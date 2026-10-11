"""Closed-loop mobile task orchestration; all planning uses observation-only modules."""
from dataclasses import asdict,dataclass,field,replace
from pathlib import Path
import json
import time
import traceback
import zipfile
import numpy as np
from scipy.spatial import cKDTree
from .types import EpisodeRecord
from .robocasa_backend import RoboCasaSimulator,CAMERAS
from .mobile_evaluation import MobileEvaluator,scene_eligibility
from .config import fingerprint
from .geometry import project,unproject,grasp_candidates,part_cloud
from .grounding import GroundingResolver,proposal_mask,pixels,lift_ground_path
from .mapping import DepthMap
from .planning import RobotModel,Planner,time_parameterize
from .navigation import NavigationCollision,SE2Planner,velocity_command,wrap_angle,transform_body
from .recording import save_observation,save_episode
from .vlm import VLM
from .view_advisor import suggest_mounted_cameras


def grounded_world(obs,g):
    if g.frame_id!=obs.frame_id or g.camera_id!=obs.camera_id:raise ValueError('stale grounding')
    if g.kind=='points':xy=g.points[0]
    else:
        mask=proposal_mask(g,obs)
        import cv2
        mask &= np.isfinite(obs.depth)&(obs.depth>.05)&(obs.depth<10)
        d=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
        if d.max()==0:raise ValueError('grounding has no valid interior')
        y,x=np.unravel_index(d.argmax(),d.shape);h,w=d.shape;xy=(x/(w-1),y/(h-1))
    uv=pixels([xy],obs.depth.shape)[0];depth=obs.depth[uv[1],uv[0]]
    if not np.isfinite(depth) or not .05<depth<10:raise ValueError('invalid grounding depth')
    return unproject(uv,depth,obs.K,obs.T_wc)


def support_plane(points,anchor=None):
    """Horizontal plane fitting in observed local geometry, with no scene heights."""
    points=np.asarray(points);points=points[np.isfinite(points).all(axis=1)]
    if anchor is not None:
        points=points[(np.linalg.norm(points[:,:2]-anchor[:2],axis=1)<.4)&(points[:,2]<anchor[2]-.006)]
    if len(points)<30:raise ValueError('insufficient observed support surface')
    levels,counts=np.unique(np.round(points[:,2]/.005).astype(int),return_counts=True)
    valid=levels[counts>=max(20,counts.max()*.2)]
    if not len(valid):raise ValueError('no stable horizontal plane')
    level=valid.max() if anchor is not None else valid.min()
    selected=points[np.abs(points[:,2]-level*.005)<.008]
    return float(np.median(selected[:,2])),selected


def payload_envelope(points,spacing=.01):
    """Conservative filled observed AABB, not just an empty surface point set."""
    lo=np.min(points,axis=0)-.005;hi=np.max(points,axis=0)+.005
    axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/spacing))+1)) for a,b in zip(lo,hi)]
    return np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)


def envelope_boundary(points,tolerance=.011):
    """Exterior of an observed filled envelope for continuous sweep checks."""
    points=np.asarray(points);lo=points.min(0);hi=points.max(0)
    return points[np.any((points<=lo+tolerance)|(points>=hi-tolerance),axis=1)]


@dataclass
class ObservedPayload:
    """Provisional collision geometry is distinct from an observed held object.

    All points come from RGB-D and robot kinematics. This estimate is deliberately
    conservative and never supplies the independent grasp-success score.
    """
    source: np.ndarray
    surface: np.ndarray
    envelope: np.ndarray
    support_z: float
    camera_id: str
    state: str = 'closed_unverified'
    source_retired: bool = False
    pending_views: list = field(default_factory=list,repr=False)

    @property
    def holding(self):return self.state=='held'

    def remember_view(self,observation,tracked,p,R):
        # Save the actual world attachment pose now: old arm joints alone cannot
        # reconstruct it once navigation changes base / torso auxiliary state.
        tracked=None if tracked is None else np.asarray(tracked).copy()
        self.pending_views.append((observation.frame_id,observation.camera_id,tracked,p.copy(),R.copy()))
        self.pending_views=self.pending_views[-12:]

    def visible_in(self,observation,p,R):
        uv,z=project(self.surface@R.T+p,observation.K,observation.T_wc)
        h,w=observation.depth.shape
        return np.count_nonzero((z>.05)&np.isfinite(uv).all(axis=1)&
            (uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1))>=20

    def retire_views(self,maps):
        """Retire measured associated surfaces, never the entire envelope."""
        self.require_holding()
        report=[];local_surfaces=[]
        for frame_id,camera_id,tracked,p,R in self.pending_views:
            if tracked is None:continue
            points=np.asarray(tracked);points=points[np.isfinite(points).all(axis=1)]
            expected=self.surface@R.T+p
            close=cKDTree(expected).query(points)[0]<=.025
            # Unrelated geometry returned by a loose tracker remains occupied.
            if len(points)<20 or close.sum()<20 or close.mean()<.8:continue
            matched=points[close]
            for map_ in maps:map_.remove_target_endpoints(matched)
            local_surfaces.append((matched-p)@R)
            report.append({'frame_id':frame_id,'camera_id':camera_id,'associated_points':len(matched),
                           'attachment_position':p.tolist(),'attachment_rotation':R.tolist()})
        if local_surfaces:
            # Newly associated side faces belong in future collision geometry
            # as well as being retired from the static map. Fuse only measured
            # points in the captured attachment frame; preserve axis extrema.
            combined=np.vstack([self.surface,*local_surfaces])
            _,indices=np.unique(np.round(combined/.005).astype(np.int32),axis=0,return_index=True)
            extrema=np.r_[combined.argmin(axis=0),combined.argmax(axis=0)]
            self.surface=combined[np.unique(np.r_[indices,extrema])]
            self.envelope=payload_envelope(self.surface)
        self.pending_views.clear()
        return report

    def verify(self,tracked,p,R,gripper_q):
        expected=self.surface@R.T+p
        tracked=np.asarray(tracked) if tracked is not None else np.empty((0,3))
        points=tracked[np.isfinite(tracked).all(axis=1)]
        aperture=float(abs(gripper_q[0]-gripper_q[1]))
        close=cKDTree(expected).query(points)[0]<=.025 if len(points) else np.zeros(0,dtype=bool)
        # A newly exposed upper face at the old source is not lift evidence.
        # Require both upward displacement and association at the moved pose.
        lift=float(np.median(points[:,2])-np.median(self.source[:,2])) if len(points) else None
        matched=int(close.sum())
        confirmed=bool(len(points)>=20 and matched>=20 and close.mean()>=.8
                       and lift is not None and lift>=.08 and aperture>.006)
        previously_held=self.state in ('held','held_unverified')
        self.state='held' if confirmed else ('held_unverified' if previously_held else 'lift_unverified')
        if not confirmed:self.pending_views.clear()
        return {'holding':confirmed,'state':self.state,'tracked_points':len(points),
                'associated_points':matched,'observed_lift_m':lift,'gripper_aperture_m':aperture}

    def require_holding(self):
        if not self.holding:
            raise ValueError('execution: payload unverified; observe the gripper and object before moving or placing')


def next_high_level_camera(strategy,decision,current_camera,rng):
    """Choose the next VLM camera without leaking the last safety-rig view.

    A navigation action is grounded in the decision frame. Reacquiring that
    mounted camera after the short drive preserves target association; the
    floor / overhead safety images remain geometry-only observations.
    """
    if strategy=='fixed':return CAMERAS[0]
    if strategy=='random':return str(rng.choice(CAMERAS))
    preferred=decision.observation_camera if decision.action=='observe' else decision.camera_id
    return preferred if preferred in CAMERAS else current_camera


def run_mobile_episode(task,seed,strategy,root,config):
    out=Path(root)/f'{task}_{strategy}_{seed:04d}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    record=EpisodeRecord(task,seed,strategy)
    (out/'manifest.json').write_text(json.dumps({'source_sha256':fingerprint(config),
        'task':task,'seed':seed,'strategy':strategy,'online':True,'backend':'robocasa',
        'config':config.model_dump(mode='json'),'experimental':True},indent=2))
    package=Path(__file__).parent
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.glob('*.py')):z.write(path,'src/view2grasp/'+path.name)
        for name in ('uv.lock','sources.json'):
            path=package.parents[1]/'runtimes/robocasa'/name
            if path.exists():z.write(path,'runtimes/robocasa/'+name)
    sim=robot=vlm=resolver=evaluator=None
    events=[];history=[];start=time.monotonic();rng=np.random.default_rng(seed)
    safety_observations=0;payload=None;grip=-1.;phase='navigate' if task=='NavigateKitchen' else 'find_object'
    budget=config.navigation_observations if task=='NavigateKitchen' else config.mobile_observations
    duration=config.navigation_sim_seconds if task=='NavigateKitchen' else config.mobile_sim_seconds
    def event(kind,**data):
        item={'elapsed':time.monotonic()-start,'kind':kind,**data};events.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    try:
        # Dining layout 2 is a development default. Formal scene manifests must
        # be generated and checked before using this backend in a benchmark.
        sim=RoboCasaSimulator(task,seed,config.image_size,layout=2,style=1)
        eligibility=scene_eligibility(sim)
        (out/'scene_eligibility_truth.json').write_text(json.dumps(eligibility,indent=2))
        if not eligibility['eligible']:raise RuntimeError('scene generation: source separation or fixed-base reach exclusion failed')
        evaluator=MobileEvaluator(sim)
        robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name,include_base=True)
        robot.set_auxiliary_state(sim.auxiliary_state())
        # Occupied endpoints accumulate, while free-space ray tests use only a
        # bounded local window. Four fresh safety views are acquired after every
        # short drive, so forgetting older free rays is conservative and keeps
        # collision-query cost independent of travelled distance.
        local=DepthMap(.005,workspace_bounds=None,max_depth=10,max_views=12)
        # Keep metric evidence at the protocol's 5 mm resolution. The SE(2)
        # search lattice is independently 20 mm; coarsening depth voxels also
        # inflated occupied endpoints to 26 mm and falsely rejected the valid
        # initial robot next to kitchen fixtures.
        navmap=DepthMap(config.voxel_resolution_m,workspace_bounds=None,max_depth=10,max_views=12)
        for map_ in (local,navmap):map_.certify_home(*[robot.points(sim.q,sim.gripper_q)[0],robot.volumes(sim.q,sim.gripper_q)])
        arm=Planner(robot,local,rng);arm.gripper=sim.gripper_q
        vlm=VLM(out/'vlm');resolver=GroundingResolver.from_config(config)
        event('perception_backend',sam_available=resolver.segmenter is not None,reason=resolver.sam_unavailable_reason)
        t0=sim.time

        def step(q,gripper,velocity=None):
            if sim.time-t0>=duration:raise TimeoutError('simulation time budget exhausted')
            sim.step(q,gripper,velocity);evaluator.update()

        def acquire(camera=None,safety=False):
            nonlocal safety_observations
            if not safety:
                if record.observations>=budget:raise RuntimeError('observation budget exhausted')
                record.observations+=1
            else:safety_observations+=1
            robot.set_auxiliary_state(sim.auxiliary_state())
            obs=sim.observe(camera)
            save_observation(out/('safety_observations' if safety else 'observations'),obs)
            self_depth=robot.depth(obs)
            for map_ in (local,navmap):
                map_.integrate(obs,robot.points(obs.q,obs.gripper_q)[0],self_depth,
                    robot.volumes(obs.q,obs.gripper_q),robot.visual_volumes(obs.q,obs.gripper_q))
                map_.certify_robot_state(robot.points(obs.q,obs.gripper_q)[0],robot.volumes(obs.q,obs.gripper_q))
            if payload is not None and payload.holding:
                p,R=robot.pose(obs.q)
                if payload.visible_in(obs,p,R):
                    tracked=resolver.track(obs,payload.surface@R.T+p,self_depth,support_z=payload.support_z)
                    payload.remember_view(obs,tracked,p,R)
            return obs,self_depth

        obs,self_depth=acquire()

        def verify_payload(observation,robot_depth):
            nonlocal phase
            if payload is None or payload.state=='closed_unverified':return
            p,R=robot.pose(observation.q)
            saved=next((view for view in reversed(payload.pending_views) if view[0]==observation.frame_id),None)
            tracked=saved[2] if saved is not None else resolver.track(
                observation,payload.surface@R.T+p,robot_depth,support_z=payload.support_z)
            evidence=payload.verify(tracked,p,R,observation.gripper_q)
            event('payload_observation',frame_id=observation.frame_id,
                camera_id=observation.camera_id,**evidence)
            if payload.holding:
                if saved is None:payload.remember_view(observation,tracked,p,R)
                for map_ in (local,navmap):
                    if not payload.source_retired:map_.remove_target_endpoints(payload.source)
                associated=payload.retire_views((local,navmap))
                arm.attached=payload.envelope
                event('payload_map_association',views=associated)
                payload.source_retired=True
                payload.camera_id=observation.camera_id
            else:
                phase='verify_grasp'
                history.append({'feedback':'Held-object association is missing or inconsistent. Base motion and placement are blocked; observe the gripper/object again.',
                                'payload_evidence':evidence})

        def nav_planner():
            floor_z,floor=support_plane(navmap.surface)
            world=robot.boundary_points(sim.q,sim.gripper_q)
            # Match the 2 cm navigation lattice while preserving volumetric coverage.
            _,indices=np.unique(np.round(world/.02).astype(np.int32),axis=0,return_index=True)
            world=world[indices]
            base=sim.base_pose
            # Rotate into current base coordinates. Z remains world vertical.
            body=transform_body(world-np.r_[base[:2],0],[0,0,-base[2]])
            # Initial contact is a robot-standing scene precondition; its exact
            # current footprint is certified, not unexplored neighboring floor.
            feet=world[world[:,2]<=floor_z+.025]
            if len(feet):floor=np.vstack([floor,feet])
            carried=None
            if payload is not None:
                p,R=robot.pose(sim.q);held=payload.envelope@R.T+p
                carried=transform_body(envelope_boundary(held)-np.r_[base[:2],0],[0,0,-base[2]])
            return SE2Planner(NavigationCollision(navmap,body,floor,floor_z,carried,
                shadow_pose=base,shadow_distance=.03)),floor_z

        def drive(goal,hints=()):
            nonlocal obs,self_depth
            if payload is not None:
                payload.require_holding()
                check_obs,check_depth=acquire(payload.camera_id,safety=True)
                verify_payload(check_obs,check_depth);payload.require_holding()
            sim.phase='navigate_carrying' if payload is not None else 'navigate'
            hold=sim.q.copy();initial=sim.base_pose.copy()
            segments=0
            while segments<16:
                current=sim.base_pose.copy();delta=np.asarray(goal)-current;delta[2]=wrap_angle(delta[2])
                distance=np.linalg.norm(delta[:2]);yaw=delta[2]
                if distance<.018 and abs(yaw)<.035:break
                if np.linalg.norm(current[:2]-initial[:2])>=.25 or abs(wrap_angle(current[2]-initial[2]))>=.4:break
                local_goal=current.copy()
                if distance>=.018:
                    local_goal[:2]+=delta[:2]*min(1.,.025/distance)
                else:
                    local_goal[2]+=np.sign(yaw)*min(abs(yaw),.025)
                planning_start=time.monotonic();nav,_=nav_planner()
                # Sensor rendering and physical execution are not search time.
                # Apply the planning budget independently to every freshly
                # observed short segment instead of expiring it during RGB-D I/O.
                path=nav.plan(current,local_goal,hints=hints,
                    time_budget=min(5.,config.planning_seconds))
                record.planning_seconds+=time.monotonic()-planning_start
                if path is None:
                    diagnostic=nav.diagnostic()
                    diagnostic.update(segment=segments,proposed_goal=np.asarray(goal).tolist(),
                        local_goal=local_goal.tolist(),stage='plan')
                    diagnostic_path=out/f'navigation_rejection_{len(events):04d}.json'
                    diagnostic_path.write_text(json.dumps(diagnostic,indent=2))
                    event('navigation_rejected',diagnostic=diagnostic_path.name,stage='plan')
                    raise ValueError('planning: '+nav.last_failure+'; '+nav.valid.last_failure+' (3 cm self-shadow certificate)')
                np.save(out/f'navigation_path_{len(events):04d}_{segments:02d}.npy',path)
                event('navigation_plan',waypoints=len(path),segment=segments,
                    proposed_goal=np.asarray(goal).tolist(),executed_local_goal=local_goal.tolist())
                for waypoint in path[1:]:
                    for _ in range(40):
                        now=sim.base_pose
                        if np.linalg.norm(now[:2]-waypoint[:2])<.008 and abs(wrap_angle(now[2]-waypoint[2]))<.018:break
                        velocity=velocity_command(now,waypoint,max_speed=.12,max_yaw=.25)
                        predicted=now.copy();c,s=np.cos(now[2]),np.sin(now[2])
                        predicted[:2]+=.05*np.array([[c,-s],[s,c]])@velocity[:2];predicted[2]+=.05*velocity[2]
                        if not nav.edge(now,predicted):
                            diagnostic=nav.diagnostic()
                            diagnostic.update(segment=segments,proposed_goal=np.asarray(goal).tolist(),
                                local_goal=local_goal.tolist(),velocity=velocity.tolist(),stage='control')
                            diagnostic_path=out/f'navigation_rejection_{len(events):04d}.json'
                            diagnostic_path.write_text(json.dumps(diagnostic,indent=2))
                            event('navigation_rejected',diagnostic=diagnostic_path.name,stage='control')
                            raise ValueError('planning: newly occupied base sweep')
                        step(hold,grip,velocity)
                for _ in range(2):step(hold,grip,np.zeros(3))
                moved=sim.base_pose-current
                if np.linalg.norm(moved[:2])<.003 and abs(moved[2])<.005:
                    raise RuntimeError('execution: mobile base made no progress')
                # Shared low-level safety rig: one outward pair in the motion
                # direction plus near-floor and top-down coverage. These images
                # update geometry only and are not shown to the VLM implicitly.
                if np.linalg.norm(moved[:2])>.002:
                    c,s=np.cos(current[2]),np.sin(current[2]);body=np.array([[c,s],[-s,c]])@moved[:2]
                    side=('front','left','rear','right')[int(np.argmax([body[0],body[1],-body[0],-body[1]]))]
                else:side='left' if moved[2]>=0 else 'right'
                for camera in (f'robot0_nav_{side}',f'robot0_nav_high_{side}',
                               'robot0_nav_floor','robot0_nav_overhead'):
                    obs,self_depth=acquire(camera,safety=True)
                if payload is not None:
                    # Recheck the object from the last camera that saw it. The
                    # floor safety images alone cannot confirm it is still held.
                    obs,self_depth=acquire(payload.camera_id,safety=True)
                    verify_payload(obs,self_depth);payload.require_holding()
                segments+=1
            for _ in range(3):step(hold,grip,np.zeros(3))

        def arm_execute(path,closing,stage):
            sim.phase=stage;np.save(out/f'arm_path_{len(events):04d}_{stage}.npy',path)
            event('arm_execute',phase=stage,waypoints=len(path))
            for q in time_parameterize(path):step(q,closing)
            for _ in range(8):step(path[-1],closing)
            if np.max(np.abs(sim.q-path[-1]))>.12:raise RuntimeError('execution: arm tracking error')

        while record.observations<=budget:
            history.append({'phase':phase,'available_cameras':list(CAMERAS),'remaining_observations':budget-record.observations,
                'camera_descriptions':{'robot0_nav_front':'wide front/down view of floor and fixtures',
                    'robot0_nav_rear':'wide rear/down view; useful to back away from a counter',
                    'robot0_nav_left':'wide left/down view','robot0_nav_right':'wide right/down view',
                    'robot0_nav_high_front':'high front view for distant fixtures and upper obstacles',
                    'robot0_nav_high_rear':'high rear view for distant fixtures and upper obstacles',
                    'robot0_nav_high_left':'high left view for distant fixtures and upper obstacles',
                    'robot0_nav_high_right':'high right view for distant fixtures and upper obstacles',
                    'robot0_nav_floor':'straight-down wide view of the floor immediately around the wheels; use when unknown floor support blocks motion',
                    'robot0_nav_overhead':'high downward view of the complete robot silhouette and nearby floor; use when the full-body sweep remains unknown',
                    'robot0_agentview_left':'close manipulation view','robot0_agentview_right':'close manipulation view',
                    'robot0_eye_in_hand':'wrist camera; pose changes with the arm'},
                'holding':payload is not None and payload.holding,
                'payload_state':'empty' if payload is None else payload.state,'base_pose':sim.base_pose.tolist(),
                'feedback':'Navigate using visible floor goals. Base motion is limited to 25 cm before observing again. Grasp only when target is reachable; place only when holding.'})
            decision=vlm.decide(obs,sim.instruction,history)
            save_observation(out/'observations',obs,decision);event('decision',**decision.model_dump())
            record.grounding_types.extend(g.kind for g in decision.groundings)
            try:
                if decision.action=='abort':record.failure='agent_abort: '+decision.reason;break
                if decision.action=='finish':
                    sim.phase='verify'
                    for _ in range(45):step(sim.q,grip)
                    break
                if decision.action=='navigate':
                    goals=[g for g in decision.groundings if g.role=='navigation_goal']
                    if not goals:raise ValueError('grounding: navigation_goal required')
                    position=grounded_world(obs,goals[0]);_,floor=support_plane(navmap.surface)
                    floor_z=float(np.median(floor[:,2]))
                    if abs(position[2]-floor_z)>.06:raise ValueError('grounding: navigation goal is not on observed floor')
                    look=[g for g in decision.groundings if g.role=='look_at']
                    direction=grounded_world(obs,look[0])[:2]-position[:2] if look else position[:2]-sim.base_pose[:2]
                    goal=np.r_[position[:2],np.arctan2(direction[1],direction[0])]
                    paths=[g for g in decision.groundings if g.role=='ground_path']
                    hints=lift_ground_path(obs,paths[0],floor_z) if paths else ()
                    drive(goal,hints)
                elif decision.action=='observe':
                    motion=decision.observation_motion if strategy=='active' else (
                        str(rng.choice(('none','turn_left','turn_right','wrist'))) if strategy=='random' else 'none')
                    if motion in ('turn_left','turn_right'):
                        goal=sim.base_pose.copy();goal[2]+=np.pi/8*(1 if motion=='turn_left' else -1)
                        drive(goal)
                    elif motion=='wrist':
                        if payload is not None:
                            payload.require_holding()
                            check_obs,check_depth=acquire(payload.camera_id,safety=True)
                            verify_payload(check_obs,check_depth);payload.require_holding()
                        path=None
                        for delta in (.3,-.3):
                            q=sim.q.copy();q[-1]+=delta
                            path=arm.plan(sim.q,q,timeout=2)
                            if path is not None:break
                        if path is None:raise ValueError('planning: wrist observation sweep lacks clearance')
                        arm_execute(path,grip,'observe_wrist')
                elif decision.action=='grasp':
                    if task=='NavigateKitchen' or payload is not None:raise ValueError('invalid grasp phase')
                    if record.attempts>=config.max_attempts:raise RuntimeError('grasp attempt budget exhausted')
                    proposals=[g for g in decision.groundings if g.role in ('grasp_contact','grasp_part','object')]
                    if not proposals:raise ValueError('grounding: grasp region required')
                    anchor=grounded_world(obs,proposals[0]);support,_=support_plane(local.surface,anchor)
                    local.set_table(support)
                    result=resolver.resolve_grasp(obs,decision,support,self_depth)
                    from .recording import save_grounding_result
                    save_grounding_result(out,obs,result)
                    candidates,target=grasp_candidates(obs,result.decision,support,robot.pose(sim.q)[1],robot.grasp_center_offset(),
                        local.surface,result.object_mask,result.part_mask)
                    local.set_target(part_cloud(obs,result.decision,target,result.part_mask))
                    np.savez_compressed(out/f'candidates_{obs.frame_id:03d}.npz',
                        positions=np.array([g.position for g in candidates]),rotations=np.array([g.rotation for g in candidates]),
                        widths=np.array([g.width for g in candidates]),ray_angles=np.array([g.ray_angle for g in candidates]),target=target)
                    timer=time.monotonic();chosen=arm.choose(sim.q,candidates,max_candidates=config.max_candidates,time_budget=config.planning_seconds)
                    record.planning_seconds+=time.monotonic()-timer
                    unknown=arm.recovery_unknown()
                    (out/f'planning_diagnostic_{obs.frame_id:03d}.json').write_text(json.dumps({
                        'failure':arm.last_failure,'failure_counts':dict(arm.failure_counts),
                        'closest_invalid':arm.closest_invalid,'closest_by_failure':arm.closest_by_failure,
                        'recovery_unknown_points':len(unknown),'checks':arm.checks},indent=2))
                    np.save(out/f'planning_unknown_{obs.frame_id:03d}.npy',unknown)
                    if chosen is None:
                        if len(unknown):
                            calibrated=[]
                            for camera in CAMERAS:
                                K,T_wc=sim.camera_calibration(camera)
                                calibrated.append(replace(obs,K=K,T_wc=T_wc,camera_id=camera))
                            advice=suggest_mounted_cameras(robot,calibrated,unknown,budget=3,
                                observed_points=local.surface)
                            event('mounted_recovery_advice',**advice)
                            history.append({'mounted_coverage_advice':advice,
                                'feedback':'The arm planner rejected these exact unknown path points. Observe the first recommended mounted camera that has not yet supplied the needed coverage; no unseen image was previewed.'})
                        raise ValueError('planning: '+arm.last_failure)
                    candidate,transit,approach=chosen
                    arm_execute(transit,-1,'pregrasp');arm_execute(approach,-1,'approach')
                    sim.phase='close';record.attempts+=1;sim.episode_attempt=record.attempts
                    for _ in range(25):step(sim.q,1)
                    grip=1.;p,R=robot.pose(sim.q)
                    payload=ObservedPayload(target.copy(),(target-p)@R,
                        (payload_envelope(target)-p)@R,support,obs.camera_id)
                    # Keep a possible object's envelope in collision checks,
                    # but do not tell the policy that closing proves a grasp.
                    arm.attached=payload.envelope;arm.gripper=sim.gripper_q
                    local.set_payload_source(target);local.remove_target_endpoints(target)
                    arm.begin_payload_lift()
                    obs,self_depth=acquire()
                    # Dynamic refresh is conservative: any unresolved moving-object
                    # evidence remains an obstacle rather than silently being erased.
                    local.remove_target_endpoints(target)
                    lift=[sim.q];arm.allow_payload_support=True
                    try:
                        try:
                            for dz in np.linspace(.01,.13,14):
                                q=arm.feasible_ik(p+[0,0,dz],R,lift[-1],contact=True,connected=True)
                                if q is None:raise ValueError('planning: lift '+arm.last_failure)
                                lift.append(q)
                        finally:arm.allow_payload_support=False;arm.end_payload_lift()
                    except ValueError:
                        # No lift motion was executed. Release at the original
                        # support rather than retaining a false carried state.
                        sim.phase='release_failed_grasp';grip=-1.
                        for _ in range(25):step(sim.q,grip)
                        arm.attached=None;payload=None;arm.gripper=sim.gripper_q
                        for map_ in (local,navmap):
                            map_.set_target(np.empty((0,3)))
                            map_.set_payload_source(np.empty((0,3)))
                        phase='find_object'
                        obs,self_depth=acquire(safety=True)
                        event('grasp_released',reason='lift rejected before motion')
                        raise
                    payload.state='lift_unverified';phase='verify_grasp'
                    arm_execute(np.array(lift),grip,'lift')
                    sim.phase='hold'
                    for _ in range(45):step(sim.q,grip)
                    obs,self_depth=acquire(payload.camera_id)
                    verify_payload(obs,self_depth);payload.require_holding()
                    phase='carry_to_destination'
                elif decision.action=='place':
                    if payload is None:raise ValueError('place requires a held object')
                    payload.require_holding()
                    check_obs,check_depth=acquire(payload.camera_id,safety=True)
                    verify_payload(check_obs,check_depth);payload.require_holding()
                    regions=[g for g in decision.groundings if g.role=='place_region']
                    if not regions:raise ValueError('grounding: placement region required')
                    point=grounded_world(obs,regions[0]);p,R=robot.pose(sim.q)
                    local.set_table(point[2])
                    bottom=float(np.min(payload.envelope@R.T,axis=0)[2])
                    goal=point.copy();goal[2]-=bottom;goal[2]+=.01
                    above=goal+[0,0,.13];q=robot.ik(above,R,sim.q)
                    transfer=None if q is None else arm.plan(sim.q,q)
                    if transfer is None:raise ValueError('planning: placement transfer '+arm.last_failure)
                    arm_execute(transfer,grip,'transfer')
                    lower=[sim.q];arm.allow_payload_support=True
                    try:
                        for pnext in np.linspace(above,goal,20)[1:]:
                            q=arm.feasible_ik(pnext,R,lower[-1],contact=True,connected=True)
                            if q is None:raise ValueError('planning: placement approach '+arm.last_failure)
                            lower.append(q)
                    finally:arm.allow_payload_support=False
                    arm_execute(np.array(lower),grip,'lower');grip=-1.;sim.phase='release'
                    for _ in range(45):step(sim.q,grip)
                    arm.attached=None;payload=None;phase='verify'
                    local.clear_dynamic();navmap.clear_dynamic()
                    obs,self_depth=acquire(safety=True)
                    arm.gripper=sim.gripper_q
                    # Released geometry is now an ordinary obstacle. Retreat
                    # through fresh RGB-D with no carried-object exemption.
                    start_p,start_R=robot.pose(sim.q);retreat=[sim.q.copy()]
                    for height in np.linspace(.01,.18,18):
                        q=arm.feasible_ik(start_p+[0,0,height],start_R,retreat[-1],connected=True)
                        if q is None:raise ValueError('planning: release retreat '+arm.last_failure)
                        retreat.append(q)
                    arm_execute(np.array(retreat),grip,'retreat')
                else:raise ValueError('unsupported action')
                history.append({'decision':decision.model_dump(),'feedback':'action completed; verify from new observation'})
            except ValueError as exc:
                history.append({'feedback':str(exc)});event('rejected',reason=str(exc));record.failure=str(exc)
                record.geometry_rejections.append(str(exc))
            if record.observations>=budget:break
            camera=next_high_level_camera(strategy,decision,sim.camera,rng)
            obs,self_depth=acquire(camera)
            if payload is not None and payload.state!='closed_unverified':
                p,R=robot.pose(obs.q)
                # An image looking elsewhere is not evidence of a drop. When
                # the predicted object is in view, update even a previously
                # held estimate before the next policy decision.
                if not payload.holding or payload.visible_in(obs,p,R):
                    verify_payload(obs,self_depth)
                    if payload.holding:phase='carry_to_destination'
        if not record.failure:record.failure='execution: independent success criterion not met'
    except KeyboardInterrupt:
        record.failure='interrupted during development';event('interrupted')
        raise
    except Exception as exc:
        record.failure=str(exc);(out/'error.txt').write_text(traceback.format_exc());event('error',message=str(exc))
    finally:
        if vlm:record.vlm_calls=vlm.calls;record.inference_seconds=vlm.seconds;vlm.client.close()
        if resolver:record.segmentation_calls=resolver.calls;record.perception_seconds=resolver.seconds
        record.elapsed_seconds=time.monotonic()-start
        if evaluator:
            truth=evaluator.result();record.success=truth['success'];record.collision=truth['collision']
            record.first_grasp_success=truth['first_grasp_success']
            record.official_success=truth['official_success'];record.collision_free_success=truth['collision_free_success']
            record.path_length_m=truth['path_length_m'];record.held_path_length_m=truth['held_path_length_m']
            (out/'evaluation_truth.json').write_text(json.dumps(truth,indent=2))
            if record.success:record.failure=''
        (out/'mobile_metrics.json').write_text(json.dumps({'safety_observations':safety_observations,'final_phase':phase,
            'payload_state':'empty' if payload is None else payload.state,
            'holding_estimate':payload is not None and payload.holding},indent=2))
        if sim:save_episode(out,sim,record,events);sim.close()
        else:(out/'result.json').write_text(json.dumps(asdict(record),indent=2))
        if robot:robot.close()
    return asdict(record)
