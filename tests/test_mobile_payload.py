"""Observation-only carrying and separately privileged mobile grasp scoring."""
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from view2grasp.mobile_evaluation import MobileEvaluator
from view2grasp.mobile_runner import ObservedPayload,payload_envelope


def observed_payload():
    source=np.array([[x,y,z] for x in np.linspace(-.02,.02,5)
                     for y in np.linspace(-.02,.02,5) for z in (.81,.84,.87)])
    origin=np.array([0.,0.,.84])
    estimate=ObservedPayload(source,source-origin,payload_envelope(source)-origin,
        .8,'robot0_agentview_left',state='lift_unverified')
    return estimate,origin


def test_closing_is_not_holding_and_does_not_allow_transport():
    estimate,_=observed_payload();estimate.state='closed_unverified'
    assert not estimate.holding
    with pytest.raises(ValueError,match='unverified'):estimate.require_holding()


def test_observed_lift_confirms_holding_but_missing_evidence_blocks_it_again():
    estimate,origin=observed_payload();p=origin+[0,0,.13]
    report=estimate.verify(estimate.source+[0,0,.13],p,np.eye(3),[.02,-.02])
    assert report['holding'] and report['observed_lift_m']==pytest.approx(.13)
    estimate.require_holding()
    envelope=estimate.envelope.copy()
    report=estimate.verify(None,p,np.eye(3),[.02,-.02])
    assert not report['holding'] and estimate.state=='held_unverified'
    assert np.array_equal(estimate.envelope,envelope)  # possible payload remains a collision volume
    with pytest.raises(ValueError,match='unverified'):estimate.require_holding()


def test_all_verified_camera_surfaces_retire_at_their_recorded_world_pose():
    from view2grasp.geometry import cloud
    from view2grasp.grounding import track_surface
    from view2grasp.mapping import DepthMap,OCCUPIED
    from test_geometry import observation
    K=np.array([[500.,0.,30.],[0.,500.,30.],[0.,0.,1.]])
    def frame(number,offset,side=False):
        depth=np.full((60,60),2.)
        if side:
            depth[25:35,30:40]=1.005
            depth[40:45,30:35]=1.005  # separate nearby obstacle, inside tracking search bounds
        else:depth[25:35,20:30]=1.
        pose=np.eye(4);pose[0,3]=offset
        return replace(observation(),frame_id=number,camera_id='side' if side else 'front',
            depth=depth,rgb=np.zeros((60,60,3),dtype=np.uint8),K=K,T_wc=pose)
    original=frame(0,0)
    front_points=cloud(original,1);front_points=front_points[front_points[:,2]<1.5]
    source=front_points-[0,0,.13]
    payload=ObservedPayload(source,source.copy(),payload_envelope(source),.8,'front',state='held')
    side=frame(1,.1,True);front=frame(2,.2)
    first_pose=np.array([.1,0,.13]);last_pose=np.array([.2,0,.13]);R=np.eye(3)
    side_points=track_surface(side,payload.surface+first_pose,support_z=.8)
    seen_front=track_surface(front,payload.surface+last_pose,support_z=.8)
    assert side_points is not None and len(side_points)==100
    maps=[DepthMap(workspace_bounds=None),DepthMap(workspace_bounds=None)]
    for map_ in maps:
        map_.integrate(side,np.empty((0,3)));map_.integrate(front,np.empty((0,3)))
        map_.remove_target_endpoints(seen_front)
        assert np.all(map_.classify(side_points)==OCCUPIED)
    payload.remember_view(side,side_points,first_pose,R)
    payload.remember_view(front,seen_front,last_pose,R)
    assert payload.verify(seen_front,last_pose,R,[.02,-.02])['holding']
    reports=payload.retire_views(maps)
    assert {row['camera_id'] for row in reports}=={'side','front'}
    assert reports[0]['attachment_position']==first_pose.tolist()
    assert payload.envelope[:,0].max()>=np.max(side_points[:,0]-first_pose[0])
    obstacle=np.array([[.102, .02412,1.005]])
    for map_ in maps:
        assert not np.any(map_.classify(side_points)==OCCUPIED)
        assert map_.classify(obstacle)[0]==OCCUPIED


def test_missing_held_evidence_discards_pending_surface_retirement():
    from test_geometry import observation
    payload,origin=observed_payload();payload.state='held'
    held=payload.source+[0,0,.13]
    payload.remember_view(observation(),held,origin+[0,0,.13],np.eye(3))
    assert payload.pending_views
    assert not payload.verify(None,origin+[0,0,.13],np.eye(3),[.02,-.02])['holding']
    assert not payload.pending_views
    with pytest.raises(ValueError,match='unverified'):payload.retire_views(())


@pytest.mark.parametrize('case',['stationary','closed_empty','too_few','nearby_clutter','nonfinite'])
def test_grasp_confirmation_rejects_inconsistent_or_insufficient_evidence(case):
    estimate,origin=observed_payload();p=origin+[0,0,.13]
    tracked=estimate.source+[0,0,.13];gripper=[.02,-.02]
    if case=='stationary':tracked=estimate.source
    elif case=='closed_empty':gripper=[.001,-.001]
    elif case=='too_few':tracked=tracked[:19]
    elif case=='nearby_clutter':tracked=tracked+[.04,0,0]
    elif case=='nonfinite':tracked[:]=np.nan
    assert not estimate.verify(tracked,p,np.eye(3),gripper)['holding']


def scoring_sim(task='DeliverCanToBowl'):
    names=['gripper0_left','gripper0_right','can','floor','robot0_arm']
    model=SimpleNamespace(ngeom=len(names),geom_id2name=lambda i:names[i],geom_name2id=names.index)
    data=SimpleNamespace(contact=[],body_xpos=np.array([[0.,0.,.8]]),time=0.)
    env=SimpleNamespace(sim=SimpleNamespace(model=model,data=data),
        objects={'soda_can':SimpleNamespace(contact_geoms=['can'])},obj_body_id={'soda_can':0},
        _check_success=lambda:False)
    grip=SimpleNamespace(important_geoms={'left_finger':['gripper0_left'],'right_finger':['gripper0_right']})
    sim=SimpleNamespace(task=task,env=env,robot=SimpleNamespace(gripper={'right':grip}),
        base_pose=np.zeros(3),phase='observe',episode_attempt=0)
    return sim


def advance_score(sim,evaluator,count,*,height=.9,contacts=(0,1),attempt=1):
    sim.episode_attempt=attempt
    for _ in range(count):
        sim.env.sim.data.time+=.05
        sim.env.sim.data.body_xpos[0,2]=height
        sim.env.sim.data.contact=[SimpleNamespace(geom1=i,geom2=2,dist=-.0001) for i in contacts]
        evaluator.update()


def test_mobile_first_grasp_is_unmeasured_until_an_actual_attempt():
    sim=scoring_sim();evaluator=MobileEvaluator(sim)
    assert evaluator.result()['first_grasp_success'] is None
    evaluator.update()
    assert evaluator.result()['first_grasp_success'] is None
    advance_score(sim,evaluator,1,height=.8)
    assert evaluator.result()['first_grasp_success'] is False
    navigation=MobileEvaluator(scoring_sim('NavigateKitchen'))
    assert navigation.result()['first_grasp_success'] is None


def test_mobile_first_grasp_requires_both_contacts_eight_cm_and_two_seconds():
    sim=scoring_sim();evaluator=MobileEvaluator(sim)
    advance_score(sim,evaluator,45,contacts=(0,))
    advance_score(sim,evaluator,45,height=.87)
    advance_score(sim,evaluator,40)
    assert evaluator.result()['first_grasp_success'] is False
    advance_score(sim,evaluator,1)
    assert evaluator.result()['first_grasp_success'] is True
    # A later drop cannot rewrite whether the first stable grasp happened.
    advance_score(sim,evaluator,10,contacts=(),height=.8)
    assert evaluator.result()['first_grasp_success'] is True


def test_interrupted_or_second_attempt_cannot_be_counted_as_first_grasp():
    sim=scoring_sim();evaluator=MobileEvaluator(sim)
    advance_score(sim,evaluator,25)
    advance_score(sim,evaluator,1,contacts=())
    advance_score(sim,evaluator,25)
    assert evaluator.result()['first_grasp_success'] is False
    advance_score(sim,evaluator,50,attempt=2)
    assert evaluator.result()['first_grasp_success'] is False


def test_native_placement_does_not_fabricate_a_first_grasp_or_held_transport():
    sim=scoring_sim();sim.env._check_success=lambda:True
    evaluator=MobileEvaluator(sim)
    advance_score(sim,evaluator,50,contacts=(),attempt=0)
    result=evaluator.result()
    assert result['official_success'] and result['stable_success']
    assert result['first_grasp_success'] is None
    assert not result['success'] and not result['required_held_transport_met']


def test_positive_contact_margin_is_not_physical_two_finger_contact():
    sim=scoring_sim();evaluator=MobileEvaluator(sim)
    sim.episode_attempt=1;sim.env.sim.data.body_xpos[0,2]=.9
    sim.env.sim.data.contact=[SimpleNamespace(geom1=i,geom2=2,dist=.001) for i in (0,1)]
    for _ in range(50):
        sim.env.sim.data.time+=.05;sim.base_pose[0]+=.01;evaluator.update()
    result=evaluator.result()
    assert not result['first_grasp_success'] and result['held_path_length_m']==0.


@pytest.mark.parametrize('scenario',['planning_rejected','lift_unverified','wrist_drop','drive_drop','visible_drop'])
def test_lift_or_later_evidence_failure_never_allows_unverified_motion(monkeypatch,tmp_path,scenario):
    """Exercise the runner's real failure branch, with no renderer or VLM service."""
    import view2grasp.mobile_runner as runner
    import view2grasp.recording as recording
    from view2grasp.config import ExperimentConfig
    from view2grasp.types import AgentDecision,GroundingProposal
    from test_geometry import observation
    source=observed_payload()[0].source
    lift_plans=scenario!='planning_rejected'
    later_drop=scenario in ('wrist_drop','drive_drop','visible_drop')
    class Sim:
        def __init__(self,*args,**kwargs):
            self.q=np.zeros(7);self.gripper_q=np.array([.02,-.02]);self.time=0.
            self.camera='robot0_agentview_left';self.base_pose=np.zeros(3)
            self.robot_xml='';self.joint_names=[];self.eef_name='';self.instruction='carry can'
            self.frame_id=0;self.phase='observe'
        def auxiliary_state(self):return {}
        def observe(self,camera=None):
            self.camera=camera or self.camera;self.frame_id+=1
            return replace(observation(),frame_id=self.frame_id,camera_id=self.camera,
                q=self.q,gripper_q=self.gripper_q,timestamp=self.time)
        def step(self,q,gripper,velocity=None):self.time+=.05;self.q=q.copy()
        def close(self):pass
    class Robot:
        def __init__(self,*args,**kwargs):pass
        def set_auxiliary_state(self,*args):pass
        def points(self,*args):return (np.zeros((1,3)),)
        def volumes(self,*args):return []
        def visual_volumes(self,*args):return []
        def depth(self,obs):return np.full_like(obs.depth,10.)
        def pose(self,q):return np.array([0.,0.,.84+q[0]]),np.eye(3)
        def grasp_center_offset(self):return np.zeros(3)
        def close(self):pass
    class Map:
        surface=source
        def __init__(self,*args,**kwargs):pass
        def __getattr__(self,name):return lambda *args,**kwargs:None
    planners=[]
    class Arm:
        def __init__(self,*args):
            self.attached=None;self.last_failure='unknown space';self.failure_counts={}
            self.closest_invalid=None;self.closest_by_failure={};self.checks=0;planners.append(self)
        def choose(self,*args,**kwargs):return None,np.zeros((2,7)),np.zeros((2,7))
        def recovery_unknown(self):return np.empty((0,3))
        def begin_payload_lift(self):pass
        def end_payload_lift(self):pass
        def feasible_ik(self,p,*args,**kwargs):return np.r_[p[2]-.84,np.zeros(6)] if lift_plans else None
    histories=[]
    class FakeVLM:
        calls=0;seconds=0.
        def __init__(self,*args):self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history):
            self.calls+=1;histories.append(json.loads(json.dumps(history)))
            if self.calls>2:return AgentDecision(action='abort',frame_id=obs.frame_id,camera_id=obs.camera_id)
            if self.calls==2 and later_drop:
                if scenario=='drive_drop':
                    return AgentDecision(action='navigate',frame_id=obs.frame_id,camera_id=obs.camera_id,
                        groundings=[GroundingProposal(kind='points',role='navigation_goal',frame_id=obs.frame_id,
                            camera_id=obs.camera_id,points=[(.5,.5)])])
                return AgentDecision(action='observe',frame_id=obs.frame_id,camera_id=obs.camera_id,
                    observation_motion='wrist')
            if self.calls==2 and lift_plans:
                return AgentDecision(action='place',frame_id=obs.frame_id,camera_id=obs.camera_id)
            return AgentDecision(action='grasp',frame_id=obs.frame_id,camera_id=obs.camera_id,
                groundings=[GroundingProposal(kind='points',role='object',frame_id=obs.frame_id,
                    camera_id=obs.camera_id,points=[(.5,.5)])])
    tracking_calls=[]
    def track(*args,**kwargs):
        tracking_calls.append(args[0].frame_id)
        return source+[0,0,.13] if later_drop and len(tracking_calls)==1 else None
    resolver=SimpleNamespace(segmenter=None,sam_unavailable_reason='test',calls=0,seconds=0.,
        resolve_grasp=lambda obs,decision,*args:SimpleNamespace(decision=decision,object_mask=None,part_mask=None),
        track=track)
    evaluator=SimpleNamespace(update=lambda:None,result=lambda:{'success':False,'collision':False,
        'first_grasp_success':False,'official_success':False,'collision_free_success':False,
        'path_length_m':0.,'held_path_length_m':0.})
    monkeypatch.setattr(runner,'RoboCasaSimulator',Sim)
    monkeypatch.setattr(runner,'scene_eligibility',lambda sim:{'eligible':True})
    monkeypatch.setattr(runner,'MobileEvaluator',lambda sim:evaluator)
    monkeypatch.setattr(runner,'RobotModel',Robot);monkeypatch.setattr(runner,'DepthMap',Map)
    monkeypatch.setattr(runner,'Planner',Arm);monkeypatch.setattr(runner,'VLM',FakeVLM)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:resolver)
    monkeypatch.setattr(runner,'support_plane',lambda points,anchor=None:(.8,source) if anchor is not None else
        (1.,np.array([[0.,0.,1.]])))
    monkeypatch.setattr(runner,'grasp_candidates',lambda *args:([],source))
    monkeypatch.setattr(runner,'part_cloud',lambda *args:source)
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    if scenario in ('wrist_drop','drive_drop'):
        # The high-level frame looks elsewhere; fresh manipulation-camera
        # evidence must still be required before either kind of motion.
        monkeypatch.setattr(runner.ObservedPayload,'visible_in',lambda *args:False)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:
        (out/'events.json').write_text(json.dumps(events)))
    result=runner.run_mobile_episode('DeliverCanToBowl',3,'active',tmp_path,ExperimentConfig())
    second_state=next(row for row in reversed(histories[1]) if 'holding' in row)
    assert 'target_framing' not in second_state, 'pre-grasp static evidence must expire when the arm acts'
    events=json.loads((tmp_path/'DeliverCanToBowl_active_0003/events.json').read_text())
    assert not result['success']
    if later_drop:
        assert result['attempts']==1 and planners[0].attached is not None
        if scenario=='visible_drop':assert not second_state['holding']
        else:assert second_state['holding']
        assert any(row['kind']=='payload_observation' and row['state']=='held_unverified' for row in events)
        assert any(row['kind']=='rejected' and 'payload unverified' in row['reason'] for row in events)
        assert not any(row['kind']=='navigation_plan' or
            (row['kind']=='arm_execute' and row['phase']=='observe_wrist') for row in events)
    elif lift_plans:
        # A lift command alone is not a held object: keep its possible collision
        # envelope, reject placement, and do not count another grasp attempt.
        assert result['attempts']==1 and planners[0].attached is not None
        assert not second_state['holding']
        assert second_state['payload_state']=='lift_unverified'
        assert not any(row['kind']=='grasp_released' for row in events)
        assert any(row['kind']=='rejected' and 'payload unverified' in row['reason'] for row in events)
        assert not any(row['kind']=='arm_execute' and row['phase']=='transfer' for row in events)
    else:
        # Rejected planning has not moved the arm: release on its original
        # support and allow a fresh second grasp with new observations.
        assert result['attempts']==2 and planners[0].attached is None
        assert not second_state['holding']
        assert second_state['payload_state']=='empty'
        assert len([row for row in events if row['kind']=='grasp_released'])==2
