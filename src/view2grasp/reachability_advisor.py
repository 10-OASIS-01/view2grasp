"""Bounded next-view kinematics advice from observed points and known robot geometry.

No candidate scene image is acquired and no map or executable grasp is changed.
A recommended camera must be observed and the target grounded again before use.
"""
import time

import mujoco
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation

from .geometry import orbit_pose, project


def _orbit_views(obs):
    # A task-independent orbit grid, using the current calibrated orbit radius.
    distance=float(np.clip(obs.view[2],.55,1.3))
    return [(float(az),float(el),distance)
            for el in (0,15,30) for az in range(-180,180,30)]


def _robot_blocked(camera,points,volumes):
    """Segment/convex-hull intersection with known robot visual geometry only."""
    directions=points-camera
    blocked=np.zeros(len(points),dtype=bool)
    for A,b,_,_ in volumes:
        numerator=A@camera+b
        denominator=directions@A.T
        parallel=np.abs(denominator)<1e-12
        with np.errstate(divide='ignore',invalid='ignore'):
            t=-numerator/denominator
        low=np.maximum(0,np.max(np.where(denominator < -1e-12,t,-np.inf),axis=1))
        high=np.minimum(1-1e-6,np.min(np.where(denominator > 1e-12,t,np.inf),axis=1))
        blocked|=(low<=high)&~np.any(parallel&(numerator>0),axis=1)
    return blocked


def _visibility(obs,points,T_wc,volumes,observed_points):
    """Projection and occlusion estimates; never a certificate of visibility."""
    uv,z=project(points,obs.K,T_wc);h,w=obs.depth.shape
    inside=(z>.01)&np.isfinite(uv).all(axis=1)&(uv[:,0]>0)&(uv[:,0]<w-1)&(uv[:,1]>0)&(uv[:,1]<h-1)
    robot_clear=~_robot_blocked(T_wc[:3,3],points,volumes)
    visible=inside&robot_clear
    if len(observed_points):
        scene_uv,scene_z=project(observed_points,obs.K,T_wc)
        valid=(scene_z>.01)&np.isfinite(scene_uv).all(axis=1)
        ids=np.flatnonzero(valid)
        ij=np.rint(scene_uv[ids]).astype(int)
        valid=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        ij=ij[valid];ids=ids[valid]
        flat=np.full(h*w,np.inf)
        np.minimum.at(flat,ij[:,1]*w+ij[:,0],scene_z[ids])
        # Only already observed endpoints can reject a line of sight. Their
        # absence does not certify the unobserved volume or an unseen back face.
        zbuffer=minimum_filter(flat.reshape(h,w),size=3,mode='constant',cval=np.inf)
        ids=np.flatnonzero(inside)
        ij=np.rint(uv[ids]).astype(int)
        visible[ids]&=z[ids]<=zbuffer[ij[:,1],ij[:,0]]+.008
    return uv,z,inside,robot_clear,visible


def _rotations(ray,part,anchor,hint):
    x=hint[:,0]-np.dot(hint[:,0],ray)*ray
    if np.linalg.norm(x)<1e-6:
        ref=np.eye(3)[np.argmin(np.abs(ray))];x=np.cross(ref,ray)
    x/=np.linalg.norm(x);y=np.cross(ray,x)
    planar=(part-anchor)@np.column_stack([x,y])
    values,vectors=np.linalg.eigh(np.cov(planar.T))
    if values[1]>2*max(values[0],1e-7):
        x=np.column_stack([x,y])@vectors[:,0];y=np.cross(ray,x)
    orientations=[]
    for roll in np.linspace(0,2*np.pi,24,endpoint=False):
        R=np.column_stack([x,y,ray])@Rotation.from_rotvec([0,0,roll]).as_matrix()
        width=float(np.ptp((part-anchor)@R[:,0])+.008)
        if width<=.077:
            orientations.append((R,width,float(np.rad2deg(roll))))
    return sorted(orientations,key=lambda item:np.linalg.norm(Rotation.from_matrix(item[0]@hint.T).as_rotvec()))


def suggest_reachable_views(robot,obs,contact_world,part_cloud,observed_points=None,
                            *,budget=3,max_ik_calls=72,time_budget=4.):
    """Suggest new tabletop observations using bounded center-ray pregrasp IK.

    `contact_world` and `part_cloud` must be the caller's current resolved RGB-D
    contact and associated grasp-part surface. They are a static observation
    hypothesis, never an object pose from a simulator. Optional `observed_points`
    are already observed surface endpoints, used only to reject occluded views.
    Only the center ray (0 degrees within the existing 15-degree approach cone)
    is tested; failure is not proof that a view's entire cone is unreachable.
    The known robot model's complete qpos, including gripper/auxiliary joints,
    is restored even when advice fails. No Simulator or renderer is used.
    """
    started=time.monotonic()
    if (not isinstance(budget,int) or budget<0 or not isinstance(max_ik_calls,int)
            or max_ik_calls<0 or not np.isfinite(time_budget) or time_budget<0):
        raise ValueError('invalid reachability advice budget')
    anchor=np.asarray(contact_world,dtype=float)
    part=np.asarray(part_cloud,dtype=float)
    observed=np.empty((0,3)) if observed_points is None else np.asarray(observed_points,dtype=float)
    if (anchor.shape!=(3,) or part.ndim!=2 or part.shape[1]!=3 or observed.ndim!=2 or observed.shape[1]!=3
            or not np.isfinite(anchor).all() or not np.isfinite(part).all() or not np.isfinite(observed).all()):
        raise ValueError('reachability advice requires finite observed 3-D points')
    K=np.asarray(obs.K);T=np.asarray(obs.T_wc)
    if (obs.camera_id!='activeview' or len(obs.depth.shape)!=2 or min(obs.depth.shape)<2
            or K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all()
            or not np.isfinite(T).all() or min(K[0,0],K[1,1])<=0
            or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-6)
            or not np.allclose(T[3],[0,0,0,1]) or not np.isfinite(obs.q).all()):
        raise ValueError('reachability advice requires calibrated tabletop observation')
    result={'views':[],'candidates':[],'source_frame_id':obs.frame_id,'source_camera_id':obs.camera_id,
        'contact_world':anchor.tolist(),'requires_new_observation':True,'requires_new_grounding':True,
        'scope':'Kinematics and predicted occlusion from known robot/already observed surfaces only; NOT scene clearance, aperture/part semantics, a motion plan, or permission to grasp. No unvisited scene images were acquired.',
        'search':{'ik_calls':0,'max_ik_calls':max_ik_calls,'tested_views':0,'grid_views':0,
                  'time_budget_seconds':time_budget,'budget_exhausted':False}}
    def finish(reason):
        result['search'].update(stop_reason=reason,elapsed_seconds=time.monotonic()-started)
        result['search']['budget_exhausted']=reason in ('time_budget','ik_call_budget')
        return result
    if budget==0:return finish('no_advice_requested')
    if max_ik_calls==0:return finish('ik_call_budget')
    if time_budget==0:return finish('time_budget')
    local=part[np.linalg.norm(part-anchor,axis=1)<.04]
    if len(local)<4:return finish('insufficient_observed_part')
    # Bounded geometric samples do not create any additional surface evidence.
    sample=part[np.linspace(0,len(part)-1,min(48,len(part)),dtype=int)]
    points=np.vstack([anchor,sample])
    observed=observed[np.linspace(0,len(observed)-1,min(20000,len(observed)),dtype=int)]
    result['search'].update(part_samples=len(sample),observed_surface_samples=len(observed))
    values,vectors=np.linalg.eigh(np.cov((local-anchor).T))
    normal=vectors[:,0] if values[1]>1e-7 and values[0]<.15*values[1] else None
    original_qpos=robot.data.qpos.copy()
    try:
        robot.set_q(obs.q,obs.gripper_q)
        _,hint=robot.pose();offset=robot.grasp_center_offset()
        volumes=robot.visual_volumes(obs.q,obs.gripper_q)
        choices=[]
        for view in _orbit_views(obs):
            if time.monotonic()-started>=time_budget:return finish('time_budget')
            result['search']['grid_views']+=1
            pose=orbit_pose(*view)
            if np.allclose(pose,obs.T_wc,atol=1e-6):continue
            uv,z,inside,robot_clear,visible=_visibility(obs,points,pose,volumes,observed)
            if not visible[0] or np.mean(visible[1:])<.5:continue
            ray=anchor-pose[:3,3];ray/=np.linalg.norm(ray)
            orientations=_rotations(ray,local,anchor,hint)
            if not orientations:continue
            h,w=obs.depth.shape;xy=uv[0]/[w-1,h-1]
            margin=float(min(*xy,*(1-xy)))
            incidence=None if normal is None else float(abs(normal@ray))
            score=(float(np.mean(visible[1:])),0. if incidence is None else incidence,margin)
            choices.append({'view':view,'ray':ray,'orientations':orientations,'score':score,
                'evidence':{'view':list(view),'contact_normalized':xy.tolist(),
                    'predicted_part_unoccluded_fraction':float(np.mean(visible[1:])),
                    'predicted_robot_unoccluded_fraction':float(np.mean(robot_clear[1:])),
                    'observed_surface_incidence_abs_cosine':incidence}})
        choices.sort(key=lambda c:c['score'],reverse=True)
        # Breadth before depth: one roll per view per pass, so one impossible
        # ray cannot consume every IK call. All tested poses use an existing
        # insertion depth and the exact candidate-generation grasp-center offset.
        solved=set();tested=set()
        for roll_index in range(24):
            for index,choice in enumerate(choices):
                if index in solved or roll_index>=len(choice['orientations']):continue
                if time.monotonic()-started>=time_budget:return finish('time_budget')
                if result['search']['ik_calls']>=max_ik_calls:return finish('ik_call_budget')
                R,width,roll=choice['orientations'][roll_index];ray=choice['ray']
                insertion=.024
                pre=anchor+ray*insertion-R@offset-ray*.08
                tested.add(index);result['search']['tested_views']=len(tested)
                result['search']['ik_calls']+=1
                q=robot.ik(pre,R,obs.q,max_iter=120)
                if q is None:continue
                q=np.asarray(q)
                if q.shape!=np.asarray(obs.q).shape or not np.isfinite(q).all():continue
                if np.any(q<robot.limits[:,0]+.015-1e-9) or np.any(q>robot.limits[:,1]-.015+1e-9):continue
                actual,actual_R=robot.pose(q)
                ep=float(np.linalg.norm(pre-actual))
                er=float(np.linalg.norm(Rotation.from_matrix(R@actual_R.T).as_rotvec()))
                if ep>=.0015 or er>=.015:continue
                evidence=choice['evidence']
                evidence['pregrasp_ik']={'position_error_m':ep,'orientation_error_rad':er,
                    'approach_angle_to_new_camera_ray_degrees':0.,'approach_cone_degrees':15.,
                    'roll_degrees':roll,'insertion_m':insertion,'part_width_m':width,
                    'minimum_joint_limit_margin_rad':float(np.min(np.minimum(q-robot.limits[:,0],robot.limits[:,1]-q))),
                    'checked':'Pregrasp pose only; approach, transit, collision and grasp were not checked.'}
                result['views'].append(list(choice['view']));result['candidates'].append(evidence);solved.add(index)
                if len(solved)>=budget:return finish('requested_advice_found')
        return finish('tested_center_rays_exhausted')
    finally:
        robot.data.qpos[:]=original_qpos
        mujoco.mj_forward(robot.model,robot.data)
