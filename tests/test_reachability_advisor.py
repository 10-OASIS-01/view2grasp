from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_geometry import observation
from view2grasp.geometry import orbit_pose
from view2grasp.reachability_advisor import suggest_reachable_views, _robot_blocked, _visibility
import view2grasp.reachability_advisor as advisor


def inputs():
    obs=replace(observation(),T_wc=orbit_pose(0,15,.8),view=(0.,15.,.8))
    anchor=np.array([-.12,0.,.95])
    part=anchor+np.array([[0.,y,z] for y in (-.004,0,.004) for z in (-.015,0,.015)])
    return obs,anchor,part


class KnownRobot:
    """Stateful FK/IK double; rejects any rendering or scene acquisition."""
    def __init__(self,success=True):
        self.model=object()
        self.data=SimpleNamespace(qpos=np.r_[np.linspace(-.3,.3,7),.02,-.025,1.2])
        self.qidx=np.arange(7);self.gq=np.array([7,8])
        self.limits=np.tile([-3.,3.],(7,1));self.success=success
        self.calls=[];self.solutions={};self.offset=np.array([0.,0.,-.0036])
    def set_q(self,q,gripper=None):
        self.data.qpos[self.qidx]=q
        self.data.qpos[self.gq]=[.04,-.04] if gripper is None else gripper
    def pose(self,q=None):
        if q is not None:self.set_q(q)
        return self.solutions.get(tuple(self.data.qpos[self.qidx]),(np.array([-.1,0,1.]),np.eye(3)))
    def visual_volumes(self,q,gripper):
        self.set_q(q,gripper);return []
    def grasp_center_offset(self):return self.offset.copy()
    def ik(self,p,R,seed,max_iter=120):
        self.calls.append((p.copy(),R.copy(),seed.copy(),max_iter))
        # Deliberately modify arm, gripper and an auxiliary joint to ensure the
        # adviser restores the full state, not only the arm or source obs.q.
        self.data.qpos[:]=.7
        if isinstance(self.success,Exception):raise self.success
        if not self.success:return None
        q=np.full(7,.1*len(self.calls))
        self.solutions[tuple(q)]=(p.copy(),R.copy())
        return q
    def depth(self,*args):raise AssertionError('no new renderer is allowed')
    def observe(self,*args):raise AssertionError('no scene images are allowed')


@pytest.fixture(autouse=True)
def deterministic_grid_and_forward(monkeypatch):
    monkeypatch.setattr(advisor,'_orbit_views',lambda obs:[(-90.,15.,.8),(90.,15.,.8),(-180.,15.,.8)])
    monkeypatch.setattr(advisor.mujoco,'mj_forward',lambda model,data:None)


def test_success_is_new_view_advice_with_center_ray_ik_not_execution():
    obs,anchor,part=inputs();robot=KnownRobot();before=robot.data.qpos.copy()
    original=(obs.q.copy(),obs.gripper_q.copy(),obs.T_wc.copy(),anchor.copy(),part.copy())
    result=suggest_reachable_views(robot,obs,anchor,part,budget=2)
    assert len(result['views'])==2 and len(robot.calls)==2
    assert result['requires_new_observation'] and result['requires_new_grounding']
    assert 'NOT scene clearance' in result['scope'] and 'No unvisited scene images' in result['scope']
    for candidate,(p,R,seed,iterations) in zip(result['candidates'],robot.calls):
        ray=anchor-orbit_pose(*candidate['view'])[:3,3];ray/=np.linalg.norm(ray)
        assert np.allclose(R[:,2],ray)
        assert np.allclose(p,anchor+ray*.024-R@robot.offset-ray*.08)
        assert np.allclose(seed,obs.q) and iterations==120
        assert candidate['pregrasp_ik']['position_error_m']==0
        assert candidate['pregrasp_ik']['approach_angle_to_new_camera_ray_degrees']==0
        assert 'not checked' in candidate['pregrasp_ik']['checked']
        assert 'q' not in candidate and 'grasp' not in candidate
    assert np.array_equal(robot.data.qpos,before)
    for actual,expected in zip((obs.q,obs.gripper_q,obs.T_wc,anchor,part),original):
        assert np.array_equal(actual,expected)


def test_no_solution_stops_at_call_budget_and_restores_full_robot_state():
    obs,anchor,part=inputs();robot=KnownRobot(False);before=robot.data.qpos.copy()
    result=suggest_reachable_views(robot,obs,anchor,part,max_ik_calls=2)
    assert result['views']==[] and len(robot.calls)==2
    assert result['search']['stop_reason']=='ik_call_budget'
    assert result['search']['budget_exhausted']
    # Fair first pass visits different rays rather than exhausting one ray's rolls.
    assert not np.allclose(robot.calls[0][1][:,2],robot.calls[1][1][:,2])
    assert np.array_equal(robot.data.qpos,before)


def test_unexpected_failure_restores_robot_state():
    obs,anchor,part=inputs();robot=KnownRobot(RuntimeError('IK backend error'));before=robot.data.qpos.copy()
    with pytest.raises(RuntimeError,match='IK backend'):
        suggest_reachable_views(robot,obs,anchor,part)
    assert np.array_equal(robot.data.qpos,before)


@pytest.mark.parametrize('limits,wrong_pose',[(True,False),(False,True)])
def test_solver_return_is_rechecked_for_limits_and_actual_fk(limits,wrong_pose):
    obs,anchor,part=inputs();robot=KnownRobot()
    if limits:robot.limits[:]=[-.05,.05]
    if wrong_pose:robot.pose=lambda q=None:(np.array([8.,8.,8.]),np.eye(3))
    result=suggest_reachable_views(robot,obs,anchor,part,max_ik_calls=2)
    assert result['views']==[]


def box(center,half=.05):
    A=np.vstack([np.eye(3),-np.eye(3)]);center=np.array(center)
    return A,np.r_[-center-half,center-half],center-half,center+half


def test_robot_occlusion_is_segment_bounded_not_infinite_ray():
    points=np.array([[0.,0.,1.],[.5,0,1.],[0.,0.,.2]])
    assert _robot_blocked(np.zeros(3),points,[box([0,0,.5])]).tolist()==[True,False,False]
    assert not _robot_blocked(np.zeros(3),points,[box([0,0,-.5])]).any()
    assert not _robot_blocked(np.zeros(3),points,[box([0,0,2.])]).any()
    assert _robot_blocked(np.zeros(3),points,[box([0,0,0])]).all()


def test_visibility_uses_projection_robot_and_only_observed_occluders():
    obs=observation();points=np.array([[0.,0.,1.],[10.,0.,1.],[0,0,-1.]])
    assert _visibility(obs,points,np.eye(4),[],np.empty((0,3)))[-1].tolist()==[True,False,False]
    assert not _visibility(obs,points,np.eye(4),[box([0,0,.5])],np.empty((0,3)))[-1].any()
    assert not _visibility(obs,points,np.eye(4),[],np.array([[0.,0.,.5]]))[-1].any()
    # The target's own already observed endpoint is not a foreground occluder.
    assert _visibility(obs,points,np.eye(4),[],points[:1])[-1][0]


def test_current_view_is_not_recommended_and_no_visible_contact_means_no_ik(monkeypatch):
    obs,anchor,part=inputs();robot=KnownRobot()
    monkeypatch.setattr(advisor,'_orbit_views',lambda obs:[obs.view])
    result=suggest_reachable_views(robot,obs,anchor,part)
    assert result['views']==[] and not robot.calls
    monkeypatch.setattr(advisor,'_orbit_views',lambda obs:[(90.,15.,.8)])
    monkeypatch.setattr(robot,'visual_volumes',lambda q,g:[box(anchor,.03)])
    result=suggest_reachable_views(robot,obs,anchor,part)
    assert result['views']==[] and not robot.calls


def test_budget_and_input_failures_do_not_touch_robot():
    obs,anchor,part=inputs();robot=KnownRobot();before=robot.data.qpos.copy()
    for kwargs,reason in [({'budget':0},'no_advice_requested'),({'time_budget':0},'time_budget'),({'max_ik_calls':0},'ik_call_budget')]:
        assert suggest_reachable_views(robot,obs,anchor,part,**kwargs)['search']['stop_reason']==reason
    for bad in (np.array([np.nan,0,1.]),np.ones(2)):
        with pytest.raises(ValueError,match='finite observed'):
            suggest_reachable_views(robot,obs,bad,part)
    with pytest.raises(ValueError,match='tabletop'):
        suggest_reachable_views(robot,replace(obs,camera_id='wrist'),anchor,part)
    with pytest.raises(ValueError,match='budget'):
        suggest_reachable_views(robot,obs,anchor,part,max_ik_calls=-1)
    result=suggest_reachable_views(robot,obs,anchor,part[:3])
    assert result['search']['stop_reason']=='insufficient_observed_part'
    assert not robot.calls and np.array_equal(robot.data.qpos,before)


def test_time_budget_stops_search_and_preserves_partial_advice(monkeypatch):
    obs,anchor,part=inputs();robot=KnownRobot();before=robot.data.qpos.copy();clock=[0.]
    monkeypatch.setattr(advisor.time,'monotonic',lambda:clock[0])
    original=robot.ik
    def slow(*args,**kwargs):
        out=original(*args,**kwargs);clock[0]+=.7;return out
    monkeypatch.setattr(robot,'ik',slow)
    result=suggest_reachable_views(robot,obs,anchor,part,budget=3,time_budget=.5)
    assert len(result['views'])==1 and len(robot.calls)==1
    assert result['search']['stop_reason']=='time_budget'
    assert np.array_equal(robot.data.qpos,before)
