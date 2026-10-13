"""Nominal gripper evidence orders candidates; full planning stays mandatory."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from view2grasp.mapping import UNKNOWN,FREE,OCCUPIED
from view2grasp.planning import Planner,RobotModel
from view2grasp.types import GraspCandidate


def grasp(x=.18,angle=0.,score=0.):
    return GraspCandidate(np.array([x,0.,0.]),Rotation.from_rotvec([angle,0,0]).as_matrix(),
        .04,np.array([1.,0,0]),score,0.)


class Robot:
    def __init__(self):
        self.limits=np.tile([-20.,20.],(7,1));self.data=SimpleNamespace(contact=[])
        self.calls=[];self.local=np.array([[0.,0,0],[0.,1,0]]);self.pads=np.array([True,False])
        self.actual_shift=0.
    def gripper_local_points(self,q,gripper):return self.local.copy(),self.pads.copy()
    def pose(self,q=None):return np.zeros(3),np.eye(3)
    def ik(self,p,R,seed):
        self.calls.append(p.copy());q=np.zeros(7);q[0]=p[0];return q
    def points(self,q,gripper):return self.local+np.array([q[0]+self.actual_shift,0,0]),self.pads.copy()


def planner(rule):
    robot=Robot();map_=SimpleNamespace(classify=lambda p:np.asarray(rule(p),dtype=np.uint8),
        is_target=lambda p:np.ones(len(p),dtype=bool),contact_allowed=lambda p:np.ones(len(p),dtype=bool))
    return Planner(robot,map_,np.random.default_rng(0)),robot


def test_complete_approach_precheck_finds_obstacle_after_unknown_pregrasp():
    p,r=planner(lambda pts:np.where((pts[:,0]>.13)&(pts[:,0]<.14),OCCUPIED,UNKNOWN))
    g=grasp();assert p._geometry_priority(np.zeros(7),[g])==[]
    assert p.geometry_ranking['nominal_blocked_candidates']==1
    assert p.geometry_ranking['checked_candidates']==1
    assert not r.calls and p.checks==0 and not len(p.recovery_unknown())


def test_nominally_clear_candidate_outside_old_subset_is_searched_first():
    p,r=planner(lambda pts:np.where(pts[:,0]<.5,OCCUPIED,FREE))
    old=grasp();preferred=grasp(1.18,angle=1.)
    assert p._candidate_subset([old,preferred],np.eye(3),1)[0] is old
    p.approach=lambda q,g:np.array([q,q]);p.plan=lambda a,b:np.array([a,b])
    result=p.choose(np.zeros(7),[old,preferred],max_candidates=1,prioritize_gripper_geometry=True)
    assert result[0] is preferred and len(r.calls)==1 and p.checks==1
    assert p.geometry_ranking['selected_candidate_indices']==[1]
    assert p.geometry_ranking['selected_priority_candidates']==1


def test_nominal_obstacle_never_hard_rejects_a_tolerated_actual_ik_solution():
    p,r=planner(lambda pts:np.where(pts[:,0]<.5,OCCUPIED,FREE))
    r.actual_shift=1.  # Deliberately exaggerate nominal/IK discrepancy.
    g=grasp();p.approach=lambda q,g:np.array([q,q]);p.plan=lambda a,b:np.array([a,b])
    result=p.choose(np.zeros(7),[g],max_candidates=1,prioritize_gripper_geometry=True)
    assert result[0] is g and len(r.calls)==1
    assert p.geometry_ranking['nominal_blocked_candidates']==1
    assert p.geometry_ranking['selected_priority_candidates']==0


def test_unknown_is_a_priority_hypothesis_but_still_rejected_by_actual_planner():
    p,r=planner(lambda pts:np.full(len(pts),UNKNOWN));g=grasp()
    assert p.choose(np.zeros(7),[g],max_candidates=1,prioritize_gripper_geometry=True) is None
    assert p.geometry_ranking['nominal_clear_candidates']==1
    assert p.checks==8 and p.failure_counts['pregrasp: unknown space']==8
    assert p.last_failure=='unknown space'


@pytest.mark.parametrize('threshold,pads,clear',[(.1649,[True,True],False),(.17,[True,True],True),(.17,[True,False],False)])
def test_only_permitted_pads_in_original_last_12mm_get_contact_exception(threshold,pads,clear):
    p,r=planner(lambda pts:np.where(pts[:,0]>=threshold,OCCUPIED,UNKNOWN));r.pads=np.array(pads)
    g=grasp();assert bool(p._geometry_priority(np.zeros(7),[g])) is clear


def test_pad_near_independent_occupied_endpoint_still_blocks_when_contact_not_allowed():
    p,r=planner(lambda pts:np.where(pts[:,0]>=.17,OCCUPIED,UNKNOWN));r.pads[:]=True
    p.map.contact_allowed=lambda pts:np.zeros(len(pts),dtype=bool)
    assert not p._geometry_priority(np.zeros(7),[grasp()])


def test_precheck_consumes_original_deadline_and_partial_candidate_is_not_clear(monkeypatch):
    import view2grasp.planning as module
    clock=[0.];p,r=planner(lambda pts:np.full(len(pts),FREE))
    def classify(points):
        clock[0]+=.01;return np.full(len(points),FREE,dtype=np.uint8)
    p.map.classify=classify
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    ik_times=[];original_ik=r.ik
    def ik(*args):ik_times.append(clock[0]);return original_ik(*args)
    r.ik=ik;p.approach=lambda q,g:np.array([q,q]);p.plan=lambda a,b:np.array([a,b])
    a,b=grasp(),grasp(1.18)
    result=p.choose(np.zeros(7),[a,b],max_candidates=1,time_budget=1.,prioritize_gripper_geometry=True)
    d=p.geometry_ranking
    assert d['status']=='time_budget' and d['stop_reason']=='precheck_time_budget'
    assert d['incomplete_candidates']==1 and d['nominal_clear_candidates']==1 and d['checked_candidates']==1
    assert d['precheck_budget_seconds']==.25 and .25<=d['precheck_seconds']<.26
    assert ik_times and ik_times[0]<.26 and result[0] is a and p.checks==1
    # Neither the partial candidate nor its partially visited ray is certified.
    assert d['selected_priority_candidates']==1


def test_geometry_budget_never_extends_global_deadline(monkeypatch):
    import view2grasp.planning as module
    p,r=planner(lambda pts:np.full(len(pts),FREE));clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    def classify(points):
        clock[0]=1.1;return np.full(len(points),FREE,dtype=np.uint8)
    p.map.classify=classify
    assert p.choose(np.zeros(7),[grasp()],time_budget=1.,prioritize_gripper_geometry=True) is None
    assert p.geometry_ranking['stop_reason']=='planning_time_budget'
    assert p.geometry_ranking['incomplete_candidates']==1 and not r.calls
    assert p.last_failure=='planning budget exhausted: candidate geometry ranking'


def test_geometry_budget_is_capped_at_eight_seconds():
    p,r=planner(lambda pts:np.full(len(pts),FREE));r.ik=lambda *args:None
    p.choose(np.zeros(7),[grasp()],time_budget=120.,prioritize_gripper_geometry=True)
    assert p.geometry_ranking['precheck_budget_seconds']==8.


def test_priority_ik_diagnostics_distinguish_complete_failure_from_a_valid_pose():
    p,r=planner(lambda pts:np.full(len(pts),FREE));r.ik=lambda *args:None
    assert p.choose(np.zeros(7),[grasp(),grasp(1.18)],prioritize_gripper_geometry=True) is None
    d=p.geometry_ranking
    assert d['priority_pregrasp_ik_attempts']==16 and d['priority_candidates_completed']==2
    assert d['priority_pregrasp_ik_solutions']==0
    # A returned pose rejected only for UNKNOWN is not a kinematic failure.
    p,r=planner(lambda pts:np.full(len(pts),UNKNOWN))
    assert p.choose(np.zeros(7),[grasp()],prioritize_gripper_geometry=True) is None
    assert p.geometry_ranking['priority_pregrasp_ik_solutions']==8
    assert p.geometry_ranking['priority_candidates_completed']==1


def test_deadline_during_priority_ik_does_not_report_complete_failure(monkeypatch):
    import view2grasp.planning as module
    p,r=planner(lambda pts:np.full(len(pts),FREE));clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    def ik(*args):clock[0]+=.3;return None
    r.ik=ik
    assert p.choose(np.zeros(7),[grasp()],time_budget=1.,prioritize_gripper_geometry=True) is None
    assert p.geometry_ranking['priority_pregrasp_ik_attempts']==4
    assert p.geometry_ranking['priority_candidates_completed']==0
    assert p.last_failure=='planning budget exhausted: pregrasp IK'


def test_priority_fills_from_original_diverse_fallback_without_mutating_candidates():
    p,r=planner(lambda pts:np.where(pts[:,0]<.5,OCCUPIED,FREE))
    candidates=[grasp(.18+j*.008,angle=i*.1,score=i*.01+j*.008) for i in range(20) for j in range(3)]+[grasp(1.18,angle=2.)]
    before=[(g.position.copy(),g.rotation.copy(),g.approach.copy(),g.score) for g in candidates]
    fallback=p._candidate_subset(candidates,np.eye(3),12)
    r.ik=lambda *args:None
    assert p.choose(np.zeros(7),candidates,max_candidates=12,prioritize_gripper_geometry=True) is None
    selected=p.geometry_ranking['selected_candidate_indices']
    indices={id(g):i for i,g in enumerate(candidates)}
    expected=([60]+[indices[id(g)] for g in fallback if g is not candidates[60]])[:12]
    assert selected==expected and len(selected)<=12 and len(set(selected))==len(selected)
    for g,(pos,rot,ray,score) in zip(candidates,before):
        assert np.array_equal(g.position,pos) and np.array_equal(g.rotation,rot)
        assert np.array_equal(g.approach,ray) and g.score==score


def test_known_gripper_proxy_excludes_arm_and_restores_full_state(monkeypatch):
    import view2grasp.planning as module
    robot=object.__new__(RobotModel);saved=np.arange(10,dtype=float)
    robot.model=object();robot.data=SimpleNamespace(qpos=saved.copy())
    robot.parts=[(0,np.zeros((1,3)),False,'robot0_link_collision'),
        (1,np.zeros((2,3)),False,'gripper0_hand_collision'),
        (2,np.zeros((1,3)),True,'gripper0_pad_collision')]
    R=Rotation.from_rotvec([0,0,.7]).as_matrix();position=np.array([.2,.3,.4])
    expected=np.array([[.01,0,0],[.02,0,0],[.03,0,0]])
    def points(q,gripper):
        robot.data.qpos[:]=123.
        return np.vstack([[9.,9,9],expected@R.T+position]),np.array([False,False,False,True])
    robot.points=points;robot.pose=lambda:(position,R)
    monkeypatch.setattr(module.mujoco,'mj_forward',lambda model,data:None)
    local,pads=robot.gripper_local_points(np.zeros(7),np.array([.04,-.04]))
    assert np.allclose(local,expected) and pads.tolist()==[False,False,True]
    assert np.array_equal(robot.data.qpos,saved)
    def fail(*args):robot.data.qpos[:]=99.;raise ValueError('bad robot')
    robot.points=fail
    with pytest.raises(ValueError,match='bad robot'):robot.gripper_local_points(np.zeros(7))
    assert np.array_equal(robot.data.qpos,saved)


def test_geometry_priority_disabled_by_default_preserves_original_subset():
    p,r=planner(lambda pts:np.full(len(pts),FREE));a,b=grasp(),grasp(1.18,angle=1.)
    def forbidden(*args):raise AssertionError('default path must not run geometry priority')
    r.gripper_local_points=forbidden
    p.approach=lambda q,g:np.array([q,q]);p.plan=lambda a,b:np.array([a,b])
    assert p.choose(np.zeros(7),[a,b],max_candidates=1)[0] is a
    assert p.geometry_ranking['status']=='disabled'
