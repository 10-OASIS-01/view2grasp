"""Fair multistart ordering retains the shared budget and all motion checks."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from view2grasp.mapping import FREE, OCCUPIED, UNKNOWN
from view2grasp.planning import Planner, PlanningBudgetExceeded
from view2grasp.types import GraspCandidate


def candidates():
    return [GraspCandidate(np.array([x+.08,0.,0.]),np.eye(3),.04,
        np.array([1.,0.,0.]),float(i),0.) for i,x in enumerate((.2,.6))]


class Robot:
    def __init__(self,grasps):
        self.limits=np.tile([-3.,3.],(7,1))
        self.data=SimpleNamespace(contact=[])
        self.pre=[g.position-g.approach*.08 for g in grasps]
        self.pre_calls=[];self.all_calls=[];self.pre_callback=None

    def pose(self,q=None):return np.zeros(3),np.eye(3)

    def gripper_local_points(self,q,gripper):
        return np.zeros((1,3)),np.array([False])

    def points(self,q,gripper):return np.array([[q[0],0.,0.]]),np.array([False])

    def ik(self,position,rotation,seed):
        self.all_calls.append((position.copy(),seed.copy()))
        index=next((i for i,p in enumerate(self.pre) if np.allclose(position,p,atol=1e-10)),None)
        if index is not None:
            self.pre_calls.append((index,seed.copy()))
            if self.pre_callback is not None:return self.pre_callback(index,position,seed)
        q=np.zeros(7);q[0]=position[0];return q


def setup(state=FREE,rng_seed=29):
    grasps=candidates();robot=Robot(grasps)
    map_=SimpleNamespace(classify=lambda points:np.full(len(points),state,dtype=np.uint8),
        contact_allowed=lambda points:np.zeros(len(points),bool),
        is_target=lambda points:np.zeros(len(points),bool))
    return Planner(robot,map_,np.random.default_rng(rng_seed)),robot,grasps


def calls(robot):return [index for index,_ in robot.pre_calls]


@pytest.mark.parametrize('diversify,expected',[(False,[0]*8+[1]),(True,[0,1])])
def test_later_candidate_can_succeed_after_first_transit_fails(diversify,expected):
    planner,robot,grasps=setup();transits=[]
    def plan(start,goal):
        index=0 if np.isclose(goal[0],.2) else 1
        transits.append(index)
        if index==0:planner.last_failure='RRT timeout';return None
        return np.array([start,goal])
    planner.plan=plan
    result=planner.choose(np.zeros(7),grasps,prioritize_gripper_geometry=True,
        diversify_attempts=diversify)
    assert result[0] is grasps[1] and len(result[2])==17
    assert calls(robot)==transits==expected
    # The real approach calls feasible IK, valid and edge for every step.
    assert len(robot.all_calls)==17*len(expected)
    assert planner.checks==49*len(expected)
    assert planner.failure_counts=={'RRT timeout':expected.count(0)}
    info=planner.geometry_ranking
    assert info['selected_candidate_indices']==[0,1]
    assert info['priority_pregrasp_ik_attempts']==info['priority_pregrasp_ik_solutions']==len(expected)
    assert info['priority_candidates_completed']==(0 if diversify else 1)
    assert planner.deadline is None


@pytest.mark.parametrize('diversify',[False,True])
def test_each_candidate_keeps_eight_attempts_seed_scales_and_limit_clipping(diversify):
    planner,robot,grasps=setup();robot.pre_callback=lambda *args:None
    initial=np.linspace(2.7,2.9,7)
    assert planner.choose(initial,grasps,prioritize_gripper_geometry=True,
        diversify_attempts=diversify) is None
    order=([(i,attempt) for attempt in range(8) for i in range(2)] if diversify
        else [(i,attempt) for i in range(2) for attempt in range(8)])
    expected_rng=np.random.default_rng(29)
    for (index,seed),(expected_index,attempt) in zip(robot.pre_calls,order,strict=True):
        expected=(initial if attempt==0 else np.clip(initial+
            expected_rng.normal(0,.25 if attempt<4 else .8,7),-2.98,2.98))
        assert index==expected_index and np.array_equal(seed,expected)
    assert Counter(calls(robot))=={0:8,1:8}
    info=planner.geometry_ranking
    assert info['priority_pregrasp_ik_attempts']==16 and info['priority_pregrasp_ik_solutions']==0
    assert info['priority_candidates_completed']==2
    assert info['attempt_order']==('round_robin' if diversify else 'candidate_major')
    assert planner.failure_counts=={'pregrasp IK':16}


def test_omitted_option_keeps_original_candidate_major_order():
    planner,robot,grasps=setup();robot.pre_callback=lambda *args:None
    assert planner.choose(np.zeros(7),grasps) is None
    assert calls(robot)==[0]*8+[1]*8
    assert planner.geometry_ranking['attempt_order']=='candidate_major'


def test_rounds_reuse_ranked_subset_including_nonpriority_fallback(monkeypatch):
    planner,robot,grasps=setup();robot.pre_callback=lambda *args:None
    def priority(q,values):
        planner.geometry_ranking={'status':'complete'}
        return [values[1]]
    monkeypatch.setattr(planner,'_geometry_priority',priority)
    before=[(g.position.copy(),g.rotation.copy(),g.approach.copy(),g.score) for g in grasps]
    assert planner.choose(np.zeros(7),grasps,prioritize_gripper_geometry=True,
        diversify_attempts=True) is None
    assert calls(robot)==[1,0]*8
    info=planner.geometry_ranking
    assert info['selected_candidate_indices']==[1,0] and info['selected_priority_candidates']==1
    assert info['priority_pregrasp_ik_attempts']==8 and info['priority_candidates_completed']==1
    for grasp,(position,rotation,approach,score) in zip(grasps,before,strict=True):
        assert np.array_equal(grasp.position,position) and np.array_equal(grasp.rotation,rotation)
        assert np.array_equal(grasp.approach,approach) and grasp.score==score


@pytest.mark.parametrize('diversify,succeeds',[(False,False),(True,True)])
def test_round_robin_does_not_extend_the_shared_budget(monkeypatch,diversify,succeeds):
    import view2grasp.planning as module
    planner,robot,grasps=setup();clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    deadlines=[]
    def plan(start,goal):
        deadlines.append(planner.deadline)
        if np.isclose(goal[0],.2):
            clock[0]+=4.;planner.last_failure='RRT timeout';return None
        return np.array([start,goal])
    planner.plan=plan
    result=planner.choose(np.zeros(7),grasps,time_budget=5.,
        prioritize_gripper_geometry=True,diversify_attempts=diversify)
    assert (result is not None)==succeeds
    assert calls(robot)==([0,1] if diversify else [0,0])
    assert deadlines==[5.,5.] and planner.deadline is None
    assert planner.geometry_ranking['priority_candidates_completed']==0
    if not succeeds:assert planner.last_failure=='planning budget exhausted: RRT timeout'


@pytest.mark.parametrize('limit,completed',[(2,0),(14,0),(15,1)])
def test_budget_interruption_counts_only_candidates_with_all_eight_failures(monkeypatch,limit,completed):
    import view2grasp.planning as module
    planner,robot,grasps=setup();clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    def fail(*args):clock[0]+=1.;return None
    robot.pre_callback=fail
    assert planner.choose(np.zeros(7),grasps,time_budget=limit,
        prioritize_gripper_geometry=True,diversify_attempts=True) is None
    assert calls(robot)==([0,1]*8)[:limit]
    info=planner.geometry_ranking
    assert info['priority_pregrasp_ik_attempts']==limit
    assert info['priority_candidates_completed']==completed
    assert planner.failure_counts=={'pregrasp IK':limit}
    assert planner.last_failure=='planning budget exhausted: pregrasp IK' and planner.deadline is None


@pytest.mark.parametrize('phase',['valid','approach','plan'])
def test_exception_inside_final_attempt_does_not_mark_it_completed(monkeypatch,phase):
    planner,robot,grasps=setup()
    real_valid=planner.valid
    def valid(q,contact=False):
        if phase=='valid' and len(robot.pre_calls)==15:raise PlanningBudgetExceeded
        return real_valid(q,contact)
    def approach(q,g):
        if phase=='approach' and len(robot.pre_calls)==15:raise PlanningBudgetExceeded
        return np.array([q,q])
    def plan(a,b):
        if phase=='plan' and len(robot.pre_calls)==15:raise PlanningBudgetExceeded
        planner.last_failure='RRT timeout';return None
    monkeypatch.setattr(planner,'valid',valid)
    monkeypatch.setattr(planner,'approach',approach)
    monkeypatch.setattr(planner,'plan',plan)
    assert planner.choose(np.zeros(7),grasps,prioritize_gripper_geometry=True,
        diversify_attempts=True) is None
    assert len(robot.pre_calls)==15
    info=planner.geometry_ranking
    assert info['priority_pregrasp_ik_attempts']==info['priority_pregrasp_ik_solutions']==15
    assert info['priority_candidates_completed']==0
    assert planner.failure_counts=={'RRT timeout':14} and planner.deadline is None


def test_success_on_eighth_attempt_is_not_reported_as_exhausted():
    planner,robot,grasps=setup();counts=Counter()
    planner.approach=lambda q,g:np.array([q,q])
    def plan(a,b):
        index=0 if np.isclose(b[0],.2) else 1;counts[index]+=1
        if index==0 and counts[index]==8:return np.array([a,b])
        planner.last_failure='RRT timeout';return None
    planner.plan=plan
    result=planner.choose(np.zeros(7),grasps,prioritize_gripper_geometry=True,diversify_attempts=True)
    assert result[0] is grasps[0] and calls(robot)==[0,1]*7+[0]
    assert planner.geometry_ranking['priority_candidates_completed']==0
    assert planner.geometry_ranking['priority_pregrasp_ik_attempts']==15


@pytest.mark.parametrize('state,reason',[(UNKNOWN,'unknown space'),(OCCUPIED,'observed obstacle')])
def test_round_robin_does_not_bypass_pregrasp_collision_or_unknown_checks(state,reason):
    planner,robot,grasps=setup(state)
    def forbidden(*args):raise AssertionError('invalid pregrasp must stop before approach/transit')
    planner.approach=forbidden;planner.plan=forbidden
    assert planner.choose(np.zeros(7),grasps,diversify_attempts=True) is None
    assert calls(robot)==[0,1]*8 and planner.checks==16
    assert planner.failure_counts=={'pregrasp: '+reason:16}


def test_approach_ik_failure_still_blocks_transit_for_every_candidate():
    planner,robot,grasps=setup();original=robot.ik
    def ik(position,rotation,seed):
        if not any(np.allclose(position,p,atol=1e-10) for p in robot.pre):return None
        return original(position,rotation,seed)
    robot.ik=ik
    def forbidden(*args):raise AssertionError('failed approach must not enter transit')
    planner.plan=forbidden
    assert planner.choose(np.zeros(7),grasps,prioritize_gripper_geometry=True,
        diversify_attempts=True) is None
    assert calls(robot)==[0,1]*8
    assert planner.failure_counts=={'approach: IK':16}
    assert planner.geometry_ranking['priority_candidates_completed']==2
