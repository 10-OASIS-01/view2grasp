"""View advice consumes effective refusals, never raw waived UNKNOWN cells."""
from types import SimpleNamespace
import numpy as np
import pytest

from view2grasp.mapping import FREE, UNKNOWN, OCCUPIED
from view2grasp.planning import Planner, PlanningBudgetExceeded


def make_planner(points, states, pads=None):
    points=np.asarray(points,dtype=float).reshape(-1,3)
    robot=SimpleNamespace(limits=np.array([[-2.,2.]]),data=SimpleNamespace(contact=[]),
        points=lambda q,g:(points.copy(),np.zeros(len(points),dtype=bool) if pads is None else pads),
        pose=lambda:(np.zeros(3),np.eye(3)))
    map_=SimpleNamespace(classify=lambda p:np.asarray(states,dtype=np.uint8).copy(),
        contact_allowed=lambda p:np.zeros(len(p),dtype=bool),
        is_target=lambda p:np.zeros(len(p),dtype=bool))
    return Planner(robot,map_,np.random.default_rng(0))


def test_effective_robot_refusal_excludes_permitted_contact_and_resets():
    planner=make_planner([[0,0,0],[1,0,0],[2,0,0]],
        [UNKNOWN,OCCUPIED,FREE],np.ones(3,dtype=bool))
    planner.map.contact_allowed=lambda p:np.array([True,False,False])
    assert not planner.valid(np.array([.2]),contact=True)
    report=planner.last_validation
    assert report['complete'] and report['valid'] is False
    assert report['robot_unknown_count']==0 and report['robot_unknown_points']==[]
    assert report['robot_occupied_points']==[[1.,0.,0.]]
    planner.map.classify=lambda p:np.full(len(p),FREE,dtype=np.uint8)
    assert planner.valid(np.array([.3]),contact=True)
    assert planner.last_validation['valid'] is True
    assert planner.last_validation['failure']==''
    assert planner.last_validation['robot_occupied_points']==[]
    assert planner.last_validation['q']==[.3]


def test_payload_refusal_keeps_effective_state_and_exact_pose_provenance():
    planner=make_planner([],[])
    planner.attached=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    planner.map.classify=lambda p:np.array([UNKNOWN,UNKNOWN,OCCUPIED],dtype=np.uint8) if len(p) else np.empty(0,dtype=np.uint8)
    planner.map.is_payload_source=lambda p:np.array([True,False,True])
    assert not planner.valid(np.array([.1]))
    report=planner.last_validation
    assert report['payload_unknown_points']==[[1.,0.,0.]]
    assert report['payload_occupied_points']==[[2.,0.,0.]]
    assert report['payload_unknown_count']==report['payload_occupied_count']==1
    assert planner.payload_invalid['q']==[.1]
    assert planner.payload_invalid['point_states']==[UNKNOWN,OCCUPIED]
    assert planner.payload_invalid['payload_point_indices']==[1,2]
    # A subsequent early robot refusal must not carry the previous payload's
    # samples into current view advice. Historical payload diagnostics remain.
    planner.robot.points=lambda q,g:(np.array([[3.,0.,0.]]),np.array([False]))
    planner.map.classify=lambda p:np.full(len(p),OCCUPIED,dtype=np.uint8)
    assert not planner.valid(np.array([.2]))
    assert planner.last_validation['robot_occupied_points']==[[3.,0.,0.]]
    assert planner.last_validation['payload_unknown_points']==[]
    assert planner.payload_invalid['q']==[.1]


def test_deadline_marks_new_validation_incomplete():
    planner=make_planner([],[])
    assert planner.valid(np.zeros(1))
    planner.deadline=0.
    with pytest.raises(PlanningBudgetExceeded):planner.valid(np.array([.5]))
    assert planner.last_validation['complete'] is False
    assert planner.last_validation['valid'] is None
    assert planner.last_validation['q']==[.5]


def test_diagnostic_sample_limit_does_not_understate_total_refusals():
    planner=make_planner(np.zeros((250,3)),np.full(250,UNKNOWN))
    assert not planner.valid(np.zeros(1))
    assert planner.last_validation['robot_unknown_count']==250
    assert len(planner.last_validation['robot_unknown_points'])==200
