"""Phase reserves and transfer recovery use actual, current UNKNOWN evidence."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from view2grasp.mapping import FREE, OCCUPIED, UNKNOWN
from view2grasp.planning import PlanningBudgetExceeded
from view2grasp.transfer_recovery import plan_transfer_attempt, tabletop_observation_budget


def test_named_config_explicitly_reserves_postlift_without_changing_default_protocol():
    from pathlib import Path
    from view2grasp.config import ExperimentConfig, load_config
    default = ExperimentConfig()
    development = load_config(Path(__file__).parents[1] / 'configs/occluded-transfer.yaml')
    assert default.max_observations == 6 and default.post_lift_observations == 0
    assert development.max_observations == 10 and development.post_lift_observations == 2
    assert development.image_size == 1024
    assert development.refine_grasp_grounding is False
    assert development.voxel_resolution_m == default.voxel_resolution_m
    with pytest.raises(ValueError, match='initial and a post-close'):
        ExperimentConfig(max_observations=3, post_lift_observations=2)


@pytest.mark.parametrize('task,maximum,reserve,expected',[
    ('occluded',6,0,(4,6,2,0)),('occluded',8,0,(6,8,2,0)),
    ('occluded',10,2,(6,8,2,2)),('occluded',10,1,(7,9,2,1)),
    ('mug',10,2,(7,10,3,0)),('block',6,0,(4,6,2,0)),
])
def test_phase_limits_reserve_postlift_before_preclose_recovery(task,maximum,reserve,expected):
    budget = tabletop_observation_budget(task,maximum,reserve)
    assert (budget.pregrasp_limit,budget.postclose_limit,budget.postclose_observations,
        budget.postlift_observations) == expected


@pytest.mark.parametrize('maximum,reserve',[(2,1),(3,2),(10,-1),(10,3)])
def test_invalid_budget_cannot_consume_initial_or_postclose_frame(maximum,reserve):
    with pytest.raises(ValueError):tabletop_observation_budget('occluded',maximum,reserve)


def fake_planner(outcome='unknown'):
    planner = SimpleNamespace(rng=np.random.default_rng(3),deadline=None,checks=17,
        last_failure='stale observed obstacle',closest_by_failure={'unknown space':{'points':[[99.,0,0]]}},
        payload_invalid={'unknown':1,'points':[[99.,0,0]]})
    planner.robot = SimpleNamespace(ik=lambda *args: None if outcome=='ik' else np.ones(7))
    planner.map = SimpleNamespace(classify=lambda p:np.array([
        UNKNOWN if q[0] == 1 else OCCUPIED if q[0] == 2 else FREE for q in p],dtype=np.uint8))
    planner.recovery_unknown = lambda: np.array([[99.,0,0]])
    planner.random_draws=[]
    def plan(start,goal,timeout):
        assert 0 < timeout <= 5
        assert planner.deadline is not None
        assert planner.closest_by_failure == {} and planner.payload_invalid is None
        planner.random_draws.append(float(planner.rng.random()))
        planner.checks += 2
        if outcome=='budget':raise PlanningBudgetExceeded
        if outcome=='exception':raise RuntimeError('unexpected failure')
        if outcome=='success':return np.array([start,goal])
        planner.closest_by_failure={'unknown space':{'points':[[1.,0,0],[2.,0,0],[3.,0,0]],
            'count':3,'unknown':1}}
        planner.last_failure = 'observed obstacle' if outcome=='occupied' else 'RRT timeout'
        return None
    planner.plan=plan
    return planner


def attempt(planner):
    return plan_transfer_attempt(planner,np.zeros(7),np.zeros(3),np.eye(3),time_budget=10.)


def test_retry_unknowns_are_phase_local_and_reclassified_without_occupied_samples():
    planner=fake_planner();result=attempt(planner)
    assert result.path is None and result.report['observation_recovery_eligible']
    assert result.unknown_points.tolist()==[[1.,0,0]]
    assert result.report['checks']==2 and result.report['elapsed_seconds']>=0
    assert planner.deadline is None
    # Restoring the transfer checkpoint, after a real observation in the
    # runner, repeats this same search rather than relying on a new lucky RNG.
    planner.rng.bit_generator.state=copy.deepcopy(result.rng_state)
    attempt(planner)
    assert planner.random_draws[0]==planner.random_draws[1]


def test_known_obstacle_does_not_trigger_observation_retry_from_other_unknown_branch():
    result=attempt(fake_planner('occupied'))
    assert result.report['failure']=='observed obstacle'
    assert result.unknown_points.tolist()==[[1.,0,0]]
    assert not result.report['observation_recovery_eligible']


def test_ik_failure_clears_stale_diagnostics_and_does_not_claim_unknown_recovery():
    result=attempt(fake_planner('ik'))
    assert result.report['failure']=='IK'
    assert result.report['closest_by_failure']=={} and result.report['payload_invalid'] is None
    assert not result.report['observation_recovery_eligible']
    assert len(result.unknown_points)==0


@pytest.mark.parametrize('outcome',['budget','exception'])
def test_deadline_is_restored_when_attempt_raises(outcome):
    planner=fake_planner(outcome)
    if outcome=='exception':
        with pytest.raises(RuntimeError,match='unexpected'):attempt(planner)
    else:
        result=attempt(planner)
        assert result.report['failure']=='planning budget exhausted'
        assert not result.report['observation_recovery_eligible']
    assert planner.deadline is None


def test_success_returns_only_planners_checked_path_and_no_recovery_request():
    result=attempt(fake_planner('success'))
    assert result.path.shape==(2,7)
    assert result.report['path_found'] and result.report['failure']==''
    assert not result.report['observation_recovery_eligible'] and not len(result.unknown_points)
