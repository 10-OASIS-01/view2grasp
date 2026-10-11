"""The reserved ninth and tenth real frames gate held-object transfer."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_contact_reassociation import annotations, scene
from view2grasp.types import GraspCandidate


def run_transfer_fixture(tmp_path, monkeypatch, *, outcome='recover', reserve=2):
    from view2grasp import recording, runner
    from view2grasp.grounding import GroundingResolver

    obs, xyz, mask, _ = scene()
    target = xyz[mask]
    part_mask = mask.copy(); part_mask[:, 10:] = False
    part = xyz[part_mask]
    candidate = GraspCandidate(np.zeros(3), np.eye(3), .02, np.array([0., 0., 1.]),
        0., 0., surface_point=target[0], grasp_center=target[0])
    evidence = SimpleNamespace(events=[], observations=[], associations=[], associated_clouds={}, retired=[], targets=[],
        random_draws=[], plans=[], advice=[], sources=[], held_requests=[], camera_moves=[], clock=0., steps=0)
    monkeypatch.setattr(runner.time, 'monotonic', lambda: evidence.clock)

    class Sim:
        def __init__(self, *args):
            self.q = np.zeros(7); self.gripper_q = np.array([.02, -.02])
            self.robot_xml = ''; self.joint_names = []; self.eef_name = ''
            self.phase = 'observe'; self.frame = 0; self.view = obs.view
            evidence.sim = self
        def observe(self):
            self.frame += 1
            evidence.observations.append((self.frame, self.phase, evidence.steps))
            return replace(obs, frame_id=self.frame, q=self.q.copy(), gripper_q=self.gripper_q.copy(), view=self.view)
        def move_camera(self, *args):
            evidence.camera_moves.append((self.frame, tuple(args), evidence.steps)); self.view = tuple(args)
        def validate_initial_proxies(self, *args): pass
        def step(self, q, grip): self.q = np.asarray(q).copy(); evidence.steps += 1
        def close(self): pass

    class Robot:
        def __init__(self, *args): pass
        def points(self, *args): return np.zeros((1, 3)), np.zeros(1, dtype=bool)
        def volumes(self, *args): return ()
        def visual_volumes(self, *args): return ()
        def depth(self, sample): return np.full_like(sample.depth, 10.)
        def pose(self, q=None): return evidence.sim.q[:3].copy() if q is None else q[:3].copy(), np.eye(3)
        def grasp_center_offset(self): return np.zeros(3)
        def ik(self, p, R, q): return np.r_[p, q[3:]]
        def set_q(self, *args): pass
        def close(self): pass

    class Map:
        def __init__(self, *args):
            self.resolution = .005; self.table_z = .95; self.surface = target.copy(); self.views = []
        def certify_home(self, *args): pass
        def certify_robot_state(self, *args): pass
        def integrate(self, sample, *args): self.views.append(sample)
        def set_table(self, z): self.table_z = z
        def classify(self, points): return np.zeros(len(points), dtype=np.uint8)
        def set_target(self, points): evidence.targets.append((evidence.sim.frame, points.copy()))
        def set_payload_source(self, points): evidence.sources.append((evidence.sim.frame, points.copy()))
        def remove_target_endpoints(self, points): evidence.retired.append((evidence.sim.frame, points.copy()))
        def clear_dynamic(self): pass

    class Plan:
        add_observed_payload = runner.Planner.add_observed_payload
        payload_collision_local = runner.Planner.payload_collision_local
        def __init__(self, robot, map_, rng):
            self.robot = robot; self.map = map_; self.rng = rng; self.deadline = None
            self.checks = 0; self.failure_counts = {}; self.last_failure = ''
            self.closest_invalid = None; self.closest_by_failure = {}; self.payload_invalid = None
            evidence.planner = self
        def choose(self, q, candidates, **kwargs):
            return candidates[0], np.array([q, q]), np.array([q, q])
        def recovery_unknown(self): return np.empty((0, 3))
        def begin_payload_lift(self): pass
        def end_payload_lift(self): pass
        def feasible_ik(self, p, R, q, **kwargs): return self.robot.ik(p, R, q)
        def edge(self, start, goal, *, contact):
            assert contact
            if outcome == 'lower_failure':
                evidence.clock += 3.; self.last_failure = 'observed obstacle'; return False
            return True
        def plan(self, start, goal, timeout):
            assert 0 < timeout <= 5 and self.deadline is not None
            evidence.clock += 7.
            evidence.random_draws.append(self.rng.random())
            evidence.plans.append((evidence.sim.frame, self.attached.copy(), self.gripper.copy()))
            self.checks += 1
            if outcome == 'exception': raise RuntimeError('transfer test exception')
            if len(evidence.plans) == 1 and outcome not in ('lower_failure', 'held_recover_success') or outcome == 'residual':
                self.last_failure = 'observed obstacle' if outcome == 'occupied' else 'RRT timeout'
                self.closest_by_failure = {'unknown space': {'points': [[.7, .8, 1.2]], 'unknown': 1}}
                return None
            return np.array([start, goal])

    class Decisions:
        def __init__(self, *args, **kwargs):
            self.calls = 0; self.seconds = 0.; self.client = SimpleNamespace(close=lambda: None)
        def decide(self, sample, instruction, history, **kwargs):
            self.calls += 1
            result = annotations(sample, mask, part_mask)
            if history and 'held_verification' in history[-1]:
                evidence.held_requests.append((sample.frame_id, history[-1], kwargs))
                result = result.model_copy(update={'azimuth': 90.})
                if outcome == 'held_abort': result = result.model_copy(update={'action': 'abort'})
                if outcome == 'held_stale': result = result.model_copy(update={'frame_id': 8})
                if outcome == 'held_repeat': result = result.model_copy(update={'azimuth': sample.view[0]})
                return result
            if sample.frame_id == 6 and not kwargs.get('force_observe') and outcome != 'pregrasp_overrun':
                return result.model_copy(update={'action': 'grasp', 'place_point': (.5, .5)})
            if kwargs.get('observe_purpose') != 'annotation' and sample.frame_id >= 9:
                # Camera/view work is allowed to draw RNG; transfer retry must
                # still replay its own original search checkpoint.
                evidence.planner.rng.random(13)
            return result

    resolver = GroundingResolver()
    resolver.resolve_grasp = lambda sample, decision, *args: SimpleNamespace(
        decision=decision, object_mask=mask, part_mask=part_mask, report={'kinds': ['mask']})
    resolver.track = lambda *args, **kwargs: pytest.fail('Loose tracking cannot authorize held contacts')
    def associate(sample, annotation, original_object, original_part, identity, *args, source_supported=False):
        evidence.associations.append((sample.frame_id, source_supported, original_object.copy(), original_part.copy()))
        if (sample.frame_id == 9 and outcome.startswith('held_')
                or sample.frame_id == 10 and outcome in ('recovery_failure', 'held_failure')):
            raise ValueError('unverified held surface')
        # Held views reveal new sides, not merely subsets of the original
        # cloud. They must enter collision checks without moving fixed anchors
        # or turning the whole object's surfaces into permitted-part contacts.
        current_object = original_object.copy()
        if sample.frame_id >= 9:
            delta = [0., 0., -.014] if sample.frame_id == 9 else [0., .020, 0.]
            current_object = np.vstack([current_object, current_object[0] + delta])
        if outcome == 'low_lift':current_object = current_object - [0., 0., .13]
        current_part = original_part[::2] if sample.frame_id == 10 else original_part
        evidence.associated_clouds[sample.frame_id] = (current_object.copy(), current_part.copy())
        return SimpleNamespace(object_points=current_object.copy(), part_points=current_part.copy(),
            object_mask=mask, part_mask=part_mask, report={'frame_id': sample.frame_id})
    resolver.reassociate_contact = associate
    def views(robot, sample, unknown, **kwargs):
        evidence.advice.append((sample.frame_id, np.asarray(unknown).copy(), kwargs))
        return {'views': [[0., 55., .8]]}
    def no_crop(*args, **kwargs): raise ValueError('Use original fixture image')

    for name, value in [('Simulator', Sim), ('RobotModel', Robot), ('DepthMap', Map),
            ('Planner', Plan), ('VLM', Decisions)]: monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner.GroundingResolver, 'from_config', lambda config: resolver)
    monkeypatch.setattr(runner, 'Evaluator', lambda sim: SimpleNamespace(result=lambda: {
        'success': False, 'collision': False, 'first_grasp_success': True, 'collision_free_success': False}))
    monkeypatch.setattr(runner, 'grasp_candidates', lambda *args: ([candidate], target.copy()))
    monkeypatch.setattr(runner, 'part_cloud', lambda *args: part.copy())
    monkeypatch.setattr(runner, 'fit_table', lambda *args: .95)
    monkeypatch.setattr(runner, 'nominal_grasp', lambda *args: None)
    monkeypatch.setattr(runner, 'suggest_views', views)
    monkeypatch.setattr(runner, 'crop_observation', no_crop)
    monkeypatch.setattr(runner, 'time_parameterize', lambda path: path)
    monkeypatch.setattr(recording, 'save_grounding_result', lambda *args: None)
    monkeypatch.setattr(runner, 'save_episode', lambda out, sim, record, events: evidence.events.extend(events))
    placement = SimpleNamespace(transfer_position=np.array([.2, .2, .13]),
        release_position=np.array([.2, .2, .02]), rotation=np.eye(3), diagnostic=lambda: {})
    monkeypatch.setattr(runner, 'tabletop_placement', lambda **kwargs: placement)
    result = runner.run_episode('occluded', 3, 'active', tmp_path, replay=tmp_path / 'offline',
        config=runner.ExperimentConfig(max_observations=10 if reserve else 8,
            post_lift_observations=reserve, perception_backend='rgbd'))
    evidence.result = result; evidence.out = tmp_path / 'occluded_active_0003'
    return evidence


def executed(evidence, phase):
    return any(e['kind'] == 'execute' and e['phase'] == phase for e in evidence.events)


def test_reserved_real_held_then_unknown_view_strictly_gates_same_search_retry(tmp_path, monkeypatch):
    e = run_transfer_fixture(tmp_path, monkeypatch)
    assert [frame for frame, _, _ in e.observations] == list(range(1, 11))
    assert [(frame, supported) for frame, supported, _, _ in e.associations] == [(7, True), (8, True), (9, False), (10, False)]
    assert e.observations[8][2] == e.observations[9][2]  # Camera-only recovery.
    assert [frame for frame, _, _ in e.plans] == [9, 10]
    assert e.random_draws[0] == e.random_draws[1]
    assert np.array_equal(e.plans[0][1], e.plans[1][1]) and np.array_equal(e.plans[0][2], e.plans[1][2])
    assert np.array_equal(e.associations[2][2], e.associations[3][2])
    assert np.array_equal(e.associations[2][3], e.associations[3][3])
    assert [p.tolist() for frame, p in e.retired if frame == 10] == [e.associated_clouds[10][0].tolist()]
    assert [p.tolist() for frame, p in e.targets if frame == 10] == [e.associated_clouds[10][1].tolist()]
    # Both newly exposed surfaces survive in the collision-only envelope and
    # receive only the existing narrow observed-source certificate.
    from scipy.spatial import cKDTree
    local = e.planner.payload_collision_local()
    source = [p for frame, p in e.sources if frame == 10][-1]
    for frame in (9, 10):
        new_world = e.associated_clouds[frame][0][-1]
        assert cKDTree(source).query(new_world)[0] < 1e-12
        assert cKDTree(local).query(new_world - [0., 0., .13])[0] < 1e-12
    assert e.advice[-1][1].tolist() == [[.7, .8, 1.2]]
    assert e.advice[-1][2]['required_points'].tolist() == [[.7, .8, 1.2]]
    assert executed(e, 'transfer') and executed(e, 'lower')
    assert e.result['planning_seconds'] == 14.
    reports = [json.loads((e.out / f'transfer_planning_diagnostic_{i:03d}.json').read_text()) for i in (0, 1)]
    assert [r['rng_restored'] for r in reports] == [False, True]
    assert [r['path_found'] for r in reports] == [False, True]


@pytest.mark.parametrize('outcome,frames,attempts',[
    ('held_failure', 10, 0), ('recovery_failure', 10, 1), ('occupied', 9, 1), ('residual', 10, 2),
    ('exception', 9, 1), ('pregrasp_overrun', 6, 0),
])
def test_no_motion_or_budget_overrun_when_evidence_or_planning_fails(tmp_path, monkeypatch, outcome, frames, attempts):
    e = run_transfer_fixture(tmp_path, monkeypatch, outcome=outcome)
    assert e.result['observations'] == frames and len(e.plans) == attempts
    assert not executed(e, 'transfer')
    assert e.result['planning_seconds'] >= 7 * attempts
    assert e.planner.deadline is None
    if outcome == 'recovery_failure':
        assert not any(frame == 10 for frame, _ in e.retired)
        assert not any(frame == 10 for frame, _ in e.targets)
    if outcome == 'held_failure':
        assert executed(e, 'lower')
        assert not any(frame >= 9 for frame, _ in e.retired)
        assert not any(frame >= 9 for frame, _ in e.targets)
        assert e.planner.observed_payload_local is None
    if outcome == 'exception':
        report = json.loads((e.out / 'transfer_planning_diagnostic_000.json').read_text())
        assert report['failure'] == 'transfer test exception' and report['elapsed_seconds'] == 7.


def test_default_protocol_adds_no_postlift_or_transfer_recovery_frame(tmp_path, monkeypatch):
    e = run_transfer_fixture(tmp_path, monkeypatch, reserve=0)
    assert e.result['observations'] == 8
    assert [frame for frame, _, _ in e.plans] == [8]
    assert [(frame, supported) for frame, supported, _, _ in e.associations] == [(7, True), (8, True)]
    assert not executed(e, 'transfer')


def test_lower_failure_restores_support_flag_and_counts_failed_planning(tmp_path, monkeypatch):
    e = run_transfer_fixture(tmp_path, monkeypatch, outcome='lower_failure')
    assert executed(e, 'transfer') and not executed(e, 'lower')
    assert not e.planner.allow_payload_support
    assert e.result['planning_seconds'] == 10.


@pytest.mark.parametrize('outcome,transfer_passes', [('held_recover_success', True), ('held_recover_unknown', False)])
def test_failed_held_annotation_can_use_last_real_view_but_no_transfer_observation(tmp_path, monkeypatch, outcome, transfer_passes):
    e = run_transfer_fixture(tmp_path, monkeypatch, outcome=outcome)
    assert e.result['observations'] == 10
    assert [(frame, supported) for frame, supported, _, _ in e.associations] == [(7, True), (8, True), (9, False), (10, False)]
    assert np.array_equal(e.associations[2][2], e.associations[3][2])
    assert np.array_equal(e.associations[2][3], e.associations[3][3])
    assert e.observations[8][2] == e.observations[9][2]
    assert e.camera_moves[-1][0:2] == (9, (90., 55., .8))
    assert [frame for frame, _, _ in e.plans] == [10]
    assert executed(e, 'transfer') is transfer_passes
    assert not any(frame == 9 for frame, _ in e.retired)
    assert not any(frame == 9 for frame, _ in e.targets)
    assert [p.tolist() for frame, p in e.targets if frame == 10] == [e.associated_clouds[10][1].tolist()]
    frame, request, kwargs = e.held_requests[0]
    assert frame == 9 and kwargs['force_observe']
    assert request['held_verification']['remaining_transfer_mapping_observations_after_this_view'] == 0
    assert request['coverage_advice']['purpose'] == 'held_object_verification'
    assert not len(e.advice[-1][1])  # No invented UNKNOWN-path evidence.
    assert 'required_points' not in e.advice[-1][2]
    assert np.array_equal(e.advice[-1][2]['semantic_focus'], e.associations[2][2])
    report = json.loads((e.out / 'transfer_planning_diagnostic_000.json').read_text())
    assert report['post_lift_observations_used'] == report['post_lift_observations_reserved'] == 2
    assert not (e.out / 'transfer_planning_diagnostic_001.json').exists()
    if not transfer_passes: assert e.result['failure'] == 'planning: transfer RRT timeout'


@pytest.mark.parametrize('outcome', ['held_abort', 'held_stale', 'held_repeat'])
def test_bad_held_recovery_view_decision_returns_without_extra_frame_or_transfer(tmp_path, monkeypatch, outcome):
    e = run_transfer_fixture(tmp_path, monkeypatch, outcome=outcome)
    assert e.result['observations'] == 9 and not e.plans
    assert executed(e, 'lower') and not executed(e, 'transfer')
    assert not any(frame >= 9 for frame, _ in e.retired)
    assert any(x['kind'] == 'held_verification_recovery_rejected' for x in e.events)


def test_successful_association_with_failed_measured_lift_does_not_spend_last_view(tmp_path, monkeypatch):
    e = run_transfer_fixture(tmp_path, monkeypatch, outcome='low_lift')
    assert e.result['observations'] == 9 and not e.held_requests and not e.plans
    assert executed(e, 'lower') and not executed(e, 'transfer')


def test_single_reserved_postlift_frame_cannot_trigger_extra_verification_image(tmp_path, monkeypatch):
    e = run_transfer_fixture(tmp_path, monkeypatch, outcome='held_failure', reserve=1)
    assert e.result['observations'] == 9 and not e.held_requests and not e.plans
    assert executed(e, 'lower') and not executed(e, 'transfer')
