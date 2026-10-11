"""Whole-object evidence must not enlarge the runner's finger contact region."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.types import AgentDecision, GroundingProposal, GraspCandidate, Observation


def run_contact_lifecycle(tmp_path, monkeypatch, *, object_seeds, explicit_part, ownership):
    from view2grasp import recording, runner
    # The separately observed body is inside the legacy 4 cm neighborhood but
    # outside the measured grasp part. Both remain in the collision envelope.
    part = np.array([[x, y, 1.] for x in np.arange(4)*.002 for y in np.arange(4)*.002])
    body = part + [.024, 0., 0.]
    target = np.vstack([part, body])
    current = target + [0., 0., .002]
    part_mask = np.zeros((12, 12), bool); part_mask[4:6, 4:6] = True
    object_mask = part_mask.copy(); object_mask[4:6, 8:10] = True
    sample = Observation(1, np.zeros((12, 12, 3), np.uint8), np.ones((12, 12)),
        np.eye(3), np.eye(4), np.zeros(7), np.array([.02, -.02]), 0., (0., 45., .8))
    candidate = GraspCandidate(np.zeros(3), np.eye(3), .02, np.array([0., 0., 1.]),
        1., 0., surface_point=part[0], grasp_center=part[0])
    e = SimpleNamespace(targets=[], events=[], track_calls=[], payload_sources=[],
        annotations=[], executed=[], original_part=part, body=body, original_ray=candidate.approach.copy())

    class Sim:
        def __init__(self, *args):
            self.q = np.zeros(7); self.gripper_q = sample.gripper_q.copy(); self.phase = 'observe'
            self.robot_xml = ''; self.joint_names = []; self.eef_name = ''; self.frame = 0; e.sim = self
        def observe(self):
            self.frame += 1
            return replace(sample, frame_id=self.frame, q=self.q.copy(), gripper_q=self.gripper_q.copy())
        def move_camera(self, *args): pass
        def validate_initial_proxies(self, *args): pass
        def step(self, q, grip): self.q = np.asarray(q).copy(); e.executed.append(self.phase)
        def close(self): pass

    class Robot:
        def __init__(self, *args): pass
        def points(self, *args): return np.zeros((1, 3)), np.zeros(1, bool)
        def volumes(self, *args): return ()
        def visual_volumes(self, *args): return ()
        def depth(self, obs): return np.full_like(obs.depth, 10.)
        def pose(self, q=None):
            q = e.sim.q if q is None else q
            return np.array([0., 0., q[0]]), np.eye(3)
        def grasp_center_offset(self): return np.zeros(3)
        def ik(self, p, R, q): return np.r_[p[2], q[1:]]
        def set_q(self, *args): pass
        def close(self): pass

    class Map(runner.DepthMap):
        def __init__(self, resolution):
            super().__init__(resolution, workspace_bounds=None); e.map = self
        def integrate(self, obs, *args): self.views.append(obs)
        def certify_home(self, *args): pass
        def certify_robot_state(self, *args): pass
        def set_table(self, z): self.table_z = z
        def classify(self, points): return np.full(len(points), int(len(self.views) >= 4), np.uint8)
        def set_target(self, points):
            e.targets.append((e.sim.phase, points.copy())); super().set_target(points)
        def set_payload_source(self, points):
            e.payload_sources.append(points.copy()); super().set_payload_source(points)

    class Plan:
        add_observed_payload = runner.Planner.add_observed_payload
        payload_collision_local = runner.Planner.payload_collision_local
        def __init__(self, robot, map_, rng):
            self.rng = rng; self.checks = 0; self.last_failure = ''
            self.failure_counts = {}; self.closest_invalid = None; self.closest_by_failure = {}
            self.payload_invalid = None; self.observed_payload_local = None; self.registered_payload_local = None
            self.attached = None; e.planner = self
        def choose(self, q, candidates, **kwargs): return candidate, np.array([q, q]), np.array([q, q])
        def recovery_unknown(self): return np.empty((0, 3))
        def begin_payload_lift(self): pass
        def end_payload_lift(self): pass
        def feasible_ik(self, p, R, q, **kwargs): return np.r_[p[2], q[1:]]

    class Decisions:
        def __init__(self, *args, **kwargs):
            self.calls = 0; self.seconds = 0.; self.client = SimpleNamespace(close=lambda: None)
        def decide(self, obs, *args, **kwargs):
            self.calls += 1
            if kwargs.get('force_observe'): return AgentDecision(action='observe', frame_id=obs.frame_id)
            gs = [GroundingProposal(kind='box', role='object', frame_id=obs.frame_id,
                box=(.2, .2, .9, .9), target_id='mug')]
            if object_seeds:
                gs.append(GroundingProposal(kind='points', role='object', frame_id=obs.frame_id,
                    points=[(.35, .45), (.75, .45)], target_id='mug'))
            if explicit_part:
                gs.append(GroundingProposal(kind='box', role='grasp_part', frame_id=obs.frame_id,
                    box=(.2, .2, .5, .6), target_id='mug'))
            elif object_seeds:
                # Independent contact semantics are required by the real
                # resolver even when no separate grasp_part region is given.
                gs.append(GroundingProposal(kind='points', role='grasp_contact', frame_id=obs.frame_id,
                    points=[(.35, .45)], target_id='mug'))
            return AgentDecision(action='grasp', frame_id=obs.frame_id, groundings=gs,
                target='mug', point=(.35, .45), bbox=(.2, .2, .9, .9), part_bbox=(.2, .2, .5, .6))

    class Resolver:
        calls = 0; seconds = 0.; segmenter = None; sam_unavailable_reason = 'test'
        def resolve_grasp(self, obs, decision, *args):
            # The resolver owns semantic masks. This test isolates what the
            # real runner subsequently does with its separate body/part result.
            return SimpleNamespace(decision=decision, object_mask=object_mask,
                part_mask=part_mask, report={'kinds': ['box', 'points']})
        def track(self, obs, *args, **kwargs):
            e.track_calls.append((e.sim.phase, obs.frame_id))
            return current + ([0., 0., .13] if e.sim.phase == 'hold' else np.zeros(3))

    def annotate(*args, **kwargs):
        e.annotations.append(args[1].frame_id)
        return SimpleNamespace(ownership=None, report={'semantic_requests': 1, 'vlm_calls': 0})

    for name, value in (('Simulator', Sim), ('RobotModel', Robot), ('DepthMap', Map),
            ('Planner', Plan), ('VLM', Decisions), ('annotate_source_object', annotate)):
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner.GroundingResolver, 'from_config', lambda config: Resolver())
    monkeypatch.setattr(runner, 'Evaluator', lambda sim: SimpleNamespace(result=lambda: {
        'success': False, 'collision': False, 'first_grasp_success': False, 'collision_free_success': False}))
    monkeypatch.setattr(runner, 'grasp_candidates', lambda *args: ([candidate], target.copy()))
    monkeypatch.setattr(runner, 'part_cloud', lambda *args: part.copy())
    monkeypatch.setattr(runner, 'check_vertical_part_midpoint', lambda *args, **kwargs: {'status': 'unresolved'})
    monkeypatch.setattr(runner, 'cloud', lambda obs: target)
    monkeypatch.setattr(runner, 'fit_table', lambda *args: .9)
    monkeypatch.setattr(runner, 'suggest_views', lambda *args, **kwargs: {'views': [[0., 55., .8]]})
    monkeypatch.setattr(runner, 'time_parameterize', lambda path: path)
    monkeypatch.setattr(runner, 'save_observation', lambda *args: None)
    monkeypatch.setattr(recording, 'save_grounding_result', lambda *args: None)
    monkeypatch.setattr(runner, 'save_episode', lambda out, sim, record, events: e.events.extend(events))
    result = runner.run_episode('mug', 3, 'active', tmp_path, replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=5, source_object_ownership=ownership))
    assert result['observations'] == 5 and result['attempts'] == 1
    assert result['failure'] == 'execution: evaluator success criterion not met'
    assert not result['online'] and not result['success']
    assert 'lift' in e.executed and any(x['kind'] == 'visual_verification' and x['lifted'] for x in e.events)
    assert np.array_equal(candidate.approach, e.original_ray)
    # All source points are still carried through the collision-only target.
    payload = e.planner.attached
    assert np.all(cKDTree(payload).query(current)[0] < .005)
    return e


@pytest.mark.parametrize('explicit_part,ownership', [(False, False), (True, False), (True, True)])
def test_explicit_whole_object_seeds_never_expand_contact_at_any_runner_phase(
        tmp_path, monkeypatch, explicit_part, ownership):
    e = run_contact_lifecycle(tmp_path, monkeypatch,
        object_seeds=True, explicit_part=explicit_part, ownership=ownership)
    assert all(np.array_equal(points, e.original_part) for _, points in e.targets)
    assert {phase for phase, _ in e.targets} >= {'observe', 'close'}
    assert not e.map.contact_allowed(e.body).any()
    # A successful post-lift tracker result is not a semantic contact update.
    assert not any(phase == 'hold' for phase, _ in e.targets)


@pytest.mark.parametrize('explicit_part', [False, True])
def test_ownership_mode_preserves_measured_part_even_without_initial_object_seeds(
        tmp_path, monkeypatch, explicit_part):
    e = run_contact_lifecycle(tmp_path, monkeypatch,
        object_seeds=False, explicit_part=explicit_part, ownership=True)
    assert e.annotations
    assert all(np.array_equal(points, e.original_part) for _, points in e.targets)
    assert not e.map.contact_allowed(e.body).any()


def test_legacy_unseeded_nonownership_path_keeps_existing_contact_behavior(tmp_path, monkeypatch):
    e = run_contact_lifecycle(tmp_path, monkeypatch,
        object_seeds=False, explicit_part=False, ownership=False)
    assert not e.annotations
    assert any(np.any(cKDTree(points).query(e.body)[0] < .005) for _, points in e.targets)
    assert any(phase == 'hold' for phase, _ in e.targets)
