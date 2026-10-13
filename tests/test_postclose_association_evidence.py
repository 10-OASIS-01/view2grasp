"""A failed extra camera annotation cannot erase static, verified evidence."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_contact_reassociation import annotations, scene
from view2grasp.grounding import GroundingResolver
from view2grasp.types import GraspCandidate


@pytest.mark.parametrize('first_verified', [True, False])
def test_postclose_evidence_survives_failed_second_view_without_bypassing_lift_checks(
        tmp_path, monkeypatch, first_verified):
    from view2grasp import recording, runner

    obs, xyz, mask, _ = scene()
    part_mask = mask.copy()
    target = xyz[mask]
    candidate = GraspCandidate(np.zeros(3), np.eye(3), .02, np.array([0., 0., 1.]),
        0., 0., surface_point=target[0], grasp_center=target[0])
    evidence = SimpleNamespace(events=[], physical_steps=0, camera_steps=[], observation_steps=[],
        targets=[], lift_checks=0, annotation_frames=[])

    class Sim:
        def __init__(self, *args):
            self.q = np.zeros(7); self.gripper_q = np.array([.02, -.02])
            self.robot_xml = ''; self.joint_names = []; self.eef_name = ''
            self.phase = 'observe'; self.frame = 0
            evidence.sim = self
        def observe(self):
            self.frame += 1
            evidence.observation_steps.append((self.frame, evidence.physical_steps))
            return replace(obs, frame_id=self.frame, q=self.q.copy(), gripper_q=self.gripper_q.copy())
        def move_camera(self, *args): evidence.camera_steps.append(evidence.physical_steps)
        def validate_initial_proxies(self, *args): pass
        def step(self, q, grip):
            self.q = np.asarray(q).copy(); evidence.physical_steps += 1
        def close(self): pass

    class Robot:
        def __init__(self, *args): pass
        def points(self, *args): return np.zeros((1, 3)), np.zeros(1, dtype=bool)
        def volumes(self, *args): return ()
        def visual_volumes(self, *args): return ()
        def depth(self, sample): return np.full_like(sample.depth, 10.)
        def pose(self, q=None): return np.zeros(3), np.eye(3)
        def grasp_center_offset(self): return np.zeros(3)
        def ik(self, p, R, q): return q.copy()
        def set_q(self, *args): pass
        def close(self): pass

    class Map:
        def __init__(self, *args):
            self.resolution = .005; self.table_z = .95; self.surface = target.copy(); self.views = []
        def certify_home(self, *args): pass
        def certify_robot_state(self, *args): pass
        def integrate(self, sample, *args): self.views.append(sample)
        def set_table(self, z): self.table_z = z
        # Unresolved sweep evidence requires the second view even after the
        # first semantic association succeeds; no FREE evidence is invented.
        def classify(self, points): return np.zeros(len(points), dtype=np.uint8)
        def set_target(self, points): evidence.targets.append(points.copy())
        def set_payload_source(self, *args): pass
        def remove_target_endpoints(self, *args): pass

    class Plan:
        def __init__(self, robot, map_, rng):
            self.rng = rng; self.checks = 0; self.failure_counts = {}; self.last_failure = ''
            self.closest_invalid = None; self.closest_by_failure = {}; self.payload_invalid = None
        def choose(self, q, candidates, **kwargs):
            return candidates[0], np.array([q, q]), np.array([q, q])
        def recovery_unknown(self): return np.empty((0, 3))
        def begin_payload_lift(self): pass
        def end_payload_lift(self): pass
        def feasible_ik(self, *args, **kwargs):
            evidence.lift_checks += 1
            self.last_failure = 'deliberate strict lift rejection'
            return None

    class Decisions:
        def __init__(self, *args, **kwargs):
            self.calls = 0; self.seconds = 0.; self.client = SimpleNamespace(close=lambda: None)
        def decide(self, sample, instruction, history, **kwargs):
            self.calls += 1
            if sample.frame_id == 1 and not kwargs.get('force_observe'):
                return annotations(sample, mask, part_mask).model_copy(update={'action': 'grasp'})
            if kwargs.get('observe_purpose') == 'annotation':
                evidence.annotation_frames.append(sample.frame_id)
                result = annotations(sample, mask, part_mask)
                if sample.frame_id != 2 or not first_verified:
                    result = result.model_copy(update={'groundings': result.groundings[:1]})
                return result
            return annotations(sample, mask).model_copy(update={'groundings': []})

    resolver = GroundingResolver()
    resolver.resolve_grasp = lambda sample, decision, *args: SimpleNamespace(
        decision=decision, object_mask=mask, part_mask=part_mask, report={'kinds': ['mask']})
    resolver.track = lambda *args, **kwargs: pytest.fail('Unverified tracking must not replace strict association')
    monkeypatch.setattr(runner, 'Simulator', Sim)
    monkeypatch.setattr(runner, 'RobotModel', Robot)
    monkeypatch.setattr(runner, 'DepthMap', Map)
    monkeypatch.setattr(runner, 'Planner', Plan)
    monkeypatch.setattr(runner, 'VLM', Decisions)
    monkeypatch.setattr(runner.GroundingResolver, 'from_config', lambda config: resolver)
    monkeypatch.setattr(runner, 'Evaluator', lambda sim: SimpleNamespace(result=lambda: {
        'success': False, 'collision': False, 'first_grasp_success': False, 'collision_free_success': False}))
    monkeypatch.setattr(runner, 'grasp_candidates', lambda *args: ([candidate], target.copy()))
    monkeypatch.setattr(runner, 'part_cloud', lambda *args: target.copy())
    monkeypatch.setattr(runner, 'fit_table', lambda *args: .95)
    monkeypatch.setattr(runner, 'nominal_grasp', lambda *args: None)
    monkeypatch.setattr(runner, 'suggest_views', lambda *args, **kwargs: {'views': [[0., 55., .8]]})
    monkeypatch.setattr(runner, 'save_observation', lambda *args: None)
    monkeypatch.setattr(runner, 'time_parameterize', lambda path: path)
    monkeypatch.setattr(recording, 'save_grounding_result', lambda *args: None)
    monkeypatch.setattr(runner, 'save_episode', lambda out, sim, record, events: evidence.events.extend(events))
    def no_crop(*args, **kwargs): raise ValueError('Use the original test image')
    monkeypatch.setattr(runner, 'crop_observation', no_crop)

    result = runner.run_episode('occluded', 3, 'active', tmp_path, replay=tmp_path / 'offline',
        config=runner.ExperimentConfig(max_observations=4, perception_backend='rgbd'))
    assert evidence.annotation_frames == [2, 3]
    assert evidence.observation_steps[1][1] > 0
    assert evidence.observation_steps[1][1] == evidence.observation_steps[2][1] == evidence.physical_steps
    assert evidence.camera_steps == [evidence.physical_steps] * 2
    rejected = [event for event in evidence.events if event['kind'] == 'contact_association_rejected']
    assert [event['frame_id'] for event in rejected] == ([3] if first_verified else [2, 3])
    assert result['success'] is False
    assert not any(event['kind'] == 'execute' and event['phase'] == 'lift' for event in evidence.events)
    if first_verified:
        assert evidence.lift_checks == 1
        assert result['failure'] == 'planning: lift deliberate strict lift rejection'
        assert len(evidence.targets) == 2  # Failed view neither clears nor broadens the verified part.
        assert np.array_equal(np.unique(evidence.targets[0], axis=0), evidence.targets[1])
    else:
        assert evidence.lift_checks == 0
        assert result['failure'] == 'execution: post-close object/part association unverified; lift blocked'


def test_real_camera_and_observe_methods_do_not_advance_physics(monkeypatch):
    from view2grasp import sim as module

    simulator = object.__new__(module.Simulator)
    state = SimpleNamespace(qpos=np.arange(9, dtype=float), time=12.5)
    before = state.qpos.copy(); calls = []
    state.set_mocap_pos = lambda *args: calls.append('camera_position')
    state.set_mocap_quat = lambda *args: calls.append('camera_rotation')
    def render(**kwargs):
        calls.append('render')
        return np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4))
    backend = SimpleNamespace(data=state, forward=lambda: calls.append('forward'), render=render,
        _render_context_offscreen=SimpleNamespace(gl_ctx=SimpleNamespace(make_current=lambda: None)))
    simulator.env = SimpleNamespace(sim=backend, step=lambda *args: pytest.fail('Camera observations must not step physics'))
    simulator.qidx = list(range(7)); simulator.gripidx = [7, 8]
    simulator.size = 4; simulator.frame_id = 0
    monkeypatch.setattr(module, 'get_camera_intrinsic_matrix', lambda *args: np.eye(3))
    monkeypatch.setattr(module, 'get_camera_extrinsic_matrix', lambda *args: np.eye(4))
    monkeypatch.setattr(module, 'get_real_depth_map', lambda sim, depth: depth)

    simulator.move_camera(0., 55., .8)
    first = simulator.observe()
    simulator.move_camera(90., 55., .8)
    second = simulator.observe()
    assert calls == ['camera_position', 'camera_rotation', 'forward', 'render'] * 2
    assert np.array_equal(state.qpos, before)
    assert np.array_equal(first.q, second.q) and np.array_equal(first.gripper_q, second.gripper_q)
    assert first.timestamp == second.timestamp == 12.5
    assert second.frame_id == first.frame_id + 1
