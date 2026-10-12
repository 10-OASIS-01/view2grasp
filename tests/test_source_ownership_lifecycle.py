"""Runner source ownership changes collision evidence, never contact or FREE."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from test_object_ownership import scene, annotation
from view2grasp.object_ownership import resolve_object_ownership
from view2grasp.payload_registration import PayloadRegistrationError
from view2grasp.types import AgentDecision, GraspCandidate


def _contains(haystack, needles):
    return bool(len(haystack) and np.all(cKDTree(haystack).query(needles)[0] < 1e-10))


def run_lifecycle(tmp_path, monkeypatch, *, source_ok=True, current_ok=True,
                  registration_ok=True, tracked_current=False, enabled=True, task='mug',
                  complete_lift=False, max_observations=4, initialize_reference=False,
                  initialization_ok=True, initial_identity=True, initialize_part=False, part_initialization_ok=True,
                  pose_recovery=False,recovery_registration_ok=False,recovery_failure='ambiguous_local_pose',
                  pose_recovery_at=0,initialize_part_count=1,fail_part_frame=None,source_pose_views=1):
    from view2grasp import recording, runner
    sample, xyz, mask, original = scene()
    part_mask = np.zeros_like(mask); part_mask[6:10, 2:4] = True
    source_points = xyz[mask]
    current_depth = sample.depth.copy(); current_depth[mask] += .003
    current_sample = replace(sample, depth=current_depth)
    current_owned_template = resolve_object_ownership(current_sample,
        annotation(current_sample, mask), source_points, 'mug', .9)
    current_points = current_owned_template.object_points
    predicted = source_points + [.020, .010, .010]
    # Distinct from both the original source pixels and the measured close cloud.
    assert not _contains(np.vstack([original, source_points, current_points]), predicted)
    neighbor = source_points[-1] + [0., .0062, 0.]
    position = np.array([.2, -.3, .8])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    candidate = GraspCandidate(np.zeros(3), np.eye(3), .02, np.array([0., 0., 1.]),
        1., 0., surface_point=original[0], grasp_center=original[0])
    evidence = SimpleNamespace(events=[], annotations=[], registrations=[], retired=[],
        advice=[], targets=[], source_certificates=[], observations=[], executed=[],
        track_calls=[], removals=[], initializations=[], registration_references=[],
        operation_order=[],camera_requests=[], original=original, source_points=source_points,
        current_points=current_points, predicted=predicted, neighbor=neighbor,
        candidate=candidate, original_ray=candidate.approach.copy())
    evidence.extra_initial_part=source_points[-4:].copy()

    class Sim:
        def __init__(self, *args):
            self.q = np.zeros(7); self.gripper_q = np.array([.02, -.02]); self.phase = 'observe'
            self.requested_view=None
            self.robot_xml = ''; self.joint_names = []; self.eef_name = ''; self.frame = 0
            evidence.sim = self
        def observe(self):
            self.frame += 1; evidence.observations.append((self.frame, self.phase))
            template = current_sample if self.phase == 'close' else sample
            return replace(template, frame_id=self.frame, q=self.q.copy(),
                gripper_q=self.gripper_q.copy(),
                timestamp=1. if initialize_reference and self.phase=='observe' else float(self.frame),
                view=(self.requested_view if self.phase!='observe' and self.requested_view is not None
                    else (30.*self.frame,15.,.8)) if pose_recovery else template.view)
        def move_camera(self, *args): self.requested_view=tuple(args)
        def validate_initial_proxies(self, *args): pass
        def step(self, q, grip):
            self.q = np.asarray(q).copy(); evidence.executed.append(self.phase)
        def close(self): pass

    class Robot:
        def __init__(self, *args): pass
        def points(self, *args): return np.array([[-.8, -.7, .9]]), np.zeros(1, bool)
        def volumes(self, *args): return ()
        def visual_volumes(self, *args): return ()
        def depth(self, obs): return np.full_like(obs.depth, 10.)
        def pose(self, *args):
            q = args[0] if args else evidence.sim.q
            return position + [0., 0., q[0]], rotation.copy()
        def grasp_center_offset(self): return np.zeros(3)
        def ik(self, p, R, q):
            result = q.copy(); result[0] = p[2] - position[2]
            return result
        def set_q(self, *args): pass
        def close(self): pass

    class Map(runner.DepthMap):
        def __init__(self, resolution):
            super().__init__(resolution, workspace_bounds=None)
            self.surface = np.vstack([source_points, neighbor]); self.tree = cKDTree(self.surface)
            self.table_z = .9; evidence.map = self
        def certify_home(self, *args): pass
        def certify_robot_state(self, *args): pass
        def integrate(self, obs, *args): self.views.append(obs)
        def set_table(self, z): self.table_z = z
        def classify(self, points):
            # One success-path case exhausts UNKNOWN before the total limit,
            # leaving an actual post-lift image for visual verification.
            covered = complete_lift and max_observations > 4 and len(self.views) >= 4
            return np.full(len(points), 1 if covered else 0, np.uint8)
        def set_target(self, points):
            evidence.targets.append(points.copy()); super().set_target(points)
        def set_payload_source(self, points):
            evidence.source_certificates.append(points.copy())
            evidence.operation_order.append('source_certificate')
            super().set_payload_source(points)
        def remove_target_endpoints(self, points, radius=None):
            evidence.removals.append((evidence.sim.phase, points.copy()))
            super().remove_target_endpoints(points, radius)
        def retire_owned_source(self, ownership, obs):
            collision_world = evidence.planner.payload_collision_local() @ rotation.T + position
            expected = predicted if obs.frame_id in (tuple(range(1,2+initialize_part_count)) if initialize_part else (1,)) else ownership.object_points
            # Exact retirement is reached only after the entire qualified cloud
            # has collision coverage in the actual attachment coordinates.
            assert _contains(collision_world, expected)
            assert not _contains(evidence.planner.attached @ rotation.T + position, predicted)
            before = self.surface.copy()
            removed = super().retire_owned_source(ownership, obs)
            evidence.retired.append((obs.frame_id, removed))
            evidence.operation_order.append(('retire', obs.frame_id))
            assert _contains(self.surface, neighbor[None])
            assert len(before) - len(self.surface) == removed
            return removed

    class Plan:
        add_observed_payload = runner.Planner.add_observed_payload
        payload_collision_local = runner.Planner.payload_collision_local
        def __init__(self, robot, map_, rng):
            self.rng = rng; self.checks = 0; self.last_failure = ''
            self.failure_counts = {}; self.closest_invalid = None; self.closest_by_failure = {}
            self.payload_invalid = None; self.observed_payload_local = None; self.registered_payload_local = None
            self.attached = None; evidence.planner = self
        def choose(self, q, candidates, **kwargs):
            assert candidates[0] is candidate
            return candidate, np.array([q, q]), np.array([q, q])
        def recovery_unknown(self): return np.empty((0, 3))
        def begin_payload_lift(self): evidence.operation_order.append('begin_lift')
        def end_payload_lift(self): pass
        def feasible_ik(self, p, R, q, **kwargs):
            if complete_lift:
                result = q.copy(); result[0] = p[2] - position[2]
                return result
            self.last_failure = 'deliberate lifecycle stop'
            return None

    class Decisions:
        def __init__(self, *args, **kwargs):
            self.calls = 0; self.seconds = 0.; self.client = SimpleNamespace(close=lambda: None)
        def decide(self, obs, *args, **kwargs):
            self.calls += 1
            request=args[-1][-1].get('source_pose_verification') if args and args[-1] and isinstance(args[-1][-1],dict) else None
            if request is not None:
                evidence.camera_requests.append(request.copy())
                az,el,dist=request['reference_view']
                return AgentDecision(action='observe',frame_id=obs.frame_id,azimuth=az,elevation=el,distance=dist)
            if kwargs.get('force_observe') or (initialize_reference and obs.frame_id==1) or (initialize_part and obs.frame_id<=1+initialize_part_count):
                return AgentDecision(action='observe', frame_id=obs.frame_id)
            decision = annotation(obs, mask)
            explicit_part = annotation(obs, part_mask).groundings[0].model_copy(update={'role': 'grasp_part'})
            return decision.model_copy(update={'action': 'grasp', 'target': 'mug',
                'groundings': [*(decision.groundings if initial_identity else []), explicit_part]})

    class Resolver:
        calls = 0; seconds = 0.; segmenter = None; sam_unavailable_reason = 'test'
        def resolve_grasp(self, obs, decision, *args):
            return SimpleNamespace(decision=decision, object_mask=mask, part_mask=part_mask,
                report={'kinds': ['mask']})
        def track(self, obs, *args, **kwargs):
            evidence.track_calls.append((evidence.sim.frame, obs.frame_id, evidence.sim.phase))
            if complete_lift and evidence.sim.phase == 'hold':
                return current_points + [0., 0., evidence.sim.q[0]]
            return current_points.copy() if tracked_current and obs.frame_id > 1 else None

    def annotate(vlm, obs, reference, identity, support, **kwargs):
        assert identity == 'mug' and kwargs['allow_partial_seed_components']
        assert kwargs['robot_depth'].shape == obs.depth.shape
        evidence.annotations.append((obs.frame_id, reference.copy()))
        vlm.calls += 1
        accepted = source_ok if obs.frame_id == 1 else current_ok
        ownership = resolve_object_ownership(obs, annotation(obs, mask), reference,
            identity, support, kwargs['robot_depth'], allow_partial_seed_components=True) if accepted else None
        return SimpleNamespace(ownership=ownership, report={'semantic_requests': 1, 'vlm_calls': 1,
            'accepted': accepted})

    def register(reference, measured, source):
        expected=(np.vstack([original,*([source_points]*(1+initialize_part_count if initialize_part else 1))])
            if initialize_reference else original)
        assert np.array_equal(reference, expected)
        if initialize_reference:assert not reference.flags.writeable
        evidence.registration_references.append(reference.copy())
        assert np.array_equal(source,np.vstack([source_points]*(1+initialize_part_count)) if initialize_part else source_points)
        evidence.registrations.append(measured.copy())
        # No acquired current surface means no registration, even if the old
        # pre-close target is geometrically identical to the original reference.
        ok=(registration_ok and not (pose_recovery and len(evidence.registrations)==1)) or (
            pose_recovery and recovery_registration_ok and len(evidence.registrations)>=2)
        if not len(measured) or not ok:
            raise PayloadRegistrationError('deliberate rejection', {'accepted': False,
                'failure':'no_current_evidence' if not len(measured) else recovery_failure if pose_recovery else 'test'})
        return SimpleNamespace(transformed_source_points=np.vstack([predicted]*(1+initialize_part_count)) if initialize_part else predicted.copy(), report={'accepted': True})

    def initialize(vlm, source, side, reference, identity, support, **kwargs):
        assert not evidence.executed and evidence.sim.phase=='observe'
        is_part=kwargs.get('side_part') is not None
        assert side.frame_id==(2+initialize_part_count if initialize_part else 2)
        assert source.frame_id in (range(2,2+initialize_part_count) if is_part else (1,))
        if is_part:assert np.array_equal(kwargs['side_part'],original)
        assert source.timestamp==side.timestamp
        assert np.array_equal(reference,original)
        assert not evidence.retired and not evidence.source_certificates
        evidence.initializations.append((source.frame_id,side.frame_id,reference.copy()))
        vlm.calls+=1
        accepted=(part_initialization_ok and source.frame_id!=fail_part_frame) if is_part else initialization_ok
        ownership=resolve_object_ownership(source,annotation(source,mask),reference,
            identity,support,kwargs['robot_depth']) if accepted else None
        return SimpleNamespace(ownership=ownership,report={'semantic_requests':1,'vlm_calls':1},
            initial_part_points=evidence.extra_initial_part.copy() if is_part and accepted else None,
            initial_part_mask=mask.copy() if is_part and accepted else None)

    def advice(*args, **kwargs):
        evidence.advice.append({'observed': kwargs['observed_points'].copy(),
            'coverage': np.asarray(args[2]).copy(), 'static_surface': evidence.map.surface.copy()})
        return {'views': [[0., 55., .8]]}

    def preview(planner, payload, start, gripper):
        # This fixture exercises ownership/registration lifecycles, with its
        # original synthetic coverage schedule. Real strict geometry, map
        # isolation and restoration are tested in test_postclose_preview.
        from view2grasp.postclose_preview import LiftPreview
        if pose_recovery and len(evidence.map.views)>=pose_recovery_at:
            return LiftPreview(np.empty((0,3)),np.empty((0,3)),
                {'mapping_evidence_available':False,'occupied_unresolved':payload.registered_source_points is None,
                    'incomplete':payload.registered_source_points is None},evidence.map.surface.copy())
        carried = [payload.target, payload.contact_points]
        carried.extend(owned.object_points for owned, _ in payload.current_ownership)
        if payload.registered_source_points is not None:
            carried.append(payload.registered_source_points)
        covered = complete_lift and max_observations > 4 and len(evidence.map.views) >= 4
        unknown = np.empty((0, 3)) if covered else np.vstack(carried) + [0., 0., .13]
        return LiftPreview(np.empty((0, 3)), unknown,
            {'mapping_evidence_available': bool(len(unknown)),
                'occupied_unresolved': False, 'incomplete': not covered},
            evidence.map.surface.copy())

    monkeypatch.setattr(runner, 'Simulator', Sim)
    monkeypatch.setattr(runner, 'RobotModel', Robot)
    monkeypatch.setattr(runner, 'DepthMap', Map)
    monkeypatch.setattr(runner, 'Planner', Plan)
    monkeypatch.setattr(runner, 'VLM', Decisions)
    monkeypatch.setattr(runner.GroundingResolver, 'from_config', lambda config: Resolver())
    monkeypatch.setattr(runner, 'annotate_source_object', annotate)
    monkeypatch.setattr(runner, 'annotate_initial_object_reference', initialize)
    if initialize_part:
        def select(views,*args,**kwargs):
            candidates=[v for v in views if 2<=v.frame_id<2+initialize_part_count and v.frame_id not in kwargs['excluded_frames']]
            chosen=candidates[0] if candidates else None
            return chosen,{'selected_frame':None if chosen is None else chosen.frame_id}
        monkeypatch.setattr(runner,'select_initial_part_view',select)
    monkeypatch.setattr(runner, 'register_payload_source', register)
    monkeypatch.setattr(runner, 'Evaluator', lambda sim: SimpleNamespace(result=lambda: {
        'success': False, 'collision': False, 'first_grasp_success': False, 'collision_free_success': False}))
    monkeypatch.setattr(runner, 'grasp_candidates', lambda *args: ([candidate], original.copy()))
    monkeypatch.setattr(runner, 'part_cloud', lambda *args: original.copy())
    monkeypatch.setattr(runner, 'check_vertical_part_midpoint', lambda *args, **kwargs: {'status': 'unresolved'})
    monkeypatch.setattr(runner, 'cloud', lambda obs: original)
    monkeypatch.setattr(runner, 'fit_table', lambda *args: .9)
    monkeypatch.setattr(runner, 'suggest_views', advice)
    monkeypatch.setattr(runner, 'preview_lift', preview)
    monkeypatch.setattr(runner, 'nominal_grasp', lambda *args: None)
    monkeypatch.setattr(runner, 'time_parameterize', lambda path: path)
    monkeypatch.setattr(runner, 'save_observation', lambda *args: None)
    monkeypatch.setattr(recording, 'save_grounding_result', lambda *args: None)
    monkeypatch.setattr(runner, 'save_episode', lambda out, sim, record, events: evidence.events.extend(events))
    result = runner.run_episode(task, 3, 'active', tmp_path, replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=max_observations, source_object_ownership=enabled,
            initial_multiview_reference=initialize_reference,initial_multiview_part=initialize_part,
            initial_part_views=initialize_part_count if initialize_part else 1,
            source_pose_recovery_views=source_pose_views))
    expected_failure = ('execution: evaluator success criterion not met' if complete_lift
        else 'planning: lift deliberate lifecycle stop')
    if initialize_reference and not initialization_ok:
        expected_failure='perception: initial multiview object reference unverified'
    if initialize_reference and not initial_identity:
        expected_failure='perception: initial object reference requires one explicit target identity'
    if initialize_part and (not part_initialization_ok or fail_part_frame is not None):
        expected_failure='perception: initial multiview part reference unverified'
    assert result['failure'] == expected_failure
    assert not result['success'] and not result['online']
    assert np.array_equal(candidate.approach, evidence.original_ray)
    return result, evidence


def test_registered_points_are_collision_only_and_retirement_is_after_representation(tmp_path, monkeypatch):
    result, e = run_lifecycle(tmp_path, monkeypatch)
    assert result['observations'] == 4 and result['attempts'] == 1
    assert [frame for frame, _ in e.annotations] == [1, 2, 3, 4]
    assert [frame for frame, _ in e.retired] == [1, 2, 3, 4]
    assert e.planner.registered_payload_local is not None
    assert e.planner.observed_payload_local is not None
    actual = e.source_certificates[-1]
    assert _contains(actual, e.source_points) and _contains(actual, e.current_points)
    assert not _contains(actual, e.predicted)
    assert all(np.array_equal(points, e.original) for points in e.targets)
    assert not e.map.contact_allowed(e.predicted).any()
    assert _contains(e.map.surface, e.neighbor[None])
    assert ('retire', 1) in e.operation_order[:e.operation_order.index('begin_lift')]


def test_extra_current_owned_surfaces_and_registered_hypothesis_reach_next_view_advice(tmp_path, monkeypatch):
    _, e = run_lifecycle(tmp_path, monkeypatch)
    assert len(e.advice) == 3
    assert not _contains(e.advice[0]['observed'], e.current_points)
    assert not _contains(e.advice[1]['static_surface'], e.current_points)
    assert _contains(e.advice[1]['observed'], e.current_points)
    assert _contains(e.advice[1]['observed'], e.predicted)
    assert _contains(e.advice[1]['coverage'], e.current_points + [0., 0., .13])


def test_failed_registration_keeps_historical_source_but_carries_and_retires_actual_current(tmp_path, monkeypatch):
    _, e = run_lifecycle(tmp_path, monkeypatch, registration_ok=False)
    assert [frame for frame, _ in e.retired] == [2, 3, 4]
    assert e.planner.registered_payload_local is None
    assert e.planner.observed_payload_local is not None
    assert _contains(e.map.surface, e.source_points[-1:])
    assert _contains(e.source_certificates[-1], e.current_points)
    assert not _contains(e.source_certificates[-1], e.source_points)
    assert not _contains(e.source_certificates[-1], e.predicted)


def test_failed_source_annotation_does_not_disable_independent_current_ownership(tmp_path, monkeypatch):
    _, e = run_lifecycle(tmp_path, monkeypatch, source_ok=False)
    assert not e.registrations
    assert [frame for frame, _ in e.retired] == [2, 3, 4]
    assert e.planner.registered_payload_local is None
    assert _contains(e.map.surface, e.source_points[-1:])
    assert not _contains(e.source_certificates[-1], e.source_points)


def test_no_actual_postclose_evidence_never_registers_against_stale_preclose_target(tmp_path, monkeypatch):
    _, e = run_lifecycle(tmp_path, monkeypatch, current_ok=False, tracked_current=False)
    assert e.registrations and all(points.shape == (0, 3) for points in e.registrations)
    assert not e.retired
    assert e.planner.registered_payload_local is None
    assert e.planner.observed_payload_local is None
    assert _contains(e.map.surface, e.source_points[-1:])
    assert np.array_equal(e.source_certificates[-1], e.original)


def test_actual_postclose_tracking_can_supply_registration_when_object_annotation_abstains(tmp_path, monkeypatch):
    _, e = run_lifecycle(tmp_path, monkeypatch, current_ok=False, tracked_current=True)
    assert e.registrations and all(_contains(points, e.current_points) for points in e.registrations)
    assert [frame for frame, _ in e.retired] == [1]
    assert e.planner.observed_payload_local is None
    assert e.planner.registered_payload_local is not None
    # The baseline tracker voxel-downsamples its target. Every retained point
    # must remain an actual measurement; it need not retain all raw pixels.
    certificate = e.source_certificates[-1]
    assert _contains(np.vstack([e.current_points, e.source_points]), certificate)
    assert np.any(cKDTree(e.current_points).query(certificate)[0] < 1e-10)
    assert _contains(e.source_certificates[-1], e.source_points)
    assert not _contains(e.source_certificates[-1], e.predicted)


def test_ownership_does_not_retrack_history_for_broad_endpoint_removal(tmp_path, monkeypatch):
    _, e = run_lifecycle(tmp_path, monkeypatch, current_ok=False, tracked_current=True)
    assert e.track_calls == [(2, 2, 'close'), (3, 3, 'close'), (4, 4, 'close')]
    # Legacy source/target cleanup remains, but there is no historical tracker
    # union whose radius could erase independently abstained source surfaces.
    assert len(e.removals) == 2


@pytest.mark.parametrize('maximum', [4, 5])
def test_completed_lift_does_not_promote_fk_estimates_to_measured_source(tmp_path, monkeypatch, maximum):
    result, e = run_lifecycle(tmp_path, monkeypatch, complete_lift=True,
        max_observations=maximum)
    assert result['observations'] == maximum and result['attempts'] == 1
    assert 'lift' in e.executed and 'hold' in e.executed
    assert len(e.source_certificates) == 1
    assert all(phase == 'close' for phase, _ in e.removals)
    actual = np.vstack([e.original, e.source_points, e.current_points])
    assert _contains(actual, e.source_certificates[0])
    assert not _contains(e.source_certificates[0], e.predicted)
    assert not _contains(e.source_certificates[0], e.original + [0., 0., .13])
    assert e.planner.registered_payload_local is not None
    assert e.planner.observed_payload_local is not None
    if maximum == 5:
        assert any(item['kind'] == 'visual_verification' and item['lifted'] for item in e.events)
    else:
        assert any(item['kind'] == 'verification_deferred_to_independent_evaluator' for item in e.events)


@pytest.mark.parametrize('enabled,expected_observations', [(True, 5), (False, 4)])
def test_only_ownership_development_mode_can_spend_unused_pregrasp_slot_on_real_mapping(
        tmp_path, monkeypatch, enabled, expected_observations):
    result, e = run_lifecycle(tmp_path, monkeypatch, enabled=enabled, max_observations=5)
    assert result['observations'] == expected_observations
    assert len(e.advice) == expected_observations - 1
    assert e.observations == [(1, 'observe'),
        *[(frame, 'close') for frame in range(2, expected_observations + 1)]]
    assert result['observations'] <= 5
    assert result['attempts'] == 1


@pytest.mark.parametrize('enabled,task', [(False, 'mug'), (True, 'block')])
def test_opt_in_is_mug_only_and_default_path_has_no_new_ownership_operations(tmp_path, monkeypatch, enabled, task):
    _, e = run_lifecycle(tmp_path, monkeypatch, enabled=enabled, task=task)
    assert not e.annotations and not e.registrations and not e.retired
    assert e.planner.observed_payload_local is None and e.planner.registered_payload_local is None
    assert np.array_equal(e.source_certificates[-1], e.original)
