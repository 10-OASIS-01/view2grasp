"""Strict post-close advice must have the final attachment's safety semantics."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.mapping import DepthMap, FREE, OCCUPIED, UNKNOWN
from view2grasp.planning import Planner, PlanningBudgetExceeded
from view2grasp.postclose_preview import (LiftEvidence, preview_lift,
    setup_lift_attachment, snapshot_depth_map)
from test_object_ownership import annotation, resolve, scene


class Robot:
    """No simulation: known kinematics with a linear TCP and explicit state."""
    def __init__(self):
        self.qidx = np.arange(7)
        self.gq = np.array([7, 8])
        self.data = SimpleNamespace(qpos=np.r_[np.zeros(7), .02, -.02], contact=[])
        self.limits = np.tile([-3., 3.], (7, 1))
        self.ik_calls = 0
        self.noncontact_intersection = False

    def set_q(self, q, gripper=None):
        self.data.qpos[self.qidx] = q
        self.data.qpos[self.gq] = [.04, -.04] if gripper is None else gripper

    def pose(self, q=None):
        if q is not None:
            self.set_q(q)
        return np.array([0., 0., self.data.qpos[0]]), np.eye(3)

    def ik(self, position, rotation, seed):
        self.ik_calls += 1
        q = np.asarray(seed).copy()
        q[0] = position[2]
        self.set_q(q)  # IK may also change the robot-only gripper state.
        return q

    def points(self, q, gripper):
        self.set_q(q, gripper)
        return np.array([[.5, 0., .5]]), np.array([False])

    def payload_intersects_noncontact_robot(self, points):
        return self.noncontact_intersection


def box_volume(lo, hi):
    return np.vstack([np.eye(3), -np.eye(3)]), np.r_[-hi, lo], lo, hi


def planner_and_evidence(target=None):
    map_ = DepthMap(workspace_bounds=None)
    robot = Robot()
    map_.home_volumes = [box_volume(np.array([.49, -.01, .49]), np.array([.51, .01, .51]))]
    planner = Planner(robot, map_, np.random.default_rng(7))
    planner.gripper = np.array([.02, -.02])
    points = np.array([[0., 0., .8]]) if target is None else np.asarray(target)
    return planner, LiftEvidence(points, points.copy(), True)


def legacy_setup(planner, map_, p, R, evidence):
    """Frozen pre-refactor production sequence, for exact parity validation."""
    planner.attached = (evidence.target-p)@R
    planner.observed_payload_local = None
    planner.registered_payload_local = None
    planner.attached_contact_local = (evidence.part-p)@R
    actual = evidence.target
    measured = np.vstack([*(item.object_points for item, _ in evidence.current_ownership), evidence.contact_points])
    if len(measured):
        planner.add_observed_payload(measured, p, R)
        actual = np.vstack([actual, measured])
    retired = []
    if evidence.registered_source_points is not None:
        planner.registered_payload_local = (evidence.registered_source_points-p)@R
        actual = np.vstack([actual, evidence.source_ownership.object_points])
        retired.append(map_.retire_owned_source(evidence.source_ownership, evidence.source_observation))
    for owned, obs in evidence.current_ownership:
        retired.append(map_.retire_owned_source(owned, obs))
    map_.set_payload_source(actual)
    map_.remove_target_endpoints(evidence.target)
    planner.begin_payload_lift()
    return retired


@pytest.mark.parametrize('registered', [False, True])
def test_shared_setup_matches_final_clouds_and_exact_retirement_preserving_retained_neighbors(registered):
    obs, xyz, mask, reference = scene()
    owned = resolve(obs, annotation(obs, mask), reference)
    current_mask = mask.copy(); current_mask[:, :10] = False
    current = resolve(obs, annotation(obs, current_mask, seeds=((10, 6),)), xyz[mask])
    retained = owned.object_points[[0]]
    neighbor = owned.object_points[[1]]+[0., 0., .0062]
    target = owned.object_points[[4]]
    contact = owned.object_points[[6]]+[0., 0., .001]
    predicted = owned.object_points+[0., 0., .02] if registered else None
    evidence = LiftEvidence(target, target, True, ((current, obs),), contact,
        owned, obs, predicted)
    map_ = DepthMap(workspace_bounds=None)
    map_.surface = np.vstack([owned.object_points, neighbor])
    map_.tree = cKDTree(map_.surface)
    map_.retain_observed_endpoints(retained)
    map_.set_target(target)
    p = np.array([.2, -.3, .7])
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    old_map, new_map = snapshot_depth_map(map_), snapshot_depth_map(map_)
    old = Planner(Robot(), old_map, np.random.default_rng(7))
    new = Planner(Robot(), new_map, np.random.default_rng(7))
    old_counts = legacy_setup(old, old_map, p, R, evidence)
    # Audit ordering at the real map method boundary: every newly retired
    # current/source cloud is already represented by its carried counterpart.
    actual_retire = new_map.retire_owned_source
    def retire(item, frame):
        carried = new.payload_collision_local()@R.T+p
        expected = predicted if registered and item is owned else item.object_points
        assert np.all(cKDTree(carried).query(expected)[0] < 1e-12)
        return actual_retire(item, frame)
    new_map.retire_owned_source = retire
    result = setup_lift_attachment(new, new_map, p, R, evidence)
    assert [r['exact_endpoints'] for r in result.retirements] == old_counts
    assert np.array_equal(new_map.surface, old_map.surface)
    assert np.array_equal(new_map.payload_points, old_map.payload_points)
    for name in ('attached', 'attached_contact_local', 'observed_payload_local', 'registered_payload_local'):
        assert np.array_equal(getattr(new, name), getattr(old, name))
    assert np.all(new_map.classify(np.vstack([retained, neighbor])) == OCCUPIED)
    assert np.array_equal(new.attached_contact_local, (target-p)@R)
    if registered:
        assert not np.any(new_map.is_payload_source(predicted))
    else:
        # Without registration, historical ownership has no retirement or
        # source certificate beyond independently measured current/target data.
        assert all(r['kind'] == 'current_object_retired' for r in result.retirements)


def test_snapshot_copies_mutable_geometry_certificates_and_masks_but_shares_acquired_images():
    planner, _ = planner_and_evidence()
    map_ = planner.map
    obs, _, _, _ = scene()
    map_.views = [obs]
    map_.self_masks = [np.zeros_like(obs.depth, bool)]
    map_.occlusion_masks = [np.ones_like(obs.depth, bool)]
    map_.surface = np.array([[0., 0., .8]])
    map_.tree = cKDTree(map_.surface)
    map_.retain_observed_endpoints(map_.surface)
    map_.set_payload_source(map_.surface)
    snapshot = snapshot_depth_map(map_)
    assert snapshot.views[0] is obs and snapshot.views is not map_.views
    snapshot.surface[:] = 9.
    snapshot.tree.data[:] = 8.
    snapshot.retained_endpoints[:] = 7.
    snapshot.payload_points[:] = 6.
    snapshot.self_masks[0][:] = True
    snapshot.occlusion_masks[0][:] = False
    snapshot.home_volumes[0][0][:] = 5.
    assert np.array_equal(map_.surface, [[0., 0., .8]])
    assert np.array_equal(map_.tree.data, [[0., 0., .8]])
    assert np.array_equal(map_.retained_endpoints, [[0., 0., .8]])
    assert np.array_equal(map_.payload_points, [[0., 0., .8]])
    assert not map_.self_masks[0].any() and map_.occlusion_masks[0].all()
    assert np.max(map_.home_volumes[0][0]) == 1.


def test_multisource_registration_missing_one_cloud_stops_before_map_retirement():
    obs, _, mask, reference = scene()
    owned = resolve(obs, annotation(obs, mask), reference)
    planner, _ = planner_and_evidence()
    planner.map.surface = owned.object_points.copy()
    planner.map.tree = cKDTree(planner.map.surface)
    evidence = LiftEvidence(owned.object_points[:1], owned.object_points[:1], True,
        source_ownership=owned, source_observation=obs,
        registered_source_points=owned.object_points + [0., 0., .02],
        additional_source_ownership=((owned, obs),))
    before = planner.map.surface.copy()
    with pytest.raises(ValueError, match='every historical source point'):
        setup_lift_attachment(planner, planner.map, np.zeros(3), np.eye(3), evidence)
    assert np.array_equal(planner.map.surface, before)
    assert planner.map.payload_points is None
    assert planner.registered_payload_local is None


def test_payload_unknown_and_occupied_are_separate_with_no_live_mutation_or_contact_growth():
    planner, evidence = planner_and_evidence([[0., 0., .8], [.1, 0., .8]])
    planner.map.surface = np.array([[0., 0., .81]])
    planner.map.tree = cKDTree(planner.map.surface)
    planner.map.set_target(evidence.part[:1])
    planner.closest_by_failure = {'old': {'count': 7}}
    planner.robot.set_q(np.r_[.2, np.zeros(6)], np.array([.031, -.031]))
    prior_state = planner.robot.data.qpos.copy()
    prior_rng = copy.deepcopy(planner.rng.bit_generator.state)
    prior_surface = planner.map.surface.copy()
    report = preview_lift(planner, evidence, np.zeros(7), np.array([.02, -.02]))
    assert report.report['last_validation']['payload_unknown_count'] == 1
    assert report.report['last_validation']['payload_occupied_count'] == 1
    assert report.report['occupied_unresolved'] and report.report['mapping_evidence_available']
    assert not len(report.robot_unknown)
    assert np.allclose(report.payload_unknown, [[.1, 0., .81]])
    assert np.array_equal(planner.map.surface, prior_surface)
    assert planner.map.payload_points is None and planner.attached is None
    assert planner.closest_by_failure == {'old': {'count': 7}}
    assert planner.checks == 0 and planner.last_validation is None
    assert planner.rng.bit_generator.state == prior_rng
    assert np.array_equal(planner.robot.data.qpos, prior_state)


def test_raw_unknown_exempted_by_measured_source_never_becomes_preview_advice():
    planner, evidence = planner_and_evidence([[0., 0., .8], [0., 0., .81]])
    report = preview_lift(planner, evidence, np.zeros(7), planner.gripper)
    # At +1 cm, the lower point reaches an actual original source surface;
    # the leading point remains unknown. Advice contains only that leading point.
    assert np.allclose(report.payload_unknown, [[0., 0., .82]])
    assert report.report['last_validation']['payload_unknown_count'] == 1


def test_permitted_finger_unknown_is_not_misreported_as_required_arm_coverage():
    planner, evidence = planner_and_evidence()
    planner.map.home_volumes = []
    planner.map.set_target(evidence.part)
    def pad_points(q, grip):
        planner.robot.set_q(q, grip)
        return evidence.part+[0.,0.,q[0]], np.array([True])
    planner.robot.points = pad_points
    result = preview_lift(planner, evidence, np.zeros(7), planner.gripper)
    assert not len(result.robot_unknown)
    assert result.report['last_validation']['robot_unknown_count'] == 0
    assert np.allclose(result.payload_unknown, [[0.,0.,.81]])


@pytest.mark.parametrize('kind', ['occupied', 'robot_unknown', 'noncontact_robot'])
def test_nonmapping_refusals_never_invent_payload_unknown(kind):
    planner, evidence = planner_and_evidence()
    if kind == 'occupied':
        planner.map.surface = np.array([[0., 0., .81]])
        planner.map.tree = cKDTree(planner.map.surface)
    elif kind == 'robot_unknown':
        planner.map.home_volumes = []
    else:
        planner.robot.noncontact_intersection = True
    report = preview_lift(planner, evidence, np.zeros(7), planner.gripper)
    assert not len(report.payload_unknown)
    assert bool(len(report.robot_unknown)) == (kind == 'robot_unknown')
    assert report.report['mapping_evidence_available'] == (kind == 'robot_unknown')
    assert report.report['occupied_unresolved'] == (kind == 'occupied')
    assert not report.report['nominal_path_valid']


def test_edge_checks_interpolated_samples_and_reports_that_actual_refusal():
    planner, evidence = planner_and_evidence(np.empty((0, 3)))
    # Endpoints at 0 and .03 are in disjoint exact known robot volumes, but
    # the interpolated q=.015 robot point has no free-space evidence.
    planner.map.home_volumes = [box_volume(np.array([x-.001, -.001, .499]),
        np.array([x+.001, .001, .501])) for x in (0., .03)]
    planner.robot.points = lambda q, g: (planner.robot.set_q(q, g) or np.array([[q[0], 0., .5]]), np.array([False]))
    planner.robot.ik = lambda p, R, seed: np.r_[.03, np.zeros(6)]
    result = preview_lift(planner, evidence, np.zeros(7), planner.gripper)
    assert result.report['failure_stage'] == 'edge'
    assert result.report['last_validation']['q'][0] == pytest.approx(.015)
    assert np.allclose(result.robot_unknown, [[.015, 0., .5]])


@pytest.mark.parametrize('where', ['IK', 'valid', 'edge'])
@pytest.mark.parametrize('failure', ['timeout', 'exception'])
def test_failure_at_any_stage_restores_state_without_exposing_stale_diagnostics(monkeypatch, where, failure):
    planner, evidence = planner_and_evidence(np.empty((0, 3)))
    before = planner.robot.data.qpos.copy()
    def fail(*args, **kwargs):
        planner.robot.set_q(np.ones(7), np.array([.039, -.039]))
        raise PlanningBudgetExceeded if failure == 'timeout' else RuntimeError('test incomplete preview')
    if where == 'IK':
        monkeypatch.setattr(planner.robot, 'ik', fail)
    else:
        monkeypatch.setattr(Planner, where, fail)
    result = preview_lift(planner, evidence, np.zeros(7), planner.gripper)
    assert result.report['incomplete'] and not result.report['nominal_path_valid']
    assert result.report['last_validation'] is None
    assert not result.report['mapping_evidence_available']
    assert np.array_equal(planner.robot.data.qpos, before)


def test_successful_nominal_preview_still_returns_no_execution_path_and_checks_all_edges(monkeypatch):
    planner, evidence = planner_and_evidence(np.empty((0, 3)))
    real_edge = Planner.edge
    edges = []
    def edge(self, a, b, contact=False):
        edges.append((a.copy(), b.copy(), contact))
        return real_edge(self, a, b, contact)
    monkeypatch.setattr(Planner, 'edge', edge)
    result = preview_lift(planner, evidence, np.zeros(7), planner.gripper)
    assert result.report['completed_waypoints'] == 14
    assert result.report['nominal_path_valid'] and not result.report['incomplete']
    assert len(edges) == 14 and all(contact for _, _, contact in edges)
    assert all(np.array_equal(edges[i-1][1], edges[i][0]) for i in range(1, 14))
    assert not hasattr(result, 'path') and planner.attached is None


def test_cooperative_deadline_overrun_is_not_reported_as_valid(monkeypatch):
    from view2grasp import postclose_preview
    planner, evidence = planner_and_evidence(np.empty((0, 3)))
    clock = [0.]
    monkeypatch.setattr(postclose_preview.time, 'monotonic', lambda: clock[0])
    calls = [0]
    real_edge = Planner.edge
    def edge(self, *args, **kwargs):
        value = real_edge(self, *args, **kwargs)
        calls[0] += 1
        if calls[0] == 14:
            clock[0] = 4.
        return value
    monkeypatch.setattr(Planner, 'edge', edge)
    result = preview_lift(planner, evidence, np.zeros(7), planner.gripper, time_budget=3.)
    assert result.report['completed_waypoints'] == 14
    assert result.report['timed_out'] and result.report['incomplete']
    assert not result.report['nominal_path_valid']
