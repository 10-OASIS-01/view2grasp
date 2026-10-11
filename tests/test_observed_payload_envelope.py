"""New measured held faces strengthen collision checks without new contact."""
from types import SimpleNamespace

import numpy as np
import pytest

from view2grasp.mapping import DepthMap, FREE, OCCUPIED, UNKNOWN
from view2grasp.planning import Planner


def planner_fixture(state=FREE, noncontact=False):
    pose = np.zeros(3)
    original = np.array([[0., 0., 0.]])
    new_side = np.array([[.020, 0., 0.]])
    source = DepthMap(); source.set_payload_source(np.vstack([original, new_side]))
    def classify(points):
        blocked = (points[:, 0] > .015) & (points[:, 1] > .09)
        return np.where(blocked, state, FREE).astype(np.uint8)
    def points(q, grip):
        pose[:] = [0., q[0], 0.]
        return np.empty((0, 3)), np.empty(0, dtype=bool)
    def hits_robot(held):
        return bool(noncontact and np.any((held[:, 0] > .015) & (held[:, 1] > .09)))
    robot = SimpleNamespace(limits=np.array([[-1., 1.]]), points=points,
        data=SimpleNamespace(contact=[]), pose=lambda: (pose.copy(), np.eye(3)),
        payload_intersects_noncontact_robot=hits_robot)
    map_ = SimpleNamespace(classify=classify, is_payload_source=source.is_payload_source)
    planner = Planner(robot, map_, np.random.default_rng(0)); planner.attached = original.copy()
    return planner, original, new_side, source


@pytest.mark.parametrize('state,noncontact,reason',[
    (OCCUPIED, False, 'carried object observed obstacle'),
    (UNKNOWN, False, 'carried object unknown space'),
    (FREE, True, 'carried object intersects non-contact robot'),
])
def test_new_side_can_block_a_future_path_that_original_payload_missed(state, noncontact, reason):
    planner, original, new_side, _ = planner_fixture(state, noncontact)
    assert planner.valid(np.array([.1]))  # Original observed top alone misses it.
    planner.add_observed_payload(new_side, np.zeros(3), np.eye(3))
    assert planner.valid(np.array([0.]))
    assert not planner.valid(np.array([.1]))
    assert planner.last_failure == reason
    assert np.array_equal(planner.attached, original)


def test_extra_surface_source_explains_only_unknown_and_never_occupied():
    planner, original, new_side, source = planner_fixture()
    planner.add_observed_payload(new_side, np.zeros(3), np.eye(3))
    state = UNKNOWN
    planner.map.classify = lambda p: np.where(p[:, 0] > .015, state, FREE).astype(np.uint8)
    assert planner.valid(np.zeros(1))
    assert source.is_payload_source(new_side).all()
    state = OCCUPIED
    assert not planner.valid(np.zeros(1))
    assert planner.last_failure == 'carried object observed obstacle'
    assert np.array_equal(planner.attached, original)


@pytest.mark.parametrize('contact', [False, True])
def test_collision_only_surface_never_expands_permitted_pad_contacts(contact):
    planner, original, new_side, source = planner_fixture()
    source.set_target(original)
    planner.add_observed_payload(new_side, np.zeros(3), np.eye(3))
    planner.robot.points = lambda q, g: (new_side.copy(), np.array([True]))
    planner.map.classify = lambda p: np.where(p[:, 0] > .015, UNKNOWN, FREE).astype(np.uint8)
    planner.map.is_target = source.is_target
    planner.map.contact_allowed = source.contact_allowed
    assert not source.contact_allowed(new_side).any()
    assert not planner.valid(np.zeros(1), contact=contact)
    assert planner.last_failure == 'unknown space'
    assert np.array_equal(planner.attached, original)


def test_measured_world_surface_is_transformed_to_local_without_reference_drift():
    planner, original, _, _ = planner_fixture()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    position = np.array([.2, -.3, .8])
    local = np.array([[.02, .03, -.01]])
    measured = local @ rotation.T + position
    planner.add_observed_payload(measured, position, rotation)
    assert np.allclose(planner.observed_payload_local, local)
    planner.add_observed_payload(measured, position, rotation)
    assert len(planner.observed_payload_local) == 1
    assert np.array_equal(planner.attached, original)
    assert np.allclose(planner.payload_collision_local(), np.vstack([original, local]))
