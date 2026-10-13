"""Independent regressions for object membership versus attached pad authority."""
from types import SimpleNamespace

import numpy as np
import pytest

from view2grasp.mapping import DepthMap, FREE, OCCUPIED, UNKNOWN
from view2grasp.planning import Planner


PART = np.array([[0., 0., 0.]])
BODY = np.array([[.03, 0., 0.]])


def setup(pad_local=BODY, *, state=UNKNOWN, position=None, rotation=None, is_pad=True):
    position = np.array([.2, -.1, 1.]) if position is None else np.asarray(position)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    whole = np.vstack([PART, BODY])
    measured = whole @ rotation.T + position
    evidence = DepthMap()
    evidence.set_target(PART @ rotation.T + position)
    evidence.set_payload_source(measured)
    robot = SimpleNamespace(limits=np.array([[-1., 1.]]),
        points=lambda q, g: (np.asarray(pad_local) @ rotation.T + position, np.array([is_pad])),
        data=SimpleNamespace(contact=[]), pose=lambda: (position.copy(), rotation.copy()))
    map_ = SimpleNamespace(classify=lambda points: np.full(len(points), state, np.uint8),
        contact_allowed=evidence.contact_allowed, is_target=evidence.is_target,
        is_payload_source=evidence.is_payload_source)
    planner = Planner(robot, map_, np.random.default_rng(3))
    planner.attached = whole.copy()
    return planner, evidence, measured


@pytest.mark.parametrize('contact', [False, True])
def test_membership_body_cannot_waive_unknown_pad_outside_independent_part(contact):
    planner, evidence, measured = setup()
    assert not evidence.contact_allowed(measured[1:]).any()
    assert planner.attached_contact_local is None
    # Exact legacy counterexample: the whole attachment used to waive this.
    assert planner.valid(np.zeros(1), contact=contact)
    planner.attached_contact_local = PART.copy()
    source_before = evidence.payload_points.copy()
    assert not planner.valid(np.zeros(1), contact=contact)
    assert planner.last_failure == 'unknown space'
    assert np.array_equal(planner.last_invalid, [0])  # Robot pad, before payload checks.
    assert np.array_equal(planner.payload_collision_local(), np.vstack([PART, BODY]))
    assert np.array_equal(evidence.payload_points, source_before)
    assert evidence.is_payload_source(measured).all()
    assert np.all(planner.map.classify(measured) == UNKNOWN)  # No map FREE mutation.


def test_explicit_empty_contact_scope_never_falls_back_to_whole_attachment():
    planner, _, _ = setup(PART)
    assert planner.valid(np.zeros(1), contact=False)
    planner.attached_contact_local = np.empty((0, 3))
    assert not planner.valid(np.zeros(1), contact=False)
    assert planner.last_failure == 'unknown space'
    assert np.array_equal(planner.attached, np.vstack([PART, BODY]))


@pytest.mark.parametrize('position,rotation', [
    ([.2, -.1, 1.], np.eye(3)),
    ([-.25, .3, .85], np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])),
])
def test_contact_part_uses_current_tcp_translation_and_rotation(position, rotation):
    part_local = np.array([[.01, .02, .005]])
    # Offset the entire synthetic attachment so an implementation that treats
    # local points as world coordinates cannot accidentally pass this check.
    planner, evidence, _ = setup(part_local, position=position, rotation=rotation)
    planner.attached = np.vstack([part_local, BODY])
    measured = planner.attached @ rotation.T + position
    evidence.set_payload_source(measured)
    planner.attached_contact_local = part_local.copy()
    assert planner.valid(np.zeros(1), contact=False)
    planner.robot.points = lambda q, g: (BODY @ rotation.T + position, np.array([True]))
    assert np.linalg.norm(BODY - part_local) > .012
    assert not planner.valid(np.zeros(1), contact=False)
    assert planner.last_failure == 'unknown space'


@pytest.mark.parametrize('scope', [None, PART, np.empty((0, 3))])
def test_attachment_contact_exception_never_waives_actual_occupied(scope):
    planner, _, _ = setup(PART, state=OCCUPIED)
    planner.attached_contact_local = scope
    assert not planner.valid(np.zeros(1), contact=False)
    assert planner.last_failure == 'observed obstacle'
    assert np.array_equal(planner.last_invalid, [0])


def test_nonpad_robot_geometry_does_not_receive_part_touching_exception():
    planner, _, _ = setup(PART, is_pad=False)
    planner.attached_contact_local = PART.copy()
    assert not planner.valid(np.zeros(1), contact=False)
    assert planner.last_failure == 'unknown space'


@pytest.mark.parametrize('extra_field', ['observed_payload_local', 'registered_payload_local'])
def test_extra_collision_surfaces_cannot_join_attached_contact_scope(extra_field):
    extra = np.array([[.065, 0., 0.]])
    planner, _, _ = setup(extra)
    planner.attached_contact_local = PART.copy()
    setattr(planner, extra_field, extra.copy())
    assert np.array_equal(planner.payload_collision_local(), np.vstack([PART, BODY, extra]))
    assert not planner.valid(np.zeros(1), contact=False)
    assert planner.last_failure == 'unknown space'  # Pad rejected before carried surface checks.
    assert np.array_equal(planner.attached_contact_local, PART)


def test_independent_contact_scope_does_not_remove_body_from_collision_checks():
    planner, _, _ = setup(PART, state=FREE)
    planner.attached_contact_local = PART.copy()
    planner.robot.payload_intersects_noncontact_robot = lambda points: bool(np.any(points[:, 0] > .22))
    assert not planner.valid(np.zeros(1))
    assert planner.last_failure == 'carried object intersects non-contact robot'
