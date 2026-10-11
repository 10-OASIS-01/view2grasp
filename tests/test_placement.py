import itertools
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from view2grasp.placement import tabletop_placement


def case(**overrides):
    # A measured top patch with an off-center TCP; no bottom/volume completion.
    points = np.array(list(itertools.product([.020, .060], [-.040, 0.], [.010, .012])))
    arguments = dict(destination_xy=np.array([.1, .2]), observed_payload_local=points,
        source_tcp_position=np.array([-.1, -.2, .832]), source_rotation=np.eye(3),
        current_tcp_position=np.array([-.1, -.2, .962]), current_rotation=np.eye(3),
        source_support_z=.8, destination_support_z=.8, source_supported=True)
    arguments.update(overrides)
    return tabletop_placement(**arguments)


@pytest.mark.parametrize('angle', [0., 27., 90., -145.])
def test_eccentric_observed_reference_not_tcp_is_placed_at_destination(angle):
    rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    plan = case(source_rotation=rotation, current_rotation=rotation)
    assert np.allclose((plan.release_position + rotation @ plan.reference_local)[:2], [.1, .2])
    assert np.allclose((plan.transfer_position + rotation @ plan.reference_local)[:2], [.1, .2])
    assert np.linalg.norm(plan.release_position[:2] - [.1, .2]) > .04
    assert np.array_equal(plan.transfer_position[:2], plan.release_position[:2])
    assert plan.transfer_position[2] == .962


def test_saved_top_only_cloud_case_does_not_lower_as_if_observed_top_were_bottom():
    # occluded02 public RGB-D min(z) and robot-only FK, not scene object truth.
    source_tcp_z = .8323836749770893
    observed_min_z = .8344302062824902
    observed_max_z = .8437887490736178
    observed = np.array(list(itertools.product([-.02, .02], [-.02, .02],
        [observed_min_z - source_tcp_z, observed_max_z - source_tcp_z])))
    plan = case(observed_payload_local=observed,
        source_tcp_position=[0., 0., source_tcp_z], current_tcp_position=[0., 0., source_tcp_z + .13])
    old_release_z = .8 - observed[:, 2].min() + .008
    assert source_tcp_z - old_release_z == pytest.approx(.0264302062824902)
    assert plan.release_position[2] == pytest.approx(source_tcp_z + .020)
    assert plan.release_position[2] > source_tcp_z


def test_hidden_bottom_and_newly_visible_lower_surface_do_not_change_release_height():
    top = case()
    more_surface = np.vstack([np.array(list(itertools.product([.020, .060], [-.040, 0.], [.010, .012]))),
        [.020, -.040, -.025], [.060, 0., -.025]])
    fuller = case(observed_payload_local=more_surface)
    assert np.array_equal(top.release_position, fuller.release_position)
    assert not np.array_equal(top.reference_local, fuller.reference_local)


@pytest.mark.parametrize('support_delta', [-.05, 0., .05])
def test_known_horizontal_support_height_change_preserves_source_tcp_clearance(support_delta):
    plan = case(destination_support_z=.8 + support_delta)
    assert plan.release_position[2] - plan.destination_support_z == pytest.approx(.832 - .8 + .020)


def test_small_current_fk_error_never_changes_requested_source_orientation():
    source = Rotation.from_euler('xyz', [.2, -.3, .6]).as_matrix()
    current = Rotation.from_rotvec([0., .014, 0.]).as_matrix() @ source
    plan = case(source_rotation=source, current_rotation=current)
    assert np.array_equal(plan.rotation, source)
    assert plan.rotation_difference_rad == pytest.approx(.014)
    assert np.allclose((plan.release_position + source @ plan.reference_local)[:2], [.1, .2])


def test_orientation_change_is_rejected_instead_of_reusing_unsupported_bottom_assumption():
    with pytest.raises(ValueError, match='orientation differs'):
        case(current_rotation=Rotation.from_rotvec([0., .016, 0.]).as_matrix())


def test_current_height_below_release_height_is_rejected():
    with pytest.raises(ValueError, match='lift clearance'):
        case(current_tcp_position=[0., 0., .840])


def test_unestablished_source_support_is_not_inferred_from_the_cloud():
    with pytest.raises(ValueError, match='source support'):
        case(source_supported=False)


def test_diagnostic_is_json_safe_and_explicit_about_observation_and_support_assumptions():
    report = json.loads(json.dumps(case().diagnostic(), allow_nan=False))
    assert report['physical_bottom_inferred'] is False
    assert report['collision_exemption'] is False
    assert report['clearance_is_certified_error_bound'] is False
    assert 'not_complete_object_center' in report['reference_kind']
    assert 'rigid_grasp_without_object_slip_or_reorientation' in report['required_assumptions']


@pytest.mark.parametrize('override', [
    {'destination_xy': [np.nan, 0.]}, {'destination_xy': [0., 0., 0.]},
    {'observed_payload_local': np.zeros((0, 3))}, {'observed_payload_local': np.zeros((3, 3))},
    {'observed_payload_local': [[0., 0., np.nan]] * 4},
    {'source_tcp_position': [0., 0., np.inf]}, {'current_tcp_position': [0., np.nan, 1.]},
    {'source_rotation': np.diag([-1., 1., 1.])}, {'current_rotation': np.eye(3) * 2},
    {'source_support_z': np.inf}, {'destination_support_z': np.nan},
    {'clearance_m': -.001}, {'clearance_m': np.nan},
    {'rotation_tolerance_rad': -.001}, {'rotation_tolerance_rad': np.pi},
])
def test_invalid_inputs_fail_closed(override):
    with pytest.raises(ValueError, match='placement:'):
        case(**override)
