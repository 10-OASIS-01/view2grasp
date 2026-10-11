"""Semantic contradiction checks must not turn into a contact-placement rule."""
from dataclasses import replace
import json

import numpy as np
import pytest

from view2grasp.grasp_semantics import (GraspSemanticsError, VerticalPartMidpointConfig,
    check_vertical_part_midpoint)
from view2grasp.types import Observation


def scene(rows=slice(12, 15), cols=slice(30, 70), contact=(50, 13)):
    K = np.array([[200., 0, 49.5], [0, 200., 49.5], [0, 0, 1.]])
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]; T[2, 3] = 1.
    obs = Observation(9, np.zeros((100, 100, 3), np.uint8), np.ones((100, 100)),
        K, T, np.zeros(7), np.array([.04, -.04]), 1.5, (-150., 30., 1.3), 'current_camera')
    object_mask = np.zeros((100, 100), bool); object_mask[10:90, 20:80] = True
    part_mask = np.zeros_like(object_mask); part_mask[rows, cols] = True
    obs.depth[:] = 2.  # Measured farther background around the actual object.
    obs.depth[object_mask] = 1.
    return obs, object_mask, part_mask, np.array(contact) / 99


def check(case, **kwargs):
    report = check_vertical_part_midpoint(*case, **kwargs)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    return report


@pytest.mark.parametrize('rows,contact', [(slice(12, 15), (50, 13)), (slice(85, 88), (50, 86))])
def test_only_strong_horizontal_end_region_is_rejected(rows, contact):
    case = scene(rows=rows, contact=contact)
    original_masks = [case[1].copy(), case[2].copy()]; original_contact = case[3].copy()
    with pytest.raises(GraspSemanticsError, match='THIS image') as caught:
        check(case)
    error = caught.value; d = error.diagnostics
    assert isinstance(error, ValueError)
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d['status'] == 'rejected' and all(d['contradiction_conditions'].values())
    assert d['frame_id'] == 9 and d['camera_id'] == 'current_camera'
    assert d['actual_view'] == [-150., 30., 1.3]
    assert d['contact']['pixel'] == list(contact)
    assert d['part']['principal_vertical_alignment'] < 1e-10
    assert d['part']['principal_variance_ratio'] > 4
    assert np.array_equal(case[1], original_masks[0]) and np.array_equal(case[2], original_masks[1])
    assert np.array_equal(case[3], original_contact)  # No corrected point is chosen.


def test_short_horizontal_contact_patch_at_true_middle_is_not_rejected():
    report = check(scene(rows=slice(49, 52), contact=(50, 50)))
    assert report['status'] == 'no_contradiction'
    assert report['contradiction_conditions']['near_horizontal']
    assert not report['contradiction_conditions']['height_extreme']
    assert .4 < report['contact']['observed_height_fraction'] < .6


def test_truncating_object_mask_cannot_make_a_real_midpoint_into_an_endpoint():
    obs, obj, part, contact = scene(rows=slice(49, 52), contact=(50, 50))
    assert check((obs, obj, part, contact))['status'] == 'no_contradiction'
    # Same RGB-D and part: the upper object is still actually visible. Only the
    # semantic mask is cut, falsely putting the selected midpoint at its top.
    obj[:48] = False
    report = check((obs, obj, part, contact))
    assert all(report['contradiction_conditions'].values())
    assert report['status'] == 'unresolved'
    assert report['reason'] == 'observed_height_boundary_not_established'
    assert report['height_boundary']['counts']['surface_continuation'] > 0


@pytest.mark.parametrize('outside_depth,reason', [(.8, 'foreground'), (np.nan, 'invalid_depth'),
    (1.002, 'surface_continuation')])
def test_occluded_or_unmeasured_height_boundary_cannot_reject_semantics(outside_depth, reason):
    obs, obj, part, contact = scene(rows=slice(49, 52), contact=(50, 50))
    obj[:48] = False
    obs.depth[47, 20:80] = outside_depth
    report = check((obs, obj, part, contact))
    assert report['status'] == 'unresolved'
    assert report['height_boundary']['counts'][reason] > 0


def test_height_extreme_alone_does_not_reject_a_vertical_observed_part():
    report = check(scene(rows=slice(12, 29), cols=slice(45, 49), contact=(46, 13)))
    assert report['status'] == 'no_contradiction'
    assert report['contradiction_conditions']['height_extreme']
    assert report['part']['principal_vertical_alignment'] > .99


def test_isotropic_end_patch_is_not_assumed_to_be_a_horizontal_bar():
    report = check(scene(rows=slice(12, 16), cols=slice(45, 49), contact=(46, 13)))
    assert report['status'] == 'no_contradiction'
    assert not report['contradiction_conditions']['elongated']


def test_wide_region_with_substantial_vertical_extent_does_not_satisfy_thin_end_evidence():
    report = check(scene(rows=slice(12, 29), cols=slice(20, 80), contact=(50, 13)))
    assert report['status'] == 'no_contradiction'
    assert report['contradiction_conditions']['near_horizontal']
    assert report['contradiction_conditions']['elongated']
    assert not report['contradiction_conditions']['small_vertical_extent']


def test_camera_roll_does_not_turn_world_horizontal_into_world_vertical():
    obs, obj, part, contact = scene()
    roll = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    T = obs.T_wc.copy(); T[:3, :3] = T[:3, :3] @ roll
    rolled = replace(obs, T_wc=T, rgb=np.rot90(obs.rgb), depth=np.rot90(obs.depth))
    new_contact = np.array([contact[1], 1 - contact[0]])
    with pytest.raises(GraspSemanticsError) as original:
        check_vertical_part_midpoint(obs, obj, part, contact)
    with pytest.raises(GraspSemanticsError) as rotated:
        check_vertical_part_midpoint(rolled, np.rot90(obj), np.rot90(part), new_contact)
    a, b = original.value.diagnostics, rotated.value.diagnostics
    assert np.allclose(a['contact']['world_xyz_m'], b['contact']['world_xyz_m'])
    assert b['part']['principal_vertical_alignment'] < 1e-10
    assert b['contact']['observed_height_fraction'] == pytest.approx(a['contact']['observed_height_fraction'])


@pytest.mark.parametrize('fault,reason', [
    ('few_part_points', 'insufficient_current_mask_depth'),
    ('few_object_points', 'insufficient_current_mask_depth'),
    ('partial_object_height', 'insufficient_observed_object_height'),
    ('invalid_depth', 'insufficient_current_mask_depth'),
    ('invalid_contact_depth', 'no_valid_current_contact_depth'),
    ('outside_contact', 'contact_not_on_current_observed_part'),
    ('missing_contact', 'no_valid_current_contact'),
    ('clipped_object', 'object_mask_reaches_image_boundary'),
    ('outside_part', 'part_not_inside_current_object_mask')])
def test_inadequate_observations_return_no_determination(fault, reason):
    obs, obj, part, contact = scene()
    if fault == 'few_part_points': part[:] = False; part[13, 50:53] = True
    elif fault == 'few_object_points': obj = part = np.zeros_like(obj); obj[13, 45:65] = True
    elif fault == 'partial_object_height': obj = part.copy()
    elif fault == 'invalid_depth': obs.depth[:] = np.nan
    elif fault == 'invalid_contact_depth': obs.depth[13, 50] = np.nan
    elif fault == 'outside_contact': contact = np.array([.5, .5])
    elif fault == 'missing_contact': contact = None
    elif fault == 'clipped_object': obj[0, 50] = True
    elif fault == 'outside_part': part[0, 0] = True
    report = check((obs, obj, part, contact))
    assert report['status'] == 'unresolved' and report['reason'] == reason


def test_public_thresholds_can_demand_more_evidence_and_are_recorded():
    config = VerticalPartMidpointConfig(min_principal_variance_ratio=1000.)
    report = check(scene(), config=config)
    assert report['status'] == 'no_contradiction'
    assert report['thresholds']['min_principal_variance_ratio'] == 1000.


@pytest.mark.parametrize('kwargs', [{'min_object_points': 3}, {'min_part_points': 3},
    {'min_object_points': float('nan')}, {'min_part_points': 4.5}, {'min_part_points': True},
    {'min_object_height_m': np.nan}, {'height_quantile': .5}, {'height_extreme_fraction': .5},
    {'min_principal_variance_ratio': 1}, {'max_principal_vertical_alignment': 1},
    {'max_part_height_fraction': 0}])
def test_invalid_threshold_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match='thresholds'):
        VerticalPartMidpointConfig(**kwargs)


def test_diagnostics_include_current_mask_bounds_in_both_coordinate_units():
    with pytest.raises(GraspSemanticsError) as caught:
        check(scene())
    d = caught.value.diagnostics
    assert d['part']['mask_bbox']['pixel'] == [30, 12, 69, 14]
    assert np.allclose(d['part']['mask_bbox']['normalized'], np.array([30, 12, 69, 14]) / 99)
    assert 'truth' not in d and d['evidence_source'] == 'current_resolved_masks_and_calibrated_rgbd'
