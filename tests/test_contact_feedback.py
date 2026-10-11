"""A failed semantic association explains observed pixels without relaxing it."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.geometry import unproject
from view2grasp.grounding import ContactAssociationError, GroundingResolver
from test_contact_reassociation import annotations, scene


def diagnostic_for(obs, mask, reference, support=.95, robot_depth=None, **kwargs):
    with pytest.raises(ContactAssociationError) as caught:
        GroundingResolver().reassociate_contact(obs, annotations(obs, mask, kwargs.get('part_mask')),
            reference, kwargs.get('reference_part', reference), 'target', support, robot_depth)
    error = caught.value
    # Reject NaN / Infinity explicitly, not merely Python's permissive JSON.
    assert json.loads(json.dumps(error.diagnostics, allow_nan=False)) == error.diagnostics
    assert isinstance(error, ValueError)
    assert len(error.diagnostics['rejected_samples']) <= 12
    return error, error.diagnostics


def assert_source_pixels(diagnostics, obs):
    assert diagnostics['frame_id'] == obs.frame_id
    assert diagnostics['camera_id'] == obs.camera_id
    assert diagnostics['source_shape_hw'] == list(obs.depth.shape)
    assert diagnostics['coordinate_space'] == 'source_image_normalized_pixel_centers'
    h, w = obs.depth.shape
    for sample in diagnostics['rejected_samples']:
        assert np.array_equal(np.rint(np.array(sample['normalized']) * [w - 1, h - 1]), sample['pixel'])


def test_support_rejection_explains_wrong_pixels_with_bounded_source_coordinates():
    obs, xyz, mask, reference = scene()
    obs = replace(obs, camera_id='current_camera')
    error, d = diagnostic_for(obs, mask, reference, support=1.)
    assert str(error) == 'contact association has insufficient filtered depth'
    assert d['failure'] == 'insufficient_filtered_depth' and d['role'] == 'object'
    assert d['proposed_pixels'] == 16 and d['valid_pixels'] == d['anchor_pixels'] == 0
    assert d['rejection_counts']['support'] == d['rejected_pixel_count'] == 16
    assert d['samples_truncated'] and len(d['rejected_samples']) == 12
    assert all(s['reasons'] == ['support'] and s['world_z_m'] == 1. for s in d['rejected_samples'])
    assert_source_pixels(d, obs)
    assert d['reference_projection']['kind'] == 'hypothesis_not_current_visibility'
    assert d['reference_projection']['intersects_image']
    assert np.allclose(d['reference_projection']['bbox_normalized'], np.array([8, 10, 9, 13]) / 23)


@pytest.mark.parametrize('robot_distance', [1., .9, np.nan, np.inf])
def test_robot_self_and_foreground_rejections_never_report_nonfinite_json(robot_distance):
    obs, xyz, mask, reference = scene()
    self_depth = np.full(obs.depth.shape, robot_distance)
    error, d = diagnostic_for(obs, mask, reference, robot_depth=self_depth)
    assert str(error) == 'contact association has insufficient filtered depth'
    assert d['valid_pixels'] == 0 and d['rejection_counts']['robot'] == 16
    assert all(s['reasons'] == ['robot'] for s in d['rejected_samples'])
    assert all(s['robot_depth_m'] == (robot_distance if np.isfinite(robot_distance) else None)
        for s in d['rejected_samples'])
    assert_source_pixels(d, obs)


def test_invalid_depth_is_distinct_from_support_and_preserves_null_depth():
    obs, xyz, mask, reference = scene()
    bad_depth = obs.depth.copy(); bad_depth[mask] = np.nan
    bad = replace(obs, depth=bad_depth)
    _, d = diagnostic_for(bad, mask, reference)
    assert d['valid_pixels'] == 0 and d['rejection_counts']['invalid_depth'] == 16
    assert d['rejection_counts']['support'] == 0
    assert all(s['reasons'] == ['invalid_depth'] and s['depth_m'] is None for s in d['rejected_samples'])


def test_single_expansion_violation_is_not_hidden_by_filtered_background():
    obs, xyz, mask, reference = scene()
    mask[2:8, 2:18] = True
    depth = obs.depth.copy(); depth[2:8, 2:18] = .95
    depth[12, 10] = 1.036  # A measured face beyond the fixed 35 mm source bound.
    obs = replace(obs, depth=depth)
    error, d = diagnostic_for(obs, mask, reference)
    assert str(error) == 'contact association is inconsistent with the original observed object'
    assert d['failure'] == 'reference_inconsistency'
    assert d['proposed_pixels'] == 112 and d['valid_pixels'] == 16
    assert d['anchor_pixels'] >= 4
    assert d['rejection_counts']['support'] == 96
    assert d['rejection_counts']['reference_distance'] == 1
    assert d['rejected_pixel_count'] == 1 and not d['samples_truncated']
    sample, = d['rejected_samples']
    assert sample['pixel'] == [10, 12] and sample['reasons'] == ['reference_distance']
    observed = unproject([10, 12], depth[12, 10], obs.K, obs.T_wc)
    assert sample['reference_distance_m'] == pytest.approx(cKDTree(reference).query(observed)[0])
    assert sample['reference_distance_m'] > .035
    assert_source_pixels(d, obs)
    # Diagnostic generation must not remove the point and accept the remainder.
    assert obs.depth[12, 10] == 1.036 and mask[12, 10]


def test_missing_original_overlap_reports_anchor_shortfall_without_expansion_violation():
    obs, xyz, mask, reference = scene()
    reference = reference + [0, 0, .01]
    _, d = diagnostic_for(obs, mask, reference)
    assert d['valid_pixels'] == 16 and d['anchor_pixels'] == 0
    assert d['rejection_counts']['reference_distance'] == 0
    assert d['rejection_counts']['reference_overlap_insufficient'] == 16
    assert d['rejected_pixel_count'] == 16 and d['samples_truncated']
    assert all(s['reasons'] == ['reference_overlap_insufficient'] for s in d['rejected_samples'])
    assert all(.004 <= s['reference_distance_m'] <= .035 for s in d['rejected_samples'])


def test_unanchored_neighbor_feedback_identifies_only_the_disconnected_neighbor():
    obs, xyz, mask, reference = scene(); mask[10:14, 12:16] = True
    error, d = diagnostic_for(obs, mask, reference)
    assert str(error) == 'contact association contains an unanchored semantic surface component'
    assert d['failure'] == 'unanchored_component'
    assert d['valid_pixels'] == 32 and d['anchor_pixels'] >= 4
    assert d['rejection_counts']['reference_distance'] == 0
    assert d['rejection_counts']['unanchored_component'] == d['rejected_pixel_count'] == 16
    assert d['samples_truncated']
    for sample in d['rejected_samples']:
        x, y = sample['pixel']
        assert 12 <= x < 16 and 10 <= y < 14
        assert sample['reasons'] == ['unanchored_component']
        assert sample['reference_distance_m'] > .009
    assert_source_pixels(d, obs)


def test_part_reference_failure_is_not_reported_as_object_failure():
    obs, xyz, mask, reference = scene()
    object_reference = xyz[mask]
    _, d = diagnostic_for(obs, mask, object_reference, reference_part=reference + [0, 0, .01])
    assert d['role'] == 'grasp_part' and d['failure'] == 'reference_inconsistency'
    assert d['anchor_pixels'] == 0


@pytest.mark.parametrize('offset', [[2, 0, 0], [0, 0, -2]])
def test_offscreen_or_behind_camera_reference_is_not_labeled_visible(offset):
    obs, xyz, mask, reference = scene()
    _, d = diagnostic_for(obs, mask, reference + offset)
    assert d['reference_projection'] == {'kind': 'hypothesis_not_current_visibility',
        'bbox_normalized': None, 'intersects_image': False}


def test_success_keeps_original_points_and_report_without_diagnostic_side_effects():
    obs, xyz, mask, reference = scene()
    result = GroundingResolver().reassociate_contact(obs, annotations(obs, mask), reference,
        reference, 'target', .95)
    assert np.array_equal(result.object_mask, mask)
    assert np.array_equal(result.object_points, xyz[mask])
    assert np.array_equal(result.part_points, xyz[mask])
    assert result.report['object_pixels'] == result.report['part_pixels'] == 16
    assert result.report['grasp_ray_changed'] is False
