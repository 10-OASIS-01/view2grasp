"""Measured supported sides may exceed a top-only reference's 35 mm tube."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.geometry import unproject
from view2grasp.grounding import ContactAssociationError, GroundingResolver, _supported_downward_points
from view2grasp.types import Observation
from test_contact_reassociation import annotations, scene


def supported_side():
    """Calibrated abstraction of the recorded 43.79 mm supported block.

    The old reference sees its top. A later side view measures a continuous
    4 mm pixel lattice down to 8 mm above the observed support. The lowest
    measured row is 36 mm from the reference, despite belonging to that side.
    No simulator object dimensions, poses or labels enter this fixture.
    """
    depth = np.full((28,24),2.)
    mask = np.zeros(depth.shape,bool);mask[12:22,8:13] = True
    depth[mask] = 1.
    K = np.array([[250.,0,10.],[0,250.,12.],[0,0,1.]])
    transform = np.eye(4)
    transform[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
    transform[:3,3] = [0,-1,.844]
    rgb = np.zeros((*depth.shape,3),np.uint8);rgb[mask] = [8,32,145]
    obs = Observation(4,rgb,depth,K,transform,np.zeros(7),np.array([.04,-.04]),0.,(0.,30.,1.))
    x,y = np.meshgrid(np.arange(-.012,.0121,.004),[-.008,0.,.008])
    reference = np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.844)])
    v,u = np.mgrid[:28,:24]
    xyz = unproject(np.stack([u,v],-1),depth,K,transform)
    return obs,mask,reference,xyz


def associate(obs,mask,reference,*,source_supported=False,part_mask=None,reference_part=None):
    return GroundingResolver().reassociate_contact(obs,annotations(obs,mask,part_mask),
        reference,reference if reference_part is None else reference_part,'target',.8,
        source_supported=source_supported)


def test_supported_44mm_top_only_reference_recovers_actual_connected_side_without_filling():
    obs,mask,reference,xyz = supported_side()
    assert np.max(cKDTree(reference).query(xyz[mask])[0]) == pytest.approx(.036)
    with pytest.raises(ContactAssociationError,match='inconsistent'):
        associate(obs,mask,reference)
    result = associate(obs,mask,reference,source_supported=True)
    assert np.array_equal(result.object_mask,mask) and np.array_equal(result.part_mask,mask)
    assert np.array_equal(result.object_points,xyz[mask])
    assert np.array_equal(result.part_points,xyz[mask])
    assert result.report['grasp_ray_changed'] is False
    for role in ('object','grasp_part'):
        evidence = result.report['supported_downward_association'][role]
        assert evidence['minimum_z_exclusive_m'] == pytest.approx(.806)
        assert evidence['maximum_z_inclusive_m'] == pytest.approx(.844)
        assert evidence['eligible_beyond_reference_limit_pixels'] == 5
    json.dumps(result.report,allow_nan=False)


@pytest.mark.parametrize('fault',['outside_xy','upward','red_neighbor'])
def test_supported_interval_never_expands_laterally_or_upward_or_to_red_neighbor(fault):
    obs,mask,reference,xyz = supported_side()
    depth = obs.depth.copy();rgb = obs.rgb.copy()
    if fault == 'upward':pixel = (3,10)
    else:pixel = (21,22)
    mask[pixel] = True;depth[pixel] = 1.
    if fault == 'red_neighbor':rgb[pixel] = [145,15,8]
    obs = replace(obs,depth=depth,rgb=rgb)
    with pytest.raises(ContactAssociationError,match='inconsistent') as caught:
        associate(obs,mask,reference,source_supported=True)
    diagnostic = caught.value.diagnostics
    assert diagnostic['rejection_counts']['reference_distance'] == 1
    sample, = diagnostic['rejected_samples']
    assert sample['pixel'] == list(pixel[::-1])
    assert sample['reasons'] == ['reference_distance']
    assert sample['supported_downward_eligible'] is False
    # The other five long-range side samples are eligible, not misreported as
    # reference-distance violations merely because the whole proposal fails.
    assert diagnostic['supported_downward_association']['eligible_beyond_reference_limit_pixels'] == 5


def test_supported_downward_points_still_need_fixed_original_anchors():
    obs,mask,reference,xyz = supported_side();reference[:,2] += .01
    with pytest.raises(ContactAssociationError,match='inconsistent') as caught:
        associate(obs,mask,reference,source_supported=True)
    diagnostic = caught.value.diagnostics
    assert diagnostic['anchor_pixels'] == 0
    assert diagnostic['rejection_counts']['reference_distance'] == 0
    assert diagnostic['rejection_counts']['reference_overlap_insufficient'] == mask.sum()


def test_supported_downward_points_still_need_original_6mm_four_connected_chain():
    obs,mask,reference,xyz = supported_side();mask[19] = False
    with pytest.raises(ContactAssociationError,match='unanchored semantic surface') as caught:
        associate(obs,mask,reference,source_supported=True)
    diagnostic = caught.value.diagnostics
    assert diagnostic['failure'] == 'unanchored_component'
    assert diagnostic['rejection_counts']['unanchored_component'] == 10
    assert diagnostic['rejection_counts']['reference_distance'] == 0
    beyond = [s for s in diagnostic['rejected_samples'] if s['reference_distance_m'] > .035]
    assert len(beyond) == 5
    assert all(s['supported_downward_eligible'] and s['reasons'] == ['unanchored_component'] for s in beyond)
    json.dumps(diagnostic,allow_nan=False)


def test_degenerate_reference_footprint_keeps_strict_original_bound():
    obs,mask,reference,xyz = supported_side();reference = reference[reference[:,1] == 0]
    with pytest.raises(ContactAssociationError,match='inconsistent') as caught:
        associate(obs,mask,reference,source_supported=True)
    diagnostic = caught.value.diagnostics
    assert diagnostic['supported_downward_association']['status'] == 'unavailable_reference_footprint'
    assert diagnostic['rejection_counts']['reference_distance'] == 5


def test_part_must_keep_its_own_reference_and_remain_inside_object():
    obs,mask,reference,xyz = supported_side();part = mask.copy();part[11,10] = True
    with pytest.raises(ContactAssociationError,match='inside the explicit object'):
        associate(obs,mask,reference,source_supported=True,part_mask=part)
    with pytest.raises(ContactAssociationError,match='inconsistent') as caught:
        associate(obs,mask,reference,source_supported=True,reference_part=reference+[0,0,.01])
    assert caught.value.diagnostics['role'] == 'grasp_part'
    assert caught.value.diagnostics['anchor_pixels'] == 0


def test_original_10mm_independent_neighbor_still_cannot_borrow_target_anchors():
    obs,xyz,mask,reference = scene();mask[10:14,12:16] = True
    with pytest.raises(ContactAssociationError,match='unanchored semantic surface'):
        GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,
            'target',.95,source_supported=True)


def test_footprint_margin_is_euclidean_4mm_even_at_hull_corners():
    obs,mask,reference,xyz = supported_side()
    # An offset half-plane intersection would admit the 3 mm / 3 mm corner,
    # whose true distance from the observed footprint exceeds the 4 mm bound.
    points = np.array([[.015,0.,.808],[.017,0.,.808],[.015,.011,.808]])
    allowed,evidence = _supported_downward_points(points,reference,.8)
    assert allowed.tolist() == [True,False,False]
    assert evidence['xy_tolerance_m'] == .004


def test_support_and_robot_filters_still_exclude_measured_returns():
    obs,mask,reference,xyz = supported_side()
    robot_depth = np.full(obs.depth.shape,10.);robot_depth[21,8] = 1.
    result = GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,
        'target',.8,robot_depth,source_supported=True)
    assert not result.object_mask[21,8] and not result.part_mask[21,8]
    # Advancing the observed support removes the low row through the original
    # filter, rather than reintroducing it through the downward exception.
    result = GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,
        'target',.804,source_supported=True)
    assert not result.object_mask[21].any()
