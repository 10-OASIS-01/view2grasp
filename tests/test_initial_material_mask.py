"""Dense initial labels are fresh semantics, never implicit seed expansion."""
from dataclasses import replace

import numpy as np
import pytest

from view2grasp.initial_object_reference import (
    _resolve_initial_object_reference, annotate_initial_object_reference)
from view2grasp.object_ownership import ObjectOwnershipError
from test_initial_object_reference import ScriptedVLM
from test_object_ownership import annotation, neighbor_scene, scene


def resolve(obs, decision, reference, **kwargs):
    return _resolve_initial_object_reference(obs,decision,reference,'mug',.9,
        material_mask=True,**kwargs)


def test_fresh_mask_labels_unseeded_disconnected_material_without_reinterpreting_default():
    obs,xyz,mask,reference=neighbor_scene()
    mask[7,10]=False  # An explicit hole is not filled by region connectivity.
    decision=annotation(obs,mask)
    sparse=_resolve_initial_object_reference(obs,decision,reference,'mug',.9)
    assert sparse.object_mask.sum()==24
    dense=resolve(obs,decision,reference)
    assert np.array_equal(dense.object_mask,mask)
    assert np.array_equal(dense.object_points,xyz[mask])
    assert not dense.object_mask[7,10]
    assert dense.report['component_seed_gate_applied'] is False
    assert dense.report['membership_authority']=='fresh_initial_vlm_dense_material_region'
    assert dense.report['side_anchor_pixels']>=4
    assert dense.report['thresholds']['component_step_inclusive_m'] is None
    assert not dense.report['grants_contact_permission'] and not dense.report['grants_free_space']
    assert not dense.report['map_modified'] and not dense.object_points.flags.writeable


def test_detection_box_never_becomes_dense_material_authority():
    obs,_,mask,reference=scene()
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs,annotation(obs,mask,kind='box'),reference)
    assert caught.value.diagnostics['failure']=='initial_material_mask_required'


def test_material_outside_fixed_envelope_rejects_whole_region_without_trimming():
    obs,_,mask,reference=scene(end=40)
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs,annotation(obs,mask),reference)
    assert caught.value.diagnostics['failure']=='initial_material_outside_fixed_envelope'
    assert caught.value.diagnostics['rejected_pixel_count']==4
    assert all(x==39 for x,y in caught.value.diagnostics['rejected_pixels_xy'])


def test_invalid_seed_and_unseeded_outlier_are_both_reported():
    obs,_,mask,reference=scene(end=40)
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs,annotation(obs,mask,seeds=((2,6),(0,0))),reference)
    failures=caught.value.diagnostics['abstentions']
    assert [f['failure'] for f in failures]==['invalid_explicit_seed','initial_material_outside_fixed_envelope']


def test_dense_material_cannot_self_supply_independent_side_anchors():
    obs,_,mask,reference=neighbor_scene()
    mask[:,:8]=False
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs,annotation(obs,mask,seeds=((9,6),)),reference)
    assert caught.value.diagnostics['failure']=='insufficient_original_overlap'


def test_mask_does_not_bypass_robot_or_support_seed_filters():
    obs,_,mask,reference=scene()
    for depth in (obs.depth.copy(),np.zeros_like(obs.depth)):
        with pytest.raises(ObjectOwnershipError) as caught:
            resolve(obs,annotation(obs,mask),reference,robot_depth=depth)
        assert 'robot_occluded_or_unverified' in caught.value.diagnostics['reasons']


def test_fresh_mask_transaction_requires_new_complete_reply_and_keeps_two_reply_cap():
    obs,xyz,mask,reference=neighbor_scene()
    client=ScriptedVLM([annotation(obs,mask,kind='box'),annotation(obs,mask)])
    result=annotate_initial_object_reference(client,obs,replace(obs,frame_id=8),reference,'mug',.9,
        material_mask=True)
    assert result.attempts[0].ownership is None
    assert np.array_equal(result.ownership.object_points,xyz[mask])
    assert result.report['explicit_initial_material_mask']
    assert result.report['semantic_requests']==2 and not result.report['reply_results_merged']
    assert 'EVERY depth-valid selected pixel' in client.requests[0][1]
    assert 'NO box' in client.requests[0][1]


def test_dense_object_mode_cannot_change_independent_part_initialization():
    obs,_,mask,reference=scene()
    client=ScriptedVLM([])
    result=annotate_initial_object_reference(client,obs,replace(obs,frame_id=8),reference,'mug',.9,
        material_mask=True,side_part=reference)
    assert result.ownership is None and not client.requests
