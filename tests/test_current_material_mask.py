"""Fresh current material labels: atomic semantics, unchanged map boundaries."""
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from view2grasp.config import ExperimentConfig,load_config
from view2grasp.object_annotation import annotate_source_object
from view2grasp.object_ownership import ObjectOwnershipError,resolve_object_ownership,source_endpoint_retirement_mask
from test_object_ownership import annotation,neighbor_scene,scene
from test_object_annotation import FakeVLM,source_scene


def resolve(obs,decision,reference,**kwargs):
    return resolve_object_ownership(obs,decision,reference,'mug',.9,material_mask=True,
        allow_partial_seed_components=True,**kwargs)


def test_current_dense_mask_is_new_semantic_authority_not_default_seed_expansion():
    obs,xyz,mask,reference=neighbor_scene();mask[7,10]=False
    decision=annotation(obs,mask)
    old=resolve_object_ownership(obs,decision,reference,'mug',.9,allow_partial_seed_components=True)
    dense=resolve(obs,decision,reference)
    assert old.object_mask.sum()==24 and np.array_equal(dense.object_mask,mask)
    assert np.array_equal(dense.object_points,xyz[mask]) and not dense.object_points.flags.writeable
    assert not dense.object_mask[7,10] and not len(dense.abstained_source_points)
    assert dense.report['membership_authority']=='fresh_vlm_dense_material_region'
    assert dense.report['allow_partial_seed_components'] is False
    assert not any(dense.report[key] for key in ('grants_contact_permission','grants_free_space','map_modified'))
    # Still only exact endpoints of THIS image may retire, including no
    # neighboring voxel or hole endpoint, even after explicit dense semantics.
    query=np.vstack([xyz[mask],xyz[7,10],xyz[mask][0]+[1e-7,0,0]])
    retire=source_endpoint_retirement_mask(query,dense,obs,table_z=.9)
    assert retire[:mask.sum()].all() and not retire[-2:].any()


def test_dense_current_box_and_invalid_seed_are_rejected_atomically_even_in_partial_mode():
    obs,_,mask,reference=scene()
    for decision,failure in [(annotation(obs,mask,kind='box'),'material_mask_required'),
            (annotation(obs,mask,seeds=((2,6),(0,0))),'invalid_explicit_seed')]:
        with pytest.raises(ObjectOwnershipError) as caught:resolve(obs,decision,reference)
        assert caught.value.diagnostics['failure']==failure


def test_outside_material_is_not_trimmed_and_cannot_self_anchor():
    obs,_,mask,reference=scene(end=40)
    with pytest.raises(ObjectOwnershipError) as caught:resolve(obs,annotation(obs,mask),reference)
    assert caught.value.diagnostics['failure']=='material_outside_fixed_envelope'
    assert caught.value.diagnostics['rejected_pixel_count']==4
    obs,_,mask,reference=neighbor_scene();mask[:,:8]=False
    with pytest.raises(ObjectOwnershipError) as caught:resolve(obs,annotation(obs,mask,seeds=((9,6),)),reference)
    assert caught.value.diagnostics['failure']=='insufficient_original_overlap'


def test_current_material_preserves_self_support_and_negative_pixel_filters():
    obs,_,mask,reference=scene()
    for kwargs in ({'robot_depth':obs.depth.copy()},{'robot_depth':np.zeros_like(obs.depth)}):
        with pytest.raises(ObjectOwnershipError) as caught:resolve(obs,annotation(obs,mask),reference,**kwargs)
        assert caught.value.diagnostics['failure']=='invalid_explicit_seed'
    decision=annotation(obs,mask);decision.groundings[0].negative_points=[(3/(mask.shape[1]-1),7/(mask.shape[0]-1))]
    with pytest.raises(ObjectOwnershipError) as caught:resolve(obs,decision,reference)
    assert caught.value.diagnostics['failure']=='invalid_explicit_seed'
    # A support-plane point cannot seed an otherwise explicit material region.
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve_object_ownership(obs,annotation(obs,mask),reference,'mug',1.,material_mask=True)
    assert 'support' in caught.value.diagnostics['reasons']


def test_dense_current_transaction_requests_fresh_labels_and_never_merges_replies():
    obs,reference,_=source_scene();client=FakeVLM(['left_bad','right'])
    result=annotate_source_object(client,obs,reference,'mug',.9,material_mask=True,allow_partial_seed_components=True)
    assert len(result.attempts)==2 and result.attempts[0].ownership is None
    assert result.ownership is not None and not result.report['reply_results_merged']
    assert result.report['explicit_material_mask'] and result.report['semantic_requests']==2
    assert all('EXPLICIT MATERIAL' in request['instruction'] for request in client.requests)
    assert all('A detection box is not accepted' in request['instruction'] for request in client.requests)
    assert all('ONE confident interior positive seed' in request['instruction'] for request in client.requests)
    bad=FakeVLM(['left_bad','all_bad'])
    rejected=annotate_source_object(bad,obs,reference,'mug',.9,material_mask=True,allow_partial_seed_components=True)
    assert len(rejected.attempts)==2 and rejected.ownership is None


def test_current_material_is_opt_in_and_only_config_difference():
    root=Path(__file__).resolve().parents[1]
    prior=load_config(root/'configs/mug-ranked-recovery.yaml').model_dump()
    dense=load_config(root/'configs/mug-current-material.yaml').model_dump()
    assert prior.pop('current_object_material_mask') is False
    assert dense.pop('current_object_material_mask') is True
    assert prior==dense and dense['max_observations']==12
    with pytest.raises(ValidationError,match='requires initial multiview object reference'):
        ExperimentConfig(current_object_material_mask=True)


def test_runner_applies_dense_current_authority_only_to_new_postclose_images(tmp_path,monkeypatch):
    from test_source_ownership_lifecycle import run_lifecycle,_contains
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,max_observations=5,
        current_material=True,registration_ok=False)
    assert e.material_modes and all(enabled and frame>=3 for frame,enabled in e.material_modes)
    assert [f for f,_ in e.retired]==[f for f,_ in e.material_modes]
    assert e.planner.registered_payload_local is None
    assert not e.map.contact_allowed(e.predicted).any() and _contains(e.map.surface,e.neighbor[None])
    assert not result['success']
