"""Boundary-repair diagnostics never accept a partial or invented surface."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from view2grasp.grounding import ContactAssociationError, GroundingResolver
from view2grasp.runner import _contact_annotation_progressed
from view2grasp.types import AgentDecision, GroundingProposal, Observation
from test_contact_reassociation import annotations, scene


def rejected(obs, decision, reference, *, robot_depth=None, support=.95):
    with pytest.raises(ContactAssociationError) as caught:
        GroundingResolver().reassociate_contact(obs,decision,reference,reference,
            'target',support,robot_depth)
    return caught.value.diagnostics


def synthetic_pair():
    obs,xyz,mask,reference=scene()
    neighbor=mask.copy();neighbor[10:14,12:16]=True
    first=rejected(obs,annotations(obs,neighbor),reference)
    clipped=mask.copy();clipped[:,10]=False
    second=rejected(obs,annotations(obs,clipped),reference)
    return first,second


def test_observed_boundary_repair_only_unlocks_final_reply_without_acceptance():
    first,second=synthetic_pair()
    assert first['failure']==second['failure']=='unanchored_component'
    assert first['anchor_pixels']>second['anchor_pixels']>=4
    assert first['component_anchor_repairability']['anchor_reachable_fraction']==0.
    assert second['component_anchor_repairability']['anchor_reachable_fraction']==1.
    assert _contact_annotation_progressed(first,second)
    assert not _contact_annotation_progressed(first,first)
    assert not _contact_annotation_progressed(second,second)
    reverse=copy.deepcopy(first)
    reverse['anchor_pixels']=second['anchor_pixels']  # Do not invoke the preserved legacy anchor rule.
    assert not _contact_annotation_progressed(second,reverse)
    # Diagnostic fields include no replacement mask or proposed positive pixels.
    metric=second['component_anchor_repairability']
    assert metric['grants_semantic_or_contact_permission'] is False
    assert not {'mask','points','pixel','proposed_region'} & metric.keys()


def test_saved_frame4_boundary_progress_and_actual_third_response_remain_strict():
    fixture=np.load(Path(__file__).parent/'fixtures/contact_topology_progress.npz')
    metadata=json.loads(str(fixture['metadata']))
    obs=Observation(4,fixture['rgb'],fixture['depth'],fixture['K'],fixture['T_wc'],
        fixture['q'],fixture['gripper_q'],float(fixture['timestamp']),tuple(fixture['view']))
    resolver=GroundingResolver();diagnostics=[]
    for key,anchors,rejected_count in [('first',385,15),('second',383,17)]:
        with pytest.raises(ContactAssociationError) as caught:
            resolver.reassociate_contact(obs,AgentDecision.model_validate_json(str(fixture[key])),
                fixture['reference_object'],fixture['reference_part'],'blue_block',
                float(fixture['support_z']),fixture['robot_depth'],source_supported=True)
        report=caught.value.diagnostics
        assert report['anchor_pixels']==anchors and report['rejected_pixel_count']==rejected_count
        diagnostics.append(report)
    assert [d['component_anchor_repairability']['anchor_reachable_fraction'] for d in diagnostics]==[0.,1.]
    assert _contact_annotation_progressed(*diagnostics)
    association=resolver.reassociate_contact(obs,AgentDecision.model_validate_json(str(fixture['third'])),
        fixture['reference_object'],fixture['reference_part'],'blue_block',
        float(fixture['support_z']),fixture['robot_depth'],source_supported=True)
    assert np.array_equal(association.object_mask,fixture['third_object_mask'])
    assert np.array_equal(association.part_mask,fixture['third_part_mask'])
    assert association.report['object_pixels']==association.report['part_pixels']==712
    assert association.report['original_surface_overlap_pixels']=={'object':379,'grasp_part':379}
    assert metadata['source_snapshot_sha256']
    assert 'NOT episode success' in metadata['third_reply']


@pytest.mark.parametrize('fault',['background','invalid','support','robot_at_bridge','robot_in_front'])
def test_non_object_or_unmeasured_returns_cannot_make_a_repair_bridge(fault):
    obs,xyz,mask,reference=scene();mask[:,10]=False
    depth=obs.depth.copy();robot_depth=None
    if fault=='background':depth[10:14,10]=2.
    elif fault=='invalid':depth[10:14,10]=np.nan
    elif fault=='support':depth[10:14,10]=.95
    else:
        robot_depth=np.full(depth.shape,10.)
        robot_depth[10:14,10]=1. if fault=='robot_at_bridge' else .9
    report=rejected(replace(obs,depth=depth),annotations(obs,mask),reference,robot_depth=robot_depth)
    assert report['failure']=='unanchored_component'
    assert report['component_anchor_repairability']['anchor_reachable_pixels']==0


@pytest.mark.parametrize('exclusion',['negative_point','polygon_hole'])
def test_explicit_negative_semantics_cannot_supply_outside_boundary_bridge(exclusion):
    obs,xyz,mask,reference=scene();depth=obs.depth.copy();depth[10:14,8:16]=1.
    obs=replace(obs,depth=depth);mask[:]=False;mask[10:14,8:10]=True;mask[10:14,15]=True
    decision=annotations(obs,mask)
    if exclusion=='negative_point':
        decision=decision.model_copy(update={'groundings':[g.model_copy(update={
            'negative_points':[(12/23,11/23)]}) for g in decision.groundings]})
    else:
        decision=decision.model_copy(update={'groundings':[
            GroundingProposal(kind='polygon',role=role,frame_id=4,target_id='target',
                points=[(8/23,10/23),(15/23,10/23),(15/23,13/23),(8/23,13/23)],
                holes=[[(10/23,9/23),(14/23,9/23),(14/23,14/23),(10/23,14/23)]])
            for role in ('object','grasp_part')]})
    report=rejected(obs,decision,reference)
    assert report['failure']=='unanchored_component'
    assert report['component_anchor_repairability']['anchor_reachable_pixels']==0


def test_ten_mm_neighbor_and_missing_original_anchors_still_fail_closed():
    obs,xyz,mask,reference=scene();mask[10:14,12:16]=True
    report=rejected(obs,annotations(obs,mask),reference)
    assert report['component_anchor_repairability']['anchor_reachable_pixels']==0
    report=rejected(obs,annotations(obs,mask),reference+[0,0,.02])
    assert report['failure']=='reference_inconsistency' and report['anchor_pixels']==0
    assert 'component_anchor_repairability' not in report


@pytest.mark.parametrize('fault',['frame','camera','different_role','few_anchors','expansion',
    'wrong_failure','missing_metric','different_evidence','permission','partial_reachability',
    'zero_denominator','bool_count','rounded_ratio_lies'])
def test_topology_gate_rejects_stale_ineligible_or_incomplete_progress(fault):
    previous,current=synthetic_pair();current=copy.deepcopy(current)
    metric=current['component_anchor_repairability']
    if fault=='frame':current['frame_id']+=1
    elif fault=='camera':current['camera_id']='other'
    elif fault=='different_role':previous['role']='grasp_part'
    elif fault=='few_anchors':current['anchor_pixels']=3
    elif fault=='expansion':current['rejection_counts']['reference_distance']=1
    elif fault=='wrong_failure':current['failure']='reference_inconsistency'
    elif fault=='missing_metric':current.pop('component_anchor_repairability')
    elif fault=='different_evidence':metric['evidence_sha256']='different reference or image'
    elif fault=='permission':metric['grants_semantic_or_contact_permission']=True
    elif fault=='partial_reachability':metric['anchor_reachable_pixels']-=1
    elif fault=='zero_denominator':metric['component_pixels']=0
    elif fault=='bool_count':metric['component_pixels']=True
    elif fault=='rounded_ratio_lies':metric['anchor_reachable_pixels']=0;metric['anchor_reachable_fraction']=1.
    assert not _contact_annotation_progressed(previous,current)


def test_original_anchor_rejection_and_role_progress_rules_are_preserved():
    previous,current=synthetic_pair()
    current['anchor_pixels']=previous['anchor_pixels']+1
    assert _contact_annotation_progressed(previous,current)
    current['anchor_pixels']=previous['anchor_pixels']
    current['rejected_pixel_count']=previous['rejected_pixel_count']-1
    assert _contact_annotation_progressed(previous,current)
    current['anchor_pixels']=0;current['role']='grasp_part'
    assert _contact_annotation_progressed(previous,current)
    assert not _contact_annotation_progressed(None,current)


@pytest.mark.parametrize('field,value',[('frame_id',99),('camera_id','stale')])
def test_even_legacy_progress_requires_the_same_source_frame_and_camera(field,value):
    previous,current=synthetic_pair()
    current['anchor_pixels']=previous['anchor_pixels']+1
    current[field]=value
    assert not _contact_annotation_progressed(previous,current)
