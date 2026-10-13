"""Object membership may grow without moving contact or broadening its part."""
import copy

import numpy as np
import pytest

from view2grasp.geometry import pixel_point
from view2grasp.grounding import GroundingResolver,GroundingSeedError,mask_rle
from view2grasp.types import AgentDecision,GroundingProposal,Observation


def scene():
    h,w=32,48
    depth=np.full((h,w),np.nan)
    depth[8:15,6:13]=1.
    depth[8:15,30:34]=1.03
    depth[22:27,20:25]=1.2
    return Observation(1,np.zeros((h,w,3),np.uint8),depth,
        np.array([[400.,0,24],[0,400.,16],[0,0,1]]),np.eye(4),
        np.zeros(7),np.array([.04,-.04]),0.,(0.,55.,.8))


def xy(x,y):return (x/47,y/31)


def proposal(kind,role,**kwargs):
    return GroundingProposal(kind=kind,role=role,frame_id=1,target_id='one_object',**kwargs)


def decision(*,object_seed=True,embedded=False,contact=True):
    object_box=proposal('box','object',box=(*xy(4,6),*xy(36,28)),
        points=[xy(9,11)] if object_seed and embedded else [])
    proposals=[object_box]
    if object_seed and not embedded:proposals.append(proposal('points','object',points=[xy(9,11)]))
    proposals.append(proposal('box','grasp_part',box=(*xy(29,7),*xy(34,15))))
    if contact:proposals.append(proposal('points','grasp_contact',points=[xy(31,11)]))
    return AgentDecision(action='grasp',frame_id=1,groundings=proposals)


@pytest.mark.parametrize('embedded',[False,True])
@pytest.mark.parametrize('contact',[False,True])
def test_explicit_object_seed_adds_measured_component_without_contact_or_part_change(embedded,contact):
    obs=scene();before=decision(object_seed=False,contact=contact)
    supplied=decision(embedded=embedded,contact=contact);snapshot=copy.deepcopy(supplied)
    baseline=GroundingResolver().resolve_grasp(obs,before,.5)
    resolver=GroundingResolver();actual=resolver.resolve_grasp(obs,supplied,.5)
    assert actual.object_mask.sum()==77 and baseline.object_mask.sum()==28
    assert actual.object_mask[11,9] and actual.object_mask[11,31]
    assert not actual.object_mask[24,22]  # No seed selects this third island.
    assert np.array_equal(actual.part_mask,baseline.part_mask)
    assert actual.decision.point==baseline.decision.point
    assert np.array_equal(pixel_point(obs,actual.decision.point),pixel_point(obs,baseline.decision.point))
    assert actual.report['point_selection']==baseline.report['point_selection']
    assert actual.decision.groundings==supplied.groundings and supplied==snapshot
    assert resolver.calls==actual.report['sam_calls']==0
    assert actual.report['explicit_object_seed_components'] is True
    assert actual.report['explicit_object_seed_count']==1
    assert 'explicit_object_seed_components' not in baseline.report


def test_extra_object_component_does_not_restore_disconnected_part_permission():
    obs=scene()
    object_box=decision(object_seed=False).groundings[0]
    wide_part=object_box.model_copy(update={'role':'grasp_part'})
    two_part_seeds=proposal('points','grasp_part',points=[xy(9,11),xy(31,11)])
    baseline=AgentDecision(action='grasp',frame_id=1,groundings=[object_box,wide_part,two_part_seeds])
    extra=proposal('points','object',points=[xy(31,11)])
    supplied=baseline.model_copy(update={'groundings':[object_box,extra,wide_part,two_part_seeds]})
    old=GroundingResolver().resolve_grasp(obs,baseline,.5)
    actual=GroundingResolver().resolve_grasp(obs,supplied,.5)
    assert old.object_mask.sum()==old.part_mask.sum()==49
    assert actual.object_mask.sum()==77
    assert np.array_equal(actual.part_mask,old.part_mask)
    assert not actual.part_mask[11,31] and actual.object_mask[11,31]
    assert actual.decision.point==old.decision.point


@pytest.mark.parametrize('fault,reason',[
    ('outside','outside_proposed_region'),('depth','invalid_depth'),
    ('support','support_plane'),('robot','robot_self_return'),
    ('negative','negative_point'),('hole','outside_proposed_region')])
def test_any_invalid_body_seed_rejects_the_whole_result_with_object_diagnostics(fault,reason):
    obs=scene();supplied=decision();own=np.full_like(obs.depth,10.)
    if fault=='outside':supplied.groundings[0]=supplied.groundings[0].model_copy(update={'box':(*xy(20,6),*xy(36,28))})
    elif fault=='depth':obs.depth[11,9]=np.nan
    elif fault=='support':obs.depth[11,9]=.5
    elif fault=='robot':own[11,9]=obs.depth[11,9]
    elif fault=='negative':supplied.groundings[0]=supplied.groundings[0].model_copy(update={'negative_points':[xy(9,11)]})
    elif fault=='hole':
        supplied.groundings[0]=proposal('polygon','object',
            points=[xy(4,6),xy(36,6),xy(36,28),xy(4,28)],
            holes=[[xy(8,10),xy(10,10),xy(10,12),xy(8,12)]])
    before=copy.deepcopy(supplied)
    with pytest.raises(GroundingSeedError) as caught:
        GroundingResolver().resolve_grasp(obs,supplied,.5,own)
    details=caught.value.diagnostics
    assert details['role']=='object' and details['rejected_seed_count']==1
    assert details['rejected_seeds'][0]['seed_origin']=='explicit_object'
    assert details['rejected_seeds'][0]['pixel']==[9,11]
    assert reason in details['rejected_seeds'][0]['reasons']
    assert 'All supplied seeds remain binding' in str(caught.value)
    assert 'omit optional contact' not in str(caught.value)
    assert supplied==before


def test_valid_body_does_not_rescue_a_resolved_contact_outside_object_region():
    obs=scene();supplied=decision()
    supplied.groundings[0]=supplied.groundings[0].model_copy(update={'box':(*xy(4,6),*xy(15,17))})
    with pytest.raises(GroundingSeedError) as caught:
        GroundingResolver().resolve_grasp(obs,supplied,.5)
    assert caught.value.diagnostics['role']=='object'
    assert caught.value.diagnostics['rejected_seeds']==[{
        'seed_index':1,'normalized':list(xy(31,11)),'pixel':[31,11],
        'reasons':['outside_proposed_region'],'seed_origin':'resolved_contact'}]


def test_polygon_vertices_are_not_object_membership_seeds():
    obs=scene();supplied=decision(object_seed=False)
    supplied.groundings[0]=proposal('polygon','object',
        points=[xy(4,6),xy(36,6),xy(36,28),xy(4,28)])
    # Every polygon vertex is invalid depth. Treating vertices as seeds would fail.
    actual=GroundingResolver().resolve_grasp(obs,supplied,.5)
    assert actual.object_mask.sum()==actual.part_mask.sum()==28
    assert actual.decision.point==xy(31,11)


def test_missing_object_role_does_not_reinterpret_part_seeds_as_object_seeds():
    obs=scene();box=decision(object_seed=False).groundings[0].model_copy(update={'role':'grasp_part'})
    supplied=AgentDecision(action='grasp',frame_id=1,groundings=[box,
        proposal('points','grasp_part',points=[xy(9,11),xy(31,11)])])
    actual=GroundingResolver().resolve_grasp(obs,supplied,.5)
    assert actual.object_mask.sum()==actual.part_mask.sum()==49


def test_no_object_seed_preserves_masks_and_legacy_point_selection_policy():
    obs=scene();box_input=decision(object_seed=False)
    box_result=GroundingResolver().resolve_grasp(obs,box_input,.5)
    mask=np.zeros(obs.depth.shape,bool);mask[6:29,4:37]=True
    masked=box_input.model_copy(update={'groundings':[
        proposal('mask','object',mask_size=mask.shape,mask_rle=mask_rle(mask)),*box_input.groundings[1:]]})
    legacy=AgentDecision(action='grasp',frame_id=1,point=xy(31,11),
        bbox=box_input.groundings[0].box,part_bbox=box_input.groundings[1].box)
    for supplied in (masked,legacy):
        actual=GroundingResolver().resolve_grasp(obs,supplied,.5)
        assert np.array_equal(actual.object_mask,box_result.object_mask)
        assert np.array_equal(actual.part_mask,box_result.part_mask)
        if supplied is masked:
            assert actual.decision.point==box_result.decision.point
        else:
            # The legacy point seeds the part; it is not an explicit contact.
            assert actual.decision.point==xy(31,9)
            assert actual.report['point_selection']['method']=='seeded_part_global_interior'


def test_contact_component_recheck_does_not_repeat_segmenter_calls():
    class Segmenter:
        def __init__(self):self.proposals=[]
        def segment(self,obs,proposal):
            self.proposals.append(proposal)
            return np.ones(obs.depth.shape,bool)
    obs=scene();segmenter=Segmenter();resolver=GroundingResolver(segmenter)
    result=resolver.resolve_grasp(obs,decision(),.5)
    assert result.object_mask.sum()==77 and result.part_mask.sum()==28
    assert len(segmenter.proposals)==resolver.calls==result.report['sam_calls']==3


def test_object_membership_alone_cannot_choose_contact_or_grasp_part():
    obs=scene();supplied=decision()
    supplied=supplied.model_copy(update={'groundings':[
        g for g in supplied.groundings if g.role=='object']})
    with pytest.raises(ValueError,match='independent grasp_part or grasp_contact'):
        GroundingResolver().resolve_grasp(obs,supplied,.5)


@pytest.mark.parametrize('embedded',[False,True])
def test_independent_contact_fallback_never_uses_membership_seeds_for_part_or_sam(embedded):
    class Segmenter:
        def __init__(self):self.proposals=[]
        def segment(self,obs,proposal):
            self.proposals.append(proposal)
            return np.ones(obs.depth.shape,bool)
    obs=scene();supplied=decision(embedded=embedded)
    supplied=supplied.model_copy(update={'groundings':[
        g for g in supplied.groundings if g.role!='grasp_part']})
    segmenter=Segmenter();actual=GroundingResolver(segmenter).resolve_grasp(obs,supplied,.5)
    assert actual.decision.point==xy(31,11)
    assert actual.object_mask.sum()==77 and actual.part_mask.sum()==28
    assert segmenter.proposals[0].role=='grasp_contact'
    assert segmenter.proposals[0].points==[xy(31,11)]
    assert all(g.role=='object' for g in segmenter.proposals[1:])
