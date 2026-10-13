"""Unseeded surface review is advisory; only explicit seeds grant ownership."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from view2grasp.geometry import unproject
from view2grasp.grounding import mask_rle
from view2grasp.mapping import DepthMap, OCCUPIED
from view2grasp.object_ownership import (ObjectOwnershipError, resolve_object_ownership,
    source_endpoint_retirement_mask)
from view2grasp.types import AgentDecision, GroundingProposal, Observation


def scene(blobs=((2,2,6,5),(16,6,20,9)),pitch=.001):
    depth=np.full((24,80),np.nan)
    masks=[]
    for x0,y0,x1,y1 in blobs:
        mask=np.zeros(depth.shape,bool);mask[y0:y1,x0:x1]=True
        depth[mask]=1.;masks.append(mask)
    obs=Observation(7,np.zeros((*depth.shape,3),np.uint8),depth,
        np.diag([1/pitch,1/pitch,1.]),np.eye(4),np.zeros(7),np.array([.04,-.04]),
        12.,(0.,55.,.8))
    yy,xx=np.mgrid[:24,:80]
    xyz=unproject(np.stack([xx,yy],axis=-1),depth,obs.K,obs.T_wc)
    reference=xyz[np.any(masks,axis=0)]
    return obs,xyz,masks,reference


def annotation(obs,masks,seeds=((3,3),),kind='mask',negatives=(),holes=()):
    h,w=obs.depth.shape
    def xy(points):return [(x/(w-1),y/(h-1)) for x,y in points]
    common=dict(role='object',target_id='mug',frame_id=obs.frame_id,camera_id=obs.camera_id)
    if kind=='mask':
        mask=np.any(masks,axis=0)
        region=GroundingProposal(kind='mask',mask_size=mask.shape,mask_rle=mask_rle(mask),**common)
    else:
        region=GroundingProposal(kind='polygon',points=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)],
            holes=[xy(ring) for ring in holes],**common)
    return AgentDecision(action='observe',frame_id=obs.frame_id,camera_id=obs.camera_id,
        groundings=[region,GroundingProposal(kind='points',points=xy(seeds),
            negative_points=xy(negatives),**common)])


def resolve(obs,reply,reference,**kwargs):
    return resolve_object_ownership(obs,reply,reference,'mug',.9,**kwargs)


def test_review_is_not_ownership_and_does_not_retire_unseeded_material():
    obs,xyz,masks,reference=scene();reply=annotation(obs,masks)
    before=copy.deepcopy(reply);old_depth=obs.depth.copy();old_reference=reference.copy()
    strict=resolve(obs,reply,reference)
    result=resolve(obs,reply,reference,allow_partial_seed_components=True)
    assert result.report['unresolved_measured_components']==[
        {'pixels':12,'original_anchor_pixels':12,'bbox_xyxy':[16,6,19,8]}]
    assert result.report['unresolved_measured_component_count']==1
    assert 'not object identity' in result.report['review_scope']
    assert 'inclusive source-image pixels' in result.report['review_scope']
    assert np.array_equal(result.object_mask,strict.object_mask)
    assert np.array_equal(result.object_points,strict.object_points)
    assert np.array_equal(result.source_pixels_xy,strict.source_pixels_xy)
    assert not result.object_mask[masks[1]].any()
    assert reply==before and np.array_equal(obs.depth,old_depth,equal_nan=True)
    assert np.array_equal(reference,old_reference)
    for value in (result.object_mask,result.object_points,result.source_pixels_xy,
                  result.abstained_source_pixels_xy,result.abstained_source_points):
        assert not value.flags.writeable
    surface=xyz[np.any(masks,axis=0)]
    remove=source_endpoint_retirement_mask(surface,result,obs)
    assert np.array_equal(surface[remove],xyz[masks[0]])
    map_=DepthMap();map_.surface=surface.copy()
    map_.retire_owned_source(result,obs)
    assert np.all(map_.classify(xyz[masks[1]])==OCCUPIED)


def test_default_strict_report_has_no_review_fields_or_behavior_change():
    obs,xyz,masks,reference=scene();reply=annotation(obs,masks)
    result=resolve(obs,reply,reference)
    assert not any(k.startswith('unresolved_measured') or k=='review_scope' for k in result.report)
    assert np.array_equal(result.object_mask,masks[0])
    assert np.array_equal(result.object_points,xyz[masks[0]])
    assert result.report['unselected_valid_region_pixels']==12


def test_review_caps_display_at_three_with_stable_ties_and_total_count():
    blobs=((2,2,6,5),(15,2,18,4),(28,2,32,5),(40,2,44,5),(52,2,55,5),(64,2,66,4))
    obs,_,masks,reference=scene(blobs)
    result=resolve(obs,annotation(obs,masks),reference,allow_partial_seed_components=True)
    assert result.report['unresolved_measured_component_count']==5
    assert result.report['unresolved_measured_components']==[
        {'pixels':12,'original_anchor_pixels':12,'bbox_xyxy':[28,2,31,4]},
        {'pixels':12,'original_anchor_pixels':12,'bbox_xyxy':[40,2,43,4]},
        {'pixels':9,'original_anchor_pixels':9,'bbox_xyxy':[52,2,54,4]}]


def test_source_image_border_is_not_dropped_from_complete_component():
    obs,_,masks,reference=scene(((2,2,6,5),(78,0,80,2)))
    result=resolve(obs,annotation(obs,masks),reference,allow_partial_seed_components=True)
    assert result.report['unresolved_measured_components']==[
        {'pixels':4,'original_anchor_pixels':4,'bbox_xyxy':[78,0,79,1]}]


def test_10mm_neighbor_cannot_borrow_seeded_objects_anchors_for_review():
    obs,xyz,masks,_=scene(((2,2,6,5),(6,2,10,5)))
    reference=xyz[masks[0]].copy();depth=obs.depth.copy();depth[masks[1]]=1.01
    obs=replace(obs,depth=depth)
    result=resolve(obs,annotation(obs,masks),reference,allow_partial_seed_components=True)
    assert result.report['unresolved_measured_components']==[]
    assert result.report['unresolved_measured_component_count']==0
    assert np.array_equal(result.object_mask,masks[0])


@pytest.mark.parametrize('anchors',[0,3])
def test_unseeded_component_requires_four_own_fixed_reference_anchors(anchors):
    obs,xyz,masks,_=scene(((2,2,6,5),(16,6,20,7)),pitch=.0055)
    # A distant reference bounds the envelope without supplying close anchors.
    reference=np.vstack([xyz[masks[0]],xyz[masks[1]][:anchors],xyz[6,19]+[0,0,.02]])
    result=resolve(obs,annotation(obs,masks),reference,allow_partial_seed_components=True)
    assert result.report['unresolved_measured_component_count']==0
    assert result.report['unresolved_measured_components']==[]


@pytest.mark.parametrize('exclusion',['hole','negative','self','support','invalid'])
def test_holes_negative_points_and_existing_depth_filters_cannot_create_review(exclusion):
    obs,xyz,masks,reference=scene(((2,2,6,5),(16,6,19,9)))
    own=np.full(obs.depth.shape,10.)
    kwargs={'kind':'polygon'}
    if exclusion=='hole':kwargs['holes']=[[(15,5),(19,5),(19,9),(15,9)]]
    elif exclusion=='negative':kwargs['negatives']=[(17,7)]
    elif exclusion=='self':own[masks[1]]=obs.depth[masks[1]]
    else:
        depth=obs.depth.copy();depth[masks[1]]=.905 if exclusion=='support' else np.nan
        obs=replace(obs,depth=depth)
    result=resolve(obs,annotation(obs,masks,**kwargs),reference,robot_depth=own,
        allow_partial_seed_components=True)
    assert result.report['unresolved_measured_components']==[]
    assert result.report['unresolved_measured_component_count']==0
    assert np.array_equal(result.object_points,xyz[masks[0]])


def test_complete_unseeded_component_crossing_aabb_is_not_clipped_into_review():
    obs,xyz,masks,_=scene(((2,2,6,5),(20,6,70,9)))
    reference=np.vstack([xyz[masks[0]],xyz[6:9,20:24].reshape(-1,3)])
    result=resolve(obs,annotation(obs,masks),reference,allow_partial_seed_components=True)
    assert result.report['unresolved_measured_components']==[]
    assert result.report['unresolved_measured_component_count']==0
    assert result.report['unselected_valid_region_pixels']==150
    assert not result.object_mask[masks[1]].any()


def test_accepted_and_abstained_components_are_not_offered_for_review():
    obs,xyz,masks,_=scene(((2,2,6,5),(16,6,20,9),(28,6,32,9),(40,6,44,9)))
    reference=np.vstack([xyz[masks[0]],xyz[masks[1]],xyz[masks[3]]])
    reply=annotation(obs,masks,seeds=((3,3),(17,7),(29,7)))
    result=resolve(obs,reply,reference,allow_partial_seed_components=True)
    assert len(result.report['selected_components'])==2
    assert result.report['abstentions'][0]['failure']=='unanchored_seeded_component'
    assert len(result.abstained_source_points)==12
    assert result.report['unresolved_measured_components']==[
        {'pixels':12,'original_anchor_pixels':12,'bbox_xyxy':[40,6,43,8]}]
    assert result.report['unresolved_measured_component_count']==1


def test_review_can_explain_a_failed_reply_without_turning_it_into_ownership():
    obs,_,masks,reference=scene()
    reply=annotation(obs,masks,seeds=((70,20),))
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs,reply,reference,allow_partial_seed_components=True)
    assert caught.value.diagnostics['failure']=='no_accepted_components'
    assert caught.value.diagnostics['unresolved_measured_component_count']==2
    assert len(caught.value.diagnostics['unresolved_measured_components'])==2
