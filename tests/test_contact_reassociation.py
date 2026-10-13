"""Current-frame semantics gate contact; geometric tracking cannot grant it."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.geometry import unproject
from view2grasp.grounding import GroundingResolver,mask_rle,track_surface
from view2grasp.mapping import DepthMap
from view2grasp.types import AgentDecision,GroundingProposal,Observation


def scene():
    depth=np.full((24,24),2.);depth[10:14,8:12]=1.;depth[10:14,12:16]=1.01
    obs=Observation(4,np.zeros((24,24,3),np.uint8),depth,
        np.array([[300.,0,11.5],[0,300.,11.5],[0,0,1.]]),np.eye(4),
        np.zeros(7),np.array([.04,-.04]),0.,(0.,55.,.8))
    v,u=np.mgrid[:24,:24];xyz=unproject(np.stack([u,v],-1),depth,obs.K,obs.T_wc)
    object_mask=np.zeros((24,24),bool);object_mask[10:14,8:12]=True
    reference=xyz[10:14,8:10].reshape(-1,3)
    return obs,xyz,object_mask,reference


def annotations(obs,object_mask,part_mask=None,target_id='target'):
    return AgentDecision(action='observe',frame_id=obs.frame_id,camera_id=obs.camera_id,
        groundings=[GroundingProposal(kind='mask',role=role,frame_id=obs.frame_id,
            camera_id=obs.camera_id,target_id=target_id,mask_size=mask.shape,mask_rle=mask_rle(mask))
            for role,mask in (('object',object_mask),('grasp_part',object_mask if part_mask is None else part_mask))])


def test_explicit_mask_recovers_new_side_but_nearby_independent_object_stays_obstacle():
    obs,xyz,mask,reference=scene();neighbor=xyz[10:14,12:16].reshape(-1,3)
    # This is the actual counterexample to granting contact from track_surface.
    tracked=track_surface(obs,xyz[mask],support_z=.95)
    assert len(tracked)==32 and np.count_nonzero(cKDTree(neighbor).query(tracked)[0]<1e-9)==16
    association=GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,'target',.95)
    assert len(association.object_points)==len(association.part_points)==16
    assert np.min(cKDTree(neighbor).query(association.part_points)[0])>.009
    map_=DepthMap(.005);map_.surface=np.vstack([xyz[mask],neighbor]);map_.tree=cKDTree(map_.surface)
    new_side=xyz[11,11][None];map_.set_target(reference)
    assert not map_.contact_allowed(new_side)[0]
    map_.set_target(association.part_points)
    assert map_.contact_allowed(new_side)[0]
    assert not map_.contact_allowed(neighbor).any()


def test_even_an_overbroad_semantic_mask_cannot_bridge_the_10mm_depth_gap():
    obs,xyz,mask,reference=scene();mask[10:14,12:16]=True
    with pytest.raises(ValueError,match='unanchored semantic surface'):
        GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,'target',.95)


def test_part_contact_does_not_expand_to_other_object_surfaces():
    obs,xyz,mask,reference=scene();part=np.zeros_like(mask);part[10:14,8:10]=True
    result=GroundingResolver().reassociate_contact(obs,annotations(obs,mask,part),reference,reference,'target',.95)
    assert len(result.object_points)==16 and len(result.part_points)==8
    assert np.array_equal(np.sort(result.part_points,axis=0),np.sort(reference,axis=0))


@pytest.mark.parametrize('fault',['stale','wrong_camera','wrong_target','missing_part','box','outside_part','different_part'])
def test_missing_or_inconsistent_semantics_do_not_authorize_contact(fault):
    obs,xyz,mask,reference=scene();decision=annotations(obs,mask)
    if fault=='stale':decision=decision.model_copy(update={'frame_id':3})
    elif fault=='wrong_camera':decision=decision.model_copy(update={'camera_id':'unseen'})
    elif fault=='wrong_target':decision=annotations(obs,mask,target_id='neighbor')
    elif fault=='missing_part':decision=decision.model_copy(update={'groundings':decision.groundings[:1]})
    elif fault=='box':
        decision=decision.model_copy(update={'groundings':[g.model_copy(update={'kind':'box','box':(.2,.2,.8,.8)}) for g in decision.groundings]})
    elif fault=='outside_part':
        part=mask.copy();part[10,12]=True;decision=annotations(obs,mask,part)
    else:reference=xyz[10:14,12:16].reshape(-1,3)
    with pytest.raises(ValueError):
        GroundingResolver().reassociate_contact(obs,decision,reference,reference,'target',.95)


@pytest.mark.parametrize('robot_distance',[1.,.9])
def test_support_and_robot_returns_are_excluded_even_inside_semantic_masks(robot_distance):
    obs,xyz,mask,reference=scene();robot_depth=np.full(obs.depth.shape,10.)
    robot_depth[10,8]=robot_distance
    result=GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,'target',.95,robot_depth)
    assert not result.object_mask[10,8] and not result.part_mask[10,8]
    with pytest.raises(ValueError,match='insufficient filtered depth'):
        GroundingResolver().reassociate_contact(obs,annotations(obs,mask),reference,reference,'target',1.)


def test_saved_side_surface_can_be_associated_only_with_new_explicit_semantics():
    """Offline human annotation is a regression fixture, not model success."""
    fixture=np.load(Path(__file__).parent/'fixtures/occluded_contact_reassociation.npz')
    metadata=json.loads(str(fixture['metadata']))
    obs=Observation(4,fixture['rgb'],fixture['depth'],fixture['K'],fixture['T_wc'],
        np.zeros(7),np.array([.04,-.04]),0.,(0.,0.,1.))
    mask=fixture['semantic_mask'];reference=fixture['reference_object']
    result=GroundingResolver().reassociate_contact(obs,annotations(obs,mask,target_id='blue_block'),
        reference,fixture['reference_part'],'blue_block',float(fixture['support_z']))
    endpoint=fixture['new_side_endpoint']
    assert cKDTree(reference).query(endpoint)[0]>.008
    assert cKDTree(result.part_points).query(endpoint)[0]<1e-9
    assert metadata['annotation_source']=='offline human polygon; not an online VLM result'
    assert metadata['source_frame_id']==4 and result.report['grasp_ray_changed'] is False
