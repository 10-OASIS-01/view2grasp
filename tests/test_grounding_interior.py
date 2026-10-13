"""Unseeded contact prefers the center only among maximum-interior pixels."""
import cv2
import numpy as np
import pytest

from view2grasp.grounding import GroundingResolver,mask_rle
from view2grasp.types import AgentDecision,GroundingProposal,Observation


def resolve(mask,seed_role=None,seed=None):
    size=mask.shape[0]
    obs=Observation(1,np.zeros((*mask.shape,3),np.uint8),np.ones(mask.shape),
        np.array([[300.,0,(size-1)/2],[0,300.,(size-1)/2],[0,0,1.]]),
        np.eye(4),np.zeros(7),np.zeros(2),0.,(0.,0.,1.))
    proposals=[GroundingProposal(kind='mask',role=role,frame_id=1,
        mask_size=mask.shape,mask_rle=mask_rle(mask),
        points=[seed] if seed_role==role else []) for role in ('object','grasp_part')]
    if seed_role=='grasp_contact':
        proposals.append(GroundingProposal(kind='points',role=seed_role,frame_id=1,points=[seed]))
    result=GroundingResolver().resolve_grasp(obs,
        AgentDecision(action='grasp',frame_id=1,groundings=proposals),.5)
    pixel=np.rint(np.asarray(result.decision.point)*[size-1,size-1]).astype(int)
    return result,tuple(pixel)


@pytest.mark.parametrize('transpose',[False,True])
def test_long_narrow_unseeded_region_centers_equal_clearance_pixels(transpose):
    mask=np.zeros((41,41),bool);mask[10:31,19:22]=True
    if transpose:mask=mask.T.copy()
    result,pixel=resolve(mask)
    assert pixel==(20,20)
    assert np.array_equal(result.part_mask,mask)
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    assert distance[pixel[1],pixel[0]]==distance.max()
    assert np.unravel_index(distance.argmax(),distance.shape)!=(20,20)


def test_mask_hole_is_never_replaced_by_the_region_centroid():
    mask=np.zeros((41,41),bool);mask[5:36,5:36]=True;mask[12:29,12:29]=False
    result,(x,y)=resolve(mask)
    assert not mask[20,20] and mask[y,x]
    assert np.array_equal(result.part_mask,mask)
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    assert distance[y,x]==distance.max()


def test_disconnected_regions_keep_existing_selected_depth_component():
    mask=np.zeros((41,41),bool);mask[6:35,8:11]=True;mask[6:35,30:33]=True
    result,pixel=resolve(mask)
    expected=np.zeros_like(mask);expected[6:35,8:11]=True
    assert np.array_equal(result.part_mask,expected)
    assert pixel==(9,20)
    assert not mask[20,20]  # The centroid of both islands must never be chosen.


def test_explicit_contact_point_is_preserved_exactly():
    mask=np.zeros((41,41),bool);mask[10:31,19:22]=True
    seed=(20/40,11/40)
    result,pixel=resolve(mask,'grasp_contact',seed)
    assert result.decision.point==seed and pixel==(20,11)


def test_explicit_part_seed_keeps_existing_contact_selection_behavior():
    mask=np.zeros((41,41),bool);mask[10:31,19:22]=True
    # A part seed selects connectivity, whereas grasp_contact fixes the contact.
    # This change intentionally leaves both explicitly seeded paths unchanged.
    result,pixel=resolve(mask,'grasp_part',(20/40,27/40))
    assert pixel==(20,11)
    assert result.decision.groundings[1].points==[(20/40,27/40)]
