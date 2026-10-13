"""Region-only axial representatives stay inside the actual measured surface."""
import cv2
import numpy as np
import pytest

from test_grounding_interior import resolve


def wide_junction():
    mask=np.zeros((121,121),bool)
    mask[10:111,58:63]=True;mask[96:111,35:86]=True
    return mask


@pytest.mark.parametrize('turns',[0,1,2,3])
def test_long_region_with_wide_end_uses_axial_middle_not_global_inradius_or_centroid(turns):
    mask=np.rot90(wide_junction(),turns).copy()
    result,(x,y)=resolve(mask)
    report=result.report['point_selection']
    assert report['method']=='observed_axis_central_third_interior'
    assert .45<report['selected_axis_fraction']<.55
    assert np.array_equal(result.part_mask,mask) and np.array_equal(result.object_mask,mask)
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    assert mask[y,x] and distance[y,x]<distance.max()
    assert np.linalg.norm(np.array([y,x])-np.argwhere(mask).mean(axis=0))>10


def test_diagonal_rod_uses_measured_axis_without_world_vertical_assumption():
    mask=np.zeros((121,121),np.uint8);cv2.line(mask,(22,20),(95,98),1,5)
    result,(x,y)=resolve(mask.astype(bool));report=result.report['point_selection']
    assert report['method']=='observed_axis_central_third_interior'
    assert abs(report['principal_axis_world'][0])>.5 and abs(report['principal_axis_world'][1])>.5
    assert mask[y,x] and np.linalg.norm(np.array([x,y])-[59,59])<3


def test_axial_aperture_does_not_fill_hole_or_choose_boundary_nearest_centroid():
    mask=np.zeros((121,121),bool);mask[10:111,45:76]=True;mask[25:96,49:72]=False
    result,(x,y)=resolve(mask)
    assert not mask[60,60] and mask[y,x]
    assert abs(y-60)<=1 and x in (46,47,73,74)
    assert np.array_equal(mask,result.part_mask)
    assert result.report['point_selection']['interior_distance_pixels']>1.5


def test_branched_region_stays_on_observed_pixels_without_claiming_semantic_branch():
    mask=np.zeros((121,121),bool);mask[10:111,58:63]=True;mask[45:50,58:98]=True
    result,(x,y)=resolve(mask);report=result.report['point_selection']
    assert mask[y,x] and 1/3<=report['selected_axis_fraction']<=2/3
    assert np.array_equal(mask,result.part_mask)
    assert report['meaning']=='observed_region_representative_not_hidden_part_midpoint_or_safety'


def test_partial_observed_segment_is_not_completed_to_hidden_whole_bar_midpoint():
    mask=np.zeros((121,121),bool);mask[10:72,58:63]=True
    result,(x,y)=resolve(mask)
    assert x==60 and 39<=y<=42 and mask[y,x]
    assert np.array_equal(result.part_mask,mask)
    assert 'not_hidden_part_midpoint' in result.report['point_selection']['meaning']


def test_compact_block_preserves_original_maximum_interior_tie_policy():
    mask=np.zeros((121,121),bool);mask[40:80,40:80]=True
    result,pixel=resolve(mask)
    assert pixel==(59,59)
    assert result.report['point_selection']['method']=='global_interior_centroid_tie'


@pytest.mark.parametrize('role',['grasp_contact','grasp_part'])
def test_explicit_seeds_bypass_axial_policy_without_moving_proposal(role):
    mask=wide_junction();seed=(60/120,103/120)
    result,(x,y)=resolve(mask,role,seed)
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    if role=='grasp_contact':assert result.decision.point==seed
    else:
        assert (y,x)==np.unravel_index(distance.argmax(),distance.shape)
        assert result.decision.groundings[1].points==[seed]
    assert result.report['point_selection']['basis']=='unchanged_explicit_seed_policy'


def test_disconnected_depth_regions_still_use_only_original_selected_component():
    mask=np.zeros((121,121),bool);mask[10:111,28:33]=True;mask[10:111,88:93]=True
    result,(x,y)=resolve(mask)
    assert x==30 and abs(y-60)<=1
    assert not result.part_mask[:,88:93].any() and not result.part_mask[:,33:88].any()
