"""Object framing uses measured source pixels and calibration, not scene truth."""
from dataclasses import replace
import json

import numpy as np
import pytest

from test_geometry import observation
from view2grasp.target_framing import TargetFraming
from view2grasp.types import GroundingProposal


def obs(frame=1):
    return replace(observation(),frame_id=frame,timestamp=float(frame),
        camera_id='robot0_nav_right',base_pose=np.zeros(3))


def proposal(image,**updates):
    fields=dict(kind='points',role='object',target_id='can',frame_id=image.frame_id,
        camera_id=image.camera_id,points=[(.75,.5)])
    fields.update(updates)
    return GroundingProposal(**fields)


def test_exact_object_point_and_source_are_preserved_across_camera_change():
    image=obs();memory=TargetFraming()
    assert memory.refresh(image,[proposal(image)])
    target=memory.target
    assert target.source_pixels==((23,16),)
    assert target.source_normalized_points==((.75,.5),)
    assert np.allclose(target.world_points,[[7/30,0.,1.]])
    assert target.selection=='explicit_object_points'
    later=replace(obs(2),camera_id='robot0_agentview_left')
    assert not memory.refresh(later,[proposal(later,role='look_at',target_id='floor')])
    assert memory.target is target
    context=memory.context(later,{})
    assert context['source_frame_id']==1 and context['current_frame_id']==2
    assert context['source_camera_id']=='robot0_nav_right' and context['source_age_frames']==1
    assert context['source_pixels']==[[23,16]]
    assert context['recommended_camera'] is None
    json.dumps(context,allow_nan=False)
    memory.clear()
    assert memory.target is None and memory.context(later,{}) is None


@pytest.mark.parametrize('role',['look_at','navigation_goal','grasp_part','grasp_contact','place_region'])
def test_non_object_grounding_never_creates_target(role):
    image=obs();memory=TargetFraming()
    assert not memory.refresh(image,[proposal(image,role=role)])
    assert memory.target is None


@pytest.mark.parametrize('depth',[0.,.05,10.,float('nan'),float('inf')])
def test_invalid_explicit_depth_is_rejected_without_replacement(depth):
    image=obs();image.depth[16,23]=depth
    memory=TargetFraming()
    with pytest.raises(ValueError,match='invalid framing point depth'):
        memory.refresh(image,[proposal(image)])
    assert memory.target is None
    assert image.depth[16,22]==1. # Adjacent valid pixels cannot replace the point.


@pytest.mark.parametrize('own,error',[(1.,'robot self return'),(1.002,'robot self return'),
    (.8,'robot occludes'),(0.,'invalid robot depth'),
    (float('nan'),'invalid robot depth'),(float('inf'),'invalid robot depth')])
def test_robot_or_unusable_self_depth_cannot_become_target(own,error):
    image=obs();robot_depth=np.full_like(image.depth,100.);robot_depth[16,23]=own
    memory=TargetFraming()
    with pytest.raises(ValueError,match=error):memory.refresh(image,[proposal(image)],robot_depth)
    assert memory.target is None


def test_robot_depth_is_only_used_for_source_validation():
    image=obs();memory=TargetFraming()
    assert memory.refresh(image,[proposal(image)],np.full_like(image.depth,2.))
    context=memory.context(image,{})
    assert context['robot_depth_checked']
    assert 'current visibility' in context['scope'] and 'No candidate camera image was read' in context['scope']


@pytest.mark.parametrize('updates',[{'frame_id':0},{'camera_id':'robot0_nav_left'}])
def test_stale_source_grounding_is_rejected(updates):
    image=obs();memory=TargetFraming()
    with pytest.raises(ValueError,match='stale'):memory.refresh(image,[proposal(image,**updates)])
    assert memory.target is None


def test_ambiguous_or_changed_target_requires_explicit_clear():
    image=obs();memory=TargetFraming()
    with pytest.raises(ValueError,match='one explicit target_id'):
        memory.refresh(image,[proposal(image),proposal(image,target_id='bowl')])
    memory.refresh(image,[proposal(image)]);old=memory.target
    with pytest.raises(ValueError,match='clear previous'):
        memory.refresh(obs(2),[proposal(obs(2),target_id='bowl')])
    assert memory.target is old
    memory.clear()
    assert memory.refresh(obs(2),[proposal(obs(2),target_id='bowl')])
    assert memory.target.target_id=='bowl'


def test_failed_refresh_keeps_old_evidence_with_its_old_source():
    image=obs();memory=TargetFraming();memory.refresh(image,[proposal(image)])
    later=obs(2);later.depth[16,23]=0.
    with pytest.raises(ValueError):memory.refresh(later,[proposal(later)])
    context=memory.context(later,{})
    assert context['source_frame_id']==1 and context['source_age_frames']==1
    with pytest.raises(ValueError,match='older observation'):
        memory.refresh(obs(0),[proposal(obs(0))])
    with pytest.raises(ValueError,match='predates'):memory.context(obs(0),{})


def test_point_excluded_by_region_or_negative_point_stays_rejected():
    image=obs();memory=TargetFraming()
    with pytest.raises(ValueError,match='outside proposed object region'):
        memory.refresh(image,[proposal(image),proposal(image,kind='box',points=[],box=(.1,.1,.4,.9))])
    with pytest.raises(ValueError,match='negative point'):
        memory.refresh(image,[proposal(image,negative_points=[(.75,.5)])])
    assert memory.target is None


def test_region_uses_real_interior_depth_without_filling_polygon_hole():
    image=obs();memory=TargetFraming()
    region=proposal(image,kind='polygon',points=[(.2,.2),(.8,.2),(.8,.8),(.2,.8)],
        holes=[[(.4,.4),(.6,.4),(.6,.6),(.4,.6)]])
    assert memory.refresh(image,[region],np.full_like(image.depth,2.))
    target=memory.target;x,y=target.source_pixels[0]
    assert target.selection=='derived_object_region_interior'
    assert 6<=x<=25 and 6<=y<=25 and not (12<=x<=19 and 12<=y<=19)
    assert np.allclose(target.world_points[0],[(x-16)/30,(y-16)/30,1.])
    assert target.source_normalized_points==((x/31,y/31),)


def test_no_region_point_is_invented_when_all_returns_are_robot():
    image=obs();memory=TargetFraming()
    region=proposal(image,kind='box',points=[],box=(.2,.2,.8,.8))
    with pytest.raises(ValueError,match='empty valid grounding region'):
        memory.refresh(image,[region],image.depth.copy())
    assert memory.target is None


def test_region_behind_known_robot_is_not_used_as_target():
    image=obs();memory=TargetFraming()
    region=proposal(image,kind='box',points=[],box=(.2,.2,.8,.8))
    with pytest.raises(ValueError,match='empty valid grounding region'):
        memory.refresh(image,[region],np.full_like(image.depth,.8))
    assert memory.target is None


def test_known_camera_translation_predicts_centering_without_candidate_images():
    image=obs();memory=TargetFraming();memory.refresh(image,[proposal(image)])
    # Source point is x=7/30,z=1. A parallel camera shifted +.75m sees
    # u=.5px; a camera shifted by the point x sees u=16px. These are analytic
    # pinhole expectations, independent of the helper's project implementation.
    clipped=np.eye(4);clipped[0,3]=.75
    centered=np.eye(4);centered[0,3]=7/30
    current=replace(obs(2),camera_id='robot0_agentview_left',T_wc=clipped)
    report=memory.context(current,{'robot0_agentview_right':(image.K,centered)})
    cameras={p['camera_id']:p for p in report['camera_projections']}
    assert np.allclose(cameras[current.camera_id]['projected_center_normalized'],[.5/31,16/31])
    assert np.allclose(cameras['robot0_agentview_right']['projected_center_normalized'],[16/31,16/31])
    assert report['recommended_camera']=='robot0_agentview_right'
    assert report['source_pixels']==[[23,16]]
    # Advice cannot consume candidate RGB/depth: only K/T matrices are supplied.
    empty_depth=replace(current,rgb=np.full_like(current.rgb,255),depth=np.full_like(current.depth,np.nan))
    assert memory.context(empty_depth,{'robot0_agentview_right':(image.K,centered)})==report


def test_out_of_frame_and_behind_camera_are_not_visibility_claims():
    image=obs();memory=TargetFraming();memory.refresh(image,[proposal(image)])
    outside=np.eye(4);outside[0,3]=2.
    behind=np.eye(4);behind[:3,:3]=np.diag([-1.,1.,-1.])
    report=memory.context(image,{'robot0_eye_in_hand':(image.K,outside),
        'robot0_nav_rear':(image.K,behind)})
    cameras={p['camera_id']:p for p in report['camera_projections']}
    assert cameras['robot0_eye_in_hand']['in_frame_points']==0
    assert cameras['robot0_eye_in_hand']['minimum_border_margin']<0
    assert cameras['robot0_nav_rear']['projected_points_normalized']==[None]
    assert cameras['robot0_nav_rear']['minimum_border_margin'] is None
    assert report['recommended_camera'] is None
    json.dumps(report,allow_nan=False)


def test_current_image_calibration_wins_over_stale_entry():
    image=obs();memory=TargetFraming();memory.refresh(image,[proposal(image)])
    stale=np.eye(4);stale[0,3]=100.
    context=memory.context(image,{image.camera_id:(image.K,stale)})
    assert context['camera_projections'][0]['in_frame_points']==1


def test_invalid_camera_calibration_and_unknown_camera_fail_closed():
    image=obs();memory=TargetFraming();memory.refresh(image,[proposal(image)])
    with pytest.raises(ValueError,match='unknown mounted'):
        memory.context(image,{'teleport_camera':(image.K,image.T_wc)})
    with pytest.raises(ValueError,match='invalid framing camera calibration'):
        memory.context(image,{'robot0_eye_in_hand':(np.zeros((3,3)),image.T_wc)})
    with pytest.raises(ValueError,match='robot depth shape mismatch'):
        memory.refresh(image,[proposal(image)],np.zeros((1,1)))
