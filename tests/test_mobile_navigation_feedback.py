"""Navigation advice is calibrated image evidence, never a motion certificate."""
import json
from dataclasses import replace

import numpy as np
import pytest

from test_mobile_contracts import navigation_region_scene
from view2grasp.mobile_runner import (NavigationGroundingError,grounded_world,
                                     navigation_observation_context,navigation_floor,support_plane)
from view2grasp.types import GroundingProposal


def observed_floor():
    return np.array([[x,y,0.] for x in np.linspace(-1,1,10) for y in np.linspace(-1,1,10)])


def robot_boundary(floor_z=0.):
    # The actual mobile robot's sampled feet start 2 mm above its standing plane.
    return np.array([[0.,0.,floor_z+.002],[.2,.2,floor_z+.7]])


def test_non_floor_point_rejection_preserves_measured_pixel_and_height():
    obs,_=navigation_region_scene()
    proposal=GroundingProposal(kind='points',role='navigation_goal',frame_id=1,points=[(.5,.5)])
    with pytest.raises(NavigationGroundingError) as caught:
        grounded_world(obs,proposal,floor_z=0.)
    report=caught.value.diagnostics
    assert report['frame_id']==1 and report['camera_id']==obs.camera_id
    assert report['pixel']==[16,16] and report['normalized_pixel']==[.5,.5]
    assert report['world_xyz_m'][2]==pytest.approx(1.)
    assert report['height_above_floor_m']==pytest.approx(1.)
    assert report['floor_z_m']==0 and report['floor_tolerance_m']==.06
    assert str(caught.value)=='grounding: navigation goal is not on observed floor'
    json.dumps(report,allow_nan=False)


def test_region_rejection_reports_current_frame_and_empty_floor_support():
    obs,proposal=navigation_region_scene()
    obs.depth[:]=1.
    with pytest.raises(NavigationGroundingError) as caught:
        grounded_world(obs,proposal,floor_z=0.)
    assert caught.value.diagnostics['floor_pixels_in_region']==0
    assert caught.value.diagnostics['proposal_kind']=='box'


def test_floor_samples_are_bounded_current_frame_measurements_without_mutating_inputs():
    obs,_=navigation_region_scene()
    obs=replace(obs,frame_id=8,camera_id='robot0_nav_floor')
    depth=obs.depth.copy();surface=observed_floor();before=surface.copy()
    context=navigation_observation_context(obs,surface,robot_boundary=robot_boundary())
    assert context['frame_id']==8 and context['camera_id']==obs.camera_id
    assert 1<=len(context['floor_samples'])<=3
    for sample in context['floor_samples']:
        x,y=sample['pixel']
        assert 21<=x<28 and 10<=y<24 and np.isfinite(obs.depth[y,x])
        assert sample['world_xyz_m'][2]==pytest.approx(0.)
        assert np.allclose(sample['normalized_pixel'],[x/31,y/31])
    assert 'do not establish robot clearance' in context['evidence_scope']
    assert 'destination identity' in context['evidence_scope']
    assert np.array_equal(obs.depth,depth,equal_nan=True) and np.array_equal(surface,before)
    json.dumps(context,allow_nan=False)


def test_floor_advice_excludes_robot_returns_and_occluded_background():
    obs,_=navigation_region_scene()
    # One floor-like island is robot self-depth, the other is behind the robot.
    obs.depth[:]=1.;obs.depth[7:11,5:9]=2.;obs.depth[20:24,24:28]=2.
    own=np.full_like(obs.depth,20.);own[7:11,5:9]=2.;own[20:24,24:28]=1.5
    context=navigation_observation_context(obs,observed_floor(),own,robot_boundary=robot_boundary())
    assert context['observed_floor_pixels']==0 and context['floor_samples']==[]


def test_floor_advice_does_not_fabricate_support_or_reuse_prior_samples():
    obs,_=navigation_region_scene()
    first=navigation_observation_context(obs,observed_floor(),robot_boundary=robot_boundary())
    assert first['floor_samples']
    obs=replace(obs,frame_id=2);obs.depth[:]=np.nan
    unavailable=navigation_observation_context(obs,np.empty((0,3)),robot_boundary=robot_boundary())
    assert unavailable['frame_id']==2 and unavailable['floor_samples']==[]
    assert 'floor_evidence_unavailable' in unavailable
    empty=navigation_observation_context(obs,observed_floor(),robot_boundary=robot_boundary())
    assert empty['observed_floor_pixels']==0 and empty['floor_samples']==[]
    json.dumps(empty,allow_nan=False)


def test_countertop_only_is_not_navigation_floor_but_remains_a_grasp_support():
    obs,_=navigation_region_scene();counter=observed_floor()*[.2,.2,1.]+[0,0,.92]
    with pytest.raises(ValueError,match='navigation floor unobserved.*observe robot0_nav_floor'):
        navigation_floor(counter,robot_boundary())
    context=navigation_observation_context(obs,counter,robot_boundary=robot_boundary())
    assert context['floor_samples']==[] and 'floor_z_m' not in context
    assert 'observe robot0_nav_floor' in context['floor_evidence_unavailable']
    z,_=support_plane(counter,np.array([0.,0.,1.1]))
    assert z==pytest.approx(.92)


@pytest.mark.parametrize('floor_z',[0.,1.75])
def test_floor_fit_uses_calibrated_robot_height_not_world_zero_or_largest_plane(floor_z):
    floor=observed_floor()+[0,0,floor_z]
    counter=np.tile(observed_floor()+[0,0,floor_z+.92],(20,1))
    measured=np.vstack([counter,floor,[[0,0,np.nan]]])
    z,selected=navigation_floor(measured,robot_boundary(floor_z))
    assert z==pytest.approx(floor_z) and len(selected)==len(floor)
    assert np.allclose(selected[:,2],floor_z)
    obs,_=navigation_region_scene();obs.T_wc[2,3]+=floor_z
    context=navigation_observation_context(obs,measured,robot_boundary=robot_boundary(floor_z))
    assert context['floor_z_m']==pytest.approx(floor_z) and context['floor_samples']
    assert all(sample['world_xyz_m'][2]==pytest.approx(floor_z) for sample in context['floor_samples'])


def test_robot_prior_does_not_supply_missing_floor_samples_or_expand_height_tolerance():
    with pytest.raises(ValueError,match='navigation floor unobserved'):
        navigation_floor(observed_floor()[:29],robot_boundary())
    with pytest.raises(ValueError,match='navigation floor unobserved'):
        navigation_floor(observed_floor()+[0,0,.063],robot_boundary())
    z,_=navigation_floor(observed_floor()+[0,0,.061],robot_boundary())
    assert z==pytest.approx(.061)
