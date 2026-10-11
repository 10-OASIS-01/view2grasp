"""Navigation advice is calibrated image evidence, never a motion certificate."""
import json
from dataclasses import replace

import numpy as np
import pytest

from test_mobile_contracts import navigation_region_scene
from view2grasp.mobile_runner import (NavigationGroundingError,grounded_world,
                                     navigation_observation_context)
from view2grasp.types import GroundingProposal


def observed_floor():
    return np.array([[x,y,0.] for x in np.linspace(-1,1,10) for y in np.linspace(-1,1,10)])


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
    context=navigation_observation_context(obs,surface)
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
    context=navigation_observation_context(obs,observed_floor(),own)
    assert context['observed_floor_pixels']==0 and context['floor_samples']==[]


def test_floor_advice_does_not_fabricate_support_or_reuse_prior_samples():
    obs,_=navigation_region_scene()
    first=navigation_observation_context(obs,observed_floor())
    assert first['floor_samples']
    obs=replace(obs,frame_id=2);obs.depth[:]=np.nan
    unavailable=navigation_observation_context(obs,np.empty((0,3)))
    assert unavailable['frame_id']==2 and unavailable['floor_samples']==[]
    assert 'floor_evidence_unavailable' in unavailable
    empty=navigation_observation_context(obs,observed_floor())
    assert empty['observed_floor_pixels']==0 and empty['floor_samples']==[]
    json.dumps(empty,allow_nan=False)
