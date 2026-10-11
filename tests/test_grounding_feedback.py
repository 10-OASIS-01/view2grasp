"""Explicit bad contacts remain errors; feedback identifies observed exclusions."""
import json
import numpy as np
import pytest
from test_geometry import observation
from view2grasp.grounding import GroundingSeedError,depth_component,proposal_mask
from view2grasp.types import GroundingProposal


@pytest.mark.parametrize('reason,remaining',[
    ('robot_self_return',1023),('outside_proposed_region',1023),
    ('support_plane',1023),('negative_point',999),
])
def test_bad_seed_reports_filter_reason_without_using_remaining_region(reason,remaining):
    obs=observation();allowed=np.ones(obs.depth.shape,dtype=bool);kwargs={}
    if reason=='robot_self_return':
        robot_depth=np.full_like(obs.depth,10.);robot_depth[16,16]=1.
        kwargs['robot_depth']=robot_depth
    elif reason=='outside_proposed_region':allowed[16,16]=False
    elif reason=='support_plane':
        obs.depth[16,16]=.8;kwargs['support_z']=.8
    elif reason=='negative_point':kwargs['negative_points']=[(.5,.5)]
    original=allowed.copy();seeds=[(.5,.5)]
    with pytest.raises(GroundingSeedError) as caught:
        depth_component(obs,allowed,seeds,**kwargs)
    error=caught.value;details=error.diagnostics
    assert details['valid_candidate_pixels']==remaining
    assert details['frame_id']==obs.frame_id and details['camera_id']==obs.camera_id
    assert details['rejected_seeds']==[{'seed_index':0,'normalized':[.5,.5],
        'pixel':[16,16],'reasons':[reason]}]
    assert details['rejected_seed_count']==1 and not details['rejected_seeds_truncated']
    assert json.dumps(details,separators=(',',':')) in str(error)
    assert 'omit optional contact points while still supplying valid object/part regions' in str(error)
    assert 'not semantic correctness or graspability evidence' in str(error)
    assert np.array_equal(allowed,original) and seeds==[(.5,.5)]


@pytest.mark.parametrize('depth',[np.nan,np.inf,0.,.05,10.])
def test_invalid_depth_is_not_mislabeled_as_support_or_robot_contact(depth):
    obs=observation();obs.depth[16,16]=depth
    with np.errstate(invalid='ignore'),pytest.raises(GroundingSeedError) as caught:
        depth_component(obs,np.ones(obs.depth.shape,bool),[(.5,.5)],support_z=.8,
            robot_depth=np.ones_like(obs.depth)*2)
    assert caught.value.diagnostics['rejected_seeds'][0]['reasons']==['invalid_depth']
    assert caught.value.diagnostics['valid_candidate_pixels']==1023


def test_polygon_hole_reports_mask_exclusion_without_guessing_its_semantics():
    obs=observation()
    polygon=GroundingProposal(kind='polygon',role='grasp_part',frame_id=1,
        points=[(.1,.1),(.9,.1),(.9,.9),(.1,.9)],
        holes=[[ (.4,.4),(.6,.4),(.6,.6),(.4,.6) ]])
    allowed=proposal_mask(polygon,obs)
    with pytest.raises(GroundingSeedError) as caught:
        depth_component(obs,allowed,[(.5,.5)])
    details=caught.value.diagnostics
    assert details['rejected_seeds'][0]['reasons']==['outside_proposed_region']
    assert details['valid_candidate_pixels']==int(allowed.sum())
    assert 'hole' not in str(caught.value)


def test_overlapping_exclusions_and_one_bad_seed_reject_the_whole_proposal():
    obs=observation();allowed=np.ones(obs.depth.shape,bool);allowed[16,16]=False
    robot_depth=np.full_like(obs.depth,10.);robot_depth[16,16]=1.
    with pytest.raises(GroundingSeedError) as caught:
        depth_component(obs,allowed,[(.25,.25),(.5,.5)],negative_points=[(.5,.5)],
            robot_depth=robot_depth)
    details=caught.value.diagnostics
    assert details['valid_candidate_pixels']==999
    assert details['rejected_seed_count']==1
    assert details['rejected_seeds'][0]['seed_index']==1
    assert details['rejected_seeds'][0]['reasons']==[
        'outside_proposed_region','robot_self_return','negative_point']


def test_empty_filtered_region_does_not_suggest_omitting_the_seed():
    obs=observation();obs.depth[:]=0
    with pytest.raises(GroundingSeedError) as caught:
        depth_component(obs,np.ones(obs.depth.shape,bool),[(.5,.5)])
    assert caught.value.diagnostics['valid_candidate_pixels']==0
    assert 'omit optional contact' not in str(caught.value)
    assert 'no filtered pixels remain' in str(caught.value)


def test_error_payload_is_bounded_without_ignoring_additional_bad_seeds():
    obs=observation();allowed=np.ones(obs.depth.shape,bool);allowed[16,16]=False
    with pytest.raises(GroundingSeedError) as caught:
        depth_component(obs,allowed,[(.5,.5)]*17)
    details=caught.value.diagnostics
    assert details['rejected_seed_count']==17 and len(details['rejected_seeds'])==16
    assert details['rejected_seeds_truncated']


def test_region_only_and_valid_seed_keep_existing_connected_depth_selection():
    obs=observation();allowed=np.ones(obs.depth.shape,bool)
    robot_depth=np.full_like(obs.depth,10.);robot_depth[16,16]=1.
    expected=allowed.copy();expected[16,16]=False
    for seeds in ((),[(.25,.25)]):
        actual=depth_component(obs,allowed,seeds,support_z=.8,robot_depth=robot_depth)
        assert np.array_equal(actual,expected)
