from dataclasses import replace
import numpy as np
import pytest
from view2grasp.types import GroundingProposal, AgentDecision
from view2grasp.grounding import (proposal_mask,mask_rle,restore_crop,GroundingResolver,lift_ground_path)
from view2grasp.observation_policy import tabletop_view
from test_geometry import observation


def proposal(**kwargs):
    return GroundingProposal(frame_id=1,kind='polygon',role='grasp_part',
        points=[(.1,.1),(.9,.1),(.9,.9),(.1,.9)],**kwargs)


def test_polygon_hole_and_mask_rle_roundtrip():
    obs=observation()
    p=proposal(holes=[[ (.4,.4),(.6,.4),(.6,.6),(.4,.6) ]])
    mask=proposal_mask(p,obs)
    h,w=mask.shape
    assert not mask[h//2,w//2] and mask[h//4,w//4]
    rle=GroundingProposal(kind='mask',role='grasp_part',frame_id=1,
        mask_size=mask.shape,mask_rle=mask_rle(mask))
    assert np.array_equal(proposal_mask(rle,obs),mask)


def test_region_only_grasp_does_not_require_a_point_at_interface():
    obs=observation()
    p=proposal()
    d=AgentDecision(action='grasp',frame_id=1,groundings=[p])
    assert d.point is None
    result=GroundingResolver().resolve_grasp(obs,d,.5)
    assert result.decision.point is not None
    assert result.decision.groundings==[p] and d.point is None
    assert result.report['validated']


def test_mask_seed_on_hole_is_rejected():
    obs=observation()
    p=proposal(holes=[[ (.4,.4),(.6,.4),(.6,.6),(.4,.6) ]])
    seed=GroundingProposal(kind='points',role='grasp_contact',points=[(.5,.5)],frame_id=1)
    d=AgentDecision(action='grasp',frame_id=1,groundings=[p,seed])
    with pytest.raises(ValueError,match='hole'):
        GroundingResolver().resolve_grasp(obs,d,.5)


def test_coordinate_validation_and_stale_frame():
    with pytest.raises(ValueError):
        GroundingProposal(kind='points',role='object',frame_id=1,points=[(1.1,.5)])
    with pytest.raises(ValueError,match='frame'):
        proposal_mask(proposal(),replace(observation(),frame_id=2))
    with pytest.raises(ValueError,match='camera'):
        proposal_mask(proposal(),replace(observation(),camera_id='wrist'))
    assert np.allclose(restore_crop([[0,0],[1,1]],(10,20,30,40),(100,100)),np.array([[10,20],[30,40]])/99)


def test_path_semantics_and_depth_validation():
    obs=observation()
    p=GroundingProposal(kind='path',role='ground_path',frame_id=1,points=[(.3,.3),(.6,.6)])
    world=lift_ground_path(obs,p,1.)
    assert np.allclose(world[:,2],1.)
    with pytest.raises(ValueError,match='only ground_path'):
        lift_ground_path(obs,p.model_copy(update={'role':'end_effector_path'}),1.)
    obs.depth[:]=np.nan
    with pytest.raises(ValueError,match='depth'):
        lift_ground_path(obs,p,1.)


def test_sam_unavailable_is_explicit_and_no_free_space_is_created():
    result=GroundingResolver(sam_unavailable_reason='not installed').resolve_grasp(
        observation(),AgentDecision(action='grasp',frame_id=1,groundings=[proposal()]),.5)
    assert result.report['source']=='rgbd' and result.report['sam_calls']==0
    assert result.report['sam_unavailable']=='not installed'


def test_observation_policy_applies_to_every_phase():
    rng=np.random.default_rng(3)
    for phase in ('initial','recovery','postgrasp'):
        assert tabletop_view('fixed',(0,80,1.3),rng,(-90,0,.7))==(0,80,1.3)
        assert tabletop_view('active',(0,80,1.3),rng,(-90,0,.7))==(-90,0,.7)
    views=np.array([tabletop_view('random',(0,80,1.3),rng) for _ in range(100)])
    assert views[:,1].min()<0 and views[:,2].min()<.75


def test_mask_size_and_candidate_scope():
    with pytest.raises(ValueError,match='RLE'):
        GroundingProposal(kind='mask',role='object',frame_id=1,mask_size=(4,4),mask_rle=[2,3])
    p=GroundingProposal(kind='candidate',role='object',frame_id=1,candidate_id='unseen')
    with pytest.raises(ValueError,match='not provided'):
        proposal_mask(p,observation(),{})


def test_part_identity_mismatch_gets_schema_repair_not_a_new_observation():
    a=proposal(target_id='mug')
    b=GroundingProposal(kind='box',role='object',box=(.1,.1,.9,.9),frame_id=1,target_id='handle')
    with pytest.raises(ValueError,match='ONE target_id'):
        AgentDecision(action='grasp',frame_id=1,groundings=[a,b])


def test_tracking_preserves_new_side_faces_and_disconnected_visible_islands():
    from view2grasp.grounding import track_surface
    from view2grasp.geometry import unproject
    obs=observation()
    h,w=obs.depth.shape
    obs.depth[:]=2
    mask=np.zeros((h,w),bool)
    mask[h//2-1:h//2+2,w//2-2:w//2-1]=True
    mask[h//2-1:h//2+2,w//2+1:w//2+2]=True
    obs.depth[mask]=.98
    # Previous observation exposed only the face at z=1. The new surfaces are
    # lower, but still above the actual observed support at z=.95.
    reference=unproject(np.array([[w//2-2,h//2],[w//2+1,h//2]]),np.ones(2),obs.K,obs.T_wc)
    found=track_surface(obs,reference,segment=lambda _:mask,support_z=.95)
    assert found is not None and len(found)==mask.sum()
    assert np.allclose(found[:,2],.98)
    assert track_surface(obs,reference,segment=lambda _:mask,support_z=.99) is None


def test_failed_grounding_time_is_included():
    resolver=GroundingResolver();obs=observation();obs.depth[:]=0
    with pytest.raises(ValueError):
        resolver.resolve_grasp(obs,AgentDecision(action='grasp',frame_id=1,groundings=[proposal()]),.5)
    assert resolver.seconds>0


def test_end_effector_path_is_ranking_only_not_background_depth():
    from view2grasp.geometry import grasp_candidates
    obs=observation();obs.K[0,0]=obs.K[1,1]=300
    decision=AgentDecision(action='grasp',frame_id=1,point=(.5,.5),bbox=(.3,.3,.7,.7),part_bbox=(.3,.3,.7,.7))
    original,_=grasp_candidates(obs,decision,.5)
    hint=GroundingProposal(kind='path',role='end_effector_path',frame_id=1,points=[(.1,.2),(.2,.2)])
    obs.depth[:5,:5]=np.nan
    altered,_=grasp_candidates(obs,decision.model_copy(update={'groundings':[hint]}),.5)
    signature=lambda items:sorted(tuple(np.round(np.r_[g.position,g.rotation.ravel()],8)) for g in items)
    assert signature(original)==signature(altered)
    assert all(g.ray_angle<=15 for g in altered)
