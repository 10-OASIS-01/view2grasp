"""A refinement crop only changes coordinates; it never approves a grasp."""
from dataclasses import replace
import numpy as np
import pytest
from test_observation_crop import observation
from view2grasp.geometry import unproject
from view2grasp.grounding import mask_rle,pixels,proposal_mask
from view2grasp.observation_crop import crop_grasp_proposal
from view2grasp.types import AgentDecision,GroundingProposal


def proposal(obs,role='object',kind='polygon',**kwargs):
    shape={'points':[(.35,.3),(.65,.3),(.65,.7),(.35,.7)],
        'holes':[[[.45,.4],[.55,.4],[.55,.6],[.45,.6]]]} if kind=='polygon' else {'box':(.36,.32,.44,.68)}
    shape.update(kwargs)
    return GroundingProposal(kind=kind,role=role,frame_id=obs.frame_id,camera_id=obs.camera_id,**shape)


def decision(obs):
    return AgentDecision(action='grasp',frame_id=obs.frame_id,camera_id=obs.camera_id,
        groundings=[proposal(obs),proposal(obs,'grasp_part','box')])


def test_proposal_bounds_preserve_disconnected_body_and_hole_without_resolving_contact():
    obs=observation();coarse=decision(obs)
    crop=crop_grasp_proposal(obs,coarse,robot_depth=obs.depth+.5)
    x0,y0,x1,y1=crop.bounds_xyxy
    all_pixels=pixels(coarse.groundings[0].points,obs.depth.shape)
    assert np.all(all_pixels>=[x0,y0]) and np.all(all_pixels<[x1,y1])
    assert crop.observation.rgb.shape[:2]==(128,128)
    assert np.array_equal(crop.observation.rgb,obs.rgb[y0:y1,x0:x1])
    assert crop.provenance()['reference_kind']=='current_vlm_object_pixel_region'
    assert crop.provenance()['surface_margin_m']==0 and not crop.provenance()['new_observation']
    # The aperture remains raw observed background, not an invented filled mask.
    u,v=pixels([[.5,.5]],obs.depth.shape)[0]
    assert not proposal_mask(coarse.groundings[0],obs)[v,u]
    assert np.array_equal(crop.observation.rgb[v-y0,u-x0],obs.rgb[v,u])


def test_explicit_grasp_restore_preserves_semantics_pixels_and_world_contact():
    obs=observation();crop=crop_grasp_proposal(obs,decision(obs))
    refined=decision(crop.observation).model_copy(update={'point':(.4,.5),'reason':'fresh crop material'})
    restored=crop.restore_grasp_decision(refined)
    assert restored.action=='grasp' and restored.reason==refined.reason
    assert restored.frame_id==obs.frame_id and restored.camera_id==obs.camera_id
    assert restored.point==tuple(crop.restore_points([refined.point])[0])
    uv=pixels([refined.point],crop.observation.depth.shape)
    z=crop.observation.depth[uv[:,1],uv[:,0]]
    assert np.allclose(unproject(uv,z,crop.observation.K,crop.observation.T_wc),
        unproject(pixels([restored.point],obs.depth.shape),z,obs.K,obs.T_wc))
    for old,new in zip(refined.groundings,restored.groundings):
        full=proposal_mask(new,obs);local=proposal_mask(old,crop.observation)
        x0,y0,x1,y1=crop.bounds_xyxy
        expected=np.zeros(obs.depth.shape,dtype=bool);expected[y0:y1,x0:x1]=local
        assert np.array_equal(full,expected)
    with pytest.raises(ValueError,match='only accepts observe'):
        crop.restore_decision(refined)


def test_refinement_mask_restores_exact_rle_holes():
    obs=observation();crop=crop_grasp_proposal(obs,decision(obs));shape=crop.observation.depth.shape
    object_mask=np.zeros(shape,dtype=bool);object_mask[10:-10,12:-12]=True;object_mask[30:50,30:50]=False
    part_mask=np.zeros(shape,dtype=bool);part_mask[12:26,15:26]=True
    refined=AgentDecision(action='grasp',frame_id=obs.frame_id,camera_id=obs.camera_id,
        groundings=[proposal(crop.observation,role,'mask',points=[],box=None,mask_size=shape,mask_rle=mask_rle(mask))
            for role,mask in [('object',object_mask),('grasp_part',part_mask)]])
    restored=crop.restore_grasp_decision(refined)
    x0,y0,x1,y1=crop.bounds_xyxy
    for g,mask in zip(restored.groundings,[object_mask,part_mask]):
        full=proposal_mask(g,obs)
        assert np.array_equal(full[y0:y1,x0:x1],mask) and full.sum()==mask.sum()


@pytest.mark.parametrize('action',['observe','place','navigate','finish'])
def test_annotation_or_camera_action_is_never_promoted_to_grasp(action):
    obs=observation();crop=crop_grasp_proposal(obs,decision(obs))
    reply=AgentDecision(action=action,frame_id=obs.frame_id,camera_id=obs.camera_id)
    with pytest.raises(ValueError,match='only accepts grasp or abort'):crop.restore_grasp_decision(reply)


def test_abort_remains_abort_and_stale_reply_remains_invalid():
    obs=observation();crop=crop_grasp_proposal(obs,decision(obs))
    reply=AgentDecision(action='abort',frame_id=obs.frame_id,camera_id=obs.camera_id,reason='uncertain')
    assert crop.restore_grasp_decision(reply).model_dump()==reply.model_dump()
    with pytest.raises(ValueError,match='source frame and camera'):
        crop.restore_grasp_decision(reply.model_copy(update={'frame_id':999}))


def test_refinement_cannot_reuse_only_points_or_legacy_contact():
    obs=observation();crop=crop_grasp_proposal(obs,decision(obs))
    points=AgentDecision(action='grasp',frame_id=obs.frame_id,camera_id=obs.camera_id,
        groundings=[GroundingProposal(kind='points',role=role,frame_id=obs.frame_id,
            camera_id=obs.camera_id,points=[(.5,.5)]) for role in ('object','grasp_part')])
    with pytest.raises(ValueError,match='fresh explicit'):crop.restore_grasp_decision(points)
    legacy=AgentDecision(action='grasp',frame_id=obs.frame_id,camera_id=obs.camera_id,
        point=(.5,.5),bbox=(.1,.1,.9,.9),part_bbox=(.2,.2,.8,.8))
    with pytest.raises(ValueError,match='fresh explicit'):crop.restore_grasp_decision(legacy)


@pytest.mark.parametrize('case',['missing','legacy','robot','empty','stale'])
def test_invalid_crop_does_not_substitute_a_new_object_or_whole_image(case):
    obs=observation();coarse=decision(obs);own=None
    if case=='missing':coarse.groundings=coarse.groundings[1:]
    elif case=='legacy':coarse.groundings[0].source='legacy'
    elif case=='robot':own=obs.depth.copy()
    elif case=='empty':obs=replace(obs,depth=np.full_like(obs.depth,np.nan))
    else:coarse=coarse.model_copy(update={'frame_id':99})
    with pytest.raises(ValueError):crop_grasp_proposal(obs,coarse,robot_depth=own)
