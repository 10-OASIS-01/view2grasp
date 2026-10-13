"""Explicit measured object components survive without contact/map expansion."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.geometry import (extract_target,grasp_candidates,
    has_explicit_object_seeds,part_cloud,pixel_point,unproject)
from view2grasp.grounding import GroundingResolver
from view2grasp.types import AgentDecision,GroundingProposal,Observation


def scene():
    depth=np.full((64,64),np.nan)
    depth[24:33,22:29]=1.
    depth[24:33,34:41]=1.02
    # An unseeded visible component inside the broad box is never membership.
    depth[38:43,27:32]=1.08
    return Observation(7,np.zeros((64,64,3),np.uint8),depth,
        np.array([[700.,0,32],[0,700.,32],[0,0,1]]),np.eye(4),
        np.zeros(7),np.array([.04,-.04]),0.,(0.,55.,.8))


def xy(x,y):return (x/63,y/63)


def decision(*,seeded=True):
    def proposal(kind,role,**kw):
        return GroundingProposal(kind=kind,role=role,frame_id=7,**kw)
    whole=(*xy(20,22),*xy(43,45))
    grounds=[proposal('box','object',box=whole),
        proposal('box','grasp_part',box=whole),
        proposal('points','grasp_contact',points=[xy(25,28)])]
    if seeded:grounds.append(proposal('points','object',points=[xy(37,28)]))
    return AgentDecision(action='grasp',frame_id=7,groundings=grounds)


def measured(obs,mask):
    y,x=np.where(mask)
    return unproject(np.c_[x,y],obs.depth[mask],obs.K,obs.T_wc)


def test_disconnected_body_remains_in_complete_current_target_without_part_expansion():
    obs=scene();resolver=GroundingResolver()
    old=resolver.resolve_grasp(obs,decision(seeded=False),.5)
    resolved=resolver.resolve_grasp(obs,decision(),.5)
    p,target=extract_target(obs,resolved.decision,.5,object_mask=resolved.object_mask)
    _,old_target=extract_target(obs,old.decision,.5,object_mask=old.object_mask)
    body=resolved.object_mask & ~resolved.part_mask
    assert body.sum()==resolved.part_mask.sum()==63
    assert cKDTree(measured(obs,body)).query(measured(obs,resolved.part_mask))[0].min()>.013
    assert np.array_equal(target,measured(obs,resolved.object_mask))
    assert len(target)==126 and np.all(old_target[:,2]==1.)
    assert np.array_equal(resolved.part_mask,old.part_mask)
    assert np.array_equal(p,pixel_point(obs,old.decision.point,.5))
    part=part_cloud(obs,resolved.decision,target,resolved.part_mask)
    assert np.array_equal(part,measured(obs,resolved.part_mask))
    assert np.all(part[:,2]==1.)
    candidates,actual=grasp_candidates(obs,resolved.decision,.5,
        object_mask=resolved.object_mask,part_mask=resolved.part_mask)
    assert candidates and np.array_equal(actual,target)
    assert all(np.array_equal(g.surface_point,p) for g in candidates)
    assert all(g.ray_angle<=15 for g in candidates)
    # Even this body lies within the old 4 cm local-part radius. It must not
    # affect candidates when the already-validated part mask excludes it.
    assert np.linalg.norm(measured(obs,body)-p,axis=1).max()<.04
    contact_only=resolved.decision.model_copy(update={'bbox':old.decision.bbox})
    baseline,_=grasp_candidates(obs,contact_only,.5,
        object_mask=resolved.part_mask,part_mask=resolved.part_mask)
    assert len(candidates)==len(baseline)
    for actual,expected in zip(candidates,baseline):
        assert np.array_equal(actual.position,expected.position)
        assert np.array_equal(actual.rotation,expected.rotation)
        assert actual.width==expected.width


def test_full_resolution_preserves_seeded_island_missed_by_legacy_stride():
    obs=scene();obs.depth[24:33,34:41]=np.nan
    obs.depth[25:30,35]=1.02
    supplied=decision()
    supplied.groundings[-1]=supplied.groundings[-1].model_copy(update={'points':[xy(35,27)]})
    resolved=GroundingResolver().resolve_grasp(obs,supplied,.5)
    _,target=extract_target(obs,resolved.decision,.5,object_mask=resolved.object_mask)
    assert resolved.object_mask.sum()==len(target)==68
    assert (target[:,2]==1.02).sum()==5


def test_historical_surfaces_behind_current_depth_cannot_enter_explicit_target_or_part():
    obs=scene();resolved=GroundingResolver().resolve_grasp(obs,decision(),.5)
    # All project inside selected pixels, are above support, and within 13 mm
    # of current surface: the old projection-only map fusion would admit them.
    uv=np.array([[25,28],[37,28],[26,29]])
    extra=unproject(uv,[1.008,1.028,.994],obs.K,obs.T_wc)
    before=extra.copy();mask_before=resolved.object_mask.copy()
    _,expected=extract_target(obs,resolved.decision,.5,object_mask=resolved.object_mask)
    _,actual=extract_target(obs,resolved.decision,.5,extra,resolved.object_mask)
    assert np.array_equal(actual,expected)
    assert cKDTree(actual).query(extra)[0].min()>.001
    assert np.array_equal(extra,before) and np.array_equal(resolved.object_mask,mask_before)


def test_broad_box_and_object_points_do_not_bypass_resolved_masks():
    obs=scene();supplied=decision()
    with pytest.raises(ValueError,match='resolved object mask'):
        extract_target(obs,supplied,.5)
    resolved=GroundingResolver().resolve_grasp(obs,supplied,.5)
    with pytest.raises(ValueError,match='resolved grasp-part mask'):
        grasp_candidates(obs,resolved.decision,.5,object_mask=resolved.object_mask)


@pytest.mark.parametrize('fault',['shape','depth','support','frame','camera'])
def test_explicit_components_reject_inconsistent_source_evidence(fault):
    obs=scene();resolved=GroundingResolver().resolve_grasp(obs,decision(),.5)
    supplied=resolved.decision;mask=resolved.object_mask.copy()
    if fault=='shape':mask=mask[:-1]
    if fault=='depth':obs.depth[24,34]=np.nan
    if fault=='support':obs.depth[24,34]=.5
    if fault=='frame':supplied=supplied.model_copy(update={'frame_id':6})
    if fault=='camera':supplied=supplied.model_copy(update={'camera_id':'old'})
    with pytest.raises(ValueError):extract_target(obs,supplied,.5,object_mask=mask)


def test_polygon_vertices_and_part_seeds_do_not_enable_component_preservation():
    supplied=decision(seeded=False)
    assert not has_explicit_object_seeds(supplied)
    object_box=supplied.groundings[0]
    for replacement in (
        object_box.model_copy(update={'kind':'polygon','box':None,
            'points':[xy(20,22),xy(43,22),xy(43,45),xy(20,45)]}),
        object_box.model_copy(update={'role':'grasp_part','points':[xy(37,28)]}),
    ):
        assert not has_explicit_object_seeds(supplied.model_copy(
            update={'groundings':[replacement,*supplied.groundings[1:]]}))


def test_no_object_seed_keeps_legacy_stride_map_fusion_and_contact_component():
    obs=scene();resolved=GroundingResolver().resolve_grasp(obs,decision(seeded=False),.5)
    extra=unproject(np.array([[25,28],[37,28]]),[1.008,1.028],obs.K,obs.T_wc)
    p,target=extract_target(obs,resolved.decision,.5,extra,resolved.object_mask)
    # Historical behavior samples the 7x9 contact region at stride 2 and
    # admits the projected nearby map point; the other image island is absent.
    y,x=np.mgrid[24:33:2,22:29:2]
    expected=np.vstack([unproject(np.stack([x,y],-1),obs.depth[y,x],obs.K,obs.T_wc).reshape(-1,3),extra[:1]])
    assert len(target)==21 and np.array_equal(target,expected)
    assert np.array_equal(p,pixel_point(obs,resolved.decision.point,.5))
    assert 'explicit_object_seed_components' not in resolved.report
