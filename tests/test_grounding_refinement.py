"""The same observed crop can receive one actual resolver-error correction."""
from dataclasses import replace
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from test_observation_crop import observation
from view2grasp.grounding import GroundingResolver,GroundingSeedError
from view2grasp.grounding_refinement import refine_grasp_grounding
from view2grasp.observation_crop import crop_grasp_proposal
from view2grasp.types import AgentDecision,GroundingProposal


def grasp(obs,object_box=(.1,.1,.9,.9),part_box=(.65,.3,.8,.7)):
    return AgentDecision(action='grasp',frame_id=obs.frame_id,camera_id=obs.camera_id,groundings=[
        GroundingProposal(kind='box',role=role,frame_id=obs.frame_id,camera_id=obs.camera_id,box=box)
        for role,box in [('object',object_box),('grasp_part',part_box)]])


def setup(replies):
    obs=replace(observation(),base_pose=None,camera_id='activeview')
    coarse=grasp(obs,(.3,.3,.7,.7),(.4,.4,.6,.6));own=obs.depth+.5
    evidence=SimpleNamespace(requests=[],resolutions=[],events=[])
    def decide(image,instruction,history,**kwargs):
        evidence.requests.append((image,copy.deepcopy(kwargs)))
        response=replies[len(evidence.requests)-1]
        if response=='error':raise RuntimeError('API unavailable')
        if response in ('abort','observe'):
            return AgentDecision(action=response,frame_id=image.frame_id,camera_id=image.camera_id)
        return grasp(image,(.1,.1,.4,.8) if response=='bad' else (.1,.1,.9,.9))
    real=GroundingResolver()
    def resolve(source,decision,support,depth):
        assert source is obs and depth is own and support==-10.
        result=real.resolve_grasp(source,decision,support,depth)
        evidence.resolutions.append(result)
        return result
    client=SimpleNamespace(decide=decide);resolver=SimpleNamespace(resolve_grasp=resolve)
    def run():return refine_grasp_grounding(obs,coarse,resolver,client,'task',[], -10.,robot_depth=own,
        event=lambda kind,**data:evidence.events.append({'kind':kind,**copy.deepcopy(data)}))
    return run,evidence,obs,coarse


def test_actual_outside_object_seed_gets_one_same_crop_retry_and_source_resolution():
    run,e,obs,coarse=setup(['bad','good'])
    decision,resolved=run()
    assert len(e.requests)==2 and len(e.resolutions)==1 and resolved is e.resolutions[0]
    assert decision.action=='grasp' and resolved.report['validated']
    first,second=e.requests
    assert first[0] is second[0] and np.array_equal(first[0].rgb,second[0].rgb)
    assert 'grounding_refinement_feedback' not in first[1]
    feedback=second[1]['grounding_refinement_feedback']
    assert feedback['repair_attempt']==feedback['max_geometry_repairs']==1
    assert feedback['previous_crop_decision']['groundings'][0]['box']==(.1,.1,.4,.8)
    rejected=next(v for v in e.events if v['kind']=='grounding_refinement_geometry_rejected')
    source_seed=rejected['diagnostics']['rejected_seeds'][0]
    seed=feedback['geometry_diagnostics']['rejected_seeds'][0]
    x0,y0,_,_=first[1]['grounding_refinement']['bounds_xyxy'];ch,cw=first[0].depth.shape
    expected=np.array(source_seed['pixel'])-[x0,y0]
    assert seed['pixel']==expected.tolist() and np.allclose(seed['normalized'],expected/[cw-1,ch-1])
    assert seed['reasons']==['outside_proposed_region'] and seed['inside_crop'] is True
    assert 'source_pixel' not in seed
    # The bad object was not unioned with its part. Only the independently
    # supplied second semantic object region reached a successful resolution.
    assert resolved.object_mask.sum()>0 and resolved.part_mask.sum()>0
    assert resolved.report['image_context']==first[1]['grounding_refinement']


def test_second_geometry_failure_is_final_without_old_coarse_fallback():
    run,e,_,_=setup(['bad','bad','good'])
    with pytest.raises(GroundingSeedError,match='outside_proposed_region'):run()
    assert len(e.requests)==2 and not e.resolutions
    assert sum(v['kind']=='grounding_refinement_geometry_rejected' for v in e.events)==2


@pytest.mark.parametrize('replies,count', [(['error'],1),(['bad','error'],2)])
def test_api_failure_does_not_start_another_geometry_retry(replies,count):
    run,e,_,_=setup(replies)
    with pytest.raises(RuntimeError,match='API unavailable'):run()
    assert len(e.requests)==count and not e.resolutions


@pytest.mark.parametrize('replies,count', [(['abort'],1),(['bad','abort'],2)])
def test_abort_keeps_its_action_and_never_returns_cached_coarse_grounding(replies,count):
    run,e,_,_=setup(replies);decision,resolved=run()
    assert decision.action=='abort' and resolved is None and len(e.requests)==count and not e.resolutions


def test_annotation_action_is_not_a_grasp_or_a_geometric_retry():
    run,e,_,_=setup(['observe','good'])
    with pytest.raises(ValueError,match='only accepts grasp or abort'):run()
    assert len(e.requests)==1 and not e.resolutions


def test_generic_resolver_failure_has_text_feedback_without_fabricated_pixels():
    obs=replace(observation(),base_pose=None,camera_id='activeview');coarse=grasp(obs)
    calls=[]
    def decide(image,*args,**kwargs):calls.append(kwargs);return grasp(image)
    def reject(*args):raise ValueError('empty valid grounding region')
    with pytest.raises(ValueError,match='empty valid'):
        refine_grasp_grounding(obs,coarse,SimpleNamespace(resolve_grasp=reject),SimpleNamespace(decide=decide),'task',[],0.)
    assert len(calls)==2
    assert calls[1]['grounding_refinement_feedback']['geometry_diagnostics']=={'reason':'empty valid grounding region'}


def test_out_of_crop_diagnostic_is_not_clipped_to_a_new_seed_or_mutated():
    obs=observation();crop=crop_grasp_proposal(obs,grasp(obs,(.3,.3,.7,.7),(.4,.4,.6,.6)))
    diagnostics={'frame_id':obs.frame_id,'camera_id':obs.camera_id,'valid_candidate_pixels':10,
        'rejected_seeds':[{'pixel':[0,0],'normalized':[0.,0.],'reasons':['outside_proposed_region']}],
        'rejected_seed_count':1,'rejected_seeds_truncated':False}
    before=copy.deepcopy(diagnostics);feedback=crop.localize_grounding_error(GroundingSeedError(diagnostics))
    seed=feedback['rejected_seeds'][0]
    assert seed['pixel']==[-crop.bounds_xyxy[0],-crop.bounds_xyxy[1]]
    assert min(seed['normalized'])<0 and seed['inside_crop'] is False and diagnostics==before
    diagnostics['frame_id']=999
    with pytest.raises(ValueError,match='current source-frame'):crop.localize_grounding_error(GroundingSeedError(diagnostics))
