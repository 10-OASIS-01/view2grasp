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


def setup(replies,history=None):
    obs=replace(observation(),base_pose=None,camera_id='activeview')
    coarse=grasp(obs,(.3,.3,.7,.7),(.4,.4,.6,.6));own=obs.depth+.5
    evidence=SimpleNamespace(requests=[],resolutions=[],events=[],histories=[])
    def decide(image,instruction,current_history,**kwargs):
        evidence.requests.append((image,copy.deepcopy(kwargs)))
        evidence.histories.append(copy.deepcopy(current_history))
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
    def run():return refine_grasp_grounding(obs,coarse,resolver,client,'task',history or [], -10.,robot_depth=own,
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


def fallback_setup(replies,history=None,*,one_possible_pixel=False,support=.8,fault=None):
    """Wrong finite-background crop; another observed full-image surface exists."""
    obs=replace(observation(),base_pose=None,camera_id='activeview',T_wc=np.eye(4))
    depth=np.full_like(obs.depth,4.)
    depth[60:220,200:300]=1.
    depth[70:100,:128]=support  # Support is not object material.
    depth[100:110,:128]=np.nan
    obs=replace(obs,depth=depth)
    coarse=grasp(obs,(.04,.3,.18,.65),(.08,.4,.15,.55))
    initial_crop=crop_grasp_proposal(obs,coarse)
    if one_possible_pixel:
        x0,y0,_,_=initial_crop.bounds_xyxy
        depth[y0+2,x0+2]=1.  # Outside the proposal but inside its image context.
    own=depth+.5
    if fault=='calibration':
        K=obs.K.copy();K[0,0]=-1.;obs=replace(obs,K=K)
    elif fault=='stale':coarse=coarse.model_copy(update={'frame_id':999})
    elif fault=='all_self':own=depth.copy()
    evidence=SimpleNamespace(requests=[],histories=[],resolutions=[],events=[])
    def decide(image,instruction,current_history,**kwargs):
        evidence.requests.append((image,copy.deepcopy(kwargs)))
        evidence.histories.append(copy.deepcopy(current_history))
        response=replies[len(evidence.requests)-1]
        if response=='error':raise RuntimeError('API unavailable')
        if response in ('abort','observe'):
            return AgentDecision(action=response,frame_id=image.frame_id,camera_id=image.camera_id)
        result=grasp(image,(.68,.3,.7,.8) if response=='bad' else (.68,.3,.9,.8),(.72,.4,.8,.6))
        return result.model_copy(update={'frame_id':999}) if response=='stale' else result
    real=GroundingResolver()
    def resolve(source,decision,support_z,robot_depth):
        assert source is obs and robot_depth is own and support_z==support
        result=real.resolve_grasp(source,decision,support_z,robot_depth)
        evidence.resolutions.append(result)
        return result
    def run():return refine_grasp_grounding(obs,coarse,SimpleNamespace(resolve_grasp=resolve),
        SimpleNamespace(decide=decide),'Grasp the requested visible part',history or [],support,
        robot_depth=own,event=lambda kind,**data:evidence.events.append({'kind':kind,**copy.deepcopy(data)}))
    return run,evidence,obs,initial_crop


def test_zero_possible_contact_crop_uses_only_same_full_frame_and_original_resolver():
    run,e,obs,proposal_crop=fallback_setup(['good'])
    decision,resolved=run()
    assert len(e.requests)==len(e.resolutions)==1
    image,options=e.requests[0]
    assert image is obs and np.array_equal(image.rgb,obs.rgb)
    context=options['grounding_refinement']
    assert context==options['derived_image_context']
    assert context['kind']=='full_acquired_observation'
    assert context['source_shape_hw']==context['crop_shape_hw']==list(obs.depth.shape)
    assert context['bounds_xyxy']==[0,0,obs.depth.shape[1],obs.depth.shape[0]]
    assert context['new_observation'] is False and context['resized'] is False
    assert np.array_equal(image.K,obs.K) and np.array_equal(image.T_wc,obs.T_wc)
    request=e.events[0]
    assert request['crop_depth_screen']['possible_contact_pixels']==0
    assert request['proposal_image_context']['bounds_xyxy']==list(proposal_crop.bounds_xyxy)
    assert request['crop_depth_screen']['full_frame_fallback']
    assert decision.action=='grasp' and resolved.report['image_context']==context
    assert resolved.report['validated'] and resolved is e.resolutions[0]
    # The independently returned full-image region, not the old coarse crop,
    # supplies the measured material and normalized source contact.
    assert resolved.decision.point[0]>.7


def test_one_possible_pixel_in_crop_context_does_not_trigger_full_frame_fallback():
    run,e,obs,proposal_crop=fallback_setup(['abort'],one_possible_pixel=True)
    decision,resolved=run()
    assert decision.action=='abort' and resolved is None
    assert e.requests[0][0].depth.shape==proposal_crop.observation.depth.shape
    assert e.requests[0][0] is not obs
    request=e.events[0]
    assert request['crop_depth_screen']['possible_contact_pixels']==1
    assert request['crop_depth_screen']['full_frame_fallback'] is False


@pytest.mark.parametrize('depth,support,possible', [
    (.05,-1.,0),(.050001,-1.,1),(3.,.8,0),(2.999999,.8,1),
    (.806,.8,0),(.8060001,.8,1),(np.nan,.8,0),(np.inf,.8,0)])
def test_crop_screen_uses_only_strict_existing_depth_and_loose_support_bounds(depth,support,possible):
    from view2grasp.grounding_refinement import _possible_contact_pixels
    obs=replace(observation((2,2)),depth=np.full((2,2),np.nan),T_wc=np.eye(4))
    obs.depth[0,0]=depth
    assert _possible_contact_pixels(obs,support)==possible


def test_crop_screen_uses_calibrated_world_height_not_camera_depth():
    from view2grasp.grounding_refinement import _possible_contact_pixels
    obs=replace(observation((2,2)),depth=np.full((2,2),1.),T_wc=np.eye(4))
    obs.T_wc[2,3]=-.3
    assert _possible_contact_pixels(obs,.8)==0  # World z=.7 despite depth=1.
    obs.T_wc[2,3]=.4;obs.depth[:]=.5
    assert _possible_contact_pixels(obs,.8)==4  # World z=.9 despite depth=.5.


@pytest.mark.parametrize('replies,outcome', [
    (['bad','good'],'success'),(['bad','bad','good'],'geometry'),
    (['abort'],'abort'),(['bad','abort'],'abort'),(['error'],'api'),
    (['bad','error'],'api'),(['stale'],'stale'),(['observe'],'action')])
def test_full_frame_fallback_keeps_abort_errors_and_two_answer_limit(replies,outcome):
    run,e,obs,_=fallback_setup(replies)
    if outcome=='geometry':
        with pytest.raises(GroundingSeedError,match='outside_proposed_region'):run()
    elif outcome=='api':
        with pytest.raises(RuntimeError,match='API unavailable'):run()
    elif outcome in ('stale','action'):
        with pytest.raises(ValueError,match='source frame and camera' if outcome=='stale' else 'only accepts grasp or abort'):run()
    else:
        decision,resolved=run()
        assert decision.action==('abort' if outcome=='abort' else 'grasp')
        assert (resolved is None)==(outcome=='abort')
    assert len(e.requests)==min(2,len(replies))
    assert all(image is obs for image,_ in e.requests)
    if len(e.requests)==2:
        feedback=e.requests[1][1]['grounding_refinement_feedback']
        assert feedback['repair_attempt']==feedback['max_geometry_repairs']==1
        assert feedback['image_shape_hw']==list(obs.depth.shape)
        sample=feedback['geometry_diagnostics']['rejected_seeds'][0]
        source=next(item for item in e.events if item['kind']=='grounding_refinement_geometry_rejected')
        assert sample['pixel']==source['diagnostics']['rejected_seeds'][0]['pixel']


@pytest.mark.parametrize('fault,match', [('calibration','calibrated'),('stale','current-frame'),
    ('all_self','no measured non-robot')])
def test_invalid_source_or_crop_never_gets_generic_full_frame_error_fallback(fault,match):
    run,e,_,_=fallback_setup(['good'],fault=fault)
    with pytest.raises(ValueError,match=match):run()
    assert not e.requests and not e.resolutions


def test_nonfinite_support_cannot_masquerade_as_an_empty_crop():
    run,e,_,_=fallback_setup(['good'],support=np.nan)
    with pytest.raises(ValueError,match='finite observed support'):run()
    assert not e.requests


@pytest.mark.parametrize('full_frame',[False,True])
def test_refinement_sends_only_current_transaction_and_keeps_old_coordinates_local(full_frame):
    history=[{'frame_id':1,'feedback':'old coordinate sentinel (0.123,0.456)',
        'old_decision':{'point':[.123,.456]}}]
    before=copy.deepcopy(history)
    run,e,obs,_=(fallback_setup(['bad','good'],history) if full_frame else setup(['bad','good'],history))
    run()
    assert history==before and e.events[0]['planning_history']==before
    assert len(e.histories)==2 and e.histories[0]==e.histories[1]
    for sent in e.histories:
        assert len(sent)==1
        transaction=sent[0]['grounding_refinement_transaction']
        assert transaction['frame_id']==obs.frame_id and transaction['camera_id']==obs.camera_id
        assert transaction['full_frame_fallback'] is full_frame
        assert 'old coordinate sentinel' not in str(sent)
        assert 'old_decision' not in str(sent) and 'coarse_decision' not in str(sent)
