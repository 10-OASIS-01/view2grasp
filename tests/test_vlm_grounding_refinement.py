"""Same acquired crop refines a new grasp; it is never an observe annotation."""
import base64
import copy
from dataclasses import replace
import hashlib
import json

import httpx
import numpy as np
from PIL import Image
import pytest

from test_geometry import observation
from test_vlm import isolated_api_settings, retry_sleeps
from view2grasp.vlm import VLM


def provenance(obs):
    return {'kind':'crop_of_acquired_observation','source_frame_id':obs.frame_id,
        'source_camera_id':obs.camera_id,'source_shape_hw':[100,100],
        'crop_shape_hw':list(obs.depth.shape),'bounds_xyxy':[20,30,52,62],
        'coordinates':'normalized_to_crop_pixel_centers','new_observation':False,'resized':False}


def grasp(obs):
    return {'action':'grasp','frame_id':obs.frame_id,'camera_id':obs.camera_id,
        'groundings':[{'kind':'polygon','role':role,'frame_id':obs.frame_id,
            'camera_id':obs.camera_id,'source':'vlm','target_id':'mug',
            'points':[[.2,.2],[.4,.2],[.4,.4],[.2,.4]]} for role in ('object','grasp_part')]+[
                {'kind':'points','role':'object','frame_id':obs.frame_id,'camera_id':obs.camera_id,
                 'source':'vlm','target_id':'mug','points':[[.3,.3]]}]}


def client_with_replies(tmp_path,replies,requests):
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(replies.pop(0))}}]})
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_refinement_logs_exact_crop_and_demands_fresh_grasp_before_ray(tmp_path):
    obs=observation();obs.rgb[:]=np.random.default_rng(6).integers(0,256,obs.rgb.shape,dtype=np.uint8)
    evidence=provenance(obs);requests=[];reply=grasp(obs)
    client=client_with_replies(tmp_path,[reply],requests)
    history=[{'feedback':'Old mapping-only advice preserves the prior ray'}]
    try:answer=client.decide(obs,'Grasp the handle',history,force_grasp=True,
        grounding_refinement=evidence,derived_image_context=evidence)
    finally:client.client.close()
    assert answer.action=='grasp' and answer.point is None and client.calls==1
    assert [list(point) for point in answer.groundings[0].points]==reply['groundings'][0]['points']
    logged=json.loads((tmp_path/'call_000.json').read_text())['request']
    context=logged['context'];refinement=context['grounding_refinement_context']
    assert context['allowed_actions']==['grasp','abort']
    assert context['must_ground_now'] and not context['must_observe_now']
    assert context['image_size']=={'width':32,'height':32} and context['history']==history
    assert not any(k in context for k in ('mapping_context','annotation_context','current_frame_reselection_context'))
    assert refinement['purpose']=='before_ray_same_frame_grasp_refinement'
    assert refinement['frame_id']==obs.frame_id and refinement['camera_id']==obs.camera_id
    assert refinement['acquires_new_observation'] is False
    assert refinement['derived_image_context']==evidence
    assert 'both ends of the visible straight bar' in refinement['instructions']
    assert 'prefer ONE tight object BOX' in refinement['instructions']
    assert 'mug body AND handle together' in refinement['instructions']
    assert 'SEPARATE tight visible grasp_part region inside the object region' in refinement['instructions']
    assert 'any polygon hole must lie inside its outer boundary' in refinement['instructions']
    # The preference does not rewrite the model's polygon into a synthesized box.
    assert answer.groundings[0].kind=='polygon'
    assert 'Use derived_image_context for its bounds' in context['response_contract']
    assert 'Do not return observe' in context['response_contract']
    image_path=logged['messages'][1]['content'][1]['image_url']['url']
    assert image_path.startswith('input_frame_001_')
    assert np.array_equal(np.asarray(Image.open(tmp_path/image_path)),obs.rgb)
    assert logged['image_sha256']==hashlib.sha256(obs.rgb.tobytes()).hexdigest()
    sent_url=requests[0]['messages'][1]['content'][1]['image_url']['url']
    assert base64.b64decode(sent_url.split(',',1)[1])==(tmp_path/image_path).read_bytes()
    assert len(requests[0]['messages'][1]['content'])==2


def test_observe_and_legacy_grounding_are_rejected_not_coerced_to_grasp(tmp_path):
    obs=observation();evidence=provenance(obs);requests=[]
    replies=[{'action':'observe','frame_id':1,'azimuth':0,'elevation':20,'distance':1.},
        {'action':'grasp','frame_id':1,'point':[.3,.3],'bbox':[.2,.2,.4,.4],'part_bbox':[.2,.2,.4,.4]},
        {'action':'abort','frame_id':1,'reason':'part cannot be located'}]
    client=client_with_replies(tmp_path,replies,requests)
    try:answer=client.decide(obs,'Grasp the handle',[],force_grasp=True,
        grounding_refinement=evidence,derived_image_context=evidence)
    finally:client.client.close()
    assert answer.action=='abort' and client.calls==3
    records=[json.loads((tmp_path/f'call_{i:03d}.json').read_text()) for i in range(3)]
    assert 'action is not supported' in records[0]['error']
    assert 'fresh explicit object AND grasp_part regions' in records[1]['error']
    assert records[2]['error'] is None
    assert [len(r['request']['messages']) for r in records]==[2,4,6]
    assert 'supplied image_size' in requests[1]['messages'][-1]['content']


def test_full_acquired_refinement_has_identity_coordinate_contract(tmp_path):
    obs=observation();evidence=provenance(obs);requests=[]
    evidence.update(kind='full_acquired_observation',source_shape_hw=list(obs.depth.shape),
        bounds_xyxy=[0,0,obs.depth.shape[1],obs.depth.shape[0]])
    client=client_with_replies(tmp_path,[grasp(obs)],requests)
    try:answer=client.decide(obs,'Grasp the handle',[],force_grasp=True,
        grounding_refinement=evidence,derived_image_context=evidence)
    finally:client.client.close()
    assert answer.action=='grasp' and len(requests)==1
    context=json.loads((tmp_path/'call_000.json').read_text())['request']['context']
    assert 'COMPLETE current acquired frame, with identity' in context['grounding_refinement_context']['instructions']
    assert 'not the original full image' not in context['response_contract']
    assert context['derived_image_context']['bounds_xyxy']==[0,0,32,32]


@pytest.mark.parametrize('fault',['frame','camera','source','points_only','candidate_only','no_object_seeds','contact_only'])
def test_refinement_repairs_wrong_source_or_nonregion_proposals(tmp_path,fault):
    obs=observation();evidence=provenance(obs);requests=[];invalid=grasp(obs)
    if fault=='frame':invalid['frame_id']=99
    elif fault=='camera':invalid['camera_id']='other'
    elif fault=='source':invalid['groundings'][0]['source']='rgbd'
    elif fault=='points_only':
        for g in invalid['groundings']:g.update(kind='points',points=[[.3,.3]])
    elif fault=='no_object_seeds':invalid['groundings'].pop()
    elif fault=='contact_only':invalid['groundings'][-1]['role']='grasp_contact'
    else:
        for g in invalid['groundings']:g.update(kind='candidate',candidate_id='old-target')
    client=client_with_replies(tmp_path,[invalid,grasp(obs)],requests)
    try:answer=client.decide(obs,'Grasp the handle',[],force_grasp=True,
        grounding_refinement=evidence,derived_image_context=evidence)
    finally:client.client.close()
    assert answer.action=='grasp' and client.calls==2
    assert json.loads((tmp_path/'call_000.json').read_text())['error'] is not None
    assert 'frame_id must be 1' in requests[1]['messages'][-1]['content']
    assert 'camera_id must be "activeview"' in requests[1]['messages'][-1]['content']


@pytest.mark.parametrize('fault',['unforced','force_observe','mounted','other_purpose','reselection',
    'diagnostic','missing_derived','wrong_type','source_frame','source_camera','shape','new_observation','mismatch'])
def test_refinement_scope_and_provenance_are_checked_before_any_request(tmp_path,fault):
    obs=observation();evidence=provenance(obs)
    kwargs={'force_grasp':True,'grounding_refinement':evidence,'derived_image_context':dict(evidence)}
    if fault=='unforced':kwargs['force_grasp']=False
    elif fault=='force_observe':kwargs['force_observe']=True
    elif fault=='mounted':obs=replace(obs,base_pose=np.zeros(3))
    elif fault=='other_purpose':kwargs['observe_purpose']='annotation'
    elif fault=='reselection':kwargs['current_frame_reselection']={}
    elif fault=='diagnostic':kwargs['annotation_diagnostic']=object()
    elif fault=='missing_derived':kwargs['derived_image_context']=None
    elif fault=='wrong_type':kwargs['grounding_refinement']='crop'
    elif fault=='mismatch':kwargs['derived_image_context']['bounds_xyxy']=[21,30,53,62]
    else:
        key,value={'source_frame':('source_frame_id',99),'source_camera':('source_camera_id','other'),
            'shape':('crop_shape_hw',[128,128]),'new_observation':('new_observation',True)}[fault]
        kwargs['grounding_refinement'][key]=value;kwargs['derived_image_context'][key]=value
    client=VLM(tmp_path)
    try:
        with pytest.raises(ValueError):client.decide(obs,'Grasp the handle',[],**kwargs)
    finally:client.client.close()
    assert client.calls==0 and not list(tmp_path.iterdir())


def geometry_feedback(obs):
    return {'frame_id':obs.frame_id,'camera_id':obs.camera_id,
        'coordinate_space':'supplied_crop_normalized_pixel_centers',
        'image_shape_hw':list(obs.depth.shape),'repair_attempt':1,'max_geometry_repairs':1,
        'previous_crop_decision':grasp(obs),
        'geometry_diagnostics':{'reason':'contact seed outside object mask',
            'rejected_seeds':[{'pixel':[23,9],'normalized':[23/31,9/31],
                'reasons':['outside_object_region']}],'valid_candidate_pixels':17}}


def test_one_geometry_feedback_reuses_exact_crop_and_preserves_raw_reply_and_rejection(tmp_path):
    obs=observation();evidence=provenance(obs);feedback=geometry_feedback(obs)
    original=copy.deepcopy(feedback);requests=[];reply=grasp(obs)
    # A fresh box is a model choice. The client does not synthesize or union it.
    reply['groundings'][0].update(kind='box',box=[.1,.1,.9,.9],points=[])
    client=client_with_replies(tmp_path,[grasp(obs),reply],requests)
    try:
        client.decide(obs,'Grasp the handle',[],force_grasp=True,
            grounding_refinement=evidence,derived_image_context=evidence)
        answer=client.decide(obs,'Grasp the handle',[],force_grasp=True,
            grounding_refinement=evidence,derived_image_context=evidence,
            grounding_refinement_feedback=feedback)
    finally:client.client.close()
    assert feedback==original and answer.groundings[0].box==(.1,.1,.9,.9)
    logs=[json.loads((tmp_path/f'call_{i:03d}.json').read_text())['request'] for i in range(2)]
    assert 'grounding_refinement_feedback' not in logs[0]['context']
    assert logs[1]['context']['grounding_refinement_feedback']==original
    instructions=logs[1]['context']['grounding_refinement_context']['instructions']
    assert 'ONE geometry correction in the SAME supplied crop' in instructions
    assert 'not correct semantic labels or suggested replacement contact points' in instructions
    assert 'Do not mechanically union, expand or clip regions' in instructions
    assert logs[0]['image_sha256']==logs[1]['image_sha256']
    assert requests[0]['messages'][1]['content'][1]==requests[1]['messages'][1]['content'][1]
    assert client.calls==2 and len(list(tmp_path.glob('input_frame_*.png')))==1


@pytest.mark.parametrize('fault',['missing_refinement','wrong_type','frame','camera','space','shape',
    'attempt','attempt_bool','limit','diagnostics','previous_missing','previous_action',
    'previous_frame','previous_camera','previous_grounding_frame','previous_grounding_source'])
def test_geometry_feedback_rejects_stale_or_out_of_scope_transaction_before_request(tmp_path,fault):
    obs=observation();evidence=provenance(obs);feedback=geometry_feedback(obs)
    kwargs={'force_grasp':True,'grounding_refinement':evidence,'derived_image_context':evidence,
        'grounding_refinement_feedback':feedback}
    if fault=='missing_refinement':
        kwargs.pop('grounding_refinement');kwargs.pop('derived_image_context')
    elif fault=='wrong_type':kwargs['grounding_refinement_feedback']='retry'
    elif fault=='previous_missing':feedback.pop('previous_crop_decision')
    elif fault.startswith('previous_grounding_'):
        key,value=('frame_id',99) if fault.endswith('frame') else ('source','rgbd')
        feedback['previous_crop_decision']['groundings'][0][key]=value
    elif fault.startswith('previous_'):
        key,value={'previous_action':('action','observe'),'previous_frame':('frame_id',99),
            'previous_camera':('camera_id','other')}[fault]
        feedback['previous_crop_decision'][key]=value
    else:
        key,value={'frame':('frame_id',99),'camera':('camera_id','other'),
            'space':('coordinate_space','full_image'),'shape':('image_shape_hw',[128,128]),
            'attempt':('repair_attempt',2),'attempt_bool':('repair_attempt',True),
            'limit':('max_geometry_repairs',2),'diagnostics':('geometry_diagnostics','guess')}[fault]
        feedback[key]=value
    client=VLM(tmp_path)
    try:
        with pytest.raises(ValueError):client.decide(obs,'Grasp the handle',[],**kwargs)
    finally:client.client.close()
    assert client.calls==0 and not list(tmp_path.iterdir())


def test_refinement_correction_logs_same_pixel_diagnostic_with_exact_wire_images(tmp_path):
    from view2grasp.annotation_diagnostic import build_annotation_diagnostic
    obs=observation();obs.rgb[:]=np.random.default_rng(45).integers(0,256,obs.rgb.shape,dtype=np.uint8)
    evidence=provenance(obs);feedback=geometry_feedback(obs);requests=[]
    feedback['rejected_samples']=feedback['geometry_diagnostics']['rejected_seeds']
    aid=build_annotation_diagnostic(obs,feedback)
    client=client_with_replies(tmp_path,[grasp(obs)],requests)
    try:answer=client.decide(obs,'Grasp the handle',[],force_grasp=True,
        grounding_refinement=evidence,derived_image_context=evidence,
        grounding_refinement_feedback=feedback,annotation_diagnostic=aid)
    finally:client.client.close()
    assert answer.action=='grasp' and client.calls==1
    request=json.loads((tmp_path/'call_000.json').read_text())['request']
    assert request['annotation_diagnostic_sha256']==aid.provenance()['diagnostic_rgb_sha256']
    context=request['context']
    assert context['image_size']=={'width':32,'height':32}
    assert context['grounding_refinement_feedback']==feedback
    assert 'not target labels or replacement seeds' in context['grounding_refinement_context']['instructions']
    assert len(requests[0]['messages'][1]['content'])==3
    for actual,logged in zip(requests[0]['messages'][1]['content'][1:],request['messages'][1]['content'][1:]):
        png=base64.b64decode(actual['image_url']['url'].split(',',1)[1])
        assert png==(tmp_path/logged['image_url']['url']).read_bytes()
    assert np.array_equal(aid.rgb[::aid.scale,::aid.scale],obs.rgb)


def test_refinement_correction_rejects_another_images_coordinate_aid(tmp_path):
    from view2grasp.annotation_diagnostic import build_annotation_diagnostic
    obs=observation();other=replace(obs,rgb=obs.rgb.copy());other.rgb[0,0]=[1,2,3]
    evidence=provenance(obs);feedback=geometry_feedback(obs)
    aid=build_annotation_diagnostic(other,feedback)
    client=VLM(tmp_path)
    try:
        with pytest.raises(ValueError,match='does not match the current unmarked image'):
            client.decide(obs,'Grasp the handle',[],force_grasp=True,
                grounding_refinement=evidence,derived_image_context=evidence,
                grounding_refinement_feedback=feedback,annotation_diagnostic=aid)
    finally:client.client.close()
    assert client.calls==0 and not list(tmp_path.iterdir())
