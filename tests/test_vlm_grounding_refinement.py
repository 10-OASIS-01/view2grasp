"""Same acquired crop refines a new grasp; it is never an observe annotation."""
import base64
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
            'points':[[.2,.2],[.4,.2],[.4,.4],[.2,.4]]} for role in ('object','grasp_part')]}


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
    assert 'not the original full image' in context['response_contract']
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
    assert 'supplied crop image_size' in requests[1]['messages'][-1]['content']


@pytest.mark.parametrize('fault',['frame','camera','source','points_only','candidate_only'])
def test_refinement_repairs_wrong_source_or_nonregion_proposals(tmp_path,fault):
    obs=observation();evidence=provenance(obs);requests=[];invalid=grasp(obs)
    if fault=='frame':invalid['frame_id']=99
    elif fault=='camera':invalid['camera_id']='other'
    elif fault=='source':invalid['groundings'][0]['source']='rgbd'
    elif fault=='points_only':
        for g in invalid['groundings']:g.update(kind='points',points=[[.3,.3]])
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
