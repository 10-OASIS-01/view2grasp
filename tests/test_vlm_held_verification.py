"""The last reserved camera view verifies held material, not an UNKNOWN path."""
import copy
from dataclasses import replace
import json

import httpx
import numpy as np
import pytest

from test_geometry import observation
from test_vlm import isolated_api_settings, retry_sleeps
from view2grasp.vlm import VLM


def held_request(obs):
    return {'coverage_advice':{'purpose':'held_object_verification',
        'views':[[90.,0.,1.3]],'views_are_alternatives':True,'max_observations_to_acquire':1,
        'reference_kind':'fixed_observed_attachment_at_current_robot_pose'},
        'held_verification':{'frame_id':obs.frame_id,'camera_id':obs.camera_id,
            'strict_association_verified':False,'remaining_real_observations':1,
            'remaining_transfer_mapping_observations_after_this_view':0},
        'feedback':'The stationary held object needs one different actual verification view.'}


def decide(tmp_path,obs,history,**kwargs):
    requests=[];answer={'action':'observe','frame_id':obs.frame_id,'camera_id':obs.camera_id,
        'azimuth':90.,'elevation':0.,'distance':1.3}
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        (requests.append(json.loads(request.content)) or httpx.Response(200,
            json={'choices':[{'message':{'content':json.dumps(answer)}}]}))))
    try:result=client.decide(obs,'Verify held object',history,**kwargs)
    finally:client.client.close()
    return result,json.loads(requests[0]['messages'][1]['content'][0]['text']),requests[0]


def test_current_latest_held_transaction_overrides_unknown_mapping_purpose(tmp_path):
    obs=observation();latest=held_request(obs)
    history=[{'coverage_advice':{'views':[[0.,60.,1.3]],'unknown_points':99}},
        {'view':[90.,0.,1.3],'feedback':'before lifting'},latest]
    before=copy.deepcopy(history)
    result,context,request=decide(tmp_path,obs,history,force_observe=True)
    held=context['held_verification_context']
    assert result.action=='observe' and context['must_observe_now'] is True
    assert 'mapping_context' not in context and 'annotation_context' not in context
    assert held['purpose']=='held_object_verification'
    assert held['frame_id']==obs.frame_id and held['camera_id']==obs.camera_id
    assert held['latest_coverage_advice']==latest['coverage_advice']
    assert held['different_advised_views']==[[90.,0.,1.3]]  # Pre-lift visit is not excluded.
    assert held['remaining_real_observations']==1
    assert held['remaining_transfer_mapping_observations_after_this_view']==0
    assert 'no rejected path samples are supplied' in held['instructions']
    assert 'fixed attachment reference' in held['instructions']
    assert 'last reserved real observation' in held['instructions']
    assert 'held-object verification' in context['response_contract']
    assert 'mapping-only coverage preferences below do not apply' in request['messages'][0]['content']
    assert history==before


@pytest.mark.parametrize('fault',['frame','camera','verified','missing_advice','wrong_purpose',
    'zero_remaining','two_remaining','bool_remaining','transfer_still_available',
    'later_feedback','later_coverage','later_held_stale'])
def test_old_or_incomplete_held_evidence_does_not_trigger_special_context(tmp_path,fault):
    obs=observation();latest=held_request(obs);history=[latest]
    if fault=='missing_advice':latest.pop('coverage_advice')
    elif fault=='wrong_purpose':latest['coverage_advice']['purpose']='mapping_only'
    elif fault=='later_feedback':history.append({'feedback':'another transaction'})
    elif fault=='later_coverage':history.append({'coverage_advice':{'views':[]}})
    elif fault=='later_held_stale':
        later=held_request(obs);later['held_verification']['frame_id']=99;history.append(later)
    else:
        key,value={'frame':('frame_id',99),'camera':('camera_id','other'),
            'verified':('strict_association_verified',True),
            'zero_remaining':('remaining_real_observations',0),
            'two_remaining':('remaining_real_observations',2),
            'bool_remaining':('remaining_real_observations',True),
            'transfer_still_available':('remaining_transfer_mapping_observations_after_this_view',1)}[fault]
        latest['held_verification'][key]=value
    _,context,_=decide(tmp_path,obs,history,force_observe=True)
    assert 'held_verification_context' not in context
    assert context['mapping_context']['purpose']=='mapping_only'


@pytest.mark.parametrize('scope',['unforced','annotation','mounted'])
def test_held_history_does_not_relabel_other_request_purposes(tmp_path,scope):
    obs=observation();kwargs={'force_observe':True}
    if scope=='unforced':kwargs={}
    elif scope=='annotation':kwargs['observe_purpose']='annotation'
    else:obs=replace(obs,base_pose=np.zeros(3))
    _,context,_=decide(tmp_path,obs,[held_request(obs)],**kwargs)
    assert 'held_verification_context' not in context
    if scope=='annotation':assert context['annotation_context']['purpose']=='current_frame_annotation'


def test_held_transaction_still_repairs_grasp_action_without_promoting_it(tmp_path):
    obs=observation();responses=[{'action':'grasp','frame_id':obs.frame_id,
        'point':[.5,.5],'bbox':[.1,.1,.9,.9],'part_bbox':[.3,.3,.7,.7]},
        {'action':'observe','frame_id':obs.frame_id,'azimuth':90.,'elevation':0.,'distance':1.3}]
    client=VLM(tmp_path);client.client.close();requests=[]
    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(responses.pop(0))}}]})
    client.client=httpx.Client(transport=httpx.MockTransport(response))
    try:answer=client.decide(obs,'Verify', [held_request(obs)],force_observe=True)
    finally:client.client.close()
    assert client.calls==2 and answer.action=='observe'
    assert 'held object verification step requires an observe action' in json.loads((tmp_path/'call_000.json').read_text())['error']
    assert 'held-object verification' in requests[1]['messages'][-1]['content']
