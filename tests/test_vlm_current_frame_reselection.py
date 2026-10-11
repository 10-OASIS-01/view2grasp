import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from test_geometry import observation
from test_vlm import isolated_api_settings, retry_sleeps
from view2grasp.vlm import VLM


def region(role,frame=1):
    return dict(kind='polygon',role=role,frame_id=frame,camera_id='activeview',
        target_id='mug',source='vlm',points=[[.2,.2],[.4,.2],[.4,.4],[.2,.4]])


def test_current_frame_reselection_demands_fresh_roles_and_rejects_stale_grounding(tmp_path):
    replies=[dict(action='grasp',frame_id=1,point=[.3,.3],bbox=[.2,.2,.4,.4],part_bbox=[.2,.2,.4,.4]),
        dict(action='grasp',frame_id=1,groundings=[region('object',0),region('grasp_part',0)]),
        dict(action='grasp',frame_id=1,groundings=[region('object'),region('grasp_part')])]
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(replies.pop(0))}}]})
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    history=[{'feedback':'Old mapping transaction: preserve the old grasp ray'}]
    evidence={'previous_grounding_frame':0,'cause':'tested clear candidates have no pregrasp IK'}
    try:answer=client.decide(observation(),'Grasp the handle',history,current_frame_reselection=evidence)
    finally:client.client.close()
    assert answer.action=='grasp' and client.calls==3
    context=json.loads(requests[0]['messages'][1]['content'][0]['text'])
    assert context['history']==history and 'mapping_context' not in context
    assert not context['must_ground_now'] and not context['must_observe_now']
    new=context['current_frame_reselection_context']
    assert new['evidence']==evidence and new['acquires_new_observation'] is False
    assert new['frame_id']==1 and new['camera_id']=='activeview'
    assert 'NEW camera-to-contact ray' in new['instructions']
    first=json.loads((tmp_path/'call_000.json').read_text())
    assert 'fresh explicit object AND grasp_part' in first['error']
    assert json.loads((tmp_path/'call_001.json').read_text())['error'] is not None


@pytest.mark.parametrize('action',['observe','abort'])
def test_new_transaction_may_request_a_view_or_abort(tmp_path,action):
    reply=dict(action=action,frame_id=1,azimuth=150,elevation=10,distance=.8)
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(reply)}}]})))
    try:answer=client.decide(observation(),'Grasp the handle',[],current_frame_reselection={})
    finally:client.client.close()
    assert answer.action==action and client.calls==1


@pytest.mark.parametrize('fault',['forced_observe','forced_grasp','mounted','wrong_type'])
def test_new_transaction_cannot_override_other_contexts(tmp_path,fault):
    client=VLM(tmp_path);obs=observation();kwargs={'current_frame_reselection':{}}
    if fault=='forced_observe':kwargs['force_observe']=True
    if fault=='forced_grasp':kwargs['force_grasp']=True
    if fault=='mounted':obs=replace(obs,base_pose=np.zeros(3))
    if fault=='wrong_type':kwargs['current_frame_reselection']='change the ray'
    try:
        with pytest.raises(ValueError,match='unforced tabletop decision'):
            client.decide(obs,'Grasp the handle',[],**kwargs)
    finally:client.client.close()
    assert client.calls==0
