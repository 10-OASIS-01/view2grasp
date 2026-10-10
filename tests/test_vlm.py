import json
import httpx
import pytest
from view2grasp.vlm import VLM
from test_geometry import observation

def test_repairs_bad_json_and_stale_frame_without_logging_secret(tmp_path,monkeypatch):
    monkeypatch.setenv('VLM_API_KEY','secret-for-test')
    responses=['not json',json.dumps({'action':'abort','frame_id':99}),json.dumps({'action':'abort','frame_id':1,'reason':'unreachable'})]
    client=VLM(tmp_path);client.client.close()
    def handler(request):
        return httpx.Response(200,json={'model':'test-model','choices':[{'message':{'content':responses.pop(0)}}]})
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    answer=client.decide(observation(),'test',[])
    assert answer.frame_id==1 and client.calls==3
    records=[p.read_text() for p in sorted(tmp_path.glob('*.json'))]
    assert all('secret-for-test' not in text for text in records)
    assert json.loads(records[-1])['error'] is None
    assert len(json.loads(records[-1])['request']['messages'])==4
    client.client.close()

def test_service_error_is_bounded(tmp_path):
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(503,json={'error':'unavailable'})))
    with pytest.raises(RuntimeError,match='VLM failure'):client.decide(observation(),'test',[])
    assert client.calls==3
    client.client.close()

def test_timeout_is_bounded_and_logged(tmp_path):
    client=VLM(tmp_path);client.client.close()
    def timeout(request):raise httpx.ReadTimeout('deliberate test timeout',request=request)
    client.client=httpx.Client(transport=httpx.MockTransport(timeout))
    with pytest.raises(RuntimeError,match='VLM failure'):client.decide(observation(),'test',[])
    assert client.calls==3
    assert all('timeout' in json.loads(p.read_text())['error'] for p in tmp_path.glob('*.json'))
    client.client.close()

def test_response_replay_rejects_a_different_image(tmp_path):
    old=tmp_path/'old';old.mkdir()
    (old/'call_000.json').write_text(json.dumps({'request':{'image_sha256':'not-the-image'},'response':{}}))
    client=VLM(tmp_path/'new',replay=old)
    with pytest.raises(RuntimeError,match='image mismatch'):client.decide(observation(),'test',[])
    assert client.calls==1
    client.client.close()
