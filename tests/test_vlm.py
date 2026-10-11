import base64
import hashlib
import io
import json
import httpx
import pytest
from PIL import Image
from view2grasp.vlm import VLM
from test_geometry import observation


@pytest.fixture(autouse=True)
def isolated_api_settings(monkeypatch):
    for name in ("VLM_PROVIDER", "VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY", "ZHIPU_API_KEY",
                 "VLM_TIMEOUT_SECONDS", "VLM_MAX_TOKENS", "VLM_THINKING", "VLM_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def retry_sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr("view2grasp.vlm.time.sleep", delays.append)
    return delays


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
    messages=json.loads(records[-1])['request']['messages']
    assert len(messages)==6
    assert messages[2]=={'role':'assistant','content':'not json'}
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


def test_glm_sends_image_with_frozen_settings_and_auditable_redacted_log(tmp_path, monkeypatch):
    secret = "glm-transport-test-credential"
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    monkeypatch.setenv("ZHIPU_API_KEY", secret)
    obs = observation()
    client = VLM(tmp_path)
    assert client.client.timeout.read == 120
    client.client.close()
    # Changing the shell environment midway through an episode must not change
    # the provider or make the recorded configuration differ from the request.
    monkeypatch.setenv("VLM_BASE_URL", "https://different.test/v1")
    monkeypatch.setenv("VLM_MODEL", "different-model")
    monkeypatch.setenv("VLM_API_KEY", "different-credential")
    monkeypatch.setenv("VLM_MAX_TOKENS", "32")
    monkeypatch.setenv("VLM_THINKING", "disabled")
    monkeypatch.setenv("VLM_REASONING_EFFORT", "high")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "model": "actual-returned-vision-model", "provider_debug": secret,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "action": "abort", "frame_id": obs.frame_id, "camera_id": obs.camera_id,
                "reason": "No safe target",
            })}}],
        })

    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        answer = client.decide(obs, "Locate a graspable object", [])
    finally:
        client.client.close()
    assert answer.action == "abort" and len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert request.headers["Authorization"] == "Bearer " + secret
    payload = json.loads(request.content)
    assert payload["model"] == "glm-5.3-flashx"
    assert payload["max_tokens"] == 4096 and payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
    assert "response_format" not in payload
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
        assert image.convert("RGB").tobytes() == obs.rgb.tobytes()
    log_text = (tmp_path / "call_000.json").read_text()
    assert secret not in log_text and "different-credential" not in log_text
    assert "data:image" not in log_text
    log = json.loads(log_text)
    assert log["request"]["transport"] == "online"
    assert log["request"]["connection"] == client.settings.public_metadata()
    assert log["request"]["messages"][1]["content"][1]["image_url"]["url"] == "../observations/frame_001.png"
    assert log["request"]["image_sha256"] == hashlib.sha256(obs.rgb.tobytes()).hexdigest()
    assert log["response"]["model"] == "actual-returned-vision-model"
    assert log["response"]["provider_debug"] == "[REDACTED]"


def test_compatible_transport_keeps_minimal_request_payload(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content":
            json.dumps({"action": "abort", "frame_id": 1})}}]})

    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert str(requests[0].url) == "http://127.0.0.1:8318/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer local"
    assert set(json.loads(requests[0].content)) == {"model", "messages"}


@pytest.mark.parametrize("response", [
    None,
    {"choices": []},
    {"choices": [None]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": ""}}]},
    {"choices": [{"message": {"content": "   "}}]},
    {"choices": [{"message": {"content": ["not a text response"]}}]},
    {"choices": [{"message": {"content": None, "reasoning_content":
        json.dumps({"action": "abort", "frame_id": 1})}}]},
    {"choices": [{"finish_reason": "length", "message": {"content":
        json.dumps({"action": "abort", "frame_id": 1})}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "```json"}}]},
])
def test_malformed_or_incomplete_responses_never_become_robot_actions(tmp_path, response):
    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=response)))
    try:
        with pytest.raises(RuntimeError, match="VLM failure"):
            client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert client.calls == 3
    logs = [json.loads(path.read_text()) for path in tmp_path.glob("call_*.json")]
    assert len(logs) == 3 and all(log["error"] for log in logs)


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_errors_stop_without_spending_repair_attempts(tmp_path, status):
    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": "access denied"})))
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert client.calls == 1
    assert json.loads((tmp_path / "call_000.json").read_text())["error"]


@pytest.mark.parametrize("status", [400, 404])
def test_invalid_request_stops_once_and_reports_bounded_redacted_provider_detail(
        tmp_path, monkeypatch, retry_sleeps, status):
    secret = "invalid-request-test-credential"
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    monkeypatch.setenv("VLM_API_KEY", secret)
    monkeypatch.setenv("VLM_THINKING", "enabled")
    monkeypatch.setenv("VLM_REASONING_EFFORT", "low")
    detail = "Model or request options are unsupported for " + secret + ": " + "x" * 600 + "unbounded-tail"
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": detail}})

    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match=f"request rejected.*HTTP {status}") as error:
            client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert client.calls == 1 and len(requests) == 1
    assert retry_sleeps == []
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "low"
    message = str(error.value)
    assert secret not in message and "[REDACTED]" in message
    assert "unbounded-tail" not in message
    assert message.split(": ", 1)[1] == detail.replace(secret, "[REDACTED]")[:500]
    logs = list(tmp_path.glob("call_*.json"))
    assert len(logs) == 1
    log_text = logs[0].read_text()
    assert secret not in log_text
    log = json.loads(log_text)
    assert log["response"]["error"]["message"] == detail.replace(secret, "[REDACTED]")
    assert log["request"]["connection"]["request_options"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("retry_after,expected_delay", [("7", 7), ("nan", 1), ("999", 30)])
def test_rate_limit_retries_preserve_request_and_log_redacted_response(
        tmp_path, monkeypatch, retry_sleeps, retry_after, expected_delay):
    secret = "rate-limit-test-credential"
    monkeypatch.setenv("VLM_API_KEY", secret)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": retry_after}, json={
                "error": {"code": "rate_limited", "message": "Too many requests for " + secret},
            })
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"action": "abort", "frame_id": 1}),
        }}]})

    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        decision = client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert decision.action == "abort" and client.calls == 2
    assert retry_sleeps == [expected_delay]
    assert all(0 <= delay <= 30 for delay in retry_sleeps)
    # Retry a transport failure with the original request; do not ask a model
    # that has not produced an answer to repair its JSON.
    assert requests[0] == requests[1]
    assert "Correct your JSON response" not in json.dumps(requests[1]["messages"])
    failure_text = (tmp_path / "call_000.json").read_text()
    assert secret not in failure_text
    failure = json.loads(failure_text)
    assert failure["response"] == {"error": {
        "code": "rate_limited", "message": "Too many requests for [REDACTED]",
    }}
    assert "429" in failure["error"]
    assert json.loads((tmp_path / "call_001.json").read_text())["error"] is None


def test_glm_replay_needs_no_key_and_preserves_source_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    obs = observation()
    source = tmp_path / "source"
    source.mkdir()
    original_connection = {"provider": "openai-compatible", "base_url": "http://original.test/v1",
                           "model": "original-model", "request_options": {}}
    (source / "call_000.json").write_text(json.dumps({
        "request": {"image_sha256": hashlib.sha256(obs.rgb.tobytes()).hexdigest(),
                    "model": "original-model", "connection": original_connection},
        "response": {"model": "original-returned-model", "choices": [{"message": {"content":
            json.dumps({"action": "abort", "frame_id": obs.frame_id, "camera_id": obs.camera_id})}}]},
    }))
    client = VLM(tmp_path / "replayed", replay=source)
    assert client.settings.api_key == ""
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: pytest.fail("Response replay must not contact any API")))
    try:
        answer = client.decide(obs, "test", [])
    finally:
        client.client.close()
    assert answer.action == "abort" and client.calls == 1
    log = json.loads((tmp_path / "replayed/call_000.json").read_text())
    assert log["request"]["transport"] == "replay"
    assert log["request"]["model"] == "original-model"
    assert log["request"]["connection"] == original_connection
    assert log["response"]["model"] == "original-returned-model"
