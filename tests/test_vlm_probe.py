import base64
import io
import json

import httpx
from PIL import Image
import pytest

from view2grasp import vlm_probe
from view2grasp.vlm import VLM


@pytest.fixture(autouse=True)
def isolated_api_settings(monkeypatch):
    monkeypatch.setattr('view2grasp.vlm.time.sleep', lambda seconds: None)
    for name in ("VLM_PROVIDER", "ZHIPU_API_KEY", "GLM_API_KEY", "VLM_THINKING", "VLM_TIMEOUT_SECONDS", "VLM_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("VLM_API_KEY", "probe-test-key")
    monkeypatch.setenv("VLM_MODEL", "test-request-model")


def _mock_api(monkeypatch, response):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            "model": "actual-vision-model", "choices": [{"message": {"content": json.dumps(response)}}],
        })

    def factory(directory):
        client = VLM(directory)
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    monkeypatch.setattr(vlm_probe, "VLM", factory)
    return requests


def _proposal(contact=None):
    proposal = {
        "action": "grasp", "frame_id": 1, "camera_id": "api_probe",
        "target": "red square", "part": "red surface", "groundings": [
            {"kind": "box", "role": role, "frame_id": 1, "camera_id": "api_probe",
             "target_id": "red_square", "source": "vlm", "box": [0.11, 0.49, 0.36, 0.83]}
            for role in ("object", "grasp_part")
        ],
    }
    if contact is not None:
        proposal["groundings"].append({
            "kind": "points", "role": "grasp_contact", "frame_id": 1, "camera_id": "api_probe",
            "target_id": "red_square", "source": "vlm", "points": [contact],
        })
    return proposal


@pytest.mark.parametrize("contact", [None, [0.23, 0.65]])
def test_probe_sends_image_and_records_real_response_model(tmp_path, monkeypatch, contact):
    requests = _mock_api(monkeypatch, _proposal(contact))
    report = vlm_probe.run_vlm_probe(tmp_path / "probe")
    assert report["success"] and report["kind"] == "vlm_api_probe"
    assert report["robot_episode"] is False and report["calls"] == 1
    assert report["response_model"] == "actual-vision-model"
    assert report["response_models"] == ["actual-vision-model"]
    content = requests[0]["messages"][1]["content"]
    context = json.loads(content[0]["text"])
    assert context["must_ground_now"] is True
    assert "36" not in context["instruction"] and "0.23" not in context["instruction"]
    data = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (320, 240)
        assert image.getpixel((75, 155)) == (225, 25, 35)
    output = tmp_path / "probe"
    assert json.loads((output / "probe_report.json").read_text()) == report
    assert (output / "observations/frame_001.png").exists()
    assert (output / "vlm/call_000.json").exists()
    assert not (output / "result.json").exists()


@pytest.mark.parametrize("response", [
    {"action": "abort", "frame_id": 1, "camera_id": "api_probe", "reason": "Cannot locate it"},
    _proposal([0.76, 0.27]),
])
def test_probe_rejects_abort_and_wrong_color_even_with_correct_object_box(tmp_path, monkeypatch, response):
    _mock_api(monkeypatch, response)
    report = vlm_probe.run_vlm_probe(tmp_path / "probe")
    assert report["success"] is False
    assert report["validation"]["valid"] is False
    assert (tmp_path / "probe/probe_report.json").exists()


def test_probe_accepts_legacy_contact_coordinates(tmp_path, monkeypatch):
    _mock_api(monkeypatch, {
        "action": "grasp", "frame_id": 1, "camera_id": "api_probe",
        "point": [0.23, 0.65], "bbox": [0.11, 0.49, 0.36, 0.83],
        "part_bbox": [0.11, 0.49, 0.36, 0.83],
    })
    report = vlm_probe.run_vlm_probe(tmp_path / "probe")
    assert report["success"] is True
    assert report["validation"]["checks"][0]["source"] == "point"


def test_wrong_explicit_contact_box_is_not_hidden_by_correct_object_box(tmp_path, monkeypatch):
    proposal = _proposal()
    proposal["groundings"].append({
        "kind": "box", "role": "grasp_contact", "frame_id": 1, "camera_id": "api_probe",
        "target_id": "red_square", "source": "vlm", "box": [0.65, 0.1, 0.89, 0.42],
    })
    _mock_api(monkeypatch, proposal)
    report = vlm_probe.run_vlm_probe(tmp_path / "probe")
    assert report["success"] is False
    assert report["validation"]["checks"][0]["source"] == "grasp_contact_box_center"


def test_probe_preserves_report_when_client_configuration_fails(tmp_path, monkeypatch):
    def missing_configuration(directory):
        raise ValueError("Missing test configuration")

    monkeypatch.setattr(vlm_probe, "VLM", missing_configuration)
    report = vlm_probe.run_vlm_probe(tmp_path / "probe")
    assert report["success"] is False and report["calls"] == 0
    assert "Missing test configuration" in report["error"]
    assert (tmp_path / "probe/probe_report.json").exists()


def test_probe_records_transport_failure(tmp_path, monkeypatch):
    def factory(directory):
        client = VLM(directory)
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "unavailable"})))
        return client

    monkeypatch.setattr(vlm_probe, "VLM", factory)
    report = vlm_probe.run_vlm_probe(tmp_path / "probe")
    assert report["success"] is False and report["calls"] == 3
    assert report["error"] and report["response_model"] is None
    assert json.loads((tmp_path / "probe/probe_report.json").read_text())["success"] is False


def test_probe_refuses_to_overwrite_existing_evidence(tmp_path, monkeypatch):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("previous evidence")
    monkeypatch.setattr(vlm_probe, "VLM", lambda directory: pytest.fail("Must not call the API"))
    with pytest.raises(FileExistsError, match="overwrite"):
        vlm_probe.run_vlm_probe(tmp_path)
    assert sentinel.read_text() == "previous evidence"
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize("success,exit_code", [(True, 0), (False, 1)])
def test_check_vlm_cli_reports_probe_outcome(tmp_path, monkeypatch, success, exit_code):
    from typer.testing import CliRunner
    from view2grasp.cli import app

    output = tmp_path / "probe"
    report = {"kind": "vlm_api_probe", "robot_episode": False, "success": success}
    received = []

    def probe(directory):
        received.append(directory)
        return report

    monkeypatch.setattr(vlm_probe, "run_vlm_probe", probe)
    result = CliRunner().invoke(app, ["check-vlm", "--output", str(output)])
    assert result.exit_code == exit_code
    assert json.loads(result.stdout) == report
    assert received == [output]
