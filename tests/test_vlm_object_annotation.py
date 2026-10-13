"""Provider messages distinguish object ownership from object/part contact."""
import json

import httpx
import pytest

from view2grasp.vlm import VLM
from test_object_annotation import source_scene, reply


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in ('VLM_PROVIDER', 'VLM_BASE_URL', 'VLM_MODEL', 'VLM_API_KEY', 'ZHIPU_API_KEY',
                 'VLM_TIMEOUT_SECONDS', 'VLM_MAX_TOKENS', 'VLM_THINKING', 'VLM_REASONING_EFFORT'):
        monkeypatch.delenv(name, raising=False)


def client_with_responses(directory, responses, requests):
    client = VLM(directory)
    client.client.close()

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': responses.pop(0).model_dump_json()}}]})

    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.parametrize('object_only', [False, True])
def test_actual_provider_context_and_log_preserve_default_or_request_object_only(tmp_path, object_only):
    obs, _, _ = source_scene()
    requests = []
    client = client_with_responses(tmp_path, [reply(obs, 'good')], requests)
    kwargs = {'annotation_roles': ('object',)} if object_only else {}
    try:
        client.decide(obs, 'Audit object evidence', [], force_observe=True, observe_purpose='annotation', **kwargs)
    finally:
        client.client.close()
    context = json.loads(requests[0]['messages'][1]['content'][0]['text'])
    annotation = context['annotation_context']
    assert annotation['purpose'] == 'current_frame_annotation'
    assert annotation['acquires_new_observation'] is False
    if object_only:
        assert annotation['requested_roles'] == ['object']
        assert 'Annotate ONLY the requested object' in annotation['instructions']
        assert 'No permitted part is requested' in annotation['instructions']
        assert 'object and permitted part' not in annotation['instructions']
        assert 'only object groundings' in context['response_contract']
    else:
        assert 'requested_roles' not in annotation
        assert annotation['instructions'].startswith('Annotate the requested object and permitted part')
        assert 'only object groundings' not in context['response_contract']
    saved = json.loads((tmp_path/'call_000.json').read_text())
    assert saved['request']['context']['annotation_context'] == annotation
    assert client.calls == 1


def test_wrong_part_in_object_only_reply_gets_existing_bounded_schema_repair(tmp_path):
    obs, _, _ = source_scene()
    bad = reply(obs, 'good')
    bad.groundings[1] = bad.groundings[1].model_copy(update={'role': 'grasp_part'})
    requests = []
    client = client_with_responses(tmp_path, [bad, reply(obs, 'good')], requests)
    try:
        result = client.decide(obs, 'Audit object evidence', [], force_observe=True,
            observe_purpose='annotation', annotation_roles=('object',))
    finally:
        client.client.close()
    assert client.calls == 2
    assert all(g.role == 'object' for g in result.groundings)
    assert 'only role=object' in requests[1]['messages'][-1]['content']
    assert json.loads((tmp_path/'call_000.json').read_text())['error'] is not None


@pytest.mark.parametrize('kwargs', [
    {'annotation_roles': ('object',)},
    {'force_observe': True, 'annotation_roles': ('object',)},
    {'force_observe': True, 'observe_purpose': 'annotation', 'annotation_roles': ('grasp_part',)},
    {'force_observe': True, 'observe_purpose': 'annotation', 'annotation_roles': 'object'},
])
def test_role_override_cannot_leak_into_mapping_or_unforced_calls(tmp_path, kwargs):
    obs, _, _ = source_scene()
    client = VLM(tmp_path)
    try:
        with pytest.raises(ValueError, match='annotation_roles'):
            client.decide(obs, 'test', [], **kwargs)
    finally:
        client.client.close()
    assert client.calls == 0
