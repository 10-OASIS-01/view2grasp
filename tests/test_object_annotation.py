"""Bounded saved-image annotation, isolated coordinates and independent replies."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from view2grasp.geometry import unproject
from view2grasp.grounding import mask_rle, pixels
from view2grasp.object_annotation import annotate_source_object
from view2grasp.types import AgentDecision, GroundingProposal, Observation


def source_scene(two_parts=False):
    depth = np.full((320, 400), 2.)
    depth[210:224, 230:244] = 1.
    if two_parts: depth[210:224, 254:268] = 1.
    obs = Observation(17, np.zeros((320, 400, 3), np.uint8), depth,
        np.array([[1000., 0., 199.5], [0., 1000., 159.5], [0., 0., 1.]]),
        np.eye(4), np.zeros(7), np.array([.04, -.04]), 12.5, (23., 55., .8), 'source_camera')
    y, x = np.mgrid[:320, :400]
    xyz = unproject(np.stack([x, y], -1), depth, obs.K, obs.T_wc)
    reference = xyz[210:214, 230:234].reshape(-1, 3)
    if two_parts: reference = np.vstack([reference, xyz[210:214, 254:258].reshape(-1, 3)])
    return obs, reference, xyz


def reply(obs, mode):
    yy, xx = np.where(obs.depth == 1.)
    h, w = obs.depth.shape
    mask = np.zeros((h, w), bool)
    mask[yy.min():yy.max()+1, xx.min():xx.max()+1] = True
    seed = [int(xx.min()), int(yy.min())]
    if mode.startswith('right'): seed[0] = int(xx.max())
    seeds = [seed]
    if 'bad' in mode: seeds.append([int(xx.min()-1), int(yy.min())])
    if mode == 'all_bad': seeds = seeds[1:]
    common = dict(role='object', frame_id=obs.frame_id, camera_id=obs.camera_id, target_id='mug')
    return AgentDecision(action='observe', frame_id=obs.frame_id, camera_id=obs.camera_id,
        azimuth=obs.view[0], elevation=obs.view[1], distance=obs.view[2], groundings=[
            GroundingProposal(kind='mask', mask_size=(h, w), mask_rle=mask_rle(mask), **common),
            GroundingProposal(kind='points', points=[(x/(w-1), y/(h-1)) for x, y in seeds], **common)])


class FakeVLM:
    def __init__(self, modes):
        self.modes = modes
        self.requests = []
        self.replies = []
        self.calls = 5
        self.seconds = 1.

    def decide(self, obs, instruction, history, **kwargs):
        index = len(self.requests)
        assert index < len(self.modes), 'unexpected third semantic request'
        self.calls += 1
        self.seconds += .1
        self.requests.append({'obs': obs, 'instruction': instruction,
            'history': copy.deepcopy(history), 'kwargs': kwargs})
        # Mutating client-side request data must not mutate retained logs or
        # introduce unrelated state into the next attempt.
        history[0]['source_object_annotation']['client_mutation'] = True
        mode = self.modes[index]
        if mode == 'transport_error': raise RuntimeError('provider exhausted bounded transport retries')
        decision = reply(obs, mode)
        if mode == 'abort': decision = decision.model_copy(update={'action': 'abort'})
        elif mode == 'wrong_frame':
            decision = decision.model_copy(update={'frame_id': obs.frame_id+1})
        elif mode == 'wrong_identity':
            decision.groundings = [g.model_copy(update={'target_id': 'neighbor'}) for g in decision.groundings]
        elif mode == 'wrong_role':
            decision.groundings[1] = decision.groundings[1].model_copy(update={'role': 'grasp_part'})
        self.replies.append(decision.model_dump(mode='json'))
        return decision


def annotate(client, obs, reference, **kwargs):
    return annotate_source_object(client, obs, reference, 'mug', .9, **kwargs)


def test_one_success_uses_reference_crop_and_restores_exact_source_pixels():
    obs, reference, xyz = source_scene()
    original_reference = reference.copy()
    original_depth = obs.depth.copy()
    client = FakeVLM(['good'])
    result = annotate(client, obs, reference)
    assert result.ownership is not None and len(result.attempts) == 1
    context = result.report['image_context']
    x0, y0, x1, y1 = context['bounds_xyxy']
    assert x0 > 0 and y0 > 0
    assert client.requests[0]['obs'].depth.shape == (y1-y0, x1-x0)
    assert np.array_equal(client.requests[0]['obs'].rgb, obs.rgb[y0:y1, x0:x1])
    assert np.array_equal(result.ownership.object_points, xyz[210:224, 230:244].reshape(-1, 3))
    restored = AgentDecision.model_validate(result.attempts[0].log['source_annotation'])
    assert pixels(restored.groundings[1].points, obs.depth.shape).tolist() == [[230, 210]]
    assert np.array_equal(reference, original_reference) and np.array_equal(obs.depth, original_depth)
    assert result.report['vlm_calls'] == 1 and result.report['inference_seconds'] == pytest.approx(.1)
    assert result.report['new_observations'] == 0 and not result.report['reply_results_merged']


def test_real_rejection_gets_one_same_crop_correction_and_localized_diagnostic():
    obs, reference, _ = source_scene()
    client = FakeVLM(['left_bad', 'good'])
    result = annotate(client, obs, reference)
    assert len(result.attempts) == 2 and result.ownership is not None
    first, second = client.requests
    assert first['obs'] is second['obs']
    assert first['kwargs']['derived_image_context'] == second['kwargs']['derived_image_context']
    assert first['kwargs']['annotation_diagnostic'] is None
    diagnostic = second['kwargs']['annotation_diagnostic']
    diagnostic.validate_source(second['obs'])
    transaction = second['history'][0]['source_object_annotation']
    assert len(second['history']) == 1
    assert transaction['previous_annotation'] == client.replies[0]
    assert transaction['attempt'] == 1
    feedback = transaction['geometry_diagnostics']
    x0, y0, _, _ = result.report['image_context']['bounds_xyxy']
    expected = [229-x0, 210-y0]
    assert feedback['rejected_samples'][0]['pixel'] == expected
    assert diagnostic.rejected_pixels == (tuple(expected),)
    ch, cw = second['obs'].depth.shape
    assert feedback['rejected_samples'][0]['normalized'] == pytest.approx(np.array(expected)/[cw-1, ch-1])
    assert result.attempts[0].ownership is None
    assert 'client_mutation' not in str(result.report)
    assert 'source_pixels_xy' not in str(second['history'])
    assert 'ONLY role=object' in second['instruction']
    assert all(r['kwargs']['force_observe'] and r['kwargs']['observe_purpose'] == 'annotation'
               for r in client.requests)
    assert all(r['kwargs']['annotation_roles'] == ('object',) for r in client.requests)


def test_two_failures_stop_without_a_third_request_or_previous_result_fallback():
    obs, reference, _ = source_scene()
    client = FakeVLM(['left_bad', 'left_bad'])
    result = annotate(client, obs, reference)
    assert len(result.attempts) == 2 and result.ownership is None
    assert all(a.log['diagnostics']['failure'] == 'invalid_explicit_seed' for a in result.attempts)
    assert result.report['vlm_calls'] == 2 and result.report['semantic_requests'] == 2


def test_partial_abstention_allows_one_correction_but_reply_components_are_never_merged():
    obs, reference, _ = source_scene(two_parts=True)
    client = FakeVLM(['left_bad', 'right_bad'])
    result = annotate(client, obs, reference, allow_partial_seed_components=True)
    assert len(result.attempts) == 2 and all(a.ownership is not None for a in result.attempts)
    first, second = [a.ownership for a in result.attempts]
    assert first.report['partial'] and second.report['partial']
    assert not np.any(first.object_mask & second.object_mask)
    assert result.ownership is second
    assert len(first.object_points) == len(second.object_points) == 196
    assert not result.ownership.object_mask[210:224, 230:244].any()
    assert result.ownership.object_mask[210:224, 254:268].all()
    assert 'selected_components' not in str(client.requests[1]['history'])


def test_partial_then_abort_does_not_implicitly_resurrect_earlier_ownership():
    obs, reference, _ = source_scene()
    client = FakeVLM(['left_bad', 'abort'])
    result = annotate(client, obs, reference, allow_partial_seed_components=True)
    assert result.attempts[0].ownership is not None
    assert result.ownership is None and result.attempts[-1].ownership is None
    assert result.report['last_reply_has_ownership'] is False


@pytest.mark.parametrize('mode', ['abort', 'wrong_frame', 'wrong_identity', 'wrong_role', 'transport_error'])
def test_non_geometric_failure_stops_conservatively_without_semantic_retry(mode):
    obs, reference, _ = source_scene()
    client = FakeVLM([mode])
    result = annotate(client, obs, reference)
    assert result.ownership is None and len(result.attempts) == 1
    assert 'error' in result.attempts[0].log
    assert client.calls == 6


def test_crop_failure_makes_no_request_and_returns_a_persistable_failure_log():
    obs, _, _ = source_scene()
    client = FakeVLM([])
    result = annotate(client, obs, np.array([[0., 0., -1.], [.001, 0., -1.],
                                           [0., .001, -1.], [.001, .001, -1.]]))
    assert result.ownership is None and not result.attempts
    assert result.report['failure'] == 'crop_unavailable'
    assert result.report['vlm_calls'] == 0


def test_invalid_reference_is_rejected_before_any_model_request():
    obs, _, _ = source_scene()
    client = FakeVLM([])
    result = annotate(client, obs, np.array([[0., 0., 1.]]))
    assert result.report['failure'] == 'invalid_source_reference'
    assert result.report['vlm_calls'] == result.report['semantic_requests'] == 0


def test_helper_never_reacquires_an_image_or_changes_source_timestamp_and_camera():
    obs, reference, _ = source_scene()
    client = FakeVLM(['all_bad', 'good'])
    result = annotate(client, obs, reference, robot_depth=np.full(obs.depth.shape, 10.))
    for request in client.requests:
        assert request['obs'].frame_id == obs.frame_id
        assert request['obs'].camera_id == obs.camera_id
        assert request['obs'].timestamp == obs.timestamp
        assert request['obs'].view == obs.view
    assert result.ownership.source_frame_id == obs.frame_id
    assert result.ownership.source_camera_id == obs.camera_id
    assert result.report['new_observations'] == 0


def test_actual_saved_partial_reply_replayed_twice_remains_bounded_and_unedited():
    fixture = np.load(Path(__file__).parent/'fixtures/mug_object_ownership_reply.npz')
    metadata = json.loads(str(fixture['metadata']))
    actual = AgentDecision.model_validate_json(str(fixture['actual_response_content']))
    obs = Observation(1, fixture['rgb'], fixture['depth'], fixture['K'], fixture['T_wc'],
        np.zeros(7), np.array([.04, -.04]), metadata['source_timestamp'], (0., 80., 1.3))

    class Replay:
        calls = 0
        seconds = 0.
        requests = []

        def decide(self, observation, instruction, history, **kwargs):
            self.calls += 1
            assert self.calls <= 2
            assert np.array_equal(observation.rgb, fixture['rgb'])
            assert np.array_equal(observation.depth, fixture['depth'])
            self.requests.append(copy.deepcopy(history))
            return actual.model_copy(deep=True)

    client = Replay()
    result = annotate_source_object(client, obs, fixture['reference'], 'mug',
        float(fixture['support_z']), robot_depth=fixture['robot_depth'], allow_partial_seed_components=True)
    assert len(result.attempts) == 2
    assert all(a.log['raw_annotation'] == actual.model_dump(mode='json') for a in result.attempts)
    assert all(len(a.ownership.object_points) == 458 for a in result.attempts)
    assert all(a.ownership.report['partial'] for a in result.attempts)
    feedback = client.requests[1][0]['source_object_annotation']['geometry_diagnostics']
    assert [sample['pixel'] for sample in feedback['rejected_samples']] == [[70, 72], [57, 72]]
    assert len(result.ownership.report['abstentions']) == 2
    assert result.report['reply_results_merged'] is False
