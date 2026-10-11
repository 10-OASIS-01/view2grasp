"""Bounded saved-image annotation, isolated coordinates and independent replies."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from view2grasp.geometry import unproject
from view2grasp.grounding import mask_rle, pixels
from view2grasp.object_annotation import annotate_source_object, _correction_feedback
from view2grasp.observation_crop import crop_observation
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


def test_unseeded_component_gets_one_review_without_automatic_ownership():
    obs, reference, _ = source_scene(two_parts=True)
    client = FakeVLM(['good', 'good'])
    result = annotate(client, obs, reference, allow_partial_seed_components=True)
    assert len(result.attempts) == 2
    assert len(result.ownership.object_points) == 196
    assert not result.ownership.object_mask[210:224, 254:268].any()
    # A valid but incomplete first answer is not silently broadened. The model
    # can ignore the review, and the same surfaces remain unowned afterward.
    assert np.array_equal(result.attempts[0].ownership.object_mask, result.ownership.object_mask)
    feedback = client.requests[1]['history'][0]['source_object_annotation']['geometry_diagnostics']
    assert feedback['failures'] == feedback['rejected_samples'] == []
    assert feedback['unresolved_measured_component_count'] == 1
    component = feedback['unresolved_measured_components'][0]
    x0, y0, _, _ = result.report['image_context']['bounds_xyxy']
    expected = np.array([254-x0, 210-y0, 267-x0, 223-y0])
    assert component['bbox_pixel_xyxy'] == expected.tolist()
    h, w = client.requests[1]['obs'].depth.shape
    assert component['bbox_normalized_xyxy'] == pytest.approx(expected / [w-1, h-1, w-1, h-1])
    assert 'unresolved' in component['semantic_status']
    assert client.requests[1]['kwargs']['annotation_diagnostic'].rejected_pixels == ()
    assert 'reference overlap' in client.requests[1]['instruction']
    assert result.report['vlm_calls'] == 2 and result.report['new_observations'] == 0


def test_unresolved_review_abort_does_not_fall_back_to_successful_first_answer():
    obs, reference, _ = source_scene(two_parts=True)
    client = FakeVLM(['good', 'abort'])
    result = annotate(client, obs, reference, allow_partial_seed_components=True)
    assert result.attempts[0].ownership is not None
    assert len(result.attempts) == 2 and result.ownership is None


def test_review_accepts_second_component_only_after_new_explicit_semantic_seed():
    obs, reference, _ = source_scene(two_parts=True)

    class ConfirmBoth(FakeVLM):
        def decide(self, obs, instruction, history, **kwargs):
            decision = super().decide(obs, instruction, history, **kwargs)
            if len(self.requests) == 2:
                yy, xx = np.where(obs.depth == 1.)
                h, w = obs.depth.shape
                decision.groundings[1].points.append((int(xx.max())/(w-1), int(yy.min())/(h-1)))
            return decision

    result = annotate(ConfirmBoth(['good', 'good']), obs, reference, allow_partial_seed_components=True)
    assert len(result.attempts[0].ownership.object_points) == 196
    assert len(result.ownership.object_points) == 392
    assert result.ownership.report['unresolved_measured_component_count'] == 0
    assert result.report['reply_results_merged'] is False


@pytest.mark.parametrize('partial', [False, True])
def test_envelope_correction_marks_actual_tail_not_legal_seed_and_still_stops_at_two(partial):
    obs, reference, _ = source_scene()
    depth = obs.depth.copy()
    depth[210:224, 230:274] = 1.
    obs = replace(obs, depth=depth)
    arrays = [obs.rgb, obs.depth, obs.K, obs.T_wc, reference]
    before = [array.copy() for array in arrays]
    client = FakeVLM(['good', 'good'])
    result = annotate(client, obs, reference, allow_partial_seed_components=partial)

    assert result.ownership is None and len(result.attempts) == 2
    assert result.report['semantic_requests'] == result.report['vlm_calls'] == 2
    assert result.report['new_observations'] == 0
    assert client.requests[0]['obs'] is client.requests[1]['obs']
    diagnostics = result.attempts[0].log['diagnostics']
    failure = diagnostics['abstentions'][0] if partial else diagnostics
    assert failure['rejected_pixel_count'] == 70
    assert failure.get('pixels', failure.get('component_pixels')) == 14*44
    feedback = client.requests[1]['history'][0]['source_object_annotation']['geometry_diagnostics']
    x0, y0, _, _ = result.report['image_context']['bounds_xyxy']
    expected = np.asarray(failure['rejected_pixels_xy']) - [x0, y0]
    assert len(expected) == 12
    assert [sample['pixel'] for sample in feedback['rejected_samples']] == expected.tolist()
    assert all(sample['sample_kind'] == 'geometry_outlier' for sample in feedback['rejected_samples'])
    entry = feedback['failures'][0]
    assert entry['rejected_seed_pixels'] == []
    assert [sample['pixel'] for sample in entry['rejected_geometry_pixels']] == expected.tolist()
    if partial:
        assert [s['pixel'] for s in entry['component_seed_pixels']] == [[230-x0, 210-y0]]
        assert len(failure['source_pixels_xy']) == 14*44  # Whole component still abstains.
    assert [230-x0, 210-y0] not in expected.tolist()
    diagnostic = client.requests[1]['kwargs']['annotation_diagnostic']
    assert diagnostic.rejected_pixels == tuple(map(tuple, expected))
    diagnostic.validate_source(client.requests[1]['obs'])
    assert all(np.array_equal(array, saved) for array, saved in zip(arrays, before))
    assert all(attempt.log['raw_annotation'] == raw for attempt, raw in zip(result.attempts, client.replies))


def test_multiple_failure_samples_prioritize_real_geometry_and_share_twelve_point_cap():
    obs, reference, _ = source_scene()
    crop = crop_observation(obs, reference)
    source_reply = crop.restore_decision(reply(crop.observation, 'left_bad'))
    outside_a = [[240+x, 220] for x in range(8)]
    outside_b = [[247+x, 220] for x in range(8)]  # One duplicate across failures.
    diagnostics = {'frame_id': obs.frame_id, 'camera_id': obs.camera_id, 'abstentions': [
        {'failure': 'invalid_explicit_seed', 'seed_indices': [1], 'reasons': ['outside_region_or_hole']},
        {'failure': 'component_outside_fixed_envelope', 'seed_indices': [0],
         'rejected_pixels_xy': outside_a, 'rejected_pixel_count': 20},
        {'failure': 'component_outside_fixed_envelope', 'seed_indices': [0],
         'rejected_pixels_xy': outside_b, 'rejected_pixel_count': 24}]}
    before = copy.deepcopy(diagnostics)
    raw_before = source_reply.model_dump()
    feedback = _correction_feedback(crop, diagnostics, source_reply)
    x0, y0, _, _ = crop.bounds_xyxy
    expected = (np.array([[240+x, 220] for x in range(12)]) - [x0, y0]).tolist()
    assert [sample['pixel'] for sample in feedback['rejected_samples']] == expected
    assert all(sample['sample_kind'] == 'geometry_outlier' for sample in feedback['rejected_samples'])
    assert feedback['failures'][0]['rejected_seed_pixels']  # Invalid seed remains in text.
    assert feedback['failures'][1]['rejected_seed_pixels'] == []
    assert feedback['failures'][1]['component_seed_pixels']
    assert diagnostics == before and source_reply.model_dump() == raw_before


def test_seed_can_also_be_an_actual_outlier_without_conflicting_attribution():
    obs, reference, _ = source_scene()
    crop = crop_observation(obs, reference)
    source_reply = crop.restore_decision(reply(crop.observation, 'good'))
    seed = pixels(source_reply.groundings[1].points, obs.depth.shape)[0].tolist()
    diagnostics = {'frame_id': obs.frame_id, 'camera_id': obs.camera_id, 'abstentions': [
        {'failure': 'component_outside_fixed_envelope', 'seed_indices': [0],
         'rejected_pixels_xy': [seed], 'rejected_pixel_count': 1}]}
    feedback = _correction_feedback(crop, diagnostics, source_reply)
    failure = feedback['failures'][0]
    assert failure['component_seed_pixels'] == failure['rejected_geometry_pixels']
    assert 'does not classify' in failure['component_seed_scope']
    assert feedback['rejected_samples'][0]['sample_kind'] == 'geometry_outlier'


@pytest.mark.parametrize('fault', ['frame', 'camera', 'outside_crop'])
def test_outlier_feedback_rejects_wrong_source_or_coordinates_without_clipping(fault):
    obs, reference, _ = source_scene()
    crop = crop_observation(obs, reference)
    source_reply = crop.restore_decision(reply(crop.observation, 'good'))
    diagnostics = {'frame_id': obs.frame_id, 'camera_id': obs.camera_id,
        'failure': 'component_outside_fixed_envelope', 'rejected_pixels_xy': [[240, 220]],
        'rejected_pixel_count': 1}
    if fault == 'frame': diagnostics['frame_id'] += 1
    elif fault == 'camera': diagnostics['camera_id'] = 'other_camera'
    else: diagnostics['rejected_pixels_xy'] = [[crop.bounds_xyxy[0]-1, crop.bounds_xyxy[1]]]
    before = copy.deepcopy(diagnostics)
    with pytest.raises(ValueError):
        _correction_feedback(crop, diagnostics, source_reply)
    assert diagnostics == before
