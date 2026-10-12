"""Fresh initial semantics are explicit, bounded, and never self-anchored."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from view2grasp.initial_object_reference import annotate_initial_object_reference
from view2grasp.object_ownership import (ObjectOwnershipError, resolve_object_ownership,
                                       source_endpoint_retirement_mask)
from test_object_ownership import scene, neighbor_scene, annotation


class ScriptedVLM:
    def __init__(self, replies):
        self.replies = replies
        self.requests = []
        self.calls, self.seconds = 3, .5

    def decide(self, obs, instruction, history, **options):
        index = len(self.requests)
        assert index < len(self.replies), 'unexpected semantic request beyond supplied budget'
        self.calls += 1
        self.seconds += .1
        self.requests.append((obs, instruction, copy.deepcopy(history), options))
        history[0]['initial_object_reference']['client_mutation'] = True
        reply = self.replies[index]
        if isinstance(reply, Exception): raise reply
        return reply(obs) if callable(reply) else reply.model_copy(deep=True)


def initialize(obs, reference, replies, *, side_obs=None, robot_depth=None):
    client = ScriptedVLM(replies)
    result = annotate_initial_object_reference(client, obs,
        replace(obs, frame_id=obs.frame_id+1) if side_obs is None else side_obs,
        reference, 'mug', .9, robot_depth=robot_depth, target_description='mug')
    return result, client


def test_fresh_seeded_unanchored_surface_initializes_but_is_not_a_later_anchor_pass():
    obs, xyz, mask, reference = neighbor_scene()
    decision = annotation(obs, mask, seeds=((2, 6), (9, 6)))
    arrays = [obs.rgb, obs.depth, obs.K, obs.T_wc, obs.q, obs.gripper_q, reference, mask]
    before = [a.copy() for a in arrays]
    raw_before = decision.model_dump()
    result, client = initialize(obs, reference, [decision])
    owned = result.ownership
    assert owned is not None and np.array_equal(owned.object_mask, mask)
    assert np.array_equal(owned.object_points, xyz[mask])
    assert owned.report['novel_semantic_components'] == 1
    assert owned.report['side_anchor_pixels'] >= 4
    assert [c['side_anchor_pixels'] for c in owned.report['selected_components']][1] == 0
    assert all(c['membership_authority'] == 'fresh_initial_vlm_semantics'
               for c in owned.report['selected_components'])
    assert owned.report['later_component_anchor_gate_applied'] is False
    assert not owned.report['grants_contact_permission'] and not owned.report['grants_free_space']
    assert owned.abstained_source_points.shape == (0, 3)
    # This geometry could also be an explicitly mislabeled separate neighbor.
    # Initial VLM authority is the assumption, not a geometric identity proof.
    assert 'neighboring object was mislabeled' in owned.report['semantic_identity_limit']
    with pytest.raises(ObjectOwnershipError) as failure:
        resolve_object_ownership(obs, decision, reference, 'mug', .9)
    assert failure.value.diagnostics['failure'] == 'unanchored_seeded_component'
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before))
    assert decision.model_dump() == raw_before
    for a in (owned.object_mask, owned.object_points, owned.source_pixels_xy, owned.source_depth_m,
              owned.abstained_source_points, owned.abstained_source_pixels_xy):
        assert not a.flags.writeable
    request = client.requests[0]
    assert request[3]['force_observe'] and request[3]['annotation_roles'] == ('object',)
    assert result.report['vlm_calls'] == result.report['semantic_requests'] == 1
    assert 'client_mutation' not in str(result.report)


def test_unseeded_ten_mm_neighbor_is_not_flooded_owned_or_exactly_retired():
    obs, xyz, mask, reference = neighbor_scene()
    result, _ = initialize(obs, reference, [annotation(obs, mask)])
    owned = result.ownership
    assert owned is not None and owned.object_mask.sum() == 24
    assert not owned.object_mask[:, 8:].any()
    surface = np.vstack([xyz[mask], owned.object_points[0]+[0, 0, .0062]])
    retired = source_endpoint_retirement_mask(surface, owned, obs)
    assert retired.sum() == 24 and not retired[-1]
    assert not source_endpoint_retirement_mask(xyz[6:10, 8:12].reshape(-1, 3), owned, obs).any()


def test_novel_surface_alone_cannot_supply_its_own_side_anchors():
    obs, _, mask, reference = neighbor_scene()
    decision = annotation(obs, mask, seeds=((9, 6),))
    result, client = initialize(obs, reference, [decision, decision])
    assert result.ownership is None and len(client.requests) == 2
    assert all(a.log['diagnostics']['side_anchor_pixels'] == 0 for a in result.attempts)
    assert all(a.log['diagnostics']['failure'] == 'insufficient_original_overlap' for a in result.attempts)


def test_duplicate_seeds_do_not_multiply_one_actual_side_anchor_into_four():
    side, _, mask, reference = neighbor_scene()
    depth = side.depth.copy()
    depth[6:10, 2:8] = 2.
    depth[6, 2] = 1.
    source = replace(side, frame_id=side.frame_id-1, depth=depth)
    decision = annotation(source, mask, seeds=((2, 6),)*4+((9, 6),))
    result, _ = initialize(source, reference, [decision, decision], side_obs=side)
    assert result.ownership is None
    assert result.attempts[-1].log['diagnostics']['side_anchor_pixels'] == 1


@pytest.mark.parametrize('fault', ['time', 'q', 'gripper', 'base', 'torso', 'same_frame',
                                 'hidden_side_point', 'wrong_side_image', 'nonfinite_q'])
def test_static_epoch_and_measured_side_reference_are_required_before_any_call(fault):
    source, _, _, reference = scene()
    side = replace(source, frame_id=source.frame_id+1)
    if fault == 'time': side = replace(side, timestamp=source.timestamp+.001)
    elif fault == 'q': side = replace(side, q=source.q+.001)
    elif fault == 'gripper': side = replace(side, gripper_q=source.gripper_q+.001)
    elif fault == 'base': side = replace(side, base_pose=np.zeros(3))
    elif fault == 'torso': side = replace(side, torso_q=np.zeros(1))
    elif fault == 'same_frame': side = source
    elif fault == 'hidden_side_point': reference = reference + [0, 0, .001]
    elif fault == 'wrong_side_image': side = replace(side, depth=side.depth+.01)
    else: side = replace(side, q=np.full(7, np.nan))
    result, client = initialize(source, reference, [], side_obs=side)
    assert not client.requests and result.ownership is None
    assert result.report['failure'] == 'invalid_initial_evidence'
    assert result.report['vlm_calls'] == result.report['semantic_requests'] == 0


@pytest.mark.parametrize('fault', ['support', 'self', 'negative', 'hole', 'invalid_depth', 'outside_region'])
def test_any_invalid_seed_atomically_rejects_otherwise_good_components(fault):
    side, _, mask, reference = scene(end=25)
    source = replace(side, frame_id=side.frame_id-1)
    own = np.full(source.depth.shape, 10.)
    decision = annotation(source, mask, seeds=((2, 6), (20, 8)), kind='polygon')
    if fault == 'self': own[8, 20] = .9
    elif fault == 'negative': decision = annotation(source, mask, seeds=((2, 6), (20, 8)), negatives=((20, 8),))
    elif fault == 'hole':
        decision.groundings[0].holes = [[(19/63, 7/15), (21/63, 7/15), (21/63, 9/15), (19/63, 9/15)]]
    elif fault == 'outside_region': decision = annotation(source, mask, seeds=((2, 6), (1, 6)))
    else:
        depth = source.depth.copy(); depth[8, 20] = .905 if fault == 'support' else np.nan
        source = replace(source, depth=depth)
    result, client = initialize(source, reference, [decision, decision], side_obs=side, robot_depth=own)
    assert result.ownership is None and len(client.requests) == 2
    assert all(a.ownership is None for a in result.attempts)
    expected = {'support': 'support', 'self': 'robot_occluded_or_unverified', 'negative': 'explicit_negative',
                'hole': 'outside_region_or_hole', 'invalid_depth': 'invalid_depth',
                'outside_region': 'outside_region_or_hole'}[fault]
    assert expected in result.attempts[-1].log['diagnostics']['reasons']


def test_full_outlying_component_rejected_then_fresh_reply_not_merged_and_feedback_is_exact():
    obs, _, mask, reference = scene(end=44)
    first = annotation(obs, mask)
    second_mask = mask.copy(); second_mask[:, 8:] = False
    second = annotation(obs, second_mask)
    result, client = initialize(obs, reference, [first, second])
    assert result.attempts[0].ownership is None
    failure = result.attempts[0].log['diagnostics']
    assert failure['component_pixels'] == 168 and failure['rejected_pixel_count'] == 20
    assert len(failure['rejected_pixels_xy']) == 12
    assert result.ownership.object_mask.sum() == 24
    assert np.array_equal(result.ownership.object_mask, second_mask)
    feedback = client.requests[1][2][0]['initial_object_reference']['geometry_diagnostics']
    assert [s['pixel'] for s in feedback['rejected_samples']] == failure['rejected_pixels_xy']
    assert [2, 6] not in [s['pixel'] for s in feedback['rejected_samples']]
    diagnostic = client.requests[1][3]['annotation_diagnostic']
    diagnostic.validate_source(client.requests[1][0])
    assert diagnostic.rejected_pixels == tuple(map(tuple, failure['rejected_pixels_xy']))
    assert client.requests[0][0] is client.requests[1][0]
    assert not result.report['reply_results_merged'] and result.report['new_observations'] == 0


def test_one_correction_reports_all_invalid_seeds_and_never_accepts_a_valid_subset():
    obs, _, mask, reference = scene(end=25)
    first = annotation(obs, mask, seeds=((2, 6), (1, 6), (30, 6)))
    second = annotation(obs, mask)
    result, client = initialize(obs, reference, [first, second])
    assert result.attempts[0].ownership is None
    diagnostics = result.attempts[0].log['diagnostics']
    assert diagnostics['invalid_seed_count'] == 2
    feedback = client.requests[1][2][0]['initial_object_reference']['geometry_diagnostics']
    assert [f['seed_indices'] for f in feedback['failures']] == [[1], [2]]
    assert [s['pixel'] for s in feedback['rejected_samples']] == [[1, 6], [30, 6]]
    assert all(s['sample_kind'] == 'invalid_seed' for s in feedback['rejected_samples'])
    assert result.ownership is not None and len(client.requests) == 2
    assert not result.report['reply_results_merged']


def test_bad_third_component_does_not_return_good_anchored_and_novel_components():
    obs, _, mask, reference = neighbor_scene()
    depth = obs.depth.copy(); depth[12:14, 50:52] = 1.
    obs = replace(obs, depth=depth)
    mask[12:14, 50:52] = True
    decision = annotation(obs, mask, seeds=((2, 6), (9, 6), (50, 12)))
    abort = decision.model_copy(update={'action': 'abort'})
    result, client = initialize(obs, reference, [decision, abort])
    assert result.ownership is None and len(client.requests) == 2
    assert all(a.ownership is None for a in result.attempts)
    assert result.attempts[0].log['diagnostics']['failure'] == 'component_outside_fixed_envelope'


@pytest.mark.parametrize('fault', ['identity', 'role', 'source', 'frame', 'vertices_only', 'grasp', 'abort', 'transport'])
def test_non_geometry_protocol_failure_does_not_open_an_extra_transaction(fault):
    obs, _, mask, reference = scene()
    decision = annotation(obs, mask, kind='polygon')
    if fault in ('identity', 'role', 'source'):
        key, value = {'identity': ('target_id', 'other'), 'role': ('role', 'grasp_part'),
                      'source': ('source', 'rgbd')}[fault]
        decision.groundings[1] = decision.groundings[1].model_copy(update={key: value})
    elif fault == 'frame': decision = decision.model_copy(update={'frame_id': obs.frame_id+1})
    elif fault == 'vertices_only': decision.groundings = decision.groundings[:1]
    elif fault in ('grasp', 'abort'): decision = decision.model_copy(update={'action': fault})
    else: decision = RuntimeError('bounded provider retries exhausted')
    result, client = initialize(obs, reference, [decision])
    assert result.ownership is None and len(client.requests) == 1


def test_nonzero_crop_coordinates_restore_to_exact_source_pixels():
    from test_object_annotation import source_scene, reply
    obs, reference, xyz = source_scene()
    result, client = initialize(obs, reference, [lambda crop: reply(crop, 'good')])
    x0, y0, x1, y1 = result.report['image_context']['bounds_xyxy']
    assert x0 > 0 and y0 > 0
    assert client.requests[0][0].depth.shape == (y1-y0, x1-x0)
    assert np.array_equal(result.ownership.object_points, xyz[210:224, 230:244].reshape(-1, 3))
    assert result.ownership.source_pixels_xy.min(axis=0).tolist() == [230, 210]
