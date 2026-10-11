"""Opt-in independent ownership never repairs or retires abstained evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from view2grasp.object_ownership import (ObjectOwnershipError, resolve_object_ownership,
                                       source_endpoint_retirement_mask)
from view2grasp.types import AgentDecision, Observation
from test_object_ownership import scene, annotation, resolve, neighbor_scene


def partial(obs, decision, reference, **kwargs):
    return resolve(obs, decision, reference, allow_partial_seed_components=True, **kwargs)


def test_partial_is_opt_in_and_invalid_seed_is_logged_not_moved():
    obs, _, mask, reference = scene()
    decision = annotation(obs, mask, seeds=((2, 6), (1, 6)))
    before = decision.model_dump()
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, decision, reference)
    assert caught.value.diagnostics['failure'] == 'invalid_explicit_seed'
    result = partial(obs, decision, reference)
    assert np.array_equal(result.object_mask, mask)
    assert result.report['partial'] is True
    abstained = result.report['abstentions']
    assert len(abstained) == 1
    assert abstained[0]['seed_indices'] == [1]
    assert abstained[0]['source_pixels_xy'] == [[1, 6]]
    assert abstained[0]['reasons'] == ['outside_region_or_hole']
    assert result.abstained_source_pixels_xy.tolist() == [[1, 6]]
    assert decision.model_dump() == before


def test_independent_10mm_neighbor_abstains_whole_and_remains_unretired():
    obs, xyz, mask, reference = neighbor_scene()
    result = partial(obs, annotation(obs, mask, seeds=((2, 6), (9, 6), (10, 8))), reference)
    assert result.object_mask.sum() == 24 and not result.object_mask[:, 8:].any()
    item = result.report['abstentions'][0]
    assert item['failure'] == 'unanchored_seeded_component'
    assert item['seed_indices'] == [1, 2]
    assert item['pixels'] == len(item['source_pixels_xy']) == 16
    assert result.abstained_source_pixels_xy.shape == (16, 2)
    neighbors = xyz[6:10, 8:12].reshape(-1, 3)
    close_neighbor = result.object_points[0] + [0, 0, .0062]
    surface = np.vstack([result.object_points, neighbors, close_neighbor])
    retire = source_endpoint_retirement_mask(surface, result, obs)
    assert retire[:24].all() and not retire[24:].any()
    assert not result.report['grants_contact_permission']
    with pytest.raises(ValueError): result.abstained_source_pixels_xy[0] = 0
    with pytest.raises(ValueError): result.abstained_source_points[0] = 0


def test_partial_needs_four_anchors_per_component_not_borrowed_global_anchors():
    obs, _, mask, reference = scene()
    depth = obs.depth.copy(); depth[12, 20] = 1.
    obs = replace(obs, depth=depth)
    mask[12, 20] = True
    reference = np.vstack([reference, [.020, .012, 1.]])
    decision = annotation(obs, mask, seeds=((2, 6), (20, 12)))
    # Preserve the prior strict-mode contract: each component needs >=1 anchor
    # and there must be >=4 total. Partial mode has the stronger per-part gate.
    assert resolve(obs, decision, reference).object_mask[12, 20]
    result = partial(obs, decision, reference)
    assert not result.object_mask[12, 20]
    assert result.report['abstentions'][0]['failure'] == 'insufficient_component_anchors'
    assert result.report['abstentions'][0]['original_anchor_pixels'] == 1


@pytest.mark.parametrize('end', [40, 44])
def test_out_of_envelope_tail_abstains_entire_component_without_clipping(end):
    obs, _, mask, reference = scene(end=end)
    depth = obs.depth.copy(); depth[12:14, 2:4] = 1.
    obs = replace(obs, depth=depth)
    mask[12:14, 2:4] = True
    reference = np.vstack([reference, [[x*.001, y*.001, 1.] for y in (12, 13) for x in (2, 3)]])
    decision = annotation(obs, mask, seeds=((2, 6), (2, 12)))
    arrays = [obs.rgb, obs.depth, obs.K, obs.T_wc, reference, mask]
    before = [array.copy() for array in arrays]
    before_decision = decision.model_dump()
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, decision, reference)
    strict = caught.value.diagnostics
    result = partial(obs, decision, reference)
    assert result.object_mask.sum() == 4
    assert not result.object_mask[6:10].any()
    item = result.report['abstentions'][0]
    assert item['failure'] == 'component_outside_fixed_envelope'
    assert item['outside_fixed_envelope_pixels'] == item['rejected_pixel_count'] == 4*(end-39)
    assert len(item['source_pixels_xy']) == 4*(end-2)
    # Diagnostics preserve exactly the strict resolver's real outliers, with
    # a bounded sample and an untruncated count. The legal seed is not one.
    assert item['rejected_pixels_xy'] == strict['rejected_pixels_xy']
    assert item['rejected_pixel_count'] == strict['rejected_pixel_count']
    assert len(item['rejected_pixels_xy']) == min(12, 4*(end-39))
    assert all(39 <= x < end and 6 <= y < 10 for x, y in item['rejected_pixels_xy'])
    assert [2, 6] not in item['rejected_pixels_xy']
    assert result.abstained_source_pixels_xy.shape == (4*(end-2), 2)
    assert not source_endpoint_retirement_mask(result.abstained_source_points, result, obs).any()
    assert all(np.array_equal(array, saved) for array, saved in zip(arrays, before))
    assert decision.model_dump() == before_decision
    with pytest.raises(ValueError): result.object_mask[6, 2] = True


@pytest.mark.parametrize('fault', ['support', 'self', 'negative', 'hole', 'invalid_depth'])
def test_bad_seed_reasons_and_raw_source_pixels_survive_partial_acceptance(fault):
    obs, _, mask, reference = scene(end=25)
    own = np.full(obs.depth.shape, 10.)
    decision = annotation(obs, mask, seeds=((2, 6), (20, 8)), kind='polygon')
    if fault == 'self': own[8, 20] = .9
    elif fault == 'negative': decision = annotation(obs, mask, seeds=((2, 6), (20, 8)), negatives=((20, 8),))
    elif fault == 'hole':
        decision.groundings[0].holes = [[(19/63, 7/15), (21/63, 7/15), (21/63, 9/15), (19/63, 9/15)]]
    else:
        depth = obs.depth.copy(); depth[8, 20] = .905 if fault == 'support' else np.nan
        obs = replace(obs, depth=depth)
    result = partial(obs, decision, reference, robot_depth=own)
    assert not result.object_mask[8, 20]
    reasons = result.report['abstentions'][0]['reasons']
    assert {'support': 'support', 'self': 'robot_occluded_or_unverified', 'negative': 'explicit_negative',
            'hole': 'outside_region_or_hole', 'invalid_depth': 'invalid_depth'}[fault] in reasons
    assert result.abstained_source_pixels_xy.tolist() == [[20, 8]]
    assert np.isfinite(result.abstained_source_points).all()


def test_partial_all_bad_reply_still_fails_with_full_abstention_reasons():
    obs, _, mask, reference = neighbor_scene()
    decision = annotation(obs, mask, seeds=((9, 6), (0, 0)))
    with pytest.raises(ObjectOwnershipError) as caught:
        partial(obs, decision, reference)
    assert caught.value.diagnostics['failure'] == 'no_accepted_components'
    assert {a['failure'] for a in caught.value.diagnostics['abstentions']} == {
        'invalid_explicit_seed', 'unanchored_seeded_component'}


def test_abstained_numerical_alias_has_priority_over_accepted_retirement():
    obs, _, mask, reference = scene()
    result = partial(obs, annotation(obs, mask), reference)
    # Deliberately supply conflicting numerical evidence to the pure selection
    # helper: a coordinate present in both sets must remain an obstacle.
    protected = result.object_points[[0]].copy()
    result = replace(result, abstained_source_points=protected)
    retirement = source_endpoint_retirement_mask(result.object_points, result, obs)
    assert not retirement[0] and retirement[1:].all()


def test_unchanged_actual_saved_reply_remains_strict_failure_but_partial_has_two_whole_components():
    fixture = np.load(Path(__file__).parent/'fixtures/mug_object_ownership_reply.npz')
    metadata = json.loads(str(fixture['metadata']))
    raw = str(fixture['actual_response_content'])
    assert hashlib.sha256(raw.encode()).hexdigest() == metadata['actual_response_content_sha256']
    assert hashlib.sha256(fixture['rgb'].tobytes()).hexdigest() == metadata['source_crop_rgb_sha256']
    decision = AgentDecision.model_validate_json(raw)
    assert decision.model_dump() == AgentDecision.model_validate_json(str(fixture['annotation_crop'])).model_dump()
    before = decision.model_dump()
    obs = Observation(1, fixture['rgb'], fixture['depth'], fixture['K'], fixture['T_wc'],
        np.zeros(7), np.array([.04, -.04]), metadata['source_timestamp'], (0., 80., 1.3))
    args = (obs, decision, fixture['reference'], 'mug', float(fixture['support_z']), fixture['robot_depth'])
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve_object_ownership(*args)
    assert caught.value.diagnostics['failure'] == 'invalid_explicit_seed'
    result = resolve_object_ownership(*args, allow_partial_seed_components=True)
    assert result.report['partial'] is True
    assert [(c['pixels'], c['original_anchor_pixels'], c['seed_indices']) for c in result.report['selected_components']] == [
        (131, 18, [0, 1]), (327, 212, [2, 5])]
    assert len(result.object_points) == 458
    abstentions = result.report['abstentions']
    assert [(a['failure'], a['seed_indices']) for a in abstentions] == [
        ('invalid_explicit_seed', [3]), ('unanchored_seeded_component', [4])]
    assert abstentions[0]['reasons'] == ['support']
    x0, y0, _, _ = metadata['crop_bounds_xyxy']
    assert (result.abstained_source_pixels_xy + [x0, y0]).tolist() == [[229, 303], [242, 303]]
    # Saved obstacle endpoint coordinates are assertion-only diagnostics. They
    # are never supplied to the resolver, model, region or seed proposals.
    endpoints_xy = np.array([[264, 290], [252, 294], [254, 294], [262, 294],
                             [256, 296], [258, 296], [266, 296], [264, 298]]) - [x0, y0]
    assert result.object_mask[endpoints_xy[:, 1], endpoints_xy[:, 0]].all()
    assert not source_endpoint_retirement_mask(result.abstained_source_points, result, obs).any()
    assert decision.model_dump() == before
    assert metadata['episode_success_claimed'] is False and metadata['manually_corrected_annotation'] is False
    json.dumps(result.report)  # Full reasons/source pixels remain serializable.
