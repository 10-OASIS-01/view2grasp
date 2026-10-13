"""Object-only measured ownership never grants contact or free-space evidence."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from view2grasp.geometry import unproject
from view2grasp.grounding import mask_rle
from view2grasp.mapping import DepthMap, FREE, OCCUPIED, UNKNOWN
from view2grasp.object_ownership import (ObjectOwnershipError, resolve_object_ownership,
                                       source_endpoint_retirement_mask)
from view2grasp.types import AgentDecision, GroundingProposal, Observation


def scene(end=20):
    depth = np.full((16, 64), 2.)
    depth[6:10, 2:end] = 1.
    obs = Observation(7, np.zeros((*depth.shape, 3), np.uint8), depth,
        np.diag([1000., 1000., 1.]), np.eye(4), np.zeros(7), np.array([.04, -.04]),
        12., (0., 55., .8))
    y, x = np.mgrid[:16, :64]
    xyz = unproject(np.stack([x, y], -1), depth, obs.K, obs.T_wc)
    mask = np.zeros_like(depth, bool)
    mask[6:10, 2:end] = True
    reference = xyz[6:10, 2:4].reshape(-1, 3).copy()
    return obs, xyz, mask, reference


def annotation(obs, mask, seeds=((2, 6),), *, kind='mask', negatives=()):
    h, w = mask.shape
    xy = lambda points: [(x/(w-1), y/(h-1)) for x, y in points]
    common = dict(role='object', target_id='mug', frame_id=obs.frame_id, camera_id=obs.camera_id)
    if kind == 'mask':
        region = GroundingProposal(kind=kind, mask_size=mask.shape, mask_rle=mask_rle(mask), **common)
    else:
        y, x = np.where(mask)
        box = (x.min()/(w-1), y.min()/(h-1), x.max()/(w-1), y.max()/(h-1))
        region = GroundingProposal(kind=kind, box=box,
            points=xy([(x.min(), y.min()), (x.max(), y.min()), (x.max(), y.max()), (x.min(), y.max())])
                if kind == 'polygon' else [], **common)
    return AgentDecision(action='observe', frame_id=obs.frame_id, camera_id=obs.camera_id,
        groundings=[region, GroundingProposal(kind='points', points=xy(seeds),
            negative_points=xy(negatives), **common)])


def resolve(obs, decision, reference, **kw):
    return resolve_object_ownership(obs, decision, reference, 'mug', .9, **kw)


@pytest.mark.parametrize('kind', ['mask', 'box', 'polygon'])
def test_returns_only_measured_seeded_points_with_source_metadata_and_no_mutation(kind):
    obs, xyz, mask, reference = scene()
    decision = annotation(obs, mask, kind=kind)
    before_decision = decision.model_dump()
    result = resolve(obs, decision, reference)
    assert np.array_equal(result.object_mask, mask)
    assert np.array_equal(result.object_points, xyz[mask])
    assert np.array_equal(result.source_pixels_xy, np.argwhere(mask)[:, ::-1])
    assert np.array_equal(result.source_depth_m, obs.depth[mask])
    assert result.source_frame_id == 7 and result.source_camera_id == 'activeview'
    assert result.report['anchor_pixels'] >= 4
    assert not result.report['grants_contact_permission']
    assert not result.report['grants_free_space']
    assert not result.report['map_modified']
    assert result.report['source_pose_only']
    assert decision.model_dump() == before_decision
    assert np.array_equal(reference, xyz[6:10, 2:4].reshape(-1, 3))
    with pytest.raises(ValueError):
        result.object_points[0, 0] = 100


def test_fixed_aabb_allows_connected_inner_surface_over_35mm_from_original_front():
    obs, xyz, mask, reference = scene(end=50)
    reference = np.vstack([reference, [.052, .052, 1.]])
    result = resolve(obs, annotation(obs, mask), reference)
    assert cKDTree(reference).query(xyz[6, 49])[0] > .035
    assert result.object_mask[6, 49]
    assert result.report['original_reference_sha256'] == resolve(
        obs, annotation(obs, mask, seeds=((40, 6),)), reference).report['original_reference_sha256']


def test_complete_component_outside_aabb_is_rejected_not_clipped_or_incrementally_grown():
    obs, xyz, mask, reference = scene(end=39)
    first = resolve(obs, annotation(obs, mask), reference)
    assert first.object_mask[6, 38]  # Original max x=.003 + .035.
    obs2, _, mask2, _ = scene(end=40)
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs2, annotation(obs2, mask2), reference)
    assert caught.value.diagnostics['failure'] == 'component_outside_fixed_envelope'
    assert caught.value.diagnostics['rejected_pixel_count'] == 4
    assert caught.value.diagnostics['component_pixels'] == int(mask2.sum())
    assert first.report['fixed_original_aabb_m'][1][0] == pytest.approx(.038)


def neighbor_scene():
    obs, xyz, mask, reference = scene(end=8)
    depth = obs.depth.copy()
    depth[6:10, 8:12] = 1.01
    obs = replace(obs, depth=depth)
    y, x = np.mgrid[:16, :64]
    xyz = unproject(np.stack([x, y], -1), depth, obs.K, obs.T_wc)
    mask[6:10, 8:12] = True
    return obs, xyz, mask, reference


@pytest.mark.parametrize('seeds', [((2, 6), (9, 6)), ((9, 6),)])
def test_seeded_10mm_neighbor_cannot_borrow_other_components_anchors(seeds):
    obs, _, mask, reference = neighbor_scene()
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, annotation(obs, mask, seeds=seeds), reference)
    assert caught.value.diagnostics['failure'] == 'unanchored_seeded_component'


def test_unseeded_neighbor_in_region_is_not_owned_or_returned():
    obs, _, mask, reference = neighbor_scene()
    result = resolve(obs, annotation(obs, mask), reference)
    assert result.object_mask[:, :8].sum() == 24
    assert not result.object_mask[:, 8:].any()
    assert result.report['unselected_valid_region_pixels'] == 16


def test_all_selected_components_have_anchors_and_four_total_are_required():
    obs, xyz, mask, _ = neighbor_scene()
    reference = np.vstack([xyz[6, 2:5], xyz[6, 9]])
    result = resolve(obs, annotation(obs, mask, seeds=((2, 6), (9, 6))), reference)
    assert len(result.report['selected_components']) == 2
    assert all(c['original_anchor_pixels'] for c in result.report['selected_components'])
    one = np.zeros_like(mask)
    one[6, 2] = True
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, annotation(obs, one, seeds=((2, 6), (2, 6))), reference)
    assert caught.value.diagnostics['failure'] == 'insufficient_original_overlap'
    assert caught.value.diagnostics['anchor_pixels'] == 1


@pytest.mark.parametrize('fault', ['stale', 'camera', 'target', 'source', 'part',
                                  'missing_seed', 'missing_region', 'two_regions', 'proposal_frame'])
def test_semantic_identity_and_explicit_object_seeds_are_required(fault):
    obs, _, mask, reference = scene()
    decision = annotation(obs, mask)
    if fault == 'stale': decision = decision.model_copy(update={'frame_id': 6})
    elif fault == 'camera': decision = decision.model_copy(update={'camera_id': 'other'})
    elif fault == 'missing_seed': decision.groundings = decision.groundings[:1]
    elif fault == 'missing_region': decision.groundings = decision.groundings[1:]
    elif fault == 'two_regions': decision.groundings.append(decision.groundings[0])
    else:
        update = {'target': {'target_id': 'other'}, 'source': {'source': 'rgbd'},
                  'part': {'role': 'grasp_part'}, 'proposal_frame': {'frame_id': 6}}[fault]
        decision.groundings[0] = decision.groundings[0].model_copy(update=update)
    with pytest.raises(ObjectOwnershipError):
        resolve(obs, decision, reference)


@pytest.mark.parametrize('action', ['abort', 'grasp'])
def test_valid_regions_cannot_turn_abort_or_grasp_into_ownership_authorization(action):
    obs, _, mask, reference = scene()
    decision = annotation(obs, mask)
    decision = AgentDecision.model_validate({**decision.model_dump(), 'action': action})
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, decision, reference)
    assert caught.value.diagnostics['failure'] == 'annotation_action_required'


def test_rgb_and_depth_must_share_the_source_pixel_coordinate_system():
    obs, _, mask, reference = scene()
    decision = annotation(obs, mask)
    obs = replace(obs, rgb=obs.rgb[:-1])
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, decision, reference)
    assert caught.value.diagnostics['failure'] == 'invalid_calibration'


@pytest.mark.parametrize('fault', ['outside', 'hole', 'negative', 'nan', 'support', 'self', 'behind_robot'])
def test_invalid_seed_is_never_relocated_to_a_nearby_valid_pixel(fault):
    obs, _, mask, reference = scene()
    decision = annotation(obs, mask, kind='polygon')
    own = np.full(obs.depth.shape, 10.)
    if fault == 'outside': decision = annotation(obs, mask, seeds=((1, 6),))
    elif fault == 'hole':
        decision.groundings[0].holes = [[(1/63, 5/15), (4/63, 5/15), (4/63, 8/15), (1/63, 8/15)]]
    elif fault == 'negative': decision = annotation(obs, mask, negatives=((2, 6),))
    elif fault in ('self', 'behind_robot'): own[6, 2] = 1. if fault == 'self' else .9
    else:
        depth = obs.depth.copy()
        depth[6, 2] = np.nan if fault == 'nan' else .905
        obs = replace(obs, depth=depth)
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, decision, reference, robot_depth=own)
    assert caught.value.diagnostics['failure'] == 'invalid_explicit_seed'


@pytest.mark.parametrize('gap', ['hole', 'negative', 'self', 'support', 'invalid'])
def test_excluded_pixels_cannot_bridge_a_seeded_unanchored_surface(gap):
    obs, _, mask, reference = scene(end=25)
    decision = annotation(obs, mask, seeds=((2, 6), (22, 6)), kind='polygon')
    own = np.full(obs.depth.shape, 10.)
    if gap == 'hole':
        decision.groundings[0].holes = [[(10/63, 5/15), (12/63, 5/15), (12/63, 10/15), (10/63, 10/15)]]
    elif gap == 'negative': decision = annotation(obs, mask, seeds=((2, 6), (22, 6)), negatives=((11, 8),))
    elif gap == 'self': own[6:10, 11] = .9
    else:
        depth = obs.depth.copy()
        depth[6:10, 11] = .905 if gap == 'support' else np.nan
        obs = replace(obs, depth=depth)
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, decision, reference, robot_depth=own)
    assert caught.value.diagnostics['failure'] == 'unanchored_seeded_component'


def test_diagonal_pixels_do_not_connect_without_a_four_neighbor_chain():
    obs, _, mask, reference = scene()
    mask[:] = False
    mask[6:8, 2:4] = True
    mask[8:10, 4:6] = True
    # Separate island is within the 35 mm envelope, but farther than 4 mm from
    # every fixed reference; diagonal image adjacency cannot join the islands.
    reference = np.array([[.0, .006, 1.], [.0, .007, 1.], [.0, .008, 1.], [.0, .009, 1.]])
    with pytest.raises(ObjectOwnershipError) as caught:
        resolve(obs, annotation(obs, mask, seeds=((2, 6), (5, 9))), reference)
    assert caught.value.diagnostics['failure'] == 'unanchored_seeded_component'


def test_exact_retirement_preserves_6_2mm_neighbor_and_other_view_representatives():
    obs, _, mask, reference = scene()
    result = resolve(obs, annotation(obs, mask), reference)
    owned = result.object_points[0]
    neighbor = owned + [0, 0, .0062]
    other_view = owned + [0, 0, .00001]
    numerically_same = owned + [0, 0, 1e-12]
    surface = np.array([owned, neighbor, other_view, numerically_same])
    before = surface.copy()
    retire = source_endpoint_retirement_mask(surface, result, obs)
    assert retire.tolist() == [True, False, False, True]
    assert np.array_equal(surface, before)
    legacy = DepthMap(.005)
    legacy.surface = surface.copy()
    legacy.tree = cKDTree(legacy.surface)
    legacy.remove_target_endpoints(result.object_points)
    assert not len(legacy.surface)  # Existing 6.5 mm deletion loses the neighbor.
    assert not source_endpoint_retirement_mask(surface, result, obs, table_z=1.).any()


@pytest.mark.parametrize('change', ['frame', 'camera', 'depth', 'rgb', 'calibration'])
def test_retirement_rejects_changed_or_wrong_source_evidence(change):
    obs, _, mask, reference = scene()
    result = resolve(obs, annotation(obs, mask), reference)
    if change == 'frame': changed = replace(obs, frame_id=8)
    elif change == 'camera': changed = replace(obs, camera_id='other')
    elif change == 'depth': changed = replace(obs, depth=obs.depth + 1e-10)
    elif change == 'rgb': changed = replace(obs, rgb=obs.rgb + 1)
    else:
        K = obs.K.copy(); K[0, 0] += .001
        changed = replace(obs, K=K)
    with pytest.raises(ValueError, match='unchanged ownership source'):
        source_endpoint_retirement_mask(result.object_points, result, changed)


def test_retirement_selection_preserves_depth_views_and_never_carves_behind_object():
    obs, _, mask, reference = scene()
    result = resolve(obs, annotation(obs, mask), reference)
    map_ = DepthMap(.005)
    map_.integrate(obs, np.empty((0, 3)))
    depth_before = obs.depth.copy()
    views_before = list(map_.views)
    ray = np.array([.01, .008, 1.])
    query = ray[None] * np.array([.98, 1.02, 1.])[..., None]
    assert map_.classify(query).tolist() == [FREE, UNKNOWN, OCCUPIED]
    retire = source_endpoint_retirement_mask(map_.surface, result, obs)
    # Selection is pure. Simulate a future caller applying exactly that subset;
    # the module does not install a certificate or alter inverse-sensor images.
    assert map_.classify(query).tolist() == [FREE, UNKNOWN, OCCUPIED]
    map_.surface = map_.surface[~retire]
    map_.tree = cKDTree(map_.surface)
    map_._refresh_non_target()
    assert map_.classify(query).tolist() == [FREE, UNKNOWN, UNKNOWN]
    assert map_.views == views_before and map_.views[0] is obs
    assert np.array_equal(obs.depth, depth_before)
