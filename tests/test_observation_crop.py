from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from view2grasp.geometry import project, unproject
from view2grasp.grounding import mask_rle, pixels, proposal_mask
from view2grasp.observation_crop import crop_observation
from view2grasp.types import AgentDecision, GroundingProposal, Observation


def observation(shape=(240, 320)):
    h, w = shape
    v, u = np.mgrid[:h, :w]
    rgb = np.stack([u % 256, v % 256, (u + v) % 256], -1).astype(np.uint8)
    K = np.array([[300., 2., (w - 1) / 2], [0., 290., (h - 1) / 2], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [13, -27, 9], degrees=True).as_matrix()
    T[:3, 3] = [.2, -.4, .6]
    return Observation(17, rgb, 1 + v / 1000 + u / 2000, K, T,
        np.arange(7.) / 10, np.array([.02, -.02]), 12.5, (30., 60., 1.3),
        'wrist', np.array([.1, .2, .3]), np.array([.5]))


def reference(obs, uv=((120, 100), (145, 116))):
    return unproject(uv, [1., 1.01], obs.K, obs.T_wc)


def test_crop_preserves_original_pixels_calibration_and_proprioception():
    obs = observation()
    self_depth = obs.depth + .4
    crop = crop_observation(obs, reference(obs), robot_depth=self_depth)
    x0, y0, x1, y1 = crop.bounds_xyxy
    out = crop.observation
    assert out.depth.shape == (128, 128)
    assert np.array_equal(out.rgb, obs.rgb[y0:y1, x0:x1])
    assert np.array_equal(out.depth, obs.depth[y0:y1, x0:x1])
    assert np.array_equal(crop.robot_depth, self_depth[y0:y1, x0:x1])
    assert not np.shares_memory(out.rgb, obs.rgb)
    assert not np.shares_memory(out.depth, obs.depth)
    assert out.frame_id == obs.frame_id and out.camera_id == obs.camera_id
    assert out.timestamp == obs.timestamp and out.view == obs.view
    for name in ('T_wc', 'q', 'gripper_q', 'base_pose', 'torso_q'):
        assert np.array_equal(getattr(out, name), getattr(obs, name))
    local_uv = np.array([[0, 0], [127, 127], [53, 76]])
    depths = out.depth[local_uv[:, 1], local_uv[:, 0]]
    assert np.allclose(unproject(local_uv, depths, out.K, out.T_wc),
        unproject(local_uv + [x0, y0], depths, obs.K, obs.T_wc), atol=1e-12)
    assert np.array_equal(obs.K, observation().K)
    report = json.loads(json.dumps(crop.provenance()))
    assert report['source_frame_id'] == 17 and report['source_camera_id'] == 'wrist'
    assert report['new_observation'] is False and report['resized'] is False
    assert 'not_current_visibility' in report['reference_use']


def test_metric_margin_contains_new_surfaces_under_rotated_camera_projection():
    obs = observation()
    source = reference(obs)
    crop = crop_observation(obs, source, min_size=2, context_pixels=0)
    rng = np.random.default_rng(31)
    offsets = rng.normal(size=(1000, 3))
    offsets *= .035 / np.linalg.norm(offsets, axis=1, keepdims=True)
    novel = np.vstack([p + offsets for p in source])
    uv, _ = project(novel, obs.K, obs.T_wc)
    x0, y0, x1, y1 = crop.bounds_xyxy
    assert np.all(uv >= [x0, y0]) and np.all(uv <= [x1 - 1, y1 - 1])


@pytest.mark.parametrize('uv', [(1., 1.), (318., 238.), (1., 238.), (318., 1.)])
def test_edge_crop_shifts_inward_without_losing_minimum_size(uv):
    obs = observation()
    crop = crop_observation(obs, unproject([uv], [1.], obs.K, obs.T_wc))
    assert crop.observation.depth.shape == (128, 128)
    x0, y0, x1, y1 = crop.bounds_xyxy
    assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240
    assert x0 <= uv[0] <= x1 - 1 and y0 <= uv[1] <= y1 - 1


def test_small_source_is_clipped_without_resizing_and_empty_depth_stays_empty():
    obs = observation((64, 90))
    obs.depth[:] = np.nan
    crop = crop_observation(obs, unproject([[40, 30]], [1.], obs.K, obs.T_wc))
    assert crop.bounds_xyxy == (0, 0, 90, 64)
    assert crop.observation.depth.shape == (64, 90)
    assert np.isnan(crop.observation.depth).all()
    assert crop.robot_depth is None


def test_restore_points_round_trip_uses_pixel_centers_and_exclusive_bounds():
    obs = observation()
    crop = crop_observation(obs, reference(obs))
    h, w = crop.observation.depth.shape
    crop_uv = np.array([[0, 0], [w - 1, h - 1], [17, 83]])
    restored = crop.restore_points(crop_uv / [w - 1, h - 1])
    source_uv = pixels(restored, obs.depth.shape)
    assert np.array_equal(source_uv, crop_uv + crop.bounds_xyxy[:2])
    assert crop.restore_points([]).shape == (0, 2)
    with pytest.raises(ValueError, match='normalized'):
        crop.restore_points([[1.01, .5]])


def test_half_pixel_ties_keep_the_same_rasterized_pixels_after_translation():
    obs = observation()
    crop = crop_observation(obs, reference(obs))
    # The crop origin deliberately has odd coordinates. A half-pixel tie must
    # round in crop coordinates, not after adding the offset to the full image.
    crop = replace(crop, bounds_xyxy=(51, 53, 179, 181))
    normalized = np.array([[.5, .5], [12.5 / 127, 27.5 / 127]])
    assert np.array_equal(pixels(crop.restore_points(normalized), obs.depth.shape),
        pixels(normalized, crop.observation.depth.shape) + [51, 53])


def test_restore_annotation_preserves_all_coordinate_fields_and_semantics():
    obs = observation()
    crop = crop_observation(obs, reference(obs))
    polygon = GroundingProposal(kind='polygon', role='object', frame_id=17, camera_id='wrist',
        target_id='observed_target', points=[(.1, .1), (.8, .1), (.8, .8), (.1, .8)],
        holes=[[(.3, .3), (.5, .3), (.4, .5)]], negative_points=[(.4, .4)])
    seed = GroundingProposal(kind='points', role='grasp_contact', frame_id=17,
        camera_id='wrist', target_id='observed_target', points=[(.2, .2)])
    box = GroundingProposal(kind='box', role='grasp_part', frame_id=17,
        camera_id='wrist', target_id='observed_target', box=(.1, .1, .8, .8))
    decision = AgentDecision(action='observe', frame_id=17, camera_id='wrist', reason='annotation',
        point=(.2, .2), bbox=(.1, .1, .8, .8), part_bbox=(.15, .15, .7, .7),
        place_point=(.9, .9), groundings=[polygon, seed, box])
    out = crop.restore_decision(decision)
    assert out.frame_id == 17 and out.camera_id == 'wrist' and out.reason == decision.reason
    assert out.groundings[0].target_id == polygon.target_id and out.groundings[0].source == 'vlm'
    for name in ('point', 'place_point'):
        assert np.allclose(getattr(out, name), crop.restore_points([getattr(decision, name)])[0])
    for name in ('bbox', 'part_bbox'):
        assert np.allclose(getattr(out, name), crop.restore_points(np.array(getattr(decision, name)).reshape(2, 2)).ravel())
    for original, restored in zip(decision.groundings, out.groundings):
        assert np.allclose(np.asarray(restored.points).reshape(-1, 2), crop.restore_points(original.points))
        assert np.allclose(np.asarray(restored.negative_points).reshape(-1, 2), crop.restore_points(original.negative_points))
    assert np.allclose(out.groundings[0].holes[0], crop.restore_points(polygon.holes[0]))
    assert np.allclose(out.groundings[2].box, crop.restore_points(np.array(box.box).reshape(2, 2)).ravel())
    before = proposal_mask(polygon, crop.observation)
    after = proposal_mask(out.groundings[0], obs)
    x0, y0, x1, y1 = crop.bounds_xyxy
    assert np.array_equal(before, after[y0:y1, x0:x1])
    assert after.sum() == before.sum()
    assert decision.point == (.2, .2) and decision.groundings[0].points[0] == (.1, .1)


def test_restore_mask_preserves_exact_pixels_and_hole_and_rejects_wrong_size():
    obs = observation()
    crop = crop_observation(obs, reference(obs))
    mask = np.zeros(crop.observation.depth.shape, dtype=bool)
    mask[10:33, 7:39] = True
    mask[15:20, 14:25] = False
    proposal = GroundingProposal(kind='mask', role='grasp_part', frame_id=17,
        camera_id='wrist', mask_size=mask.shape, mask_rle=mask_rle(mask))
    decision = AgentDecision(action='observe', frame_id=17, camera_id='wrist', groundings=[proposal])
    out = crop.restore_decision(decision)
    full = proposal_mask(out.groundings[0], obs)
    x0, y0, x1, y1 = crop.bounds_xyxy
    assert full.sum() == mask.sum() and np.array_equal(full[y0:y1, x0:x1], mask)
    wrong = proposal.model_copy(update={'mask_size': (64, 256)})
    with pytest.raises(ValueError, match='mask dimensions'):
        crop.restore_decision(decision.model_copy(update={'groundings': [wrong]}))


@pytest.mark.parametrize('change,match', [({'frame_id': 16}, 'frame and camera'),
    ({'camera_id': 'different'}, 'frame and camera'), ({'action': 'grasp'}, 'observe annotations')])
def test_restore_does_not_correct_wrong_source_or_execute_actions(change, match):
    obs = observation()
    crop = crop_observation(obs, reference(obs))
    decision = AgentDecision(action='observe', frame_id=17, camera_id='wrist')
    with pytest.raises(ValueError, match=match):
        crop.restore_decision(decision.model_copy(update=change))


def test_restore_rejects_candidate_and_stale_nested_grounding():
    obs = observation()
    crop = crop_observation(obs, reference(obs))
    candidate = GroundingProposal(kind='candidate', role='object', frame_id=17,
        camera_id='wrist', candidate_id='crop-only-region')
    decision = AgentDecision(action='observe', frame_id=17, camera_id='wrist', groundings=[candidate])
    with pytest.raises(ValueError, match='candidate IDs'):
        crop.restore_decision(decision)
    stale = candidate.model_copy(update={'frame_id': 16})
    with pytest.raises(ValueError, match='source frame and camera'):
        crop.restore_decision(decision.model_copy(update={'groundings': [stale]}))


@pytest.mark.parametrize('points', [[], [[np.nan, 0, 1]], [[0, 1]], [[0, 0, 0]], [[0, 0, -1]], [[100, 0, 1]]])
def test_invalid_or_unprojectable_references_fail_closed(points):
    obs = replace(observation(), T_wc=np.eye(4))
    with pytest.raises(ValueError, match='crop reference'):
        crop_observation(obs, points)


@pytest.mark.parametrize('kwargs', [{'surface_margin_m': -1}, {'surface_margin_m': np.inf},
    {'context_pixels': -1}, {'context_pixels': .5}, {'min_size': 1}, {'min_size': 12.5},
    {'robot_depth': np.zeros((2, 2))}])
def test_invalid_crop_options_rejected(kwargs):
    obs = observation()
    with pytest.raises(ValueError):
        crop_observation(obs, reference(obs), **kwargs)


def test_invalid_calibration_and_misaligned_images_rejected():
    obs = observation()
    for altered in (replace(obs, K=np.eye(2)), replace(obs, T_wc=np.zeros((4, 4))),
                    replace(obs, rgb=obs.rgb[:-1])):
        with pytest.raises(ValueError):
            crop_observation(altered, reference(obs))
