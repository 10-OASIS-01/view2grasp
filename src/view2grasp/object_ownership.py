"""Explicit source-image object ownership, separate from contact permission.

A historical image establishes only source-pose ownership; callers must
establish a rigid attachment transform before
using its points as a carried collision envelope. Nothing here changes a map,
grasp ray, contact region, or UNKNOWN classification.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from .geometry import unproject
from .grounding import pixels, proposal_mask
from .types import AgentDecision, Observation


REFERENCE_MARGIN_M = .035
ANCHOR_DISTANCE_M = .004
COMPONENT_STEP_M = .006
# Equality tolerance for recomputing an endpoint from the SAME saved RGB-D
# calibration. This is a nanometre, not a voxel/association/collision radius.
SOURCE_EQUALITY_M = 1e-9


class ObjectOwnershipError(ValueError):
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message)


def _digest(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        value.update(str((array.dtype.str, array.shape)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def _source_digest(obs):
    return _digest(np.frombuffer(json.dumps([obs.frame_id, obs.camera_id,
        obs.timestamp], separators=(',', ':')).encode(), dtype=np.uint8),
        obs.rgb, obs.depth, obs.K, obs.T_wc)


def _readonly(array):
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


def _unresolved_component_review(candidate, xyz, labelled, reference_tree, lo, hi):
    """Report unseeded measured components without assigning them an identity."""
    h, w = candidate.shape
    seen = labelled.copy()
    eligible = []
    # Seeded floods already visited their complete components, including
    # rejected ones. Never offer accepted or abstained material for automatic
    # reconsideration, nor cross a hole, negative point or filtered depth gap.
    for y, x in np.argwhere(candidate & ~seen):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        component = []
        while stack:
            yy, xx = stack.pop()
            component.append((yy, xx))
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if (0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and not seen[ny, nx]
                        and np.linalg.norm(xyz[ny, nx] - xyz[yy, xx]) <= COMPONENT_STEP_M):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        pixels_yx = np.asarray(component, dtype=int)
        points = xyz[pixels_yx[:, 0], pixels_yx[:, 1]]
        if np.any((points < lo) | (points > hi)):
            continue
        anchors = int(np.count_nonzero(reference_tree.query(points)[0] < ANCHOR_DISTANCE_M))
        if anchors < 4:
            continue
        y0, x0 = pixels_yx.min(axis=0)
        y1, x1 = pixels_yx.max(axis=0)
        eligible.append({'pixels': len(points), 'original_anchor_pixels': anchors,
            'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)]})
    # Python's stable sort preserves source row-major component order on ties.
    eligible.sort(key=lambda item: -item['pixels'])
    return {'unresolved_measured_components': eligible[:3],
        'unresolved_measured_component_count': len(eligible),
        'review_scope': 'Unseeded measured components inside the supplied semantic region only. '
            'Geometric eligibility is not object identity, ownership, contact permission or FREE evidence. '
            'Only a new explicit semantic reply may select them; bbox_xyxy uses inclusive source-image pixels.'}


@dataclass(frozen=True)
class ObjectOwnership:
    object_mask: np.ndarray
    object_points: np.ndarray
    source_pixels_xy: np.ndarray
    source_depth_m: np.ndarray
    source_frame_id: int
    source_camera_id: str
    source_evidence_sha256: str
    abstained_source_pixels_xy: np.ndarray
    abstained_source_points: np.ndarray
    report: dict


def resolve_object_ownership(obs: Observation, decision: AgentDecision,
                             original_reference, target_id: str, support_z: float,
                             robot_depth=None, *, allow_partial_seed_components=False,
                             material_mask=False) -> ObjectOwnership:
    """Validate explicitly seeded whole components inside one object region.

    The region can be a box, polygon or mask; only separate ``object/points``
    proposals seed ownership. Unseeded islands receive no ownership. Every
    selected component is traversed before checking the fixed original AABB, so
    an outlying tail cannot be silently clipped into acceptance. The caller
    supplies the same immutable original world reference for every source image
    in a static scene. Identity still relies on explicit VLM semantics: surfaces
    separated by less than the 6 mm graph step are not geometrically separable
    in general. This contract does not claim an independent identity proof.

    The default rejects the entire reply on any invalid seed/component. Opt-in
    partial mode abstains from each failure and requires FOUR original anchor
    pixels in EACH accepted complete component. It never trims a failed
    component or moves a seed. Abstention is not contact or object permission.

    Explicit material mode is a separate, atomic semantic contract: a FRESH
    polygon/mask labels every selected measured material pixel. It cannot be
    applied to a detection box or an old seeded reply. It retains the fixed
    envelope, independent reference anchors and self/support filters, but does
    not require a separate seed in each depth fragment. Identity depends on
    that new semantic label, not on geometry proving neighboring-object identity.
    """
    report = {'frame_id': obs.frame_id, 'camera_id': obs.camera_id,
        'target_id': target_id, 'scope': 'source_image_object_ownership_only',
        'grants_contact_permission': False, 'grasp_ray_changed': False,
        'map_modified': False, 'grants_free_space': False,
        'allow_partial_seed_components': bool(allow_partial_seed_components)}
    if material_mask:
        report.update(allow_partial_seed_components=False,explicit_material_mask=True)

    def reject(reason, message, **details):
        raise ObjectOwnershipError(message, {**report, 'failure': reason, **details})

    if decision.frame_id != obs.frame_id or decision.camera_id != obs.camera_id:
        reject('source_mismatch', 'object ownership requires the explicit source frame and camera')
    if decision.action != 'observe':
        reject('annotation_action_required', 'object ownership requires an observe annotation, not an execution decision')
    proposals = decision.groundings
    if not proposals or any(g.role != 'object' or g.source != 'vlm' or
            g.target_id != target_id or g.frame_id != obs.frame_id or
            g.camera_id != obs.camera_id for g in proposals):
        reject('semantic_identity', 'object ownership requires only same-target source-frame VLM object proposals')
    regions = [g for g in proposals if g.kind in ('box', 'polygon', 'mask')]
    seeds = [p for g in proposals if g.kind == 'points' for p in g.points]
    if len(regions) != 1 or not seeds or any(g.kind not in ('box', 'polygon', 'mask', 'points')
                                          for g in proposals):
        reject('region_and_seeds_required', 'object ownership requires one region and explicit object positive seeds')
    if material_mask and regions[0].kind not in ('polygon','mask'):
        reject('material_mask_required','explicit material needs a fresh polygon or mask, never a detection box')
    reference = np.asarray(original_reference, dtype=float)
    if (reference.ndim != 2 or reference.shape[1:] != (3,) or len(reference) < 4 or
            not np.isfinite(reference).all()):
        reject('invalid_original_reference', 'object ownership requires at least four finite original surface points')
    if not np.isfinite(support_z):
        reject('invalid_support', 'object ownership requires a finite observed support height')
    if (obs.depth.ndim != 2 or min(obs.depth.shape) < 2 or
            np.shape(obs.rgb) != (*obs.depth.shape, 3) or
            np.shape(obs.K) != (3, 3) or np.shape(obs.T_wc) != (4, 4) or
            not np.isfinite(obs.K).all() or not np.isfinite(obs.T_wc).all()):
        reject('invalid_calibration', 'object ownership requires a calibrated source depth image')
    h, w = obs.depth.shape
    if robot_depth is not None and np.shape(robot_depth) != (h, w):
        reject('robot_depth_shape', 'robot depth must match the source image')

    region = proposal_mask(regions[0], obs)
    semantic = region.copy()
    for g in proposals:
        for x, y in pixels(g.negative_points, (h, w)).reshape(-1, 2):
            semantic[max(0, y-2):y+3, max(0, x-2):x+3] = False
    v, u = np.mgrid[:h, :w]
    xyz = unproject(np.stack([u, v], -1), obs.depth, obs.K, obs.T_wc)
    depth_valid = (np.isfinite(obs.depth) & (obs.depth > .05) & (obs.depth < 10) &
                   np.isfinite(xyz).all(axis=2))
    support_clear = xyz[:, :, 2] > support_z + .006
    robot_clear = np.ones((h, w), dtype=bool)
    if robot_depth is not None:
        robot_clear = np.isfinite(robot_depth) & (obs.depth < robot_depth - .003)
    valid = depth_valid & support_clear & robot_clear
    candidate = semantic & valid
    seed_pixels = pixels(seeds, (h, w)).reshape(-1, 2)
    abstained = np.zeros((h, w), dtype=bool)
    abstentions = []
    for seed_index, (x, y) in enumerate(seed_pixels):
        if not candidate[y, x]:
            reasons = []
            if not region[y, x]: reasons.append('outside_region_or_hole')
            elif not semantic[y, x]: reasons.append('explicit_negative')
            if not depth_valid[y, x]: reasons.append('invalid_depth')
            if depth_valid[y, x] and not support_clear[y, x]: reasons.append('support')
            if not robot_clear[y, x]: reasons.append('robot_occluded_or_unverified')
            details = {'seed_pixel': [int(x), int(y)], 'seed_indices': [seed_index],
                'source_pixels_xy': [[int(x), int(y)]], 'reasons': reasons,
                'inside_semantic_region': bool(semantic[y, x]), 'valid_filtered_depth': bool(valid[y, x])}
            if not allow_partial_seed_components or material_mask:
                reject('invalid_explicit_seed', 'object ownership seed is not on valid explicit object evidence', **details)
            abstained[y, x] = True
            abstentions.append({'failure': 'invalid_explicit_seed', **details})

    lo = reference.min(axis=0) - REFERENCE_MARGIN_M
    hi = reference.max(axis=0) + REFERENCE_MARGIN_M
    report.update(original_reference_sha256=_digest(reference),
        fixed_original_aabb_m=[lo.tolist(), hi.tolist()], support_z_m=float(support_z),
        thresholds={'aabb_margin_m': REFERENCE_MARGIN_M, 'anchor_distance_exclusive_m': ANCHOR_DISTANCE_M,
            'component_step_inclusive_m': COMPONENT_STEP_M, 'minimum_total_anchor_pixels': 4,
            'minimum_component_anchor_pixels': 4 if allow_partial_seed_components else 1},
        proposed_region_pixels=int(semantic.sum()), valid_region_pixels=int(candidate.sum()))
    reference_tree = cKDTree(reference)
    if material_mask:
        yy,xx=np.where(candidate);points=xyz[candidate]
        outside=np.any((points<lo)|(points>hi),axis=1)
        if outside.any():
            reject('material_outside_fixed_envelope','explicit material leaves the fixed original envelope',
                rejected_pixel_count=int(outside.sum()),rejected_pixels_xy=np.column_stack([xx,yy])[outside][:12].tolist())
        anchors=int(np.count_nonzero(reference_tree.query(points)[0]<ANCHOR_DISTANCE_M))
        if anchors<4:
            reject('insufficient_original_overlap','explicit material needs four independent original anchors',anchor_pixels=anchors)
        report.update(membership_authority='fresh_vlm_dense_material_region',
            component_seed_gate_applied=False,selected_region_kind=regions[0].kind,
            anchor_pixels=anchors,object_pixels=len(points),source_timestamp=float(obs.timestamp),
            partial=False,abstentions=[],abstained_source_pixels=0,source_pose_only=True,
            rigid_attachment_mapping_required_before_carrying=True,
            semantic_identity_limit='Fresh pixel-level VLM material labels supply identity; fixed geometry does not prove a nearby object was not mislabeled.')
        report['thresholds'].update(component_step_inclusive_m=None,minimum_component_anchor_pixels=None)
        return ObjectOwnership(_readonly(candidate),_readonly(points),_readonly(np.column_stack([xx,yy])),
            _readonly(obs.depth[candidate]),obs.frame_id,obs.camera_id,_source_digest(obs),
            _readonly(np.empty((0,2),int)),_readonly(np.empty((0,3))),report)
    selected = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), -1, dtype=int)
    outcomes = []
    components = []
    total_anchors = 0
    for seed_index, (x, y) in enumerate(seed_pixels):
        if not candidate[y, x]:
            continue
        if labels[y, x] >= 0:
            outcomes[labels[y, x]]['seed_indices'].append(seed_index)
            continue
        # No AABB clipping here: the complete selected semantic/depth component
        # must satisfy the original bound, including any measured outlying tail.
        component_id = len(outcomes)
        labels[y, x] = component_id
        stack = [(y, x)]
        component = []
        while stack:
            yy, xx = stack.pop()
            component.append((yy, xx))
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if (0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and labels[ny, nx] < 0
                        and np.linalg.norm(xyz[ny, nx] - xyz[yy, xx]) <= COMPONENT_STEP_M):
                    labels[ny, nx] = component_id
                    stack.append((ny, nx))
        component_yx = np.asarray(component, dtype=int)
        points = xyz[component_yx[:, 0], component_yx[:, 1]]
        outside = np.any((points < lo) | (points > hi), axis=1)
        anchors = int(np.count_nonzero(reference_tree.query(points)[0] < ANCHOR_DISTANCE_M))
        if outside.any() and not allow_partial_seed_components:
            rejected = component_yx[outside]
            reject('component_outside_fixed_envelope', 'object ownership component leaves the fixed original AABB',
                component_pixels=len(points), rejected_pixel_count=int(outside.sum()),
                rejected_pixels_xy=rejected[:12, ::-1].tolist())
        if not anchors and not allow_partial_seed_components:
            reject('unanchored_seeded_component', 'object ownership component lacks original surface anchors',
                component_pixels=len(points), seed_pixel=[int(x), int(y)])
        item = {'pixels': len(points), 'original_anchor_pixels': anchors,
            'seed_pixel': [int(x), int(y)], 'seed_indices': [seed_index]}
        outcomes.append(item)
        if allow_partial_seed_components and (outside.any() or anchors < 4):
            reasons = []
            if outside.any(): reasons.append('component_outside_fixed_envelope')
            if anchors < 4:
                reasons.append('unanchored_seeded_component' if not anchors else 'insufficient_component_anchors')
            item.update(failure=reasons[0], reasons=reasons,
                outside_fixed_envelope_pixels=int(outside.sum()),
                source_pixels_xy=component_yx[:, ::-1].tolist())
            if outside.any():
                # Diagnose the actual outlying depth pixels, not the valid
                # membership seed that selected this entire failed component.
                # Sampling this report never clips the component or changes
                # its full abstention/retention mask.
                item.update(rejected_pixel_count=int(outside.sum()),
                    rejected_pixels_xy=component_yx[outside][:12, ::-1].tolist())
            abstentions.append(item)
            abstained[component_yx[:, 0], component_yx[:, 1]] = True
            continue
        selected[component_yx[:, 0], component_yx[:, 1]] = True
        total_anchors += anchors
        components.append(item)
    report.update(partial=bool(abstentions), abstentions=abstentions)
    if allow_partial_seed_components:
        report.update(_unresolved_component_review(candidate, xyz, labels >= 0,
            reference_tree, lo, hi))
    if allow_partial_seed_components and not components:
        reject('no_accepted_components', 'object ownership abstained from every explicit seed component')
    if total_anchors < 4:
        reject('insufficient_original_overlap', 'object ownership needs at least four original anchor pixels',
            anchor_pixels=total_anchors)
    yy, xx = np.where(selected)
    ay, ax = np.where(abstained)
    abstained_points = xyz[abstained & depth_valid]
    report.update(selected_components=components, anchor_pixels=total_anchors,
        object_pixels=len(yy), unselected_valid_region_pixels=int(np.sum(candidate & ~selected)),
        abstained_source_pixels=len(ay),
        source_timestamp=float(obs.timestamp),
        source_pose_only=True, rigid_attachment_mapping_required_before_carrying=True)
    return ObjectOwnership(_readonly(selected), _readonly(xyz[selected]),
        _readonly(np.column_stack([xx, yy])), _readonly(obs.depth[selected]),
        obs.frame_id, obs.camera_id, _source_digest(obs), _readonly(np.column_stack([ax, ay])),
        _readonly(abstained_points), report)


def source_endpoint_retirement_mask(surface_points, ownership: ObjectOwnership,
                                    source_obs: Observation, *, table_z=None):
    """Pure selection of numerical matches to accepted source-image endpoints.

    No map or depth frame is changed. Callers must add the entire accepted cloud
    to the appropriate collision envelope before applying this selection. Other
    views' voxel representatives are retained unless they numerically coincide
    with a measured accepted source endpoint; no nearest-voxel fallback exists.
    """
    if (ownership.source_frame_id != source_obs.frame_id or
            ownership.source_camera_id != source_obs.camera_id or
            ownership.source_evidence_sha256 != _source_digest(source_obs)):
        raise ValueError('endpoint retirement requires the unchanged ownership source image')
    surface = np.asarray(surface_points, dtype=float)
    if surface.ndim != 2 or surface.shape[1:] != (3,) or not np.isfinite(surface).all():
        raise ValueError('endpoint retirement requires finite map surface points')
    support = ownership.report['support_z_m'] if table_z is None else float(table_z)
    if not np.isfinite(support):
        raise ValueError('endpoint retirement requires a finite observed support height')
    result = np.zeros(len(surface), dtype=bool)
    if len(surface) and len(ownership.object_points):
        result = cKDTree(ownership.object_points).query(surface)[0] <= SOURCE_EQUALITY_M
        # Even a numerical alias of accepted and abstained source evidence
        # must remain occupied. Abstention always has priority over retirement.
        if len(ownership.abstained_source_points):
            result &= cKDTree(ownership.abstained_source_points).query(surface)[0] > SOURCE_EQUALITY_M
        result &= np.abs(surface[:, 2] - support) >= .004
    return result
