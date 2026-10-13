"""Conservative RGB-D contradictions to an explicitly requested grasp part.

These checks do not identify parts, certify a grasp, or move contact points.
The vertical-midpoint check applies only when that semantic constraint was
explicitly requested, in a calibrated world frame whose Z axis is vertical.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import project, unproject
from .types import Observation


@dataclass(frozen=True)
class VerticalPartMidpointConfig:
    # Evidence requirements, not dimensions of a known object or target pose.
    min_object_points: int = 32
    min_part_points: int = 8
    min_object_height_m: float = .03
    height_quantile: float = .05
    # All four geometric conditions must hold to reject the annotation.
    height_extreme_fraction: float = .15
    min_principal_variance_ratio: float = 4.
    max_principal_vertical_alignment: float = .35
    max_part_height_fraction: float = .15
    # An observed height extreme also needs a real visible depth boundary.
    height_boundary_band_m: float = .006
    boundary_continuation_m: float = .006
    boundary_depth_gap_m: float = .003
    min_boundary_samples: int = 4

    def __post_init__(self):
        if (any(isinstance(value, bool) or not isinstance(value, int) or value < 4
                for value in (self.min_object_points, self.min_part_points, self.min_boundary_samples))
                or not np.isfinite(self.min_object_height_m) or self.min_object_height_m <= 0
                or not 0 <= self.height_quantile < .5
                or not 0 < self.height_extreme_fraction < .5
                or not np.isfinite(self.min_principal_variance_ratio) or self.min_principal_variance_ratio <= 1
                or not 0 <= self.max_principal_vertical_alignment < 1
                or not 0 < self.max_part_height_fraction < 1
                or any(not np.isfinite(value) or value <= 0 for value in
                    (self.height_boundary_band_m, self.boundary_continuation_m, self.boundary_depth_gap_m))):
            raise ValueError('invalid vertical-part midpoint evidence thresholds')


class GraspSemanticsError(ValueError):
    """A semantic contradiction supported by current observed geometry only."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('grasp semantics: observed part is elongated horizontally at an object-height extreme, '
            'inconsistent with the requested vertical-part midpoint. Reinspect the complete visible straight '
            'vertical bar in THIS image and annotate its middle, excluding junctions; do not reuse old coordinates '
            'or move the contact automatically.')


def _mask_bbox(mask):
    y, x = np.where(mask)
    if not len(x): return {'pixel': None, 'normalized': None}
    h, w = mask.shape
    box = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    return {'pixel': box, 'normalized': (np.array(box) / [w - 1, h - 1, w - 1, h - 1]).tolist()}


def _height_boundary_evidence(obs, object_mask, rows, cols, xyz, upper, config):
    """Veto a supposed height extreme if the mask cuts/occludes a surface.

    This does not grow the semantic mask. It only asks whether the pixels just
    beyond its height boundary actually reveal farther background, rather than
    an arbitrary annotation edge or foreground occlusion. A partial mask alone
    is never evidence that the physical object ends there.
    """
    direction = 1 if upper else -1
    extreme = xyz[:, 2].max() if upper else xyz[:, 2].min()
    ids = np.flatnonzero(direction * (extreme - xyz[:, 2]) <= config.height_boundary_band_m)
    boundary_xyz = xyz[ids]
    uv, z = project(boundary_xyz + [0, 0, direction * .01], obs.K, obs.T_wc)
    motion = uv - np.c_[cols[ids], rows[ids]]
    h, w = object_mask.shape
    counts = {key: 0 for key in ('background', 'surface_continuation', 'foreground',
        'near_surface', 'invalid_depth', 'unprojectable')}
    samples = []
    seen = set()
    for index, delta, camera_z in zip(ids, motion, z):
        if camera_z <= .01 or not np.isfinite(delta).all() or np.linalg.norm(delta) < .05:
            counts['unprojectable'] += 1
            continue
        x, y = int(cols[index]), int(rows[index])
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if np.dot(delta, [dx, dy]) < .5 * np.linalg.norm(delta): continue
            xx, yy = x + dx, y + dy
            if not (0 <= xx < w and 0 <= yy < h):
                counts['invalid_depth'] += 1
                continue
            if object_mask[yy, xx] or (x, y, xx, yy) in seen: continue
            seen.add((x, y, xx, yy))
            depth = obs.depth[yy, xx]
            if not np.isfinite(depth) or not .05 < depth < 10:
                reason = 'invalid_depth'
            else:
                outside = unproject([xx, yy], depth, obs.K, obs.T_wc)
                gap = float(depth - obs.depth[y, x])
                if not np.isfinite(outside).all(): reason = 'invalid_depth'
                elif np.linalg.norm(outside - xyz[index]) <= config.boundary_continuation_m:
                    reason = 'surface_continuation'
                elif gap < -config.boundary_depth_gap_m: reason = 'foreground'
                elif gap <= config.boundary_depth_gap_m: reason = 'near_surface'
                else: reason = 'background'
            counts[reason] += 1
            if reason != 'background' and len(samples) < 8:
                samples.append({'inside_pixel': [x, y], 'outside_pixel': [xx, yy],
                    'reason': reason})
    supported = counts['background'] >= config.min_boundary_samples and not any(
        value for key, value in counts.items() if key != 'background')
    return {'status': 'observed_background_boundary' if supported else 'unresolved',
        'direction': 'upper' if upper else 'lower', 'counts': counts,
        'rejected_boundary_samples': samples,
        'meaning': 'local_depth_boundary_only_not_full_object_visibility'}


def check_vertical_part_midpoint(obs: Observation, object_mask, part_mask, contact, *,
                                 config=VerticalPartMidpointConfig()):
    """Check only a strong contradiction to 'vertical part, at its midpoint'.

    Masks must be the current grounding resolver's measured object/part masks.
    Height is measured along calibrated world Z, not image Y. Only this frame
    supplies geometry; historical target clouds and simulator truth are unused.
    Lack of evidence returns ``unresolved``. ``no_contradiction`` is not proof
    of semantic correctness or graspability, and never adds contact permission.
    """
    if obs.depth.ndim != 2 or min(obs.depth.shape) < 2:
        raise ValueError('semantic check requires a calibrated RGB-D observation')
    object_mask, part_mask = np.asarray(object_mask, bool), np.asarray(part_mask, bool)
    if object_mask.shape != obs.depth.shape or part_mask.shape != obs.depth.shape:
        raise ValueError('semantic masks must match the current observation')
    h, w = obs.depth.shape
    report = {'frame_id': int(obs.frame_id), 'camera_id': obs.camera_id,
        'actual_view': [float(value) if np.isfinite(value) else None for value in obs.view],
        'coordinate_space': 'current_observation_normalized_pixel_centers',
        'image_shape_hw': [int(h), int(w)], 'semantic_requirement': 'vertical_part_midpoint',
        'evidence_source': 'current_resolved_masks_and_calibrated_rgbd',
        'status': 'unresolved', 'reason': '',
        'thresholds': {key: int(value) if isinstance(value, int) else float(value)
            for key, value in asdict(config).items()},
        'object': {'mask_pixels': int(object_mask.sum()), 'mask_bbox': _mask_bbox(object_mask)},
        'part': {'mask_pixels': int(part_mask.sum()), 'mask_bbox': _mask_bbox(part_mask)}}
    if np.any(part_mask & ~object_mask):
        report['reason'] = 'part_not_inside_current_object_mask'
        return report
    if object_mask[[0, -1]].any() or object_mask[:, [0, -1]].any():
        report['reason'] = 'object_mask_reaches_image_boundary'
        return report
    try:
        contact = np.asarray(contact, dtype=float)
    except (ValueError, TypeError):
        contact = np.empty(0)
    if contact.shape != (2,) or not np.isfinite(contact).all() or np.any((contact < 0) | (contact > 1)):
        report['reason'] = 'no_valid_current_contact'
        return report
    x, y = np.rint(contact * [w - 1, h - 1]).astype(int)
    report['contact'] = {'pixel': [int(x), int(y)],
        'normalized': [float(x / (w - 1)), float(y / (h - 1))]}
    if not part_mask[y, x]:
        report['reason'] = 'contact_not_on_current_observed_part'
        return report
    # Unproject only the proposed object pixels, not the whole image. The part
    # is a subset, so it uses exactly the same measured points/calibration.
    rows, cols = np.where(object_mask)
    depth = obs.depth[rows, cols]
    xyz = unproject(np.c_[cols, rows], depth, obs.K, obs.T_wc)
    valid = np.isfinite(depth) & (depth > .05) & (depth < 10) & np.isfinite(xyz).all(axis=1)
    current_object = xyz[valid]
    current_part = xyz[valid & part_mask[rows, cols]]
    report['object']['observed_points'] = len(current_object)
    report['part']['observed_points'] = len(current_part)
    if len(current_object) < config.min_object_points or len(current_part) < config.min_part_points:
        report['reason'] = 'insufficient_current_mask_depth'
        return report
    if not np.isfinite(obs.depth[y, x]) or not .05 < obs.depth[y, x] < 10:
        report['reason'] = 'no_valid_current_contact_depth'
        return report
    anchor = unproject([x, y], obs.depth[y, x], obs.K, obs.T_wc)
    if not np.isfinite(anchor).all():
        report['reason'] = 'no_valid_current_contact_depth'
        return report
    observed_range = np.quantile(current_object[:, 2], [config.height_quantile, 1 - config.height_quantile])
    height = float(observed_range[1] - observed_range[0])
    report['object'].update(z_range_m=[float(current_object[:, 2].min()), float(current_object[:, 2].max())],
        robust_z_range_m=observed_range.tolist(), robust_height_m=height)
    if height < config.min_object_height_m:
        report['reason'] = 'insufficient_observed_object_height'
        return report
    centered = current_part - current_part.mean(axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    if not np.isfinite(covariance).all():
        report['reason'] = 'insufficient_finite_part_geometry'
        return report
    values, axes = np.linalg.eigh(covariance)
    values = np.maximum(values[::-1], 0)
    axis = axes[:, -1]
    # A deterministic sign makes diagnostics comparable; the check uses |Z|.
    if axis[np.argmax(np.abs(axis))] < 0: axis = -axis
    variance_ratio = float(values[0] / max(values[1], 1e-12))
    vertical_alignment = float(abs(axis[2]))
    part_height = float(np.ptp(current_part[:, 2]))
    height_fraction = float((anchor[2] - observed_range[0]) / height)
    report['contact'].update(world_xyz_m=anchor.tolist(), observed_height_fraction=height_fraction)
    report['part'].update(xyz_extent_m=np.ptp(current_part, axis=0).tolist(),
        z_range_m=[float(current_part[:, 2].min()), float(current_part[:, 2].max())],
        principal_axis_world=axis.tolist(), principal_variances_m2=values.tolist(),
        principal_variance_ratio=variance_ratio, principal_vertical_alignment=vertical_alignment,
        height_fraction_of_observed_object=part_height / height)
    extreme = height_fraction <= config.height_extreme_fraction or height_fraction >= 1 - config.height_extreme_fraction
    elongated = variance_ratio >= config.min_principal_variance_ratio
    horizontal = vertical_alignment <= config.max_principal_vertical_alignment
    thin_in_height = part_height / height <= config.max_part_height_fraction
    report['contradiction_conditions'] = {'height_extreme': bool(extreme),
        'elongated': bool(elongated), 'near_horizontal': bool(horizontal),
        'small_vertical_extent': bool(thin_in_height)}
    if extreme and elongated and horizontal and thin_in_height:
        boundary = _height_boundary_evidence(obs, object_mask, rows[valid], cols[valid],
            current_object, height_fraction >= 1 - config.height_extreme_fraction, config)
        report['height_boundary'] = boundary
        if boundary['status'] == 'unresolved':
            report['reason'] = 'observed_height_boundary_not_established'
            return report
        report.update(status='rejected', reason='horizontal_part_at_observed_object_height_extreme')
        raise GraspSemanticsError(report)
    report.update(status='no_contradiction', reason='no_strong_vertical_midpoint_contradiction')
    return report
