"""Bounded observation-only rigid registration for collision payloads.

This is a local geometric heuristic, not object identity or attachment truth.
The caller supplies an immutable pre-close object reference, measured current
object points, and source-owned points from that same static source epoch.
Only a complete rigidly transformed collision cloud is returned. Predicted
points must never become contact permission, FREE evidence or source certificates.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class PayloadRegistrationError(ValueError):
    def __init__(self, message, report):
        self.report = report
        self.diagnostics = report
        super().__init__(message)


@dataclass(frozen=True)
class PayloadRegistration:
    R: np.ndarray
    t: np.ndarray
    transformed_source_points: np.ndarray
    report: dict


def _digest(points):
    value = np.ascontiguousarray(points)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def _readonly(array):
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


def _sample(points, maximum):
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points)-1, maximum, dtype=int)]


def _rigid_fit(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, b-rotation @ a


def _icp(reference, current, tree, rotation, translation, mutual, match_m):
    # Broad correspondences initialize a local fit; only the final 4 mm (or
    # caller-specified) residual counts as evidence. Trimming affects fitting,
    # never the returned complete source cloud or the validation denominator.
    for limit in (6.25*match_m, 3.75*match_m, 2*match_m, match_m):
        for _ in range(35):
            moved = reference @ rotation.T + translation
            distance, index = tree.query(moved)
            ids = np.flatnonzero(distance < limit)
            if mutual and len(ids):
                ids = ids[cKDTree(moved).query(current[index[ids]])[1] == ids]
            if len(ids) < 20:
                break
            ids = ids[np.argsort(distance[ids])[:max(20, int(len(ids)*.85))]]
            dr, dt = _rigid_fit(moved[ids], current[index[ids]])
            rotation, translation = dr @ rotation, dr @ translation + dt
            if np.linalg.norm(dt) < 1e-7 and np.linalg.norm(Rotation.from_matrix(dr).as_rotvec()) < 1e-6:
                break
    return rotation, translation


def register_payload_source(original_reference, current_object_points, source_owned_points, *,
                            match_distance_m=.004, max_center_motion_m=.030,
                            max_rotation_degrees=20., minimum_source_coverage=.5,
                            minimum_inlier_points=32, ambiguity_distance_m=.004):
    """Estimate ``current = source @ R.T + t`` for collision checks only.

    Gates require a majority of the full original measured reference within the
    residual bound, non-planar inlier support wider than that bound, and a
    well-conditioned local point-to-plane information matrix. Nineteen bounded
    starts with nearest and reciprocal correspondences test local ambiguity;
    comparably supported fits must agree on every source-owned point. These
    finite local tests can falsify a bad fit; they do not prove global uniqueness
    or semantic identity. No simulator pose, RGB color, or object ID is used.
    """
    report = {'scope': 'heuristic_observed_rigid_registration_for_collision_only',
        'grants_contact_permission': False, 'grants_free_space': False,
        'grants_source_certificate': False, 'map_modified': False,
        'identity_or_attachment_truth_established': False,
        'transform_convention': 'current_world = source_world @ R.T + t'}

    def reject(reason, **details):
        raise PayloadRegistrationError('payload registration rejected: '+reason,
            {**report, 'accepted': False, 'failure': reason, **details})

    parameters = np.asarray([match_distance_m, max_center_motion_m,
        max_rotation_degrees, minimum_source_coverage, ambiguity_distance_m], dtype=float)
    if (not np.isfinite(parameters).all() or np.any(parameters <= 0)
            or minimum_source_coverage > 1 or max_rotation_degrees > 45
            or type(minimum_inlier_points) is not int or minimum_inlier_points < 20):
        reject('invalid_parameters')
    arrays = []
    for name, value, minimum in (('original_reference', original_reference, minimum_inlier_points),
            ('current_object_points', current_object_points, minimum_inlier_points),
            ('source_owned_points', source_owned_points, 1)):
        points = np.asarray(value, dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < minimum
                or len(points) > 100000 or not np.isfinite(points).all()):
            reject('invalid_input_cloud', input=name)
        arrays.append(points)
    original, current, source = arrays
    report.update(input_sha256={name: _digest(points) for name, points in zip(
        ('original_reference', 'current_object_points', 'source_owned_points'), arrays)},
        input_counts={name: len(points) for name, points in zip(
        ('original_reference', 'current_object_points', 'source_owned_points'), arrays)},
        thresholds={'match_distance_exclusive_m': float(match_distance_m),
            'maximum_center_motion_m': float(max_center_motion_m),
            'maximum_rotation_degrees': float(max_rotation_degrees),
            'minimum_full_reference_coverage': float(minimum_source_coverage),
            'minimum_inlier_points': minimum_inlier_points,
            'minimum_inlier_singular_m': float(match_distance_m),
            'maximum_point_plane_condition': 100.,
            'comparable_fit_coverage_ratio': .95,
            'maximum_complete_source_pose_disagreement_m': float(ambiguity_distance_m)},
        computation_caps={'reference_fit_points': 2048, 'current_fit_points': 4096,
            'starts': 19, 'correspondence_methods': 2, 'stages': 4, 'iterations_per_stage': 35,
            'information_matrix_points': 512})
    fit_reference, fit_current = _sample(original, 2048), _sample(current, 4096)
    fit_tree, full_tree = cKDTree(fit_current), cKDTree(current)
    center = original.mean(0)
    radius = float(np.linalg.norm(original-center, axis=1).max())
    if radius <= match_distance_m:
        reject('insufficient_spatial_extent')
    starts = [('identity', np.eye(3), np.zeros(3))]
    for axis in range(3):
        for angle in (-.75, -.25, .25, .75):
            rv = np.zeros(3); rv[axis] = np.deg2rad(angle*max_rotation_degrees)
            rotation = Rotation.from_rotvec(rv).as_matrix()
            starts.append((f'rotation_{axis}_{angle}', rotation, center-rotation @ center))
        for sign in (-1, 1):
            shift = np.zeros(3); shift[axis] = sign*max_center_motion_m/3
            starts.append((f'translation_{axis}_{sign}', np.eye(3), shift))
    candidates = []
    for mutual in (False, True):
        for name, rotation, translation in starts:
            rotation, translation = _icp(fit_reference, fit_current, fit_tree,
                rotation.copy(), translation.copy(), mutual, match_distance_m)
            angle = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))
            center_motion = float(np.linalg.norm(center @ rotation.T+translation-center))
            distance = full_tree.query(original @ rotation.T+translation)[0]
            inliers = distance < match_distance_m
            count = int(inliers.sum()); coverage = float(inliers.mean())
            candidates.append({'R': rotation, 't': translation, 'inliers': inliers,
                'start': name, 'method': 'reciprocal' if mutual else 'nearest',
                'rotation_degrees': angle, 'center_motion_m': center_motion,
                'inlier_points': count, 'source_coverage': coverage,
                'inlier_rms_m': float(np.sqrt(np.mean(distance[inliers]**2))) if count else None,
                'within_motion_bound': angle <= max_rotation_degrees and center_motion <= max_center_motion_m})
    report['candidates'] = [{key: value for key, value in item.items() if key not in ('R','t','inliers')}
        for item in candidates]
    eligible = [item for item in candidates if item['within_motion_bound']
        and item['inlier_points'] >= minimum_inlier_points
        and item['source_coverage'] >= minimum_source_coverage]
    if not eligible:
        reject('insufficient_bounded_overlap')
    eligible.sort(key=lambda item: (-item['source_coverage'], item['inlier_rms_m']))
    best = eligible[0]
    comparable = [item for item in eligible if item['source_coverage'] >= .95*best['source_coverage']]
    transformed = source @ best['R'].T + best['t']
    # Check pairwise complete-cloud disagreement, not just centroid agreement.
    disagreement = 0.
    for i, first in enumerate(comparable):
        a = source @ first['R'].T + first['t']
        for second in comparable[i+1:]:
            b = source @ second['R'].T + second['t']
            disagreement = max(disagreement, float(np.linalg.norm(a-b, axis=1).max()))
    report.update(comparable_fits=len(comparable), maximum_source_disagreement_m=disagreement)
    if disagreement > ambiguity_distance_m:
        reject('ambiguous_local_pose')
    moved = original @ best['R'].T+best['t']
    inliers = moved[best['inliers']]
    singular = np.linalg.svd((inliers-inliers.mean(0))/np.sqrt(len(inliers)), compute_uv=False)
    report['inlier_coordinate_singular_m'] = singular.tolist()
    if singular[-1] < match_distance_m:
        reject('geometric_degeneracy')
    points = _sample(inliers, 512)
    index = full_tree.query(points)[1]
    neighborhoods = current[full_tree.query(current[index], k=min(12,len(current)))[1]]
    centered = neighborhoods-neighborhoods.mean(1)[:, None, :]
    _, _, vt = np.linalg.svd(centered)
    normals = vt[:, -1, :]
    lever = (points-points.mean(0))/radius
    jacobian = np.c_[np.cross(lever,normals), normals]
    eigenvalues = np.linalg.eigvalsh(jacobian.T @ jacobian/len(jacobian))
    condition = float(eigenvalues[-1]/max(eigenvalues[0],1e-15))
    report.update(point_plane_information_eigenvalues=eigenvalues.tolist(), point_plane_condition=condition)
    if condition > 100. or eigenvalues[0] <= 1e-8:
        reject('geometric_degeneracy')
    report.update(accepted=True, selected={key: value for key,value in best.items()
        if key not in ('R','t','inliers')}, R=best['R'].tolist(), t=best['t'].tolist(),
        source_center_motion_m=(center @ best['R'].T+best['t']-center).tolist(),
        complete_source_points_transformed=len(source), points_pruned_from_source=0,
        uncertainty_statement='Local fit agreement is a diagnostic, not a certified global pose-error bound. Check the complete transformed payload with unchanged collision/UNKNOWN rules.')
    return PayloadRegistration(_readonly(best['R']), _readonly(best['t']), _readonly(transformed), report)
