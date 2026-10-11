"""Observed-reference placement for a supported object on horizontal tables.

This helper does not infer an object's physical bottom from its visible cloud.
It applies only when the task contract establishes that the object was resting
on the observed source support at the saved, pre-lift TCP pose. The grasp must
remain rigid and the placement orientation must remain the source orientation.
It neither certifies a path nor grants any collision/contact exemption.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabletopPlacement:
    transfer_position: np.ndarray
    release_position: np.ndarray
    rotation: np.ndarray
    reference_local: np.ndarray
    source_tcp_position: np.ndarray
    source_support_z: float
    destination_support_z: float
    clearance_m: float
    rotation_difference_rad: float
    rotation_tolerance_rad: float

    def diagnostic(self):
        return {
            'reference_kind': 'observed_payload_local_bbox_midpoint_not_complete_object_center',
            'reference_local': self.reference_local.tolist(),
            'reference_world_at_release':
                (self.release_position + self.rotation @ self.reference_local).tolist(),
            'source_tcp_position': self.source_tcp_position.tolist(),
            'source_support_z': self.source_support_z,
            'destination_support_z': self.destination_support_z,
            'transfer_position': self.transfer_position.tolist(),
            'release_position': self.release_position.tolist(),
            'rotation': self.rotation.tolist(),
            'rotation_difference_rad': self.rotation_difference_rad,
            'rotation_tolerance_rad': self.rotation_tolerance_rad,
            'release_clearance_m': self.clearance_m,
            'height_reference': 'pre_lift_supported_tcp_plus_support_height_change_and_clearance',
            'required_assumptions': [
                'object_initially_resting_on_observed_horizontal_source_support',
                'rigid_grasp_without_object_slip_or_reorientation',
                'horizontal_destination_support_with_observed_height',
                'placement_uses_saved_source_rotation',
            ],
            'clearance_is_certified_error_bound': False,
            'physical_bottom_inferred': False,
            'collision_exemption': False,
        }


def _vector(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'placement: {name} must be finite with shape {shape}')
    return result.copy()


def _rotation(value, name):
    result = _vector(value, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
            np.linalg.det(result), 1., atol=1e-6, rtol=0):
        raise ValueError(f'placement: {name} must be a proper rotation')
    return result


def tabletop_placement(*, destination_xy, observed_payload_local,
                      source_tcp_position, source_rotation,
                      current_tcp_position, current_rotation,
                      source_support_z, destination_support_z, source_supported,
                      clearance_m=.020, rotation_tolerance_rad=.015):
    """Return TCP goals for an observed object reference at ``destination_xy``.

    ``observed_payload_local`` contains only semantically associated measured
    object points, expressed in the current rigid attachment's TCP frame. Its
    local bounding-box midpoint is an *observed reference*, not a recovered
    physical object center. It must not include unrelated scene points.

    ``source_tcp_position`` and ``source_rotation`` are measured immediately
    after closing and before lifting, while the task's source-support assumption
    still holds. Source and destination support heights come from observed
    horizontal support planes, not from a destination marker's raised surface.
    The caller must explicitly opt into that assumption with source_supported.

    The 20 mm default is an intentional release gap, not a proven bound on
    unknown geometry, support estimation, or execution errors. The orientation
    gate uses the planner's default 0.015 rad IK tolerance; returned goals always
    request the original source rotation. Existing motion checks and independent
    released-object scoring remain necessary.
    """
    if source_supported is not True:
        raise ValueError('placement: initial source support must be established by the task contract')
    destination_xy = _vector(destination_xy, (2,), 'destination_xy')
    source_p = _vector(source_tcp_position, (3,), 'source_tcp_position')
    current_p = _vector(current_tcp_position, (3,), 'current_tcp_position')
    source_R = _rotation(source_rotation, 'source_rotation')
    current_R = _rotation(current_rotation, 'current_rotation')
    points = np.asarray(observed_payload_local, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError('placement: observed_payload_local requires at least four finite 3D points')
    values = np.asarray([source_support_z, destination_support_z, clearance_m,
                         rotation_tolerance_rad], dtype=float)
    if not np.isfinite(values).all() or clearance_m < 0 or not 0 <= rotation_tolerance_rad < np.pi:
        raise ValueError('placement: support heights and nonnegative clearance/tolerance must be finite')
    relative_rotation = current_R @ source_R.T
    angle = float(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))
    if angle > rotation_tolerance_rad + 1e-12:
        raise ValueError('placement: held orientation differs from supported source reference')

    reference_local = (points.min(0) + points.max(0)) / 2
    release = np.empty(3)
    release[:2] = destination_xy - (source_R @ reference_local)[:2]
    release[2] = source_p[2] + float(destination_support_z - source_support_z) + float(clearance_m)
    if current_p[2] < release[2] - 1e-12:
        raise ValueError('placement: current lift clearance is below the supported release height')
    transfer = release.copy()
    transfer[2] = current_p[2]
    return TabletopPlacement(transfer, release, source_R, reference_local, source_p,
        float(source_support_z), float(destination_support_z), float(clearance_m),
        angle, float(rotation_tolerance_rad))
