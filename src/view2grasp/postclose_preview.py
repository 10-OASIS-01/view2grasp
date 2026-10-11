"""Shared attachment setup and isolated, bounded post-close lift diagnostics.

The preview produces view-selection evidence, never an executable path or a
free-space certificate. Final lift planning still performs its normal search.
"""
from dataclasses import dataclass, field
import copy
import time

import numpy as np
from scipy.spatial import cKDTree

from .mapping import DepthMap
from .planning import Planner, PlanningBudgetExceeded


@dataclass(frozen=True)
class LiftEvidence:
    target: np.ndarray
    part: np.ndarray
    semantic_contact_only: bool
    current_ownership: tuple = ()
    contact_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    source_ownership: object = None
    source_observation: object = None
    registered_source_points: np.ndarray | None = None


@dataclass
class LiftAttachment:
    attached_part: np.ndarray
    actual_source_points: np.ndarray
    retirements: list


def setup_lift_attachment(planner, map_, position, rotation, evidence):
    """Apply the final attachment policy, identically on real and preview maps.

    Collision envelopes precede endpoint retirement. Only current measured
    ownership authorizes current exact retirement; a registered historical
    source can retire its own endpoints. Predictions never enter the source
    certificate or independent contact part. Retained endpoints and the old
    broad target-retirement rule remain the map's responsibility.
    """
    p, R = np.asarray(position), np.asarray(rotation)
    planner.attached = (evidence.target - p) @ R
    planner.observed_payload_local = None
    planner.registered_payload_local = None
    attached_part = (evidence.part - p) @ R
    planner.attached_contact_local = attached_part.copy() if evidence.semantic_contact_only else None
    measured = np.vstack([*(item.object_points for item, _ in evidence.current_ownership),
        evidence.contact_points])
    actual_source = evidence.target
    if len(measured):
        planner.add_observed_payload(measured, p, R)
        actual_source = np.vstack([actual_source, measured])
    retirements = []
    predicted = evidence.registered_source_points
    if predicted is not None:
        if evidence.source_ownership is None or evidence.source_observation is None:
            raise ValueError('registered payload requires its actual source ownership and observation')
        planner.registered_payload_local = (predicted - p) @ R
        actual_source = np.vstack([actual_source, evidence.source_ownership.object_points])
        removed = map_.retire_owned_source(evidence.source_ownership, evidence.source_observation)
        retirements.append({'kind': 'source_object_retired',
            'frame_id': evidence.source_observation.frame_id, 'exact_endpoints': removed,
            'registered_collision_points': len(predicted)})
    for owned, source in evidence.current_ownership:
        removed = map_.retire_owned_source(owned, source)
        retirements.append({'kind': 'current_object_retired',
            'frame_id': source.frame_id, 'exact_endpoints': removed})
    map_.set_payload_source(actual_source)
    map_.remove_target_endpoints(evidence.target)
    planner.begin_payload_lift()
    return LiftAttachment(attached_part, actual_source, retirements)


def snapshot_depth_map(map_):
    """Copy mutable map state without copying immutable acquired RGB-D frames.

    No observations are integrated into this snapshot. Frame images are
    immutable classification inputs; masks and their containers are copied.
    All arrays, trees, certificates and retention state that setup can replace
    or mutate are copied. In particular this never clears dynamic evidence.
    """
    result = DepthMap.__new__(DepthMap)
    for name, value in vars(map_).items():
        if name == 'views':
            value = list(value)
        elif isinstance(value, cKDTree):
            value = cKDTree(value.data.copy())
        else:
            value = copy.deepcopy(value)
        setattr(result, name, value)
    return result


@dataclass
class LiftPreview:
    robot_unknown: np.ndarray
    payload_unknown: np.ndarray
    report: dict
    surface: np.ndarray


def preview_lift(planner, evidence, start, gripper, *, time_budget=3.):
    """Check one nominal 14-waypoint lift on an isolated evidence snapshot.

    Uses the ordinary valid/edge checks and their post-policy rejection states.
    There is no redundant-IK search, RNG consumption, motion, or live-map write.
    The cooperative deadline includes setup; a single existing IK/valid call
    can finish after it. Only an actual complete rejection supplies UNKNOWN
    samples. A timeout, IK failure or self collision is not free-space evidence.
    """
    if not np.isfinite(time_budget) or time_budget <= 0:
        raise ValueError('lift preview requires a positive finite time budget')
    started = time.monotonic()
    robot = planner.robot
    saved_q = robot.data.qpos[robot.qidx].copy()
    saved_gripper = robot.data.qpos[robot.gq].copy() if len(robot.gq) == 2 else None
    private = None
    diagnostic = None
    completed = 0
    stage = 'setup'
    failure = ''
    timed_out = False
    error = None
    setup = None
    try:
        private = Planner(robot, snapshot_depth_map(planner.map), copy.deepcopy(planner.rng))
        private.gripper = np.asarray(gripper).copy()
        private.deadline = started + time_budget
        robot.set_q(start, gripper)
        p, R = robot.pose()
        setup = setup_lift_attachment(private, private.map, p, R, evidence)
        private.allow_payload_support = True
        previous = np.asarray(start).copy()
        for dz in np.linspace(.01, .13, 14):
            if time.monotonic() >= private.deadline:
                raise PlanningBudgetExceeded
            stage = 'IK'
            q = robot.ik(p + [0., 0., dz], R, previous)
            if q is None:
                failure = 'IK'
                break
            if time.monotonic() >= private.deadline:
                raise PlanningBudgetExceeded
            stage = 'valid'
            if not private.valid(q, contact=True):
                failure = private.last_failure
                diagnostic = copy.deepcopy(private.last_validation)
                break
            stage = 'edge'
            if not private.edge(previous, q, contact=True):
                failure = private.last_failure
                diagnostic = copy.deepcopy(private.last_validation)
                break
            previous = q
            completed += 1
    except PlanningBudgetExceeded:
        failure = 'planning budget exhausted'
        timed_out = True
    except Exception as exc:
        # Advice is optional; preserve final planning and expose the incomplete
        # diagnostic rather than substituting a less strict geometric sweep.
        failure = 'preview error'
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if private is not None:
            private.allow_payload_support = False
            private.end_payload_lift()
        robot.set_q(saved_q, saved_gripper)
    elapsed = time.monotonic() - started
    timed_out = timed_out or elapsed > time_budget
    rejected = (diagnostic is not None and diagnostic.get('complete') is True
        and diagnostic.get('valid') is False)
    def points(key):
        return (np.asarray(diagnostic.get(key, []), dtype=float).reshape(-1, 3)
            if rejected else np.empty((0, 3)))
    robot_unknown = points('robot_unknown_points')
    payload_unknown = points('payload_unknown_points')
    report = {'purpose': 'strict_postclose_lift_view_selection',
        'start_q': np.asarray(start).tolist(), 'gripper_q': np.asarray(gripper).tolist(),
        'nominal_path_valid': completed == 14 and not failure and not timed_out,
        'incomplete': completed < 14 or timed_out, 'completed_waypoints': completed,
        'total_waypoints': 14, 'failure_stage': stage if failure else None,
        'failure': failure, 'elapsed_seconds': elapsed, 'time_budget_seconds': time_budget,
        'timed_out': timed_out, 'checks': 0 if private is None else private.checks,
        'last_validation': diagnostic if rejected else None,
        'robot_unknown_points': robot_unknown.tolist(),
        'payload_unknown_points': payload_unknown.tolist(),
        'occupied_unresolved': bool(rejected and (
            diagnostic.get('robot_occupied_count', 0) or diagnostic.get('payload_occupied_count', 0))),
        'mapping_evidence_available': bool(len(robot_unknown) or len(payload_unknown)),
        'retirements_on_snapshot': [] if setup is None else setup.retirements,
        'attachment_evidence': {'target_points': len(evidence.target),
            'independent_part_points': len(evidence.part),
            'current_ownership_frames': [source.frame_id for _, source in evidence.current_ownership],
            'current_owned_points': sum(len(owned.object_points) for owned, _ in evidence.current_ownership),
            'current_contact_object_points': len(evidence.contact_points),
            'registered_collision_points': (0 if evidence.registered_source_points is None else
                len(evidence.registered_source_points)),
            'actual_source_points': 0 if setup is None else len(setup.actual_source_points)},
        'scope': ('Nominal checked prefix only. UNKNOWN samples come from ordinary strict checks; '
            'observed obstacles, retained endpoints and self collisions remain unresolved. '
            'No motion or live evidence change; final lift planning is still required.')}
    if error is not None:
        report['error'] = error
    surface = planner.map.surface.copy() if private is None else private.map.surface.copy()
    return LiftPreview(robot_unknown, payload_unknown, report, surface)
