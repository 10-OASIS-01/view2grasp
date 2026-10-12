"""Bounded transfer diagnostics from actual planner refusals and observations."""
from collections import Counter
from dataclasses import dataclass
import copy
import time

import numpy as np

from .mapping import UNKNOWN
from .planning import PlanningBudgetExceeded


@dataclass(frozen=True)
class TabletopObservationBudget:
    pregrasp_limit: int
    postclose_limit: int
    postclose_observations: int
    postlift_observations: int


def tabletop_observation_budget(task, maximum, postlift_observations=0):
    """Reserve named phase allowances before any grasp recovery acquires images."""
    reserved = postlift_observations if task == 'occluded' else 0
    if reserved < 0 or reserved > 2 or maximum - reserved < 2:
        raise ValueError('post-lift reserve requires an initial and a post-close observation')
    # Mug budgets above ten keep the seven-view pregrasp cap. Extra views
    # verify the full registered payload after its pose becomes observable.
    desired = min(5,3+max(0,maximum-10)) if task=='mug' else 2
    postclose = min(desired, maximum - reserved - 1)
    return TabletopObservationBudget(maximum - reserved - postclose,
        maximum - reserved, postclose, reserved)


def different_tabletop_view(view, current):
    """Camera-only recovery requires a new view, including azimuth wraparound."""
    delta = np.asarray(view, dtype=float) - np.asarray(current, dtype=float)
    delta[0] = (delta[0] + 180.) % 360. - 180.
    return bool(np.any(np.abs(delta) > 1e-6))


@dataclass
class TransferAttempt:
    path: np.ndarray | None
    unknown_points: np.ndarray
    report: dict
    rng_state: dict


def _transfer_unknown(planner):
    """Prefer the closest rejected branch, and recheck every sample is UNKNOWN."""
    points = []
    closest = planner.closest_by_failure.get('unknown space')
    if closest is not None:
        points.extend(closest.get('points', []))
    payload = planner.payload_invalid
    if payload is not None and payload.get('unknown', 0):
        points.extend(payload.get('points', []))
    if not points:
        points = np.asarray(planner.recovery_unknown()).reshape(-1, 3).tolist()
    if not points:
        return np.empty((0, 3))
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[planner.map.classify(points) == UNKNOWN]
    return np.unique(points, axis=0)


def plan_transfer_attempt(planner, start, position, rotation, *, time_budget, rrt_timeout=5.):
    """Plan normally, preserving refusals for one evidence-driven mapping retry.

    No nominal path or sampled unknown branch is approved here. The returned
    path is exclusively the planner's fully checked path. The RNG checkpoint
    lets the caller retry the same search after acquiring real new evidence.
    """
    rng_state = copy.deepcopy(planner.rng.bit_generator.state)
    prior_deadline = planner.deadline
    timer = time.monotonic()
    planner.last_failure = '';planner.last_bad = 0;planner.best_bad = 100000
    planner.best_unknown = None;planner.closest_invalid = None;planner.closest_by_failure = {}
    planner.unknown_samples = [];planner.unknown_sample_count = 0;planner.payload_invalid = None
    planner.failure_counts = Counter()
    checks_before = planner.checks
    planner.deadline = timer + time_budget
    if prior_deadline is not None:
        planner.deadline = min(planner.deadline, prior_deadline)
    path = None;goal = None
    try:
        goal = planner.robot.ik(position, rotation, start)
        if goal is None:
            planner.last_failure = 'IK'
        else:
            remaining = planner.deadline - time.monotonic()
            if remaining <= 0:
                raise PlanningBudgetExceeded
            path = planner.plan(start, goal, timeout=min(rrt_timeout, remaining))
    except PlanningBudgetExceeded:
        planner.last_failure = 'planning budget exhausted'
    finally:
        planner.deadline = prior_deadline
    elapsed = time.monotonic() - timer
    unknown = _transfer_unknown(planner) if path is None else np.empty((0, 3))
    recoverable_reason = planner.last_failure in (
        'unknown space', 'carried object unknown space', 'RRT timeout', 'planning budget exhausted')
    report = {'phase': 'transfer', 'start_q': np.asarray(start).tolist(),
        'goal_q': None if goal is None else np.asarray(goal).tolist(),
        'destination': np.asarray(position).tolist(), 'rotation': np.asarray(rotation).tolist(),
        'path_found': path is not None, 'failure': '' if path is not None else planner.last_failure,
        'elapsed_seconds': elapsed, 'planning_budget_seconds': time_budget,
        'rrt_timeout_seconds': rrt_timeout, 'checks': planner.checks - checks_before,
        'closest_by_failure': copy.deepcopy(planner.closest_by_failure),
        'payload_invalid': copy.deepcopy(planner.payload_invalid),
        'rejected_unknown_points': unknown.tolist(),
        'observation_recovery_eligible': bool(path is None and recoverable_reason and len(unknown)),
        'scope': 'Actual rejected samples only; proposed views do not certify free space.'}
    return TransferAttempt(path, unknown, report, rng_state)
