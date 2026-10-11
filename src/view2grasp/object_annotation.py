"""Bounded annotation of one already acquired source image, without execution.

The caller owns the VLM client and persistence. Every attempt is returned with
its original crop reply, restored source reply, ownership result and diagnostic
log. Replies are never merged and historical ownership is never a current pose.
"""
from dataclasses import dataclass
import copy
import hashlib

import numpy as np

from .annotation_diagnostic import build_annotation_diagnostic
from .grounding import pixels
from .object_ownership import ObjectOwnership, ObjectOwnershipError, resolve_object_ownership
from .observation_crop import crop_observation


@dataclass(frozen=True)
class ObjectAnnotationAttempt:
    index: int
    ownership: ObjectOwnership | None
    log: dict


@dataclass(frozen=True)
class ObjectAnnotationResult:
    attempts: tuple[ObjectAnnotationAttempt, ...]
    report: dict

    @property
    def ownership(self):
        """Only the last reply's independent result; never an earlier fallback."""
        return self.attempts[-1].ownership if self.attempts else None


def _instruction(target_id, description):
    return (
        f'Annotate ONLY the supplied already acquired source-image crop. The target identity is '
        f'{target_id}: {description or target_id}. Mark the complete clearly visible material of '
        'that same object, including any visually identifiable inner wall or inner bottom. '
        'Distinguish actual material from an open hole, support, robot or independent object; '
        'do not complete invisible surfaces or assume material because it lies inside a rim. '
        'Return observe with the supplied frame_id, camera_id and CURRENT view fields. '
        'No camera or robot action from this reply will execute. Use source=vlm and '
        f'target_id={target_id} for ALL proposals. Use ONLY role=object: exactly ONE tight '
        'whole-object box, polygon or mask, plus a SEPARATE kind=points object proposal '
        'with positive seeds on each clearly identifiable visible surface region. '
        'Polygon vertices are not positive seeds. These seeds establish object membership, '
        'never grasp contact. A box can contain background; measured component filters will run. '
        'Omit uncertain clicks. No permitted part is requested: do not provide grasp_part, '
        'grasp_contact, a replacement grasp point, or a grasp action. All normalized coordinates '
        'refer to THIS unresized supplied crop. This is source-pose evidence only, with no claim '
        'that historical points describe a currently held pose. No contact, map retirement or '
        'free-space permission is granted by this annotation request.')


def _correction_feedback(crop, diagnostics, source_reply):
    """Localize actual rejection evidence; never propose replacement pixels."""
    obs = crop.observation
    if (diagnostics.get('frame_id') != obs.frame_id or
            diagnostics.get('camera_id') != obs.camera_id):
        raise ValueError('object annotation feedback must belong to the unchanged source image')
    failures = diagnostics.get('abstentions') or ([diagnostics] if diagnostics.get('failure') else [])
    h, w = obs.depth.shape
    offset = np.asarray(crop.bounds_xyxy[:2])
    seeds = [p for g in source_reply.groundings if g.role == 'object' and g.kind == 'points'
             for p in g.points]
    seed_pixels = pixels(seeds, crop.source_shape).reshape(-1, 2)
    sample_candidates = []
    entries = []
    def localized(source_pixel):
        local = np.asarray(source_pixel, dtype=int) - offset
        if not (0 <= local[0] < w and 0 <= local[1] < h):
            raise ValueError('object annotation rejected pixel lies outside the supplied crop')
        return {'pixel': local.tolist(), 'normalized': (local / [w-1, h-1]).tolist()}

    for failure in failures:
        reasons = failure.get('reasons') or [failure.get('failure', 'geometry_rejection')]
        entry = {key: copy.deepcopy(failure[key]) for key in
            ('failure', 'seed_indices', 'pixels', 'component_pixels', 'original_anchor_pixels',
             'outside_fixed_envelope_pixels', 'rejected_pixel_count', 'anchor_pixels') if key in failure}
        entry['reasons'] = list(reasons)
        component_seeds = [seed_pixels[i].tolist() for i in failure.get('seed_indices', [])]
        if not component_seeds and 'seed_pixel' in failure:
            component_seeds = [failure['seed_pixel']]
        outside_component = 'component_outside_fixed_envelope' in reasons
        invalid_seed = failure.get('failure') == 'invalid_explicit_seed'
        entry['rejected_seed_pixels'] = []
        if component_seeds:
            location_key = 'rejected_seed_pixels' if invalid_seed else 'component_seed_pixels'
            entry[location_key] = [localized(pixel) for pixel in component_seeds]
            if not invalid_seed:
                entry['component_seed_scope'] = (
                    'These region/depth-valid seeds select a rejected component. '
                    'This list alone does not classify them as inside or outside the fixed envelope; '
                    'rejected_geometry_pixels identifies actual measured outlier samples.')
        geometry_pixels = failure.get('rejected_pixels_xy', [])
        entry['rejected_geometry_pixels'] = [localized(pixel) for pixel in geometry_pixels]
        for location in entry['rejected_geometry_pixels']:
            sample_candidates.append((0, {**location, 'sample_kind': 'geometry_outlier',
                'reasons': ['component_outside_fixed_envelope'] if outside_component else list(reasons)}))
        # An envelope violation belongs to its measured outliers, not every
        # in-bounds seed. Keep those seeds as attribution text only, including
        # for historical reports that did not preserve the actual outliers.
        if not outside_component:
            for source_pixel in component_seeds:
                sample_candidates.append((1 if invalid_seed else 2,
                    {**localized(source_pixel), 'sample_kind': 'invalid_seed' if invalid_seed else 'component_seed',
                        'reasons': list(reasons)}))
        entries.append(entry)
    samples = []
    seen = set()
    # Actual outliers take priority over seed attribution across ALL failures.
    # Stable ordering and deduplication preserve the shared 12-point image cap.
    for _, sample in sorted(sample_candidates, key=lambda item: item[0]):
        key = tuple(sample['pixel'])
        if key not in seen:
            seen.add(key)
            if len(samples) < 12:
                samples.append(sample)
    review = []
    for component in diagnostics.get('unresolved_measured_components', []):
        bounds = np.asarray(component['bbox_xyxy'])
        if (bounds.shape != (4,) or not np.issubdtype(bounds.dtype, np.integer)
                or np.any(bounds[:2] > bounds[2:])):
            raise ValueError('object review requires an inclusive measured component rectangle')
        local = bounds - np.tile(offset, 2)
        if np.any(local < 0) or np.any(local > [w-1, h-1, w-1, h-1]):
            raise ValueError('unresolved component lies outside the supplied crop')
        review.append({'pixels': component['pixels'],
            'original_anchor_pixels': component['original_anchor_pixels'],
            'bbox_pixel_xyxy': local.tolist(),
            'bbox_normalized_xyxy': (local / [w-1, h-1, w-1, h-1]).tolist(),
            'bounds_convention': 'both_pixel_endpoints_inclusive',
            'semantic_status': 'unresolved; may be the target, another object or background'})
    return {'frame_id': obs.frame_id, 'camera_id': obs.camera_id,
        'coordinate_space': 'supplied_crop_normalized_pixel_centers',
        'image_shape_hw': [h, w], 'failures': entries, 'rejected_samples': samples,
        'unresolved_measured_components': review,
        'unresolved_measured_component_count': diagnostics.get('unresolved_measured_component_count', 0),
        'component_review_scope': 'These are unseeded depth components inside your previous region. '
            'Geometry only nominates areas for visual inspection; it establishes no object identity, '
            'ownership, contact or free space. Rectangles are not proposed seeds or accepted masks. '
            'Confirm from the unmarked image or leave uncertain components unowned.',
        'scope': 'Actual previous reply and measured geometry only; no replacement seeds, accepted mask or map collision points.'}


def annotate_source_object(vlm, source_obs, original_object, target_id, support_z, *,
                           robot_depth=None, allow_partial_seed_components=False,
                           target_description='') -> ObjectAnnotationResult:
    """At most two semantic replies on one fixed source crop and reference.

    Measured rejection, abstention or unresolved components permit one correction. Invalid
    actions/identity, crop failures and client failures stop conservatively.
    The VLM may have its own bounded transport/schema retries; those are counted
    separately from the maximum two semantic requests here. No observations,
    robot motion, map changes, credential loading or client closure occur here.
    """
    calls_before = getattr(vlm, 'calls', 0)
    seconds_before = getattr(vlm, 'seconds', 0.)
    report = {'frame_id': source_obs.frame_id, 'camera_id': source_obs.camera_id,
        'target_id': target_id, 'maximum_semantic_requests': 2, 'new_observations': 0,
        'map_modified': False, 'contact_permission_changed': False,
        'source_pose_only': True, 'reply_results_merged': False}
    attempts = []

    def finish():
        report.update(semantic_requests=len(attempts),
            vlm_calls=getattr(vlm, 'calls', 0)-calls_before,
            inference_seconds=getattr(vlm, 'seconds', 0.)-seconds_before,
            last_reply_has_ownership=bool(attempts and attempts[-1].ownership is not None),
            attempts=[copy.deepcopy(attempt.log) for attempt in attempts])
        return ObjectAnnotationResult(tuple(attempts), report)

    reference = np.asarray(original_object, dtype=float).copy()
    reference.flags.writeable = False
    if (reference.ndim != 2 or reference.shape[1:] != (3,) or len(reference) < 4 or
            not np.isfinite(reference).all() or not np.isfinite(support_z)):
        report.update(failure='invalid_source_reference',
            error='Source annotation requires four finite original surface points and a finite observed support height')
        return finish()
    try:
        crop = crop_observation(source_obs, reference, robot_depth=robot_depth)
    except (ValueError, RuntimeError) as exc:
        report.update(failure='crop_unavailable', error=str(exc))
        return finish()
    context = crop.provenance()
    report.update(image_context=context,
        source_rgb_sha256=hashlib.sha256(source_obs.rgb.tobytes()).hexdigest(),
        crop_rgb_sha256=hashlib.sha256(crop.observation.rgb.tobytes()).hexdigest(),
        original_reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest())
    instruction = _instruction(target_id, target_description)
    transaction = {'source_object_annotation': {'frame_id': source_obs.frame_id,
        'camera_id': source_obs.camera_id, 'target_id': target_id,
        'image_context': copy.deepcopy(context), 'attempt': 0, 'maximum_semantic_requests': 2}}
    diagnostic = None
    repairable = {'invalid_explicit_seed', 'component_outside_fixed_envelope',
        'unanchored_seeded_component', 'insufficient_original_overlap', 'no_accepted_components'}
    for index in range(2):
        log = {'index': index, 'request_history': copy.deepcopy([transaction]),
            'diagnostic': None if diagnostic is None else diagnostic.provenance()}
        ownership = None
        source_reply = None
        feedback_diagnostics = None
        try:
            raw = vlm.decide(crop.observation, instruction, copy.deepcopy([transaction]),
                force_observe=True, observe_purpose='annotation',
                annotation_roles=('object',),
                derived_image_context=copy.deepcopy(context), annotation_diagnostic=diagnostic)
            log['raw_annotation'] = raw.model_dump(mode='json')
            source_reply = crop.restore_decision(raw)
            log['source_annotation'] = source_reply.model_dump(mode='json')
            ownership = resolve_object_ownership(source_obs, source_reply, reference, target_id,
                support_z, robot_depth, allow_partial_seed_components=allow_partial_seed_components)
            log['ownership'] = copy.deepcopy(ownership.report)
            if (ownership.report.get('abstentions') or
                    ownership.report.get('unresolved_measured_components')):
                feedback_diagnostics = ownership.report
        except ObjectOwnershipError as exc:
            log.update(error=str(exc), diagnostics=copy.deepcopy(exc.diagnostics))
            if exc.diagnostics.get('failure') in repairable:
                feedback_diagnostics = exc.diagnostics
        except Exception as exc:
            # The caller can persist every attempted request even when its
            # shared client exhausted provider retries. No extra semantic retry.
            log.update(error=f'{type(exc).__name__}: {exc}')
        attempts.append(ObjectAnnotationAttempt(index, ownership, log))
        if index == 1 or feedback_diagnostics is None or source_reply is None:
            break
        try:
            feedback = _correction_feedback(crop, feedback_diagnostics, source_reply)
            diagnostic = build_annotation_diagnostic(crop.observation, feedback)
        except (ValueError, RuntimeError) as exc:
            log['correction_unavailable'] = str(exc)
            break
        log['correction_feedback'] = copy.deepcopy(feedback)
        transaction = {'source_object_annotation': {'frame_id': source_obs.frame_id,
            'camera_id': source_obs.camera_id, 'target_id': target_id,
            'image_context': copy.deepcopy(context), 'attempt': 1, 'maximum_semantic_requests': 2,
            'previous_annotation': copy.deepcopy(log['raw_annotation']),
            'geometry_diagnostics': feedback}}
        instruction = _instruction(target_id, target_description) + (
            ' This is the ONE correction of your previous reply on the SAME unmarked crop. '
            'Reinspect the entire visible object and return a fresh complete region plus explicit '
            'positive seeds. Do not mechanically move or just delete rejected clicks. The second '
            'image marks actual rejected pixels and gives a coordinate grid; it provides no object '
            'labels or desired boundaries. Both images share normalized pixel-center coordinates. '
            'For component_outside_fixed_envelope, rejected_geometry_pixels are the actual '
            'outlying depth samples; component_seed_pixels only identify the seeds that selected '
            'that failed component, not whether those seed pixels are inside or outside the envelope. '
            'If geometry_diagnostics lists unresolved_measured_components, inspect those areas '
            'independently in the unmarked image. They are measured pieces inside your own previous '
            'region without a positive seed, not confirmed target material. They may instead belong '
            'to another object or the background. Add an explicit object seed only if you visually '
            'confirm membership; do not copy the diagnostic rectangles as grounding or infer '
            'membership from reference overlap. Visible pieces can be separated by occlusion or '
            'depth discontinuities. Never bridge empty space or mark the gripper as the object. '
            'Use the FIRST unmarked image for all output coordinates. Omit uncertain surfaces. '
            'This reply is evaluated independently; earlier accepted regions are not merged into it.')
    return finish()
