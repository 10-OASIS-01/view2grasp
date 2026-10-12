"""Bounded, explicit multi-view initialization before any robot execution.

New seeded source surfaces may establish initial object membership without an
old anchor in each component. That is fresh VLM semantic authority, NOT a
successful later reassociation or independent geometric identity proof. The
side reference alone supplies the overlap test and fixed admission envelope.
Callers must freeze initialization before execution; this module never revises
retained evidence from an earlier ownership transaction.
"""
import copy
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .annotation_diagnostic import build_annotation_diagnostic
from .geometry import project, unproject
from .grounding import pixels, proposal_mask
from .object_annotation import (ObjectAnnotationAttempt, ObjectAnnotationResult,
                                _correction_feedback, _instruction)
from .object_ownership import (ANCHOR_DISTANCE_M, COMPONENT_STEP_M, REFERENCE_MARGIN_M,
    SOURCE_EQUALITY_M, ObjectOwnership, ObjectOwnershipError, _digest, _readonly, _source_digest)
from .observation_crop import _validate_source, crop_observation


@dataclass(frozen=True)
class InitialReferenceResult(ObjectAnnotationResult):
    initial_part_points: np.ndarray | None = None
    initial_part_mask: np.ndarray | None = None


def select_initial_part_view(observations, side_obs, side_part, support_z, robot_depths,
                             *, excluded_frames=(), resolution=.005, known_part_points=None):
    """Rank existing static images for fresh semantic review, never membership.

    Nearby measured material absent from the original part is a hypothesis to
    inspect. It may be another object; only the subsequent VLM transaction can
    label it. No unseen scene image, new camera or post-execution frame enters.
    """
    reference=np.asarray(side_part)
    if (reference.ndim!=2 or reference.shape[1:]!=(3,) or len(reference)<4
            or not np.isfinite(reference).all() or not np.isfinite(support_z)
            or not np.isfinite(resolution) or resolution<=0):
        raise ValueError('part view selection requires a finite measured reference and positive resolution')
    known=reference if known_part_points is None else np.asarray(known_part_points)
    if known.ndim!=2 or known.shape[1:]!=(3,) or len(known)<4 or not np.isfinite(known).all():
        raise ValueError('part novelty ranking requires finite confirmed material')
    tree=cKDTree(known)
    lo,hi=reference.min(0)-REFERENCE_MARGIN_M,reference.max(0)+REFERENCE_MARGIN_M
    ranked=[];sources={}
    for obs in observations:
        if obs.frame_id in excluded_frames or (obs.frame_id,obs.camera_id)==(side_obs.frame_id,side_obs.camera_id):
            continue
        own=robot_depths.get(obs.frame_id)
        if own is None:continue
        try:
            _validate_initial_sources(obs,side_obs,reference,support_z,own)
            crop=crop_observation(obs,reference,robot_depth=own)
        except (ValueError,RuntimeError):continue
        current=crop.observation;y,x=np.mgrid[:current.depth.shape[0],:current.depth.shape[1]]
        xyz=unproject(np.stack([x,y],-1),current.depth,current.K,current.T_wc)
        valid=(np.isfinite(xyz).all(2)&(current.depth>.05)&(current.depth<10)
            &(xyz[:,:,2]>support_z+.006)&np.isfinite(crop.robot_depth)
            &(current.depth<crop.robot_depth-.003)&np.all((xyz>=lo)&(xyz<=hi),axis=2))
        points=xyz[valid]
        novel=points[tree.query(points)[0]>=ANCHOR_DISTANCE_M] if len(points) else points
        voxels=len(np.unique(np.round(novel/resolution).astype(np.int64),axis=0))
        ranked.append({'frame_id':obs.frame_id,'camera_id':obs.camera_id,
            'measured_candidate_pixels':len(points),'novel_candidate_voxels':voxels})
        sources[obs.frame_id]=obs
    ranked.sort(key=lambda r:(-r['novel_candidate_voxels'],r['frame_id']))
    chosen=sources[ranked[0]['frame_id']] if ranked and ranked[0]['novel_candidate_voxels'] else None
    return chosen,{'scope':'observed geometry ranks a semantic review only; no part or object identity granted',
        'selected_frame':None if chosen is None else chosen.frame_id,'ranked_frames':ranked,
        'novelty_reference_sha256':_digest(known),'fixed_side_part_sha256':_digest(reference),
        'new_observations':0,'maximum_additional_source_images':1}


def _initial_part_selection(obs, decision, target_id, side_part, support_z, robot_depth):
    """Check part geometry independently so the sole correction sees all faults."""
    proposals=[g for g in decision.groundings if g.role=='grasp_part']
    def reject(failure,message,**details):
        raise ObjectOwnershipError(message,{'frame_id':obs.frame_id,'camera_id':obs.camera_id,
            'failure':failure,'scope':'fresh_initial_part_semantics','partial':False,**details})
    if (len(proposals)!=1 or proposals[0].kind not in ('box','polygon','mask')
            or any(g.source!='vlm' or g.target_id!=target_id
                or g.frame_id!=obs.frame_id or g.camera_id!=obs.camera_id for g in proposals)):
        reject('initial_part_region_required','provide one explicit same-target grasp_part region')
    proposal=proposals[0];mask=proposal_mask(proposal,obs)
    for x,y in pixels(proposal.negative_points,obs.depth.shape).reshape(-1,2):
        mask[max(0,y-2):y+3,max(0,x-2):x+3]=False
    y,x=np.mgrid[:obs.depth.shape[0],:obs.depth.shape[1]]
    xyz=unproject(np.stack([x,y],-1),obs.depth,obs.K,obs.T_wc)
    valid=(np.isfinite(xyz).all(2)&(obs.depth>.05)&(obs.depth<10)&(xyz[:,:,2]>support_z+.006))
    if robot_depth is not None:valid&=np.isfinite(robot_depth)&(obs.depth<robot_depth-.003)
    selected=mask&valid
    points=xyz[selected]
    if len(points)<4:reject('initial_part_insufficient_material','part region contains fewer than four measured material pixels')
    lo,hi=side_part.min(0)-REFERENCE_MARGIN_M,side_part.max(0)+REFERENCE_MARGIN_M
    outside=np.any((points<lo)|(points>hi),axis=1)
    if outside.any():
        reject('initial_part_outside_fixed_envelope','complete part selection leaves the fixed initial side-part envelope',
            rejected_pixel_count=int(outside.sum()),rejected_pixels_xy=np.argwhere(selected)[outside][:12,::-1].tolist())
    report={'scope':'fresh_initial_same_part_semantics_before_execution','pixels':len(points),
        'membership_authority':'explicit VLM grasp_part region inside independently seeded object material',
        'fixed_side_part_sha256':_digest(side_part),'fixed_side_part_aabb_m':[lo.tolist(),hi.tolist()],
        'part_side_anchor_pixels':int(np.count_nonzero(cKDTree(side_part).query(points)[0]<ANCHOR_DISTANCE_M)),
        'part_component_anchor_gate_applied':False,'ray_changed':False,
        'semantic_identity_limit':'The initial same-part label is a VLM assumption, not geometric proof of hidden part identity.'}
    return _readonly(points),_readonly(selected),report


def _validate_initial_part_ownership(obs, selected, ownership):
    outside=selected&~ownership.object_mask
    if outside.any():
        raise ObjectOwnershipError('part material must belong to the independently seeded initial object',
            {'frame_id':obs.frame_id,'camera_id':obs.camera_id,
             'failure':'initial_part_outside_owned_object','scope':'fresh_initial_part_semantics','partial':False,
             'rejected_pixel_count':int(outside.sum()),'rejected_pixels_xy':np.argwhere(outside)[:12,::-1].tolist()})


def _validate_initial_sources(source_obs, side_obs, side_object, support_z, robot_depth):
    """Verify static observation provenance, never infer semantic membership."""
    _validate_source(source_obs, robot_depth)
    _validate_source(side_obs, None)
    if (source_obs.frame_id, source_obs.camera_id) == (side_obs.frame_id, side_obs.camera_id):
        raise ValueError('initial reference requires distinct source and side observations')
    if (not np.isfinite(source_obs.timestamp) or not np.isfinite(side_obs.timestamp)
            or source_obs.timestamp != side_obs.timestamp):
        raise ValueError('initial reference requires the same static simulation timestamp')
    for name in ('q', 'gripper_q', 'base_pose', 'torso_q'):
        source, side = getattr(source_obs, name), getattr(side_obs, name)
        if source is None or side is None:
            if source is not None or side is not None:
                raise ValueError(f'initial reference requires unchanged {name}')
            if name in ('q', 'gripper_q'):
                raise ValueError(f'initial reference requires measured {name}')
        elif (not np.isfinite(source).all() or not np.isfinite(side).all()
                or not np.array_equal(source, side)):
            raise ValueError(f'initial reference requires unchanged {name}')
    reference = _readonly(np.asarray(side_object, dtype=float))
    if (reference.ndim != 2 or reference.shape[1:] != (3,) or len(reference) < 4
            or not np.isfinite(reference).all() or not np.isfinite(support_z)):
        raise ValueError('initial reference requires four finite side points and observed support')
    # A caller must retain the actual grounding observation, not a later
    # mapping frame or a cloud projected behind its visible depth surface.
    uv, z = project(reference, side_obs.K, side_obs.T_wc)
    h, w = side_obs.depth.shape
    if (not np.isfinite(uv).all() or np.any(z <= 0)
            or np.any(uv < -.5) or np.any(uv >= [w-.5, h-.5])):
        raise ValueError('side reference is not measured in the supplied side observation')
    ij = np.rint(uv).astype(int)
    depth = side_obs.depth[ij[:, 1], ij[:, 0]]
    measured = unproject(ij, depth, side_obs.K, side_obs.T_wc)
    if (not np.isfinite(measured).all() or np.any((depth <= .05) | (depth >= 10))
            or np.any(measured[:, 2] <= support_z + .006)
            or np.any(np.linalg.norm(measured-reference, axis=1) > SOURCE_EQUALITY_M)):
        raise ValueError('side reference must match actual side-image depth endpoints')
    return reference


def _resolve_initial_object_reference(obs, decision, side_reference, target_id, support_z,
                                      robot_depth=None):
    """Atomic initial semantic selection; no self-anchoring or partial result."""
    report = {'frame_id': obs.frame_id, 'camera_id': obs.camera_id, 'target_id': target_id,
        'scope': 'initial_multiview_object_reference', 'initial_semantic_authority': True,
        'later_component_anchor_gate_applied': False,
        'anchor_reference': 'immutable_side_observation_only',
        'grants_contact_permission': False, 'grasp_ray_changed': False,
        'map_modified': False, 'grants_free_space': False,
        'allow_partial_seed_components': False, 'partial': False, 'abstentions': [],
        'source_pose_only': True, 'rigid_attachment_mapping_required_before_carrying': True,
        'semantic_identity_limit': 'Explicit same-object VLM semantics establish new initial '
            'surfaces. Geometry cannot prove that a separately seeded neighboring object was mislabeled.'}

    def reject(failure, message, **details):
        raise ObjectOwnershipError(message, {**report, 'failure': failure, **details})

    if decision.frame_id != obs.frame_id or decision.camera_id != obs.camera_id:
        reject('source_mismatch', 'initial annotation must identify its supplied source image')
    if decision.action != 'observe':
        reject('annotation_action_required', 'initial object reference requires an observe annotation')
    proposals = decision.groundings
    if not proposals or any(g.role != 'object' or g.source != 'vlm' or g.target_id != target_id
            or g.frame_id != obs.frame_id or g.camera_id != obs.camera_id for g in proposals):
        reject('semantic_identity', 'initial reference requires only same-target VLM object proposals')
    regions = [g for g in proposals if g.kind in ('box', 'polygon', 'mask')]
    seeds = [p for g in proposals if g.kind == 'points' for p in g.points]
    if (len(regions) != 1 or not seeds
            or any(g.kind not in ('box', 'polygon', 'mask', 'points') for g in proposals)
            or any(g.points and g.kind in ('box', 'mask') for g in proposals)):
        reject('region_and_seeds_required', 'initial reference needs one region and separate explicit object seeds')
    h, w = obs.depth.shape
    region = proposal_mask(regions[0], obs)
    semantic = region.copy()
    for proposal in proposals:
        for x, y in pixels(proposal.negative_points, (h, w)).reshape(-1, 2):
            semantic[max(0, y-2):y+3, max(0, x-2):x+3] = False
    y, x = np.mgrid[:h, :w]
    xyz = unproject(np.stack([x, y], -1), obs.depth, obs.K, obs.T_wc)
    depth_valid = (np.isfinite(obs.depth) & (obs.depth > .05) & (obs.depth < 10)
                   & np.isfinite(xyz).all(axis=2))
    support_clear = xyz[:, :, 2] > support_z + .006
    robot_clear = (np.ones((h, w), bool) if robot_depth is None else
                   np.isfinite(robot_depth) & (obs.depth < robot_depth - .003))
    candidate = semantic & depth_valid & support_clear & robot_clear
    seed_pixels = pixels(seeds, (h, w)).reshape(-1, 2)
    invalid_seeds = []
    for index, (xx, yy) in enumerate(seed_pixels):
        if candidate[yy, xx]:
            continue
        reasons = []
        if not region[yy, xx]: reasons.append('outside_region_or_hole')
        elif not semantic[yy, xx]: reasons.append('explicit_negative')
        if not depth_valid[yy, xx]: reasons.append('invalid_depth')
        if depth_valid[yy, xx] and not support_clear[yy, xx]: reasons.append('support')
        if not robot_clear[yy, xx]: reasons.append('robot_occluded_or_unverified')
        invalid_seeds.append({'failure': 'invalid_explicit_seed',
            'seed_indices': [index], 'seed_pixel': [int(xx), int(yy)], 'reasons': reasons})
    failures = list(invalid_seeds)

    lo = side_reference.min(axis=0)-REFERENCE_MARGIN_M
    hi = side_reference.max(axis=0)+REFERENCE_MARGIN_M
    tree = cKDTree(side_reference)  # Never union the proposed source cloud here.
    report.update(original_reference_sha256=_digest(side_reference),
        side_reference_sha256=_digest(side_reference), fixed_side_aabb_m=[lo.tolist(), hi.tolist()],
        fixed_original_aabb_m=[lo.tolist(), hi.tolist()], support_z_m=float(support_z),
        thresholds={'aabb_margin_m': REFERENCE_MARGIN_M, 'component_step_inclusive_m': COMPONENT_STEP_M,
            'anchor_distance_exclusive_m': ANCHOR_DISTANCE_M, 'minimum_total_side_anchor_pixels': 4},
        proposed_region_pixels=int(semantic.sum()), valid_region_pixels=int(candidate.sum()))
    labels = np.full((h, w), -1, int)
    components = []
    for index, (xx, yy) in enumerate(seed_pixels):
        if not candidate[yy, xx]:
            continue
        if labels[yy, xx] >= 0:
            components[labels[yy, xx]]['seed_indices'].append(index)
            continue
        label = len(components)
        labels[yy, xx] = label
        stack = [(yy, xx)]
        component = []
        while stack:
            py, px = stack.pop()
            component.append((py, px))
            for ny, nx in ((py-1, px), (py+1, px), (py, px-1), (py, px+1)):
                if (0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and labels[ny, nx] < 0
                        and np.linalg.norm(xyz[ny, nx]-xyz[py, px]) <= COMPONENT_STEP_M):
                    labels[ny, nx] = label
                    stack.append((ny, nx))
        yx = np.asarray(component, int)
        points = xyz[yx[:, 0], yx[:, 1]]
        outside = np.any((points < lo) | (points > hi), axis=1)
        anchors = int(np.count_nonzero(tree.query(points)[0] < ANCHOR_DISTANCE_M))
        component_seeds = [index]
        if outside.any():
            failures.append({'failure': 'component_outside_fixed_envelope',
                'seed_indices': component_seeds, 'component_pixels': len(points), 'original_anchor_pixels': anchors,
                'rejected_pixel_count': int(outside.sum()), 'rejected_pixels_xy': yx[outside][:12, ::-1].tolist()})
        components.append({'pixels': len(points), 'seed_indices': component_seeds,
            'side_anchor_pixels': anchors, 'original_anchor_pixels': anchors,
            'membership_authority': 'fresh_initial_vlm_semantics',
            'independent_side_overlap': bool(anchors),
            'satisfies_later_four_anchor_component_gate': anchors >= 4})
    if failures:
        # One correction sees every actual seed/component failure together.
        # Traversal is diagnostic only: any failure rejects the whole answer.
        first = failures[0]
        reject(first['failure'], 'initial object selection contains invalid seeds or out-of-envelope components',
            **{k: v for k, v in first.items() if k != 'failure'},
            invalid_seed_count=len(invalid_seeds), abstentions=failures)
    total_anchors = sum(component['side_anchor_pixels'] for component in components)
    if total_anchors < 4:
        reject('insufficient_original_overlap', 'initial source selection needs four anchors against the side reference',
            anchor_pixels=total_anchors, side_anchor_pixels=total_anchors,
            seed_indices=list(range(len(seed_pixels))))
    selected = labels >= 0
    yy, xx = np.where(selected)
    report.update(selected_components=components, anchor_pixels=total_anchors,
        side_anchor_pixels=total_anchors, object_pixels=len(yy),
        novel_semantic_components=sum(component['side_anchor_pixels'] == 0 for component in components),
        components_below_later_anchor_gate=sum(component['side_anchor_pixels'] < 4 for component in components),
        unselected_valid_region_pixels=int(np.count_nonzero(candidate & ~selected)),
        source_timestamp=float(obs.timestamp), abstained_source_pixels=0)
    return ObjectOwnership(_readonly(selected), _readonly(xyz[selected]),
        _readonly(np.column_stack([xx, yy])), _readonly(obs.depth[selected]),
        obs.frame_id, obs.camera_id, _source_digest(obs),
        _readonly(np.empty((0, 2), int)), _readonly(np.empty((0, 3))), report)


def annotate_initial_object_reference(vlm, source_obs, side_obs, side_object, target_id,
                                     support_z, *, robot_depth=None,
                                     target_description='', side_part=None) -> ObjectAnnotationResult:
    """At most two fresh semantic replies on an existing static source image.

    Returns ObjectAnnotationResult; only its final attempt can supply ownership.
    The caller must invoke this once during initialization of a new episode,
    before execution or any ownership/retention transaction for this source.
    There is no lifecycle ledger here and no permission to reset one externally.
    """
    calls_before, seconds_before = getattr(vlm, 'calls', 0), getattr(vlm, 'seconds', 0.)
    attempts = []
    initial_part_points=initial_part_mask=None
    report = {'frame_id': source_obs.frame_id, 'camera_id': source_obs.camera_id,
        'side_frame_id': side_obs.frame_id, 'side_camera_id': side_obs.camera_id,
        'target_id': target_id, 'scope': 'initial_multiview_object_reference',
        'maximum_semantic_requests': 2, 'new_observations': 0, 'map_modified': False,
        'contact_permission_changed': False, 'grasp_ray_changed': False,
        'reply_results_merged': False, 'source_pose_only': True}

    def finish():
        report.update(semantic_requests=len(attempts), vlm_calls=getattr(vlm, 'calls', 0)-calls_before,
            inference_seconds=getattr(vlm, 'seconds', 0.)-seconds_before,
            last_reply_has_ownership=bool(attempts and attempts[-1].ownership is not None),
            attempts=[copy.deepcopy(attempt.log) for attempt in attempts])
        return InitialReferenceResult(tuple(attempts),report,initial_part_points,initial_part_mask)

    try:
        reference = _validate_initial_sources(source_obs, side_obs, side_object, support_z, robot_depth)
        if side_part is not None:
            side_part=_validate_initial_sources(source_obs,side_obs,side_part,support_z,robot_depth)
            if np.any(cKDTree(reference).query(side_part)[0]>SOURCE_EQUALITY_M):
                raise ValueError('initial side part must be a subset of the measured initial side object')
        crop = crop_observation(source_obs, reference, robot_depth=robot_depth)
    except (ValueError, RuntimeError) as exc:
        report.update(failure='invalid_initial_evidence', error=str(exc))
        return finish()
    report.update(static_epoch_verified=True, source_evidence_sha256=_source_digest(source_obs),
        side_evidence_sha256=_source_digest(side_obs), side_reference_sha256=_digest(reference),
        static_state_sha256=_digest(source_obs.q, source_obs.gripper_q), image_context=crop.provenance())
    base_instruction = _instruction(target_id, target_description) + (
        ' This is INITIAL multi-view object membership before any robot execution. Explicitly '
        'seed each separate clearly visible surface of the same object, including identifiable '
        'inner material. Each seed selects its whole depth-connected surface: use a small '
        'number of confident interior clicks, not a dense outline or repeated clicks on '
        'one surface. Stay off silhouette edges and thin open gaps; any invalid seed rejects '
        'the whole answer. Do not seed an independent object, support or the robot. No earlier '
        'unowned component is being automatically reclassified; the supplied crop is the only '
        'image for this fresh annotation. The independent side image establishes a fixed '
        'geometric admission envelope and overall overlap, not labels for new source surfaces.')
    roles=('object',)
    if side_part is not None:
        roles=('object','grasp_part')
        base_instruction=base_instruction.replace('Use ONLY role=object:', 'For the object, use role=object:')
        base_instruction=base_instruction.replace(
            'No permitted part is requested: do not provide grasp_part, grasp_contact, a replacement grasp point, or a grasp action.',
            'Also provide exactly ONE separate grasp_part box, polygon or mask on the clearly visible straight outer vertical handle bar around its midpoint, excluding both junctions and cup body. This confirms the SAME requested part from a complementary STATIC image before execution. Explicitly seed object material containing this part. Do not provide grasp_contact, a replacement contact point, or a grasp action.')
        base_instruction=base_instruction.replace(
            'No contact, map retirement or free-space permission is granted by this annotation request.',
            'Only the independently validated grasp_part material may extend the initial part reference; whole-object labels never do. This request grants no map retirement or free-space permission.')
        base_instruction+=(' The grasp_part region is a material mask, not a loose detection box: '
            'every depth-valid pixel must be on the requested bar and inside independently seeded '
            'object material. Keep all edges strictly inside the visible bar, away from its hole '
            'and background; use a narrow polygon when needed. A box crossing a hole is rejected.')
        report['initial_part_reference']=True
    transaction = {'initial_object_reference': {'frame_id': source_obs.frame_id,
        'camera_id': source_obs.camera_id, 'target_id': target_id, 'attempt': 0,
        'maximum_semantic_requests': 2, 'image_context': copy.deepcopy(crop.provenance())}}
    instruction = base_instruction
    # The initial crop can be only ~200 pixels wide. Supply the same normalized
    # pixel-center magnification on the first reply too, without any labels or
    # rejected pixels. Depth, K, crop restoration and admission are unchanged.
    diagnostic = build_annotation_diagnostic(crop.observation, {
        'frame_id': source_obs.frame_id, 'camera_id': source_obs.camera_id,
        'coordinate_space': 'supplied_crop_normalized_pixel_centers',
        'image_shape_hw': list(crop.observation.depth.shape), 'rejected_samples': []})
    repairable = {'invalid_explicit_seed', 'component_outside_fixed_envelope', 'insufficient_original_overlap',
        'initial_part_outside_owned_object','initial_part_insufficient_material','initial_part_outside_fixed_envelope'}
    for index in range(2):
        log = {'index': index, 'request_history': copy.deepcopy([transaction]),
            'diagnostic': None if diagnostic is None else diagnostic.provenance()}
        ownership = source_reply = rejection = None
        try:
            raw = vlm.decide(crop.observation, instruction, copy.deepcopy([transaction]),
                force_observe=True, observe_purpose='annotation',
                annotation_roles=('object',) if side_part is None else None,
                derived_image_context=copy.deepcopy(crop.provenance()), annotation_diagnostic=diagnostic)
            log['raw_annotation'] = raw.model_dump(mode='json')
            source_reply = crop.restore_decision(raw)
            log['source_annotation'] = source_reply.model_dump(mode='json')
            if any(g.role not in roles for g in source_reply.groundings):
                raise ValueError('unexpected role in initial reference')
            object_reply=source_reply.model_copy(update={'groundings':[g for g in source_reply.groundings if g.role=='object']})
            geometry_errors=[]
            try:
                ownership = _resolve_initial_object_reference(source_obs, object_reply, reference,
                    target_id, support_z, robot_depth)
            except ObjectOwnershipError as exc:
                geometry_errors.append(exc)
            log['object_geometry_passed']=ownership is not None
            if side_part is not None:
                try:
                    part_points,part_mask,part_report=_initial_part_selection(
                        source_obs,source_reply,target_id,side_part,support_z,robot_depth)
                    if ownership is not None:
                        _validate_initial_part_ownership(source_obs,part_mask,ownership)
                except ObjectOwnershipError as exc:
                    geometry_errors.append(exc)
            if geometry_errors:
                # Protocol errors cannot gain a retry through another role's
                # repairable geometry. No provisional point set is returned.
                primary=next((e for e in geometry_errors if e.diagnostics['failure'] not in repairable),geometry_errors[0])
                if len(geometry_errors)==1:
                    raise primary
                details=copy.deepcopy(primary.diagnostics)
                details['abstentions']=[f for e in geometry_errors
                    for f in (e.diagnostics.get('abstentions') or [e.diagnostics])]
                raise ObjectOwnershipError('initial object and part selection failed independently',details)
            if side_part is not None:
                initial_part_points,initial_part_mask=part_points,part_mask
                log['initial_part']=part_report
            log['ownership'] = copy.deepcopy(ownership.report)
        except ObjectOwnershipError as exc:
            ownership=None;initial_part_points=initial_part_mask=None
            log.update(error=str(exc), diagnostics=copy.deepcopy(exc.diagnostics))
            if exc.diagnostics['failure'] in repairable:
                rejection = exc.diagnostics
        except Exception as exc:
            ownership=None;initial_part_points=initial_part_mask=None
            log['error'] = f'{type(exc).__name__}: {exc}'
        attempts.append(ObjectAnnotationAttempt(index, ownership, log))
        if index == 1 or rejection is None or source_reply is None:
            break
        try:
            feedback = _correction_feedback(crop, rejection, source_reply)
            if log.get('object_geometry_passed') and rejection['failure']=='initial_part_outside_fixed_envelope':
                feedback['object_geometry_passed_in_rejected_reply']=True
                feedback['transaction_accepted']=False
            diagnostic = build_annotation_diagnostic(crop.observation, feedback)
        except (ValueError, RuntimeError) as exc:
            log['correction_unavailable'] = str(exc)
            break
        log['correction_feedback'] = copy.deepcopy(feedback)
        transaction = {'initial_object_reference': {'frame_id': source_obs.frame_id,
            'camera_id': source_obs.camera_id, 'target_id': target_id, 'attempt': 1,
            'maximum_semantic_requests': 2, 'image_context': copy.deepcopy(crop.provenance()),
            'previous_annotation': copy.deepcopy(log['raw_annotation']), 'geometry_diagnostics': feedback}}
        instruction = base_instruction + (
            ' This is the ONE correction on the SAME image. Return a fresh complete region and '
            'explicit seeds; replies are never merged. The diagnostic image marks actual rejected '
            'pixels only, not replacement seeds or desired boundaries. For envelope failures, '
            'rejected_geometry_pixels are measured outliers; component_seed_pixels only attribute '
            'which seeds selected that component, not whether those seeds themselves are outliers. '
            'Use the first unmarked crop for all output coordinates. Omit uncertain material. '
            'If a rejected seed was a speculative click on background or support, omit it; '
            'do not replace every rejected click just to preserve the old seed count. '
            'Use a new seed only when you can clearly identify additional object material. '
            'Unflagged seeds may be retained only if they still match the visible object; '
            'the entire fresh answer, including those seeds, is revalidated independently.')
        if feedback.get('object_geometry_passed_in_rejected_reply'):
            instruction+=(' Your previous object region and object seeds passed their own geometric checks; '
                'the transaction failed on the part region. If those object proposals still match visible '
                'material, preserve them in your fresh complete answer while fixing the part. Do not add '
                'unrelated speculative object seeds during a part-only correction. Nothing is automatically '
                'accepted or copied: all proposals in the new answer will be independently revalidated.')
    return finish()
