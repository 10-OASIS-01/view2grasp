"""Observation-only grounding conversion. Masks never certify free space."""
from dataclasses import dataclass
import json
import time
import numpy as np
import cv2
from .types import GroundingProposal


def pixels(points, shape):
    h, w = shape[:2]
    return np.rint(np.asarray(points).reshape(-1,2) * [w - 1, h - 1]).astype(int)


def restore_crop(points, crop_box, full_shape):
    """Normalized crop coordinates -> normalized full-frame coordinates."""
    h, w = full_shape[:2]
    x0, y0, x1, y1 = crop_box  # inclusive pixel-center bounds
    if not (0 <= x0 < x1 < w and 0 <= y0 < y1 < h):
        raise ValueError("crop outside original image")
    p = np.asarray(points, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid crop coordinates")
    return (p * [x1 - x0, y1 - y0] + [x0, y0]) / [w - 1, h - 1]


def mask_rle(mask):
    flat = np.asarray(mask, dtype=bool).ravel()
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, len(flat)]
    counts = np.diff(boundaries).tolist()
    return ([0] if len(flat) and flat[0] else []) + counts


def proposal_mask(proposal, obs, candidates=None):
    if proposal.frame_id != obs.frame_id or proposal.camera_id != obs.camera_id:
        raise ValueError("stale grounding frame or wrong camera")
    h, w = obs.depth.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if proposal.kind == "candidate":
        if candidates is None or proposal.candidate_id not in candidates:
            raise ValueError("candidate was not provided in this frame")
        resolved = candidates[proposal.candidate_id]
        if resolved.kind == "candidate":
            raise ValueError("recursive candidate reference")
        return proposal_mask(resolved, obs)
    if proposal.kind == "mask":
        if tuple(proposal.mask_size) != (h, w):
            raise ValueError("mask resolution differs from observation")
        return np.repeat(np.arange(len(proposal.mask_rle)) % 2,
                         proposal.mask_rle).reshape(h, w).astype(bool)
    if proposal.kind == "polygon":
        cv2.fillPoly(mask, [pixels(proposal.points, (h,w))], 1)
        for ring in proposal.holes:
            cv2.fillPoly(mask, [pixels(ring, (h,w))], 0)
    elif proposal.box is not None:
        x0,y0,x1,y1 = pixels(np.array(proposal.box).reshape(2,2), (h,w)).ravel()
        mask[y0:y1+1,x0:x1+1] = 1
    elif proposal.kind == "points":
        # A point seeds RGB-D connectivity, not a fabricated object-sized mask.
        mask[:] = 1
    else:
        raise ValueError("an image path is not a segmentation mask")
    return mask.astype(bool)


class GroundingSeedError(ValueError):
    """Strict seed rejection with observation-only, serializable feedback."""
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        correction=('Correct the explicit seed, or omit optional contact points while still supplying valid object/part regions.'
            if diagnostics['valid_candidate_pixels'] else
            'Correct the proposed region or acquire another observation; no filtered pixels remain.')
        super().__init__('grounding seed rejected: '+json.dumps(diagnostics,separators=(',',':'))+
            ' '+correction+' Candidate pixels are filtered depth only, not semantic correctness or graspability evidence.')


class ContactAssociationError(ValueError):
    """A semantic/geometric rejection with bounded observation-only feedback."""
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        # Keep the historical error string for callers and stored failure names.
        super().__init__(message)


def _supported_downward_points(points, reference_object, support_z):
    """Bound measured side surfaces using a fixed observed footprint/support.

    This is available only when the caller establishes the source-supported
    task phase. It neither completes hidden geometry nor grants contact/free
    space: original-reference anchors and measured surface connectivity still
    have to pass independently for each semantic role.
    """
    from scipy.spatial import ConvexHull, QhullError
    points = np.asarray(points)
    reference = np.asarray(reference_object)
    eligible = np.zeros(len(points), dtype=bool)
    evidence = {'source_supported': True,
        'basis': 'fixed_original_object_xy_hull_and_observed_support',
        'xy_tolerance_m': .004, 'support_z_m': float(support_z),
        'minimum_z_exclusive_m': float(support_z + .006),
        'maximum_z_inclusive_m': float(reference[:, 2].max()),
        'status': 'unavailable_reference_footprint'}
    if not np.isfinite(support_z):
        raise ValueError('supported association requires a finite observed support height')
    try:
        hull = ConvexHull(reference[:, :2])
    except QhullError:
        # A line or point reference does not define an observed XY footprint.
        return eligible, evidence
    polygon = reference[hull.vertices, :2]
    evidence.update(status='available', original_xy_hull=polygon.tolist())
    if not len(points): return eligible, evidence
    xy = points[:, :2]
    inside = np.all(xy @ hull.equations[:, :2].T + hull.equations[:, 2] <= 0, axis=1)
    # Euclidean distance to the actual hull, including its corners. Offsetting
    # half-planes alone would exceed 4 mm near a sharp corner.
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edges = b - a
    delta = xy[:, None, :] - a[None, :, :]
    fraction = np.clip(np.sum(delta * edges, axis=2) / np.sum(edges * edges, axis=1), 0, 1)
    distance = np.linalg.norm(delta - fraction[:, :, None] * edges, axis=2).min(axis=1)
    eligible = ((inside | (distance <= .004)) & (points[:, 2] > support_z + .006)
        & (points[:, 2] <= reference[:, 2].max()))
    return eligible, evidence


def _contact_subset_diagnostics(obs, object_mask, part_mask):
    """Report pixel containment only, before depth/reference checks execute."""
    h, w = obs.depth.shape
    outside = np.argwhere(part_mask & ~object_mask)
    count = len(outside)
    samples = outside if count <= 12 else outside[np.linspace(0, count - 1, 12, dtype=int)]
    return {'frame_id': int(obs.frame_id), 'camera_id': obs.camera_id,
        'coordinate_space': 'source_image_normalized_pixel_centers',
        'source_shape_hw': [int(h), int(w)], 'role': 'grasp_part',
        'failure': 'part_outside_object', 'proposed_pixels': int(part_mask.sum()),
        'object_proposed_pixels': int(object_mask.sum()),
        # Numeric zeros preserve the bounded-repair comparison protocol. The
        # explicit statuses distinguish unmeasured from a depth/anchor failure.
        'valid_pixels': 0, 'anchor_pixels': 0,
        'depth_evaluation': 'not_evaluated', 'reference_evaluation': 'not_evaluated',
        'rejection_counts': {'part_outside_object': int(count)},
        'rejected_pixel_count': int(count),
        'rejected_samples': [{'pixel': [int(x), int(y)],
            'normalized': [float(x / (w - 1)), float(y / (h - 1))],
            'reasons': ['part_outside_object']} for y, x in samples],
        'samples_truncated': count > len(samples),
        'reference_projection': {'kind': 'not_evaluated',
            'bbox_normalized': None, 'intersects_image': None},
        'thresholds': {'part_outside_object_pixels_allowed': 0}}


def _contact_diagnostics(obs, role, failure, reference, reference_tree, proposed,
                         accepted, xyz, depth_valid, support_clear, robot_clear,
                         robot_depth, distances, rejected_pixels, rejected_count,
                         component_count=0, downward_mask=None, downward_evidence=None):
    """Build detailed feedback only on rejection, never on a planning loop."""
    from .geometry import project
    h, w = obs.depth.shape
    anchors = distances < .004
    anchor_count = int(anchors.sum())
    rejected_pixels = np.asarray(rejected_pixels, dtype=int).reshape(-1, 2)
    if len(rejected_pixels) > 12:
        ids = np.linspace(0, len(rejected_pixels) - 1, 12, dtype=int)
        rejected_pixels = rejected_pixels[ids]
    samples = []
    for y, x in rejected_pixels:
        reasons = []
        if not depth_valid[y, x]: reasons.append('invalid_depth')
        if depth_valid[y, x] and not support_clear[y, x]: reasons.append('support')
        if robot_depth is not None and not robot_clear[y, x]: reasons.append('robot')
        sample = {'pixel': [int(x), int(y)],
            'normalized': [float(x / (w - 1)), float(y / (h - 1))],
            'reasons': reasons,
            'depth_m': float(obs.depth[y, x]) if np.isfinite(obs.depth[y, x]) else None}
        if np.isfinite(xyz[y, x, 2]): sample['world_z_m'] = float(xyz[y, x, 2])
        if robot_depth is not None:
            sample['robot_depth_m'] = float(robot_depth[y, x]) if np.isfinite(robot_depth[y, x]) else None
        if accepted[y, x]:
            distance = float(reference_tree.query(xyz[y, x])[0])
            sample['reference_distance_m'] = distance
            downward = downward_mask is not None and bool(downward_mask[y, x])
            if downward_evidence is not None:
                sample['supported_downward_eligible'] = bool(downward)
            if distance > .035 and not downward: reasons.append('reference_distance')
            if anchor_count < 4 and distance >= .004: reasons.append('reference_overlap_insufficient')
            if failure == 'unanchored_component': reasons.append('unanchored_component')
        samples.append(sample)
    # These are projected old observations, not evidence that the reference is
    # visible or still occupies those pixels in the current image.
    uv, z = project(reference, obs.K, obs.T_wc)
    uv = uv[(z > .01) & np.isfinite(uv).all(axis=1)]
    bbox = None
    if len(uv):
        lo, hi = np.maximum(uv.min(axis=0), [0, 0]), np.minimum(uv.max(axis=0), [w - 1, h - 1])
        if np.all(lo <= hi): bbox = np.r_[lo / [w - 1, h - 1], hi / [w - 1, h - 1]].tolist()
    expansion_violations = distances > .035
    if downward_mask is not None: expansion_violations &= ~downward_mask[accepted]
    report = {'frame_id': int(obs.frame_id), 'camera_id': obs.camera_id,
        'coordinate_space': 'source_image_normalized_pixel_centers',
        'source_shape_hw': [int(h), int(w)], 'role': role, 'failure': failure,
        'proposed_pixels': int(proposed.sum()), 'valid_pixels': int(accepted.sum()),
        'anchor_pixels': anchor_count,
        'rejection_counts': {
            'invalid_depth': int((proposed & ~depth_valid).sum()),
            'support': int((proposed & depth_valid & ~support_clear).sum()),
            'robot': int((proposed & ~robot_clear).sum()),
            'reference_distance': int(expansion_violations.sum()),
            'reference_overlap_insufficient': int((~anchors).sum()) if anchor_count < 4 else 0,
            'unanchored_component': int(component_count)},
        'rejected_pixel_count': int(rejected_count), 'rejected_samples': samples,
        'samples_truncated': bool(rejected_count > len(samples)),
        'reference_projection': {'kind': 'hypothesis_not_current_visibility',
            'bbox_normalized': bbox, 'intersects_image': bbox is not None},
        'thresholds': {'support_clearance_m': .006, 'robot_clearance_m': .003,
            'reference_anchor_m': .004, 'reference_expansion_m': .035,
            'component_step_m': .006, 'min_valid_pixels': 4, 'min_anchor_pixels': 4}}
    if downward_evidence is not None:
        report['supported_downward_association'] = downward_evidence
    return report


def _component_anchor_repairability(obs, xyz, valid, component_pixels, reference,
                                    reference_object, support_z, source_supported,
                                    semantic_exclusions):
    """Diagnose a possibly truncated boundary without proposing a new mask.

    The unchanged metric graph may reach an original anchor through measured
    pixels outside the proposed polygon. That only justifies asking the VLM
    to reinspect its boundary; those pixels have no semantic/contact authority.
    Explicit negative points and polygon holes cannot provide such a bridge.
    """
    import hashlib
    from scipy.spatial import cKDTree
    reference = np.asarray(reference)
    original = np.asarray(reference_object)
    lo, hi = reference.min(0) - .035, reference.max(0) + .035
    if source_supported:
        lo = np.minimum(lo, np.r_[original[:, :2].min(0) - .004, support_z + .006])
        hi = np.maximum(hi, np.r_[original[:, :2].max(0) + .004, original[:, 2].max()])
    candidates = (valid & ~semantic_exclusions &
        np.all((xyz >= lo) & (xyz <= hi), axis=2))
    points = xyz[candidates]
    distances = cKDTree(reference).query(points)[0]
    eligible = distances <= .035
    if source_supported:
        eligible |= _supported_downward_points(points, original, support_z)[0]
    allowed = np.zeros(valid.shape, dtype=bool)
    allowed[candidates] = eligible
    anchors = np.zeros(valid.shape, dtype=bool)
    anchors[candidates] = (distances < .004) & eligible
    reached = anchors.copy()
    stack = [tuple(pixel) for pixel in np.argwhere(anchors)]
    h, w = valid.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if (0 <= yy < h and 0 <= xx < w and allowed[yy,xx] and
                    not reached[yy,xx] and np.linalg.norm(xyz[yy,xx]-xyz[y,x]) <= .006):
                reached[yy,xx] = True
                stack.append((yy,xx))
    pixels_yx = np.asarray(component_pixels, dtype=int)
    reachable = int(reached[pixels_yx[:,0], pixels_yx[:,1]].sum())
    # Same image/reference/phase evidence is required when comparing repairs.
    digest = hashlib.sha256()
    for value in (obs.rgb, obs.depth, obs.K, obs.T_wc, valid, reference, original,
                  np.asarray([support_z, float(source_supported)])):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return {'basis': 'observed_geometry_boundary_reinspection_only',
        'evidence_sha256': digest.hexdigest(),
        'component_pixels': len(pixels_yx), 'anchor_reachable_pixels': reachable,
        'anchor_reachable_fraction': reachable / len(pixels_yx),
        'eligible_observed_pixels': int(allowed.sum()),
        'fixed_reference_anchor_pixels': int(anchors.sum()),
        'grants_semantic_or_contact_permission': False,
        'meaning': 'Possible boundary truncation, not a proposed region or proof of object identity.'}


def depth_component(obs, allowed, seeds=(), negative_points=(), support_z=None, robot_depth=None):
    """4-connected RGB-D surface; invalid depth and depth jumps split components."""
    from .geometry import unproject
    h,w = obs.depth.shape
    proposed = np.asarray(allowed, dtype=bool)
    valid = proposed.copy()
    depth_valid = np.isfinite(obs.depth) & (obs.depth > .05) & (obs.depth < 10)
    valid &= depth_valid
    robot_clear = np.ones((h,w),dtype=bool)
    if robot_depth is not None:
        robot_clear = np.abs(obs.depth - robot_depth) > .003
        valid &= robot_clear
    v,u = np.mgrid[:h,:w]
    xyz = unproject(np.stack([u,v],-1), obs.depth, obs.K, obs.T_wc)
    support_clear = np.ones((h,w),dtype=bool)
    if support_z is not None:
        support_clear = xyz[:,:,2] > support_z + .006
        valid &= support_clear
    negative_excluded = np.zeros((h,w),dtype=bool)
    for x,y in pixels(negative_points, (h,w)).reshape(-1,2):
        negative_excluded[max(0,y-2):y+3,max(0,x-2):x+3] = True
    valid &= ~negative_excluded
    if seeds:
        seed_pixels = pixels(seeds, (h,w))
        if not all(valid[y,x] for x,y in seed_pixels):
            failures=[]
            for index,(x,y) in enumerate(seed_pixels):
                if valid[y,x]:continue
                reasons=[]
                if not proposed[y,x]:reasons.append('outside_proposed_region')
                if not depth_valid[y,x]:reasons.append('invalid_depth')
                if depth_valid[y,x] and not robot_clear[y,x]:
                    reasons.append('robot_self_return' if np.isfinite(robot_depth[y,x]) else 'invalid_depth')
                if depth_valid[y,x] and not support_clear[y,x]:reasons.append('support_plane')
                if negative_excluded[y,x]:reasons.append('negative_point')
                failures.append({'seed_index':index,'normalized':list(map(float,seeds[index])),
                    'pixel':[int(x),int(y)],'reasons':reasons})
            raise GroundingSeedError({'frame_id':obs.frame_id,'camera_id':obs.camera_id,
                'valid_candidate_pixels':int(valid.sum()),'rejected_seed_count':len(failures),
                'rejected_seeds':failures[:16],'rejected_seeds_truncated':len(failures)>16})
    else:
        # Prefer the interior of the proposed region, not its bounding-box center.
        distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
        if distance.max() == 0:
            raise ValueError("empty valid grounding region")
        y,x = np.unravel_index(distance.argmax(), distance.shape)
        seed_pixels = [(x,y)]
    selected = np.zeros((h,w), dtype=bool)
    stack = [(int(x),int(y)) for x,y in seed_pixels]
    for x,y in stack:
        selected[y,x] = True
    while stack:
        x,y = stack.pop()
        for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if not (0 <= xx < w and 0 <= yy < h) or selected[yy,xx] or not valid[yy,xx]:
                continue
            pixel_spacing=1.5*obs.depth[y,x]/min(obs.K[0,0],obs.K[1,1])
            if abs(obs.depth[yy,xx]-obs.depth[y,x]) > .015 or np.linalg.norm(xyz[yy,xx]-xyz[y,x]) > max(.015,pixel_spacing):
                continue
            selected[yy,xx] = True
            stack.append((xx,yy))
    if selected.sum() < 4:
        raise ValueError("insufficient connected depth support")
    return selected


class SAMSegmenter:
    """Optional local SAM 2; no download or implicit device change during a run."""
    def __init__(self, checkpoint, model_config="configs/sam2.1/sam2.1_hiera_t.yaml", device="cpu"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(model_config, str(checkpoint), device=device))
        self.device = device
        self.image_key=None

    def segment(self, obs, proposal):
        import torch
        import hashlib
        key=(obs.camera_id,obs.frame_id,hashlib.sha256(obs.rgb.tobytes()).digest())
        if key!=self.image_key:
            self.predictor.set_image(obs.rgb);self.image_key=key
        points = proposal.points + proposal.negative_points
        coords = pixels(points, obs.depth.shape) if points else None
        labels = np.r_[np.ones(len(proposal.points)), np.zeros(len(proposal.negative_points))] if points else None
        box = None if proposal.box is None else pixels(np.array(proposal.box).reshape(2,2), obs.depth.shape).ravel()
        with torch.inference_mode():
            masks,scores,_ = self.predictor.predict(point_coords=coords,point_labels=labels,box=box,multimask_output=True)
        return masks[int(np.argmax(scores))].astype(bool)


@dataclass
class GroundingResult:
    decision: object
    object_mask: np.ndarray
    part_mask: np.ndarray
    report: dict


@dataclass
class ContactAssociation:
    object_mask: np.ndarray
    part_mask: np.ndarray
    object_points: np.ndarray
    part_points: np.ndarray
    report: dict


def _region_contact(obs,mask,distance):
    """Choose a measured interior representative, never complete hidden parts.

    For a strongly elongated observed region, prefer its central axial third
    before maximizing image-space interior distance. A wide end junction must
    not win merely because it has the largest inscribed circle. This is a
    representative of the supplied visible region, not a semantic midpoint or
    evidence of contact/path safety. Seeded decisions never use this policy.
    """
    from .geometry import unproject
    rows,cols=np.where(mask)
    pixels_yx=np.c_[rows,cols];center=pixels_yx.mean(axis=0)
    width=distance[mask]
    maxima=np.flatnonzero(width==width.max())
    selected=maxima[np.argmin(np.sum((pixels_yx[maxima]-center)**2,axis=1))]
    evidence={'method':'global_interior_centroid_tie','basis':'current_measured_part_mask',
        'meaning':'observed_region_representative_not_hidden_part_midpoint_or_safety',
        'measured_pixels':len(rows)}
    xyz=unproject(np.c_[cols,rows],obs.depth[mask],obs.K,obs.T_wc)
    if np.isfinite(xyz).all() and len(xyz)>=4:
        centered=xyz-xyz.mean(axis=0)
        values,axes=np.linalg.eigh(centered.T@centered/(len(xyz)-1))
        axis=axes[:,-1]
        if axis[np.argmax(np.abs(axis))]<0:axis=-axis
        ratio=float(values[-1]/max(values[-2],1e-12))
        projection=xyz@axis
        lower,upper=np.quantile(projection,[.05,.95]);span=float(upper-lower)
        evidence.update(principal_variance_ratio=ratio,principal_axis_world=axis.tolist(),
            robust_axis_range_m=[float(lower),float(upper)],min_variance_ratio=4.,
            extent_quantiles=[.05,.95],central_axis_fraction=[1/3,2/3])
        if ratio>=4. and span>1e-9:
            ids=np.flatnonzero((projection>=lower+span/3)&(projection<=upper-span/3))
            evidence['central_band_pixels']=len(ids)
            if len(ids):
                choices=ids[width[ids]==width[ids].max()]
                axial=np.abs(projection[choices]-(lower+upper)/2)
                index=np.lexsort((np.sum((pixels_yx[choices]-center)**2,axis=1),axial))[0]
                selected=choices[index]
                evidence.update(method='observed_axis_central_third_interior',
                    selected_axis_fraction=float((projection[selected]-lower)/span))
            else:evidence['fallback_reason']='no_measured_pixels_in_central_band'
    else:evidence['fallback_reason']='insufficient_finite_measured_geometry'
    y,x=pixels_yx[selected]
    evidence.update(pixel=[int(x),int(y)],interior_distance_pixels=float(width[selected]))
    return (y,x),evidence


class GroundingResolver:
    def __init__(self, segmenter=None, sam_unavailable_reason=None):
        self.segmenter = segmenter
        self.sam_unavailable_reason = sam_unavailable_reason
        self.calls = 0
        self.seconds = 0.

    @classmethod
    def from_config(cls, config):
        from pathlib import Path
        if config.perception_backend == 'rgbd':
            return cls(sam_unavailable_reason='disabled by frozen configuration')
        checkpoint=Path(config.sam_checkpoint)
        if not checkpoint.is_absolute():checkpoint=Path(__file__).parents[2]/checkpoint
        try:
            if not checkpoint.is_file():raise FileNotFoundError(str(checkpoint))
            segmenter=SAMSegmenter(checkpoint)
            return cls(segmenter)
        except (ImportError,FileNotFoundError,RuntimeError) as exc:
            if config.perception_backend == 'sam':raise
            return cls(sam_unavailable_reason=f'{type(exc).__name__}: {exc}')

    def resolve_grasp(self, obs, decision, support_z, robot_depth=None, candidates=None):
        start=time.monotonic()
        try:return self._resolve_grasp(obs,decision,support_z,robot_depth,candidates)
        finally:self.seconds+=time.monotonic()-start

    def _resolve_grasp(self, obs, decision, support_z, robot_depth=None, candidates=None):
        """Preserve proposals; derive executable contact only at the geometry boundary."""
        start = time.monotonic()
        proposals = list(decision.groundings)
        if not proposals:
            for role,box,points in (("object",decision.bbox,[]),
                                    ("grasp_part",decision.part_bbox,[decision.point])):
                proposals.append(GroundingProposal(kind="box",role=role,frame_id=obs.frame_id,
                    camera_id=obs.camera_id,box=box,points=points,source="legacy"))
        targets = {g.target_id for g in proposals if g.role in ("object","grasp_part","grasp_contact")}
        if len(targets) != 1:
            raise ValueError("grasp must resolve exactly one associated target")
        groups = {role:[g for g in proposals if g.role==role] for role in ("object","grasp_part","grasp_contact")}
        part_proposals = groups['grasp_part'] or groups['object'] or groups['grasp_contact']
        object_proposals = groups['object'] or part_proposals
        contact_seeds = [p for g in groups['grasp_contact'] for p in g.points]
        report = {"kinds":[g.kind for g in proposals],"source":"rgbd","sam_calls":0,
                  "sam_unavailable":self.sam_unavailable_reason,"validated":False}
        def region(items, seeds):
            allowed = np.ones(obs.depth.shape,dtype=bool)
            negatives=[]
            for g in items:
                allowed &= proposal_mask(g,obs,candidates)
                negatives.extend(g.negative_points)
                if self.segmenter is not None and g.kind in ("points","box"):
                    self.calls+=1; report['sam_calls']+=1
                    try:
                        allowed &= self.segmenter.segment(obs,g)
                        report['source']='sam+rgbd'
                    except (RuntimeError,ValueError) as exc:
                        report['sam_fallback']=str(exc)
            return depth_component(obs,allowed,seeds,negatives,support_z,robot_depth)
        part_seeds = contact_seeds or [p for g in part_proposals for p in g.points if g.kind != 'polygon']
        part_mask = region(part_proposals,part_seeds)
        # An interior valid pixel initializes connectivity when only a mask/box was supplied.
        distance = cv2.distanceTransform(part_mask.astype(np.uint8),cv2.DIST_L2,3)
        y,x = np.unravel_index(distance.argmax(),distance.shape)
        if not part_seeds:
            (y,x),report['point_selection']=_region_contact(obs,part_mask,distance)
        else:
            report['point_selection']={'method':'explicit_contact' if contact_seeds else 'seeded_part_global_interior',
                'basis':'unchanged_explicit_seed_policy'}
        h,w = obs.depth.shape
        contact = contact_seeds[0] if contact_seeds else (x/(w-1),y/(h-1))
        object_mask = region(object_proposals,[contact])
        part_mask &= object_mask
        if not part_mask.any():
            raise ValueError("part and object segmentations do not agree")
        def box(mask):
            v,u=np.where(mask)
            return (float(u.min()/(w-1)),float(v.min()/(h-1)),float(u.max()/(w-1)),float(v.max()/(h-1)))
        # Keep all original representations on the resolved decision for audit.
        updates={"point":contact,"bbox":box(object_mask),"part_bbox":box(part_mask)}
        place=[g for g in proposals if g.role=='place_region']
        if place and decision.place_point is None:
            allowed=proposal_mask(place[0],obs,candidates)
            allowed &= np.isfinite(obs.depth)&(obs.depth>.05)
            distance=cv2.distanceTransform(allowed.astype(np.uint8),cv2.DIST_L2,3)
            if distance.max()==0:raise ValueError('empty placement region')
            py,px=np.unravel_index(distance.argmax(),distance.shape)
            updates['place_point']=(px/(w-1),py/(h-1))
        resolved=decision.model_copy(update=updates)
        report.update(validated=True,contact=contact,object_pixels=int(object_mask.sum()),part_pixels=int(part_mask.sum()))
        return GroundingResult(resolved,object_mask,part_mask,report)

    def track(self,obs,reference,robot_depth=None,support_z=None):
        start=time.monotonic()
        def segment(proposal):
            self.calls+=1
            return self.segmenter.segment(obs,proposal)
        try:return track_surface(obs,reference,robot_depth,segment=segment if self.segmenter else None,support_z=support_z)
        finally:self.seconds+=time.monotonic()-start

    def reassociate_contact(self,obs,decision,reference_object,reference_part,target_id,
                            support_z,robot_depth=None,*,source_supported=False):
        start=time.monotonic()
        try:return self._reassociate_contact(obs,decision,reference_object,reference_part,target_id,
                                            support_z,robot_depth,source_supported=source_supported)
        finally:self.seconds+=time.monotonic()-start

    def _reassociate_contact(self,obs,decision,reference_object,reference_part,target_id,
                             support_z,robot_depth=None,*,source_supported=False):
        """Validate fresh semantic regions without selecting a new grasp ray.

        Proximity alone cannot authorize contact: a 10 mm neighboring object can
        belong to the tracker's depth component. Require explicit current-frame
        object AND part polygons/masks, with measured overlap to the original
        surfaces. A box or point is insufficient for this permission change.
        A source-supported phase may associate measured downward faces inside
        the fixed original object's observed XY footprint and support interval;
        it still requires the same per-role anchors and 6 mm surface chains.
        """
        from .geometry import unproject
        from scipy.spatial import cKDTree
        if decision.frame_id!=obs.frame_id or decision.camera_id!=obs.camera_id:
            raise ValueError('contact association requires the current frame and camera')
        groups={role:[g for g in decision.groundings if g.role==role]
                for role in ('object','grasp_part')}
        masks={};semantic_exclusions={}
        for role,proposals in groups.items():
            if not proposals:raise ValueError('contact association requires explicit object and grasp_part regions')
            mask=np.ones(obs.depth.shape,dtype=bool)
            excluded=np.zeros(obs.depth.shape,dtype=np.uint8)
            for proposal in proposals:
                if proposal.target_id!=target_id or proposal.source!='vlm':
                    raise ValueError('contact association target identity or semantic source differs')
                if proposal.kind not in ('polygon','mask'):
                    raise ValueError('contact association requires tight polygon or mask regions, not boxes or points')
                mask &= proposal_mask(proposal,obs)
                for ring in proposal.holes:
                    cv2.fillPoly(excluded,[pixels(ring,obs.depth.shape)],1)
                for x,y in pixels(proposal.negative_points,obs.depth.shape).reshape(-1,2):
                    mask[max(0,y-2):y+3,max(0,x-2):x+3]=False
                    excluded[max(0,y-2):y+3,max(0,x-2):x+3]=1
            masks[role]=mask
            semantic_exclusions[role]=excluded.astype(bool)
        if np.any(masks['grasp_part'] & ~masks['object']):
            raise ContactAssociationError('contact part must remain inside the explicit object region',
                _contact_subset_diagnostics(obs,masks['object'],masks['grasp_part']))
        h,w=obs.depth.shape;v,u=np.mgrid[:h,:w]
        xyz=unproject(np.stack([u,v],-1),obs.depth,obs.K,obs.T_wc)
        depth_valid=np.isfinite(obs.depth)&(obs.depth>.05)&(obs.depth<10)&np.isfinite(xyz).all(axis=2)
        support_clear=xyz[:,:,2]>support_z+.006
        robot_clear=np.ones((h,w),dtype=bool)
        if robot_depth is not None:
            # Known robot geometry in front of a depth return also invalidates
            # that return; being farther than the self-match band is not enough.
            robot_clear=np.isfinite(robot_depth)&(obs.depth<robot_depth-.003)
        valid=depth_valid&support_clear&robot_clear
        clouds={};overlap={};downward_reports={}
        for role,reference in (('object',reference_object),('grasp_part',reference_part)):
            reference=np.asarray(reference)
            if len(reference)<4 or not np.isfinite(reference).all():
                raise ValueError('contact association lacks an original observed reference')
            proposed=masks[role]
            masks[role]=proposed&valid
            points=xyz[masks[role]]
            reference_tree=cKDTree(reference)
            distances=reference_tree.query(points)[0]
            point_pixels=np.argwhere(masks[role])
            downward_mask=None;downward_evidence=None
            expansion_violations=distances>.035
            if source_supported:
                downward,downward_evidence=_supported_downward_points(points,reference_object,support_z)
                downward_evidence['eligible_beyond_reference_limit_pixels']=int(np.sum(downward&expansion_violations))
                downward_mask=np.zeros((h,w),dtype=bool);downward_mask[masks[role]]=downward
                expansion_violations &= ~downward
                downward_reports[role]=downward_evidence
            def reject(message,failure,rejected_pixels,rejected_count,component_count=0,
                       component_pixels=None):
                diagnostics=_contact_diagnostics(obs,role,failure,
                    reference,reference_tree,proposed,masks[role],xyz,depth_valid,support_clear,
                    robot_clear,robot_depth,distances,rejected_pixels,rejected_count,component_count,
                    downward_mask,downward_evidence)
                if component_pixels is not None:
                    diagnostics['component_anchor_repairability']=_component_anchor_repairability(
                        obs,xyz,valid,component_pixels,reference,reference_object,support_z,
                        source_supported,semantic_exclusions[role])
                raise ContactAssociationError(message,diagnostics)
            if len(points)<4:
                rejected=np.argwhere(proposed&~valid)
                reject('contact association has insufficient filtered depth','insufficient_filtered_depth',
                    rejected,len(rejected))
            # Fixed source references prevent incremental association drift. The
            # 4 mm overlap is tighter than the 10 mm independent-neighbor case;
            # 35 mm bounds ordinary new visible faces, not collision/contact
            # tolerances. A caller-established supported phase can also use the
            # bounded measured downward surface, without lateral/upward growth.
            anchors=distances<.004
            if anchors.sum()<4 or expansion_violations.any():
                # Show the expansion violations first, so a single wrong pixel
                # is not hidden by a much larger discarded support/background.
                rejected=point_pixels[expansion_violations]
                if not len(rejected):rejected=point_pixels[~anchors]
                reject('contact association is inconsistent with the original observed '+role,
                    'reference_inconsistency',rejected,len(rejected))
            # Each disconnected semantic island needs its own old-surface
            # evidence; an unrelated island cannot borrow another's overlap.
            anchor_mask=np.zeros((h,w),dtype=bool);anchor_mask[masks[role]]=anchors
            unseen=masks[role].copy()
            for y,x in point_pixels:
                if not unseen[y,x]:continue
                unseen[y,x]=False;stack=[(y,x)];component_anchors=0;component_count=0;component_samples=[];component_pixels=[]
                while stack:
                    yy,xx=stack.pop();component_anchors+=int(anchor_mask[yy,xx]);component_count+=1
                    component_pixels.append((yy,xx))
                    if len(component_samples)<12:component_samples.append((yy,xx))
                    for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                        if (0<=ny<h and 0<=nx<w and unseen[ny,nx] and
                                np.linalg.norm(xyz[ny,nx]-xyz[yy,xx])<=.006):
                            unseen[ny,nx]=False;stack.append((ny,nx))
                if not component_anchors:
                    reject('contact association contains an unanchored semantic surface component',
                        'unanchored_component',component_samples,component_count,component_count,component_pixels)
            clouds[role]=points;overlap[role]=int(anchors.sum())
        report={'frame_id':obs.frame_id,'camera_id':obs.camera_id,'target_id':target_id,
             'source':'explicit_current_frame_semantics+rgbd','original_surface_overlap_pixels':overlap,
             'object_pixels':len(clouds['object']),'part_pixels':len(clouds['grasp_part']),
             'grasp_ray_changed':False}
        if source_supported:report['supported_downward_association']=downward_reports
        return ContactAssociation(masks['object'],masks['grasp_part'],clouds['object'],clouds['grasp_part'],report)


def lift_ground_path(obs, proposal, floor_z, tolerance=.06):
    """Lift only verified ground points. Never borrow background depth for free-space EE paths."""
    from .geometry import unproject
    if proposal.role != "ground_path" or proposal.kind != "path":
        raise ValueError("only ground_path has a depth-defined trajectory")
    if proposal.frame_id != obs.frame_id or proposal.camera_id != obs.camera_id:
        raise ValueError("stale path frame/camera")
    uv=pixels(proposal.points,obs.depth.shape)
    d=obs.depth[uv[:,1],uv[:,0]]
    if not np.isfinite(d).all() or np.any((d <= .05) | (d >= 10)):
        raise ValueError("ground path has missing/invalid depth")
    world=unproject(uv,d,obs.K,obs.T_wc)
    if np.any(np.abs(world[:,2]-floor_z)>tolerance):
        raise ValueError("path crosses a non-ground surface; reobserve or replan")
    return world


def track_surface(obs, reference, robot_depth=None, motion_margin=.035,segment=None,support_z=None):
    """Associate by local 3D depth connectivity, never a task-specific color."""
    from .geometry import unproject
    reference=np.asarray(reference)
    if not len(reference):return None
    h,w=obs.depth.shape;v,u=np.mgrid[:h,:w]
    xyz=unproject(np.stack([u,v],-1),obs.depth,obs.K,obs.T_wc)
    allowed=np.isfinite(xyz).all(axis=2)
    allowed &= np.all((xyz>=reference.min(0)-motion_margin)&(xyz<=reference.max(0)+motion_margin),axis=2)
    if robot_depth is not None:allowed &= np.abs(obs.depth-robot_depth)>.003
    # A top-down reference may contain only the object's top face. Its minimum
    # height is NOT the support plane: using it erases newly visible side faces
    # and repeatedly shrinks the tracked object across views.
    if support_z is not None:allowed &= xyz[:,:,2]>support_z+.003
    ids=np.flatnonzero(allowed)
    if not len(ids):return None
    from scipy.spatial import cKDTree
    distance=cKDTree(reference).query(xyz.reshape(-1,3)[ids])[0]
    best=int(np.argmin(distance))
    if distance[best]>motion_margin:return None
    py,px=np.unravel_index(ids[best],(h,w))
    if segment is not None:
        ys,xs=np.where(allowed)
        if xs.max()>xs.min() and ys.max()>ys.min():
            proposal=GroundingProposal(kind='box',role='object',frame_id=obs.frame_id,camera_id=obs.camera_id,
                source='rgbd',points=[(px/(w-1),py/(h-1))],
                box=(xs.min()/(w-1),ys.min()/(h-1),xs.max()/(w-1),ys.max()/(h-1)))
            mask=np.asarray(segment(proposal),dtype=bool)&allowed
            if mask.sum()<4:return None
            # A grasped object can appear as disconnected visible islands on
            # either side of the fingers. Preserve SAM's full visible mask.
            return xyz[mask]
    try:mask=depth_component(obs,allowed,[(px/(w-1),py/(h-1))],robot_depth=robot_depth)
    except ValueError:return None
    return xyz[mask]
