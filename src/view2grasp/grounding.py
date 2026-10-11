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
