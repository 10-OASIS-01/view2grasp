"""Derived image crops from already acquired RGB-D and observed world points.

A projected reference suggests where to look; it does not establish that the
object is visible, still present, reachable, or safe to touch in this frame.
"""
from dataclasses import dataclass, replace
from itertools import product
import copy

import numpy as np

from .geometry import project
from .types import AgentDecision, Observation


@dataclass(frozen=True)
class ObservationCrop:
    observation: Observation
    bounds_xyxy: tuple[int, int, int, int]
    source_shape: tuple[int, int]
    robot_depth: np.ndarray | None
    surface_margin_m: float
    context_pixels: int
    min_size: int
    reference_point_count: int
    reference_kind: str = 'observed_world_points'

    def provenance(self):
        """JSON-safe metadata; all bounds use NumPy's half-open convention."""
        return {
            'kind': 'crop_of_acquired_observation',
            'source_frame_id': self.observation.frame_id,
            'source_camera_id': self.observation.camera_id,
            'source_timestamp': float(self.observation.timestamp),
            'source_shape_hw': list(self.source_shape),
            'crop_shape_hw': list(self.observation.depth.shape),
            'bounds_xyxy': list(self.bounds_xyxy),
            'bounds_convention': 'left_top_inclusive_right_bottom_exclusive',
            'coordinates': 'normalized_to_crop_pixel_centers',
            'surface_margin_m': self.surface_margin_m,
            'context_pixels': self.context_pixels,
            'min_size_pixels': self.min_size,
            'reference_point_count': self.reference_point_count,
            'reference_kind': self.reference_kind,
            'reference_use': ('proposal_extent_only_not_current_visibility_or_contact_evidence'
                if self.reference_kind=='current_vlm_object_pixel_region' else
                'projection_only_not_current_visibility_or_contact_evidence'),
            'new_observation': False,
            'resized': False,
        }

    def restore_points(self, points):
        """Map selected crop pixel centers to normalized source pixel centers.

        Resolve the same nearest pixel used by grounding before translation.
        Rounding only afterwards changes half-pixel ties when the crop origin
        is odd, which can grow/shrink a polygon by one pixel.
        """
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('crop points must have shape (N, 2)')
        if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError('crop points must be finite and normalized')
        h, w = self.source_shape
        x0, y0, x1, y1 = self.bounds_xyxy
        selected = np.rint(points * [x1 - x0 - 1, y1 - y0 - 1])
        return (selected + [x0, y0]) / [w - 1, h - 1]

    def restore_decision(self, decision: AgentDecision) -> AgentDecision:
        """Restore an annotation reply without changing its semantic meaning.

        Cropped requests are annotation-only. They cannot select a grasp ray or
        execute the reply's camera command. Wrong frame/camera replies remain
        invalid; candidate IDs have no transformable pixel representation.
        """
        if decision.action != 'observe':
            raise ValueError('crop restoration only accepts observe annotations')
        return self._restore_decision(decision)

    def restore_grasp_decision(self, decision: AgentDecision) -> AgentDecision:
        """Restore an explicit fresh grasp refinement, never an annotation action."""
        if decision.action not in ('grasp','abort'):
            raise ValueError('grasp refinement restoration only accepts grasp or abort')
        if decision.action=='grasp':
            roles={g.role for g in decision.groundings if g.source=='vlm' and g.kind in ('box','polygon','mask')}
            if not {'object','grasp_part'}<=roles:
                raise ValueError('grasp refinement requires fresh explicit object and grasp_part')
        return self._restore_decision(decision)

    def localize_grounding_error(self,error):
        """Express actual resolver seed failures in the supplied crop coordinates.

        The source diagnostic is retained by the caller's local log. No sample
        is clipped to the crop, moved to valid depth, or made into a new seed.
        """
        diagnostics=getattr(error,'diagnostics',None)
        if diagnostics is None:return {'reason':str(error)}
        obs=self.observation
        if (diagnostics.get('frame_id')!=obs.frame_id or diagnostics.get('camera_id')!=obs.camera_id
                or 'rejected_seeds' not in diagnostics):
            raise ValueError('grounding feedback requires current source-frame seed diagnostics')
        feedback=copy.deepcopy(diagnostics)
        ch,cw=obs.depth.shape;offset=np.array(self.bounds_xyxy[:2])
        feedback.update(coordinate_space='supplied_crop_normalized_pixel_centers',image_shape_hw=[ch,cw])
        for sample in feedback['rejected_seeds']:
            local=np.asarray(sample['pixel'],dtype=int)-offset
            sample['pixel']=local.tolist();sample['normalized']=(local/[cw-1,ch-1]).tolist()
            sample['inside_crop']=bool(np.all(local>=0) and np.all(local<[cw,ch]))
        return feedback

    def _restore_decision(self, decision):
        obs = self.observation
        if decision.frame_id != obs.frame_id or decision.camera_id != obs.camera_id:
            raise ValueError('crop annotation requires its source frame and camera')
        data = decision.model_dump()

        def restore(points):
            return self.restore_points(points).tolist()

        for name in ('point', 'place_point'):
            if data[name] is not None:
                data[name] = restore([data[name]])[0]
        for name in ('bbox', 'part_bbox'):
            if data[name] is not None:
                data[name] = np.asarray(restore(np.asarray(data[name]).reshape(2, 2))).ravel().tolist()
        for proposal in data['groundings']:
            if proposal['frame_id'] != obs.frame_id or proposal['camera_id'] != obs.camera_id:
                raise ValueError('crop grounding requires its source frame and camera')
            if proposal['kind'] == 'candidate':
                raise ValueError('crop candidate IDs cannot be restored without pixel evidence')
            proposal['points'] = restore(proposal['points'])
            proposal['negative_points'] = restore(proposal['negative_points'])
            proposal['holes'] = [restore(ring) for ring in proposal['holes']]
            if proposal['box'] is not None:
                proposal['box'] = np.asarray(restore(np.asarray(proposal['box']).reshape(2, 2))).ravel().tolist()
            if proposal['kind'] == 'mask':
                from .grounding import mask_rle
                # Pixel masks contain no coordinate transform; use their
                # validated RLE directly, preserving holes exactly.
                if tuple(proposal['mask_size']) != obs.depth.shape:
                    raise ValueError('crop mask dimensions differ from the acquired crop')
                cropped_mask = np.repeat(np.arange(len(proposal['mask_rle'])) % 2,
                    proposal['mask_rle']).reshape(obs.depth.shape).astype(bool)
                full_mask = np.zeros(self.source_shape, dtype=bool)
                x0, y0, x1, y1 = self.bounds_xyxy
                full_mask[y0:y1, x0:x1] = cropped_mask
                proposal['mask_size'] = list(self.source_shape)
                proposal['mask_rle'] = mask_rle(full_mask)
        return AgentDecision.model_validate(data)


def _axis_bounds(low, high, extent, min_size):
    # Include every pixel center in the projected envelope and shift a small
    # crop inward at image edges instead of losing its requested context size.
    low = max(0, int(np.floor(low)))
    high = min(extent, int(np.ceil(high)) + 1)
    size = min(extent, max(high - low, min_size))
    start = max(0, min(extent - size, int(np.floor((low + high - size) / 2))))
    return start, start + size


def _validate_source(obs,robot_depth):
    if obs.depth.ndim != 2 or min(obs.depth.shape) < 2 or obs.rgb.shape != (*obs.depth.shape, 3):
        raise ValueError('crop requires aligned RGB and depth images of at least two pixels')
    K,T=np.asarray(obs.K),np.asarray(obs.T_wc)
    if (K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all()
            or not np.isfinite(T).all() or K[0,0]<=0 or K[1,1]<=0
            or not np.allclose(K[2],[0,0,1]) or not np.allclose(T[3],[0,0,0,1])
            or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-6)):
        raise ValueError('crop requires finite calibrated camera matrices')
    if robot_depth is not None and np.asarray(robot_depth).shape!=obs.depth.shape:
        raise ValueError('robot depth must match the source observation shape')


def _make_crop(obs,bounds,robot_depth,surface_margin_m,context_pixels,min_size,reference_count,
               reference_kind='observed_world_points'):
    x0,y0,x1,y1=bounds
    K=np.asarray(obs.K,dtype=float).copy();K[0,2]-=x0;K[1,2]-=y0
    cropped=replace(obs,rgb=obs.rgb[y0:y1,x0:x1].copy(),depth=obs.depth[y0:y1,x0:x1].copy(),K=K)
    own=None if robot_depth is None else np.asarray(robot_depth)[y0:y1,x0:x1].copy()
    return ObservationCrop(cropped,bounds,tuple(obs.depth.shape),own,float(surface_margin_m),
        int(context_pixels),int(min_size),int(reference_count),reference_kind)


def crop_grasp_proposal(obs: Observation,decision: AgentDecision,*,robot_depth=None,
                        context_pixels=16,min_size=128) -> ObservationCrop:
    """Reinspect a current object proposal before deriving contact or a ray.

    Bounds come from the proposal, not its resolved contact-connected component:
    a coarse aperture hole must not make the crop lose the cup body or outer bar.
    Measured depth only checks that the region contains a non-robot return; it
    does not identify the object, filter/repair the proposal, or grant contact.
    """
    from .grounding import proposal_mask
    _validate_source(obs,robot_depth)
    if decision.action!='grasp' or decision.frame_id!=obs.frame_id or decision.camera_id!=obs.camera_id:
        raise ValueError('grasp crop requires a current-frame grasp proposal')
    if isinstance(context_pixels,bool) or not isinstance(context_pixels,(int,np.integer)) or context_pixels<0:
        raise ValueError('crop pixel context must be a nonnegative integer')
    if isinstance(min_size,bool) or not isinstance(min_size,(int,np.integer)) or min_size<2:
        raise ValueError('crop minimum size must be an integer of at least two pixels')
    proposals=[g for g in decision.groundings if g.role=='object' and g.source=='vlm'
        and g.kind in ('box','polygon','mask')]
    if len(proposals)!=1:raise ValueError('grasp crop requires one explicit current object region')
    mask=proposal_mask(proposals[0],obs)
    measured=mask&np.isfinite(obs.depth)&(obs.depth>.01)
    if robot_depth is not None:
        own=np.asarray(robot_depth)
        measured&=~np.isfinite(own)|(obs.depth<own-.003)
    if not measured.any():raise ValueError('object region has no measured non-robot depth for refinement')
    v,u=np.where(mask);h,w=obs.depth.shape
    x0,x1=_axis_bounds(u.min()-context_pixels,u.max()+context_pixels,w,min_size)
    y0,y1=_axis_bounds(v.min()-context_pixels,v.max()+context_pixels,h,min_size)
    return _make_crop(obs,(x0,y0,x1,y1),robot_depth,0.,context_pixels,min_size,
        measured.sum(),'current_vlm_object_pixel_region')


def crop_observation(obs: Observation, reference_points, *, surface_margin_m=.035,
                     context_pixels=16, min_size=128, robot_depth=None) -> ObservationCrop:
    """Crop a current frame around projected, previously observed target points.

    The world-axis bounding box, enlarged by ``surface_margin_m`` on all axes,
    contains every point within that metric distance of the reference cloud.
    Projecting all eight corners therefore covers newly visible surfaces within
    that bound, provided the entire envelope is in front of the camera. Image
    clipping is explicit; neither projection nor cropping checks visibility.
    """
    points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('crop reference must contain finite observed world points')
    if not np.isfinite(surface_margin_m) or surface_margin_m < 0:
        raise ValueError('crop surface margin must be finite and nonnegative')
    if isinstance(context_pixels, bool) or not isinstance(context_pixels, (int, np.integer)) or context_pixels < 0:
        raise ValueError('crop pixel context must be a nonnegative integer')
    if isinstance(min_size, bool) or not isinstance(min_size, (int, np.integer)) or min_size < 2:
        raise ValueError('crop minimum size must be an integer of at least two pixels')
    _validate_source(obs,robot_depth)
    K, T = np.asarray(obs.K), np.asarray(obs.T_wc)
    lo, hi = points.min(axis=0) - surface_margin_m, points.max(axis=0) + surface_margin_m
    envelope = np.array(list(product(*zip(lo, hi))))
    uv, z = project(envelope, K, T)
    if np.any(z <= .01) or not np.isfinite(uv).all():
        raise ValueError('crop reference envelope is behind or intersects the camera near plane')
    projected_low, projected_high = uv.min(axis=0), uv.max(axis=0)
    h, w = obs.depth.shape
    if (projected_high[0] < 0 or projected_high[1] < 0
            or projected_low[0] > w - 1 or projected_low[1] > h - 1):
        raise ValueError('crop reference projection is outside the current image')
    low, high = projected_low - context_pixels, projected_high + context_pixels
    x0, x1 = _axis_bounds(low[0], high[0], w, int(min_size))
    y0, y1 = _axis_bounds(low[1], high[1], h, int(min_size))
    return _make_crop(obs,(x0,y0,x1,y1),robot_depth,surface_margin_m,context_pixels,min_size,len(points))
