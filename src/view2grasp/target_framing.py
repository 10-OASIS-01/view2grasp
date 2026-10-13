"""Remember explicit observed object points and predict mounted-camera framing.

No candidate image is read. A projection is not visibility, object identity,
reachability, a grasp contact, or permission to move through space.
"""
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .geometry import project, unproject
from .grounding import depth_component, pixels, proposal_mask
from .mobile_cameras import CAMERAS
from .types import GroundingProposal, Observation


def _calibration(K, T_wc):
    K=np.asarray(K,dtype=float);T_wc=np.asarray(T_wc,dtype=float)
    if (K.shape!=(3,3) or T_wc.shape!=(4,4) or
            not np.isfinite(K).all() or not np.isfinite(T_wc).all() or
            K[0,0]<=0 or K[1,1]<=0 or abs(np.linalg.det(K))<1e-12 or
            not np.allclose(T_wc[3],[0,0,0,1]) or
            not np.allclose(T_wc[:3,:3].T@T_wc[:3,:3],np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(T_wc[:3,:3]),1.,atol=1e-6)):
        raise ValueError('invalid framing camera calibration')
    return K,T_wc


@dataclass(frozen=True)
class ObservedFramingTarget:
    target_id: str
    source_frame_id: int
    source_camera_id: str
    source_timestamp: float
    source_pixels: tuple[tuple[int,int],...]
    source_normalized_points: tuple[tuple[float,float],...]
    world_points: tuple[tuple[float,float,float],...]
    selection: str
    robot_depth_checked: bool


class TargetFraming:
    """A static, last-observed target hypothesis, separate from grasp/payload state.

    Refreshes without object proposals do nothing. Invalid refreshes raise and
    leave the old, explicitly dated evidence intact. Call clear() on closure,
    release, a target change, or any event that invalidates the static assumption.
    """
    def __init__(self):
        self.target: ObservedFramingTarget | None=None

    def clear(self):
        self.target=None

    def refresh(self, obs: Observation, proposals: Sequence[GroundingProposal],
                robot_depth=None):
        objects=[g for g in proposals if g.role=='object']
        if not objects:return False
        if obs.camera_id not in CAMERAS:
            raise ValueError('target framing requires a mounted source camera')
        if any(g.frame_id!=obs.frame_id or g.camera_id!=obs.camera_id for g in objects):
            raise ValueError('stale object framing proposal')
        target_ids={g.target_id for g in objects}
        if len(target_ids)!=1 or not next(iter(target_ids)).strip():
            raise ValueError('object framing requires one explicit target_id')
        target_id=next(iter(target_ids))
        if self.target is not None:
            if target_id!=self.target.target_id:
                raise ValueError('clear previous framing target before changing target_id')
            if obs.frame_id<self.target.source_frame_id or obs.timestamp<self.target.source_timestamp:
                raise ValueError('cannot refresh framing from an older observation')
        K,T_wc=_calibration(obs.K,obs.T_wc)
        h,w=obs.depth.shape
        if min(h,w)<2:raise ValueError('framing image is too small')
        if robot_depth is not None:
            robot_depth=np.asarray(robot_depth)
            if robot_depth.shape!=obs.depth.shape:raise ValueError('robot depth shape mismatch')
        allowed=np.ones((h,w),dtype=bool);explicit=[];negatives=[]
        for g in objects:
            if g.kind not in ('points','box','polygon','mask'):
                raise ValueError('unsupported object framing representation')
            allowed &= proposal_mask(g,obs)
            if g.kind!='polygon':explicit.extend(g.points)
            negatives.extend(g.negative_points)
        excluded=np.zeros((h,w),dtype=bool)
        for x,y in pixels(negatives,(h,w)):
            excluded[max(0,y-2):y+3,max(0,x-2):x+3]=True
        if explicit:
            if len(explicit)>32:raise ValueError('too many explicit object framing points')
            uv=pixels(explicit,(h,w));selection='explicit_object_points'
            for x,y in uv:
                if not allowed[y,x]:raise ValueError('framing point outside proposed object region')
                if excluded[y,x]:raise ValueError('framing point excluded by a negative point')
                d=obs.depth[y,x]
                if not np.isfinite(d) or not .05<d<10:raise ValueError('invalid framing point depth')
                if robot_depth is not None:
                    own=robot_depth[y,x]
                    if not np.isfinite(own) or own<=0:raise ValueError('invalid robot depth at framing point')
                    if abs(d-own)<=.003:raise ValueError('framing point is a robot self return')
                    if d>own:raise ValueError('known robot occludes framing point')
        else:
            # With no explicit point, derive an actual interior depth sample.
            # Polygon vertices describe a boundary and are never contact seeds.
            if robot_depth is not None:
                allowed &= np.isfinite(robot_depth)&(robot_depth>0)&(obs.depth<robot_depth-.003)
            component=depth_component(obs,allowed,negative_points=negatives,robot_depth=robot_depth)
            interior=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,3)
            y,x=np.unravel_index(interior.argmax(),interior.shape)
            uv=np.array([[x,y]]);explicit=[(x/(w-1),y/(h-1))]
            selection='derived_object_region_interior'
        world=unproject(uv,obs.depth[uv[:,1],uv[:,0]],K,T_wc)
        if not np.isfinite(world).all():raise ValueError('nonfinite object framing point')
        self.target=ObservedFramingTarget(target_id,obs.frame_id,obs.camera_id,float(obs.timestamp),
            tuple(tuple(map(int,p)) for p in uv),tuple(tuple(map(float,p)) for p in explicit),
            tuple(tuple(map(float,p)) for p in world),selection,robot_depth is not None)
        return True

    def context(self, obs: Observation, calibrations: Mapping[str,tuple[np.ndarray,np.ndarray]]):
        """Project old observed points into CURRENT known calibrations, without images.

        The caller supplies current extrinsics (including base/arm motion).
        All mounted cameras use the supplied observation's image dimensions.
        """
        if self.target is None:return None
        target=self.target
        if obs.frame_id<target.source_frame_id or obs.timestamp<target.source_timestamp:
            raise ValueError('current observation predates framing target')
        if obs.camera_id not in CAMERAS:raise ValueError('framing context requires a mounted camera')
        h,w=obs.depth.shape
        if min(h,w)<2:raise ValueError('framing image is too small')
        known=dict(calibrations)
        # The input frame's calibration is authoritative for its own camera.
        known[obs.camera_id]=(obs.K,obs.T_wc)
        projections=[]
        for camera_id,(K,T_wc) in known.items():
            if camera_id not in CAMERAS:raise ValueError('unknown mounted framing camera')
            K,T_wc=_calibration(K,T_wc)
            uv,z=project(np.asarray(target.world_points),K,T_wc)
            normalized=uv/[w-1,h-1]
            front=(z>.05)&np.isfinite(normalized).all(axis=1)
            inside=front&np.all((normalized>=0)&(normalized<=1),axis=1)
            margin=(float(np.min(np.minimum(normalized[front],1-normalized[front])))
                if front.all() else None)
            center=normalized[front].mean(0) if front.any() else None
            projections.append({'camera_id':camera_id,'is_current':camera_id==obs.camera_id,
                'projected_points_normalized':[p.tolist() if valid else None for p,valid in zip(normalized,front)],
                'in_frame_points':int(inside.sum()),'point_count':len(inside),
                'in_frame_fraction':float(inside.mean()),'minimum_border_margin':margin,
                'projected_center_normalized':None if center is None else center.tolist(),
                'centering_offset_normalized':None if center is None else (center-.5).tolist()})
        def rank(item):
            margin=item['minimum_border_margin']
            return (item['in_frame_fraction'],margin if margin is not None else -float('inf'))
        projections.sort(key=rank,reverse=True)
        current=next(p for p in projections if p['is_current']);best=projections[0]
        recommendation=(best['camera_id'] if best['in_frame_points'] and rank(best)>rank(current) else None)
        return {'target_id':target.target_id,'source_frame_id':target.source_frame_id,
            'source_camera_id':target.source_camera_id,'source_timestamp':target.source_timestamp,
            'source_pixels':[list(p) for p in target.source_pixels],
            'source_normalized_points':[list(p) for p in target.source_normalized_points],
            'observed_world_points_m':[list(p) for p in target.world_points],
            'selection':target.selection,'robot_depth_checked':target.robot_depth_checked,
            'current_frame_id':obs.frame_id,'source_age_frames':obs.frame_id-target.source_frame_id,
            'recommended_camera':recommendation,'camera_projections':projections,
            'scope':'Last-observed static object hypothesis; projected points alone do not establish object extent, '
                'semantic correctness, current visibility, reachability, graspability or clearance. '
                'No candidate camera image was read. Acquire a fresh counted image before grounding; '
                'all physical camera/base/arm motion still requires planning.'}
