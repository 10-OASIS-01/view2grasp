"""Lazy 5 mm voxel inverse-sensor map. Unknown is never implicitly free."""
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from .geometry import cloud, project

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
ROBOT_MODEL_TOLERANCE_M = 0.0001
# MuJoCo's scene and robot-only depth passes can choose adjacent triangles at a
# silhouette.  Their unprojected endpoints can therefore land a fraction of a
# millimetre outside the same visual mesh.  This bound is deliberately below a
# depth pixel / map voxel and far below the 3 mm collision-planning padding.
VISUAL_SELF_FILTER_TOLERANCE_M = 0.00075


class DepthMap:
    def __init__(self, resolution=.005, workspace_bounds=((-1.3,-1.,.4),(1.3,1.,2.)),max_depth=3.,max_views=None):
        self.resolution=resolution; self.views=[]; self.self_masks=[]; self.occlusion_masks=[]; self.surface=np.empty((0,3)); self.tree=None
        self.workspace_bounds=workspace_bounds; self.max_depth=max_depth;self.max_views=max_views
        self.home_tree=None; self.home_volumes=[]; self.target_tree=None; self.non_target_tree=None
        self.payload_tree=None;self.payload_xy_tree=None;self.payload_points=None
        self.table_z=None;self.table_hull=None
        self.retained_endpoints=np.empty((0,3));self.retained_endpoint_tree=None

    def set_table(self,z):
        """Register the observed support plane and its observed XY convex hull."""
        self.table_z=float(z)
        points=self.surface[np.abs(self.surface[:,2]-self.table_z)<.004]
        self.table_hull=ConvexHull(points[:,:2]).equations if len(points)>=3 else None

    def certify_home(self, points, volumes=()):
        # Known initial robot volume, not an assumption that its surroundings are free.
        self.home_tree=cKDTree(points)
        self.home_volumes=list(volumes)

    def certify_robot_state(self,points,volumes=()):
        """Replace the currently occupied proprioceptive robot volume.

        Classification uses these certificates only to replace UNKNOWN; an
        observed OCCUPIED endpoint always wins later in classify(). Historical
        robot poses must not accumulate as permanent free-space certificates.
        """
        self.home_tree=cKDTree(points)
        self.home_volumes=list(volumes)

    def integrate(self, obs, robot_points, robot_depth=None, robot_volumes=(),robot_visual_volumes=()):
        points=cloud(obs,2)
        valid=np.isfinite(points).all(axis=1)
        measured=obs.depth[::2,::2].ravel()
        valid &= np.isfinite(measured)&(measured>.05)&(measured<self.max_depth)
        mask=np.zeros_like(obs.depth,dtype=bool)
        blocked=np.zeros_like(obs.depth,dtype=bool)
        if robot_depth is not None:
            robot_valid=np.isfinite(robot_depth)&(robot_depth>.005)&(robot_depth<self.max_depth)
            # Exact matching returns are robot surface. At silhouette pixels the
            # scene renderer can instead return background *behind* the robot;
            # z-buffer ordering still proves that return is occluded self-edge,
            # not an obstacle. Never carve free space to that background depth.
            mask=robot_valid&(np.abs(robot_depth-obs.depth)<.0002)
            occluded=robot_valid&(robot_depth<=obs.depth+.0002)
            blocked=occluded&~mask
            valid &= ~occluded[::2,::2].ravel()
        elif len(robot_points): valid &= cKDTree(robot_points).query(points,distance_upper_bound=.014)[0]>.014
        # A scene depth endpoint strictly inside the robot's current *visual*
        # mesh is necessarily a self-render rasterization mismatch. This uses
        # neither padded collision proxies nor a distance halo, so an external
        # surface in front of / beside the robot remains evidence.
        for A,b,lo,hi in robot_visual_volumes:
            tolerance=VISUAL_SELF_FILTER_TOLERANCE_M
            ids=np.flatnonzero(valid&np.all((points>=lo-tolerance)&(points<=hi+tolerance),axis=1))
            if len(ids):
                with np.errstate(all='ignore'):
                    inside=np.all(points[ids]@A.T+b<=tolerance,axis=1)
                valid[ids[inside]]=False
        # Padded collision proxies are NOT a segmentation mask: an external
        # surface close to a robot link must remain an observed obstacle.
        for A,b,lo,hi in (() if robot_depth is not None else robot_volumes):
            ids=np.flatnonzero(valid & np.all((points>=lo)&(points<=hi),axis=1))
            if len(ids): valid[ids[np.all(points[ids]@A.T+b<1e-6,axis=1)]]=False
        points=points[valid]
        # Voxelized occupied endpoints. Keep immutable depth frames for free-space carving.
        if self.workspace_bounds is not None:
            lo,hi=np.asarray(self.workspace_bounds)
            points=points[np.all((points>lo)&(points<hi),axis=1)]
        allpts=np.vstack([self.surface,points])
        keys=np.round(allpts/self.resolution).astype(np.int32)
        _,idx=np.unique(keys,axis=0,return_index=True)
        self.surface=allpts[idx]; self.tree=cKDTree(self.surface)
        self._refresh_non_target()
        self.views.append(obs)
        self.self_masks.append(mask);self.occlusion_masks.append(blocked)
        if self.max_views is not None and len(self.views)>self.max_views:
            excess=len(self.views)-self.max_views
            del self.views[:excess];del self.self_masks[:excess];del self.occlusion_masks[:excess]

    def _classify_base(self, points):
        points=np.asarray(points)
        result=np.zeros(len(points),dtype=np.uint8)
        if not len(points): return result
        # Free space is measured only before a depth return; no inference behind occluders.
        for index,(obs,robot_mask) in enumerate(zip(self.views,self.self_masks)):
            blocked=self.occlusion_masks[index] if index<len(self.occlusion_masks) else np.zeros_like(robot_mask)
            uv,z=project(points,obs.K,obs.T_wc)
            uv=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(int)
            h,w=obs.depth.shape
            valid=(z>.01)&(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
            ids=np.where(valid)[0]
            u,v=uv[ids].T
            # Projective inverse sensor model at the measured pixel. Endpoint
            # inflation handles the finite occupied-cell footprint separately.
            d=obs.depth[v,u]
            # Ideal simulated depth has sub-mm precision. Occupied endpoints are
            # separately inflated below; a second 5 mm unknown shell would falsely
            # block motion immediately outside the known robot's own surface.
            # Self-render calibration tolerance is confined to verified robot pixels;
            # external-object returns never receive this tolerance.
            tolerance=np.where(robot_mask[v,u],.002,-.0002)
            measured=np.isfinite(d)&(d>.05)&~blocked[v,u]
            result[ids[measured & (z[ids]<np.minimum(d,self.max_depth)+tolerance)]]=FREE
        # Exact convex collision volumes occupied by the known initial robot are certified.
        # Do not inflate a nearest-neighbor point set into unobserved surrounding space.
        for A,b,lo,hi in self.home_volumes:
            ids=np.flatnonzero((result==UNKNOWN)&np.all((points>=lo-ROBOT_MODEL_TOLERANCE_M)&(points<=hi+ROBOT_MODEL_TOLERANCE_M),axis=1))
            if not len(ids): continue
            # 0.1 mm model/rasterization tolerance is explicit and much smaller
            # than the 5 mm map cell; the scene builder validates 10 mm initial
            # clearance. It applies only to known initial robot hulls, never to
            # arbitrary unknown workspace or external-object depth returns.
            with np.errstate(all='ignore'):
                inside=np.all(points[ids]@A.T+b<ROBOT_MODEL_TOLERANCE_M,axis=1)
            result[ids[inside]]=FREE
        if self.table_z is not None and self.table_hull is not None:
            inside=np.all(points[:,:2]@self.table_hull[:,:2].T+self.table_hull[:,2]<=.002,axis=1)
            # A fitted opaque support surface certifies its solid side as
            # occupied, not as endlessly unobservable free-space uncertainty.
            result[inside&(points[:,2]<=self.table_z+self.resolution*1.3)]=OCCUPIED
        if self.tree is not None:
            radius=self.resolution*1.3
            result[self.tree.query(points,distance_upper_bound=radius)[0]<radius]=OCCUPIED
        return result

    def classify(self,points):
        """Only measured space and exact robot volumes certify FREE.

        Neighbour voting cannot prove an unobserved cell is empty. In
        particular, classification must not depend on query batch size.
        """
        return self._classify_base(np.asarray(points))

    def clear_dynamic(self):
        # Old occupied/free evidence is invalid after manipulation; preserve only home certificate.
        self.views=[]; self.self_masks=[]; self.occlusion_masks=[]; self.surface=np.empty((0,3)); self.tree=None
        self.target_tree=None;self.non_target_tree=None;self.payload_tree=None
        self.payload_xy_tree=None;self.payload_points=None
        # The entire source-evidence epoch ends here, including all measured
        # surfaces/depth rays. This is not selective clearance of abstentions.
        self.retained_endpoints=np.empty((0,3));self.retained_endpoint_tree=None

    def free_margin(self,points):
        """Signed observed-space margin for IK optimization, NOT an execution bypass."""
        points=np.asarray(points); margin=np.full(len(points),-1.)
        for index,(obs,robot_mask) in enumerate(zip(self.views,self.self_masks)):
            blocked=self.occlusion_masks[index] if index<len(self.occlusion_masks) else np.zeros_like(robot_mask)
            uv,z=project(points,obs.K,obs.T_wc);uv=np.rint(uv).astype(int)
            h,w=obs.depth.shape
            ids=np.flatnonzero((z>.01)&(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1))
            u,v=uv[ids].T
            d=obs.depth[v,u]
            value=np.where(np.isfinite(d)&(d>.05)&~blocked[v,u],
                np.minimum(d,self.max_depth)+np.where(robot_mask[v,u],.002,-.0002)-z[ids],-1.)
            margin[ids]=np.maximum(margin[ids],value)
        for A,b,lo,hi in self.home_volumes:
            ids=np.flatnonzero((margin<.005)&np.all((points>=lo-.03)&(points<=hi+.03),axis=1))
            if len(ids):margin[ids]=np.maximum(margin[ids],ROBOT_MODEL_TOLERANCE_M-np.max(points[ids]@A.T+b,axis=1))
        if self.tree is not None:margin=np.minimum(margin,self.tree.query(points,distance_upper_bound=.05)[0]-self.resolution*1.3)
        return margin

    def set_target(self,points):
        self.target_tree=cKDTree(points)
        self._refresh_non_target()

    def set_payload_source(self,points):
        """Observed full-object surface at attachment, distinct from contact part."""
        points=np.asarray(points)
        self.payload_points=points.copy()
        self.payload_tree=cKDTree(points)
        self.payload_xy_tree=cKDTree(points[:,:2])

    def retire_owned_source(self,ownership,source_obs):
        """Retire only exact source endpoints after the caller carries them.

        This does not alter contact masks, depth rays, or FREE certificates.
        The accepted source cloud must already be in the collision envelope.
        """
        from .object_ownership import source_endpoint_retirement_mask
        remove=source_endpoint_retirement_mask(self.surface,ownership,source_obs,table_z=self.table_z)
        remove &= ~self._retained_endpoint_mask()
        self.surface=self.surface[~remove]
        self.tree=cKDTree(self.surface) if len(self.surface) else None
        self._refresh_non_target()
        return int(remove.sum())

    def retain_observed_endpoints(self,points):
        """Keep abstained measured endpoints through all dynamic retirement.

        Numerical source equality only: no new occupancy or spatial halo is
        introduced. The original depth rays remain unchanged.
        """
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all():
            raise ValueError('retained endpoints require finite observed points')
        self.retained_endpoints=np.unique(np.vstack([self.retained_endpoints,points]),axis=0)
        self.retained_endpoint_tree=cKDTree(self.retained_endpoints) if len(self.retained_endpoints) else None
        self._refresh_non_target()

    def _retained_endpoint_mask(self):
        return (np.zeros(len(self.surface),dtype=bool) if self.retained_endpoint_tree is None else
            self.retained_endpoint_tree.query(self.surface)[0]<=1e-9)

    def is_payload_source(self,points,radius=.00075):
        """Sparse observed payload surface neighborhood vacated by its motion.

        The 0.75 mm tube is deliberately much smaller than one 5 mm map voxel:
        it covers calibrated depth / transform quantization without turning the
        object's unobserved interior or projective shadow into free space.  The
        caller may use this to explain UNKNOWN cells only. An OCCUPIED cell is
        never waived, so a nearby independently observed obstacle remains a
        collision even inside this reconstruction tolerance.
        """
        if self.payload_tree is None:return np.zeros(len(points),dtype=bool)
        return self.payload_tree.query(points,distance_upper_bound=radius)[0]<radius

    def payload_vertical_lift_shadow(self,source_points,moved_points):
        """Certify a trailing point only along an observed source-surface tube.

        This applies to fixed-orientation vertical lift only. XY motion is
        forbidden and payload returns must densely cover the complete vertical
        interval at the source. Thus a block's already observed side wall can
        explain its own trailing sweep, while a mug-handle hole (a vertical
        evidence gap), object interior, and leading space above the object stay
        UNKNOWN.
        """
        source=np.asarray(source_points);moved=np.asarray(moved_points)
        result=np.zeros(len(moved),dtype=bool)
        if (len(source)!=len(moved) or self.payload_xy_tree is None or
                self.payload_points is None or self.table_z is None):return result
        tolerance=.00075
        candidate=np.linalg.norm(moved[:,:2]-source[:,:2],axis=1)<=tolerance
        candidate &= moved[:,2]>=source[:,2]-tolerance
        for i in np.flatnonzero(candidate):
            # Match the occupied endpoint footprint / diagonal pixel sampling
            # without crossing a two-voxel aperture.
            ids=self.payload_xy_tree.query_ball_point(source[i,:2],self.resolution*1.5)
            if not ids:continue
            z=np.sort(self.payload_points[ids,2])
            z=z[(z>=source[i,2]-tolerance)&(z<=moved[i,2]+tolerance)]
            samples=np.r_[source[i,2],z,moved[i,2]]
            if np.max(np.diff(np.sort(samples)))<=self.resolution*1.5+tolerance:
                result[i]=True
        return result

    def _refresh_non_target(self):
        if self.target_tree is None:
            self.non_target_tree=self.tree;return
        keep=self.target_tree.query(self.surface,distance_upper_bound=.004)[0]>=.004
        keep|=self._retained_endpoint_mask()
        self.non_target_tree=cKDTree(self.surface[keep]) if np.any(keep) else None

    def contact_allowed(self,points):
        allowed=self.is_target(points)
        if self.non_target_tree is not None:
            radius=self.resolution*1.3
            allowed &= self.non_target_tree.query(points,distance_upper_bound=radius)[0]>=radius
        return allowed

    def support_contact_allowed(self,points):
        """Allow contact only with the observed fitted support plane."""
        result=np.zeros(len(points),dtype=bool)
        if self.tree is None or self.table_z is None or self.table_hull is None:return result
        # A carried object starts the lift while its bottom is still touching
        # the support.  The fitted plane is continuous even where rasterized
        # depth endpoints are sparse, so membership cannot require a nearby
        # sample or positive projective free-space margin.  Instead, admit only
        # a narrow band inside the observed table hull and explicitly reject a
        # larger neighborhood containing any non-table, non-payload endpoint.
        plane_band=(points[:,2]>=self.table_z-self.resolution)&(
            points[:,2]<=self.table_z+self.resolution*2.0)
        if self.table_hull is not None:
            plane_band &= np.all(points[:,:2]@self.table_hull[:,:2].T+self.table_hull[:,2]<=.002,axis=1)
        neighbors=self.tree.query_ball_point(points,self.resolution*2.5)
        for i,ids in enumerate(neighbors):
            if not plane_band[i]:continue
            if not ids:
                result[i]=True
                continue
            # Every nearby occupied endpoint must belong to the support plane;
            # a low obstacle beside it must still reject placement. Target
            # endpoint association happens separately, never in this exception.
            ids=np.asarray(ids,dtype=int)
            # Match the occupied endpoint footprint used by classify(). A
            # sampled tabletop return can sit up to one inflated voxel above
            # the fitted plane; a distinct 10 mm obstacle remains rejected.
            result[i]=bool(np.all(np.abs(self.surface[ids,2]-self.table_z)<self.resolution*1.3))
        return result

    def remove_target_endpoints(self,points,radius=None):
        """Remove voxel representatives associated with an observed moving object.

        Association must cover the same finite cell footprint used to mark an
        endpoint occupied.  A 2 mm match left 5 mm voxel representatives behind
        even when cross-view RGB-D tracking identified the correct object.
        Support-plane samples are retained explicitly below.
        """
        if self.tree is None or not len(points):return
        radius=self.resolution*1.3 if radius is None else float(radius)
        keep=cKDTree(points).query(self.surface,distance_upper_bound=radius)[0]>radius
        if self.table_z is not None:keep|=np.abs(self.surface[:,2]-self.table_z)<.004
        keep|=self._retained_endpoint_mask()
        self.surface=self.surface[keep]
        self.tree=cKDTree(self.surface) if len(self.surface) else None
        self._refresh_non_target()

    def is_target(self,points,radius=.012):
        if self.target_tree is None: return np.zeros(len(points),dtype=bool)
        return self.target_tree.query(points,distance_upper_bound=radius)[0]<radius
