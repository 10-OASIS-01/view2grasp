"""Metric camera geometry and local, image-conditioned grasp proposals."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .types import Observation, AgentDecision, GraspCandidate


def orbit_pose(azimuth,elevation,distance):
    """Known camera orbit, independent of all task-object state."""
    az,el=np.deg2rad([azimuth,elevation]);focus=np.array([-.12,0,.95])
    pos=focus+distance*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    z=focus-pos;z/=np.linalg.norm(z)
    x=np.cross(z,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(z,x)
    T=np.eye(4);T[:3,:3]=np.column_stack([x,y,z]);T[:3,3]=pos
    return T


def unproject(uv, depth, K, T_wc):
    uv = np.asarray(uv)
    rays = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1) @ np.linalg.inv(K).T
    pc = rays * np.asarray(depth)[..., None]
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3]


def project(points, K, T_wc):
    points=np.asarray(points);K=np.asarray(K);T_wc=np.asarray(T_wc)
    if not np.isfinite(points).all() or not np.isfinite(K).all() or not np.isfinite(T_wc).all():
        raise ValueError('nonfinite camera projection input')
    with np.errstate(all='ignore'):
        pc = (points - T_wc[:3, 3]) @ T_wc[:3, :3]
        pix = pc @ K.T
        uv=pix[..., :2] / pix[..., 2:]
    return uv, pc[..., 2]


def cloud(obs: Observation, stride=2):
    h, w = obs.depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    return unproject(np.stack([u, v], -1), obs.depth[::stride, ::stride], obs.K, obs.T_wc).reshape(-1, 3)


def fit_table(points, rng):
    """RANSAC horizontal support plane; no configured table height is read."""
    points = points[np.isfinite(points).all(axis=1)]
    # Fit the support plane in the declared tabletop workspace, not the room
    # floor (which dominates a high overview). This broad ROI is workspace
    # calibration, not the simulator's table height or task-object metadata.
    points=points[(np.abs(points[:,0])<.5)&(np.abs(points[:,1])<.5)&(points[:,2]>.3)&(points[:,2]<1.2)]
    if len(points) < 100:
        raise ValueError("insufficient depth for table")
    z = points[:, 2]
    sample = rng.choice(len(z), min(100, len(z)), replace=False)
    candidates = z[sample]
    counts = np.array([(np.abs(z - c) < .004).sum() for c in candidates])
    best = candidates[counts.argmax()]
    return float(np.median(z[np.abs(z - best) < .004]))


def pixel_point(obs, xy, table_z=None):
    h, w = obs.depth.shape
    u, v = np.rint(np.asarray(xy) * [w - 1, h - 1]).astype(int)
    patch = obs.depth[max(0, v-1):v+2, max(0, u-1):u+2]
    d = obs.depth[v, u]
    if not np.isfinite(patch).all() or not .05 < d < 3:
        raise ValueError("invalid depth")
    if np.ptp(patch) > .025:
        raise ValueError("depth edge / hole; reground on a solid surface")
    p = unproject([u, v], d, obs.K, obs.T_wc)
    if table_z is not None and p[2] < table_z + .007:
        raise ValueError("grounding on table/background instead of object")
    return p


def has_explicit_object_seeds(decision):
    """Membership seeds, excluding polygon boundary vertices and part seeds."""
    return any(g.role=='object' and g.kind!='polygon' and g.points
               for g in decision.groundings)


def extract_target(obs, decision, table_z,observed_points=None,object_mask=None):
    if has_explicit_object_seeds(decision):
        # Only the resolver's seed-selected measured components are membership
        # evidence. A broad box or projection of historical map surfaces cannot
        # establish identity or visibility of pixels behind the current depth.
        if object_mask is None:
            raise ValueError('explicit object seeds require a resolved object mask')
        if decision.frame_id!=obs.frame_id or decision.camera_id!=obs.camera_id:
            raise ValueError('explicit object components require the current frame and camera')
        mask=np.asarray(object_mask,dtype=bool)
        if mask.shape!=obs.depth.shape:
            raise ValueError('object mask resolution differs from observation')
        anchor=pixel_point(obs,decision.point,table_z)
        v,u=np.where(mask)
        depth=obs.depth[v,u]
        pts=unproject(np.stack([u,v],-1),depth,obs.K,obs.T_wc)
        if (not np.isfinite(pts).all() or np.any((depth<=.05)|(depth>=10))
                or np.any(pts[:,2]<=table_z+.006)):
            raise ValueError('resolved object mask contains invalid measured surface')
        if len(pts)<8:
            raise ValueError('too few object depth points')
        # Preserve every selected current-frame component, including islands
        # that stride sampling or a second contact-only flood would discard.
        return anchor,pts
    h, w = obs.depth.shape
    x0, y0, x1, y1 = np.asarray(decision.bbox) * [w-1, h-1, w-1, h-1]
    v, u = np.mgrid[int(y0):int(y1)+1:2, int(x0):int(x1)+1:2]
    pts = unproject(np.stack([u,v], -1), obs.depth[v,u], obs.K, obs.T_wc).reshape(-1,3)
    if object_mask is not None:
        pts=pts[object_mask[v,u].ravel()]
    pts = pts[(pts[:,2] > table_z+.006) & np.isfinite(pts).all(axis=1)]
    anchor = pixel_point(obs, decision.point, table_z)
    if observed_points is not None:
        # Multi-view surface association by depth connectivity, not simulator IDs.
        extra=np.asarray(observed_points)
        extra=extra[(extra[:,2]>table_z+.006)&(np.linalg.norm(extra-anchor,axis=1)<.2)]
        if object_mask is not None and len(extra):
            uv,z=project(extra,obs.K,obs.T_wc)
            ij=np.rint(uv).astype(int)
            valid=(z>0)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
            ids=np.flatnonzero(valid)
            extra=extra[ids[object_mask[ij[ids,1],ij[ids,0]]]]
        pts=np.vstack([pts,extra])
    if len(pts) < 8:
        raise ValueError("too few object depth points")
    # Select the connected surface containing the grounded point, never simulator segmentation.
    tree = cKDTree(pts)
    start = int(tree.query(anchor)[1])
    seen = {start}; frontier = [start]
    while frontier:
        nxt = set(j for ids in tree.query_ball_point(pts[frontier], .013) for j in ids) - seen
        seen.update(nxt); frontier = list(nxt)
    target = pts[list(seen)]
    return anchor, target


def part_cloud(obs,decision,target,mask=None):
    uv,z=project(target,obs.K,obs.T_wc);h,w=obs.depth.shape
    box=np.asarray(decision.part_bbox)*[w-1,h-1,w-1,h-1]
    keep=(z>0)&(uv[:,0]>=box[0]-2)&(uv[:,0]<=box[2]+2)&(uv[:,1]>=box[1]-2)&(uv[:,1]<=box[3]+2)
    if mask is not None:
        ij=np.rint(uv).astype(int)
        valid=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        indices=np.flatnonzero(valid)
        visible=np.zeros(len(target),dtype=bool)
        visible[indices]=mask[ij[indices,1],ij[indices,0]]
        keep &= visible
    return target[keep]


def grasp_candidates(obs: Observation, decision: AgentDecision, table_z, rotation_hint=None,grasp_offset=None,observed_points=None,object_mask=None,part_mask=None):
    if decision.frame_id != obs.frame_id:
        raise ValueError("stale frame")
    if has_explicit_object_seeds(decision) and part_mask is None:
        raise ValueError('explicit object seeds require a resolved grasp-part mask')
    p, target = extract_target(obs, decision, table_z,observed_points,object_mask)
    part=part_cloud(obs,decision,target,part_mask)
    if len(part)<4:raise ValueError('insufficient observed grasp-part geometry')
    grasp_offset=np.zeros(3) if grasp_offset is None else np.asarray(grasp_offset)
    ray = p - obs.T_wc[:3,3]; ray /= np.linalg.norm(ray)
    ref = np.array([0., 0., 1.]) if abs(ray[2]) < .95 else np.array([1.,0.,0.])
    bx = np.cross(ref, ray); bx /= np.linalg.norm(bx)
    by = np.cross(ray, bx)
    directions = [(ray,0.)]
    for angle in (7.5, 15.):
        for phi in np.linspace(0,2*np.pi,8,endpoint=False):
            a = np.cos(np.deg2rad(angle))*ray + np.sin(np.deg2rad(angle))*(np.cos(phi)*bx+np.sin(phi)*by)
            directions.append((a,angle))
    result = []
    for a, angle in directions:
        x = np.cross(ref,a) if rotation_hint is None else rotation_hint[:,0]-np.dot(rotation_hint[:,0],a)*a
        if np.linalg.norm(x)<1e-6: x=np.cross(ref,a)
        x /= np.linalg.norm(x)
        y = np.cross(a,x)
        local=part[np.linalg.norm(part-p,axis=1)<.04]
        if len(local)<4:continue
        planar=(local-p)@np.column_stack([x,y])
        values,vectors=np.linalg.eigh(np.cov(planar.T))
        if values[1]>2*max(values[0],1e-7):
            x=np.column_stack([x,y])@vectors[:,0];y=np.cross(a,x)
        # Geometric antipodal symmetry does NOT imply identical wrist reachability.
        for roll in np.linspace(0,2*np.pi,24,endpoint=False):
            # Panda grip_site z = approach; x = closing (finger bodies have a 90-degree rotation).
            R = np.column_stack([x,y,a]) @ Rotation.from_rotvec([0,0,roll]).as_matrix()
            nearby = local
            if len(nearby)<4: continue
            width = float(np.ptp((nearby-p) @ R[:,0]) + .008)
            if width > .077: continue
            for insertion in (.008,.016,.024):
                center = p + a*insertion
                tcp=center-R@grasp_offset
                result.append(GraspCandidate(tcp,R,max(.018,width),a,angle/15+insertion,angle,p.copy(),center))
    hints=[g for g in decision.groundings if g.kind=='path' and g.role=='end_effector_path']
    if hints:
        hint=hints[0]
        if hint.camera_id!=obs.camera_id or hint.frame_id!=obs.frame_id:raise ValueError('stale path hint')
        h,w=obs.depth.shape
        direction=(np.array(hint.points[-1])-hint.points[-2])*[w-1,h-1]
        if np.linalg.norm(direction)>1e-8:
            direction/=np.linalg.norm(direction)
            for candidate in result:
                uv,z=project(np.stack([candidate.grasp_center-.08*candidate.approach,candidate.grasp_center]),obs.K,obs.T_wc)
                projected=uv[1]-uv[0]
                if np.all(z>0) and np.linalg.norm(projected)>1e-6:
                    # Weak ranking hint only. No background depth is sampled,
                    # no candidate is added, and the 15-degree cone is unchanged.
                    candidate.score+=.02*(1-float(projected@direction/np.linalg.norm(projected)))
    return sorted(result,key=lambda g:g.score), target
