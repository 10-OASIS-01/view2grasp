"""Next-view advice from known robot geometry, NEVER unseen scene renders.

Scores estimate visibility, not free space. The map is only updated with an
actual RGB-D observation after the agent has selected and moved the camera.
"""
from dataclasses import replace
import numpy as np
from scipy.ndimage import minimum_filter
from .geometry import orbit_pose,project
from .mapping import UNKNOWN
from .types import GraspCandidate


def nominal_grasp(robot,map_,obs):
    """Coarse reach region from observed foreground; not a grasp for execution."""
    points=map_.surface
    keep=(points[:,2]>map_.table_z+.007)&(points[:,2]<map_.table_z+.20)&(np.abs(points[:,0])<.3)&(np.abs(points[:,1])<.3)
    points=points[keep]
    if len(points)<8:return None
    position=np.median(points,axis=0)
    approach=position-obs.T_wc[:3,3];approach/=np.linalg.norm(approach)
    hint=robot.pose(obs.q)[1][:,0]
    x=hint-np.dot(hint,approach)*approach;x/=np.linalg.norm(x)
    rotation=np.column_stack([x,np.cross(approach,x),approach])
    return GraspCandidate(position,rotation,.06,approach,0.,0.)


def swept_unknown(robot,map_,obs,grasp):
    pre=robot.ik(grasp.position-grasp.approach*.08,grasp.rotation,obs.q)
    if pre is None:return np.empty((0,3))
    poses=list(np.linspace(obs.q,pre,12));q=pre
    for offset in np.linspace(.08,0,12)[1:]:
        q=robot.ik(grasp.position-grasp.approach*offset,grasp.rotation,q)
        if q is None:break
        poses.append(q)
    if q is not None:
        for dz in np.linspace(.01,.13,10):
            q=robot.ik(grasp.position+[0,0,dz],grasp.rotation,q)
            if q is None:break
            poses.append(q)
    points=np.vstack([robot.points(q,obs.gripper_q)[0] for q in poses])
    points=points[map_.classify(points)==UNKNOWN]
    if not len(points):return points
    _,index=np.unique(np.round(points/.0025).astype(int),axis=0,return_index=True)
    return points[index]


def _visible(robot,obs,points,view,observed_points=None):
    hypothetical=replace(obs,T_wc=orbit_pose(*view),view=view)
    return calibrated_visibility(robot,hypothetical,points,observed_points)


def calibrated_visibility(robot,obs,points,observed_points=None):
    """Predict visibility from calibration and known geometry, never scene RGB-D."""
    own_depth=robot.depth(obs)
    uv,z=project(points,obs.K,obs.T_wc);uv=np.rint(uv).astype(int)
    h,w=own_depth.shape
    ids=np.flatnonzero((z>.01)&(uv[:,0]>0)&(uv[:,0]<w-1)&(uv[:,1]>0)&(uv[:,1]<h-1))
    visible=np.zeros(len(points),dtype=bool)
    if len(ids):
        u,v=uv[ids].T
        # Do not score a grazing ray as useful recovery coverage. Sub-pixel
        # rasterization and the self-depth mask can leave a point unknown even
        # when it is only fractions of a millimeter in front of the current
        # robot silhouette. Five millimeters matches one map voxel.
        visible[ids]=z[ids]<own_depth[v,u]-.005
    if observed_points is not None and len(observed_points):
        # Conservative point-cloud z-buffer made only from surfaces already
        # observed in RGB-D. This prevents a recovery view from looking through
        # the known back of an object to a handle or block hidden on its front.
        scene_uv,scene_z=project(observed_points,obs.K,obs.T_wc)
        scene_uv=np.rint(scene_uv).astype(int)
        ok=(scene_z>.01)&(scene_uv[:,0]>=0)&(scene_uv[:,0]<w)&(scene_uv[:,1]>=0)&(scene_uv[:,1]<h)
        flat=np.full(h*w,np.inf)
        indices=scene_uv[ok,1]*w+scene_uv[ok,0]
        np.minimum.at(flat,indices,scene_z[ok])
        zbuffer=minimum_filter(flat.reshape(h,w),size=3,mode='constant',cval=np.inf)
        if len(ids):visible[ids]&=z[ids]<=zbuffer[v,u]+.008
    return visible


def suggest_mounted_cameras(robot,calibrated_observations,points,budget=3,observed_points=None):
    """Rank real mounted cameras by predicted coverage of rejected path points."""
    if not len(points):return {'cameras':[],'predicted_coverage':1.,'uncoverable':0}
    names=[];masks=[]
    for obs in calibrated_observations:
        visible=calibrated_visibility(robot,obs,points,observed_points)
        names.append(obs.camera_id)
        masks.append(int.from_bytes(np.packbits(visible,bitorder='little').tobytes(),'little'))
    chosen=[];covered=0
    for _ in range(min(budget,len(names))):
        remaining=[i for i in range(len(names)) if i not in chosen]
        if not remaining:break
        index=max(remaining,key=lambda i:(covered|masks[i]).bit_count())
        if (covered|masks[index])==covered:break
        chosen.append(index);covered|=masks[index]
    all_mask=0
    for mask in masks:all_mask|=mask
    return {'cameras':[names[i] for i in chosen],
        'predicted_coverage':covered.bit_count()/len(points),
        'unknown_points':len(points),'uncoverable':len(points)-all_mask.bit_count(),
        'note':'Calibration and robot/observed geometry estimate only. Unvisited camera images were NOT acquired.'}


def _semantic_projection(obs,points,view,visible):
    """Estimate annotation resolution, without claiming current visibility."""
    uv,_=project(points[visible],obs.K,orbit_pose(*view))
    pixels=np.unique(np.rint(uv).astype(int),axis=0)
    span=np.ptp(pixels,axis=0) if len(pixels) else np.zeros(2)
    minor_span=0.
    if len(pixels)>=3:
        centered=pixels-pixels.mean(axis=0)
        _,_,axes=np.linalg.svd(centered,full_matrices=False)
        minor_span=float(np.ptp(centered@axes[1]))
    return {'view':list(view),'visible_reference_points':int(np.count_nonzero(visible)),
        'unique_visible_pixels':len(pixels),'pixel_span_xy':span.tolist(),
        'minor_axis_span_pixels':minor_span}


def suggest_views(robot,obs,points,budget=3,beam_width=32,focus_points=None,observed_points=None,
                  required_points=None,semantic_focus=None):
    if not len(points) and (semantic_focus is None or not len(semantic_focus)):
        return {'views':[],'predicted_coverage':1.,'uncoverable':0}
    # A coarse orbit grid is sufficient because advice is only a visibility
    # prior; the VLM still selects the actual continuous view. Keeping this
    # small also makes failure-triggered advice cheap enough for online use.
    views=[(float(az),float(el),1.3) for el in (-10,0,30,60,80) for az in range(-180,180,30)]
    masks=[];focus_masks=[];required_masks=[];semantic=[];semantic_scores=[]
    for view in views:
        visible=_visible(robot,obs,points,view,observed_points)
        masks.append(int.from_bytes(np.packbits(visible,bitorder='little').tobytes(),'little'))
        focus_visible=np.zeros(0,dtype=bool) if focus_points is None else _visible(
            robot,obs,focus_points,view,observed_points)
        focus_masks.append(int.from_bytes(np.packbits(focus_visible,bitorder='little').tobytes(),'little'))
        required_visible=np.zeros(0,dtype=bool) if required_points is None else _visible(
            robot,obs,required_points,view,observed_points)
        required_masks.append(int.from_bytes(np.packbits(required_visible,bitorder='little').tobytes(),'little'))
        if semantic_focus is not None and len(semantic_focus):
            detail=_semantic_projection(obs,np.asarray(semantic_focus),view,
                _visible(robot,obs,semantic_focus,view,observed_points))
            semantic.append(detail)
            semantic_scores.append((detail['unique_visible_pixels'],detail['minor_axis_span_pixels']))
        else:semantic_scores.append((0,0.))
    # Beam search for complementary views rather than three near-identical maxima.
    beam=[(0,0,0,())]
    for _ in range(budget):
        candidates={}
        for required_mask,mask,focus_mask,chosen in beam:
            for index,(other,other_focus,other_required) in enumerate(zip(masks,focus_masks,required_masks)):
                if index in chosen:continue
                key=tuple(sorted((*chosen,index)));combined=mask|other
                candidates[key]=(required_mask|other_required,combined,focus_mask|other_focus)
        beam=sorted(((required,mask,focus,key) for key,(required,mask,focus) in candidates.items()),
            # Required arm and payload coverage remain the first priorities.
            # A mandatory semantic check needs distinct image pixels, not many
            # reference points compressed onto the same few grazing-view pixels.
            # For a set of views, prefer its best individual annotation image;
            # pixels from different camera projections cannot be added together.
            key=lambda v:(v[0].bit_count(),v[1].bit_count(),
                *max(semantic_scores[i] for i in v[3]),v[2].bit_count()),reverse=True)[:beam_width]
    required_mask,mask,focus_mask,chosen=beam[0];all_mask=0
    for other in masks:all_mask|=other
    result={'views':[views[i] for i in chosen],'predicted_coverage':mask.bit_count()/len(points) if len(points) else 1.,
        'unknown_points':len(points),'uncoverable':len(points)-all_mask.bit_count(),
        'note':'Robot-model visibility estimate only. Unvisited scene images were NOT acquired.'}
    if focus_points is not None and len(focus_points):
        result['predicted_target_visibility']=focus_mask.bit_count()/len(focus_points)
    if required_points is not None and len(required_points):
        all_required=0
        for other in required_masks:all_required|=other
        result['predicted_required_coverage']=required_mask.bit_count()/len(required_points)
        result['required_uncoverable']=len(required_points)-all_required.bit_count()
    if semantic:
        result['predicted_semantic_focus']=[semantic[i] for i in chosen]
        result['semantic_focus_note']='Projection of previously observed reference surfaces only; pixel extent is not semantic verification, current visibility, or contact permission.'
    return result
