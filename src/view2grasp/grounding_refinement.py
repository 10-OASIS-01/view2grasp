"""Bounded same-crop semantic correction before deriving a grasp ray."""
import copy

import numpy as np

from .geometry import unproject
from .observation_crop import ObservationCrop,crop_grasp_proposal


def _possible_contact_pixels(obs,support_z):
    """Count a superset of executable contact pixels, never select a seed.

    pixel_point later requires depth below 3 m and world height at least 7 mm
    above support. Use the looser existing 6 mm object filter here; omit robot,
    edge and semantic exclusions, so only zero can disqualify the whole crop.
    """
    if support_z is None or not np.isscalar(support_z) or not np.isfinite(support_z):
        raise ValueError('grounding refinement requires a finite observed support height')
    valid=np.isfinite(obs.depth)&(obs.depth>.05)&(obs.depth<3.)
    y,x=np.where(valid)
    points=unproject(np.column_stack([x,y]),obs.depth[y,x],obs.K,obs.T_wc)
    return int(np.count_nonzero(points[:,2]>support_z+.006))


def refine_grasp_grounding(obs,coarse,resolver,vlm,instruction,history,support_z,
                           *,robot_depth=None,event=None):
    """Return restored decision and cached source-image grounding, or raise.

    There is one initial answer and at most one correction of an actual resolver
    ValueError. Protocol retries remain the VLM's bounded responsibility. No
    coarse fallback, candidate generation, new observation or motion occurs.
    """
    emit=event or (lambda *args,**kwargs:None)
    crop=crop_grasp_proposal(obs,coarse,robot_depth=robot_depth)
    proposal_context=crop.provenance()
    possible_pixels=_possible_contact_pixels(crop.observation,support_z)
    if possible_pixels==0:
        # Replace a provably unusable crop before any semantic call. This is
        # the same acquired frame, not recovery from an abort or permission to
        # use the coarse proposal. Identity bounds preserve the source pixels.
        h,w=obs.depth.shape
        crop=ObservationCrop(obs,(0,0,w,h),(h,w),robot_depth,0.,0,min(h,w),0,
            'full_frame_after_empty_proposal_crop')
    image_context=crop.provenance()
    emit('grounding_refinement_request',frame_id=obs.frame_id,
        coarse_decision=coarse.model_dump(),planning_history=copy.deepcopy(history),
        proposal_image_context=proposal_context,image_context=image_context,
        crop_depth_screen={'possible_contact_pixels':possible_pixels,
            'full_frame_fallback':possible_pixels==0,
            'basis':'finite_depth_strictly_between_0.05_and_3m_and_world_z_above_support_plus_0.006m',
            'scope':'Entire crop; necessary conditions only, no semantic localization or contact permission.'})
    # Older full images and crops use different coordinate systems. Their
    # decisions remain in local logs, never in this new image's transaction.
    current_history=[{'grounding_refinement_transaction':{
        'frame_id':obs.frame_id,'camera_id':obs.camera_id,'image_context':image_context,
        'full_frame_fallback':possible_pixels==0,
        'instructions':'Inspect only this supplied current image and the task instruction. '
            'Return fresh object and grasp_part regions or abort; no earlier pixel coordinates apply.'}}]
    feedback=None
    for repair in range(2):
        options={} if feedback is None else {'grounding_refinement_feedback':feedback}
        raw=vlm.decide(crop.observation,instruction,copy.deepcopy(current_history),force_grasp=True,
            grounding_refinement=image_context,derived_image_context=image_context,**options)
        restored=crop.restore_grasp_decision(raw)
        emit('grounding_refinement_decision',frame_id=obs.frame_id,repair=repair,
            crop_decision=raw.model_dump(),restored_decision=restored.model_dump(),image_context=image_context)
        if restored.action=='abort':return restored,None
        try:
            resolved=resolver.resolve_grasp(obs,restored,support_z,robot_depth)
        except ValueError as exc:
            emit('grounding_refinement_geometry_rejected',frame_id=obs.frame_id,repair=repair,
                reason=str(exc),diagnostics=getattr(exc,'diagnostics',None))
            if repair:raise
            feedback={'frame_id':obs.frame_id,'camera_id':obs.camera_id,
                'coordinate_space':'supplied_crop_normalized_pixel_centers',
                'image_shape_hw':list(crop.observation.depth.shape),
                'repair_attempt':1,'max_geometry_repairs':1,'previous_crop_decision':raw.model_dump(),
                'geometry_diagnostics':crop.localize_grounding_error(exc)}
            emit('grounding_refinement_feedback',**feedback)
        else:
            resolved.report['image_context']=image_context
            return restored,resolved
