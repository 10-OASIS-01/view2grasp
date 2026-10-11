"""Bounded same-crop semantic correction before deriving a grasp ray."""
from .observation_crop import crop_grasp_proposal


def refine_grasp_grounding(obs,coarse,resolver,vlm,instruction,history,support_z,
                           *,robot_depth=None,event=None):
    """Return restored decision and cached source-image grounding, or raise.

    There is one initial answer and at most one correction of an actual resolver
    ValueError. Protocol retries remain the VLM's bounded responsibility. No
    coarse fallback, candidate generation, new observation or motion occurs.
    """
    emit=event or (lambda *args,**kwargs:None)
    crop=crop_grasp_proposal(obs,coarse,robot_depth=robot_depth)
    image_context=crop.provenance()
    emit('grounding_refinement_request',frame_id=obs.frame_id,
        coarse_decision=coarse.model_dump(),image_context=image_context)
    feedback=None
    for repair in range(2):
        options={} if feedback is None else {'grounding_refinement_feedback':feedback}
        raw=vlm.decide(crop.observation,instruction,history,force_grasp=True,
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
