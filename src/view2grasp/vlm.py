"""OpenAI-compatible HTTP transport, image grounding and auditable call logs."""
import base64
import io
import json
import time
import copy
import hashlib
import math
from pathlib import Path
import httpx
import numpy as np
from PIL import Image
from .types import AgentDecision, Observation
from .vlm_settings import VLMSettings
from .geometry import project

LEGACY_SYSTEM = '''You control a Franka Panda in a tabletop simulator using only images.
Return one JSON object (no markdown). Actions: observe, grasp, abort.
Every action includes action, frame_id, reason (short operational explanation).
observe additionally uses azimuth [-180,180] degrees, elevation [-15,85] degrees,
distance [.55,1.3] meters. Camera looks at the workspace center. Azimuth 0 is
the +world-X side, 90 is +Y. Higher elevation is more top-down.
grasp includes target, part, point [x,y], bbox [left,top,right,bottom],
part_bbox [left,top,right,bottom], all normalized to [0,1] in the CURRENT image.
For pick/place also give place_point [x,y] at the yellow destination center.
Click on a SOLID surface of the requested graspable part, never a hole/background.
For a mug grasp the VERTICAL MIDPOINT of the OUTERMOST VERTICAL HANDLE BAR
(roughly 48--52% of the way from its bottom to top), not the cup body, rim,
handle opening, or either top/bottom junction. The task mug has a deliberately
large through-hole, so the midpoint maximizes clearance for the complete Panda
hand above and below. Leave clearance from the cup wall, table, and both handle
junctions for the non-contact finger structure.
Use a LOWERED CLOSE SIDE VIEW (elevation -8 to -12 degrees and distance
0.7--0.9 m) for the final mug grounding. This places the virtual camera near
the short handle's height, producing a large image while keeping the ray nearly
horizontal so it does not sweep through the rim. View the handle BROADSIDE so its open hole
is visibly separated: the camera ray must be approximately normal to the
vertical handle plane, allowing one finger to enter the hole. A radial view
from directly outside the handle makes the bar silhouette but cannot insert
the fingers. Do not use an elevated, close, or top-down final mug view.
In the INITIAL top overview, estimate the world-XY direction from the cup center
to the handle and RECORD that estimate explicitly in your reason before following
the supplied coverage views. If coverage_advice contains an EMPTY views list,
do not invent a generic clearance view: move DIRECTLY to the computed lowered
broadside grounding view. After any supplied coverage views, use the remembered
direction and do NOT request another top overview. Choose a side-view azimuth
approximately PLUS OR MINUS 90 degrees from that direction. Of the two broadside
choices, use the one closer to azimuth +180/-180 degrees: the calibrated Panda
base is on the world -X side of the workspace, and the opposite broadside often
has no IK solution. State the computed final azimuth in the reason.
The camera-to-click ray determines gripper approach (at most 15 deg adjustment).
Prefer views that expose solid grasp surfaces and finger clearance. A nearly
top-down view works well for a block. Use the low broadside rule above for mugs.
If the planner reports unknown space, choose an unvisited view to reveal it.
History may include coverage_advice: complementary viewpoints predicted using
ONLY the known robot model and previously observed foreground. Prefer these
suggestions when they help whole-arm coverage, but choose the view yourself.
They are estimates, NOT unseen scene observations and NOT permission to move
the arm through unknown space. A top-down view alone cannot reveal free space
underneath or behind the robot links.
The camera orbit center is [-0.12,0,0.95] meters in world coordinates; a
horizontal (0 degree) view is above the table and below the resting wrist.
If grasp planning fails due to geometry or reachability, try a different angle.
There are at most SIX total RGB-D observations. For a mug with empty initial
coverage advice, ground from the lowered broadside by observation TWO whenever
possible: reserve observations three and four for planner-targeted recovery and
observations five and six for post-grasp mapping of the moved gripper and cup.
When all six mug frames are needed for safety mapping, the independent evaluator
performs final hold verification without giving truth back to you. For the other
tasks, normally return to a grasp-friendly view by observation three and retain
a final visual verification frame.
After a grounding has been submitted and the planner reports UNKNOWN SPACE,
the stored grasp ray is preserved: mapping-only recovery views do not change
it. Follow the provided recovery coverage views as closely as possible, even
if a recovery frame is not itself a good grounding view. Avoid repeats.
Do not claim success; success is verified after execution.'''

SYSTEM = '''You operate a robot using calibrated RGB-D observations, task language,
proprioception, and planning feedback. Return ONE JSON object without markdown.
Actions: observe, navigate, grasp, place, finish, abort. Always include action,
frame_id, camera_id, and a short operational reason. Use only the current image.
Top-level frame_id and camera_id ALWAYS identify the CURRENT INPUT image,
including for observe actions. Every grounding uses those same input IDs.
NEVER put a requested future camera in top-level camera_id or in a grounding.
groundings is a list of associated visual proposals. Each proposal includes:
kind (points, box, polygon, path), role, frame_id, camera_id, target_id, source="vlm".
roles: object, grasp_part, grasp_contact, place_region, navigation_goal,
ground_path, end_effector_path, look_at. Coordinates are normalized [0,1].
The image origin is top-left; x increases rightward and y downward.
kind is the representation, NOT the semantic role: an object box has
kind="box", role="object", and a box field. Never use kind="object".
points uses points:[[x,y],...]; box uses box:[left,top,right,bottom]; polygon
uses points as its outer boundary and optional holes:[[[x,y],...],...]; path
uses points in traversal order. Optional negative_points identify exclusions.
Combine representations only when useful. Do not output huge raster masks.
All proposals referring to one object and its parts MUST share the same target_id;
e.g. object=mug and grasp_part=mug_handle both use target_id="mug".
Grasp: ground object and grasp_part (box/polygon/points); optionally give
grasp_contact points on SOLID material, never a hole, background or robot.
The planner derives contacts from regions when no point is supplied. For a small
solid object whose graspable face spans only a few pixels, prefer explicit object
and grasp_part regions and OMIT optional grasp_contact / point coordinates: a
coarsely rounded click can move the approach toward the face edge. Give a point
only when a particular visible contact location is needed and can be located
precisely. Explicit points remain binding and are never silently recentered. For a
handle expose both the material and aperture; use negative points/holes to
exclude background. Grasp approach stays within 15 degrees of the camera ray
to the resolved contact. Choose a view giving reachable approach and clearance.
Place: ground a support region large enough for the object and free of clutter.
Navigate: ground navigation_goal on visible FLOOR and look_at on the destination;
optionally propose a ground_path. Paths are planning hints, NOT certified free
space. An end_effector_path has no implied depth from the image background.
For tabletop observe, specify azimuth [-180,180], elevation [-15,85], distance
[.55,1.3] as three explicit JSON fields, not only in the reason text.
Orbit center [-.12,0,.95]; azimuth 0 is world +X, 90 is +Y. The Panda
base is on -X. Low broadside views expose mug-handle apertures; overhead views
often suit blocks. Coverage advice uses only observed geometry and robot model.
tabletop_camera_context projects known world axes into THIS image. Each delta
is the normalized image displacement for a +0.1 m world-axis step at the stated
workspace reference. Use these local directions to infer world-XY azimuth;
image-left is NOT automatically world -X. They are calibration hints, not an
object position or observed free space. Side views may collapse horizontal axes.
For mounted cameras, observe selects observation_camera from available_cameras and optional
observation_motion: none, turn_left, turn_right, wrist. Motion requires planning;
do not request a virtual camera teleport. ONLY observation_camera identifies the
requested NEXT camera; top-level camera_id and all grounding camera_id values
still identify the camera of the supplied image. For every mounted-camera
response OMIT azimuth, elevation, and distance entirely. They are tabletop-only
fields: do not send null or numeric placeholders for them.
If must_ground_now is true, grasp or abort. If must_observe_now is true, observe.
Unknown-space rejection means acquire relevant additional observations. Respect
remaining budgets. finish reports a visual hypothesis, not evaluator success.
held_verification_context is a separate camera-only transaction after lifting.
Its priority is to clearly see the SAME held object and permitted part in ONE
different real view, so their explicit annotations can be checked. It supplies
NO UNKNOWN path evidence: do not prefer corridor coverage over recognizing the
held material. Keep the robot, fixed attachment reference and original grasp
ray unchanged; never propose a new grasp. The attachment projection and view
advice are hypotheses, not current visibility, grasp success or motion approval.
An earlier pre-lift camera view may be revisited. Only one reserved observation
remains; using it leaves none for transfer mapping. Strict verification follows.
The mapping-only coverage preferences below do not apply to this transaction.
Tabletop mapping-only recovery preserves the previously grounded grasp and its
camera-to-contact approach ray. The recovery camera does NOT select a new grasp
approach: it should reveal UNKNOWN space along the stored robot/payload path.
Prefer an unvisited view from the latest coverage_advice, even when that view is
poor for recognizing or grasping the target. Avoid repeating the current view
when the scene has not moved: another identical image adds no coverage. Explain
which advised view you choose or why a different unvisited view is needed.
mapping_context exposes the latest advice and which advised views have not
been acquired in that recovery transaction. Prefer its remaining views over
remembering the list from older history. These are predicted viewpoints, not
observed free space; every motion still needs geometric validation.
approach_reselection_context instead reports a bounded geometry or kinematics
search limitation for the previous grasp direction, not global infeasibility.
Choose a different semantic observation of the requested graspable part;
after it is acquired, ground again to select a NEW approach ray. It is not a
mapping-only transaction and old grasp coordinates do not apply to the new image.
Both approach_reselection_context and current_frame_reselection_context permit
revisiting an earlier view: the mapping-only preference for unvisited coverage
does not apply. Use supplied robot-only kinematic view advice to prioritize
directions; it does not prove target visibility, free space, or a safe path.
annotation_context requests regions in the supplied current image only. It does
not acquire another observation or execute a camera command. If this image is a
calibrated crop, ALL coordinates refer to that crop's image_size; never reuse old
full-frame coordinates. The crop location is only a projection hypothesis from
previous measurements, not proof that the target is visible there. Mark only
visible requested material and omit regions when you cannot identify it.
grounding_refinement_context instead requests a fresh grasp or abort in a crop
of this already acquired frame, BEFORE a grasp ray is formed. Use crop-normalized
coordinates and fresh explicit object and grasp_part regions. It is not an
observe annotation or mapping-only transaction; no camera motion is requested.
For legacy tabletop pick/place, place_point:[x,y] on the destination may accompany
the grasp proposals. Do not invent geometry or refer to unseen candidate images.'''


def _response_reference_rules(obs):
    reference=(f'For this CURRENT INPUT, top-level frame_id must be {obs.frame_id} '
        f'and top-level camera_id must be {json.dumps(obs.camera_id)}. '
        'Every grounding must use these exact same frame_id and camera_id values, '
        'with source="vlm". These source IDs do not change when requesting another view. ')
    if obs.base_pose is not None:
        return reference+('This is a mounted-camera observation. ONLY observation_camera names '
            'the requested NEXT camera, chosen from available_cameras. OMIT azimuth, elevation, '
            'and distance entirely; do not send null or numeric placeholders. '
            'observation_motion may be none, turn_left, turn_right, or wrist.')
    return reference+('For tabletop observe, include explicit numeric azimuth, elevation, '
        'and distance fields; the reason text alone does not move the camera.')


def _mapping_context(obs,history):
    # A fresh advice transaction can follow a changed robot/payload pose. Do not
    # remove views merely because the same camera was used before that advice.
    index=next((i for i in range(len(history)-1,-1,-1)
                if isinstance(history[i],dict) and 'coverage_advice' in history[i]),None)
    advice=None if index is None else history[index]['coverage_advice']
    recent=[] if index is None else history[index+1:]
    stale_image=index is not None and history[index].get('scene_changed_since_observation',False)
    views=[] if stale_image else [list(map(float,obs.view))]
    views.extend(list(map(float,item['view'])) for item in recent
                 if isinstance(item,dict) and 'view' in item)
    def key(view):
        azimuth,elevation,distance=map(float,view)
        return (round(azimuth%360,3)%360,round(elevation,3),round(distance,4))
    seen={key(view) for view in views}
    offered=advice.get('views',[]) if isinstance(advice,dict) else []
    return {'purpose':'mapping_only','latest_coverage_advice':advice,
            'scene_changed_since_observation':stale_image,
            'views_in_current_transaction':views,
            'unvisited_advised_views':[view for view in offered if key(view) not in seen]}


def _held_verification_context(obs,history):
    # Only the latest explicit transaction can override mapping semantics.
    # An older held request must not become active again after another event.
    latest=history[-1] if history and isinstance(history[-1],dict) else {}
    verification=latest.get('held_verification');advice=latest.get('coverage_advice')
    if (not isinstance(verification,dict) or not isinstance(advice,dict)
            or verification.get('frame_id')!=obs.frame_id or verification.get('camera_id')!=obs.camera_id
            or verification.get('strict_association_verified') is not False
            or type(verification.get('remaining_real_observations')) is not int
            or verification['remaining_real_observations']!=1
            or type(verification.get('remaining_transfer_mapping_observations_after_this_view')) is not int
            or verification['remaining_transfer_mapping_observations_after_this_view']!=0
            or advice.get('purpose')!='held_object_verification'):
        return None
    current=_mapping_context(obs,history)
    return {'purpose':'held_object_verification','frame_id':obs.frame_id,'camera_id':obs.camera_id,
        'current_view':list(obs.view),'remaining_real_observations':1,
        'remaining_transfer_mapping_observations_after_this_view':0,
        'latest_coverage_advice':copy.deepcopy(advice),
        'different_advised_views':current['unvisited_advised_views'],
        'instructions':'Choose ONE DIFFERENT real camera view that clearly exposes the SAME held object '
            'and permitted part for strict annotation. This is held-material verification, not UNKNOWN-path '
            'coverage: no rejected path samples are supplied. Keep the robot, fixed attachment reference '
            'and original grasp ray unchanged. Do not re-ground or propose a new grasp. The latest advice '
            'is only a visibility hypothesis; it does not establish that the object is held or approve motion. '
            'An earlier pre-lift view may be revisited if different from the current view. This consumes '
            'the last reserved real observation and leaves no transfer mapping observation. After acquisition, '
            'the same object and part still need explicit semantic, RGB-D and measured-lift verification.'}


def _tabletop_camera_context(obs):
    """Project known workspace axes, without inspecting any object or depth."""
    if obs.base_pose is not None or obs.camera_id!='activeview':return None
    h,w=obs.rgb.shape[:2]
    if min(h,w)<2:return None
    reference=np.array([-.12,0.,.95]);step=.1
    try:
        uv,depth=project(np.vstack([reference,reference+step*np.eye(3)]),obs.K,obs.T_wc)
    except ValueError:return None
    if np.any(depth<=.01) or not np.isfinite(uv).all():return None
    scale=np.array([w-1,h-1])
    delta=(uv[1:]-uv[0])/scale
    return {'reference_kind':'known camera-orbit workspace center, not a target',
            'reference_world_m':reference.tolist(),
            'reference_normalized_uv':np.round(uv[0]/scale,6).tolist(),
            'axis_step_m':step,
            'world_axes_normalized_delta':{
                axis:np.round(vector,6).tolist() for axis,vector in zip(('+X','+Y','+Z'),delta)},
            'scope':'Local calibration directions only; not object geometry or free-space evidence.'}


class VLM:
    def __init__(self, directory: Path, replay=None):
        self.settings = VLMSettings.from_env(require_api_key=replay is None)
        self.directory = directory
        self.directory.mkdir(parents=True,exist_ok=True)
        self.calls = 0; self.seconds = 0.; self.replay = replay
        self.client = httpx.Client(timeout=self.settings.timeout_seconds)

    def decide(self, obs: Observation, instruction, history, force_grasp=False,force_observe=False,
               observe_purpose='mapping_only',derived_image_context=None,annotation_diagnostic=None,
               current_frame_reselection=None,grounding_refinement=None,grounding_refinement_feedback=None):
        if observe_purpose not in ('mapping_only','approach_reselection','annotation'):
            raise ValueError('unsupported observation purpose')
        if observe_purpose!='mapping_only' and (not force_observe or force_grasp or obs.base_pose is not None):
            raise ValueError(observe_purpose.replace('_',' ')+' requires a forced tabletop observation')
        if current_frame_reselection is not None and (not isinstance(current_frame_reselection,dict)
                or force_observe or force_grasp or obs.base_pose is not None or observe_purpose!='mapping_only'):
            raise ValueError('current-frame reselection requires an unforced tabletop decision and evidence dictionary')
        if grounding_refinement is not None:
            if (not isinstance(grounding_refinement,dict) or not force_grasp or force_observe
                    or obs.base_pose is not None or observe_purpose!='mapping_only'
                    or current_frame_reselection is not None or annotation_diagnostic is not None):
                raise ValueError('grounding refinement requires an exclusive forced tabletop grasp and crop provenance dictionary')
            if (not isinstance(derived_image_context,dict) or grounding_refinement!=derived_image_context
                    or grounding_refinement.get('source_frame_id')!=obs.frame_id
                    or grounding_refinement.get('source_camera_id')!=obs.camera_id
                    or grounding_refinement.get('crop_shape_hw')!=list(obs.depth.shape)
                    or tuple(obs.rgb.shape[:2])!=tuple(obs.depth.shape)
                    or grounding_refinement.get('new_observation') is not False):
                raise ValueError('grounding refinement requires matching current source frame, camera, image shape and derived crop provenance')
        if grounding_refinement_feedback is not None:
            feedback=grounding_refinement_feedback
            if (grounding_refinement is None or not isinstance(feedback,dict)
                    or feedback.get('frame_id')!=obs.frame_id or feedback.get('camera_id')!=obs.camera_id
                    or feedback.get('coordinate_space')!='supplied_crop_normalized_pixel_centers'
                    or feedback.get('image_shape_hw')!=list(obs.depth.shape)
                    or type(feedback.get('repair_attempt')) is not int or feedback['repair_attempt']!=1
                    or type(feedback.get('max_geometry_repairs')) is not int or feedback['max_geometry_repairs']!=1
                    or not isinstance(feedback.get('geometry_diagnostics'),dict)):
                raise ValueError('grounding refinement feedback requires one geometry repair in the current crop coordinate space')
            previous=AgentDecision.model_validate(feedback.get('previous_crop_decision'))
            if (previous.action!='grasp' or previous.frame_id!=obs.frame_id or previous.camera_id!=obs.camera_id
                    or any(g.frame_id!=obs.frame_id or g.camera_id!=obs.camera_id or g.source!='vlm'
                        for g in previous.groundings)):
                raise ValueError('grounding refinement feedback requires the previous same-frame crop grasp')
        if derived_image_context is not None and observe_purpose!='annotation' and grounding_refinement is None:
            raise ValueError('derived images are supported only for current-frame annotation or grounding refinement')
        if annotation_diagnostic is not None:
            if observe_purpose!='annotation':
                raise ValueError('pixel diagnostics are supported only for current-frame annotation')
            annotation_diagnostic.validate_source(obs)
        image = io.BytesIO(); Image.fromarray(obs.rgb).save(image,format="PNG")
        image_sha256=hashlib.sha256(obs.rgb.tobytes()).hexdigest()
        image_reference=f'../observations/frame_{obs.frame_id:03d}.png'
        if derived_image_context is not None:
            image_reference=f'input_frame_{obs.frame_id:03d}_{image_sha256[:16]}.png'
            (self.directory/image_reference).write_bytes(image.getvalue())
        diagnostic_metadata=None;diagnostic_reference=None;diagnostic_image=None
        if annotation_diagnostic is not None:
            diagnostic_metadata=annotation_diagnostic.provenance()
            diagnostic_image=io.BytesIO();Image.fromarray(annotation_diagnostic.rgb).save(diagnostic_image,format='PNG')
            diagnostic_metadata['diagnostic_png_sha256']=hashlib.sha256(diagnostic_image.getvalue()).hexdigest()
            diagnostic_metadata['primary_png_sha256']=hashlib.sha256(image.getvalue()).hexdigest()
            diagnostic_reference=f'annotation_diagnostic_{obs.frame_id:03d}_{diagnostic_metadata["diagnostic_rgb_sha256"][:16]}.png'
            (self.directory/diagnostic_reference).write_bytes(diagnostic_image.getvalue())
        user = {"instruction":instruction,"frame_id":obs.frame_id,"camera_id":obs.camera_id,"view":obs.view,
                "image_size":{"width":obs.rgb.shape[1],"height":obs.rgb.shape[0]},
                "history":history,"must_ground_now":force_grasp,"must_observe_now":force_observe}
        user['allowed_actions']=['observe','grasp','abort'] if obs.base_pose is None else ['observe','navigate','grasp','place','finish','abort']
        user['response_contract']=_response_reference_rules(obs)
        camera_context=_tabletop_camera_context(obs)
        if camera_context is not None:user['tabletop_camera_context']=camera_context
        if derived_image_context is not None:
            user['derived_image_context']=copy.deepcopy(derived_image_context)
        if diagnostic_metadata is not None:
            user['annotation_diagnostic']=diagnostic_metadata
        if grounding_refinement is not None:
            user['allowed_actions']=['grasp','abort']
            user['response_contract']+=(' This is a before-ray grounding refinement. Return grasp with fresh explicit '
                'object AND grasp_part regions, or abort. Every coordinate is normalized to this supplied crop image_size, '
                'not the original full image. Do not return observe or a camera command.')
            user['grounding_refinement_context']={
                'purpose':'before_ray_same_frame_grasp_refinement','frame_id':obs.frame_id,
                'camera_id':obs.camera_id,'acquires_new_observation':False,
                'derived_image_context':copy.deepcopy(grounding_refinement),
                'instructions':'This crop contains pixels from the CURRENT already-acquired image. '
                    'No grasp ray has been accepted from the earlier coarse proposal. Reinspect this crop and return '
                    'a fresh grasp with explicit object AND grasp_part box, polygon or mask regions, or abort if uncertain. '
                    'For a mug, prefer ONE tight object BOX around the complete visible mug body AND handle together. '
                    'The object region must include the chosen handle material, not only the cup body; '
                    'do not copy the small grasp_part region as the object. A box needs no aperture hole. '
                    'Use a polygon or mask only when you can trace its complete visible extent reliably; '
                    'any polygon hole must lie inside its outer boundary and must not remove the chosen handle material. '
                    'Provide a SEPARATE tight visible grasp_part region inside the object region: '
                    'the permitted grasp_part must isolate the straight outer vertical handle bar around its midpoint, '
                    'excluding both junctions. Inspect both ends of the visible straight bar before choosing its middle; '
                    'do not copy an upper-only patch or earlier pixel coordinates. Prefer regions without an optional contact click. '
                    'If the requested part or its extent cannot be identified inside this crop, abort. '
                    'ALL output coordinates use this crop image_size and its normalized pixel centers. The source frame/camera '
                    'IDs stay unchanged. The caller will restore coordinates to the full acquired image and run ordinary '
                    'depth, grounding, ray, IK and collision checks. Cropping grants no visibility, contact or safety evidence. '
                    'Ignore older mapping-only preserve-ray instructions: this is before-ray semantic refinement, '
                    'not an observe annotation and not a request for another image.'}
            if grounding_refinement_feedback is not None:
                user['grounding_refinement_feedback']=copy.deepcopy(grounding_refinement_feedback)
                user['grounding_refinement_context']['instructions']+=(' The previous crop reply failed the actual '
                    'RGB-D grounding resolver. This is its ONE geometry correction in the SAME supplied crop. '
                    'The feedback and previous_crop_decision use this crop coordinate space. Reinspect the image and '
                    'return fresh complete object and separate part regions, or abort. Check that the whole-object '
                    'region includes the chosen visible handle material. Geometry diagnostics identify rejected pixels '
                    'or constraints, not correct semantic labels or suggested replacement contact points. '
                    'Do not mechanically union, expand or clip regions, or move a contact to a diagnostic pixel; '
                    'correct the visual annotation only when the current image supports it. All ordinary checks still apply.')
        if current_frame_reselection is not None:
            user['current_frame_reselection_context']={
                'purpose':'fresh_grounding_on_acquired_frame','frame_id':obs.frame_id,
                'camera_id':obs.camera_id,'acquires_new_observation':False,
                'evidence':copy.deepcopy(current_frame_reselection),
                'instructions':'The previous grasp transaction has ended after bounded geometry/IK search failed. '
                    'This CURRENT image was actually acquired from a different view during mapping. '
                    'If the requested part and clearance are visible, propose a NEW grasp with fresh explicit object '
                    'AND grasp_part regions in this image; the planner will derive a NEW camera-to-contact ray. '
                    'Old preserve-ray/mapping-only instructions and the initial-grounding deadline do not apply '
                    'to this new transaction. Do not reuse old coordinates, candidate IDs, or target masks. '
                    'If this image is unsuitable, choose observe or abort. Revisiting an earlier view is allowed; '
                    'do not reject a view just because it was already acquired for mapping. Prefer supplied '
                    'robot-only kinematic view advice when selecting an observation, while checking the requested '
                    'part in the next actual image: advice does not prove visibility, free space or path safety. '
                    'Observation reserves and all safety checks remain in force.'}
        if force_observe and obs.base_pose is None:
            if observe_purpose=='mapping_only':
                held_context=_held_verification_context(obs,history)
                if held_context is not None:
                    user['held_verification_context']=held_context
                    user['response_contract']+=(' This is held-object verification: return observe to choose ONE '
                        'different real view of the same held object and permitted part, not an UNKNOWN corridor '
                        'or a new grasp. Only one reserved observation remains.')
                else:user['mapping_context']=_mapping_context(obs,history)
            elif observe_purpose=='approach_reselection':
                evidence=next((item['approach_reselection'] for item in reversed(history)
                    if isinstance(item,dict) and isinstance(item.get('approach_reselection'),dict)),None)
                if evidence is not None and (evidence.get('frame_id')!=obs.frame_id
                        or evidence.get('camera_id')!=obs.camera_id):
                    evidence=None
                user['approach_reselection_context']={
                    'purpose':'approach_reselection','current_view':list(obs.view),
                    'reason':None if evidence is None else evidence.get('reason'),
                    'evidence':copy.deepcopy(evidence),
                    'instructions':'The previous grasp transaction is ending after bounded geometry or kinematics search. '
                        'Use the current evidence.reason to distinguish nominal observed obstacles from exhausted IK seeds; '
                        'neither establishes global infeasibility or absence of every valid pregrasp pose. '
                        'Select a different semantic view that exposes the same requested graspable part from a reachable direction. '
                        'This ends the failed grasp transaction: the next acquired image requires fresh grounding and a NEW ray. '
                        'Do not reuse old image coordinates or treat this as mapping-only recovery. '
                        'Revisiting an earlier view is allowed; the mapping-only preference for unvisited views does not apply. '
                        'Prioritize supplied robot-only kinematic view advice, which does not certify target visibility, '
                        'free space or path safety. '
                        'This request overrides an earlier initial-grounding deadline; remaining observation and post-close reserves still apply.'}
            else:
                user['annotation_context']={
                    'purpose':'current_frame_annotation','acquires_new_observation':False,
                    'instructions':'Annotate the requested object and permitted part in this supplied image. '
                        'Return observe with the CURRENT view fields; no camera command from this reply will execute. '
                        'Ignore earlier coverage-view advice and grounding deadlines for this annotation. '
                        'All normalized coordinates refer to this image_size, including when it is a crop. '
                        'A projected crop does not establish visibility or identity. Omit uncertain regions.'}
                if diagnostic_metadata is not None:
                    user['annotation_context']['instructions']+=(
                        ' The FIRST image is the unmarked current image and the only output coordinate system. '
                        'The SECOND image is a magnified coordinate aid from the same pixels; its grid and magenta pixel borders '
                        'are geometry diagnostics, not object labels or a suggested mask. Both images have exactly the same '
                        'field of view and normalized pixel-center coordinates: a location has the same normalized x,y in both. '
                        'Raw-pixel labels for rejected pixels are listed in annotation_diagnostic metadata. '
                        'Visually reinspect material and boundaries in the first image. '
                        'Do not blindly delete marked pixels. Return normalized FIRST-image coordinates x/(width-1), y/(height-1); '
                        'raw pixel indices always refer to the FIRST image_size, never to the magnified image or grid cells.')
        primary_url='data:image/png;base64,'+base64.b64encode(image.getvalue()).decode()
        image_references={primary_url:image_reference}
        messages = [{"role":"system","content":SYSTEM}, {"role":"user","content":[
            {"type":"text","text":json.dumps(user)},
            {"type":"image_url","image_url":{"url":primary_url}}]}]
        if diagnostic_image is not None:
            diagnostic_url='data:image/png;base64,'+base64.b64encode(diagnostic_image.getvalue()).decode()
            messages[-1]['content'].append({'type':'image_url','image_url':{'url':diagnostic_url}})
            image_references[diagnostic_url]=diagnostic_reference
        last_error = None
        for repair in range(3):
            # A rejected answer and its correction are added to the *next*
            # request. Keep this call's audit log identical to what was sent.
            request_messages=copy.deepcopy(messages)
            payload = {"model":self.settings.model,"messages":request_messages,**self.settings.request_options()}
            index = self.calls; self.calls += 1
            start = time.monotonic()
            response = None
            connection = self.settings.public_metadata()
            logged_model = payload['model']
            try:
                if self.replay:
                    saved=json.loads((Path(self.replay)/f"call_{index:03d}.json").read_text())
                    connection = saved['request'].get('connection')
                    logged_model = saved['request'].get('model', 'unknown')
                    expected=saved["request"].get("image_sha256")
                    if expected and expected!=image_sha256:
                        raise RuntimeError("response replay image mismatch; use the original recorded episode for state replay")
                    if not expected:
                        source_image=Path(self.replay).parent/'observations'/f'frame_{obs.frame_id:03d}.png'
                        if source_image.exists():
                            original=Image.open(source_image).convert('RGB').tobytes()
                            if hashlib.sha256(original).hexdigest()!=image_sha256:
                                raise RuntimeError("legacy response replay image mismatch")
                    expected_diagnostic=saved['request'].get('annotation_diagnostic_sha256')
                    actual_diagnostic=None if diagnostic_metadata is None else diagnostic_metadata['diagnostic_rgb_sha256']
                    if expected_diagnostic!=actual_diagnostic:
                        raise RuntimeError('response replay annotation diagnostic image mismatch')
                    response = saved["response"]
                else:
                    r = self.client.post(self.settings.base_url+"/chat/completions",
                        headers={"Authorization":"Bearer "+self.settings.api_key},json=payload)
                    if r.is_error:
                        try: response = r.json()
                        except ValueError: response = None
                    r.raise_for_status(); response = r.json()
                if not isinstance(response,dict) or not isinstance(response.get('choices'),list) or not response['choices']:
                    raise ValueError('VLM response has no choices')
                choice = response['choices'][0]
                if not isinstance(choice,dict) or not isinstance(choice.get('message'),dict):
                    raise ValueError('VLM response has no assistant message')
                if choice.get('finish_reason') not in (None,'stop'):
                    raise ValueError('VLM response did not complete: '+str(choice['finish_reason']))
                content = choice['message'].get('content')
                if not isinstance(content,str) or not content.strip():
                    raise ValueError('VLM response has no final text content')
                text = content.strip()
                if text.startswith("```"):
                    if '\n' not in text or not text.endswith('```'):
                        raise ValueError('VLM response has an incomplete JSON code fence')
                    text = text.split("\n",1)[1].rsplit("```",1)[0]
                decision = AgentDecision.model_validate_json(text)
                if decision.frame_id != obs.frame_id: raise ValueError("frame_id is stale")
                if decision.camera_id != obs.camera_id: raise ValueError("camera_id is stale or incorrect")
                if decision.action not in user['allowed_actions']:raise ValueError('action is not supported in the current backend')
                if decision.action=='observe' and obs.base_pose is None:
                    if not {'azimuth','elevation','distance'}<=decision.model_fields_set:
                        raise ValueError('tabletop observe requires explicit azimuth, elevation and distance JSON fields; the reason text does not move the camera')
                if any(g.source!='vlm' for g in decision.groundings):raise ValueError('your proposals must have source=vlm; only actual tool execution can claim sam/rgbd')
                if current_frame_reselection is not None and decision.action=='grasp':
                    roles={g.role for g in decision.groundings if g.kind!='candidate'}
                    if not {'object','grasp_part'}<=roles:
                        raise ValueError('current-frame reselection requires fresh explicit object AND grasp_part groundings; old points or candidate IDs are insufficient')
                if grounding_refinement is not None and decision.action=='grasp':
                    roles={g.role for g in decision.groundings if g.kind in ('box','polygon','mask')}
                    if not {'object','grasp_part'}<=roles:
                        raise ValueError('grounding refinement requires fresh explicit object AND grasp_part regions; legacy points or candidate IDs are insufficient')
                if force_grasp and decision.action == "observe": raise ValueError("this decision requires grasp or abort; another observation is not allowed at this step")
                if force_observe and decision.action != "observe":
                    purpose=observe_purpose.replace('_',' ')
                    detail='annotate the supplied image' if observe_purpose=='annotation' else 'collect additional geometry'
                    if 'held_verification_context' in user:
                        purpose='held object verification';detail='verify the same held object and permitted part'
                    raise ValueError(f'this {purpose} step requires an observe action to {detail}')
                last_error=None
                return decision
            except (ValueError, KeyError, httpx.HTTPError) as exc:
                last_error = self.settings.redact(str(exc))
                if isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code in (401,403):
                    raise RuntimeError('VLM authentication failed; check the API key and model access') from None
                if isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code in (400,404):
                    detail = response.get('error') if isinstance(response,dict) else None
                    detail = detail.get('message') if isinstance(detail,dict) else None
                    detail = self.settings.redact(detail)[:500] if isinstance(detail,str) else 'check model, endpoint and request options'
                    raise RuntimeError(f'VLM request rejected (HTTP {exc.response.status_code}): {detail}') from None
                if isinstance(exc,httpx.HTTPError):
                    # A service error is not a model formatting mistake. Keep
                    # the current conversation and respect bounded retry waits.
                    if repair < 2 and not self.replay:
                        delay = 2. ** repair
                        if isinstance(exc,httpx.HTTPStatusError):
                            try: delay = float(exc.response.headers.get('Retry-After', delay))
                            except ValueError: pass
                        if not math.isfinite(delay): delay = 2. ** repair
                        time.sleep(max(0.,min(delay,30.)))
                else:
                    # Include the rejected answer so the model can repair its
                    # actual fields, instead of guessing from an error alone.
                    choices = response.get('choices') if isinstance(response,dict) else None
                    previous = choices[0].get('message') if isinstance(choices,list) and choices and isinstance(choices[0],dict) else None
                    if isinstance(previous,dict) and isinstance(previous.get('content'),str):
                        messages.append({'role':'assistant','content':self.settings.redact(previous['content'])})
                    messages.append({"role":"user","content":"Correct your JSON response: "+last_error+
                        '\n'+user['response_contract']})
            finally:
                elapsed = time.monotonic()-start; self.seconds += elapsed
                # Credentials are never recorded. Image is stored separately in observations.
                logged=copy.deepcopy(request_messages)
                for message in logged:
                    if isinstance(message.get("content"),list):
                        for item in message["content"]:
                            if item.get("type")=="image_url":
                                item["image_url"]["url"]=image_references[item['image_url']['url']]
                request_log = {"model":logged_model,"connection":connection,
                    "transport":"replay" if self.replay else "online",
                    "context":user,"messages":logged,"repair":repair,
                    "image_sha256":image_sha256}
                if diagnostic_metadata is not None:
                    request_log['annotation_diagnostic_sha256']=diagnostic_metadata['diagnostic_rgb_sha256']
                log_text=json.dumps({"request":request_log,
                    "response":response,"seconds":elapsed,"error":last_error},ensure_ascii=False,indent=2)
                (self.directory/f"call_{index:03d}.json").write_text(self.settings.redact(log_text))
        raise RuntimeError("VLM failure: "+str(last_error))
