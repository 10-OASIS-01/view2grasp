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
from PIL import Image
from .types import AgentDecision, Observation
from .vlm_settings import VLMSettings

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
The planner derives contacts from regions when no point is supplied. For a
handle expose both the material and aperture; use negative points/holes to
exclude background. Grasp approach stays within 15 degrees of the camera ray
to the resolved contact. Choose a view giving reachable approach and clearance.
Place: ground a support region large enough for the object and free of clutter.
Navigate: ground navigation_goal on visible FLOOR and look_at on the destination;
optionally propose a ground_path. Paths are planning hints, NOT certified free
space. An end_effector_path has no implied depth from the image background.
For tabletop observe, specify azimuth [-180,180], elevation [-15,85], distance
[.55,1.3]. Orbit center [-.12,0,.95]; azimuth 0 is world +X, 90 is +Y. The Panda
base is on -X. Low broadside views expose mug-handle apertures; overhead views
often suit blocks. Coverage advice uses only observed geometry and robot model.
For mounted cameras, observe selects observation_camera from available_cameras and optional
observation_motion: none, turn_left, turn_right, wrist. Motion requires planning;
do not request a virtual camera teleport. Groundings always refer to the camera
of the supplied image, even when observe requests a different camera separately.
If must_ground_now is true, grasp or abort. If must_observe_now is true, observe.
Unknown-space rejection means acquire relevant additional observations. Respect
remaining budgets. finish reports a visual hypothesis, not evaluator success.
For legacy tabletop pick/place, place_point:[x,y] on the destination may accompany
the grasp proposals. Do not invent geometry or refer to unseen candidate images.'''


class VLM:
    def __init__(self, directory: Path, replay=None):
        self.settings = VLMSettings.from_env(require_api_key=replay is None)
        self.directory = directory
        self.directory.mkdir(parents=True,exist_ok=True)
        self.calls = 0; self.seconds = 0.; self.replay = replay
        self.client = httpx.Client(timeout=self.settings.timeout_seconds)

    def decide(self, obs: Observation, instruction, history, force_grasp=False,force_observe=False):
        image = io.BytesIO(); Image.fromarray(obs.rgb).save(image,format="PNG")
        image_sha256=hashlib.sha256(obs.rgb.tobytes()).hexdigest()
        user = {"instruction":instruction,"frame_id":obs.frame_id,"camera_id":obs.camera_id,"view":obs.view,
                "image_size":{"width":obs.rgb.shape[1],"height":obs.rgb.shape[0]},
                "history":history,"must_ground_now":force_grasp,"must_observe_now":force_observe}
        user['allowed_actions']=['observe','grasp','abort'] if obs.base_pose is None else ['observe','navigate','grasp','place','finish','abort']
        messages = [{"role":"system","content":SYSTEM}, {"role":"user","content":[
            {"type":"text","text":json.dumps(user)},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(image.getvalue()).decode()}}]}]
        last_error = None
        for repair in range(3):
            payload = {"model":self.settings.model,"messages":messages,**self.settings.request_options()}
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
                if any(g.source!='vlm' for g in decision.groundings):raise ValueError('your proposals must have source=vlm; only actual tool execution can claim sam/rgbd')
                if force_grasp and decision.action == "observe": raise ValueError("observation budget exhausted: grasp or abort")
                if force_observe and decision.action != "observe": raise ValueError("post-action map refresh requires an observe action")
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
                    messages.append({"role":"user","content":"Correct your JSON response: "+last_error})
            finally:
                elapsed = time.monotonic()-start; self.seconds += elapsed
                # Credentials are never recorded. Image is stored separately in observations.
                logged=copy.deepcopy(messages)
                for message in logged:
                    if isinstance(message.get("content"),list):
                        for item in message["content"]:
                            if item.get("type")=="image_url":
                                item["image_url"]["url"]=f"../observations/frame_{obs.frame_id:03d}.png"
                request_log = {"model":logged_model,"connection":connection,
                    "transport":"replay" if self.replay else "online",
                    "context":user,"messages":logged,"repair":repair,
                    "image_sha256":image_sha256}
                log_text=json.dumps({"request":request_log,
                    "response":response,"seconds":elapsed,"error":last_error},ensure_ascii=False,indent=2)
                (self.directory/f"call_{index:03d}.json").write_text(self.settings.redact(log_text))
        raise RuntimeError("VLM failure: "+str(last_error))
