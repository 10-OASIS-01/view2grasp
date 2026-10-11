"""Small image/JSON API check that never imports or runs a robot simulator."""
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

from .types import Observation
from .vlm import VLM


PROBE_INSTRUCTION = (
    "This is a vision API grounding check, not a robot execution. "
    "Locate the solid RED square in the current image, ignoring other colors. "
    "Return a grasp proposal for that square with object and grasp_part boxes, "
    "and optionally one grasp_contact point on its solid red surface. "
    "Use the same target_id for all proposals. Use normalized image coordinates "
    "and the supplied frame_id and camera_id. Do not request another view or "
    "claim robot-task success."
)


def _probe_image():
    image = Image.new("RGB", (320, 240), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 118, 115, 197), fill=(225, 25, 35))
    draw.rectangle((210, 24, 285, 99), fill=(25, 165, 70))
    draw.rectangle((204, 143, 268, 207), fill=(35, 85, 220))
    draw.ellipse((39, 22, 100, 83), fill=(240, 190, 30))
    rgb = np.array(image)
    red = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] < 60) & (rgb[:, :, 2] < 70)
    return image, rgb, red


def _validate_grounding(decision, red):
    if decision.action != "grasp":
        return {"valid": False, "reason": "Expected a grasp proposal, received " + decision.action}

    # Explicit contacts take precedence: a correct object box must not hide a
    # contact on the background or another colored shape.
    contacts = []
    if decision.point is not None:
        contacts.append(("point", decision.point))
    for role in ("grasp_contact", "grasp_part", "object"):
        proposals = [grounding for grounding in decision.groundings if grounding.role == role]
        if any(grounding.kind not in ("points", "box") for grounding in proposals):
            return {"valid": False, "reason": "This probe requires points or boxes for " + role}
        for grounding in proposals:
            if grounding.kind == "points":
                contacts.extend((role + "_point", point) for point in grounding.points)
            elif grounding.kind == "box":
                left, top, right, bottom = grounding.box
                contacts.append((role + "_box_center", ((left + right) / 2, (top + bottom) / 2)))
        if contacts:
            break
    if not contacts:
        return {"valid": False, "reason": "No supported contact point or object/part box was returned"}

    height, width = red.shape
    checks = []
    for source, point in contacts:
        pixel = np.rint(np.asarray(point) * [width - 1, height - 1]).astype(int)
        inside = bool(red[pixel[1], pixel[0]])
        checks.append({"source": source, "point": list(point), "pixel": pixel.tolist(), "inside_red": inside})
    valid = all(check["inside_red"] for check in checks)
    return {"valid": valid, "reason": "Grounding lies on red material" if valid else
            "At least one selected contact or region center is outside the red square", "checks": checks}


def run_vlm_probe(directory: Path) -> dict:
    """Call the configured VLM on a synthetic image and persist probe evidence.

    The return value's success describes only this API check. No robot episode,
    geometry validation, physical grasp, or benchmark result is produced.
    """
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty probe output: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    observations = directory / "observations"
    observations.mkdir()
    image, rgb, red = _probe_image()
    image.save(observations / "frame_001.png")
    # Placeholders satisfy the shared transport record only. They are never
    # interpreted as sensor evidence and never passed to a geometry/planning API.
    obs = Observation(
        frame_id=1, rgb=rgb, depth=np.ones(red.shape), K=np.eye(3), T_wc=np.eye(4),
        q=np.zeros(7), gripper_q=np.zeros(2), timestamp=0.0, view=(0.0, 80.0, 1.3),
        camera_id="api_probe",
    )
    report = {
        "kind": "vlm_api_probe", "robot_episode": False, "success": False,
        "scope": "Image transport, structured response, and red-square localization only; no robot execution or benchmark scoring.",
        "image": "observations/frame_001.png", "calls": 0, "inference_seconds": 0.0,
        "response_models": [], "decision": None, "validation": None, "error": None,
    }
    vlm = None
    try:
        vlm = VLM(directory / "vlm")
        decision = vlm.decide(obs, PROBE_INSTRUCTION, [], force_grasp=True)
        report["decision"] = decision.model_dump(mode="json")
        report["validation"] = _validate_grounding(decision, red)
        report["success"] = report["validation"]["valid"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if vlm is not None:
            report["calls"] = vlm.calls
            report["inference_seconds"] = vlm.seconds
            vlm.client.close()
        models = []
        for path in sorted((directory / "vlm").glob("call_*.json")):
            response = json.loads(path.read_text()).get("response")
            if isinstance(response, dict) and isinstance(response.get("model"), str) and response["model"]:
                models.append(response["model"])
        report["response_models"] = list(dict.fromkeys(models))
        report["response_model"] = models[-1] if models else None
        (directory / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
