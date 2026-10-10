"""The entire agent boundary: no simulator handles, object poses, or IDs."""
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator, ConfigDict


@dataclass(frozen=True)
class Observation:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    q: np.ndarray
    gripper_q: np.ndarray
    timestamp: float
    view: tuple[float, float, float]
    camera_id: str = "activeview"
    base_pose: np.ndarray | None = None
    torso_q: np.ndarray | None = None


class GroundingProposal(BaseModel):
    """Image evidence, not a command or a certificate of free space.

    RLE is row-major, starts with a background run, and preserves holes.
    Pixels are normalized to [0,1]; mask_size is (height,width).
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["points", "box", "polygon", "mask", "path", "candidate"]
    role: Literal["object", "grasp_part", "grasp_contact", "place_region",
                  "navigation_goal", "ground_path", "end_effector_path", "look_at"]
    frame_id: int = Field(ge=0)
    camera_id: str = "activeview"
    target_id: str = "target"
    source: Literal["vlm", "sam", "rgbd", "legacy"] = "vlm"
    points: list[tuple[float, float]] = Field(default_factory=list)
    negative_points: list[tuple[float, float]] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None
    holes: list[list[tuple[float, float]]] = Field(default_factory=list)
    mask_size: tuple[int, int] | None = None
    mask_rle: list[int] | None = None
    candidate_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        coords = self.points + self.negative_points + [p for ring in self.holes for p in ring]
        values = [v for p in coords for v in p] + list(self.box or ())
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("grounding coordinates must be finite and normalized")
        if self.box and not (self.box[0] < self.box[2] and self.box[1] < self.box[3]):
            raise ValueError("invalid grounding box")
        required = {"points": 1, "polygon": 3, "path": 2}.get(self.kind, 0)
        if len(self.points) < required:
            raise ValueError(f"{self.kind} requires at least {required} points")
        if self.kind == "box" and self.box is None:
            raise ValueError("box grounding requires box")
        if self.holes and (self.kind != "polygon" or any(len(r) < 3 for r in self.holes)):
            raise ValueError("holes require polygon rings with at least 3 vertices")
        if self.kind == "mask":
            if self.mask_size is None or min(self.mask_size) < 1 or max(self.mask_size) > 4096:
                raise ValueError("invalid mask dimensions")
            if self.mask_rle is None or any(n < 0 for n in self.mask_rle):
                raise ValueError("invalid mask RLE")
            if sum(self.mask_rle) != self.mask_size[0] * self.mask_size[1]:
                raise ValueError("mask RLE does not match dimensions")
        if self.kind == "candidate" and not self.candidate_id:
            raise ValueError("candidate selection requires candidate_id")
        if self.kind == "path" and self.role not in ("ground_path", "end_effector_path"):
            raise ValueError("path must state ground_path or end_effector_path semantics")
        return self


class AgentDecision(BaseModel):
    action: Literal["observe", "navigate", "grasp", "place", "finish", "abort"]
    frame_id: int
    reason: str = ""
    target: str = ""
    part: str = ""
    point: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    part_bbox: tuple[float, float, float, float] | None = None
    place_point: tuple[float, float] | None = None
    groundings: list[GroundingProposal] = Field(default_factory=list)
    camera_id: str = "activeview"
    observation_motion: Literal["none", "turn_left", "turn_right", "wrist"] = "none"
    observation_camera: str | None = None
    azimuth: float = Field(default=0, ge=-180, le=180)
    elevation: float = Field(default=55, ge=-15, le=85)
    distance: float = Field(default=.8, ge=.55, le=1.3)

    @model_validator(mode="after")
    def valid_coordinates(self):
        if self.action == "grasp" and not self.groundings and any(v is None for v in (self.point, self.bbox, self.part_bbox)):
            raise ValueError("grasp requires point, bbox and part_bbox")
        for name in ("point", "bbox", "part_bbox", "place_point"):
            value = getattr(self, name)
            if value is not None and not all(np.isfinite(x) and 0 <= x <= 1 for x in value):
                raise ValueError(f"{name} must be normalized to [0,1]")
        for name in ("bbox", "part_bbox"):
            b = getattr(self, name)
            if b and not (b[0] < b[2] and b[1] < b[3]):
                raise ValueError(f"invalid {name}")
        if any(g.frame_id != self.frame_id or g.camera_id != self.camera_id for g in self.groundings):
            raise ValueError("grounding frame/camera differs from decision")
        if self.action=='grasp' and self.groundings:
            associated={g.target_id for g in self.groundings if g.role in ('object','grasp_part','grasp_contact')}
            if len(associated)!=1:
                raise ValueError('object, grasp_part and grasp_contact must share ONE target_id (the object identity, not separate part names)')
        if self.action == "grasp" and self.point is not None and self.part_bbox is not None:
            x, y = self.point
            b = self.part_bbox
            if not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
                raise ValueError("point outside part_bbox")
        return self


@dataclass
class GraspCandidate:
    position: np.ndarray
    rotation: np.ndarray
    width: float
    approach: np.ndarray
    score: float
    ray_angle: float
    surface_point: np.ndarray | None = None
    grasp_center: np.ndarray | None = None


@dataclass
class Trajectory:
    joints: np.ndarray
    gripper: float
    phase: str


@dataclass
class EpisodeRecord:
    task: str
    seed: int
    strategy: str
    online: bool = True
    success: bool = False
    first_grasp_success: bool = False
    observations: int = 0
    attempts: int = 0
    vlm_calls: int = 0
    inference_seconds: float = 0
    planning_seconds: float = 0
    elapsed_seconds: float = 0
    failure: str = ""
    collision: bool = False
    grounding_types: list[str] = field(default_factory=list)
    segmentation_calls: int = 0
    perception_seconds: float = 0.
    official_success: bool | None = None
    collision_free_success: bool | None = None
    path_length_m: float | None = None
    held_path_length_m: float | None = None
    geometry_rejections: list[str] = field(default_factory=list)
