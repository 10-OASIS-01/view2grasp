"""Explicit, serializable experiment settings and reproducibility fingerprints."""
import hashlib
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal
import yaml


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # Six remains the frozen default. Targeted perception development may use
    # up to four extra closed-loop views for handled / holed geometry. The
    # explicitly named development config records this larger evidence budget;
    # comparisons must use the same budget and UNKNOWN remains unchanged.
    max_observations: int = Field(default=6, ge=2, le=10)
    # Opt-in occluded pick/place reserve: held-object verification followed by
    # at most one observation of an actually rejected transfer UNKNOWN region.
    # Zero preserves the previous tabletop observation allocation.
    post_lift_observations: int = Field(default=0, ge=0, le=2)
    # Mug-only same-frame crop refinement experiment. Disabled preserves the
    # original VLM call sequence and replay protocol.
    refine_grasp_grounding: bool = False
    # Mug development: independently annotate acquired object surfaces and
    # register historical ownership before using it as a carried envelope.
    source_object_ownership: bool = False
    # Opt-in mug initialization: establish explicit object material from the
    # acquired overview and side view before freezing reassociation references.
    initial_multiview_reference: bool = False
    initial_multiview_part: bool = False
    initial_part_views: int = Field(default=1,ge=1,le=2)
    # Mug development: share the existing search budget across candidate poses
    # before repeating redundant-IK seeds for any one pose.
    diversify_grasp_attempts: bool = False
    max_attempts: int = Field(default=2, ge=1, le=2)
    image_size: int = Field(default=512, ge=128, le=1024)
    voxel_resolution_m: float = Field(default=.005, ge=.002, le=.01)
    planning_seconds: float = Field(default=120, gt=0, le=600)
    max_candidates: int = Field(default=48, ge=1, le=256)
    test_seeds: tuple[int, ...] = tuple(range(100, 120))
    validation_seeds: tuple[int, ...] = (3,4,5)
    perception_backend: Literal['rgbd','auto','sam'] = 'auto'
    sam_checkpoint: str = 'checkpoints/sam2.1_hiera_tiny.pt'
    navigation_observations: int = Field(default=30,ge=1,le=30)
    mobile_observations: int = Field(default=60,ge=1,le=60)
    navigation_sim_seconds: float = Field(default=180,gt=0,le=180)
    mobile_sim_seconds: float = Field(default=360,gt=0,le=360)

    @model_validator(mode='after')
    def validate_post_lift_reserve(self):
        if self.max_observations - self.post_lift_observations < 2:
            raise ValueError('post-lift reserve requires an initial and a post-close observation')
        return self

    @model_validator(mode='after')
    def validate_initial_reference(self):
        if self.initial_multiview_reference and not self.source_object_ownership:
            raise ValueError('initial multiview reference requires source object ownership')
        if self.initial_multiview_part and not self.initial_multiview_reference:
            raise ValueError('initial multiview part requires initial multiview object reference')
        if self.initial_part_views!=1 and not self.initial_multiview_part:
            raise ValueError('additional initial part views require explicit part initialization')
        return self


def load_config(path=None):
    return ExperimentConfig() if path is None else ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def fingerprint(config):
    digest = hashlib.sha256()
    source = Path(__file__).parent
    for path in sorted(source.glob("*.py")):
        digest.update(path.name.encode() + path.read_bytes())
    lock = source.parent.parent / "uv.lock"
    if lock.exists(): digest.update(lock.read_bytes())
    for extra in ('runtimes/robocasa/uv.lock','runtimes/robocasa/sources.json','runtimes/robocasa/asset_sources.json','configs/model_weights.json'):
        path=source.parent.parent/extra
        if path.exists():digest.update(path.read_bytes())
    if config.perception_backend!='rgbd':
        checkpoint=Path(config.sam_checkpoint)
        if not checkpoint.is_absolute():checkpoint=source.parent.parent/checkpoint
        if checkpoint.exists():
            with checkpoint.open('rb') as stream:digest.update(hashlib.file_digest(stream,'sha256').digest())
        else:digest.update(b'checkpoint-unavailable')
    digest.update(config.model_dump_json().encode())
    return digest.hexdigest()
