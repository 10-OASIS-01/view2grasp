# Architecture and explicit limitations

## Trust boundary

`sim.py` constructs the scene and emits immutable observation records (array
contents are copied). `vlm.py`, `geometry.py`, `mapping.py` and `planning.py`
cannot import the simulator adapter or evaluator. `RobotModel` compiles only
the assembled Panda model exported by robosuite; it contains no task objects,
table or scene collision geometry. Its renderer produces a known-robot depth
image solely for self filtering. This is not an object segmentation oracle.

`evaluation.py` and the offline recorder are privileged consumers. Success is
not used to select views or poses. All controller decisions must come from
observations/proprioception. Runtime `env.reward()` is zero.

## Geometry and mapping

Pixel coordinates are normalized [0,1] on integer pixel centers, with
`u=x*(W-1), v=y*(H-1)`. MuJoCo normalized depth is converted to metric axial
depth. Camera transforms map OpenCV camera coordinates to world coordinates.
MSAA is disabled: its depth resolves at subpixel samples on macOS and otherwise
causes a detectable bias. Known-plane validation supplements round-trip tests.

The map voxelizes occupied depth endpoints at 5 mm and evaluates free space
projectively from stored calibrated depth images. A point behind all available
depth returns is unknown. Occupied evidence overrides free evidence. Known
robot pixels have a 2 mm rendering tolerance; object depth does not. The
initial robot's known mesh volumes certify its own occupied starting volume,
not its surrounding workspace. Visual and collision meshes both belong to
the known robot model. This assumes a valid nonintersecting initial robot.

Robot collision geometry is sampled volumetrically at approximately 10 mm,
with observed obstacles inflated by 6.5 mm. Joint path interpolation is at
most 0.025 rad per joint. These are finite-resolution checks, not continuous
collision guarantees. Self collision is checked in the robot-only MuJoCo model.
Unknown space is blocked even if the task visually appears easy. The mobile
base has one narrow exception for the self-occluded layer immediately outside
its current silhouette: the same sampled boundary may move at most 3 cm from
the last validated proprioceptive pose, with 5 mm edge interpolation. Observed
occupied returns and unknown floor still block, the certificate is refreshed
after every short segment, and arm planning never uses it.

## Planning and execution

The candidate approach axis stays within 15 degrees of the camera-to-resolved-contact
ray. Roll has 24 initial samples, insertion depths are 8/16/24 mm, pregrasp is
80 mm away. Candidate width is estimated from observed local geometry. IK
uses a damped least-squares Jacobian solver; RRT-Connect searches joint space.
Physical closing uses Panda's binary gripper actuator, not direct width control.

Transit and straight approach paths are saved before execution. Lift and
transfer are subsequently planned. Historical successes do not certify the
current tightened map/contact implementation. Tabletop payloads remain sampled
observed surfaces; the experimental mobile branch uses a filled observed AABB.
Neither uses an oracle mesh, and neither solves occluded-shape completion.
This difference must be resolved before claiming robust full-envelope validation.

Known outstanding work includes coverage planning around self occlusion,
robust map updates for moved objects, conservative full payload geometry,
closed-loop payload tracking and end-to-end success validation. The code is a research
prototype and should not be used on physical hardware without further work.

## Reproducibility

The VLM is an external service and need not produce identical responses on
repeated calls. `uv.lock`, code hashes, random seeds, RGB-D and raw responses
provide auditability. Offline response replay is labeled and excluded from
online performance. Simulator snapshots are privileged offline artifacts.

Metrics requiring scene truth are stored separately. Fixed and random methods
use the same grounding, candidates, map and planner. Active selection currently
receives heuristic guidance to acquire opposing side views; this prior is
part of the published prompt and is not a learned view policy.

## Multiform proposal boundary

`GroundingProposal` preserves representation, semantic role, frame, camera,
target association and source. Region-only grasps need no point at the VLM
interface: a contact is derived only at the geometry boundary. Polygon holes
and row-major RLE masks are preserved; SAM is an optional tool with explicit
fallback reporting. Raw model proposals and depth-filtered surface masks are
stored separately. Tracking uses local calibrated geometry and optional SAM,
not task-specific RGB colors or simulator instance labels.

Only verified floor-depth paths are lifted into world-space navigation hints.
An image end-effector path only weakly ranks existing grasp candidates by
projected direction; it samples no background depth and creates no trajectory.
Semantic masks/paths never label unknown cells free.

## Mobile and reduced validation scope

RoboCasa lives behind a separate interpreter and dependency lock. Mounted
cameras, full sampled base/arm proxies, physical-velocity adaptation, SE(2)
search, arm planning, and independent native scoring are implemented, but
end-to-end navigation and mobile manipulation have not passed acceptance.
Refer to `robocasa.md` for exact sensor and scene-filter definitions.

At the user's request, the former 300-trial protocol is no longer the work
target. Small development validation is capped at 15 episodes, with existing
recordings used for offline diagnosis first. Final-video production is paused.
