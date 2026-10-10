# Isolated RoboCasa runtime

RoboCasa is based on robosuite, but its current dependency set differs from the tabletop lock. Use `runtimes/robocasa` as an independent uv project. Do not activate or modify historical environments in other projects.

Target tasks: `NavigateKitchen` and the custom `DeliverCanToBowl`; `DeliverBrewedCoffee` is deferred as a stretch task. Robot: PandaOmron; sensors: mounted RGB-D cameras, calibrated robot pose and proprioception. Ground-truth fixture placement and task success belong only to the trusted scene generator/evaluator. Follow the [handoff roadmap](../doc/HANDOFF.zh-CN.md) for current priorities; coffee records below are historical debugging evidence.

Official installation: https://robocasa.ai/docs/build/html/introduction/installation.html

## Reproduce the isolated installation

```sh
uv run python scripts/setup_robocasa.py
uv run python scripts/download_robocasa_assets.py
uv run view2grasp doctor --backend robocasa --output artifacts/robocasa-doctor.json
uv run view2grasp run --task NavigateKitchen --seed 3 --root runs/mobile-development
uv run view2grasp run --task DeliverCanToBowl --seed 3 --root runs/can-development
```

`sources.json` pins RoboCasa `456174f62b89b8fca99eaaf33949c29fec9cfc2a` and robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. The isolated lock resolves RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1, and NumPy 2.2.5. These differ from the tabletop wheel/lock despite the same robosuite version label.

The original Box downloads failed. The downloader uses pinned official RoboCasa/NVIDIA Hugging Face snapshots: 90 archives, 5,440,027,461 compressed bytes, approximately 11 GB extracted. Each archive's LFS SHA256 is verified before safe extraction. `asset_sources.json` fixes revisions; `asset_manifest.json` records each source/hash. Archives/vendor checkouts are ignored by Git and are not relicensed MIT.

Measured: native kitchen rendering passes, reprojection round-trip error <1e-12 pixels, robot-only FK error <1e-8 m, and finite volumetric collision samples. A seed-3 512px privileged audit exposed a robot-only near-plane mismatch: the wrist camera initially retained 6,801 robot-labeled samples. Setting the robot renderer's absolute near plane to 5 mm reduced that to one sampled edge pixel and removed occupied samples at the valid current pose. Labels are diagnostic only. NumPy's pinned Apple runtime emits spurious matmul floating-point warnings although finite proxy/FK checks pass; broad numeric robustness remains an acceptance item.

The native control action layout is resolved by named parts, never guessed indices. `MobileVelocityAdapter` uses the known robot Jacobian and actuator friction/gain to convert limited physical velocity requests to the pinned controller. JOINT_POSITION remains in RoboCasa's reset-safe delta mode; planned absolute targets are converted to normalized deltas limited to 0.1 rad per control update. A RoboCasa smoke retained the official reset pose within `2.7e-9 rad`, moved 7.85 mm, and held the arm within 0.00039 rad. This is not a kitchen-navigation success.

## Cameras and observation policy

Native left/right manipulation and wrist RGB-D cameras are supplemented by a documented fixed robot-mounted rig: four lower and four high outward/down cameras, one downward near-floor camera, and one top-down camera. Extrinsics are in `mobile_cameras.py`, attached to `mobilebase0_support`; cameras are never teleported in world coordinates. Switching physical sensors does not move the base. Requested base turns and wrist motions must be planned before execution. The VLM receives one acquired image at a time, never unvisited candidate renders. All methods share this rig and low-level safety code.

Navigation uses a 2 cm SE(2) lattice, 5 mm edge interpolation, and full sampled robot/carrying geometry. It executes at most about 2.5 cm per local segment, then acquires direction-specific low/high, near-floor, and overhead safety depth before refreshing the map. Ordinary unknown space remains blocked. A bounded 3 cm self-shadow certificate admits only the same robot boundary samples displaced from the last validated proprioceptive pose; observed occupied endpoints and unknown floor still reject. This local bootstrap is an explicit limitation of onboard depth and is never used for arm planning. Arm manipulation uses the 5 mm local map and joint planner.

## Scene eligibility and outstanding work

The first mobile implementation stages base and arm motion; no SLAM or liquid-spill simulation is claimed. Both manipulation task definitions require real base transport, with source and destination separated by at least 1 m and a fixed-base reach exclusion; filtering is logged independently of policy execution. `DeliverCanToBowl` transfers a rigid can into a large bowl and has passed environment rendering/FK checks, but no complete online task run is established. Its native receptacle/release condition is combined with two-second stability and at least 0.5 m of two-finger-contact transport in the evaluator.

Coffee eligibility also requires the entire official 30 cm breakfast-plate neighborhood to lie outside a conservative fixed-base reach bound: sum of arm-chain translations plus 30 cm gripper/object allowance. This triangle-inequality bound ignores restrictive joint limits. Seed 3/layout 2 passed: fixture separation 3.053 m; closest destination 3.076 m; reach bound 1.383 m. The policy sees only task language, never these coordinates. At least 0.5 m two-finger-contact transport is independently scored, in addition to official success and two-second stability.

Current layout/style are developmental constants (2/1), not a held-out benchmark. `runs/mobile-v8-targeted/NavigateKitchen_active_0003` used live VLM point grounding and moved 26.76 cm with zero collision before accumulated occupied-sweep evidence blocked the next segment. `runs/mobile-v10-coffee-advisor/DeliverBrewedCoffee_active_0003` used calibration-only mounted-camera advice, recognized that static cameras could not see the remaining arm corridor, and retreated 32.84 cm in 15 safe segments; it still did not grasp the cup. No end-to-end mobile success is claimed. Coffee grasp/carry/place remains unvalidated. The native official success and collision-free success metrics are separate. The user cancelled the 300-trial benchmark in favor of about 10–15 practical validation runs total.
