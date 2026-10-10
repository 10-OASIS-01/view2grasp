# Results and limitations

## Historical development episodes (not a benchmark)

| Task | Episode | Online | Legacy evaluator success |
| --- | --- | --- | --- |
| block | `runs/final-online/block_active_0003` | yes | yes |
| mug | `runs/final-online/mug_active_0003` | yes | yes |
| occluded | `runs/final-online-v2/occluded_active_0003` | yes | yes |

These are single-seed development results. Their source snapshots preserve the original implementation. It inferred some unknown cells as free using neighbor votes; post-grasp observation selection also needed policy isolation. The strict implementation removes these rules. Historical successes must not be relabeled as strict-run successes.

## Current scope (updated by user)

The 300-episode benchmark is cancelled. Target about 10–15 practical validation runs across the five tasks; do not sweep many seeds or observation policies. Prefer offline diagnosis of retained failures before more online calls. Final video remains paused. `evaluate` defaults to at most 15 development episodes and rejects larger batches.

## Measured engineering progress

- Actual local VLM accepted point/box/polygon/path requests: `runs/grounding-probe-v2/report.json`. These are live calls on recorded images, **not online robot episodes**, and prove schema support rather than localization accuracy.
- Real CPU SAM 2.1 tiny segmentation: `runs/grounding-sam-validation-v2/report.json`. Box and point prompts used SAM; direct polygons used RGB-D filtering. Checkpoint/code hashes are pinned.
- Isolated RoboCasa dependencies and 90 checked asset archives installed; native kitchen RGB-D render passes. See `artifacts/robocasa-doctor-v3.json` and the separate seed-3 512px self-depth audit under `artifacts/mobile-depth-audit/`.
- Fixed the PandaOmron arm-controller contract: RoboCasa reset must retain delta-input semantics, while absolute planned joint targets are converted to bounded normalized deltas. The official initial pose now survives reset/hold within `2.7e-9 rad`; the RoboCasa motion smoke moved 7.85 mm with 0.00039 rad arm hold error (`artifacts/robocasa-mobile-motion-smoke.json`).
- Fixed the robot-only self-depth near plane from 10.8 cm to 5 mm. Before the fix the wrist camera leaked 6,801 robot-labeled samples into the obstacle map; afterward only one sampled labeled edge pixel remained in the privileged audit and the valid current pose had zero occupied clearance samples.
- Added a bounded 3 cm local self-shadow certificate for mobile-base motion. It never waives observed occupied returns or unknown floor, uses 5 mm path interpolation, expires after every short segment, and is not used by arm planning.
- Offline payload tracing on `strict-sam-v8` found that a 2 mm moving-target association radius left all 3/3 rejected payload samples inside the map's 6.5 mm occupied footprint. Using the same 6.5 mm scale with 922 RGB-D tracked target points removed 3/3; the nearest remaining static endpoints were 3.94–4.40 cm away. The recorded-response replay then advanced from “carried object observed obstacle” to “carried object unknown space”, confirming the false occupied evidence was removed without claiming task success.
- Post-grasp view scoring now treats whole-arm visibility as a lexicographic requirement and maximizes payload lift-sweep coverage among equally safe views. The old recorded response still selects its historical 30° side view, so it cannot validate this new online choice.
- `targeted-block-v1` validated that priority online: the chosen post-close view predicted 100% coverage of the three exact rejected arm-path points. It reduced the refresh to two arm and 34 payload unknown samples but one view was insufficient, so all grasp tasks now reserve up to two post-close mapping frames; the episode remains a failure.
- Added calibrated mounted-camera recovery advice. It ranks real onboard cameras using only current intrinsics/extrinsics, the robot-only depth model, and already observed surfaces; it does not render candidate scene images. In `mobile-v10` the VLM followed the recommended right and overhead cameras, then safely commanded a base retreat once all static cameras were predicted unable to see the residual path.
- Unknown-space neighbor filling, batch-size-dependent classification, broad payload-obstacle waivers, stale dynamic-contact associations, and post-grasp active-policy leakage into baselines were removed. Failed perception time is now counted.
- Fixed new-recording scene XML capture; `runs/strict-sam-v8/block_active_0003/replay-check.mp4` was regenerated successfully from states. Its **task failed**, so successful replay does not imply task success. Old recordings with robot-only XML remain non-replayable; original execution videos are preserved.

## Representative online development outcomes (not pooled performance)

| Record | Outcome | Interpretation |
| --- | --- | --- |
| `strict-v2/block_active_0003` | success, no collision | Intermediate unknown-space fix, before later contact/source tightening; not current-code acceptance |
| `strict-sam-v4/mug_active_0003` | planning: unknown space | No grasp executed |
| `strict-sam-v7/occluded_active_0003` | carried-object unknown space | Grasp attempted; target side-face tracking fixed, lift not certified |
| `strict-sam-v8/block_active_0003` | carried-object observed obstacle | Grasp attempted, lift rejected; new complete scene/state replay verified |
| `mobile-v2/DeliverBrewedCoffee_active_0003` | planning budget / unknown space | Four online observations, cup/handle grounded, no grasp executed; eligible 3.053 m source/destination scene |
| `mobile-v3/NavigateKitchen_active_0003` | clearance rejection, then abort | 30 online decisions, no base movement; retained as failed development, not evidence of navigation |
| `mobile-v5/NavigateKitchen_active_0003` | unknown/occupied robot sweep, then abort | 15 decisions; fixed mounted rig exposes floor but full-body near-field map remains insufficient |
| `mobile-v6-controller-fix/NavigateKitchen_active_0003` | agent abort, no motion | Four-view smoke budget located no grounded approach; no collision |
| `mobile-v7-navigation/NavigateKitchen_active_0003` | 5.30 cm base motion, then abort | Live VLM point grounding drove three closed-loop 2.5 cm-prefix plans, zero collision. A wall-clock/search-budget coupling then rejected the next segment; fixed afterward. The agent later exhausted its eight-view smoke budget. Not task success |
| `mobile-v8-targeted/NavigateKitchen_active_0003` | 26.76 cm base motion, then abort | Live VLM searched mounted views, used an overhead aisle waypoint, and executed segmented motion without collision. Accumulated occupied sweep evidence beside the counter still blocked continuation. Not task success |
| `targeted-block-v1/block_active_0003` | post-grasp carried-object unknown space | Live SAM/VLM grounding, grasp close, and new required-point view selection executed; 100% predicted required coverage, but one post-close frame left 2 arm and 34 payload samples unknown. No collision; not success |
| `targeted-mug-v1/mug_active_0003` | pre-grasp unknown space / abort | VLM supplied box + holed polygon + negative hole point + solid-handle contact. Recovery reduced the closest path miss from 372 unknown samples to one, while other candidates had non-pad target interference. No grasp executed |
| `targeted-occluded-v2/occluded_active_0003` | residual recovery then new-ray unknown space | The exact residual view was acquired, exposing an orchestration bug: it re-grounded a new ray instead of retrying stored candidates. Recovery is now transactional; no success claimed yet |
| `mobile-v10-coffee-advisor/DeliverBrewedCoffee_active_0003` | 32.84 cm retreat, then abort | Mounted advisor was followed online; VLM detected static-camera unobservability, inspected rear floor, and retreated in 15 safe segments. Final closest grasp corridor still had 46 unknown samples; no grasp, transport, or collision |
| `replay-block-payload-radius/block_active_0003` | offline replay: carried-object unknown space | Not online and excluded from evaluation. Confirms the target-endpoint occupied false positive is gone; the old recorded post-grasp view still leaves seven payload sweep points unknown |

All versions, failures, raw requests, and observations are retained under `runs/`; the rows above are an index of debugging milestones, not a selectively filtered evaluation dataset.

## Remaining acceptance work

- Resolve tabletop moved-object map association, conservative payload envelope, and preplanned lift coverage; revalidate all three tasks.
- Resolve the remaining occupied-sweep rejection after navigation map accumulation; do not launch a seed sweep.
- Complete base-motion navigation and grasp–carry–place–retreat in an eligible kitchen scene. Current mobile code is experimental; first-grasp scoring, carried-object association, and actual successful transport still need validation.
- Preserve the same perception capabilities for every compared policy; freeze a small validation configuration once tasks work.
- Collect a genuine online successful and replayable episode for each task, within the reduced testing scope; organize final demos only after the video pause is lifted.

No statistical performance or superiority claim is supported yet.
