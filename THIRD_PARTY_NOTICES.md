# Third-party components

Only View2Grasp-authored code is covered by this repository's MIT license. Dependencies, downloaded assets, and checkpoints retain upstream licenses and attribution requirements. Do not redistribute the ignored vendor/download/checkpoint directories as MIT assets.

| Component | Source / pin | License / attribution |
| --- | --- | --- |
| robosuite | https://github.com/ARISE-Initiative/robosuite ; tabletop 1.5.2, mobile commit in `runtimes/robocasa/sources.json` | MIT, Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab |
| RoboCasa | https://github.com/robocasa/robocasa ; commit in `sources.json` | MIT, RoboCasa Team; preserve the pinned checkout's LICENSE |
| MuJoCo | https://github.com/google-deepmind/mujoco ; 3.3.7 tabletop / 3.3.1 mobile | Apache-2.0 |
| SAM 2.1 code and tiny checkpoint | https://github.com/facebookresearch/sam2 ; code/weights pinned in `configs/model_weights.json` | Apache-2.0; optional connected-components third-party notice in `LICENSE_cctorch` also applies if used |
| RoboCasa assets | https://huggingface.co/datasets/robocasa/robocasa-assets | Dataset metadata: CC-BY-4.0; preserve contained per-asset notices and Objaverse provenance |
| Lightwheel kitchen assets | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF | Dataset metadata: CC-BY-4.0; retain NVIDIA/Lightwheel and per-asset attribution |
| RoboPoint | https://robo-point.github.io/ | Research/interface inspiration only; no RoboPoint code or weights bundled |

The two uv locks enumerate all transitive Python packages; this table highlights major components, not a legal audit of every dependency. Model weights and kitchen archives are fetched separately with fixed revisions and checksums. Review upstream notices before redistributing selected third-party assets. The local VLM service is user-provided; no service credentials or model weights are bundled.
