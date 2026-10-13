# View2Grasp 项目交接：当前进度、后续实施与验收

更新时间：2026-10-10。代码基线：[`58d4e10`](https://github.com/10-OASIS-01/view2grasp/commit/58d4e10c6419702f17761ce704b86e1927c162bf)。仓库：[10-OASIS-01/view2grasp](https://github.com/10-OASIS-01/view2grasp)。本文面向接手实现的同学；“建议”与“已验证结果”分别注明。

2026-10-11 接手机器的新修复、GLM 在线运行和验收状态见 [本机验证记录](handoff-local-validation.zh-CN.md)。下文保留原始交接快照；导师原始 `runs/` 尚未取得，本机新增记录与历史证据分别注明。

先看 [4 分 51 秒系统演示](../artifacts/demo-v1/view2grasp_explained_1080p.mp4)，再看本文。视频包含主动观察、grounding、深度反投影、抓取姿态、未知空间和运动规划原理，也保留了未完成的运行。视频中的成功不代表五个任务已经全部完成。

## 1. 接手后要完成什么

把现有研究原型完成为一个可安装、可复现的视觉机器人闭环项目：自然语言任务 → 根据图像选择观察动作 → 多形式 grounding → RGB-D 几何验证 → 规划 → 执行 → 再观察与验证。

当前优先目标为五个任务，每个至少有一个使用真实在线 VLM、完整完成任务、满足无碰撞判据且可回放的记录：

1. `block`：Panda 抓起红色方块。
2. `occluded`：Panda 将蓝色方块放入黄色区域，保持干扰物稳定。
3. `mug`：Panda 通过杯柄抓取并抬升杯子。
4. `NavigateKitchen`：PandaOmron 导航到目标设施附近并正确朝向。
5. `DeliverCanToBowl`：PandaOmron 从工作台抓起罐子，移动到餐台，放进大碗后撤离夹爪。

已确认的范围调整：

- `DeliverBrewedCoffee` 暂缓，保留为进阶任务；先用罐子完成移动操作闭环。
- 暂不加入 SLAM，使用机器人自身定位和标定；不新增液体动力学、实机部署或复杂装配。
- 不做 180/300 次评测；采用少量针对性复跑。约十几次是工作规模意图，不是已有日志的数量，也不是必须再跑满 15 次的要求。
- 第一版视频已完成且放在中英文 README 顶部；不再把视频作为阻塞开发的前置条件。五任务完成后再更新最终演示。
- 公共 GitHub 仓库已创建并推送，接手同学在此仓库继续协作即可。

## 2. 当前进度：哪些完成，哪些还没有

| 项目 | 核实的状态 | 距离完成还差什么 |
|---|---|---|
| 方块 | 新版抓取实现有在线成功：8 次观测、1 次抓取、无记录碰撞 | 在接手环境复现；验证完整状态回放 |
| 遮挡取放 | 新版完整抓取、抬升、搬运、下放已在 VLM 响应回放中成功：8 次观测、1 次抓取、无记录碰撞 | 补一次真实在线 VLM 完整成功，不能把响应回放计入在线结果 |
| 杯柄 | 旧版有在线成功；新版已完成语义选视角、柄部定位、接近和闭爪，但抬升被安全检查拒绝 | 解决抬升路径剩余未知空间，补完整在线成功 |
| 导航 | 环境、相机、地图、短段规划与控制已接通；一次在线记录行驶 0.2676 m 后中止，无记录碰撞 | 解决后续扫掠碰撞/覆盖问题，真正到达目标并正确朝向 |
| 简单移动操作 | `DeliverCanToBowl` 已定义、注册；独立环境渲染、标定、机器人 FK 已检查 | 尚未找到此任务的完整 `result.json` 运行记录；取物、持物移动、放置全链路仍待调通 |
| 多形式 grounding | 点、框、多边形/带孔区域、mask、图像路径的接口和可视化已实现；本地 VLM 表示能力做过实际请求验证 | 表示支持不等于定位准确；仍需跨帧关联、失效恢复和任务级验证 |
| SAM | SAM 2.1 tiny CPU 实际验证过；不可用时支持明确记录的 RGB-D 回退 | 新机器重装权重，核实本次运行到底用了哪个后端 |
| 工程测试 | 基线推送前 82 项通过 | 这是工程测试，不是 82 次在线任务成功；修改后做相应回归 |
| 交付 | MIT、双语 README、两套锁文件、第三方说明、日志接口、第一版视频、公开仓库已具备 | 更新过时文档、补齐可复现资料和最终五任务视频 |

旧版三个桌面任务确实都存在在线成功记录，但旧版使用了推断未知空间等较宽松规则；这些记录只能说明当时版本的行为。新版成功必须保留自己的源码快照和配置，不能换个标签复用旧结论。当前没有足够证据宣称主动视角优于固定/随机视角。

## 3. 关键运行记录与证据

以下路径均相对项目根目录。`runs/` 未提交到 GitHub，需要从导师本机单独交接。先读 `result.json` 与 `events.json`，再看图像、规划诊断和轨迹。

| 用途 | 本地 episode 路径 | 应关注的证据 |
|---|---|---|
| 新版方块在线成功 | `runs/targeted-block-v9-online/block_active_0003` | `online=true`、`success=true`、`collision=false`；`path_*_lift.npy`、post-grasp 观测 |
| 新版遮挡回放成功 | `runs/targeted-occluded-v4-surface-sweep-replay/occluded_active_0003` | `online=false`、`success=true`；完整 lift/transfer/lower 路径 |
| 新版杯柄主要失败样本 | `runs/targeted-mug-v7-semantic-broadside-online/mug_active_0003` | `lift_planning_diagnostic.json`：22 个 payload unknown、0 个 occupied；`events.json`、第 2 帧 grounding |
| 导航主要失败样本 | `runs/mobile-v8-targeted/NavigateKitchen_active_0003` | `path_length_m=0.2676164497`，后续 occupied-sweep 拒绝；高层与 safety 观测分开保存 |
| 旧版杯柄成功/视频素材 | `runs/final-online/mug_active_0003` | 旧版成功，仅作历史参考；保留原始执行视频 |
| 旧版其他桌面成功 | `runs/final-online/block_active_0003`；`runs/final-online-v2/occluded_active_0003` | 不作为新版严格检查的通过证明 |
| 挂载相机建议与退让参考 | `runs/mobile-v10-coffee-advisor/DeliverBrewedCoffee_active_0003` | 0.3284 m 安全后退，但没有抓取；只复用通用观察经验，不恢复咖啡开发优先级 |

`artifacts/doctor_deliver_can.json` 记录了简单移动任务的环境检查：渲染成功，重投影误差约 `1.42e-13 px`，机器人 FK 误差约 `3.84e-13 m`。这些数值不证明任务成功，也不单独证明自遮挡处理完美。

记录结构：

- `observations/`：RGB、深度、K、T_wc、机器人状态、相机和 frame ID；`decision_*.json` 是模型动作。
- `vlm/call_*.json`：请求与响应，含实际返回的模型信息；回放按图像哈希匹配。
- `grounding_*_validation.json`、`*_masks.npz`：语义提议与几何过滤结果。
- `candidates_*.npz`、`path_*.npy`、`navigation_path_*.npy`：候选及规划轨迹。
- `planning_diagnostic_*.json`、`lift_planning_diagnostic.json`：拒绝原因与具体样本。
- `trajectory.npz`、`scene.xml`：离线状态回放；旧版有部分 XML 不完整，不能保证回放兼容。
- `manifest.json`、`source_snapshot.zip`：配置和实现快照。`source_sha256` 是代码、锁文件、配置等的组合指纹，不是 Git commit SHA；修改视频模块也可能改变该指纹。
- `evaluation_truth.json`、`scene_eligibility_truth.json`：独立评估/场景筛选真值，只允许评估和离线审计读取。

## 4. 新同学第一步：拿到资料并复现环境

### 4.1 交接资料

GitHub 已包含代码、锁文件、安装脚本、文档、第一版视频、字幕及来源清单。下列内容故意未进 Git：

| 内容 | 本机位置 | 交接方式 |
|---|---|---|
| 原始实验 | `runs/`，约 1.3 GB | 至少复制上表关键 episode；建议保留全量失败历史 |
| SAM 权重 | `checkpoints/`，约 160 MB | 从带哈希的脚本重新下载，或交接现有权重并校验 |
| RoboCasa 环境与资产 | `runtimes/robocasa/` 本机约 18 GB | 源码/锁/清单在 Git；vendor、下载包和虚拟环境需重建或另行转交 |
| 诊断素材 | `artifacts/` 除发布的视频外 | 交接 doctor、mobile-depth-audit 等关键报告；GitHub 中不包含所有本机图片 |
| VLM 服务 | 导师机器的本地服务 | 为接手机器提供可访问的服务或在本机启动，不能默认 localhost 指向导师电脑 |

导师本机项目目录是 `outputs/view2grasp/`。不要复制 `.venv` 后假设它在另一台机器可用；通过锁文件重建。实验日志不要因为它们失败就删除。不要将个人服务密钥提交到 Git。

### 4.2 桌面环境

```sh
git clone https://github.com/10-OASIS-01/view2grasp.git
cd view2grasp
uv sync --locked

# 这是导师本机已用过的服务配置；在新机器上需确认地址确实可达。
export VLM_BASE_URL=http://127.0.0.1:8318/v1
export VLM_API_KEY=local
export VLM_MODEL=gt-6-as-a

uv run view2grasp doctor --output artifacts/handoff-doctor.json
uv run pytest -q
```

桌面版本为 Python 3.11、robosuite 1.5.2、MuJoCo 3.3.7；依赖以 `uv.lock` 为准。本地服务使用 `/chat/completions`，请求模型名是 `gt-6-as-a`；实际响应模型曾为 `gpt-6-astra`，应继续记录实际返回值而非写死。`doctor` 检查渲染与模型列表，不等于已经通过一次带图像的端到端 VLM 调用。

可选 SAM：

```sh
SAM2_BUILD_CUDA=0 uv sync --locked --extra perception
uv run --extra perception python scripts/download_sam.py
```

安装 extra 后用 `uv run --extra perception ...` 保持同一依赖集合。`auto` 会在 SAM 不可用时回退 RGB-D；复现时必须检查日志里的实际后端。macOS 交互查看器用 `mjpython`；现有视频来自离屏渲染。

### 4.3 RoboCasa 独立环境

```sh
uv run python scripts/setup_robocasa.py
uv run python scripts/download_robocasa_assets.py
uv run view2grasp doctor --backend robocasa --output artifacts/handoff-kitchen-doctor.json

# 检查自定义简单任务；这里显式使用隔离的解释器。
PYTHONPATH=src runtimes/robocasa/.venv/bin/python -m view2grasp.mobile_cli \
  doctor --task DeliverCanToBowl --seed 3 --render \
  --output artifacts/handoff-can-doctor.json
```

RoboCasa 锁定环境目前解析为 RoboCasa 1.0.1、robosuite 1.5.2、MuJoCo 3.3.1、NumPy 2.2.5。即使 robosuite 版本字符串相同，两套源码/依赖也不能互相覆盖。资产下载是约 5.44 GB 压缩包、约 11 GB 解压数据，需预留空间。主 CLI 会自动切换 RoboCasa 解释器。

## 5. 建议实施顺序与可交付里程碑

推荐顺序：环境复现 → 桌面证据补齐 → 导航闭环 → 简单移动操作闭环 → 少量回归与交付。每一阶段先检查旧日志，再针对改动跑 1–3 次；未修复原因前不要重复启动同样的失败。

### M0：可复现的基线

交付：安装说明、doctor 报告、测试结果、能打开的关键日志和视频。先重现方块基线，不修改地图容差来“先跑起来”。确认 SAM/VLM/相机/控制器版本和记录一致。

所有运行使用新 `--root`，程序会拒绝覆盖已有 episode。下面命令是手动按阶段使用的示例，不应一次全部执行。

```sh
uv run --extra perception view2grasp run --task block --seed 3 --strategy active \
  --config configs/tabletop-targeted.yaml --root runs/handoff-block-01
```

M0 完成标准：本机可渲染与控制；82 项基线测试在兼容环境通过或环境差异得到解释；拿到完整关键记录；方块复现结果有明确结论。

### M1：收尾三个桌面任务

**遮挡取放：优先补在线证据。** 已有新版回放成功，先按相同配置做一次在线运行。若失败，比较 grounding、视角和规划差异，不能直接把回放的坐标硬编码进在线策略。

```sh
uv run --extra perception view2grasp run --task occluded --seed 3 --strategy active \
  --config configs/tabletop-targeted.yaml --root runs/handoff-occluded-01
```

**杯柄：当前最明确的桌面阻塞。** 新版第 2 帧从约 azimuth `-165°`、elevation `15°` 的实际视角定位到杯柄外侧竖杆。mapping-only 恢复会保留这条抓取射线，最终已经执行 pregrasp/approach/close。总计 9 次观测，只剩 2 次闭爪后观测；抬升首段仍有 22 个持物未知点、0 个 occupied 点。因此当前不能声称完成提杯，也不能确定只多看一帧就一定成功。

建议同学做的第一处有针对性的修改：

1. 检查 `runner.py` 的 `reserve_after_close`、残余未知空间的 follow-up 分支和 `max_post_views`。初始分配虽为杯柄预留 3 帧，后续分支仍可能消耗该预算。
2. 优先修正整个状态机的预算预留逻辑，使需要时真正剩下 3 次 post-grasp 观测；增加针对这条分支的回归测试。
3. 如果 9 帧无法同时覆盖前后阶段，可作为明确标注的开发配置试验将上限增至 10；当前 `config.py` 上限仍是 9，不能只改 YAML。同步修改配置校验测试，所有比较策略遵守相同预算。
4. 用实际新增视角覆盖诊断中的未知点；检查持物关联与动态地图刷新，不扩大一般 UNKNOWN 豁免。
5. 做一次在线确认。第三帧是否足够是待验证假设；仍失败时继续定位具体遮挡来源。

```sh
# 当前 9 帧配置；如实施了预算修改，改用新命名的开发配置。
uv run --extra perception view2grasp run --task mug --seed 3 --strategy active \
  --config configs/mug-targeted.yaml --root runs/handoff-mug-01
```

关键入口：`runner.py`、`view_advisor.py`、`mapping.py`、`planning.py`，测试入口 `test_runner.py`、`test_map_planner.py`、`test_reporting.py`。`scripts/diagnose_lift.py` 可用来追溯深度证据，但运行前阅读参数与重建假设，不能把诊断结果当在线任务成功。

M1 完成标准：三个桌面任务各有一个新版在线、无碰撞、完整成功 episode；保存独立评分；对新记录实际执行状态回放。方块和杯柄均需抬升 ≥8 cm 并稳定 2 s，杯柄还需满足柄部双指接触；遮挡任务需正确放置、释放后稳定 2 s、干扰物未撞倒。

### M2：完成 `NavigateKitchen`

当前主要困难是机器人本体遮挡与多视角地图积累后的全身扫掠检查，尚不能归因于“VLM 不会导航”。导航已经能从图像地面点生成短段动作；失败在进一步推进时缺少可用空间证据或存在 occupied 拒绝。

建议顺序：

1. 离线定位 `mobile-v8-targeted` 首次阻止继续运动的扫掠样本，将它关联到实际 RGB-D 帧、已知机器人自深度、地面与柜体边界，分清真障碍、自遮挡残留、深度边缘还是旧地图残留。
2. 必要时增加每次拒绝的坐标、来源帧、状态类型和控制段日志。不要只留下一个 `occupied robot sweep` 字符串。
3. 检查挂载相机补盲、机器人状态刷新、地面支持检查和已占据证据的生命周期。修复应保留真实障碍，不能用清空地图来制造可走路线。
4. 保持“观测 → 短段导航 → 安全观测 → 重规划”闭环，处理停滞、绕行与朝向。每个高层动作最多推进约 25 cm，内部段约 2.5 cm；两者不要混淆。
5. 完成抵达和朝向后，由独立评估器读取任务原生成功；`finish` 只代表 agent 提议结束，不能直接记成功。

```sh
uv run view2grasp run --task NavigateKitchen --seed 3 --strategy active \
  --config configs/mobile-targeted.yaml --root runs/handoff-navigation-01
```

主要入口：`mobile_runner.py` 的 `acquire/nav_planner/drive`、`navigation.py`、`mobile_cameras.py`、`mobile_control.py`、`robocasa_backend.py`。测试重点：未知地面、全机器人扫掠、路径中间碰撞、控制器跟踪、相机策略隔离与停滞恢复。

M2 完成标准：一次在线完整导航成功；`official_success=true` 且 `collision_free_success=true`；日志能解释实际行驶路线与最终朝向，不能只展示局部移动。

### M3：完成 `DeliverCanToBowl`

采用简单刚体罐子和大碗，目的在于完成移动操作系统闭环。自定义任务位于 `robocasa_tasks.py`。场景过滤要求源与目的设施相隔至少 1 m，并通过保守臂展界证明初始固定底座无法完成目的地放置。判据在 `mobile_evaluation.py::scene_eligibility`，过滤真值不能传给 agent。

按以下顺序完成，而不是直接重复跑整段：

1. 在工作台前完成罐子 grounding、可达抓取、闭爪、抬升与基于观测的持物确认。
2. 持物导航复用 M2，碰撞几何包含持物包络，底盘运动时保持抓取与臂姿态。
3. 从图像识别餐台和大碗，在可达位置停靠；从观测估计碗内放置区域和高度，不能把碗沿/背景深度当作碗底。
4. 规划搬运、下放、松爪、撤离，再观测，稳定后由独立评估器评分。
5. 明确失败状态：抓空、掉落、抬升被拒绝、目的地不可达、放置失败；最多 2 次抓取，不得因内部标志错误而假定一直持物。

代码审阅发现、需要同学验证的具体风险：

- 当前 `mobile_runner.py` 在闭爪后即设置 `payload`；尚无完善的在线持物确认。`payload is not None` 是内部估计，不是抓取成功证据；抬升失败后还要能恢复状态。
- 抓取后的目标端点移除首先作用于局部 arm 地图；导航地图也需基于观测正确处理已移动的目标，避免旧物体仍被视作障碍。不能直接按仿真位姿删点。
- 桌面持物几何主要是已观测表面，移动分支为填充的已观测 AABB；均不等于完整未知形状。先让简单罐子得到有界、保守的包络，再考虑通用形状。
- 移动分支没有显式更新 `record.first_grasp_success`，目前默认 false 不足以表达真实首次抓取结果；应补独立计分或将未测项目表示为不可用。
- 放置后的姿态撤离和稳定验证仍需实测。`DeliverCanToBowl` 的原生成功还要求夹爪远离物体。

```sh
uv run view2grasp run --task DeliverCanToBowl --seed 3 --strategy active \
  --config configs/mobile-targeted.yaml --root runs/handoff-can-01
```

`mobile-targeted.yaml` 当前只给移动操作 12 次高层观测，是调试配置，不能默认足以完成全程；必要时定义有明确理由的新配置。配置上限为 60 次高层观测、360 s 仿真时间。底层 safety 观测单独统计，不能把“12 次高层观测”写成仅采集了 12 张图。

M3 完成标准：有效跨区域布局；真实在线抓取—持物移动—放置—释放—撤离全部完成；官方成功、无碰撞成功均为 true。额外独立判据要求双指接触期间累计移动 ≥0.5 m，以及放置成功稳定 2 s；保留 `scene_eligibility_truth.json` 和 `held_path_length_m`。

### M4：整理为完整可交付项目

- 在功能改动收敛后冻结配置和 commit。先用已知 seed=3 调通，再选择少量新种子做回归；失败全部进入报告。
- 先 `evaluate --dry-run` 检查清单。`evaluate` 默认最多 15 个开发 episode，不应因有该命令就立即跑满；桌面和移动通常需各自合适的配置。
- 报告任务成功、无碰撞成功、观察次数、safety 图像次数、VLM/分割调用、推理/规划耗时和分阶段失败。样本少时只做开发验证结论。
- 若补视角策略对比，固定/随机/主动必须共享感知工具和几何规则，仅改变高层观察选择；没有对照数据就不写主动优于基线。
- 更新双语 README、架构/结果文档、安装流程、第三方许可、关键实验记录索引。
- 当前视频生成器包含针对本次四条素材的讲解与结果文案，不能直接换 episode 就假设全部说明会自动更新。五任务视频要重审结果页、历史标签、素材来源及原理段。
- README 顶部保留最新视频入口；另附各任务完整录像和可复现命令。完整仿真回放与 VLM 响应回放分别标注。

## 6. 代码导航与不能破坏的边界

| 模块 | 职责 |
|---|---|
| `types.py` | Observation、GroundingProposal、AgentDecision、候选、记录接口 |
| `vlm.py` | 图像调用、结构校验、最多两次格式修复、请求响应记录 |
| `grounding.py` | SAM/RGB-D、区域与孔洞、部件表面、路径深度验证 |
| `geometry.py` | 反投影、支撑平面、目标点云、15° 候选 |
| `mapping.py` | 占据/自由/未知、自深度过滤、目标关联与动态刷新 |
| `planning.py` | 机器人模型、IK、全臂采样碰撞、RRT-Connect、持物检查 |
| `runner.py` / `mobile_runner.py` | 两套任务状态机、观察预算与动作执行 |
| `sim.py` / `robocasa_backend.py` | 仿真接口；不作为 agent 可读取真值的通道 |
| `evaluation.py` / `mobile_evaluation.py` | 独立真值评分与有效场景检查 |
| `recording.py` / `video.py` / `demo*.py` | 记录、状态回放、带解释的视频 |

必须保留的约束：

1. agent 和 planner 只使用 RGB-D、相机标定、机器人模型及本体状态，不读取物体/设施真值位置、实例标签、物体碰撞网格、奖励和成功标志。
2. grounding 的 frame、camera、target、role 保持关联；图像路径是意图，不能直接当作控制轨迹。悬空末端路径不能直接采样背景深度。
3. mug 孔洞不是可抓表面；SAM 多边形不能把 UNKNOWN 变成 FREE。
4. 15° 接近方向容差、全臂与持物碰撞、路径插值检查需保留。当前是有限分辨率采样，不可称为数学上的连续碰撞保证。
5. 当前安全实现有两项明确、受限的自遮挡处理，并非“完全无例外的未知空间阻挡”：移动底盘的 3 cm 局部自阴影证书，以及桌面固定姿态垂直抬升时基于密集观测表面的局部阴影规则。后一规则涉及 0.75 mm 源表面容差和有界的竖直覆盖。两者不得覆盖真实 occupied，也不能扩大成任意未知区通行许可；需要保持针对孔洞、侧向移动和障碍的回归测试。
6. 接触例外应只覆盖预期的指腹与已观测目标部位。不得为了让杯柄成功，把整个杯子或整只夹爪加入无条件忽略列表。
7. replay 必须记录 `online=false`；真实 VLM 调用也不自动代表机器人任务成功。以独立 evaluator 和实际执行记录为准。

## 7. 日常调试和向导师汇报

每个问题按“失败证据 → 原因假设 → 最小修改 → 对应测试 → 一次针对性在线确认”的顺序推进。不要同时改 prompt、几何容差、相机预算和控制器，否则很难解释为什么成功或失败。

建议每次汇报包含：

| 项目 | 填写内容 |
|---|---|
| 版本 | commit、配置名、episode ID、在线/回放、实际感知后端 |
| 结果 | 官方/任务成功、无碰撞成功；失败具体发生在哪一阶段 |
| 证据 | 一段执行视频、关键帧、拒绝坐标或日志位置 |
| 修改 | 哪个模块、什么原因、哪项测试覆盖 |
| 资源 | VLM 调用数、观测数、耗时；是否超出开发预算 |
| 下一步 | 仅列下一项最可能消除阻塞的工作和完成条件 |

状态回放示例（拿到原始记录后运行，输出另起文件名）：

```sh
uv run view2grasp replay runs/targeted-block-v9-online/block_active_0003 \
  --output artifacts/handoff-block-state-replay.mp4
```

响应回放使用 `run --replay <原 episode 的 vlm/ 目录>`；它会重新执行当前规划器，要求图像哈希匹配。配置或控制变化导致图像不同后应停止，不得跳过检查强行套用旧响应。状态回放只是重现保存的动作，不验证新策略。

## 8. “项目做完整”的最终检查表

- [ ] 新机器按文档安装成功；桌面与 RoboCasa 环境相互隔离。
- [ ] 五个任务各至少一个在线、完整成功、无碰撞的 episode；原始响应、RGB-D、配置、源码快照齐全。
- [ ] 每个验收 episode 都实际完成状态回放验证；历史不兼容的 XML 不充数。
- [ ] mug 通过柄部接触判据；简单移动操作通过真实持物移动和稳定放置判据。
- [ ] 未知空间、孔洞、全臂/持物碰撞、策略隔离、坏 JSON、无效深度、超时等回归仍通过；新增失败分支有测试。
- [ ] 原始失败仍保留；小规模结果诚实说明范围，不声称完成大样本评测。
- [ ] 确认数据交接可用：仅能 clone 代码、却拿不到关键原始记录，不算完成复现交付。
- [ ] 更新含系统原理、五任务实际执行和真实限制的视频；README 顶部链接正确。
- [ ] 文档、代码、必要公开素材推送 GitHub；运行环境、下载资产、权重和个人凭据不混进代码提交。

导师需要评估的核心是：同学能否把感知、观察选择、几何检查和执行组成可解释、可复现的闭环。当前最值得投入的是杯柄 post-grasp 覆盖、导航近场扫掠地图和简单持物状态机，而不是先扩大模型、引入 SLAM 或增加评测数量。
