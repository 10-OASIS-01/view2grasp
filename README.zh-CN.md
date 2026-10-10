# View2Grasp

[![观看 View2Grasp 系统演示](artifacts/demo-v1/view2grasp_explained_1080p_previews/00_intro.jpg)](artifacts/demo-v1/view2grasp_explained_1080p.mp4)

**[▶ 观看系统演示 — 4 分 51 秒 · 1080p · 中英字幕](artifacts/demo-v1/view2grasp_explained_1080p.mp4)**

包含系统原理讲解与三个桌面任务，在线运行、响应回放和旧版结果均有标注。[视频说明与素材来源](docs/demo-v1.md)。

通过语言和多形式视觉 grounding 驱动主动观察、抓取、导航与移动操作。项目代码使用 MIT 许可；仿真资产与模型权重保留各自许可。

## 真实进度

当前桌面记录：方块新版在线成功；遮挡取放在录制 VLM 响应回放下成功；杯柄新版完成定位和闭爪，抬升被未知空间检查阻止。三个任务的旧版在线成功记录也全部保留，但旧版地图含未知空间推断，不能作为新版安全检查通过的证明。用户已取消大规模评测，改为约 10–15 次针对性验证。已先制作包含原理讲解的[第一版演示](../view2grasp/artifacts/demo-v1/view2grasp_explained_1080p.mp4)，见[视频与素材说明](docs/demo-v1.md)。

Grounding 支持点、框、多边形/mask 和图像路径。SAM 是可选的精细分割工具，VLM 输出不是直接控制指令。深度、机器人模型、完整路径及持物碰撞检查决定能否执行；未知空间不可通行。

目前已实际验证四类表示的本地 VLM 请求、SAM 2.1 CPU 分割、独立 RoboCasa 环境、厨房 RGB-D 渲染、机器人模型 FK 与底盘控制。一次导航开发运行完成了 26.76 cm 的无碰撞底盘运动；咖啡任务曾安全后退 32.84 cm。两次都未完成任务，不能算成功；失败均保留。工程回归测试属于离线验证，不计为在线机器人评测。

## 安装与运行

```sh
uv sync --locked
export VLM_BASE_URL=http://127.0.0.1:8318/v1
export VLM_API_KEY=local
export VLM_MODEL=gt-6-as-a
uv run view2grasp doctor
uv run pytest -q
uv run view2grasp run --task block --seed 3 --root runs/development
```

启用已验证的可选 SAM：

```sh
SAM2_BUILD_CUDA=0 uv sync --locked --extra perception
uv run --extra perception python scripts/download_sam.py
uv run --extra perception view2grasp run --task block --seed 3 --root runs/development-sam
```

SAM 不可用时，`auto` 明确记录原因并回退 RGB-D；`sam` 则直接报错，不冒充调用成功。包和权重 SHA256 均已固定。地面图像路径用于导航搜索引导；悬空末端路径仅用于候选排序，不套用背景深度。

桌面运行环境固定为 Python 3.11、robosuite 1.5.2、MuJoCo 3.3.7。RoboCasa 使用独立环境，不能覆盖桌面依赖，见 [安装记录](docs/robocasa.md)。

保留方块、杯柄、遮挡取放三个任务；后续接入 `NavigateKitchen` 和简化移动操作 `DeliverCanToBowl`。`DeliverBrewedCoffee` 暂缓，保留为进阶任务。程序化杯柄刻意放大以适配 Panda 整只夹爪，不等同于普通家用杯子。

固定、随机和主动策略共享感知工具与规划器，但目前不扫种子、不穷举策略。小规模验证默认五任务 × 开发种子 3/4/5 × 主动策略，最多 15 次。先复用失败日志离线排错，有实质修复再在线重跑。失败全部保留，离线回放不计入在线结果；不作统计优势结论。

`evaluate --dry-run` 只打印 15 次小规模清单，不执行；命令拒绝超过 15 次的批量请求。可用 `validation_seeds` 再减少运行次数。旧版正式种子字段仅为兼容保留，不再被此命令使用。厨房布局 2 的记录明确标为开发数据。

日志保存原始 RGB-D、标定、模型请求/响应、grounding、候选、轨迹及独立评估数据。第一版桌面演示已加入系统原理动画；旧版成功、新版响应回放和新版未完成阶段分别标注，五任务最终演示待后续补齐。完整命令说明见 [English README](README.md)。

已修复旧记录的场景 XML 导出错误：新版本在编译机器人规划模型前保存完整场景，随后校验回放的状态维数。部分旧记录只能看原始执行录像，不能据此声称完整状态回放已通过。新记录 `runs/strict-sam-v8/block_active_0003` 的状态回放已实测通过，但该次任务本身失败。第三方许可见 [说明](THIRD_PARTY_NOTICES.md)。
