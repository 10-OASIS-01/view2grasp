# 第一版系统演示 / First system demo

[观看 1080p 演示](../artifacts/demo-v1/view2grasp_explained_1080p.mp4) · [双语字幕](../artifacts/demo-v1/view2grasp_explained_1080p.srt) · [素材与时间轴](../artifacts/demo-v1/view2grasp_explained_1080p.json)

这是带中英文字幕的阶段演示，无配音。先解释系统原理，再展示三个任务的实际记录和新版杯柄的安全停止。完整成功复验留待后续。视频不报告未完成的策略对照实验，也不把旧版或响应回放算作新版在线成功。

## 原理演示

- 主动选视角：新版杯柄实际第 1、2 帧对比，先决策再采集。
- Grounding：同一杯柄的框、带孔多边形、接触点，以及 SAM + RGB-D 表面验证。
- 反投影：实测像素、米制深度、内参和相机位姿，逐步算出三维表面点。
- 抓取姿态：846 个记录候选中的采样动画，显示 15° 射线锥、滚转、开口宽度、表面点、指间中心及 TCP 的区别。候选不是执行轨迹。
- 地图反馈：自由、占据、未知三类空间的教学示意图，明确标注为非实测地图；旁边展示真实杯柄日志中的残余未知空间拒绝。
- 轨迹执行：根据遮挡取放保存的关节路径和实际关节状态做离线 FK；以不同颜色显示各阶段规划路径，以白色轨迹显示实际运动。背景为 RGB-D 点云，不显示仿真物体真值。

## 来源与结果

| 素材 | 来源 | 结果及范围 |
|---|---|---|
| 方块 | `targeted-block-v9-online/block_active_0003` | 新版在线成功 |
| 遮挡取放 | `targeted-occluded-v4-surface-sweep-replay/occluded_active_0003` | 新版响应回放成功；规划与执行重新运行 |
| 杯柄成功 | `final-online/mug_active_0003` | 旧版在线成功；不代表新版安全检查通过 |
| 杯柄进展 | `targeted-mug-v7-semantic-broadside-online/mug_active_0003` | 新版在线定位、接近、闭爪完成；抬升被未知空间阻止 |

每段保留源码快照哈希、执行视频 SHA256、结果及时间轴，详见 JSON。所有观测在执行前作为回顾展示，不声称这是原始事件的交错时间轴。执行片段保留原速和全部源视频帧；右侧 grounding 是静态参考图，不伪装为实时同步相机。原始执行视频为 640×480，观测为 512×512，1080p 指成片画布，未添加合成细节。

## 复现

从项目目录运行；需保留以下本地 episode 原始文件：

```sh
uv run view2grasp render-video \
  runs/targeted-block-v9-online/block_active_0003 \
  runs/targeted-occluded-v4-surface-sweep-replay/occluded_active_0003 \
  runs/final-online/mug_active_0003 \
  runs/targeted-mug-v7-semantic-broadside-online/mug_active_0003 \
  --annotations configs/demo-v1.json \
  --output artifacts/demo-v1/view2grasp_explained_1080p.mp4
```

`episode.mp4` 保存观测回顾、原速动作和运行结果；`execution.mp4` 仅含完整原速执行；`trajectory.npz` 是状态与控制记录。历史 XML 不保证兼容状态重放，原始执行录像保留。`artifacts/view2grasp_demo_1080p.mp4` 和不带 `explained` 的同目录初稿均为先前草稿，以本页链接为当前演示。

The bilingual demo explains the method using measured RGB-D and recorded robot motion. Occupancy illustrations are explicitly schematic. It includes current online success, response replay, historical success, and an unsuccessful current mug lift, with their provenance kept separate. No benchmark superiority or current-build mug success is claimed.
