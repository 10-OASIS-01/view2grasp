# M1 工程交接状态

日期：2026-10-12。根据用户最新要求，M1 按“基本可交接”范围收尾。**主要工程修复和验证已交付，原交接文档要求的三个任务全部在线成功验收尚未达成：杯柄完整抬升与保持仍是已知未完成项。**

## 可以交接的结果

| 任务 | 已验证结果 | 代表记录 |
| --- | --- | --- |
| 积木抓取 | 智谱 GLM 在线成功、无记录碰撞；抬升约 12.97 cm，独立成功条件持续约 3.3 s；232 状态实际回放通过 | `runs/handoff-block-05/block_active_0003`，源码 `7f5428c` |
| 遮挡取放 | 在线完成抓取、抬升、保持、搬运和释放；独立评分与 408 状态实际回放通过 | `runs/handoff-occluded-14/occluded_active_0003`，源码 `1dd2c72` |
| 杯柄抓取 | 已实现在线定位、接近、闭爪和补拍恢复；已有双指柄部接触且无禁止碰撞的状态记录，完整抬升/保持尚未通过 | `runs/handoff-mug-31/mug_active_0004`，源码 `d2eec7d`；219 状态实际回放通过，任务结果仍为失败 |

三项是分别保存源码的开发记录，不能合并为成功率或策略优越性结论。接手机器的这些运行使用 RGB-D 回退，不能称为使用了 SAM。

## 已完成的工程修复

- 抓前、闭爪后与重试共享观测预算，杯柄可使用明确标注的 12 张开发配置。
- 视角恢复按已知机器人遮挡和已观测表面选择；来源姿态恢复可在总预算内使用两个不同视角。
- 标注加入同像素放大辅助图、实际拒绝位置反馈，以及初始补充部位视角的可见性检查。
- 固定初始多视角参考参与当前姿态配准；当前观测、预测碰撞包络和允许接触部位保持独立。
- 增加默认关闭的显式材料标注选项，要求新的模型 polygon/mask 标注，处理曲面深度碎片；不把旧检测框自动当作材料权限。
- 保存调用、源码、配置、拒绝原因、独立评分和执行状态，失败记录不被成功探针覆盖。

最后一轮 `handoff-mug-34` 使用 `685947f`，初始三项语义和实际闭爪完成后，按本次收尾范围正常中断；246 个保存状态独立核对及实际回放完成（12.3 s），未执行 lift/hold。这是部分执行记录，不计为完整验收。

最近实现版本：`685947f`。完整本机回归 **1176 项通过**，日志 `artifacts/handoff/regression-m1-current-material.log`；该结果验证工程行为，不等于杯柄任务成功。

## 后续使用

智谱官方 API 已接入，参见项目 README 的 GLM 配置说明。本机密钥继续保存在忽略提交的 `.env`。后续确实需要继续杯柄实验时，使用独立输出目录，例如：

```sh
MUJOCO_GL=cgl VLM_TIMEOUT_SECONDS=240 \
VLM_REASONING_EFFORT=high VLM_MAX_TOKENS=8192 \
uv run --no-sync --env-file .env view2grasp run \
  --task mug --seed 4 --strategy active \
  --config configs/mug-current-material.yaml --root runs/m1-next-mug
```

`MUJOCO_GL=cgl` 是本机 macOS 设置；其他平台需使用适合其渲染环境的后端。新配置是实验选项，默认配置不启用材料标注。模型需重新确认可见材料，任何非法种子或越界选区仍可能使一次运行停止。

## 保留的限制

杯柄剩余问题是模型标注稳定性，以及完整载物抬升路径上的未知空间/占据校验。局部配准通过、首步障碍减少、测试通过或视频可回放都不代表完整任务通过。后续只有取得独立验证的抬升至少 8 cm、双指柄部接触并稳定保持 2 s 的在线记录，才能改为完整验收。

原始导师运行档案仍未取得，M0 暂缓。M2/M3 的已有修改保留，本次不继续其在线开发。当前分支包含此前工作，不应描述成只有 M1 改动的分支。

## 证据入口

- [详细开发记录与失败索引](handoff-local-validation.zh-CN.md)
- [结果与限制](results.md)
- [原始交接说明](HANDOFF.zh-CN.md)（保留原验收标准）
- 积木独立验收：`artifacts/handoff/block-05-acceptance-audit.json`
- 遮挡独立验收：`artifacts/handoff/occluded-14-acceptance-audit.json`
- 最后一轮收尾记录：`artifacts/handoff/mug-34-closeout.json`、`artifacts/handoff/mug-34-acceptance/acceptance-audit.json`、`artifacts/handoff/mug-34-replay-report.json`
- 杯柄部分执行与回放：`artifacts/handoff/mug-31-acceptance/acceptance-audit.json`、`artifacts/handoff/mug-31-replay-report.json`

详细 `runs/` 与诊断产物保存在本机、被 Git 忽略；这里的路径是本机证据索引，不表示大文件已经推送到 GitHub。
