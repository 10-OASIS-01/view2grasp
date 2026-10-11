# GLM 视觉 API 接入

GLM 预设复用项目现有的 OpenAI-compatible `/chat/completions` 接口，仅配置模型连接与请求参数。视觉提议仍经过原有 schema、RGB-D、地图和规划检查；更换模型不改变机器人执行判据。

## 本地配置

安装桌面依赖后，首次创建本地配置：

```sh
uv sync --locked
cp -n .env.glm.example .env
chmod 600 .env
```

用本地编辑器打开 `.env`，填写 `ZHIPU_API_KEY`。不要把真实密钥直接写进命令、截图、提交或 PR。`.env` 被 Git 忽略；示例文件保持空密钥。已有 `.env` 时，`cp -n` 不覆盖它，请按 `.env.glm.example` 调整已有内容。

项目通过环境变量读取配置，**不会自动加载 `.env`**。本文所有运行命令都使用 `uv run --env-file .env` 显式加载。若当前终端已执行过 README 中的本地服务 `export` 命令，请先清除旧值，再从 `.env` 加载，避免旧地址、模型或 `VLM_API_KEY=local` 覆盖 GLM 配置：

```sh
unset VLM_PROVIDER VLM_BASE_URL VLM_MODEL VLM_API_KEY ZHIPU_API_KEY
unset VLM_THINKING VLM_REASONING_EFFORT VLM_MAX_TOKENS VLM_TIMEOUT_SECONDS
```

| 环境变量 | `VLM_PROVIDER=glm` 时的行为 |
| --- | --- |
| `VLM_PROVIDER` | 设置为 `glm` 启用预设；未设置时仍为 `openai-compatible` |
| `VLM_BASE_URL` | 默认 `https://open.bigmodel.cn/api/paas/v4`；可覆盖，不要包含 `/chat/completions`、密钥或查询参数 |
| `VLM_MODEL` | 默认 `glm-5.3-flashx`；可换成账户支持、能够接收 PNG Base64 的视觉模型 |
| `VLM_API_KEY` | 非空时优先使用；通常保留未设置，改填下面的 GLM 密钥变量 |
| `ZHIPU_API_KEY` | GLM 模式下，在 `VLM_API_KEY` 未设置或为空时使用；缺少密钥会报错 |
| `VLM_THINKING` | 默认 `enabled`；仅对允许关闭思考的模型使用 `disabled`，空值表示不发送该参数 |
| `VLM_REASONING_EFFORT` | 默认 `low`；可设置 `low`、`high`、`max`，空值表示不发送；须符合所选模型能力 |
| `VLM_MAX_TOKENS` | 默认 `4096`；可设置正整数，空值表示不发送该参数；须符合所选模型限制 |
| `VLM_TIMEOUT_SECONDS` | 默认 `120`，必须是有限正数，单位为秒 |

GLM 预设针对智谱国内官方 API。使用其他服务时，需要同时确认地址、密钥归属、视觉模型名和参数兼容性；账户权限与费用以实际服务为准，不保证免费。

请求沿用文本与 `image_url` 组合，图像为 PNG Base64。智谱文档明确 `response_format` 仅适用于文本模型，因此这里不给 GLM 视觉请求添加 `json_object` 或 `json_schema`，而是要求模型返回 JSON，再由本地 schema 校验。思考内容 `reasoning_content` 不作为动作使用。[智谱 API 文档](https://docs.bigmodel.cn/api-reference/模型-api/对话补全)、[官方视觉调用示例](https://github.com/zai-org/GLM-V/blob/main/skills/glmv-caption/scripts/glmv_caption.py)

默认模型 `glm-5.3-flashx` 始终开启思考，会拒绝 `disabled`，应使用 `low`、`high` 或 `max` 控制推理强度。当前预设与 `.env.glm.example` 使用以下已通过合成图检查的组合：

```dotenv
VLM_MODEL=glm-5.3-flashx
VLM_THINKING=enabled
VLM_REASONING_EFFORT=low
VLM_MAX_TOKENS=4096
```

换成其他视觉模型时，需要同时调整其支持的参数。例如尝试 `glm-4.6v-flash` 时，可设置 `VLM_THINKING=disabled` 和 `VLM_REASONING_EFFORT=`，显式关闭推理强度字段；仅删除后者会重新使用 GLM 预设的 `low`。每次更换模型或服务后，仍应运行下面的检查。

## 先检查带图请求

```sh
uv run --env-file .env view2grasp check-vlm --output artifacts/glm-check-01
```

此命令生成含多种颜色图形的小图，实际调用当前配置的 VLM，要求定位红色方块。它不导入或启动机器人仿真。通过条件是模型返回符合项目 schema 的抓取提议，并且所选接触点或区域中心落在红色方块内；只收到 HTTP 200 或合法 JSON 不算通过。这里不是精确框重叠率评测，也不验证深度几何、抓取执行或机器人任务成功。

每个输出目录只能使用一次；重试请改为 `artifacts/glm-check-02` 等新目录，保留失败证据。主要文件如下：

| 文件 | 内容 |
| --- | --- |
| `probe_report.json` | `success`、`robot_episode=false`、定位检查、错误、调用次数、实际响应模型 |
| `observations/frame_001.png` | 实际发送的合成图 |
| `vlm/call_*.json` | 请求模型、连接参数、图像哈希、原始响应、耗时及错误；不记录鉴权头 |

`success=true` 只表示这一张合成图的 API 检查通过。它不产生在线机器人 episode，不计入任务成功率，也不能证明复杂场景定位质量。请求的模型名与服务实际返回的模型名可能不同，应保留两者。

2026-10-11 的真实调用使用上述默认组合，1 次调用、2.144 秒通过检查，实际响应模型为 `glm-5.3-flashx`。本地报告为 `artifacts/glm-check-03/probe_report.json`，其中 `success=true`、`robot_episode=false`。此前 `glm-4.6v-flash` 的字段校验失败与限流、`glm-5.3-flashx` 关闭思考被 HTTP 400 拒绝的记录均已保留；这次成功只覆盖合成图 API 检查，尚不能据此声称机器人任务成功。

`doctor` 另行检查仿真渲染、标定和 `/models` 列表。它不发送视觉问题，不能替代 `check-vlm`。若服务未开放 `/models`，也不能仅凭该检查失败判定 `/chat/completions` 不可用。

## 单次方块验证

带图检查通过后，可手动运行一个方块 episode：

```sh
uv run --env-file .env view2grasp run \
  --task block --seed 3 --strategy active \
  --config configs/tabletop-targeted.yaml \
  --root runs/glm-block-01
```

此配置允许最多 8 次观测，属于针对性开发验证。默认感知后端为 `auto`；没有安装可选 SAM 或缺少权重时，会明确记录原因并回退 RGB-D，无需先安装 SAM 才能尝试。请检查 `events.json` 的 `perception_backend` 事件，不能将 RGB-D 回退标成 SAM 运行。

该命令只运行一次任务，不会启动批量评测。以 `result.json`、独立评估和实际执行记录判断结果；API 检查通过不保证这次抓取成功。后续重试使用新的 `--root`，先依据已有失败记录排查，不覆盖旧 episode。

## 失败诊断

| 现象 | 检查方向 |
| --- | --- |
| 提示缺少密钥 | 确认命令带 `--env-file .env`，且本地密钥已填写；不要打印密钥来诊断 |
| HTTP 401 / 403 | 检查密钥是否属于该地址、模型权限是否开放，以及旧 `VLM_API_KEY` 是否优先覆盖；鉴权失败直接停止 |
| HTTP 400 / 404 | 直接停止，先看日志中的服务错误消息；检查地址、视觉模型与请求参数。始终思考的模型需 `enabled` 和受支持的推理强度；不接受可选参数的服务则应将对应变量留空 |
| HTTP 429 / 5xx / 超时 | 检查服务状态、额度、并发限制；总共最多尝试 3 次。数字秒数形式的 `Retry-After` 会作为重试等待时间，上限为 30 秒；调整后用新目录再测 |
| `finish_reason=length` | 输出被截断，不能作为动作执行；检查 `VLM_MAX_TOKENS`。可关闭思考的模型可禁用 thinking；始终思考的模型可降低受支持的推理强度，或在模型限额内增加输出预算 |
| 缺少最终 `content` 或非正常结束 | 查看保存的响应与 `finish_reason`；思考内容、工具调用或被拦截的内容不能替代最终 JSON |
| JSON / frame / camera / 坐标错误 | 本地校验会拒绝并要求修正；首次调用加最多两次修复，耗时和失败全部记录 |
| 定位检查失败 | 查看合成图和报告中的点/框中心；这是视觉提议错误，不能通过放宽机器人地图或碰撞规则解决 |

非正常 `finish_reason`（例如 `length`、`sensitive`、`model_context_window_exceeded`）会拒绝解析为动作。服务返回的 JSON 错误体也会保存到调用日志，并对当前配置的密钥脱敏；不要将错误响应当成 JSON 格式修复问题反复调用。[官方响应字段说明](https://docs.z.ai/api-reference/llm/chat-completion)

## 切回原兼容服务

在本地 `.env` 中将 `VLM_PROVIDER` 改为 `openai-compatible`，并恢复原服务的 `VLM_BASE_URL`、`VLM_MODEL`、`VLM_API_KEY`。删除或留空 `VLM_THINKING`、`VLM_REASONING_EFFORT`、`VLM_MAX_TOKENS`，可恢复原先不发送这些可选参数的行为；仍使用 `uv run --env-file .env ...`。

参考 `.env.example` 的原默认值：地址 `http://127.0.0.1:8318/v1`、模型 `gt-6-as-a`、本地占位密钥 `local`、超时 120 秒。在通用模式下不使用 `ZHIPU_API_KEY`。更换 provider 不会自动清除显式设置的模型、地址或可选参数，因此请一并调整，且不要覆盖保存真实密钥的本地配置文件。
