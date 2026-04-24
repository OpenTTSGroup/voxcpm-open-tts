# voxcpm-open-tts

[English](./README.md) · **中文**

兼容 OpenAI 的 HTTP 服务，封装 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) ——单镜像同时提供零样本声音克隆、自然语言声音设计、分块流式合成。遵循 [open-tts 规范](https://github.com/OpenTTSGroup/open-tts-spec)。

VoxCPM2 原生输出 48 kHz、支持 30+ 种语言，提供三种克隆策略（`reference`、`continuation`、`ref_continuation`）。本服务通过规范化的 open-tts 端点把它们全部对外暴露。

## 快速开始

```bash
# 1. 拉取镜像
docker pull ghcr.io/openttsgroup/voxcpm-open-tts:latest

# 2. 准备声音和缓存目录
mkdir -p ./voices ./cache

# 3. 运行（GPU；首次启动会把权重下载到 ./cache）
docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$PWD/cache:/root/.cache" \
  -v "$PWD/voices:/voices:ro" \
  -e VOXCPM_MODEL=openbmb/VoxCPM2 \
  ghcr.io/openttsgroup/voxcpm-open-tts:latest

# 4. 查看就绪状态
curl -s http://localhost:8000/healthz | jq
```

或使用 `docker compose`（见 [docker/docker-compose.example.yml](./docker/docker-compose.example.yml)）：

```bash
cp docker/docker-compose.example.yml docker-compose.yml
docker compose up -d
```

首次启动会把 VoxCPM 权重（VoxCPM2 约 4 GB）下载到 `./cache`。下载完成后可设 `HF_HUB_OFFLINE=1` 进入离线模式。

### 回退 CPU

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/cache:/root/.cache" -v "$PWD/voices:/voices:ro" \
  -e VOXCPM_DEVICE=cpu -e VOXCPM_DTYPE=float32 \
  ghcr.io/openttsgroup/voxcpm-open-tts:latest
```

CPU 推理比 RTX 4090 慢约 10 倍。`VOXCPM_COMPILE` 默认已关闭，且在 CPU 下也会自动禁用。

## 声音目录

把声音参考按三件套形式放进挂载的 `./voices`：

```
./voices/
├── alice.wav   # 3–15 秒参考音频，单/双声道任意采样率皆可
├── alice.txt   # 参考音频对应的文本（UTF-8）
└── alice.yml   # [可选] 元数据（name、gender、tags …）
```

客户端在 `/v1/audio/speech` 或 `/v1/audio/realtime` 中使用 `voice="file://alice"` 引用。本服务不内置任何音色——VoxCPM 是纯克隆引擎（`builtin_voices=false`）。

### 克隆模式

`VOXCPM_CLONE_MODE` 决定文件克隆的默认策略：

- `ref_continuation`（**默认**，仅 VoxCPM2）——同时使用文本+音频 prompt 与参考音频；音色相似度最高。
- `reference`（仅 VoxCPM2）——仅使用音频参考（忽略 `.txt`）。最快；转录不可靠时适用。
- `continuation`——文本 + 音频 prompt；VoxCPM1.x 和 VoxCPM2 都支持。

如果加载的不是 VoxCPM2，任何 `reference`/`ref_continuation` 设置会自动降级为 `continuation`。每个请求可用 `clone_mode` 字段覆盖默认。

### Instructions（风格 / 声音设计）

VoxCPM 通过在输入文本前加括号描述来控制声音/风格。本服务会自动注入 `instructions`（`/design` 端点为 `instruct`）：

```
input         = "你好，欢迎使用 VoxCPM。"
instructions  = "年轻女性，声音温柔甜美"
实际送给模型   = "(年轻女性，声音温柔甜美)你好，欢迎使用 VoxCPM。"
```

## 配置

所有配置通过环境变量读取（不区分大小写）。引擎相关用 `VOXCPM_` 前缀，服务通用变量无前缀。

### 引擎

| 变量 | 默认 | 说明 |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HuggingFace 仓库 ID 或本地目录 |
| `VOXCPM_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOXCPM_CUDA_INDEX` | `0` | CUDA 下的 GPU 索引 |
| `VOXCPM_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`；在 `/healthz` 中回显；CPU 路径自动降为 `float32` |
| `VOXCPM_COMPILE` | `false` | 启用 `torch.compile` 和预热合成。CPU 环境下自动关闭。GPU 环境可选开启，启动多一次 1–2 分钟预热，推理延迟可降 20–30% |
| `VOXCPM_LOAD_DENOISER` | `false` | 加载 ZipEnhancer 去噪器（约 700 MB）。客户端传 `denoise=true` 时需要 |
| `VOXCPM_ZIPENHANCER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | ModelScope 去噪模型 ID |
| `VOXCPM_LOCAL_FILES_ONLY` | `false` | 禁止 `snapshot_download` 联网（也可用 `HF_HUB_OFFLINE=1`） |
| `VOXCPM_CLONE_MODE` | `ref_continuation` | `reference` / `continuation` / `ref_continuation` |
| `VOXCPM_PROMPT_CACHE_SIZE` | `16` | 预留字段，当前未启用 |
| `VOXCPM_DEFAULT_CFG_VALUE` | `2.0` | 请求未显式传 `cfg_value` 时的默认值 |
| `VOXCPM_DEFAULT_INFERENCE_TIMESTEPS` | `10` | 扩散步数默认值 |
| `VOXCPM_DEFAULT_MIN_LEN` | `2` | 音频最小 token 长度默认值 |
| `VOXCPM_DEFAULT_MAX_LEN` | `4096` | 音频最大 token 长度默认值 |
| `VOXCPM_DEFAULT_NORMALIZE` | `false` | VoxCPM 文本归一化默认值 |
| `VOXCPM_DEFAULT_DENOISE` | `false` | 参考音频去噪默认值 |
| `VOXCPM_DEFAULT_RETRY_BADCASE` | `true` | 坏例自动重试默认值 |
| `VOXCPM_DEFAULT_RETRY_BADCASE_MAX_TIMES` | `3` | 重试次数默认值 |
| `VOXCPM_DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD` | `6.0` | 音频/文本比触发阈值默认值 |

### 服务

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | 绑定地址 |
| `PORT` | `8000` | 绑定端口 |
| `LOG_LEVEL` | `info` | uvicorn 日志级别 |
| `VOICES_DIR` | `/voices` | 容器内声音目录 |
| `MAX_INPUT_CHARS` | `8000` | `input` 长度上限 |
| `MAX_AUDIO_BYTES` | `20971520` | `/v1/audio/clone` 上传上限（20 MB） |
| `MAX_CONCURRENCY` | `1` | 同时推理的请求数 |
| `MAX_QUEUE_SIZE` | `0` | 队列上限（0 = 不限） |
| `QUEUE_TIMEOUT` | `0` | 队列等待超时秒数（0 = 不限） |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | 默认 `response_format` |
| `CORS_ENABLED` | `false` | 启用通配 CORS |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/healthz` | 状态、采样率、能力矩阵 |
| GET  | `/v1/audio/voices` | 列出文件克隆声音 |
| GET  | `/v1/audio/voices/preview?id=<name>` | 下载原始参考 WAV |
| POST | `/v1/audio/speech` | OpenAI 兼容合成 |
| POST | `/v1/audio/clone` | 一次性 multipart 克隆 |
| POST | `/v1/audio/design` | 基于自然语言描述的声音设计 |
| POST | `/v1/audio/realtime` | 分块流式合成 |

### `POST /v1/audio/speech`

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `model` | string | null | `ignored` | 兼容 OpenAI，接受但忽略 |
| `input` | string | — | `required` | 待合成文本（`1..MAX_INPUT_CHARS`） |
| `voice` | string | — | `required` | 必须 `file://<id>`（VoxCPM 无内置音色） |
| `response_format` | enum | `mp3` | `supported` | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm` |
| `speed` | float | `1.0` | `ignored` | VoxCPM 无原生变速能力 |
| `instructions` | string | null | `supported` | 以 `(<instructions>)` 前缀注入文本（VoxCPM 括号语法） |
| `clone_mode` | enum | null | `extension` | 覆盖 `VOXCPM_CLONE_MODE`：`reference` / `continuation` / `ref_continuation` |
| `cfg_value` | float | null | `extension` | CFG 引导强度，`[0.1, 10.0]` |
| `inference_timesteps` | int | null | `extension` | 扩散步数，`[1, 100]` |
| `min_len` | int | null | `extension` | 音频最小 token 长度，`[1, 16384]` |
| `max_len` | int | null | `extension` | 音频最大 token 长度，`[1, 16384]` |
| `normalize` | bool | null | `extension` | VoxCPM 文本归一化 |
| `denoise` | bool | null | `conditional` | 对参考音频去噪；需 `VOXCPM_LOAD_DENOISER=true` |
| `retry_badcase` | bool | null | `extension` | 自动重试坏例 |
| `retry_badcase_max_times` | int | null | `extension` | 重试次数，`[0, 10]` |
| `retry_badcase_ratio_threshold` | float | null | `extension` | 触发阈值，`[1.0, 20.0]` |

### `POST /v1/audio/clone`

`multipart/form-data`。字段与 `/v1/audio/speech` 相同，**除了** `voice` 由上传的 `audio` 取代，并新增必填的 `prompt_text`：

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `audio` | 文件 | — | `required` | 参考音频（`wav`/`mp3`/`flac`/`ogg`/`opus`/`m4a`/`aac`/`webm`），≤ `MAX_AUDIO_BYTES` |
| `prompt_text` | string | `""` | `conditional` | 参考音频的文本。仅当引擎是 VoxCPM2 时允许为空（自动切到 `reference` 模式） |
| `input` | string | — | `required` | 待合成文本 |
| `response_format` | enum | `mp3` | `supported` | 同 `/speech` |
| `speed` | float | `1.0` | `ignored` | |
| `instructions` | string | null | `supported` | 括号前缀注入 |
| `clone_mode` | enum | null | `extension` | 同 `/speech` |
| `cfg_value`、`inference_timesteps`、`min_len`、`max_len`、`normalize`、`denoise`、`retry_badcase`、`retry_badcase_max_times`、`retry_badcase_ratio_threshold` | — | null | `extension` | 同 `/speech` |
| `model` | string | null | `ignored` | |

### `POST /v1/audio/design`

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `input` | string | — | `required` | 待合成文本 |
| `instruct` | string | null | `supported` | 自然语言音色描述；`null`/空串时走 VoxCPM 默认声音。以 `(<instruct>)` 前缀注入 |
| `response_format` | enum | `mp3` | `supported` | 同 `/speech` |
| `cfg_value`、`inference_timesteps`、`min_len`、`max_len`、`normalize`、`retry_badcase`、`retry_badcase_max_times`、`retry_badcase_ratio_threshold` | — | null | `extension` | 同 `/speech` |

### `POST /v1/audio/realtime`

请求体与 `/v1/audio/speech` 相同。响应 `Transfer-Encoding: chunked`。仅 `mp3`/`pcm`/`opus`/`aac` 可流式；`wav`/`flac` 返回 HTTP 422。

## 开发

```bash
# 升级 VoxCPM 子模块到新 commit：
git -C engine fetch
git -C engine checkout <commit>
git add engine && git commit -m "engine: bump to <commit>"

# 本地直接运行 app/（需要匹配 Dockerfile 的 CUDA + PyTorch）：
PYTHONPATH=./engine/src VOXCPM_MODEL=openbmb/VoxCPM2 \
  uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## 相关链接

- 规范：[OpenTTSGroup/open-tts-spec](https://github.com/OpenTTSGroup/open-tts-spec)
- 上游引擎：[OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- 同系项目：[cosyvoice-open-tts](https://github.com/OpenTTSGroup/cosyvoice-open-tts)

上游 VoxCPM 遵循 Apache-2.0 协议。使用前请查阅其模型卡以确认数据归属和商用条款。
