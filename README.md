# voxcpm-open-tts

**English** · [中文](./README.zh.md)

OpenAI-compatible HTTP service wrapping [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — zero-shot voice cloning, natural-language voice design, and chunked streaming synthesis in a single CUDA image. Conforms to the [open-tts specification](https://github.com/OpenTTSGroup/open-tts-spec).

VoxCPM2 ships native 48 kHz output, 30+ languages, and supports three clone strategies (`reference`, `continuation`, `ref_continuation`). This service exposes all of them via the regulated open-tts endpoints.

## Quick start

```bash
# 1. Pull the image
docker pull ghcr.io/openttsgroup/voxcpm-open-tts:latest

# 2. Prepare voice & cache directories
mkdir -p ./voices ./cache

# 3. Run (GPU; weights download to ./cache on first boot)
docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$PWD/cache:/root/.cache" \
  -v "$PWD/voices:/voices:ro" \
  -e VOXCPM_MODEL=openbmb/VoxCPM2 \
  ghcr.io/openttsgroup/voxcpm-open-tts:latest

# 4. Check readiness
curl -s http://localhost:8000/healthz | jq
```

Or via `docker compose` (see [docker/docker-compose.example.yml](./docker/docker-compose.example.yml)):

```bash
cp docker/docker-compose.example.yml docker-compose.yml
docker compose up -d
```

First boot downloads the VoxCPM weights (~4 GB for VoxCPM2) into `./cache`. Set `HF_HUB_OFFLINE=1` to refuse any network activity after the initial download.

### CPU fallback

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/cache:/root/.cache" -v "$PWD/voices:/voices:ro" \
  -e VOXCPM_DEVICE=cpu -e VOXCPM_DTYPE=float32 \
  ghcr.io/openttsgroup/voxcpm-open-tts:latest
```

CPU inference is ~10× slower than an RTX 4090. `VOXCPM_COMPILE` is off by default and is ignored on CPU anyway.

## Voice directory

Place voice references in the mounted `./voices` directory as file triples:

```
./voices/
├── alice.wav   # 3-15 s reference audio, any mono/stereo sample rate
├── alice.txt   # Reference transcript (UTF-8)
└── alice.yml   # [Optional] metadata (name, gender, tags, …)
```

Clients address the voice as `voice="file://alice"` in any `/v1/audio/speech` or `/v1/audio/realtime` call. Built-in voices are not shipped — VoxCPM is a clone-only engine (`builtin_voices=false`).

### Clone modes

`VOXCPM_CLONE_MODE` chooses the default strategy for file-clone voices:

- `ref_continuation` (**default**, VoxCPM2 only) — both prompt (text+audio) and reference audio; best speaker similarity.
- `reference` (VoxCPM2 only) — audio-only reference (ignore `.txt`). Fastest; use when transcription is unreliable.
- `continuation` — prompt text + audio; works on VoxCPM1.x and VoxCPM2.

If the loaded model is not VoxCPM2, any `reference`/`ref_continuation` setting is auto-downgraded to `continuation`. Per-request `clone_mode` overrides the default.

### Instructions (style / voice design)

VoxCPM controls voice characteristics via a parenthetical prefix inside the input text. This service injects `instructions` (or `instruct` on `/design`) automatically:

```
input         = "你好，欢迎使用 VoxCPM。"
instructions  = "年轻女性，声音温柔甜美"
actual text   = "(年轻女性，声音温柔甜美)你好，欢迎使用 VoxCPM。"
```

## Configuration

All settings read from environment variables (case-insensitive). Prefix `VOXCPM_` is used for engine-specific knobs; service-level variables are prefix-free.

### Engine

| Variable | Default | Description |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HuggingFace repo id or local directory |
| `VOXCPM_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOXCPM_CUDA_INDEX` | `0` | GPU index when device is CUDA |
| `VOXCPM_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`. Reported in `/healthz`; CPU path auto-downgrades to `float32`. |
| `VOXCPM_COMPILE` | `false` | Enable `torch.compile` and a warm-up synthesis. Disabled automatically on CPU. Opt in on GPU for ~20–30% latency reduction after a one-time ~1–2 min warm-up. |
| `VOXCPM_LOAD_DENOISER` | `false` | Load the ZipEnhancer denoiser (~700 MB). Required if clients pass `denoise=true`. |
| `VOXCPM_ZIPENHANCER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | ModelScope denoiser id |
| `VOXCPM_LOCAL_FILES_ONLY` | `false` | Refuse network `snapshot_download` calls (also honours `HF_HUB_OFFLINE=1`) |
| `VOXCPM_CLONE_MODE` | `ref_continuation` | `reference` / `continuation` / `ref_continuation` |
| `VOXCPM_PROMPT_CACHE_SIZE` | `16` | Reserved for future LRU; currently inert |
| `VOXCPM_DEFAULT_CFG_VALUE` | `2.0` | Default `cfg_value` when the request omits it |
| `VOXCPM_DEFAULT_INFERENCE_TIMESTEPS` | `10` | Default diffusion steps |
| `VOXCPM_DEFAULT_MIN_LEN` | `2` | Default minimum audio token length |
| `VOXCPM_DEFAULT_MAX_LEN` | `4096` | Default maximum audio token length |
| `VOXCPM_DEFAULT_NORMALIZE` | `false` | Default VoxCPM text normalisation |
| `VOXCPM_DEFAULT_DENOISE` | `false` | Default reference-audio denoising |
| `VOXCPM_DEFAULT_RETRY_BADCASE` | `true` | Default bad-case retry behaviour |
| `VOXCPM_DEFAULT_RETRY_BADCASE_MAX_TIMES` | `3` | Default retry budget |
| `VOXCPM_DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD` | `6.0` | Default retry audio/text-ratio trigger |

### Service

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `VOICES_DIR` | `/voices` | Voice directory inside the container |
| `MAX_INPUT_CHARS` | `8000` | Hard cap for `input` length |
| `MAX_AUDIO_BYTES` | `20971520` | `/v1/audio/clone` upload cap (20 MB) |
| `MAX_CONCURRENCY` | `1` | Concurrent inference jobs |
| `MAX_QUEUE_SIZE` | `0` | Queue cap (0 = unbounded) |
| `QUEUE_TIMEOUT` | `0` | Queue wait timeout (0 = unbounded) |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | Fallback `response_format` |
| `CORS_ENABLED` | `false` | Enable wildcard CORS |

## API

| Method | Path | Notes |
|---|---|---|
| GET  | `/healthz` | Status, sample rate, capabilities |
| GET  | `/v1/audio/voices` | List file voices |
| GET  | `/v1/audio/voices/preview?id=<name>` | Download raw reference WAV |
| POST | `/v1/audio/speech` | OpenAI-compatible synthesis |
| POST | `/v1/audio/clone` | One-shot multipart clone |
| POST | `/v1/audio/design` | Voice design from a natural-language hint |
| POST | `/v1/audio/realtime` | Chunked streaming synthesis |

### `POST /v1/audio/speech`

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `model` | string | null | `ignored` | Accepted for OpenAI compatibility |
| `input` | string | — | `required` | Text to synthesize (`1..MAX_INPUT_CHARS`) |
| `voice` | string | — | `required` | Must be `file://<id>` (VoxCPM has no built-in voices) |
| `response_format` | enum | `mp3` | `supported` | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm` |
| `speed` | float | `1.0` | `ignored` | VoxCPM has no native speed control |
| `instructions` | string | null | `supported` | Injected as `(<instructions>)` prefix (VoxCPM parenthetical syntax) |
| `clone_mode` | enum | null | `extension` | Override `VOXCPM_CLONE_MODE`: `reference` / `continuation` / `ref_continuation` |
| `cfg_value` | float | null | `extension` | CFG guidance strength, `[0.1, 10.0]` |
| `inference_timesteps` | int | null | `extension` | Diffusion decoder steps, `[1, 100]` |
| `min_len` | int | null | `extension` | Minimum audio token length, `[1, 16384]` |
| `max_len` | int | null | `extension` | Maximum audio token length, `[1, 16384]` |
| `normalize` | bool | null | `extension` | VoxCPM text normalisation |
| `denoise` | bool | null | `conditional` | Denoise reference audio; requires `VOXCPM_LOAD_DENOISER=true` |
| `retry_badcase` | bool | null | `extension` | Auto-retry when output ratio looks off |
| `retry_badcase_max_times` | int | null | `extension` | Retry attempts, `[0, 10]` |
| `retry_badcase_ratio_threshold` | float | null | `extension` | Ratio trigger, `[1.0, 20.0]` |

### `POST /v1/audio/clone`

`multipart/form-data`. Same fields as `/v1/audio/speech` **except** `voice` (replaced by the uploaded `audio`) and with a required `prompt_text`:

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `audio` | file | — | `required` | Reference audio (`wav`/`mp3`/`flac`/`ogg`/`opus`/`m4a`/`aac`/`webm`), ≤ `MAX_AUDIO_BYTES` |
| `prompt_text` | string | `""` | `conditional` | Transcript of the reference. Empty allowed only when the engine is VoxCPM2 (auto-switches to `reference` mode) |
| `input` | string | — | `required` | Text to synthesize |
| `response_format` | enum | `mp3` | `supported` | Same set as `/speech` |
| `speed` | float | `1.0` | `ignored` | |
| `instructions` | string | null | `supported` | Parenthetical prefix injection |
| `clone_mode` | enum | null | `extension` | Same as `/speech` |
| `cfg_value`, `inference_timesteps`, `min_len`, `max_len`, `normalize`, `denoise`, `retry_badcase`, `retry_badcase_max_times`, `retry_badcase_ratio_threshold` | — | null | `extension` | See `/speech` |
| `model` | string | null | `ignored` | |

### `POST /v1/audio/design`

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `input` | string | — | `required` | Text to synthesize |
| `instruct` | string | null | `supported` | Natural-language voice hint; `null`/empty falls back to VoxCPM's internal default voice. Injected as `(<instruct>)` prefix. |
| `response_format` | enum | `mp3` | `supported` | Same set as `/speech` |
| `cfg_value`, `inference_timesteps`, `min_len`, `max_len`, `normalize`, `retry_badcase`, `retry_badcase_max_times`, `retry_badcase_ratio_threshold` | — | null | `extension` | See `/speech` |

### `POST /v1/audio/realtime`

Same JSON body as `/v1/audio/speech`. Response is `Transfer-Encoding: chunked`. Only `mp3`/`pcm`/`opus`/`aac` are streamable; `wav`/`flac` return HTTP 422.

## Development

```bash
# Update the VoxCPM submodule to a newer upstream commit:
git -C engine fetch
git -C engine checkout <commit>
git add engine && git commit -m "engine: bump to <commit>"

# Run app/ locally without Docker (needs CUDA + PyTorch matching the Dockerfile):
PYTHONPATH=./engine/src VOXCPM_MODEL=openbmb/VoxCPM2 \
  uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Links

- Spec: [OpenTTSGroup/open-tts-spec](https://github.com/OpenTTSGroup/open-tts-spec)
- Upstream engine: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- Sibling service: [cosyvoice-open-tts](https://github.com/OpenTTSGroup/cosyvoice-open-tts)

Upstream VoxCPM is released under Apache-2.0. Consult the model card for dataset attribution and commercial-use terms.
