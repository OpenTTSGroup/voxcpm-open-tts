from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
CloneMode = Literal["reference", "continuation", "ref_continuation"]


class Capabilities(BaseModel):
    clone: bool = Field(description="Zero-shot cloning support.")
    streaming: bool = Field(description="Chunked realtime synthesis support.")
    design: bool = Field(description="Text-only voice design support.")
    languages: bool = Field(description="Explicit language list support.")
    builtin_voices: bool = Field(description="Engine ships built-in voices.")


class ConcurrencySnapshot(BaseModel):
    max: int = Field(description="Global concurrency ceiling.")
    active: int = Field(description="Currently in-flight synthesis jobs.")
    queued: int = Field(description="Waiters blocked on the semaphore.")


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"] = Field(
        description="Engine readiness state."
    )
    model: str = Field(description="Loaded model identifier.")
    sample_rate: int = Field(description="Inference output sample rate (Hz).")
    capabilities: Capabilities = Field(description="Discovered engine capabilities.")
    device: Optional[str] = Field(default=None, description='e.g. "cuda:0" or "cpu".')
    dtype: Optional[str] = Field(default=None, description='e.g. "float16".')
    concurrency: Optional[ConcurrencySnapshot] = Field(
        default=None, description="Live concurrency snapshot."
    )


class VoiceInfo(BaseModel):
    id: str = Field(
        description='Voice identifier. "file://<name>" for disk voices, raw name for built-ins.'
    )
    preview_url: Optional[str] = Field(
        description="Preview URL for file voices; null for built-ins."
    )
    prompt_text: Optional[str] = Field(
        description="Reference transcript for file voices; null for built-ins."
    )
    metadata: Optional[dict[str, Any]] = Field(
        description="Optional metadata dict from <id>.yml."
    )


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo] = Field(description="Discovered voices.")


class _VoxCPMGenerationFields(BaseModel):
    """Shared VoxCPM-specific optional generation knobs.

    All fields are optional; ``None`` lets the engine wrapper fall back to its
    VOXCPM_DEFAULT_* setting. Unknown fields are ignored (``extra='ignore'``).
    """

    model_config = ConfigDict(extra="ignore")

    cfg_value: Optional[float] = Field(
        default=None, ge=0.1, le=10.0,
        description="Classifier-free guidance strength.",
    )
    inference_timesteps: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Diffusion decoder step count.",
    )
    min_len: Optional[int] = Field(
        default=None, ge=1, le=16384,
        description="Minimum audio token length.",
    )
    max_len: Optional[int] = Field(
        default=None, ge=1, le=16384,
        description="Maximum audio token length.",
    )
    normalize: Optional[bool] = Field(
        default=None,
        description="Run VoxCPM text normalisation before synthesis.",
    )
    retry_badcase: Optional[bool] = Field(
        default=None,
        description="Let VoxCPM retry when the output audio/text ratio is anomalous.",
    )
    retry_badcase_max_times: Optional[int] = Field(
        default=None, ge=0, le=10,
        description="Retry attempt budget (0 disables retries).",
    )
    retry_badcase_ratio_threshold: Optional[float] = Field(
        default=None, ge=1.0, le=20.0,
        description="Audio-to-text ratio threshold triggering a retry.",
    )


class SpeechRequest(_VoxCPMGenerationFields):
    model: Optional[str] = Field(
        default=None,
        description="Accepted for OpenAI compatibility; ignored.",
    )
    input: str = Field(
        min_length=1,
        description="Text to synthesize.",
    )
    voice: str = Field(
        description='Voice identifier; VoxCPM requires "file://<id>" (no built-in voices).'
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Output container/codec; defaults to the service setting.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Accepted for OpenAI compatibility; ignored by the VoxCPM engine.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language style hint. Injected as a '(<instructions>)' "
            "prefix to the synthesis text — VoxCPM's parenthetical control syntax."
        ),
    )
    denoise: Optional[bool] = Field(
        default=None,
        description="Denoise the reference/prompt audio (requires VOXCPM_LOAD_DENOISER=true).",
    )
    clone_mode: Optional[CloneMode] = Field(
        default=None,
        description=(
            "Override VOXCPM_CLONE_MODE for this request. "
            "'reference' / 'ref_continuation' require VoxCPM2."
        ),
    )


# Realtime shares the same request shape.
RealtimeRequest = SpeechRequest


class DesignRequest(_VoxCPMGenerationFields):
    input: str = Field(
        min_length=1,
        description="Text to synthesize.",
    )
    instruct: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language voice description. Injected as a '(<instruct>)' "
            "prefix; empty/null lets VoxCPM pick its internal default voice."
        ),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Output container/codec; defaults to the service setting.",
    )
