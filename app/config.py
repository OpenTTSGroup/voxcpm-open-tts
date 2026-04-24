from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Engine (VOXCPM_* prefix) --------------------------------------------
    voxcpm_model: str = Field(
        default="openbmb/VoxCPM2",
        description="HuggingFace repo id or local directory for the VoxCPM model.",
    )
    voxcpm_device: Literal["auto", "cuda", "cpu"] = "auto"
    voxcpm_cuda_index: int = Field(default=0, ge=0)
    voxcpm_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    voxcpm_compile: bool = Field(
        default=False,
        description="Enable torch.compile-based optimisation inside VoxCPM.from_pretrained.",
    )
    voxcpm_load_denoiser: bool = Field(
        default=False,
        description="Load the ZipEnhancer denoiser alongside the TTS model.",
    )
    voxcpm_zipenhancer_model: str = Field(
        default="iic/speech_zipenhancer_ans_multiloss_16k_base",
        description="ModelScope denoiser id used when voxcpm_load_denoiser is true.",
    )
    voxcpm_local_files_only: bool = Field(
        default=False,
        description="Refuse network access during snapshot_download (HF_HUB_OFFLINE also works).",
    )
    voxcpm_clone_mode: Literal["reference", "continuation", "ref_continuation"] = Field(
        default="ref_continuation",
        description=(
            "Default clone mode when voice=file://<id>. "
            "ref_continuation and reference require VoxCPM2; "
            "older VoxCPM/VoxCPM1.5 models auto-downgrade to continuation."
        ),
    )
    voxcpm_prompt_cache_size: int = Field(
        default=16,
        ge=1,
        description="Reserved for future prompt-cache LRU; unused in v1.",
    )

    # Default generation parameters (all per-request fields may override these).
    voxcpm_default_cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    voxcpm_default_inference_timesteps: int = Field(default=10, ge=1, le=100)
    voxcpm_default_min_len: int = Field(default=2, ge=1, le=16384)
    voxcpm_default_max_len: int = Field(default=4096, ge=1, le=16384)
    voxcpm_default_normalize: bool = False
    voxcpm_default_denoise: bool = False
    voxcpm_default_retry_badcase: bool = True
    voxcpm_default_retry_badcase_max_times: int = Field(default=3, ge=0, le=10)
    voxcpm_default_retry_badcase_ratio_threshold: float = Field(
        default=6.0, ge=1.0, le=20.0
    )

    # --- Service-level (no prefix) -------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(default=8000, ge=1)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = "mp3"
    max_concurrency: int = Field(default=1, ge=1)
    max_queue_size: int = Field(default=0, ge=0)
    queue_timeout: float = Field(default=0.0, ge=0.0)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    cors_enabled: bool = False

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)

    @property
    def resolved_device(self) -> str:
        if self.voxcpm_device == "cpu":
            return "cpu"
        if self.voxcpm_device == "cuda":
            return f"cuda:{self.voxcpm_cuda_index}"
        import torch

        if torch.cuda.is_available():
            return f"cuda:{self.voxcpm_cuda_index}"
        return "cpu"

    @property
    def use_fp16(self) -> bool:
        return (
            self.voxcpm_dtype == "float16"
            and self.resolved_device.startswith("cuda")
        )

    @property
    def effective_dtype(self) -> str:
        # CPU path cannot reliably run fp16 inference; report float32.
        if not self.resolved_device.startswith("cuda"):
            return "float32"
        return self.voxcpm_dtype


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
