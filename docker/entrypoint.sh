#!/usr/bin/env bash
set -euo pipefail

# Engine defaults
: "${VOXCPM_MODEL:=openbmb/VoxCPM2}"
: "${VOXCPM_DEVICE:=auto}"
: "${VOXCPM_DTYPE:=float16}"
: "${VOXCPM_OPTIMIZE:=true}"
: "${VOXCPM_LOAD_DENOISER:=false}"
: "${VOXCPM_CLONE_MODE:=ref_continuation}"

# Service-level defaults
: "${VOICES_DIR:=/voices}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${CORS_ENABLED:=false}"
: "${PYTHONPATH:=/opt/api:/opt/api/engine/src}"
: "${VOXCPM_ROOT:=/opt/api/engine}"

export VOXCPM_MODEL VOXCPM_DEVICE VOXCPM_DTYPE VOXCPM_OPTIMIZE \
       VOXCPM_LOAD_DENOISER VOXCPM_CLONE_MODE \
       VOICES_DIR HOST PORT LOG_LEVEL CORS_ENABLED \
       PYTHONPATH VOXCPM_ROOT

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
