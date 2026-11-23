#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RESEARCH_ROOT"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$RESEARCH_ROOT/.venv/bin/python" -m research "$@"
