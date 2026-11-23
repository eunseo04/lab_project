#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RESEARCH_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
mkdir -p research_outputs
exec .model-venv/bin/python research/model_server.py --model "$RESEARCH_ROOT/models/policy" --encoder "$RESEARCH_ROOT/models/encoder" "$@"
