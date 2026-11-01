#!/usr/bin/env bash
# Train the DEFENSE on the shell cyber-range until it beats CAI, model held fixed (Qwen3-8B).
# Closed-loop learning each iteration: GRPO (LoRA) on exported data + blue authors real Wazuh
# rules for missed techniques (detection climbs) + blue applies the learned host containment
# (patch climbs). Agents use the base policy on :8000; GRPO runs on GPU3 (training performed).
set -uo pipefail
cd /home/jeonges/lab_project
unset RED_ENDPOINT BLUE_ENDPOINT RED_ENDPOINT_MODEL BLUE_ENDPOINT_MODEL WAZUH_AGENT_NAME
export QWEN_ENDPOINT=http://localhost:8000
export QWEN_ENDPOINT_MODEL=Qwen/Qwen3-8B
export REWARD_MODEL_URL=http://localhost:8000
export QWEN_MODEL=Qwen/Qwen3-8B
# GRPO (LoRA) training on the free GPU; no adapter re-serving (robust long run)
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRPO_USE_LORA=True GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-4}" GRPO_NUM_GENERATIONS=4 GRPO_BATCH=4
export GRPO_MAX_COMPLETION_LEN=512 GRPO_REWARD_MAX_TOKENS=256
export KEEP_RANGE=1 WAZUH_POLL_WAIT=4 WAZUH_WARMUP=60 WAZUH_POLL_MAX=20 WAZUH_POLL_INTERVAL=4

ITERS="${1:-6}"
# expand ON: red invents on-surface /ping injection VARIANTS. With structure/decoder-based
# detection (encoding-agnostic) these are all caught regardless of encoding, so red's novelty
# no longer evades the SIEM -- it just exercises blue's (now robust) detection + containment.
python -u main.py --live --skip-coverage --skip-train-hint --train-loop \
  --ctf-name shell --profile shell_linux --iterations "$ITERS" --max-steps 8 --benign-rate 3
