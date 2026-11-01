#!/usr/bin/env bash
# True GRPO before/after pipeline:
#   :8000 (AWQ)  = reward/eval judge        (REWARD_MODEL_URL)
#   :8001 (bf16, --enable-lora) = POLICY served to red/blue agents
#   iter1 = base policy (BEFORE). After each iteration close_grpo_loop GRPO-trains the role
#   (LoRA), redeploy_lora.sh hot-loads the adapter into :8001, and the NEXT iteration's agent
#   requests model="red"/"blue" (the trained adapter) => AFTER.
set -uo pipefail
cd /home/jeonges/lab_project

# --- agents use the POLICY server (:8001); reward/eval stay on the AWQ server (:8000) ---
export QWEN_ENDPOINT=http://localhost:8001
export QWEN_ENDPOINT_MODEL=Qwen/Qwen3-8B
export RED_ENDPOINT=http://localhost:8001
export BLUE_ENDPOINT=http://localhost:8001
export RED_ENDPOINT_MODEL=Qwen/Qwen3-8B      # base until the first redeploy
export BLUE_ENDPOINT_MODEL=Qwen/Qwen3-8B
export RED_LORA_NAME=red
export BLUE_LORA_NAME=blue
export RED_PORT=8001 BLUE_PORT=8001 POLICY_PORT=8001
export REWARD_MODEL_URL=http://localhost:8000
export REWARD_MODEL_NAME=Qwen/Qwen3-8B
export QWEN_MODEL=Qwen/Qwen3-8B
export REDEPLOY_CMD='bash redeploy_lora.sh {role} {ckpt} {port}'

# --- GRPO (LoRA) training config: trains on GPU3; reward=:8000, policy=:8001 ---
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRPO_USE_LORA=True
export GRPO_MAX_STEPS=6
export GRPO_NUM_GENERATIONS=4
export GRPO_BATCH=4
export GRPO_MAX_COMPLETION_LEN=512
export GRPO_REWARD_MAX_TOKENS=256
export GRPO_TRAIN_TIMEOUT=7200

# --- live range + real Wazuh ---
export KEEP_RANGE=1 WAZUH_POLL_WAIT=4 WAZUH_WARMUP=90 WAZUH_POLL_MAX=12 WAZUH_POLL_INTERVAL=4

python -u main.py --live --fresh --skip-coverage --skip-train-hint --train-loop \
  --iterations 4 --max-steps 8 --benign-rate 3
