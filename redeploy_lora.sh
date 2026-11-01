#!/usr/bin/env bash
# REDEPLOY_CMD target: hot-swap a freshly-trained LoRA adapter into the RUNNING policy vLLM
# (started with --enable-lora + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1), with NO restart.
# Invoked by close_grpo_loop as:  redeploy_lora.sh <role> <ckpt_dir> <port>
# After this, requests to the policy server with model="<role>" use the trained adapter,
# so the next self-play iteration reflects GRPO training (the "after" policy).
set -uo pipefail
ROLE="${1:?role}"; CKPT="${2:?ckpt}"; PORT="${POLICY_PORT:-8001}"
URL="http://localhost:${PORT}"

# The adapter (adapter_config.json + weights) lives in the newest checkpoint-* subdir.
ADAPTER="$(ls -dt "${CKPT}"/checkpoint-* 2>/dev/null | head -1)"
[ -z "$ADAPTER" ] && ADAPTER="$CKPT"
if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
  echo "[redeploy] no adapter_config.json under ${ADAPTER} -> abort"; exit 1
fi
ADAPTER="$(cd "$ADAPTER" && pwd)"   # absolute path (vLLM resolves server-side)

echo "[redeploy] ${ROLE}: loading adapter ${ADAPTER} into ${URL}"
# unload a previous same-named adapter (ignore if absent), then load the new one
curl -s -m 30 -X POST "${URL}/v1/unload_lora_adapter" \
  -H 'Content-Type: application/json' -d "{\"lora_name\":\"${ROLE}\"}" >/dev/null 2>&1 || true
code=$(curl -s -m 120 -o /tmp/redeploy_${ROLE}.out -w "%{http_code}" -X POST "${URL}/v1/load_lora_adapter" \
  -H 'Content-Type: application/json' -d "{\"lora_name\":\"${ROLE}\",\"lora_path\":\"${ADAPTER}\"}")
echo "[redeploy] load_lora_adapter http=${code} resp=$(head -c 200 /tmp/redeploy_${ROLE}.out 2>/dev/null)"
# verify the adapter is now a served model id
if curl -s -m 15 "${URL}/v1/models" | grep -q "\"${ROLE}\""; then
  echo "[redeploy] ${ROLE}: adapter is now served as model '${ROLE}'"; exit 0
fi
[ "$code" = "200" ] && exit 0 || exit 1
