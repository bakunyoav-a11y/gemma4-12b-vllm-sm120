#!/usr/bin/env bash
# Serve AEON gemma-4-12B (abliterated, Mixed NVFP4+FP8) with MTP speculative decode.
# OpenAI-compatible API on :8000. Two modes (set MODE):
#   MODE=solo  (default) -> 1 GPU, ~50 tok/s single-stream, up to -c 32768, frees 6 GPUs
#   MODE=fast            -> 4 GPUs (no-P2P flags), ~118 tok/s single-stream, huge KV
# Measured 2026-06-07 on RTX PRO 2000 Blackwell (16GB, SM120, no NVLink/P2P).
set -euo pipefail

MODE="${MODE:-solo}"
PORT="${PORT:-8000}"
# Point MODELS at the dir holding the two downloaded repos (see README "Download").
MODELS="${MODELS:-./models}"
TARGET="${TARGET:-$MODELS/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8}"
DRAFT="${DRAFT:-$MODELS/gemma-4-12B-it-assistant}"
SPEC='{"method":"mtp","model":"/draft","num_speculative_tokens":3}'   # k=3 = peak

if [ "$MODE" = fast ]; then
  GPUS="${GPUS:-0,1,4,5}"; TP=4; MAXLEN="${MAXLEN:-32768}"; UTIL=0.90
  EXTRA=(--tensor-parallel-size 4 --disable-custom-all-reduce)
  ENVS=(-e NCCL_P2P_DISABLE=1)                 # required at TP>1 on this no-P2P box
else
  GPUS="${GPUS:-0}"; TP=1; MAXLEN="${MAXLEN:-32768}"; UTIL=0.95
  EXTRA=(--tensor-parallel-size 1 --max-num-seqs 4)   # low seqs -> KV room for 32k ctx
  ENVS=()
fi

docker run --rm -d --name gemma4-12b-aeon \
  --gpus "\"device=$GPUS\"" --ipc=host --shm-size 16gb -p "$PORT":8000 \
  -e HF_HUB_DISABLE_TELEMETRY=1 "${ENVS[@]}" \
  -v "$TARGET":/model:ro -v "$DRAFT":/draft:ro \
  vllm/vllm-openai:nightly \
    --model /model --served-model-name gemma-4-12b-aeon \
    --quantization modelopt --kv-cache-dtype fp8 \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
    --speculative-config "$SPEC" --trust-remote-code "${EXTRA[@]}"

echo "[$MODE] gemma-4-12b-aeon -> http://localhost:$PORT  (GPUs $GPUS, TP=$TP, -c $MAXLEN)"
echo "Logs: docker logs -f gemma4-12b-aeon   Stop: docker stop gemma4-12b-aeon"
