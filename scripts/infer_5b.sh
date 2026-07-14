#!/usr/bin/env bash
set -euo pipefail

: "${TASK:?Set TASK to text2all, video2flow, or flow2video}"
: "${OUTPUT:?Set OUTPUT to an .mp4 path}"

args=(
  --task "$TASK"
  --modality "${MODALITY:-depth}"
  --prompt "${PROMPT:-}"
  --output "$OUTPUT"
  --num-frames "${NUM_FRAMES:-33}"
  --height "${HEIGHT:-256}"
  --width "${WIDTH:-256}"
  --steps "${INFERENCE_STEPS:-20}"
  --seed "${SEED:-42}"
)

[[ -z "${WAN22_TI2V5B_DIR:-}" ]] || args+=(--model-root "$WAN22_TI2V5B_DIR")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${RGB_VIDEO:-}" ]] || args+=(--rgb-video "$RGB_VIDEO")
[[ -z "${CONDITION_VIDEO:-}" ]] || args+=(--condition-video "$CONDITION_VIDEO")

unityvideo-infer "${args[@]}"
