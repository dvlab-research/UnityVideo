#!/usr/bin/env bash
set -euo pipefail

: "${WAN22_TI2V5B_DIR:?Set WAN22_TI2V5B_DIR to the Wan2.2-TI2V-5B directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${METADATA_DEPTH:?Set METADATA_DEPTH}"
: "${METADATA_DENSEPOSE:?Set METADATA_DENSEPOSE}"
: "${METADATA_RAFT:?Set METADATA_RAFT}"
: "${METADATA_SEGMENTATION:?Set METADATA_SEGMENTATION}"
: "${METADATA_SKELETON:?Set METADATA_SKELETON}"

args=(
  --model-root "$WAN22_TI2V5B_DIR"
  --modalities depth,densepose,raft,segmentation,skeleton
  --metadatas "$METADATA_DEPTH,$METADATA_DENSEPOSE,$METADATA_RAFT,$METADATA_SEGMENTATION,$METADATA_SKELETON"
  --modality-weights "${MODALITY_WEIGHTS:-0.2,0.2,0.2,0.4,0.4}"
  --task-weights "${TASK_WEIGHTS:-0.5,0.25,0.25}"
  --output "$OUTPUT_DIR"
  --num-frames "${NUM_FRAMES:-33}"
  --height "${HEIGHT:-256}"
  --width "${WIDTH:-256}"
  --steps "${STEPS:-30000}"
  --lr "${LR:-1e-5}"
  --flow-lr-mult "${FLOW_LR_MULT:-10}"
  --warmup "${WARMUP:-200}"
  --lr-min-ratio "${LR_MIN_RATIO:-1.0}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.5}"
  --ema-decay "${EMA_DECAY:-0.9999}"
  --step-per-ema "${STEP_PER_EMA:-8}"
  --prompt-dropout "${PROMPT_DROPOUT:-0.1}"
  --save-every "${SAVE_EVERY:-1000}"
)
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")

torchrun \
  --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  -m unityvideo.train "${args[@]}"
