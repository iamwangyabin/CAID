#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/soyo_cddb_official_compatible/${RUN_ID}}"
SUMMARY_FILE="${SUMMARY_FILE:-${OUTPUT_DIR}/soyo_run_${RUN_ID}.tsv}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

# Defaults track the released SOYO-ViT path exposed by configs/reproduce/soyo.yaml.
LOG_BACKEND="${LOG_BACKEND:-swanlab}"
LOG_MODE="${LOG_MODE:-cloud}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-16}"
INIT_EPOCH="${INIT_EPOCH:-50}"
EPOCHS="${EPOCHS:-50}"
SOYO_EPOCH="${SOYO_EPOCH:-30}"
SOYO_LR="${SOYO_LR:-0.1}"
SOYO_WEIGHT_DECAY="${SOYO_WEIGHT_DECAY:-0.0002}"
GMM_COMPONENTS="${GMM_COMPONENTS:-2}"
RESAMPLE_PER_DOMAIN="${RESAMPLE_PER_DOMAIN:-256}"
SELECTOR_BATCH_SIZE="${SELECTOR_BATCH_SIZE:-128}"

BASE_OVERRIDES=(
  "output_dir=${OUTPUT_DIR}"
  "device=${DEVICE}"
  "logging.backend=${LOG_BACKEND}"
  "logging.mode=${LOG_MODE}"
  "train.batch_size=${BATCH_SIZE}"
  "train.num_workers=${NUM_WORKERS}"
  "method.implementation=official"
  "method.net_type=soyo_vit"
  "method.detector_cfg.backbone.type=timm"
  "method.detector_cfg.backbone.name=vit_base_patch16_224"
  "method.detector_cfg.backbone.pretrained=true"
  "method.init_epoch=${INIT_EPOCH}"
  "method.epochs=${EPOCHS}"
  "method.soyo_epoch=${SOYO_EPOCH}"
  "method.soyo_lr=${SOYO_LR}"
  "method.soyo_weight_decay=${SOYO_WEIGHT_DECAY}"
  "method.gmm_components=${GMM_COMPONENTS}"
  "method.resample_per_domain=${RESAMPLE_PER_DOMAIN}"
  "method.selector_batch_size=${SELECTOR_BATCH_SIZE}"
  "method.normalize_selector_features=true"
)

if [[ -n "${DATA_LOCAL_DIR:-}" ]]; then
  BASE_OVERRIDES+=("scenario.data.remote.local_dir=${DATA_LOCAL_DIR}")
fi

if [[ -n "${PROTOCOL:-}" ]]; then
  BASE_OVERRIDES+=("scenario.protocol=${PROTOCOL}")
fi

EXTRA_OVERRIDES=()
if [[ -n "${OVERRIDES:-}" ]]; then
  EXTRA_OVERRIDES=($OVERRIDES)
fi

echo "[soyo] root=${ROOT_DIR}"
echo "[soyo] run_id=${RUN_ID}"
echo "[soyo] output_dir=${OUTPUT_DIR}"
echo "[soyo] summary=${SUMMARY_FILE}"
echo "[soyo] log_dir=${LOG_DIR}"
echo "[soyo] config=configs/reproduce/soyo.yaml"
echo "[soyo] overrides=${BASE_OVERRIDES[*]} ${EXTRA_OVERRIDES[*]-}"
echo "[soyo] note: official SOYO-CLIP text-prompt path is not implemented; this runs SOYO-ViT."

CONFIGS="configs/reproduce/soyo.yaml" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
LOG_DIR="$LOG_DIR" \
NUM_WORKERS="" \
OVERRIDES="${BASE_OVERRIDES[*]} ${EXTRA_OVERRIDES[*]-}" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
