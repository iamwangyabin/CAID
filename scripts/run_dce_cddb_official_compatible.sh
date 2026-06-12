#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dce_cddb_official_compatible/${RUN_ID}}"
SUMMARY_FILE="${SUMMARY_FILE:-${OUTPUT_DIR}/dce_run_${RUN_ID}.tsv}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

# Keep the DCE defaults close to Lain810/DCE while still allowing command-line
# overrides through CAIDBench's config system.
LOG_BACKEND="${LOG_BACKEND:-swanlab}"
LOG_MODE="${LOG_MODE:-cloud}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-16}"

BASE_OVERRIDES=(
  "output_dir=${OUTPUT_DIR}"
  "device=${DEVICE}"
  "logging.backend=${LOG_BACKEND}"
  "logging.mode=${LOG_MODE}"
  "train.batch_size=${BATCH_SIZE}"
  "train.num_workers=${NUM_WORKERS}"
  "method.implementation=official_compatible"
  "method.prompt_type=one"
  "method.feature_scaling_mode=1"
  "method.zero_class_count=0.1"
  "method.share_covariance_within_task=true"
  "method.use_official_expert_optimizer=true"
  "method.use_test_transform_for_stats=true"
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

echo "[dce] root=${ROOT_DIR}"
echo "[dce] run_id=${RUN_ID}"
echo "[dce] output_dir=${OUTPUT_DIR}"
echo "[dce] summary=${SUMMARY_FILE}"
echo "[dce] log_dir=${LOG_DIR}"
echo "[dce] config=configs/reproduce/dce.yaml"
echo "[dce] overrides=${BASE_OVERRIDES[*]} ${EXTRA_OVERRIDES[*]-}"
echo "[dce] note: for paper-number reproduction, build the CDDB train split with the official Lain810/DCE utils/data.py::make_imb table."

CONFIGS="configs/reproduce/dce.yaml" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
LOG_DIR="$LOG_DIR" \
NUM_WORKERS="" \
OVERRIDES="${BASE_OVERRIDES[*]} ${EXTRA_OVERRIDES[*]-}" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
