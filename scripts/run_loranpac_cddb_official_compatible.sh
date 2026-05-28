#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/loranpac_cddb_official_compatible/${RUN_ID}}"
SUMMARY_FILE="${SUMMARY_FILE:-${OUTPUT_DIR}/loranpac_run_${RUN_ID}.tsv}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

# Defaults mirror liangzu/loranpac exps/tsvd/cddb.json where possible.
LOG_BACKEND="${LOG_BACKEND:-swanlab}"
LOG_MODE="${LOG_MODE:-cloud}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TSVD_BATCH_SIZE="${TSVD_BATCH_SIZE:-1000}"
TSVD_UPDATE_THRESHOLD="${TSVD_UPDATE_THRESHOLD:-10000}"
E="${E:-100000}"
RANK="${RANK:-20000}"
TRUNCATE_PERCENT="${TRUNCATE_PERCENT:-25}"
RIDGE="${RIDGE:-0}"

BASE_OVERRIDES=(
  "output_dir=${OUTPUT_DIR}"
  "device=${DEVICE}"
  "logging.backend=${LOG_BACKEND}"
  "logging.mode=${LOG_MODE}"
  "train.batch_size=${BATCH_SIZE}"
  "train.num_workers=${NUM_WORKERS}"
  "method.detector_cfg.backbone.out_dim=null"
  "method.E=${E}"
  "method.rank=${RANK}"
  "method.truncate_percent=${TRUNCATE_PERCENT}"
  "method.ridge=${RIDGE}"
  "method.tsvd_batch_size=${TSVD_BATCH_SIZE}"
  "method.tsvd_update_threshold=${TSVD_UPDATE_THRESHOLD}"
  "method.use_test_transform_for_tsvd=true"
  "method.use_RE=true"
  "method.use_relu=true"
  "method.coslinear=false"
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

echo "[loranpac] root=${ROOT_DIR}"
echo "[loranpac] run_id=${RUN_ID}"
echo "[loranpac] output_dir=${OUTPUT_DIR}"
echo "[loranpac] summary=${SUMMARY_FILE}"
echo "[loranpac] log_dir=${LOG_DIR}"
echo "[loranpac] config=configs/loranpac.yaml"
echo "[loranpac] overrides=${BASE_OVERRIDES[*]} ${EXTRA_OVERRIDES[*]-}"
echo "[loranpac] note: official-size E=100000 can be memory-heavy; set E/RANK lower only for debugging."

CONFIGS="configs/loranpac.yaml" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
LOG_DIR="$LOG_DIR" \
NUM_WORKERS="" \
OVERRIDES="${BASE_OVERRIDES[*]} ${EXTRA_OVERRIDES[*]-}" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
