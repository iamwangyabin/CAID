#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/stitched_method_runs_${RUN_ID}.tsv}"
NUM_WORKERS="${NUM_WORKERS:-16}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OVERRIDES="${OVERRIDES:-logging.backend=swanlab logging.mode=cloud}"

CONFIGS=(
  configs/stitched/base.yaml
  configs/stitched/prompt2guard.yaml
  configs/stitched/sprompts.yaml
  configs/stitched/ranpac.yaml
  configs/stitched/loranpac.yaml
  configs/stitched/soyo.yaml
  configs/stitched/dce.yaml
  configs/stitched/layup.yaml
  configs/stitched/pina.yaml
  configs/stitched/cp_prompt.yaml
  configs/stitched/duct.yaml
)

echo "[stitched] root=${ROOT_DIR}"
echo "[stitched] run_id=${RUN_ID}"
echo "[stitched] summary=${SUMMARY_FILE}"
echo "[stitched] continue_on_error=${CONTINUE_ON_ERROR}"
echo "[stitched] configs=${CONFIGS[*]}"

CONFIGS="${CONFIGS[*]}" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
NUM_WORKERS="$NUM_WORKERS" \
CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
OVERRIDES="$OVERRIDES" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
