#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/caidbench_method_runs_${RUN_ID}.tsv}"
NUM_WORKERS="${NUM_WORKERS:-16}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OVERRIDES="${OVERRIDES:-logging.backend=swanlab logging.mode=cloud}"

CONFIGS=(
  configs/caidbench/base.yaml
  configs/caidbench/prompt2guard.yaml
  configs/caidbench/sprompts.yaml
  configs/caidbench/ranpac.yaml
  configs/caidbench/loranpac.yaml
  configs/caidbench/soyo.yaml
  configs/caidbench/dce.yaml
  configs/caidbench/layup.yaml
  configs/caidbench/pina.yaml
  configs/caidbench/cp_prompt.yaml
  configs/caidbench/duct.yaml
)

echo "[caidbench] root=${ROOT_DIR}"
echo "[caidbench] run_id=${RUN_ID}"
echo "[caidbench] summary=${SUMMARY_FILE}"
echo "[caidbench] continue_on_error=${CONTINUE_ON_ERROR}"
echo "[caidbench] configs=${CONFIGS[*]}"

CONFIGS="${CONFIGS[*]}" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
NUM_WORKERS="$NUM_WORKERS" \
CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
OVERRIDES="$OVERRIDES" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
