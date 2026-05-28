#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/current_cddb_method_runs_${RUN_ID}.tsv}"
NUM_WORKERS="${NUM_WORKERS:-16}"

CONFIGS=(
  configs/duct.yaml
  configs/soyo.yaml
  configs/loranpac.yaml
  configs/dce.yaml
)

echo "[current] root=${ROOT_DIR}"
echo "[current] run_id=${RUN_ID}"
echo "[current] summary=${SUMMARY_FILE}"
echo "[current] configs=${CONFIGS[*]}"

CONFIGS="${CONFIGS[*]}" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
NUM_WORKERS="$NUM_WORKERS" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
