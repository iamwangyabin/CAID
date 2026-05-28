#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/duct_cddb_hard/duct_run_${RUN_ID}.tsv}"

CONFIGS="configs/duct.yaml" \
RUN_ID="$RUN_ID" \
SUMMARY_FILE="$SUMMARY_FILE" \
"$ROOT_DIR/scripts/run_cddb_hard_methods.sh"
