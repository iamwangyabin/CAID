#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${RAW_ROOT:-/data/caid_raw}
PROCESSED_ROOT=${PROCESSED_ROOT:-/data/caid_processed}
ARROW_ROOT=${ARROW_ROOT:-/data/caid_arrow}
FORMAT=${FORMAT:-aid}

mkdir -p "$ARROW_ROOT"

run_if_exists() {
  local kind=$1
  local root=$2
  local out=$3
  shift 3
  if [[ -d "$root" ]]; then
    echo "[pack] $kind: $root -> $out"
    caid-pack-dataset --kind "$kind" --root "$root" --out "$out" --format "$FORMAT" "$@"
  else
    echo "[skip] missing: $root"
  fi
}

run_if_exists cddb "$RAW_ROOT/CDDB" "$ARROW_ROOT/cddb"
run_if_exists cnn_detection "$RAW_ROOT/CNNDetection" "$ARROW_ROOT/cnn_detection"
run_if_exists genimage "$RAW_ROOT/GenImage" "$ARROW_ROOT/genimage"
run_if_exists deepfakebench "$PROCESSED_ROOT/deepfakebench_faces" "$ARROW_ROOT/deepfakebench_faces" --preprocess-profile sur_lid_deepfakebench_v1
run_if_exists tifs_cail "$RAW_ROOT/TIFS_CAIL_protocol2" "$ARROW_ROOT/tifs_cail_protocol2"
