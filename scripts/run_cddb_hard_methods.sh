#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/cddb_hard_method_runs_${RUN_ID}.tsv}"
OVERRIDES="${OVERRIDES:-logging.backend=swanlab logging.mode=cloud}"
NUM_WORKERS="${NUM_WORKERS-4}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_DONE="${INCLUDE_DONE:-0}"

mkdir -p "$(dirname "$SUMMARY_FILE")"

if [[ -n "${TRAIN_CMD:-}" ]]; then
  TRAIN_PARTS=($TRAIN_CMD)
elif command -v caid-train >/dev/null 2>&1; then
  TRAIN_PARTS=(caid-train)
else
  TRAIN_PARTS=(python3 -m caidbench.cli.train)
fi

if [[ -n "${CONFIGS:-}" ]]; then
  RUN_CONFIGS=($CONFIGS)
else
  RUN_CONFIGS=(
    configs/reproduce/ranpac.yaml
    configs/reproduce/layup.yaml
    configs/reproduce/pina.yaml
    configs/reproduce/cp_prompt.yaml
    configs/reproduce/duct.yaml
    configs/reproduce/soyo.yaml
    configs/reproduce/loranpac.yaml
    configs/reproduce/dce.yaml
  )
  if [[ "$INCLUDE_DONE" == "1" ]]; then
    RUN_CONFIGS+=(configs/reproduce/sprompts.yaml configs/reproduce/prompt2guard.yaml)
  fi
fi

config_output_dir() {
  local config="$1"
  local out
  out="$(awk '/^output_dir:[[:space:]]*/ { sub(/^output_dir:[[:space:]]*/, ""); print; exit }' "$config")"
  if [[ -z "$out" ]]; then
    out="outputs/$(basename "$config" .yaml)"
  fi
  printf "%s" "$out"
}

override_output_dir() {
  local part
  for part in ${OVERRIDES:-}; do
    if [[ "$part" == output_dir=* ]]; then
      printf "%s" "${part#output_dir=}"
      return 0
    fi
  done
}

printf "index\tmethod\tconfig\tstatus\tstart_time\tseconds\tlog\toutput_dir\n" > "$SUMMARY_FILE"

echo "[run] root=$ROOT_DIR"
echo "[run] batch_id=$RUN_ID"
echo "[run] summary=$SUMMARY_FILE"
echo "[run] logs=${LOG_DIR:-<config output_dir>/logs}"
echo "[run] overrides=${OVERRIDES:-<none>}"
echo "[run] num_workers_default=${NUM_WORKERS:-config}"
echo "[run] train_cmd=${TRAIN_PARTS[*]}"

run_one() {
  local index="$1"
  local config="$2"
  local method
  local method_output_dir
  local log_dir
  local log_file
  local method_start_stamp
  local start_time
  local start_sec
  local end_sec
  local elapsed
  local status
  local cmd
  local -a override_parts

  if [[ ! -f "$config" ]]; then
    echo "[error] missing config: $config" >&2
    return 2
  fi

  method="$(basename "$config" .yaml)"
  method_output_dir="$(override_output_dir)"
  if [[ -z "$method_output_dir" ]]; then
    method_output_dir="$(config_output_dir "$config")"
  fi
  log_dir="${LOG_DIR:-$method_output_dir/logs}"
  mkdir -p "$log_dir"
  method_start_stamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$log_dir/$(printf "%02d" "$index")_${method}_${method_start_stamp}.log"
  start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_sec="$(date +%s)"

  cmd=("${TRAIN_PARTS[@]}" --config "$config")
  override_parts=()
  if [[ -n "$OVERRIDES" ]]; then
    override_parts=($OVERRIDES)
  fi
  if [[ -n "$NUM_WORKERS" && " ${OVERRIDES:-} " != *" train.num_workers="* ]]; then
    override_parts+=("train.num_workers=$NUM_WORKERS")
  fi
  if [[ "${#override_parts[@]}" -gt 0 ]]; then
    cmd+=(--override "${override_parts[@]}")
  fi

  echo
  echo "[run] $index/${#RUN_CONFIGS[@]} method=$method config=$config"
  echo "[run] output_dir=$method_output_dir"
  echo "[run] log=$log_file"
  echo "[cmd] ${cmd[*]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$method" "$config" "DRY_RUN" "$start_time" "0" "$log_file" "$method_output_dir" >> "$SUMMARY_FILE"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  end_sec="$(date +%s)"
  elapsed=$((end_sec - start_sec))

  if [[ "$status" -eq 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$method" "$config" "OK" "$start_time" "$elapsed" "$log_file" "$method_output_dir" >> "$SUMMARY_FILE"
    echo "[ok] method=$method seconds=$elapsed log=$log_file output_dir=$method_output_dir"
  else
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$method" "$config" "FAIL:$status" "$start_time" "$elapsed" "$log_file" "$method_output_dir" >> "$SUMMARY_FILE"
    echo "[fail] method=$method status=$status seconds=$elapsed log=$log_file output_dir=$method_output_dir" >&2
    return "$status"
  fi
}

for i in "${!RUN_CONFIGS[@]}"; do
  index=$((i + 1))
  if ! run_one "$index" "${RUN_CONFIGS[$i]}"; then
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
      echo "[warn] continuing after failure because CONTINUE_ON_ERROR=1" >&2
    else
      echo "[stop] failed. Set CONTINUE_ON_ERROR=1 to keep running later configs." >&2
      exit 1
    fi
  fi
done

echo
echo "[done] all requested configs finished"
echo "[done] summary=$SUMMARY_FILE"
