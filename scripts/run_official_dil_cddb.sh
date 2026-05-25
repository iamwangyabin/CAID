#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/official_dil_cddb_runs/$RUN_ID/logs}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/official_dil_cddb_runs/$RUN_ID/summary.tsv}"
OVERRIDES="${OVERRIDES:-logging.backend=swanlab logging.mode=cloud}"
NUM_WORKERS="${NUM_WORKERS-4}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_DONE="${INCLUDE_DONE:-0}"

mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_FILE")"

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
    configs/ranpac.yaml
    configs/layup.yaml
    configs/pina.yaml
    configs/cp_prompt.yaml
    configs/duct.yaml
    configs/soyo.yaml
    configs/loranpac.yaml
    configs/dce.yaml
  )
  if [[ "$INCLUDE_DONE" == "1" ]]; then
    RUN_CONFIGS+=(configs/sprompts.yaml configs/prompt2guard.yaml)
  fi
fi

printf "index\tmethod\tconfig\tstatus\tstart_time\tseconds\tlog\n" > "$SUMMARY_FILE"

echo "[run] root=$ROOT_DIR"
echo "[run] logs=$LOG_DIR"
echo "[run] summary=$SUMMARY_FILE"
echo "[run] overrides=${OVERRIDES:-<none>}"
echo "[run] num_workers_default=${NUM_WORKERS:-config}"
echo "[run] train_cmd=${TRAIN_PARTS[*]}"

run_one() {
  local index="$1"
  local config="$2"
  local method
  local log_file
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
  log_file="$LOG_DIR/$(printf "%02d" "$index")_${method}.log"
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
  echo "[cmd] ${cmd[*]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$method" "$config" "DRY_RUN" "$start_time" "0" "$log_file" >> "$SUMMARY_FILE"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  end_sec="$(date +%s)"
  elapsed=$((end_sec - start_sec))

  if [[ "$status" -eq 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$method" "$config" "OK" "$start_time" "$elapsed" "$log_file" >> "$SUMMARY_FILE"
    echo "[ok] method=$method seconds=$elapsed log=$log_file"
  else
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$index" "$method" "$config" "FAIL:$status" "$start_time" "$elapsed" "$log_file" >> "$SUMMARY_FILE"
    echo "[fail] method=$method status=$status seconds=$elapsed log=$log_file" >&2
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
