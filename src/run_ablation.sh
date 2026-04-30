#!/usr/bin/env bash
# ============================================================
# run_ablation.sh
# 消融实验：DEIA_no_intent + DEIA_no_priority
# 全 5 layout × 2 position × 5 episode
# ============================================================
set -euo pipefail

HORIZON=400
EPISODE=5
LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit")
GPT_MODEL="Qwen/Qwen2.5-7B-Instruct"
PROMPT_LEVEL="l2-ap"
RETRIVAL_METHOD="recent_k"
K=1
BELIEF_REVISION=false

if [[ ! -f "main.py" ]]; then
  echo "[ERROR] 请在 src/ 目录下运行此脚本"; exit 1
fi

BATCH_TS="$(date +'%Y%m%d_%H%M%S')"
BASE_DIR="experiments/batch_${BATCH_TS}_ablation_H${HORIZON}_E${EPISODE}"
mkdir -p "${BASE_DIR}"
TABLE_FILE="${BASE_DIR}/results_table.txt"
LOG_FILE="${BASE_DIR}/run.log"

extract_mean_std () {
  /home/aj/miniconda3/envs/proagent/bin/python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"{float(d.get('mean_result',float('nan'))):.1f}±{float(d.get('std_result',float('nan'))):.1f}")
except: print("N/A")
PY
}

log () { echo "$*" | tee -a "${LOG_FILE}"; }

{
  echo "========== 消融实验结果汇总 =========="
  echo "模型: ${GPT_MODEL}  Horizon: ${HORIZON}  Episode: ${EPISODE}"
  echo ""
  printf "%-24s %-18s %-10s %-12s %-14s\n" "Layout" "Agent" "P0/P1" "Position" "Mean±Std"
  printf "%-24s %-18s %-10s %-12s %-14s\n" "------" "-----" "----" "--------" "--------"
} > "${TABLE_FILE}"

run_one () {
  local layout="$1" p0="$2" p1="$3" tag="$4"
  local log_dir="${BASE_DIR}/${layout}/${tag}"
  mkdir -p "${log_dir}"

  local cmd=(/home/aj/miniconda3/envs/proagent/bin/python main.py
    --layout "${layout}" --p0 "${p0}" --p1 "${p1}"
    --horizon "${HORIZON}" --episode "${EPISODE}"
    --save True --log_dir "${log_dir}"
    --gpt_model "${GPT_MODEL}" --prompt_level "${PROMPT_LEVEL}"
    --retrival_method "${RETRIVAL_METHOD}" --K "${K}"
    --belief_revision "${BELIEF_REVISION}"
  )

  log "[RUN] ${layout} | ${p0} vs ${p1} | tag=${tag} | $(date)"
  local start_ts=$SECONDS
  "${cmd[@]}" >> "${log_dir}/stdout.log" 2>&1
  local elapsed=$(( SECONDS - start_ts ))
  log "  Done: ${elapsed}s"

  local json score="N/A"
  json="$(ls -1 "${log_dir}"/results*.json 2>/dev/null | head -n1 || true)"
  [[ -n "${json}" ]] && score="$(extract_mean_std "${json}")"

  local agent="${p0}"; [[ "${p0}" == "BC" ]] && agent="${p1}"
  printf "%-24s %-18s %-10s %-12s %-14s\n" \
    "${layout}" "${agent}" "${p0}" "${tag}" "${score}" >> "${TABLE_FILE}"
  log "  Score: ${score}"
}

TOTAL=$(( ${#LAYOUTS[@]} * 4 ))
CURRENT=0

for layout in "${LAYOUTS[@]}"; do
  for variant in "DEIA_no_intent" "DEIA_no_priority"; do
    CURRENT=$(( CURRENT + 1 ))
    log ">>> [${CURRENT}/${TOTAL}] ${variant} P0 | ${layout}"
    run_one "${layout}" "${variant}" "BC" "${variant}_p0"

    CURRENT=$(( CURRENT + 1 ))
    log ">>> [${CURRENT}/${TOTAL}] ${variant} P1 | ${layout}"
    run_one "${layout}" "BC" "${variant}" "${variant}_p1"
  done
done

log ""; log "实验完成：$(date)"; log "结果：${BASE_DIR}"
cat "${TABLE_FILE}" | tee -a "${LOG_FILE}"
