#!/usr/bin/env bash
# ============================================================
# run_itdp_tonight.sh
# ITDP+BC 全 5 layout × 5 episode（纯规则，不调用 LLM，速度快）
# 作为 DEIA+BC 的对照组，量化 LLM 贡献
#
# 用法：
#   cd src
#   bash run_itdp_tonight.sh
# ============================================================
set -euo pipefail

HORIZON=400
EPISODE=5

LAYOUTS=(
  "cramped_room"
  "asymmetric_advantages"
  "coordination_ring"
  "forced_coordination"
  "counter_circuit"
)

if [[ ! -f "main.py" ]]; then
  echo "[ERROR] 请在 src/ 目录下运行此脚本"
  exit 1
fi

BATCH_TS="$(date +'%Y%m%d_%H%M%S')"
BASE_DIR="experiments/batch_${BATCH_TS}_ITDP_full_H${HORIZON}_E${EPISODE}"
mkdir -p "${BASE_DIR}"

TABLE_FILE="${BASE_DIR}/results_table.txt"
LOG_FILE="${BASE_DIR}/run.log"

extract_mean_std () {
  local json_path="$1"
  /home/aj/miniconda3/envs/proagent/bin/python - "$json_path" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    mean = d.get("mean_result", float("nan"))
    std  = d.get("std_result",  float("nan"))
    print(f"{float(mean):.1f}±{float(std):.1f}")
except Exception:
    print("N/A")
PY
}

log () { echo "$*" | tee -a "${LOG_FILE}"; }

{
  echo "========== ITDP 实验结果汇总（对照组）=========="
  echo "Agent: ITDP (纯规则, 无LLM)  Horizon: ${HORIZON}  Episode: ${EPISODE}"
  echo ""
  printf "%-24s %-10s %-10s %-10s %-14s\n" \
         "Layout" "P0" "P1" "Position" "Mean±Std"
  printf "%-24s %-10s %-10s %-10s %-14s\n" \
         "------" "--" "--" "--------" "--------"
} > "${TABLE_FILE}"

run_one () {
  local layout="$1"
  local p0="$2"
  local p1="$3"
  local tag="$4"

  local log_dir="${BASE_DIR}/${layout}/${tag}"
  mkdir -p "${log_dir}"

  local cmd=(/home/aj/miniconda3/envs/proagent/bin/python main.py
    --layout "${layout}"
    --p0 "${p0}" --p1 "${p1}"
    --horizon "${HORIZON}"
    --episode "${EPISODE}"
    --save True
    --log_dir "${log_dir}"
  )

  log ""
  log "============================================================"
  log "[RUN] ${layout} | P0=${p0} P1=${p1} | tag=${tag}"
  log "  Started: $(date)"

  local start_ts=$SECONDS
  "${cmd[@]}" >> "${log_dir}/stdout.log" 2>&1
  local elapsed=$(( SECONDS - start_ts ))

  log "  Finished: $(date) | elapsed: ${elapsed}s"

  local json
  json="$(ls -1 "${log_dir}"/results*.json 2>/dev/null | head -n1 || true)"
  local score="N/A"
  [[ -n "${json}" ]] && score="$(extract_mean_std "${json}")"

  printf "%-24s %-10s %-10s %-10s %-14s\n" \
         "${layout}" "${p0}" "${p1}" "${tag}" "${score}" >> "${TABLE_FILE}"

  log "  Score: ${score}"
}

TOTAL_GROUPS=$(( ${#LAYOUTS[@]} * 2 ))
CURRENT=0

for layout in "${LAYOUTS[@]}"; do

  CURRENT=$(( CURRENT + 1 ))
  log ">>> [${CURRENT}/${TOTAL_GROUPS}] ITDP as P0 | ${layout}"
  run_one "${layout}" "ITDP" "BC" "ITDP_p0"

  CURRENT=$(( CURRENT + 1 ))
  log ">>> [${CURRENT}/${TOTAL_GROUPS}] ITDP as P1 | ${layout}"
  run_one "${layout}" "BC" "ITDP" "ITDP_p1"

done

log ""
log "=========================================="
log "实验全部完成：$(date)"
log "结果目录：${BASE_DIR}"
log "=========================================="
cat "${TABLE_FILE}" | tee -a "${LOG_FILE}"
log ""
log "完整日志：${LOG_FILE}"
