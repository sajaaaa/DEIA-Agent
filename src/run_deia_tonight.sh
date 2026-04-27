#!/usr/bin/env bash
# ============================================================
# run_deia_tonight.sh
# 今晚实验：DEIA + BC 全 5 layout × 5 episode
#           ITDP  + BC 全 5 layout × 5 episode（纯规则，速度快）
#
# Baseline 说明：
#   BC+BC / Greedy+BC / SP/FCP/MEP/PBT/COLE 直接引用 ProAgent 论文数据
#   ProAgent+BC(Qwen) 若时间允许在此脚本末尾追加
#
# 用法：
#   cd src
#   bash run_deia_tonight.sh
#   bash run_deia_tonight.sh --with-proagent   # 额外跑 ProAgent+BC
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

GPT_MODEL="Qwen/Qwen2.5-7B-Instruct"
PROMPT_LEVEL="l2-ap"
RETRIVAL_METHOD="recent_k"
K=1
BELIEF_REVISION=false

# 是否额外跑 ProAgent
WITH_PROAGENT=false
for arg in "$@"; do
  [[ "$arg" == "--with-proagent" ]] && WITH_PROAGENT=true
done

if [[ ! -f "main.py" ]]; then
  echo "[ERROR] 请在 src/ 目录下运行此脚本"
  exit 1
fi

BATCH_TS="$(date +'%Y%m%d_%H%M%S')"
BASE_DIR="experiments/batch_${BATCH_TS}_DEIA_full_H${HORIZON}_E${EPISODE}"
mkdir -p "${BASE_DIR}"

TABLE_FILE="${BASE_DIR}/results_table.txt"
LOG_FILE="${BASE_DIR}/run.log"

# ── 工具函数 ────────────────────────────────────────────────

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

# ── 表头 ────────────────────────────────────────────────────
{
  echo "========== DEIA 实验结果汇总 =========="
  echo "模型: ${GPT_MODEL}  Horizon: ${HORIZON}  Episode: ${EPISODE}"
  echo ""
  printf "%-24s %-10s %-10s %-10s %-14s\n" \
         "Layout" "P0" "P1" "Position" "Mean±Std"
  printf "%-24s %-10s %-10s %-10s %-14s\n" \
         "------" "--" "--" "--------" "--------"
} > "${TABLE_FILE}"

# ── 单次实验 ────────────────────────────────────────────────
run_one () {
  local layout="$1"
  local p0="$2"
  local p1="$3"
  local tag="$4"    # 用于目录名

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

  # DEIA / ProAgent / ITDP 需要 LLM 参数
  if [[ "${p0}" == "DEIA" || "${p1}" == "DEIA" || \
        "${p0}" == "ProAgent" || "${p1}" == "ProAgent" || \
        "${p0}" == "ITDP" || "${p1}" == "ITDP" ]]; then
    cmd+=(
      --gpt_model "${GPT_MODEL}"
      --prompt_level "${PROMPT_LEVEL}"
      --retrival_method "${RETRIVAL_METHOD}"
      --K "${K}"
      --belief_revision "${BELIEF_REVISION}"
    )
  fi

  log ""
  log "============================================================"
  log "[RUN] ${layout} | P0=${p0} P1=${p1} | tag=${tag}"
  log "  ${cmd[*]}"
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

# ── 实验矩阵 ────────────────────────────────────────────────
#
# 每个 layout 跑 2 组：
#   DEIA as P0 vs BC
#   DEIA as P1 vs BC
#
# 时间估算（DEIA≈700s/ep）：
#   5 layout × 2 pos × 5 ep × 700s ≈ 9.7h
#   建议 nohup 后台运行

TOTAL_GROUPS=$(( ${#LAYOUTS[@]} * 2 ))
CURRENT=0

for layout in "${LAYOUTS[@]}"; do

  CURRENT=$(( CURRENT + 1 ))
  log ">>> [${CURRENT}/${TOTAL_GROUPS}] DEIA as P0 | ${layout}"
  run_one "${layout}" "DEIA" "BC" "DEIA_p0"

  CURRENT=$(( CURRENT + 1 ))
  log ">>> [${CURRENT}/${TOTAL_GROUPS}] DEIA as P1 | ${layout}"
  run_one "${layout}" "BC" "DEIA" "DEIA_p1"

done

# ── 可选：ProAgent + BC ──────────────────────────────────────
if [[ "${WITH_PROAGENT}" == "true" ]]; then
  log ""
  log ">>> [extra] ProAgent+BC on all layouts (--with-proagent)"
  for layout in "${LAYOUTS[@]}"; do
    run_one "${layout}" "ProAgent" "BC" "ProAgent_p0"
    run_one "${layout}" "BC" "ProAgent" "ProAgent_p1"
  done
fi

# ── 汇总输出 ────────────────────────────────────────────────
log ""
log "=========================================="
log "实验全部完成：$(date)"
log "结果目录：${BASE_DIR}"
log "=========================================="
cat "${TABLE_FILE}" | tee -a "${LOG_FILE}"
log ""
log "完整日志：${LOG_FILE}"
