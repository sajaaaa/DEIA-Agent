#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 第二个点基线实验：DEIA vs 未知队友（SP / FCP / MEP）
# 目的：量化固定 BC 先验的 DEIA 在非 BC 队友上的得分下降幅度
#
# 实验设计：
#   - 5 layouts × 3 opponents (SP/FCP/MEP) × 2 positions × 5 ep
#   - 共 30 组，约 150 episode
#
# 输出：
#   experiments/batch_<ts>_DEIA_generalization_H400_E5/results_table.txt
# ============================================================

HORIZON=400
EPISODE=5

LAYOUTS=(
  "cramped_room"
  "asymmetric_advantages"
  "coordination_ring"
  "forced_coordination"
  "counter_circuit"
)

OPPONENTS=(
  "SP"
  "FCP"
  "MEP"
)

GPT_MODEL="Qwen/Qwen2.5-7B-Instruct"
PROMPT_LEVEL="l2-ap"
RETRIVAL_METHOD="recent_k"
K=1

if [[ ! -f "main.py" ]]; then
  echo "[ERROR] main.py not found. Run this script from the src/ directory."
  exit 1
fi

BATCH_TS="$(date +'%Y%m%d_%H%M%S')"
BASE_DIR="experiments/batch_${BATCH_TS}_DEIA_generalization_H${HORIZON}_E${EPISODE}"
mkdir -p "${BASE_DIR}"

TABLE_FILE="${BASE_DIR}/results_table.txt"

{
  echo "========== DEIA 泛化基线实验结果 =========="
  echo "模型: ${GPT_MODEL}  Horizon: ${HORIZON}  Episode: ${EPISODE}"
  echo ""
  printf "%-24s %-12s %-12s %-12s %-14s\n" "Layout" "P0" "P1" "DEIA_pos" "Mean±Std"
  printf "%-24s %-12s %-12s %-12s %-14s\n" "------" "--" "--" "--------" "--------"
} > "${TABLE_FILE}"

extract_result () {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    mean = d.get("mean_result", "N/A")
    std  = d.get("std_result",  "N/A")
    if mean != "N/A" and std != "N/A":
        print(f"{float(mean):.1f}±{float(std):.1f}")
    else:
        print("N/A")
except Exception:
    print("N/A")
PY
}

run_one () {
  local layout="$1"
  local p0="$2"
  local p1="$3"
  local deia_pos="$4"   # DEIA_p0 / DEIA_p1
  local opp="$5"

  local log_dir="${BASE_DIR}/${layout}/${opp}_${deia_pos}"
  mkdir -p "${log_dir}"

  cmd=(python3 main.py
    --layout "${layout}"
    --p0 "${p0}"
    --p1 "${p1}"
    --horizon "${HORIZON}"
    --episode "${EPISODE}"
    --save True
    --log_dir "${log_dir}"
    --gpt_model "${GPT_MODEL}"
    --prompt_level "${PROMPT_LEVEL}"
    --retrival_method "${RETRIVAL_METHOD}"
    --K "${K}"
  )

  echo "============================================================"
  echo "[RUN] layout=${layout}  p0=${p0}  p1=${p1}  (${deia_pos})"
  echo "${cmd[@]}"
  "${cmd[@]}"

  local results_json
  results_json="$(ls -1 "${log_dir}"/results*.json 2>/dev/null | head -n 1 || true)"

  local result="N/A"
  if [[ -n "${results_json}" ]]; then
    result="$(extract_result "${results_json}")"
  fi

  printf "%-24s %-12s %-12s %-12s %-14s\n" \
    "${layout}" "${p0}" "${p1}" "${deia_pos}" "${result}" >> "${TABLE_FILE}"
}

echo "Starting DEIA generalization baseline..."
echo "BASE_DIR=${BASE_DIR}"
echo "Opponents: ${OPPONENTS[*]}"
echo ""

for layout in "${LAYOUTS[@]}"; do
  for opp in "${OPPONENTS[@]}"; do
    run_one "${layout}" "DEIA" "${opp}" "DEIA_p0" "${opp}"
    run_one "${layout}" "${opp}" "DEIA" "DEIA_p1" "${opp}"
  done
done

echo ""
cat "${TABLE_FILE}"
echo ""
echo "[DONE] Table saved to: ${TABLE_FILE}"
