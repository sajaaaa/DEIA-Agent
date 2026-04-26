#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 对比实验脚本：比较 ProAgent vs LLM_Bayesian vs ITDP
# 
# 用法：
#   ./run_comparison.sh
# ============================================================

HORIZON=400
EPISODE=3

LAYOUTS=(
  "cramped_room"
  # "asymmetric_advantages"
  # "coordination_ring"
  # "forced_coordination"
  # "counter_circuit"
)

OPPONENTS=(
  "SP"
  # "PBT"
  # "FCP"
)

# 三种Agent类型进行对比
AGENTS=(
  "ProAgent"
  "LLM_Bayesian"
  "ITDP"
)

# LLM配置
GPT_MODEL="Qwen/Qwen2.5-7B-Instruct"
PROMPT_LEVEL="l2-ap"
RETRIVAL_METHOD="recent_k"
K=1
BELIEF_REVISION=false

if [[ ! -f "main.py" ]]; then
  echo "[ERROR] main.py not found. Please run this script in the directory that contains main.py."
  exit 1
fi

BATCH_TS="$(date +'%Y%m%d_%H%M%S')"
BASE_DIR="experiments/comparison_${BATCH_TS}_H${HORIZON}_E${EPISODE}"
mkdir -p "${BASE_DIR}"

TABLE_FILE="${BASE_DIR}/comparison_results.txt"

# Header
{
  echo "========== Agent Comparison Results =========="
  echo "Comparing: ProAgent vs LLM_Bayesian vs ITDP"
  echo ""
  printf "%-22s %-12s %-12s %-12s %-10s\n" "Layout" "Agent" "Opponent" "Position" "Score"
  printf "%-22s %-12s %-12s %-12s %-10s\n" "------" "-----" "--------" "--------" "-----"
} > "${TABLE_FILE}"

# Helper function to extract mean_result
extract_mean () {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    v = d.get("mean_result", None)
    if v is None:
        print("N/A")
    else:
        print(f"{float(v):.1f}")
except Exception:
    print("N/A")
PY
}

run_one () {
  local layout="$1"
  local agent="$2"
  local opponent="$3"
  local position="$4"  # p0 or p1

  local log_dir="${BASE_DIR}/${layout}/${agent}_vs_${opponent}_${position}"
  mkdir -p "${log_dir}"

  local cmd
  if [[ "${position}" == "p0" ]]; then
    cmd=(python main.py
      --layout "${layout}"
      --p0 "${agent}"
      --p1 "${opponent}"
      --horizon "${HORIZON}"
      --episode "${EPISODE}"
      --save True
      --log_dir "${log_dir}"
    )
  else
    cmd=(python main.py
      --layout "${layout}"
      --p0 "${opponent}"
      --p1 "${agent}"
      --horizon "${HORIZON}"
      --episode "${EPISODE}"
      --save True
      --log_dir "${log_dir}"
    )
  fi

  # Add LLM params for agents that need them
  if [[ "${agent}" == "ProAgent" || "${agent}" == "ITDP" || "${agent}" == "LLM_Bayesian" ]]; then
    cmd+=( --gpt_model "${GPT_MODEL}"
           --prompt_level "${PROMPT_LEVEL}"
           --retrival_method "${RETRIVAL_METHOD}"
           --K "${K}"
           --belief_revision "${BELIEF_REVISION}" )
  fi

  echo "============================================================"
  echo "[RUN] layout=${layout} agent=${agent} vs opponent=${opponent} position=${position}"
  echo "${cmd[@]}"
  "${cmd[@]}"

  local results_json
  results_json="$(ls -1 "${log_dir}"/results*.json 2>/dev/null | head -n 1 || true)"

  local result="N/A"
  if [[ -n "${results_json}" ]]; then
    result="$(extract_mean "${results_json}")"
  fi

  printf "%-22s %-12s %-12s %-12s %-10s\n" "${layout}" "${agent}" "${opponent}" "${position}" "${result}" >> "${TABLE_FILE}"
}

echo "Starting comparison experiments..."
echo "BASE_DIR=${BASE_DIR}"
echo "Results will be saved to: ${TABLE_FILE}"
echo

for layout in "${LAYOUTS[@]}"; do
  for opponent in "${OPPONENTS[@]}"; do
    for agent in "${AGENTS[@]}"; do
      # Run as P0
      run_one "${layout}" "${agent}" "${opponent}" "p0"
      # Run as P1
      run_one "${layout}" "${agent}" "${opponent}" "p1"
    done
  done
done

echo ""
echo "========== FINAL RESULTS =========="
cat "${TABLE_FILE}"
echo ""
echo "[DONE] Results saved to: ${TABLE_FILE}"
