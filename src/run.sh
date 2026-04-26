#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run all experiments (requirements unchanged):
# - ITDP (our agent / ProAgent) vs {SP, PBT, FCP, MEP, COLE}
# - 5 layouts
# - swap positions once per layout (ITDP as P0 and as P1)
# - episode=5, horizon=400
#
# Final output:
# - ONE summary table (no DBE column), printed to terminal AND saved to:
#     experiments/batch_.../results_table.txt
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
  # "SP"
  # "PBT"
  # "FCP"
  # "MEP"
  # "COLE"
  "BC"
)

# ITDP args (defaults shown here, change if needed)
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
BASE_DIR="experiments/batch_${BATCH_TS}_ITDP_vs_others_H${HORIZON}_E${EPISODE}"
mkdir -p "${BASE_DIR}"

TABLE_FILE="${BASE_DIR}/results_table.txt"

# Header (align with fixed-width columns)
{
  echo "========== 结果汇总 =========="
  printf "%-22s %-10s %-10s %-10s\n" "Layout" "P0" "P1" "Result"
} > "${TABLE_FILE}"

# helper: extract mean_result from results json, else N/A
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
        # keep it simple: print with minimal rounding (1 decimal like many papers)
        try:
            print(f"{float(v):.1f}")
        except Exception:
            print(str(v))
except Exception:
    print("N/A")
PY
}

run_one () {
  local layout="$1"
  local p0="$2"
  local p1="$3"
  local swap_tag="$4"   # itdp_as_p0 / itdp_as_p1

  local log_dir="${BASE_DIR}/${layout}/${swap_tag}_${p0}_vs_${p1}"
  mkdir -p "${log_dir}"

  cmd=(python main.py
    --layout "${layout}"
    --p0 "${p0}"
    --p1 "${p1}"
    --horizon "${HORIZON}"
    --episode "${EPISODE}"
    --save True
    --log_dir "${log_dir}"
  )

  # pass ITDP params when ITDP involved
  if [[ "${p0}" == "ITDP" || "${p1}" == "ITDP" ]]; then
    cmd+=( --gpt_model "${GPT_MODEL}"
           --prompt_level "${PROMPT_LEVEL}"
           --retrival_method "${RETRIVAL_METHOD}"
           --K "${K}"
           --belief_revision "${BELIEF_REVISION}" )
  fi

  echo "============================================================"
  echo "[RUN] layout=${layout} p0=${p0} p1=${p1} horizon=${HORIZON} episode=${EPISODE}"
  echo "${cmd[@]}"
  "${cmd[@]}"

  local results_json
  results_json="$(ls -1 "${log_dir}"/results*.json 2>/dev/null | head -n 1 || true)"

  local result="N/A"
  if [[ -n "${results_json}" ]]; then
    result="$(extract_mean "${results_json}")"
  fi

  # keep labels as-is in the table (show ITDP directly)
  printf "%-22s %-10s %-10s %-10s\n" "${layout}" "${p0}" "${p1}" "${result}" >> "${TABLE_FILE}"
}

echo "Starting batch experiments..."
echo "BASE_DIR=${BASE_DIR}"
echo "Results table will be saved to: ${TABLE_FILE}"
echo

for layout in "${LAYOUTS[@]}"; do
  for opp in "${OPPONENTS[@]}"; do
    # swap once on each layout:
    run_one "${layout}" "ProAgent" "${opp}" "itdp_as_p0"
    run_one "${layout}" "${opp}" "ProAgent" "itdp_as_p1"
  done
done

echo
cat "${TABLE_FILE}"
echo
echo "[DONE] Table saved to: ${TABLE_FILE}"
