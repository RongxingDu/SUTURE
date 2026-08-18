#!/usr/bin/env bash
#
# One-click pipeline: run the full workflow optimization, then the official
# held-out test evaluation, against the same config/dataset.
#
# Usage:
#   bash experiments/scripts/run_optimize_and_test.sh \
#       --config experiments/configs/math_aliyun_qwen35_full.yaml \
#       --benchmark math \
#       --data /path/to/math_level5_four_domains_aflow_full.jsonl \
#       --api-key "$ALIYUN_API_KEY"
#
# Options:
#   --config PATH        Experiment YAML config (required)
#   --benchmark NAME     Benchmark type: code_gen|math|agent|gpqa|mmlu|scicode
#   --data PATH          Full ordered benchmark JSONL
#   --api-key KEY        API key; exported as ALIYUN_API_KEY. Falls back to the
#                        existing ALIYUN_API_KEY environment variable.
#   --workflow PATH      Initial workflow YAML (optional)
#   --skip-optimize      Skip optimization; only run held-out test on an
#                        existing results.json.
#   --skip-test          Only run optimization; do not run held-out test.
#   --dry-run            Print the commands that would run and exit.
#   --allow-local-code-execution / --use-bubblewrap-code-sandbox /
#   --allow-zero-hard-reward --scicode-hdf5 PATH --scicode-protocol NAME
#       Passthrough flags forwarded to both sub-scripts (see their docs).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG=""
BENCHMARK=""
DATA=""
API_KEY="${ALIYUN_API_KEY:-}"
WORKFLOW=""
SKIP_OPTIMIZE=0
SKIP_TEST=0
DRY_RUN=0

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --workflow) WORKFLOW="$2"; shift 2 ;;
        --skip-optimize) SKIP_OPTIMIZE=1; shift ;;
        --skip-test) SKIP_TEST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --allow-local-code-execution|--use-bubblewrap-code-sandbox|--allow-zero-hard-reward)
            EXTRA+=("$1"); shift ;;
        --scicode-hdf5|--scicode-protocol)
            EXTRA+=("$1" "$2"); shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            sed -n '2,30p' "$0" >&2
            exit 2 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required" >&2; exit 2
fi
if [[ -z "$BENCHMARK" ]]; then
    echo "ERROR: --benchmark is required" >&2; exit 2
fi
if [[ -z "$API_KEY" ]]; then
    echo "ERROR: --api-key is required (or set ALIYUN_API_KEY)" >&2; exit 2
fi
export ALIYUN_API_KEY="$API_KEY"

if [[ -f "$CONFIG" ]]; then
    CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
else
    echo "ERROR: config not found: $CONFIG" >&2; exit 2
fi

# Resolve the results directory from the config so the test step always reads
# the exact results.json produced by optimization.
RESULTS_DIR="$(python3 - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from awf.config.loader import load_config
cfg = load_config(Path(sys.argv[1]))
print(Path(cfg.output_dir) / cfg.name)
PY
)"
RESULTS_JSON="${RESULTS_DIR}/results.json"
mkdir -p "$RESULTS_DIR"

LOG_FILE="${RESULTS_DIR}/pipeline.log"
echo "==== AWF optimize + test pipeline ===="
echo "  config    : $CONFIG"
echo "  benchmark : $BENCHMARK"
echo "  data      : ${DATA:-<default>}"
echo "  results   : $RESULTS_JSON"
echo "  log       : $LOG_FILE"

run_step() {
    local label="$1"; shift
    echo "" | tee -a "$LOG_FILE"
    echo "==== $(date '+%F %T') $label ====" | tee -a "$LOG_FILE"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '  (dry-run) %q' "$1"; shift
        for arg in "$@"; do printf ' %q' "$arg"; done
        echo
        return 0
    fi
    "$@" | tee -a "$LOG_FILE"
}

if [[ "$SKIP_OPTIMIZE" -eq 1 ]]; then
    echo "Skipping optimization (--skip-optimize)." | tee -a "$LOG_FILE"
elif [[ -f "$RESULTS_JSON" ]] && grep -q '"test_metrics"' "$RESULTS_JSON"; then
    echo "results.json already contains test metrics; skipping optimization." \
        | tee -a "$LOG_FILE"
else
    OPT_ARGS=(--config "$CONFIG" --benchmark "$BENCHMARK")
    if [[ -n "$DATA" ]]; then OPT_ARGS+=(--data "$DATA"); fi
    if [[ -n "$WORKFLOW" ]]; then OPT_ARGS+=(--workflow "$WORKFLOW"); fi
    OPT_ARGS+=("${EXTRA[@]}")
    run_step "Optimization" python3 -m experiments.scripts.run_optimization \
        "${OPT_ARGS[@]}"
fi

if [[ "$SKIP_TEST" -eq 1 ]]; then
    echo "Skipping held-out test (--skip-test)." | tee -a "$LOG_FILE"
    echo "DONE (optimize only). Best checkpoint: ${RESULTS_JSON}" | tee -a "$LOG_FILE"
    exit 0
fi

if [[ ! -f "$RESULTS_JSON" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "results.json will be produced here after optimization: $RESULTS_JSON"
    else
        echo "ERROR: results.json not found after optimization: $RESULTS_JSON" >&2
        echo "       Re-run with --skip-test to debug the optimization step." >&2
        exit 1
    fi
fi

TEST_ARGS=(--results "$RESULTS_JSON" --config "$CONFIG" --benchmark "$BENCHMARK")
if [[ -n "$DATA" ]]; then TEST_ARGS+=(--data "$DATA"); fi
TEST_ARGS+=("${EXTRA[@]}")
run_step "Held-out test" python3 -m experiments.scripts.run_test "${TEST_ARGS[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "(dry-run) summary would be printed from: $RESULTS_JSON"
    exit 0
fi

echo "" | tee -a "$LOG_FILE"
echo "==== $(date '+%F %T') summary ====" | tee -a "$LOG_FILE"
python3 - "$RESULTS_JSON" <<'PY' | tee -a "$LOG_FILE"
import json, sys
res = json.load(open(sys.argv[1]))
print(f"best_val_score      : {res.get('best_val_score')}")
print(f"early_stop          : round={res.get('early_stop_round')} reason={res.get('early_stop_reason')}")
for r in res.get("rounds", []):
    print(f"round {r.get('round', '?')}: "
          f"candidate_val={r.get('val_score')}, accepted={r.get('accepted')}, "
          f"is_best={r.get('is_best')}")
test = res.get("official_test_metrics") or res.get("test_metrics") or {}
if test:
    print("---- held-out test ----")
    for k in ("hard_success_rate", "hard_reward", "composite_reward",
              "input_tokens", "output_tokens", "total_tokens", "llm_call_count"):
        if k in test:
            print(f"  {k:22s}: {test[k]}")
else:
    print("---- held-out test: NOT evaluated ----")
PY
echo "Full log: $LOG_FILE"
