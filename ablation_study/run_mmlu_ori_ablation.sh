#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"

DEFAULT_ENV="/pychen/Anaconda3/envs/Judge"
DEFAULT_MODEL="/pychen/Test/model/Qwen2.5-7B-Instruct"
DEFAULT_INDEX_FILE="${EXPERIMENTS_DIR}/mmlu_153_seed888_indices.json"
DEFAULT_RESULT_ROOT="${SCRIPT_DIR}/results/ori_mmlu_ablation_$(date +%Y%m%d_%H%M%S)"
DEFAULT_JUDGE_SHUFFLES="noshuffle shuffle"
DEFAULT_RANDOM_SEEDS="42 43 44"
DEFAULT_SMOKE_INDICES="91 160 545"

usage() {
  cat <<'EOF'
Usage:
  bash ablation_study/run_mmlu_ori_ablation.sh [options]

Runs CCR-Judge component ablations under the MMLU Ori online protocol.

Default matrix:
  - dataset: MMLU val, fixed mmlu_153_seed888_indices.json subset
  - protocol: ori_style_online_protocol
  - topology: FullConnected
  - execution mode: allow_kv_reuse
  - compare dense: enabled
  - shuffles: noshuffle shuffle
  - variants: Full, w/o Comparative Grouping, w/o Support Scoring,
              w/o Shortlist Compression, w/o Comparative Payload Stats,
              Stats Only, Shortlist Only, Random Shortlist seed=42/43/44

Options:
  --env PATH                 Conda env path. Default: /pychen/Anaconda3/envs/Judge
  --llm-name PATH            Model path. Default: /pychen/Test/model/Qwen2.5-7B-Instruct
  --result-root PATH         Output root. Default: ablation_study/results/ori_mmlu_ablation_<timestamp>
  --index-file PATH          MMLU index JSON. Default: experiments/mmlu_153_seed888_indices.json
  --question-indices STR     Space-separated question indices. Overrides --index-file.
  --limit-cases INT          Cap selected indices before running.
  --judge-shuffles STR       Space-separated: noshuffle shuffle. Default: both.
  --variants STR             Space-separated variant keys. Overrides defaults.
                             Keys: full no_grouping no_support no_shortlist
                                   no_payload_stats stats_only shortlist_only
                                   random42 random43 random44
  --random-shortlist-seeds STR
                             Space-separated random seeds from: 42 43 44. Default: "42 43 44".
  --no-random                Do not include random shortlist variants.
  --include-optional-payload-statistics-ablation
                             Include w/o Comparative Payload Statistics.
  --cuda-visible-devices STR CUDA_VISIBLE_DEVICES value. Default: 0.
  --smoke                    Run three built-in smoke indices.
  --dry-run                  Print commands without running them.
  -h, --help                 Show this help.

Examples:
  bash ablation_study/run_mmlu_ori_ablation.sh --smoke --dry-run

  bash ablation_study/run_mmlu_ori_ablation.sh \
    --llm-name /pychen/Test/model/Qwen2.5-7B-Instruct \
    --result-root ablation_study/results/ori_mmlu_qwen25_7b

  bash ablation_study/run_mmlu_ori_ablation.sh \
    --variants "full no_support no_shortlist" \
    --judge-shuffles "shuffle"
EOF
}

ENV_PATH="${DEFAULT_ENV}"
LLM_NAME="${DEFAULT_MODEL}"
RESULT_ROOT="${DEFAULT_RESULT_ROOT}"
INDEX_FILE="${DEFAULT_INDEX_FILE}"
QUESTION_INDICES=""
LIMIT_CASES=""
JUDGE_SHUFFLES="${DEFAULT_JUDGE_SHUFFLES}"
VARIANTS=""
RANDOM_SEEDS="${DEFAULT_RANDOM_SEEDS}"
INCLUDE_RANDOM=1
INCLUDE_OPTIONAL=0
CUDA_VISIBLE_DEVICES_VALUE="0"
SMOKE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_PATH="$2"
      shift 2
      ;;
    --llm-name)
      LLM_NAME="$2"
      shift 2
      ;;
    --result-root)
      RESULT_ROOT="$2"
      shift 2
      ;;
    --index-file)
      INDEX_FILE="$2"
      shift 2
      ;;
    --question-indices)
      QUESTION_INDICES="$2"
      shift 2
      ;;
    --limit-cases)
      LIMIT_CASES="$2"
      shift 2
      ;;
    --judge-shuffles)
      JUDGE_SHUFFLES="$2"
      shift 2
      ;;
    --variants)
      VARIANTS="$2"
      shift 2
      ;;
    --random-shortlist-seeds)
      RANDOM_SEEDS="$2"
      shift 2
      ;;
    --no-random)
      INCLUDE_RANDOM=0
      shift
      ;;
    --include-optional-payload-statistics-ablation)
      INCLUDE_OPTIONAL=1
      shift
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${ENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${EXPERIMENTS_DIR}/run_mmlu_ori_protocol.py" ]]; then
  echo "[ERROR] Missing runner: ${EXPERIMENTS_DIR}/run_mmlu_ori_protocol.py" >&2
  exit 1
fi
if [[ "${SMOKE}" -eq 1 ]]; then
  QUESTION_INDICES="${DEFAULT_SMOKE_INDICES}"
fi

if [[ -n "${QUESTION_INDICES}" ]]; then
  read -r -a QUESTION_INDEX_ARRAY <<< "${QUESTION_INDICES}"
else
  if [[ ! -f "${INDEX_FILE}" ]]; then
    echo "[ERROR] Index file not found: ${INDEX_FILE}" >&2
    exit 1
  fi
  mapfile -t QUESTION_INDEX_ARRAY < <(
    "${PYTHON_BIN}" - "${INDEX_FILE}" "${LIMIT_CASES}" <<'PY'
import json
import sys

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
payload = json.load(open(path, "r", encoding="utf-8"))
indices = payload.get("question_indices", payload if isinstance(payload, list) else [])
if limit is not None:
    indices = indices[:limit]
for index in indices:
    print(int(index))
PY
  )
fi

if [[ -n "${QUESTION_INDICES}" && -n "${LIMIT_CASES}" ]]; then
  QUESTION_INDEX_ARRAY=("${QUESTION_INDEX_ARRAY[@]:0:${LIMIT_CASES}}")
fi
if [[ "${#QUESTION_INDEX_ARRAY[@]}" -eq 0 ]]; then
  echo "[ERROR] No question indices selected." >&2
  exit 1
fi

read -r -a SHUFFLE_ARRAY <<< "${JUDGE_SHUFFLES}"

if [[ -n "${VARIANTS}" ]]; then
  read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
else
  VARIANT_ARRAY=(full no_grouping no_support no_shortlist no_payload_stats stats_only shortlist_only)
  if [[ "${INCLUDE_RANDOM}" -eq 1 ]]; then
    read -r -a RANDOM_SEED_ARRAY <<< "${RANDOM_SEEDS}"
    for seed in "${RANDOM_SEED_ARRAY[@]}"; do
      VARIANT_ARRAY+=("random${seed}")
    done
  fi
  if [[ "${INCLUDE_OPTIONAL}" -eq 1 ]]; then
    VARIANT_ARRAY+=(no_payload_stats)
  fi
fi

DEDUPED_VARIANTS=()
for variant in "${VARIANT_ARRAY[@]}"; do
  already_seen=0
  for existing in "${DEDUPED_VARIANTS[@]}"; do
    if [[ "${existing}" == "${variant}" ]]; then
      already_seen=1
      break
    fi
  done
  if [[ "${already_seen}" -eq 0 ]]; then
    DEDUPED_VARIANTS+=("${variant}")
  fi
done
VARIANT_ARRAY=("${DEDUPED_VARIANTS[@]}")

variant_decision_method() {
  case "$1" in
    full|ccr_full)
      echo "OriProtocolCCRJudgeAblationFullFinalSelect"
      ;;
    no_grouping|ccr_wo_comparative_grouping)
      echo "OriProtocolCCRJudgeNoComparativeGroupingFinalSelect"
      ;;
    no_support|ccr_wo_support_scoring)
      echo "OriProtocolCCRJudgeNoSupportScoringFinalSelect"
      ;;
    no_shortlist|ccr_wo_shortlist_compression)
      echo "OriProtocolCCRJudgeNoShortlistCompressionFinalSelect"
      ;;
    stats_only|ccr_stats_only)
      echo "OriProtocolCCRJudgeStatsOnlyFinalSelect"
      ;;
    shortlist_only|ccr_shortlist_only)
      echo "OriProtocolCCRJudgeShortlistOnlyFinalSelect"
      ;;
    no_payload_stats|ccr_wo_comparative_payload_statistics)
      echo "OriProtocolCCRJudgeNoComparativePayloadStatsFinalSelect"
      ;;
    random42|ccr_random_shortlist_seed42)
      echo "OriProtocolCCRJudgeRandomShortlistSeed42FinalSelect"
      ;;
    random43|ccr_random_shortlist_seed43)
      echo "OriProtocolCCRJudgeRandomShortlistSeed43FinalSelect"
      ;;
    random44|ccr_random_shortlist_seed44)
      echo "OriProtocolCCRJudgeRandomShortlistSeed44FinalSelect"
      ;;
    *)
      echo "[ERROR] Unsupported variant key: $1" >&2
      return 1
      ;;
  esac
}

variant_label() {
  case "$1" in
    full|ccr_full) echo "CCR-Judge (Full)" ;;
    no_grouping|ccr_wo_comparative_grouping) echo "CCR-Judge w/o Comparative Grouping" ;;
    no_support|ccr_wo_support_scoring) echo "CCR-Judge w/o Support Scoring" ;;
    no_shortlist|ccr_wo_shortlist_compression) echo "CCR-Judge w/o Shortlist Compression" ;;
    stats_only|ccr_stats_only) echo "CCR-Judge Stats Only" ;;
    shortlist_only|ccr_shortlist_only) echo "CCR-Judge Shortlist Only" ;;
    no_payload_stats|ccr_wo_comparative_payload_statistics) echo "CCR-Judge w/o Comparative Payload Statistics" ;;
    random42|ccr_random_shortlist_seed42) echo "CCR-Judge with Random Shortlist (seed=42)" ;;
    random43|ccr_random_shortlist_seed43) echo "CCR-Judge with Random Shortlist (seed=43)" ;;
    random44|ccr_random_shortlist_seed44) echo "CCR-Judge with Random Shortlist (seed=44)" ;;
    *) echo "$1" ;;
  esac
}

safe_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

mkdir -p "${RESULT_ROOT}"

cat > "${RESULT_ROOT}/ori_mmlu_ablation_manifest.md" <<EOF
# MMLU Ori-Protocol CCR Ablation Manifest

- Protocol: \`ori_style_online_protocol\`
- Dataset: \`mmlu/val\`
- Model: \`${LLM_NAME}\`
- Cases: \`${#QUESTION_INDEX_ARRAY[@]}\`
- Shuffles: \`${JUDGE_SHUFFLES}\`
- Variants: \`${VARIANT_ARRAY[*]}\`
- Result root: \`${RESULT_ROOT}\`
EOF

echo "[INFO] Result root: ${RESULT_ROOT}"
echo "[INFO] Selected cases: ${#QUESTION_INDEX_ARRAY[@]}"
echo "[INFO] Variants: ${VARIANT_ARRAY[*]}"
echo "[INFO] Shuffles: ${SHUFFLE_ARRAY[*]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

for shuffle_flag in "${SHUFFLE_ARRAY[@]}"; do
  case "${shuffle_flag}" in
    noshuffle|No|no|0)
      shuffle_label="noshuffle"
      shuffle_arg=()
      ;;
    shuffle|Yes|yes|1)
      shuffle_label="shuffle"
      shuffle_arg=(--judge-shuffle)
      ;;
    *)
      echo "[ERROR] Unsupported shuffle flag: ${shuffle_flag}" >&2
      exit 1
      ;;
  esac

  for variant in "${VARIANT_ARRAY[@]}"; do
    decision_method="$(variant_decision_method "${variant}")"
    label="$(variant_label "${variant}")"
    run_dir="${RESULT_ROOT}/ori_protocol/ablation/${shuffle_label}/$(safe_label "${label}")"
    cmd=(
      "${PYTHON_BIN}"
      "${EXPERIMENTS_DIR}/run_mmlu_ori_protocol.py"
      --llm_name "${LLM_NAME}"
      --output_dir "${run_dir}"
      --mode FullConnected
      --batch_size 1
      --decision_method "${decision_method}"
      --execution_mode allow_kv_reuse
      --judge-compare-dense
      --limit_questions "${#QUESTION_INDEX_ARRAY[@]}"
      --question-indices "${QUESTION_INDEX_ARRAY[@]}"
      --agent-role "MMLU Solver"
      --agent-temperatures 0.2
      --judge-temperature 1.0
      --max-new-tokens 512
      --stop-conditions model_eos_only
      --kv-max-anchor-num 5
      --scorer-mode mmlu_text_match_parity_aligned
      --agent_names AnalyzeAgent
      --agent_nums 4
      --run_tag "${variant}"
      --no-judge-mask-prev-agent-attn
      --no-judge-group-agents
      --no-judge-group-agents-by-position
      "${shuffle_arg[@]}"
    )
    echo "[RUN] shuffle=${shuffle_label} variant=${label}"
    run_cmd "${cmd[@]}"
  done
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_mmlu_ori_ablation.py" \
    --result-root "${RESULT_ROOT}"
fi

cat <<EOF

[DONE] MMLU Ori-protocol ablation launcher completed.

Output root:
  ${RESULT_ROOT}

Key files:
  ${RESULT_ROOT}/ori_mmlu_ablation_manifest.md
  ${RESULT_ROOT}/ori_mmlu_ablation_summary.json
  ${RESULT_ROOT}/ori_mmlu_ablation_table.md
EOF
