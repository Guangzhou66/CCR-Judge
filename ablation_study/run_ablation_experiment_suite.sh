#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"

DEFAULT_ENV="/pychen/Anaconda3/envs/Judge"
DEFAULT_MODEL="/pychen/Test/model/Qwen2.5-7B-Instruct"
DEFAULT_GSM8K_DATASET_ROOT="${PROJECT_ROOT}/data/gsm8k"
DEFAULT_DATASETS="mmlu humaneval"
DEFAULT_REGIMES="parallel_exploration progressive_refinement"
DEFAULT_ORDERINGS="noshuffle shuffle"
DEFAULT_RANDOM_SEEDS="42 43 44"
DEFAULT_RESULT_ROOT="${SCRIPT_DIR}/results/paper_ablation_qwen_$(date +%Y%m%d_%H%M%S)"
RUN_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ORIGINAL_COMMAND="bash $0 $*"

usage() {
  cat <<'EOF'
Usage:
  bash ablation_study/run_ablation_experiment_suite.sh [options]

Runs the paper-ready CCR-Judge ablation suite:
  Layer 1: fixed-slate component ablation
  Layer 2: online MMLU validation
  Layer 3: variant upgrade check against Dense/Naive/KVCOMM/PAL-KV

Options:
  --env PATH
  --llm-name PATH
  --suite-root PATH
  --datasets STR             Default: "mmlu humaneval"; add gsm8k when resources allow.
  --gsm8k-dataset-root PATH  Default: data/gsm8k under this repository.
  --regimes STR              Default: "parallel_exploration progressive_refinement"
  --orderings STR            Default: "noshuffle shuffle"
  --random-shortlist-seeds STR
  --limit-cases INT          Cap fixed-slate candidate packs and runs.
  --online-limit-cases INT   Cap online MMLU validation.
  --skip-build               Use existing candidate_cache under suite root.
  --skip-fixed               Skip fixed-slate ablation.
  --skip-baselines           Skip Dense/Naive/KVCOMM/PAL-KV upgrade baselines.
  --skip-online              Skip online MMLU validation.
  --smoke                    Tiny end-to-end check: mmlu, one regime, noshuffle, one case.
  --dry-run                  Print commands without executing.
  --cuda-visible-devices STR Default: 0.
  -h, --help

Formal minimum accepted run:
  bash ablation_study/run_ablation_experiment_suite.sh --datasets "mmlu humaneval"

Ideal run including GSM8K:
  bash ablation_study/run_ablation_experiment_suite.sh --datasets "mmlu gsm8k humaneval"
EOF
}

ENV_PATH="${DEFAULT_ENV}"
LLM_NAME="${DEFAULT_MODEL}"
GSM8K_DATASET_ROOT="${DEFAULT_GSM8K_DATASET_ROOT}"
SUITE_ROOT="${DEFAULT_RESULT_ROOT}"
DATASETS="${DEFAULT_DATASETS}"
REGIMES="${DEFAULT_REGIMES}"
ORDERINGS="${DEFAULT_ORDERINGS}"
RANDOM_SEEDS="${DEFAULT_RANDOM_SEEDS}"
LIMIT_CASES=""
ONLINE_LIMIT_CASES=""
SKIP_BUILD=0
SKIP_FIXED=0
SKIP_BASELINES=0
SKIP_ONLINE=0
SMOKE=0
DRY_RUN=0
CUDA_VISIBLE_DEVICES_VALUE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_PATH="$2"; shift 2 ;;
    --llm-name) LLM_NAME="$2"; shift 2 ;;
    --gsm8k-dataset-root) GSM8K_DATASET_ROOT="$2"; shift 2 ;;
    --suite-root) SUITE_ROOT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --regimes) REGIMES="$2"; shift 2 ;;
    --orderings) ORDERINGS="$2"; shift 2 ;;
    --random-shortlist-seeds) RANDOM_SEEDS="$2"; shift 2 ;;
    --limit-cases) LIMIT_CASES="$2"; shift 2 ;;
    --online-limit-cases) ONLINE_LIMIT_CASES="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-fixed) SKIP_FIXED=1; shift ;;
    --skip-baselines) SKIP_BASELINES=1; shift ;;
    --skip-online) SKIP_ONLINE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${SMOKE}" -eq 1 ]]; then
  DATASETS="mmlu"
  REGIMES="parallel_exploration"
  ORDERINGS="noshuffle"
  LIMIT_CASES="${LIMIT_CASES:-1}"
  ONLINE_LIMIT_CASES="${ONLINE_LIMIT_CASES:-1}"
  RANDOM_SEEDS="42"
fi

PYTHON_BIN="${ENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export KVCOMM_FORCE_BFLOAT16="${KVCOMM_FORCE_BFLOAT16:-1}"
export KVCOMM_PROGRESS_ONLY="${KVCOMM_PROGRESS_ONLY:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_cmd() {
  if [[ -n "${COMMAND_LOG:-}" ]]; then
    {
      printf '$'
      printf ' %q' "$@"
      printf '\n'
    } >> "${COMMAND_LOG}"
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a REGIME_ARRAY <<< "${REGIMES}"
read -r -a ORDERING_ARRAY <<< "${ORDERINGS}"

if [[ -d "${SUITE_ROOT}" ]] && [[ -n "$(find "${SUITE_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] Refusing to write into non-empty suite root: ${SUITE_ROOT}" >&2
  echo "[ERROR] Choose a new --suite-root to avoid overwriting existing results." >&2
  exit 1
fi

mkdir -p "${SUITE_ROOT}/candidate_cache" "${SUITE_ROOT}/fixed_slate" "${SUITE_ROOT}/variant_upgrade_baselines" "${SUITE_ROOT}/online_mmlu"
COMMAND_LOG="${SUITE_ROOT}/commands.log"

git_snapshot="not_a_git_repository"
if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_snapshot="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
gpu_info="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | paste -sd ';' - || true)"
if [[ -z "${gpu_info}" ]]; then
  gpu_info="unavailable"
fi

cat > "${SUITE_ROOT}/ablation_suite_manifest.md" <<EOF
# CCR-Judge Paper Ablation Suite Manifest

- Model: \`${LLM_NAME}\`
- Suite root: \`${SUITE_ROOT}\`
- Start UTC: \`${RUN_START_UTC}\`
- Command: \`${ORIGINAL_COMMAND}\`
- Git snapshot: \`${git_snapshot}\`
- GPU: \`${gpu_info}\`
- Datasets: \`${DATASETS}\`
- Regimes: \`${REGIMES}\`
- Orderings: \`${ORDERINGS}\`
- Random seeds: \`${RANDOM_SEEDS}\`
- Limit cases: \`${LIMIT_CASES:-none}\`
- Online limit cases: \`${ONLINE_LIMIT_CASES:-none}\`
EOF
cp "${SUITE_ROOT}/ablation_suite_manifest.md" "${SUITE_ROOT}/manifest.md"

cat > "${SUITE_ROOT}/config.yaml" <<EOF
model: "${LLM_NAME}"
suite_root: "${SUITE_ROOT}"
start_utc: "${RUN_START_UTC}"
command: "${ORIGINAL_COMMAND}"
git_snapshot: "${git_snapshot}"
gpu: "${gpu_info}"
datasets: "${DATASETS}"
regimes: "${REGIMES}"
orderings: "${ORDERINGS}"
random_seeds: "${RANDOM_SEEDS}"
limit_cases: "${LIMIT_CASES:-}"
online_limit_cases: "${ONLINE_LIMIT_CASES:-}"
skip_build: ${SKIP_BUILD}
skip_fixed: ${SKIP_FIXED}
skip_baselines: ${SKIP_BASELINES}
skip_online: ${SKIP_ONLINE}
smoke: ${SMOKE}
dry_run: ${DRY_RUN}
EOF

build_pack() {
  local dataset="$1"
  local regime="$2"
  local out_dir="${SUITE_ROOT}/candidate_cache/${dataset}"
  local out_path="${out_dir}/${dataset}_${regime}.json"
  mkdir -p "${out_dir}"
  if [[ -f "${out_path}" ]]; then
    echo "[SKIP] existing pack ${out_path}"
    return 0
  fi
  local limit_args=()
  if [[ -n "${LIMIT_CASES}" ]]; then
    limit_args=(--limit_questions "${LIMIT_CASES}")
  fi
  case "${dataset}" in
    mmlu)
      run_cmd "${PYTHON_BIN}" "${EXPERIMENTS_DIR}/build_mmlu_paper_repair_candidates.py" \
        --llm_name "${LLM_NAME}" \
        --generation_regime "${regime}" \
        --index_file "${EXPERIMENTS_DIR}/mmlu_153_seed888_indices.json" \
        "${limit_args[@]}" \
        --split val \
        --agent_name AnalyzeAgent \
        --agent_role "MMLU Solver" \
        --agent_temperature 0.2 \
        --agent_count 4 \
        --seed 888 \
        --output_path "${out_path}"
      ;;
    humaneval)
      run_cmd "${PYTHON_BIN}" "${EXPERIMENTS_DIR}/build_humaneval_paper_repair_candidates.py" \
        --llm_name "${LLM_NAME}" \
        --generation_regime "${regime}" \
        "${limit_args[@]}" \
        --split test \
        --agent_name CodeWriting \
        --agent_role "Programming Expert" \
        --agent_temperature 0.2 \
        --agent_count 4 \
        --seed 42 \
        --output_path "${out_path}"
      ;;
    gsm8k)
      run_cmd "${PYTHON_BIN}" "${EXPERIMENTS_DIR}/build_gsm8k_paper_repair_candidates.py" \
        --llm_name "${LLM_NAME}" \
        --generation_regime "${regime}" \
        --dataset_root "${GSM8K_DATASET_ROOT}" \
        --config_name main \
        "${limit_args[@]}" \
        --split test \
        --agent_name MathSolver \
        --agent_role "Math Solver" \
        --agent_temperature 0.2 \
        --agent_count 4 \
        --seed 42 \
        --output_path "${out_path}"
      ;;
    *)
      echo "[ERROR] Unsupported dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  for dataset in "${DATASET_ARRAY[@]}"; do
    for regime in "${REGIME_ARRAY[@]}"; do
      echo "[BUILD] dataset=${dataset} regime=${regime}"
      build_pack "${dataset}" "${regime}"
    done
  done
fi

if [[ "${SKIP_BASELINES}" -eq 0 ]]; then
  for dataset in "${DATASET_ARRAY[@]}"; do
    for regime in "${REGIME_ARRAY[@]}"; do
      pack_path="${SUITE_ROOT}/candidate_cache/${dataset}/${dataset}_${regime}.json"
      for ordering in "${ORDERING_ARRAY[@]}"; do
        out_dir="${SUITE_ROOT}/variant_upgrade_baselines/${dataset}/${regime}/${ordering}"
        cmd=(
          "${PYTHON_BIN}"
          "${EXPERIMENTS_DIR}/evaluate_paper_repair_judge.py"
          --frozen_pack "${pack_path}"
          --output_dir "${out_dir}"
          --llm_name "${LLM_NAME}"
          --methods naive_reuse kvcomm pal_kv
        )
        if [[ "${ordering}" == "shuffle" ]]; then
          cmd+=(--judge-shuffle)
        fi
        echo "[BASELINES] dataset=${dataset} regime=${regime} ordering=${ordering}"
        run_cmd "${cmd[@]}"
      done
    done
  done
fi

if [[ "${SKIP_FIXED}" -eq 0 ]]; then
  testbed_args=()
  for dataset in "${DATASET_ARRAY[@]}"; do
    for regime in "${REGIME_ARRAY[@]}"; do
      for ordering in "${ORDERING_ARRAY[@]}"; do
        testbed_args+=(--testbed "${dataset}:${regime}:${ordering}:main")
      done
    done
  done
  fixed_cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_ccr_ablation.py"
    --result-root "${SUITE_ROOT}"
    --output-root "${SUITE_ROOT}/fixed_slate"
    --reports-dir "${SUITE_ROOT}/fixed_slate_reports"
    --llm-name "${LLM_NAME}"
    --random-shortlist-seeds "${RANDOM_SEEDS}"
    --include-optional-payload-statistics-ablation
    "${testbed_args[@]}"
  )
  if [[ -n "${LIMIT_CASES}" ]]; then
    fixed_cmd+=(--limit-cases "${LIMIT_CASES}")
  fi
  echo "[FIXED-SLATE] running component ablation"
  run_cmd "${fixed_cmd[@]}"
fi

if [[ "${SKIP_ONLINE}" -eq 0 ]]; then
  online_variant_array=(full no_support no_shortlist no_payload_stats stats_only)
  read -r -a ONLINE_RANDOM_SEED_ARRAY <<< "${RANDOM_SEEDS}"
  for seed in "${ONLINE_RANDOM_SEED_ARRAY[@]}"; do
    online_variant_array+=("random${seed}")
  done
  online_cmd=(
    bash "${SCRIPT_DIR}/run_mmlu_ori_ablation.sh"
    --llm-name "${LLM_NAME}"
    --result-root "${SUITE_ROOT}/online_mmlu"
    --random-shortlist-seeds "${RANDOM_SEEDS}"
    --variants "${online_variant_array[*]}"
  )
  if [[ -n "${ONLINE_LIMIT_CASES}" ]]; then
    if [[ "${ONLINE_LIMIT_CASES}" == "1" ]]; then
      online_cmd+=(--question-indices "91")
    else
      online_cmd+=(--limit-cases "${ONLINE_LIMIT_CASES}")
    fi
  fi
  if [[ "${SMOKE}" -eq 1 ]]; then
    online_cmd+=(--judge-shuffles "noshuffle")
  fi
  echo "[ONLINE] running MMLU online validation"
  run_cmd "${online_cmd[@]}"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_ablation_experiments.py" \
    --suite-root "${SUITE_ROOT}" \
    --fixed-root "${SUITE_ROOT}/fixed_slate" \
    --baseline-root "${SUITE_ROOT}/variant_upgrade_baselines" \
    --online-root "${SUITE_ROOT}/online_mmlu" \
    --model "${LLM_NAME}" \
    --expected-datasets "${DATASETS}" \
    --expected-regimes "${REGIMES}" \
    --expected-orderings "${ORDERINGS}"
fi

RUN_END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat >> "${SUITE_ROOT}/manifest.md" <<EOF

## Completion

- End UTC: \`${RUN_END_UTC}\`
- Key report: \`${SUITE_ROOT}/ablation_experiment_report.md\`
- Command log: \`${SUITE_ROOT}/commands.log\`
EOF

cat <<EOF

[DONE] CCR-Judge paper ablation suite launcher completed.

Suite root:
  ${SUITE_ROOT}

Key report:
  ${SUITE_ROOT}/ablation_experiment_report.md
EOF
