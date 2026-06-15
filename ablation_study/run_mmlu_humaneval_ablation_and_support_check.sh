#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RANDOM_SEEDS="${RANDOM_SEEDS:-42 43 44}"
GPU_COOLDOWN_SEC="${GPU_COOLDOWN_SEC:-8}"
ABLATION_VARIANT_COOLDOWN_SEC="${ABLATION_VARIANT_COOLDOWN_SEC:-1}"
SUPPORT_METHOD_COOLDOWN_SEC="${SUPPORT_METHOD_COOLDOWN_SEC:-1}"
RUN_FULL_ABLATION="${RUN_FULL_ABLATION:-1}"
RUN_SUPPORT_CHECK="${RUN_SUPPORT_CHECK:-1}"
LIMIT_CASES="${LIMIT_CASES:-}"
ABLATION_CHUNK_SIZE="${ABLATION_CHUNK_SIZE:-}"
SUPPORT_CHUNK_SIZE="${SUPPORT_CHUNK_SIZE:-}"
HUMANEVAL_CHUNK_SIZE="${HUMANEVAL_CHUNK_SIZE:-}"

ABLATION_OUT="${ABLATION_OUT:-ablation_study/results/ccr_ablation_mmlu_humaneval_${TIMESTAMP}}"
SUPPORT_OUT="${SUPPORT_OUT:-ablation_study/results/ccr_support_variant_mmlu_humaneval_${TIMESTAMP}}"

export ABLATION_VARIANT_COOLDOWN_SEC
export SUPPORT_METHOD_COOLDOWN_SEC

FULL_ABLATION_TESTBEDS=(
  "mmlu:progressive_refinement:shuffle:main"
  "mmlu:progressive_refinement:noshuffle:main"
  "mmlu:parallel_exploration:shuffle:main"
  "mmlu:parallel_exploration:noshuffle:main"
  "humaneval:progressive_refinement:shuffle:main"
  "humaneval:progressive_refinement:noshuffle:main"
  "humaneval:parallel_exploration:shuffle:main"
  "humaneval:parallel_exploration:noshuffle:main"
)

SUPPORT_CHECK_TESTBEDS=(
  "mmlu:progressive_refinement:shuffle:main"
  "mmlu:progressive_refinement:noshuffle:main"
  "mmlu:parallel_exploration:shuffle:stability"
  "mmlu:parallel_exploration:noshuffle:stability"
  "humaneval:progressive_refinement:shuffle:stability"
  "humaneval:parallel_exploration:shuffle:stability"
)

cleanup_after_step() {
  local label="$1"
  echo "[cleanup] ${label}: Python process has exited; cooling GPU for ${GPU_COOLDOWN_SEC}s."
  source /root/anaconda3/etc/profile.d/conda.sh
  conda activate Judge
  python - <<'PY' || true
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
PY
  sleep "${GPU_COOLDOWN_SEC}"
}

append_limit_args() {
  if [[ -n "${LIMIT_CASES}" ]]; then
    printf '%s\n' "--limit-cases" "${LIMIT_CASES}"
  fi
}

append_humaneval_chunk_args() {
  if [[ -n "${HUMANEVAL_CHUNK_SIZE}" ]]; then
    printf '%s\n' "--humaneval-chunk-size" "${HUMANEVAL_CHUNK_SIZE}"
  fi
}

append_ablation_chunk_args() {
  if [[ -n "${ABLATION_CHUNK_SIZE}" ]]; then
    printf '%s\n' "--chunk-size" "${ABLATION_CHUNK_SIZE}"
  fi
}

append_support_chunk_args() {
  if [[ -n "${SUPPORT_CHUNK_SIZE}" ]]; then
    printf '%s\n' "--chunk-size" "${SUPPORT_CHUNK_SIZE}"
  fi
}

if [[ "${RUN_FULL_ABLATION}" == "1" ]]; then
  echo "[1/2] Running full component ablations for MMLU + HumanEval."
  for testbed in "${FULL_ABLATION_TESTBEDS[@]}"; do
    echo "  - ${testbed}"
    mapfile -t LIMIT_ARGS < <(append_limit_args)
    mapfile -t ABLATION_CHUNK_ARGS < <(append_ablation_chunk_args)
    bash ablation_study/run_ccr_ablation_split_variants.sh \
      --output-root "${ABLATION_OUT}" \
      --reports-dir "${ABLATION_OUT}" \
      --testbed "${testbed}" \
      --random-shortlist-seeds "${RANDOM_SEEDS}" \
      --include-optional-payload-statistics-ablation \
      "${LIMIT_ARGS[@]}" \
      "${ABLATION_CHUNK_ARGS[@]}"
    cleanup_after_step "full ablation ${testbed}"
  done
else
  echo "[1/2] Skipping full component ablations because RUN_FULL_ABLATION=${RUN_FULL_ABLATION}."
fi

if [[ "${RUN_SUPPORT_CHECK}" == "1" ]]; then
  echo "[2/2] Running w/o Support upgrade check for MMLU + HumanEval."
  for testbed in "${SUPPORT_CHECK_TESTBEDS[@]}"; do
    echo "  - ${testbed}"
    mapfile -t LIMIT_ARGS < <(append_limit_args)
    mapfile -t SUPPORT_CHUNK_ARGS < <(append_support_chunk_args)
    mapfile -t HUMANEVAL_CHUNK_ARGS < <(append_humaneval_chunk_args)
    bash ablation_study/run_support_variant_check.sh \
      --testbed "${testbed}" \
      --output-root "${SUPPORT_OUT}" \
      --reports-dir "${SUPPORT_OUT}" \
      --include-random \
      --random-seeds "${RANDOM_SEEDS}" \
      "${LIMIT_ARGS[@]}" \
      "${SUPPORT_CHUNK_ARGS[@]}" \
      "${HUMANEVAL_CHUNK_ARGS[@]}"
    cleanup_after_step "support check ${testbed}"
  done
else
  echo "[2/2] Skipping support-variant checks because RUN_SUPPORT_CHECK=${RUN_SUPPORT_CHECK}."
fi

echo "Full ablation outputs: ${ABLATION_OUT}"
echo "Support-variant outputs: ${SUPPORT_OUT}"
