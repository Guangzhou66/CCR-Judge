#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CCR_ABLATION_SKIP_CONDA:-0}" != "1" ]]; then
  if [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate Judge
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/result/paper_repair_formal_llama32_3b_20260421_164715}"
LLM_NAME="${LLM_NAME:-/pychen/Test/model/Llama-3.2-3B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export KVCOMM_FORCE_BFLOAT16="${KVCOMM_FORCE_BFLOAT16:-1}"
export KVCOMM_PROGRESS_ONLY="${KVCOMM_PROGRESS_ONLY:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export ABLATION_VARIANT_COOLDOWN_SEC="${ABLATION_VARIANT_COOLDOWN_SEC:-1}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA is not available for {sys.executable}. Activate the Judge env or set CCR_ABLATION_SKIP_CONDA=0."
    )
print(f"[run_ccr_ablation_split_variants] python={sys.executable}")
print(f"[run_ccr_ablation_split_variants] cuda_device={torch.cuda.get_device_name(0)}")
PY

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_ccr_ablation_split_variants.py" \
  --result-root "${RESULT_ROOT}" \
  --llm-name "${LLM_NAME}" \
  "$@"
