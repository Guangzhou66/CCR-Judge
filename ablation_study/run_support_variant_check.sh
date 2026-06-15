#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${SUPPORT_VARIANT_SKIP_CONDA:-0}" != "1" ]]; then
  if [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
    # The base Python in this container is CPU-only; the Judge env exposes CUDA.
    # Set SUPPORT_VARIANT_SKIP_CONDA=1 if you intentionally manage the env outside this script.
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate Judge
  fi
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export KVCOMM_FORCE_BFLOAT16="${KVCOMM_FORCE_BFLOAT16:-1}"
export KVCOMM_PROGRESS_ONLY="${KVCOMM_PROGRESS_ONLY:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export SUPPORT_METHOD_COOLDOWN_SEC="${SUPPORT_METHOD_COOLDOWN_SEC:-1}"

if [[ "$#" -eq 0 ]]; then
  set -- --settings all
fi

python ablation_study/run_support_variant_check.py "$@"
