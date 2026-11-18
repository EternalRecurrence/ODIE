#!/usr/bin/env bash
# One Tokyo no-router run and thirty DIV2K expert-count runs.
# The 16-expert comparison reuses the DIV2K main experiment.
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${ODIE_PYTHON:-python}" \
  scripts/run_ablations.py "$@"
