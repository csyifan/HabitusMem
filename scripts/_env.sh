#!/usr/bin/env bash
# Shared defaults. Override any variable from the environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
export DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the local PersonaMem-v2 directory}"
export WORK="${WORK:-work}"
export SPLIT="${SPLIT:-splits/personamem_v2_train.json}"
export NPROC="${NPROC:-8}"
export GPU="${GPU:-0}"
