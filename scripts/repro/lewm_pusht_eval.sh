#!/usr/bin/env bash
# Reproduce: LeWM PushT eval (world-model MPC).
#   model = quentinll/lewm-pusht @ weights.pt
#   eval  = pusht-wm-cem   (scripts/plan/eval_wm.py + config/pusht.yaml defaults)
#   seed  = 42             runtime ≈ 57 s
# Authoritative result lives in research/models.md / reproduction-log.md.
set -euo pipefail

# --- env (override STABLEWM_HOME / CUDA_VISIBLE_DEVICES if your host differs) ---
export STABLEWM_HOME="${STABLEWM_HOME:-/nas/manu/stable_worldmodel}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl

# --- repo root + venv ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

DATASET="${DATASET:-pusht_expert_train.h5}"   # set =pusht_expert_train.lance for the Lance run

echo "[repro] LeWM pusht-wm-cem  | STABLEWM_HOME=$STABLEWM_HOME  dataset=$DATASET  seed=42"
python scripts/plan/eval_wm.py \
    policy=quentinll/lewm-pusht \
    eval.dataset_name="$DATASET"

# Result + videos: $STABLEWM_HOME/checkpoints/quentinll/pusht_results.txt
# Expected (from result file, not re-verified): success_rate = 92.0% on h5.
# If the first run throws a KeyError right after decompressing the h5, just re-run (trap #4).
