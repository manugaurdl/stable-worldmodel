#!/usr/bin/env bash
# Reproduce: LeWM PushT eval (world-model MPC).
#   model = quentinll/lewm-pusht @ weights.pt
#   eval  = pusht-wm-cem   (scripts/plan/eval_wm.py + config/pusht.yaml defaults)
#   seed  = 42             runtime ≈ 57 s
# Authoritative result lives in research/models.md / reproduction-log.md.
set -euo pipefail

# --- cross-node env: loads scripts/hosts/$SWM_HOST.sh (default = hostname -s) ---
# Override the machine with SWM_HOST=<name>; see scripts/env.sh.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$SWM_REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${DATASET:-pusht_expert_train.h5}"   # set =pusht_expert_train.lance for the Lance run

echo "[repro] LeWM pusht-wm-cem  | host=$SWM_HOST  STABLEWM_HOME=$STABLEWM_HOME  dataset=$DATASET  seed=42"
"$PY" scripts/plan/eval_wm.py \
    policy=quentinll/lewm-pusht \
    eval.dataset_name="$DATASET"

# Result + videos: $STABLEWM_HOME/checkpoints/quentinll/pusht_results.txt
# Expected (from result file, not re-verified): success_rate = 92.0% on h5.
# If the first run throws a KeyError right after decompressing the h5, just re-run (trap #4).
