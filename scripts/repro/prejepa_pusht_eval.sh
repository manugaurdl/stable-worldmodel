#!/usr/bin/env bash
# Reproduce: PreJEPA (DINO-WM repro) PushT eval (world-model MPC).
#   model = pusht_dinov2_small_psmall @ weights_epoch_10.pt
#   eval  = pusht-wm-cem   (scripts/plan/eval_wm.py + config/pusht.yaml defaults)
#   seed  = 42             runtime ≈ 8728 s (≈2.4 h) — EXPENSIVE (trap #3)
# Authoritative result lives in research/models.md / reproduction-log.md.
set -euo pipefail

# --- cross-node env: loads scripts/hosts/$SWM_HOST.sh (default = hostname -s) ---
# Override the machine with SWM_HOST=<name>; see scripts/env.sh.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$SWM_REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${DATASET:-pusht_expert_train.h5}"    # eval_ff is Lance-unsafe, but eval_wm.py is fine on either
CKPT="${CKPT:-pusht_dinov2_small_psmall/weights_epoch_10.pt}"   # "last" = highest epoch

echo "[repro] PreJEPA pusht-wm-cem | host=$SWM_HOST  STABLEWM_HOME=$STABLEWM_HOME  ckpt=$CKPT  dataset=$DATASET  seed=42"
echo "[repro] NOTE: this eval takes ~2.4 h. Consider 'bf16=true' / 'compile=true' to speed up (changes the result id)."
"$PY" scripts/plan/eval_wm.py \
    policy="$CKPT" \
    eval.dataset_name="$DATASET"

# Result + videos: $STABLEWM_HOME/checkpoints/pusht_dinov2_small_psmall/pusht_results.txt
# Expected (from result file, not re-verified): success_rate = 48.0% on h5, weights_epoch_10.
