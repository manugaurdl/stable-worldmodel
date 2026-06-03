#!/usr/bin/env bash
# Native DINO-WM planning eval (eval id: pusht-dinowm-plan) for model dinowm_orig_pusht_f5h3.
# Uses the ORIGINAL repo's plan.py + conf/plan_pusht.yaml:
#   MPC-CEM (300 samples, 30 opt steps, top-30, horizon 5), n_evals=50, seed 99,
#   goal_source='dset' (goal = a val-trajectory state goal_H*frameskip=25 env-steps ahead),
#   success = block pos<20px AND angle<pi/9 at the final step (env.eval_state).
# This is DINO-WM's OWN protocol (the ~0.92 paper number). Distinct from swm pusht-wm-cem.
#
# n_plot_samples=0  -> no decoder visualization (we train has_decoder=False; decoder=None).
# WANDB_MODE=disabled -> plan.py forces wandb on in main(); disable to avoid plan-wandb runs.
# Goal/init obs come from the live PushT env (rollout), so this re-encodes images via the
# frozen DINOv2 at eval time (model patch D10 image branch) — precomputed feats unused here.
set -euo pipefail

# --- cross-node env: loads scripts/hosts/$SWM_HOST.sh (default = hostname -s) ---
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
export WANDB_MODE=disabled    # plan.py forces wandb on in main(); disable to avoid stray runs

# Native DINO-WM plan.py runs in the dino_wm conda env (DINOv2/torch), NOT the .venv.
if [[ -z "${DINOENV:-}" ]]; then
  echo "[plan] DINOENV not set for host '$SWM_HOST'. Set it in scripts/hosts/${SWM_HOST}.sh" >&2
  echo "       (see scripts/hosts/manu.sh)." >&2
  exit 1
fi

EPOCH=${EPOCH:-latest}
NEVALS=${NEVALS:-50}
GPU=${GPU:-0}

export DATASET_DIR="$STABLEWM_HOME/datasets"
cd "$DINO"

CUDA_VISIBLE_DEVICES=$GPU "$DINOENV" plan.py \
  --config-name plan_pusht.yaml \
  ckpt_base_path="$STABLEWM_HOME/dino_wm_runs" \
  model_name=dinowm_orig_pusht_f5h3 model_epoch="$EPOCH" \
  n_evals="$NEVALS" n_plot_samples=0
# results + logs -> dino_wm/plan_outputs/<ts>_dinowm_orig_pusht_f5h3_gH5/
