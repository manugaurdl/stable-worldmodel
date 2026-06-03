#!/usr/bin/env bash
# Train DINO-WM (ORIGINAL gaoyuezhou/dino_wm code) on our PushT expert data.
# Model id: dinowm_orig_pusht_f5h3   (wandb: manugaur/stable-wm, run name == model_name)
#
# Faithful recipe: frozen DINOv2 ViT-S/14, ViT causal predictor (depth6/heads16/mlp2048),
# global batch 32, predictor lr 5e-4, frameskip 5, num_hist 3, num_pred 1, 100 epochs.
# has_decoder=False  -> predictor-only = the paper's headline (0.92 "w/o decoder loss")
#   config; the decoder loss uses DETACHED latents so it never affects the predictor, and
#   it would OOM val on a 49GB GPU. Also matches the in-house PreJEPA config (fair compare).
# Uses precomputed frozen DINOv2 features (see dinowm_orig_pusht_dataprep.sh) via
#   env.dataset._target_=...load_pusht_feat_slice_train_val  (model patch D10 + dataset D11).
#   Mathematically identical to on-the-fly encoding (frozen encoder).
#
# Prereq: run dinowm_orig_pusht_dataprep.sh first (features in /dev/shm).
set -euo pipefail

# --- cross-node env: loads scripts/hosts/$SWM_HOST.sh (default = hostname -s) ---
# On a multi-GPU node:  SWM_HOST=trinity-0-3 bash scripts/repro/dinowm_orig_pusht_train.sh
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

# DINO-WM training runs in the dino_wm conda env (DINOv2/torch/accelerate), NOT the .venv.
if [[ -z "${DINOENV:-}" ]]; then
  echo "[train] DINOENV not set for host '$SWM_HOST'. Create a dino_wm conda env (DINOv2 +" >&2
  echo "        torch + accelerate) and set DINOENV in scripts/hosts/${SWM_HOST}.sh." >&2
  echo "        See scripts/hosts/manu.sh for the original env." >&2
  exit 1
fi
ACCELERATE="$(dirname "$DINOENV")/accelerate"

RUN=${RUN:-$STABLEWM_HOME/dino_wm_runs/outputs/dinowm_orig_pusht_f5h3}
EPOCHS=${EPOCHS:-100}

export DATASET_DIR="$STABLEWM_HOME/datasets"
cd "$DINO"

CUDA_VISIBLE_DEVICES=$GPUS "$ACCELERATE" launch --multi_gpu --num_processes "$NGPU" "$DINO/train.py" \
  --config-name train.yaml env=pusht frameskip=5 num_hist=3 num_pred=1 \
  has_decoder=False model.train_decoder=False training.batch_size=32 training.epochs="$EPOCHS" \
  env.num_workers=8 \
  env.dataset._target_=datasets.pusht_dset.load_pusht_feat_slice_train_val \
  hydra.run.dir="$RUN"
# checkpoints -> $RUN/checkpoints/model_{epoch}.pth (+ model_latest.pth); train cfg -> $RUN/hydra.yaml
