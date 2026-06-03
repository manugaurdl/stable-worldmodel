#!/usr/bin/env bash
# Train DINO-WM (ORIGINAL gaoyuezhou/dino_wm code) on our PushT expert data.
# Model id: dinowm_orig_pusht_f5h3_<timestamp>  (wandb: manugaur/stable-wm; run name ==
#   model_name == run-dir basename. New run each launch; pass RUN=<dir> to resume — see below.)
#
# Faithful recipe: frozen DINOv2 ViT-S/14, ViT causal predictor (depth6/heads16/mlp2048),
# global batch 32, predictor lr 5e-4, frameskip 5, num_hist 3, num_pred 1, 100 epochs.
# has_decoder=False  -> predictor-only = the paper's headline (0.92 "w/o decoder loss")
#   config; the decoder loss uses DETACHED latents so it never affects the predictor, and
#   it would OOM val on a 49GB GPU. Also matches the in-house PreJEPA config (fair compare).
# Feature mode is toggled by PRECOMP (patch D12): PRECOMP=1 (default) reads precomputed
#   frozen DINOv2 features (see dinowm_orig_pusht_dataprep.sh); PRECOMP=0 encodes online
#   (decode mp4 + DINOv2 forward every batch). Frozen encoder => the two are mathematically
#   identical; precomp is a throughput optimization only.
#
# Prereq (PRECOMP=1 only): run dinowm_orig_pusht_dataprep.sh first (features in /dev/shm;
#   /dev/shm is tmpfs, so re-run after a reboot). PRECOMP=0 needs no prep.
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

# Default = a NEW run every launch (unique timestamped dir -> new wandb id, train from
# scratch). To RESUME an existing run instead, pass its dir:
#   RUN=$STABLEWM_HOME/dino_wm_runs/outputs/dinowm_orig_pusht_f5h3_20260603_143000 \
#     bash scripts/repro/dinowm_orig_pusht_train.sh
# (train.py reuses that dir's hydra.yaml wandb_run_id + checkpoints/model_latest.pth.)
RUN=${RUN:-$STABLEWM_HOME/dino_wm_runs/outputs/dinowm_orig_pusht_f5h3_$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-100}
PRECOMP=${PRECOMP:-1}   # 1 -> precomputed feats; 0 -> encode DINOv2 online
[ "$PRECOMP" = "0" ] && PRECOMP_FEAT=False || PRECOMP_FEAT=True

export DATASET_DIR="$STABLEWM_HOME/datasets"
cd "$DINO"

CUDA_VISIBLE_DEVICES=$GPUS "$ACCELERATE" launch --multi_gpu --num_processes "$NGPU" "$DINO/train.py" \
  --config-name train.yaml env=pusht frameskip=5 num_hist=3 num_pred=1 \
  has_decoder=False model.train_decoder=False training.batch_size=64 training.epochs="$EPOCHS" \
  env.num_workers=8 \
  precomp_feat="$PRECOMP_FEAT" \
  hydra.run.dir="$RUN"
# checkpoints -> $RUN/checkpoints/model_{epoch}.pth (+ model_latest.pth); train cfg -> $RUN/hydra.yaml
