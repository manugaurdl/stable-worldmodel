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

DINO=/home/manu/stable-worldmodel/dino_wm
BIN=/nas/manu/miniconda3/envs/dino_wm/bin
RUN=/nas/manu/stable_worldmodel/dino_wm_runs/outputs/dinowm_orig_pusht_f5h3

NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
EPOCHS=${EPOCHS:-100}

export DATASET_DIR=/nas/manu/stable_worldmodel/datasets
export SDL_VIDEODRIVER=dummy
cd "$DINO"

CUDA_VISIBLE_DEVICES=$GPUS $BIN/accelerate launch --multi_gpu --num_processes "$NGPU" "$DINO/train.py" \
  --config-name train.yaml env=pusht frameskip=5 num_hist=3 num_pred=1 \
  has_decoder=False model.train_decoder=False training.batch_size=32 training.epochs="$EPOCHS" \
  env.num_workers=8 \
  env.dataset._target_=datasets.pusht_dset.load_pusht_feat_slice_train_val \
  hydra.run.dir="$RUN"
# checkpoints -> $RUN/checkpoints/model_{epoch}.pth (+ model_latest.pth); train cfg -> $RUN/hydra.yaml
