#!/usr/bin/env bash
# Data prep for the original-DINO-WM PushT reproduction (model id: dinowm_orig_pusht_f5h3).
#
# Two steps, both validated bit-exact:
#   1. Convert our expert h5 -> DINO-WM's pusht_noise layout (lossless mp4 + .pth tensors).
#   2. Precompute frozen DINOv2 ViT-S/14 patch features (256? no: 196x384) once, into
#      /dev/shm (RAM) so training reads them at RAM speed instead of decoding mp4 +
#      running the (frozen) encoder every epoch. Features are mathematically identical to
#      on-the-fly encode_obs (frozen encoder; verified maxdiff ~fp16 rounding).
#
# NOTE: features live in tmpfs (/dev/shm/pusht_feats, ~393 GB) and are VOLATILE — lost on
# reboot or if /dev/shm is cleared. Re-run step 2 to regenerate (~10 min on 8 GPUs).
# The split dirs' feats/ are symlinks into /dev/shm.
set -euo pipefail

VENV=/home/manu/stable-worldmodel/.venv/bin/python          # has h5py/decord/imageio-ffmpeg
DINOENV=/nas/manu/miniconda3/envs/dino_wm/bin/python        # has DINOv2/torch (py3.10)
SCRIPTS=/home/manu/stable-worldmodel/scripts/data
P=/nas/manu/stable_worldmodel/datasets/pusht_noise

# --- 1. convert h5 -> pusht_noise/{train,val} (one-time; ~5 min, ~7.6 GB) ---
$VENV $SCRIPTS/convert_pusht_h5_to_dinowm.py --dst "$P" --workers 8
# After this, PATCH dino_wm/datasets/pusht_dset.py:16-19 with the printed STATE/PROPRIO
# stats (repro patch D3) and :110 -> :05d (D2). (Already applied in this checkout.)

# --- 2. precompute DINOv2 features into /dev/shm (8 GPUs; ~10 min, ~393 GB tmpfs) ---
rm -rf "$P/train/feats" "$P/val/feats"
mkdir -p /dev/shm/pusht_feats/train /dev/shm/pusht_feats/val
ln -sfn /dev/shm/pusht_feats/train "$P/train/feats"
ln -sfn /dev/shm/pusht_feats/val   "$P/val/feats"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $DINOENV $SCRIPTS/precompute_dino_feats.py --split train --num-shards 8 --shard $i &
done; wait
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $DINOENV $SCRIPTS/precompute_dino_feats.py --split val --num-shards 8 --shard $i &
done; wait
echo "train feats: $(ls /dev/shm/pusht_feats/train | wc -l)/16816  val: $(ls /dev/shm/pusht_feats/val | wc -l)/1869"
