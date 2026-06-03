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

# --- cross-node env: loads scripts/hosts/$SWM_HOST.sh (default = hostname -s) ---
# Run on a multi-GPU node:  SWM_HOST=trinity-0-3 bash scripts/repro/dinowm_orig_pusht_dataprep.sh
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

# $PY (.venv) has h5py/decord/imageio-ffmpeg for the convert step; $DINOENV (conda)
# has DINOv2/torch for the feature precompute.
if [[ -z "${DINOENV:-}" ]]; then
  echo "[dataprep] DINOENV not set for host '$SWM_HOST'. Need the dino_wm conda env" >&2
  echo "           (DINOv2 + torch); set DINOENV in scripts/hosts/${SWM_HOST}.sh." >&2
  echo "           See scripts/hosts/manu.sh for the original env." >&2
  exit 1
fi

SCRIPTS="$SWM_REPO_ROOT/scripts/data"
P="$STABLEWM_HOME/datasets/pusht_noise"

# --- 1. convert h5 -> pusht_noise/{train,val} (one-time; ~5 min, ~7.6 GB) ---
"$PY" "$SCRIPTS/convert_pusht_h5_to_dinowm.py" --dst "$P" --workers 8
# After this, PATCH dino_wm/datasets/pusht_dset.py:16-19 with the printed STATE/PROPRIO
# stats (repro patch D3) and :110 -> :05d (D2). (Already applied in this checkout.)

# --- 2. precompute DINOv2 features into /dev/shm ($NGPU GPUs; ~10 min, ~393 GB tmpfs) ---
rm -rf "$P/train/feats" "$P/val/feats"
mkdir -p /dev/shm/pusht_feats/train /dev/shm/pusht_feats/val
ln -sfn /dev/shm/pusht_feats/train "$P/train/feats"
ln -sfn /dev/shm/pusht_feats/val   "$P/val/feats"
IFS=',' read -ra _gpu <<< "$GPUS"            # CUDA_VISIBLE_DEVICES list from the host config
for split in train val; do
  pids=()
  for ((i=0; i<NGPU; i++)); do
    CUDA_VISIBLE_DEVICES="${_gpu[$i]}" "$DINOENV" "$SCRIPTS/precompute_dino_feats.py" \
      --split "$split" --num-shards "$NGPU" --shard "$i" &
    pids+=($!)
  done
  # wait on each PID so a failed shard aborts (plain `wait` returns 0 and hid this before)
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "[dataprep] a $split precompute shard (pid $pid) failed -- aborting" >&2; exit 1; }
  done
done
ntr=$(ls /dev/shm/pusht_feats/train | wc -l); nva=$(ls /dev/shm/pusht_feats/val | wc -l)
echo "train feats: $ntr/16816  val: $nva/1869"
[ "$ntr" -eq 16816 ] && [ "$nva" -eq 1869 ] || { echo "[dataprep] feature count mismatch (want 16816/1869) -- aborting" >&2; exit 1; }
echo "[dataprep] OK"
