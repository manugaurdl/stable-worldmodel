#!/usr/bin/env bash
# Host: manu (original development machine). Preserved so manu-authored results
# stay reproducible. Do not run new CMU work under this config -- these paths do
# not exist on the trinity nodes (env.sh will fail loud if you pick it there).
#
# Sourced by scripts/env.sh.

export STABLEWM_HOME="/nas/manu/stable_worldmodel"             # home ~99% full; /nas is NFS
export PY="$SWM_REPO_ROOT/.venv/bin/python"                    # uv-managed py3.10 venv
export DINO="$SWM_REPO_ROOT/dino_wm"                           # vendored original DINO-WM code
export DINOENV="/nas/manu/miniconda3/envs/dino_wm/bin/python"  # DINOv2/torch/accelerate (py3.10)

export NGPU="${NGPU:-8}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
