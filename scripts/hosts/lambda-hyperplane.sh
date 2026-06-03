#!/usr/bin/env bash
# Host: lambda-hyperplane -- Lambda box, user manu, 4x A100-SXM4-80GB. `hostname -s`
# = 'lambda-hyperplane' (unlike the two lambda-scalar boxes), so it resolves without
# a ~/.swm_host marker; we drop one anyway for parity.
#
# Like lambda-74, the worldmodel cache + conda live on /data_new/manu (NOT /nas, which
# is NOT mounted here) -- so DO NOT source manu.sh/lambda-74.sh: their PY (repo .venv)
# and DINOENV (/nas/manu/miniconda3) point at paths that don't exist on this box. This
# is a standalone config.
#
# PY: there is no repo .venv on this box yet, so PY is pointed at the dino_wm conda
# python purely to satisfy env.sh's "PY must be executable" gate (DINO-WM training uses
# DINOENV, not PY). The swm eval/MPC scripts need the real `stable_worldmodel` package
# -- build the repo .venv (`uv sync --all-extras`) and repoint PY here before running
# those.
#
# Sourced by scripts/env.sh.

export STABLEWM_HOME="/data_new/manu/stable_worldmodel"                  # /nas not mounted here
export DINO="$SWM_REPO_ROOT/dino_wm"                                     # vendored original DINO-WM code
export DINOENV="/data_new/manu/miniconda3/envs/dino_wm/bin/python"       # DINOv2/torch2.3/accelerate (py3.10)
export PY="$DINOENV"                                                     # placeholder: no swm .venv yet (see header)

export NGPU="${NGPU:-4}"
export GPUS="${GPUS:-0,1,2,3}"
