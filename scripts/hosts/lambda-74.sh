#!/usr/bin/env bash
# Host: lambda-74 -- the Lambda box at 69.30.0.74 (user manu). Same home/.venv and
# (shared NFS) tooling as lambda-75; the ONLY difference is the worldmodel cache
# lives on local /data_new/manu instead of /nas/manu. Inherit lambda-75's env and
# override just the cache root. Select with:  echo lambda-74 > ~/.swm_host
#
# If /nas is NOT mounted here, the dino_wm conda env (DINOENV, set in manu.sh under
# /nas/manu/miniconda3) won't resolve -- override DINOENV below to this box's copy.
#
# Sourced by scripts/env.sh.
# shellcheck disable=SC1091
source "$SWM_REPO_ROOT/scripts/hosts/manu.sh"
export STABLEWM_HOME="/data_new/manu/stable_worldmodel"
