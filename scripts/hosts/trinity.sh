#!/usr/bin/env bash
# Host: trinity.vision.cs.cmu.edu (CMU vision lab), user mgaur.
# This is the LOGIN/head node -- no local GPUs. /data3 (NFS) and /home are shared
# with the trinity-0-* compute nodes, so those nodes inherit these paths and only
# override the GPU vars (see trinity-0-3.sh). To train, ssh to a GPU node first.
#
# Sourced by scripts/env.sh -- set vars only, no side effects.

export STABLEWM_HOME="/data3/mgaur/stable_worldmodel"   # NFS cache root (nebby.ib:/exports/data3)
                                                        # holds datasets/ + checkpoints/ + _backups/
export PY="$SWM_REPO_ROOT/.venv/bin/python"          # uv-managed py3.10 venv (swm + data scripts)
export DINO="$SWM_REPO_ROOT/dino_wm"                  # vendored original DINO-WM code (commit aac225c)

# GPUs: none on the login node. Compute nodes override these (e.g. trinity-0-3.sh).
export NGPU="${NGPU:-0}"
export GPUS="${GPUS:-}"

# DINO-WM training / feature-precompute runs in a SEPARATE conda env with DINOv2 +
# torch + accelerate (the .venv does not have them). Built from dino_wm/environment.yaml
# (`conda env create -f dino_wm/environment.yaml`) into NFS home, so it is shared across
# all CMU compute nodes. Verified 2026-06-03: torch 2.3.0+cu121, accelerate 0.26.1, CUDA x8.
export DINOENV="${DINOENV:-/home/mgaur/miniconda3/envs/dino_wm/bin/python}"
