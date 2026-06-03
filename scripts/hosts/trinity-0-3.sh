#!/usr/bin/env bash
# Compute node trinity-0-3: 8x NVIDIA RTX 6000 Ada Generation (48 GB). Shares the CMU-site NFS
# filesystem with the trinity login node, so inherit its paths and override GPUs.
# Reach it with `ssh trinity-0-3` (or select from elsewhere with SWM_HOST=trinity-0-3).
#
# Sourced by scripts/env.sh.

# shellcheck disable=SC1091
source "$SWM_REPO_ROOT/scripts/hosts/trinity.sh"

export NGPU=8
export GPUS="0,1,2,3,4,5,6,7"
