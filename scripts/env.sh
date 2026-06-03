#!/usr/bin/env bash
# ============================================================================
# Cross-node environment loader for stable-worldmodel research scripts.
#
# WHY: this repo was authored on host `manu` with absolute paths baked into the
# scripts. To run on any other machine, ALL host-specific facts live in exactly
# one place per host -- scripts/hosts/<host>.sh -- and everything else sources
# this file and uses the exported vars below. Never hardcode an absolute /home
# or /nas path anywhere except scripts/hosts/<host>.sh.
#
# USAGE (top of every repo run script, after `set -euo pipefail`):
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"   # from scripts/*/
#   then use the exported vars: $PY $STABLEWM_HOME $DINO $NGPU $GPUS
#
# PICK THE MACHINE (priority order):
#   1. SWM_HOST=<name>  env var  ->  loads scripts/hosts/<name>.sh   (explicit; wins)
#   2. ~/.swm_host  marker file  ->  one line naming this host       (per machine)
#   3. $(hostname -s)            ->  convenience fallback            (e.g. trinity-0-3)
# The marker exists because the two lambda boxes BOTH report `hostname -s` =
# 'lambda-scalar'; a one-line ~/.swm_host names each one so a bare
#   bash scripts/repro/dinowm_orig_pusht_train.sh
# works anywhere. Set it once per box:  echo lambda-75 > ~/.swm_host
#
# Override any single var per-invocation by exporting it before sourcing, e.g.
#   STABLEWM_HOME=/tmp/scratch GPUS=0,1 bash scripts/repro/<x>.sh
#
# Exports (host config MUST set STABLEWM_HOME + PY; the rest are optional):
#   SWM_REPO_ROOT  repo checkout root (derived here, never hardcoded)
#   SWM_HOST       selected host name
#   STABLEWM_HOME  cache root (datasets/, checkpoints/)         [required]
#   PY             python interpreter for swm + data scripts    [required]
#   DINO           vendored original DINO-WM code dir           [optional]
#   DINOENV        python for the DINO-WM conda env (DINOv2)    [optional]
#   NGPU, GPUS     GPU count + CUDA_VISIBLE_DEVICES list        [optional]
#   SDL_VIDEODRIVER=dummy  MUJOCO_GL=egl   (headless rendering, set here)
# ============================================================================

# Repo root from THIS file's location (scripts/env.sh -> repo root). Never hardcoded.
SWM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SWM_REPO_ROOT

# Select host: explicit SWM_HOST wins, else the ~/.swm_host marker, else hostname.
# The two lambda boxes share `hostname -s` = 'lambda-scalar', so drop a one-line
# marker naming this machine once per box (trinity nodes resolve by hostname):
#     echo lambda-75 > ~/.swm_host    # the /nas/manu box
#     echo lambda-74 > ~/.swm_host    # the /data_new/manu box
SWM_HOST="${SWM_HOST:-$(cat "$HOME/.swm_host" 2>/dev/null || true)}"
SWM_HOST="${SWM_HOST:-$(hostname -s)}"
export SWM_HOST

_swm_host_cfg="$SWM_REPO_ROOT/scripts/hosts/${SWM_HOST}.sh"
if [[ ! -f "$_swm_host_cfg" ]]; then
  {
    echo "[env.sh] No host config for SWM_HOST='$SWM_HOST'."
    echo "         set it via 'echo <name> > ~/.swm_host' (or SWM_HOST=<name>)."
    echo "         expected: scripts/hosts/${SWM_HOST}.sh"
    echo "         available hosts (pick one with SWM_HOST=<name>):"
    for _h in "$SWM_REPO_ROOT"/scripts/hosts/*.sh; do
      [[ -e "$_h" ]] && echo "           - $(basename "$_h" .sh)"
    done
    echo "         on a NEW machine: add scripts/hosts/<name>.sh (see scripts/hosts/manu.sh)."
  } >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$_swm_host_cfg"

# Host-independent: headless rendering for PushT / mujoco.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Enforce the contract every host config must satisfy.
if [[ -z "${STABLEWM_HOME:-}" || -z "${PY:-}" ]]; then
  echo "[env.sh] scripts/hosts/${SWM_HOST}.sh must export STABLEWM_HOME and PY." >&2
  return 1 2>/dev/null || exit 1
fi
export STABLEWM_HOME PY
if [[ ! -d "$STABLEWM_HOME" ]]; then
  echo "[env.sh] STABLEWM_HOME=$STABLEWM_HOME does not exist on host '$SWM_HOST'." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "[env.sh] PY=$PY is not executable on host '$SWM_HOST'." >&2
  return 1 2>/dev/null || exit 1
fi

echo "[env.sh] host=$SWM_HOST  STABLEWM_HOME=$STABLEWM_HOME  PY=$PY  NGPU=${NGPU:-?}  GPUS=${GPUS:-}" >&2
