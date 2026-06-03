#!/usr/bin/env bash
# Host: lambda-75 -- the Lambda box at 69.30.0.75 (user manu, 8x RTX 6000 Ada).
# Data root is /nas/manu, identical to the original `manu` dev config -- so just
# source it. Select with:  echo lambda-75 > ~/.swm_host
#
# Sourced by scripts/env.sh.
# shellcheck disable=SC1091
source "$SWM_REPO_ROOT/scripts/hosts/manu.sh"
