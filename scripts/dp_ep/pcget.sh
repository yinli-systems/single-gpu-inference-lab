#!/usr/bin/env bash
# Pull a remote path from ParaCloud. usage: pcget.sh <remote-path> <local-path>
set -euo pipefail
BIND="$(ipconfig getifaddr en0 || true)"
exec rsync -a ${BIND:+-e "ssh -b $BIND -o ConnectTimeout=60"} "paracloud:$1" "$2"
