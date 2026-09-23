#!/usr/bin/env bash
# Copy local file(s) to a remote directory on ParaCloud. usage: pcput.sh <file...> <remote-dir>
set -euo pipefail
BIND="$(ipconfig getifaddr en0 || true)"
n=$#; dest="${!n}"; files=("${@:1:$((n-1))}")
exec scp ${BIND:+-o BindAddress="$BIND"} -o ConnectTimeout=60 -q "${files[@]}" "paracloud:$dest"
