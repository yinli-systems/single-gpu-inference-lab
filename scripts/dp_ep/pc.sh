#!/usr/bin/env bash
# Run a command on the ParaCloud login node (bind-address required by this network).
set -uo pipefail
BIND="$(ipconfig getifaddr en0 || true)"
exec ssh ${BIND:+-b "$BIND"} -o ConnectTimeout=60 -o ServerAliveInterval=20 paracloud "$@"
