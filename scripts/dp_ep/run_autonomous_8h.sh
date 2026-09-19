#!/usr/bin/env bash
# 8-hour headless Claude Code loop on the ParaCloud login node (run inside tmux).
# Each iteration is a fresh session bounded by `timeout`; state is carried by AUTONOMY_STATE.md.
set -Eeuo pipefail
ROOT=/data/run01/scxi253/multigpu-autonomy
REPO="$ROOT/single-gpu-inference-lab"
PROMPT="$REPO/AUTONOMY_PROMPT.md"
STATE="$REPO/AUTONOMY_STATE.md"
LOGDIR="$ROOT/claude-logs"
HOURS=${1:-8}
export PATH=/data/run01/scxi253/claude-cli/bin:/data/run01/scxi253/codex-node/bin:$PATH
export CLAUDE_CONFIG_DIR=/data/run01/scxi253/claude-home
export TMPDIR=/tmp/scxi253/tmp; mkdir -p "$TMPDIR" "$LOGDIR"
[ -f "$ROOT/env.sh" ] && source "$ROOT/env.sh"   # optional: ANTHROPIC_API_KEY=... (chmod 600)
cd "$REPO"
END=$(( $(date +%s) + HOURS*60*60 ))
while (( $(date +%s) < END )); do
  TS=$(date +%Y%m%d-%H%M%S)
  timeout 52m claude -p \
    --model opus \
    --max-turns 100 \
    --output-format stream-json --verbose \
    --allowedTools "Read" "Write" "Edit" "Glob" "Grep" \
      "Bash(git status:*)" "Bash(git diff:*)" "Bash(git log:*)" "Bash(git show:*)" "Bash(git add:*)" "Bash(git commit:*)" "Bash(git checkout:*)" "Bash(git switch:*)" \
      "Bash(sbatch:*)" "Bash(squeue:*)" "Bash(sacct:*)" "Bash(scancel:*)" "Bash(sinfo:*)" \
      "Bash(ls:*)" "Bash(cat:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(grep:*)" "Bash(wc:*)" "Bash(du:*)" "Bash(df:*)" "Bash(mkdir:*)" "Bash(cp:*)" "Bash(rsync:*)" "Bash(tar:*)" \
      "Bash(python:*)" "Bash(python3:*)" "Bash(/tmp/scxi253/venv-vllm/bin/python:*)" "Bash(pytest:*)" "Bash(curl:*)" "Bash(nvidia-smi:*)" "Bash(date:*)" "Bash(sleep:*)" \
    --disallowedTools "Bash(git push:*)" "Bash(gh:*)" "Bash(git remote:*)" \
    "$(cat "$PROMPT")

Before doing anything:
1. Read $STATE.
2. Read docs/multigpu-opportunity-ledger.md.
3. Inspect git status/log.
4. Inspect all outstanding Slurm jobs (squeue -u \$USER; sacct for finished ones).
5. Read the newest experiment outputs under /data/run01/scxi253/inference/results.

Continue the highest-value unfinished research line. Kill hypotheses when their registered gate fails.
Do not wait idly for a GPU job: while a job runs, analyze completed data, research duplicate work,
prepare the next harness, or work on another line.

Before exiting this session (you have ~50 minutes; stop new work at 45), atomically update $STATE with:
completed work, exact git SHA, Slurm job IDs, result paths, measured numbers,
alive/killed/blocked decision per line, next exact command, unresolved risks. Commit locally.

Never publish GitHub content, push, merge, force-push, or sign DCO." \
    2>&1 | tee "$LOGDIR/$TS.jsonl" || true
  sleep 15
done
