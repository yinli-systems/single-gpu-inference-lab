#!/usr/bin/env bash
set -uo pipefail

ROOT="/Users/alice/Bio/inference/single-gpu-inference-lab"
PROMPT="$ROOT/AUTONOMY_PROMPT.md"
STATE="$ROOT/AUTONOMY_STATE.md"
LOGDIR="$ROOT/autonomy_logs"
CLAUDE="/Users/alice/Library/Application Support/Claude/claude-code/2.1.260/claude.app/Contents/MacOS/claude"

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

HOURS=${1:-8}
END_TIME=$(( $(date +%s) + HOURS*60*60 ))
ROUND=0

echo "================================================="
echo "Claude autonomous research watchdog started"
echo "Start: $(date)"
echo "End epoch: $END_TIME"
echo "Repo: $ROOT"
echo "================================================="

while (( $(date +%s) < END_TIME )); do
    ROUND=$((ROUND + 1))
    TS=$(date +"%Y%m%d-%H%M%S")
    LOG="$LOGDIR/round-${ROUND}-${TS}.jsonl"

    echo
    echo "================================================="
    echo "Starting Claude round $ROUND at $(date)"
    echo "Log: $LOG"
    echo "================================================="

    # Each Claude instance gets at most ~50 minutes. If it crashes, hits context limits,
    # reaches max turns, or exits normally, the outer watchdog launches a fresh instance
    # that reads AUTONOMY_STATE.md.
    PROMPT_FILE="$LOGDIR/round-${ROUND}-prompt.txt"
    {
      cat "$PROMPT"
      cat <<EOP

THIS IS AUTONOMOUS ROUND $ROUND OF AN APPROXIMATELY ${HOURS}-HOUR RESEARCH SPRINT.

Before doing ANY new work:

1. Read: $STATE
2. Read: docs/multigpu-opportunity-ledger.md
3. Inspect: git status; git log --oneline -20
4. Inspect all current/recent Slurm jobs: scripts/dp_ep/pc.sh 'squeue -u \$USER'; and sacct for finished ones.
5. Find and inspect the newest experiment outputs and server logs on the cluster.
6. Reconstruct what previous autonomous rounds actually accomplished. Do not repeat finished
   experiments unless replication is required.

AUTONOMOUS EXECUTION RULES:

- Continue working without asking the user questions.
- Use the highest-value runnable research line.
- Prefer falsification and oracle experiments before implementation.
- When a preregistered gate fails, mark the hypothesis KILLED and immediately move to the next best line.
- Do not sit idle while Slurm jobs run. During GPU execution: analyze previous data, inspect current
  upstream source, perform duplicate-work research, prepare analyzers, prepare next experiments,
  write artifacts, or work on an independent CPU-side task.
- Poll Slurm periodically rather than blocking for long periods.
- Keep GPU utilization productive where possible.
- Never fabricate a result. Preserve negative results. Do not silently change registered gates.
- Do not treat trace-mode performance as production performance.
- Do not confuse microbenchmarks with serving benchmarks.
- Do not claim novelty without current duplicate-work checks.

BEFORE THIS CLAUDE PROCESS EXITS, EVEN IF THE CURRENT EXPERIMENT IS INCOMPLETE:

Atomically rewrite $STATE with exactly these sections:
# Autonomous Research State
## Timestamp
## Git (branch, SHA, dirty files)
## Running Slurm jobs (id, purpose, expected output path)
## Completed this round
## Measured results (actual numbers only)
## Alive hypotheses (evidence + gate)
## Killed hypotheses (reason + evidence)
## Blocked hypotheses (specific blocker)
## Current strongest result (bounded wording)
## Next exact action (the exact first command/file/experiment for the next session)
## Files/artifacts created (paths)
## Risks / unresolved methodological issues

Also update docs/multigpu-opportunity-ledger.md. Commit locally. Then exit cleanly so the watchdog
can start the next fresh Claude context.
EOP
    } > "$PROMPT_FILE"
    "$CLAUDE" -p \
      --model opus \
      --effort max \
      --max-turns 120 \
      --output-format stream-json \
      --verbose \
      --permission-mode acceptEdits \
      --allowedTools "Read" "Write" "Edit" "Glob" "Grep" "Bash" \
      --disallowedTools \
        "Bash(git push:*)" \
        "Bash(gh:*)" \
        "Bash(git remote:*)" \
        "Bash(git reset --hard:*)" \
        "Bash(git clean -fdx:*)" \
        "Bash(sudo:*)" \
        "Bash(rm -rf /*)" \
        "Bash(rm -rf ~*)" \
      < "$PROMPT_FILE" > "$LOG" 2>&1 &
    CLAUDE_PID=$!
    ( sleep 3000; kill -TERM "$CLAUDE_PID" 2>/dev/null; sleep 30; kill -KILL "$CLAUDE_PID" 2>/dev/null ) &
    KILLER_PID=$!
    wait "$CLAUDE_PID"; EXIT_CODE=$?
    kill "$KILLER_PID" 2>/dev/null

    echo "Claude round $ROUND exited with code $EXIT_CODE at $(date)"

    if [[ ! -f "$STATE" ]]; then
        cat > "$STATE" <<EOS
# Autonomous Research State

## Timestamp
$(date -Iseconds)

## Status
Claude round $ROUND exited before creating a state file.

## Next exact action
Inspect autonomy_logs/round-${ROUND}-${TS}.jsonl and reconstruct progress.
EOS
    fi

    sleep 15
done

echo
echo "================================================="
echo "${HOURS}-hour watchdog finished at $(date)"
echo "Final state:"
cat "$STATE"
echo "================================================="
