#!/usr/bin/env bash
# Run a command in a detached tmux session so it survives SSH disconnects / Mac sleep.
# Output is also written to <log_dir>/<session>.log.
#
#   tools/tmux_run.sh <session> <log_dir> <command...>
#
# TMUX_RUN_INLINE=1 runs the command in the foreground instead (same log file), so several
# train.sh calls can be queued back to back inside one outer tmux session.
set -euo pipefail

# tmux rejects '.' and ':' in session names, and train.sh's default JOB_NAME contains
# DATASET_REV (e.g. v0.1).
SESSION="${1//[.:]/_}"; LOG_DIR="$2"; shift 2
if [[ "${TMUX_RUN_INLINE:-0}" == 1 ]]; then
    mkdir -p "$LOG_DIR"
    "$@" 2>&1 | tee -a "$LOG_DIR/$SESSION.log"
    exit "${PIPESTATUS[0]}"
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$SESSION.log"

# Pass this shell's PATH (activated conda env): a tmux server that is already running
# would otherwise give the new session the server's environment instead.
tmux new-session -d -s "$SESSION" \
    "env PATH=$(printf '%q' "$PATH") $(printf '%q ' "$@") 2>&1 | tee -a $(printf '%q' "$LOG"); echo '[exited]'; exec bash"
echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  log:    $LOG"
