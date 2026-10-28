#!/usr/bin/env bash
# Run a command in a detached tmux session so it survives SSH disconnects / Mac sleep.
# Output is also written to <log_dir>/<session>.log.
#
#   tools/tmux_run.sh <session> <log_dir> <command...>
set -euo pipefail

SESSION="$1"; LOG_DIR="$2"; shift 2
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$SESSION.log"

tmux new-session -d -s "$SESSION" \
    "$(printf '%q ' "$@") 2>&1 | tee -a $(printf '%q' "$LOG"); echo '[exited]'; exec bash"
echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  log:    $LOG"
