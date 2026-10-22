#!/usr/bin/env bash
# Detached reset waiter (memory: anything that must wait hours runs as a detached Git-bash with PATH pinned). Checks the
# wall clock every 60 s (so it survives laptop sleep), and at TARGET (epoch seconds, UTC) starts tb_sequencer.sh on the
# kernel list in reset_push_list.txt. The scheduled task lincs-kaggle-reset-push is the second trigger; it finds this
# sequencer running and only reports.
# Usage: reset_waiter.sh <target_epoch_utc>      (SEQ / LIST may be overridden for tests)
export PATH="/c/Users/Surya/AppData/Local/Programs/Python/Python314/Scripts:/c/Users/Surya/AppData/Local/Programs/Python/Python314:/usr/bin:/bin:/mingw64/bin"
TARGET=$1
SEQ=${SEQ:-/c/Projects/LINCS/external/tb_sequencer.sh}
LIST=${LIST:-/c/Projects/LINCS/external/reset_push_list.txt}
LOG=${LOG:-/c/Projects/LINCS/external/v10c_tb_sequencer.log}
echo "$(date -u) waiter armed for $(date -u -d @"$TARGET") (pid $$)" >> "$LOG"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 60; done
mapfile -t DIRS < <(tr -d '\r' < "$LIST" | grep -v '^[[:space:]]*$')
echo "$(date -u) waiter firing: ${DIRS[*]}" >> "$LOG"
exec bash "$SEQ" "${DIRS[@]}"
