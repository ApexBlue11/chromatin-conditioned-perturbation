#!/usr/bin/env bash
# Detached sequencer for the SECOND Kaggle account (KAGGLE_CONFIG_DIR below): tb_sequencer.sh with its own kernel root and log.
# Kaggle refuses it (2-session cap); a kernel counts as pushed only when Kaggle says "successfully pushed".
# Usage: tb_sequencer.sh kern_dir1 kern_dir2 ...   (directories under external/kaggle_kernels)
export PATH="/c/Users/Surya/AppData/Local/Programs/Python/Python314/Scripts:/c/Users/Surya/AppData/Local/Programs/Python/Python314:/usr/bin:/bin:/mingw64/bin"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-C:\Users\Surya\.kaggle_alt}"
LOG=${LOG:-/c/Projects/LINCS/external/alt_sequencer.log}
cd ${KERN_ROOT:-/c/Projects/LINCS/external/kaggle_kernels_alt} || exit 1
echo "$(date -u) start (v2): $*" >> "$LOG"
slug() { python -c "import json,sys; print(json.load(open(sys.argv[1]+'/kernel-metadata.json'))['id'])" "$1"; }
for d in "$@"; do
  s=$(slug "$d")
  if kaggle kernels status "$s" 2>&1 | grep -q -E "RUNNING|QUEUED|COMPLETE"; then
    echo "$(date -u) $s already exists; skipping" >> "$LOG"; continue
  fi
  for i in $(seq 1 400); do
    out=$(cd "$d" && kaggle kernels push -p . --accelerator NvidiaTeslaT4 2>&1)
    echo "$(date -u) push $s: $(echo "$out" | tail -1)" >> "$LOG"
    if echo "$out" | grep -q "successfully pushed"; then break; fi
    if echo "$out" | grep -q -i "quota"; then echo "$(date -u) quota reached; stopping" >> "$LOG"; exit 2; fi
    sleep 300
  done
  sleep 60
done
echo "$(date -u) all pushed" >> "$LOG"
