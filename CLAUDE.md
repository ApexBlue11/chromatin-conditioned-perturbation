# LINCS — standing instructions from the principal (read every session)

## Who writes code
- **Code is delegated to `agy` (Antigravity) workers**, e.g. Gemini 3.8 Flash / 3.1 Pro or Claude Opus 4.6 via
  `C:\Users\Surya\AppData\Local\agy\bin\agy.exe -p ... --model <id> --dangerously-skip-permissions`.
  The PI (Claude) writes the brief/contract and the tests' expectations, then **verifies**: runs the tests
  itself, reads the diff, never trusts a worker's report. Full routing: `orchestration/ORCHESTRATION.md` §6/§6b.
- The PI writes code itself only for small glue (a few-line patch, a record script) or when a fix is
  faster than a brief. If the PI writes more than that, it says so in its reply.

## Compute
- **No local GPU training** (this is a laptop; no thermal throttling, no multi-hour runs). Tiny inference/tests only.
- GPU: **Kaggle** (weekly quota 30 h, T4x2) and **Lightning AI** (Studio `s_01m3b77d8c9cw7ex0j1e9agmtr@ssh.lightning.ai`,
  python `/home/zeus/miniconda3/envs/cloudspace/bin/python`; billed in credits — stay within the budget the
  principal sets). Optimise every GPU hour, and **run a GPU plan past the critic before committing hours.**
- GPU spend on Kaggle, file deletion and pushing to our own repo do not need approval. Human checkpoint ~every 30 responses.

## Packages
- If an open-source package does the job and is not installed, **install it or ask** — do not write a
  substitute script.

## Rigour
- Pre-register before code and data (RESULTS.md); critic review via `orchestration/bus` for every result and every
  GPU plan; "reported, not read"; grep the repo before claiming something is needed (method rule 20).
- Live state: `orchestration/bus/state.json`, `model/results/RESULTS.md`, `IDEAS.md`.

## Mechanics
- Work in `C:\Projects\LINCS` on master; commit by explicit path; file-based patch scripts (no inline heredoc
  edits of Python); end commits with the Co-Authored-By line from the harness.
