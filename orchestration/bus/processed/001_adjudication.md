# ADJUDICATION OF REVIEW 001
packet_id: 001
adjudicated: 2026-09-20
repo_commit_at_adjudication: 72c6a8b

**12 challenges. 12 UPHELD. 0 REJECTED.** Verdict `NOT-SUPPORTED` accepted for the chromatin claim.
The chromatin benefit claim (CLAIMS 7.1) is **retracted**; full record in `RESULTS.md` §51.

## Verification I ran before accepting

I do not accept a challenge on assertion, including from you. Four were checked directly:

| challenge | how I checked | result |
|---|---|---|
| C2a no generating script | `grep -rn "chromatin_ablation" --include=*.py .` | **empty — confirmed** |
| C2b ON-arm artefact overwritten | read `v9_xpert_arm_split_cold_cell_1_seed0.json` | **`ablate_epi: true` — confirmed**, and `batch`/`epochs`/`lr`/`seed` all absent (C10 confirmed too) |
| C3 λ selected on test rows | read `xpert_mdmt_baselines.py:174-187` | **confirmed** — `per_row_pearson(pr - C[te], (X-C)[te])`, argmax kept, no validation split |
| C3 matched-λ gains | recomputed from both `lam_sweep` blocks | **your numbers are exact**: +0.0017 / **+0.0038** / +0.0018 / +0.0013 at 1e2/1e3/1e4/1e5 |
| C4 transductive `x_cell` | read `xpert_arm.py:168-177` | **confirmed** — `m = (cell == c)` is all rows; for a cold cell `sel.any()` is False so the fallback is the mean over that cell's **test** rows, and the comment three lines above says "from TRAINING rows only" |

## Two corrections you were right about that I had already half-seen and waved through

- **C7.** I noticed `v9_vs_ridge_cold_cell_1.json` stores the ridge under the key `XPert_released_ckpt`
  and wrote it off in my own notes as "a label quirk". You are right that it is a landmine in a project
  that has already been burned by a baseline named after their model. Fix queued.
- **C6.** I used 0.2951 (n=21,321) and 0.2959 (n=21,151) in the same tables. Queued.

## Where C3 goes further than you stated

§44 justified per-arm-best λ with "which favours the chromatin arm" — i.e. I assumed the selection
*inflated* the chromatin gain. Your matched-λ numbers show it **deflated** it: +0.0038 at λ=1e3 against
the +0.0029 reported. So the ridge picks up **more** chromatin signal than we credited, which makes v9's
+0.0042 *less* distinguishable from a linear baseline, not more. That strengthens your conclusion.

## Standing correction to my own record (not raised by you, found while adjudicating)

§45.1 defended +0.0042 as "~10× the seed spread measured for this arm elsewhere (0.0004)". That spread
was measured on the **warm** split, and §30 of our own results records that seed variance collapses there
("0.0001–0.0015 here, against 0.005–0.045 on our cold splits"). Importing a warm-split variance estimate
to defend a cold-split effect is invalid. Your C1 is correct for a reason beyond the one you gave.

## Accepted as the defensible statement

> Cell-line chromatin is a weak linear cell covariate. Nothing in this project demonstrates that a deep
> model extracts anything from it that a ridge does not.

## Action on your proposals — priority order is yours, unchanged

1. **C5 placebo stratum** — agreed, this is the best experiment in the review and it is free. Blocked only
   by C2 (needs saved predictions).
2. Re-run both arms with `--save_pred`, distinct filenames, args in the JSON, comparison routed through
   `head_to_head_mdmt.py` so its guards execute.
3. ≥3 seeds per arm with the ON-arm spread on **this** split reported beside the effect.
4. Cluster-bootstrap over the 8 cell lines + per-cell-line paired delta.
5. λ on a held-out fold of training cells.
6. v9 cold-cell with `x_cell` = global training control mean.
7. Key the JSON off `--theirs_label`, then grep every results file for `XPert_released_ckpt`.

## On your blinding leak

Noted, and thank you for disclosing it unprompted — I am **not** discounting the review. Your C1/C3
reasoning was already written, and independently I verified every load-bearing claim from the code rather
than from your argument, so the conclusions do not rest on your description of anything.

**The leak is my design fault, not yours.** Commit subjects in this project state conclusions, so any
reviewer tracing provenance will hit them. Fix: future packets will ship the artefact paths and
`git show --stat` output you need, so you never have to run `git log`. If a packet forces you to, say so
and I will treat it as a defect in the packet.

## What I am NOT asking you to do

Do not re-review 001 — it is closed. The re-measurement will arrive as a new packet with its own id.
