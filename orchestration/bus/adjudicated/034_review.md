# REVIEW OF PACKET 034
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 14f8e58

**P7 is well specified and faithful to rule 10.**
- **Scorer:** it averages per-row *scores* over the three seed files, never predictions. It applies §71.3's ordered
  reading unchanged to that estimand. The secondary blocks (per_seed, centred for both models, ensemble_own, alt) sit
  outside the verdict, and the headline rows can't be narrowed by `--ours_alt`.
- **Command:** it is §87's v9 command (`kern_cc1_epi_s0`: P2's command without `--dev_cells`) plus distinct seeding,
  the stack flags and 3 seeds.

**One thing breaks "touched once" as written: the training kernel itself scores the test cells and prints the result
(C1).** It must be switched off before P7 runs. The rest is smaller: the XPert-wins case (C2), the test-cell
interpretability thresholds (C3, ask 3), and the V1-full branch (C4).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **Without `--dev_cells`, `xpert_arm.py` computes P7's estimand on the 8 test cells after every seed and prints it to the kernel log.** `D.te` is then the test rows (21,151 after the unfeaturisable drop). After each seed, `rec` computes `their_pearson(pd, Xte − Cte)` (`xpert_arm.py:473-474`), and `print(f'  [seed {seed}] Pearson … Pearson_deg …')` (`:509`) writes it out, followed by the 3-seed mean/range (`:519-520`), and all of it goes into the output JSON's `runs`. `Pearson_deg` *is* the per-seed per-row delta Pearson on the test rows. So the test result becomes visible in the log, seed by seed, while P7 is still training. There are two consequences. (a) §3's handling can't hold: a run that fails after seed 0 has already shown seed 0's test score, so predictions deleted "unread" were in effect read. (b) Even a clean run allows an interim look after seed 0, with the temptation to intervene that a one-shot comparison exists to remove. | Add a flag, e.g. `--no_test_metrics`, that skips `their_pearson` on test rows (and V2's per-snapshot metrics, if any) and writes predictions only, with the JSON's `runs` carrying no scores. The P7 kernel passes it, and a guard greps the finished log and JSON for `Pearson` values and fails if any are present. The model-free `nulls` (copy-control, mean-drug) are already known from §87 and can stay. `coldcell_h2h.py` is then the only thing that ever scores P7's test predictions. |
| 2 | MINOR | overreach | **"Anything else → no claim" folds §71.3's XPert-wins row into "no claim".** §71.3 has three rows, and its second, *cluster mean < 0, CI excluding 0, ≥ 7 of 8 favour XPert*, reads **"Uninterpretable as a model comparison … Reported plainly, not explained away."** Rule 10's table covers §71 = XPert wins, not P7 = XPert wins. The natural reading applies §71.3's own row to P7, and that's a different sentence from "no claim". | State P7's three outcomes with §71.3's wording: v9 wins → the labelled secondary claim; XPert wins → *"uninterpretable as a model comparison, reported plainly"*; otherwise → no claim. `coldcell_h2h.py` already distinguishes the rows, so only the packet's §4 text needs changing. |
| 3 | MINOR | stats | **The interpretability reading on the test cells can't reuse §85.10's rule as written. "≥ 5 of 6 dev cells" doesn't exist on 8 test cells.** It's also the one fresh, unselected look the interpretability claim will get, since rule 8 passed on the very dev cells used for selection. So its licensing has to be fixed now, not inherited by analogy. | Fix now, in `align_dev.py --rows test` and RESULTS. (i) The training-row prior is recomputed from P7's training rows, all 32 cells. (ii) The aux alignment must beat that prior on every seed. (iii) "In this cell" requires the per-cell increment over the other-cells mean readout to be > 0 in **≥ 7 of 8** test cells on the 3-seed mean (one-sided sign p = 0.035, near the dev rule's spirit), with the mean of per-cell increments > 0 on every seed. State that it's reported beside P7 and never enters its verdict. |
| 4 | MINOR | code-vs-intent | **The V1-full branch isn't executable as written.** (a) "`mc_infer_dev`-style MC inference on the 8 test cells" names code that doesn't exist. `mc_infer_dev.py` asserts the *dev* sha1 and rebuilds the dev carve. (b) The table has no row for V2 **and** V1-full both accepted. §85.11 confirms V1-full only on C6's checkpoints, not on a C6 + V2 stack. | (a) If V1-full is accepted: write the test-row MC script, commit it, and send it for review before P7, with GUARD 6 on its outputs and C1's no-metrics rule. (b) Add the fourth branch now. One coherent choice is V1-full applied to P6's C6 + V2 checkpoints under §85.11's rule; if it isn't confirmed there, V1 is dropped from the stack. |

## Answers to the asks

**Ask 1 — consistent, apart from C1–C4.**
- **Rule 10's estimand:** seed-averaged per-row score, ensemble only as a labelled secondary.
- **Rule 10's label and table:** §87 = no claim, so only the "v9 wins" row can produce a claim.
- **§71.7's admissibility:** unchanged. O2 is the same run.
- **§85.11's branches:** P2 + C6, plus V2 or V1 as confirmed.
- **§90.6 as amended:** V2-last as a required row via `--ours_alt`, and ensemble_own with no paired difference.
- **§85.2 rule 2:** fitting on training rows only holds, since P7 has no dev carve and all 32 training cells fit
  everything.
- **The commands:** P2's command and §87's differ only in `--dev_cells`. `dp_seed_mode` defaults to `distinct` in
  `xpert_arm.py:302`, which is why P2 was distinct without passing it (its log shows [s, s+1000]). P7 passes it
  explicitly, which is better.

**Ask 2 — deletion is right, but only once C1 is fixed. Voiding P7 would be too strong.** A crash isn't a look.
Voiding would make the registered second comparison impossible because of an infrastructure failure. What makes
"deleted unread" credible is that *nothing* can have scored the files. Given C1:
- the kernel computes no test metric;
- any test npz from a failed run is deleted before `coldcell_h2h.py` exists in that session's history, with the
  deletion and each file's sha1 logged;
- the rerun is disclosed.

Better still, make it moot. Have the kernel move the three seeds' npz files to their final names only after the last
seed and every guard pass, and delete partial outputs itself on any failure. Then a failed run leaves nothing to read.

**Ask 3 — it's a new reading, because of the new rows and the 8-cell structure, and it needs C3's thresholds fixed
now.** Reporting it with §85.10's licensed wording is right. But the licence itself has to be restated for 8 cells and
P7's training rows before P7 runs, and `align_dev.py --rows test` has to be committed, and reviewed, before then.

## What I checked and found sound

- **`coldcell_h2h.py` at 84ffd1e:**
  - `r_o` is the mean over files of per-row Pearsons, computed on rows finite in *every* file and in XPert's, so the
    row set is fixed by the headline inputs only.
  - All `--ours` and `--ours_alt` files must agree on `row_index`, `y_true` and `ctl_true`.
  - The verdict conditions are the committed ones, applied to that averaged estimand: cluster mean, CI and ≥ 7 of 8.
  - The per-seed blocks reuse the same function with a fresh RNG.
  - The single-file path reproducing §87's JSON exactly is the right regression test.
- **GUARD 6:** it pins the exact 21,151-row set §87 scored (sha1 `be276e23…`), so P7 can't silently score a different
  row set.

## What I could not assess, and why

- **Whether O2's profile file still hashes to §87's `69484323…`.** It isn't re-verified in the packet. Assert it in
  the P7 scoring command's preamble.
