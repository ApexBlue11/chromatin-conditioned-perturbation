# REVIEW OF PACKET 087 (v10-C L1–L3 data: selection, signal, features, substrate; D timing v2 / v3 fixes)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: bcbeb8a (substrate as amended at 6d00986)

**The data work is careful and outcome-free.**
- the uniform source;
- exact-name matching;
- the perturbed-sample exclusion;
- missingness kept missing;
- byte-identical benchmark rows;
- the 96.6 same-molecule rule.

**But the substrate inherits a cell-identity leak from the benchmark itself** (C1). It is the one data issue that would make a
test outcome leak.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | leak (unseen cells; the benchmark and the substrate) | **`NCIH1975` is test cell `H1975` under another name, and it sits in unseen-cell training.** That is true both in XPert's `split_cold_cell_1` and in the v10-C substrate.<br>**The evidence** (my checks; the bundle and the substrate):<br>• **The same experiments:** the 54 `NCIH1975` training rows cover the **same 54 (compound, dose, time)** as `H1975`'s 54 test rows.<br>• **The same cell:** mean-X r = **0.9997**. The next-closest test / training pair in the 217-cell release is 0.89 (`BJAB`–`OCILY19`).<br>• **A leak, not a coincidence:** per-row Δ r between the twins is **0.417**, against **0.150** for the same conditions in other cells.<br>• **Its effect on the existing head-to-head** (P7 against O2, rescored by me): H1975 is where both models score highest among the small cells (v9 0.548, XPert 0.583; BJAB 0.33, CD34 0.29–0.33). Excluding it moves the cell-cluster v9 − XPert from **+0.050 (6 / 8)** to **+0.062 (6 / 7)**.<br>**How the substrate inherits it:** through "the benchmark's own training rows are always kept" (Ask 4).<br>**Nothing else qualifies:** no other test cell has an alias. The only other pair above 0.99 (`MCLF035SZ` / `MCLF056CN`) is two training cells, and no dev cell is involved. | **(a) The substrate:** exclude `NCIH1975` from unseen-cell training, and treat it as `H1975` everywhere (the chromatin already shares the track). The identity rules must override the forced benchmark rows.<br>**(b) The record:** log the benchmark leak in RESULTS. Add a reported line excluding H1975 to every unseen-cell reading (§71.3 / §85.14, C2a / C2b / C2c, O2).<br>**(c) T-C:** register it over the **7 clean cells**, with H1975 reported apart (rule in review 086, C2). |
| 2 | MINOR | rule order (Ask 4) | **"Always keep the benchmark's training rows" should rank below the identity rules** (cell alias, same molecule). Otherwise the substrate preserves exactly the benchmark leaks the new rules exist to catch. C1 is a live case. | State the order: identity exclusions first, then keep the remaining benchmark training rows. Record which benchmark training rows the identity rules remove (54 for C1; check the compound split's same-molecule IDs too). |
| 3 | MINOR | sample rule (Ask 1) | **Tier N** (no control term, no perturbation term) is probably mostly untreated baseline, but nothing verifies that. | Report tier composition per cell, and run the reported mechanism check (gate against accessibility) on E / C-only cells as a sensitivity. Don't fall back to perturbed samples: a treated cell's chromatin is not its baseline. Missing is the right answer. |

## Answers to the asks

**Ask 1 — keep tier N, with C3's reporting. No perturbed fallback.**

**Ask 2 — yes, lose the 9 rather than mix representations.** They are training cells, and two return through methylation. A
second representation would reintroduce the per-lab differences the uniform source removes.

**Ask 3 — acceptable, and arguably helpful.** Gene-relative ranks within a cell are what "which genes are open in this cell"
means. Discarding global level shifts also weakens chromatin's use as a cell-ID shortcut. State it as a property of the
representation.

**Ask 4 — compounds shared across the cold-cell split are by design; keep them.** Forcing the benchmark's training rows is
right *except* where an identity rule excludes them (C1, C2).

**Ask 5 — C1 is the leak.** Otherwise:
- the chromatin data are cell-level, from outside sources, and outcome-free;
- the quantiser is fitted on training rows;
- the dev cells are removed from training;
- the dev carve is clean (no aliases).

**The D fixes (v2 / v3):** both are environment issues, with a regression test added. No objection. (D's timing has since read
P62 = 15.8 h, so "not run (cost)", §98.22.)

## What I checked and found sound

- **The substrate:** keys, splits, the excluded counts by reason, and the `NCIH1975` / `H1975` rows in both splits.
- **The benchmark bundle:** the alias pair's conditions, X / control differences, profile correlations, and the per-row Δ
  agreement against other cells.
- **A cell-similarity scan** (mean X) over all 40 benchmark and 217 full-release cells.
- **P7 / O2 per-cell rescoring,** with and without H1975.
- **The §100.1–100.3 text** against the packet. The disclosed deviations (local metadata streaming, whole-file reads, the
  gene-body proxy, the lost cells, the strengthened duplicate rule, metadata-tuned regexes) are outcome-free.

## What I could not assess, and why

- **The 356 tracks' signal values and the feature builder's numerics.** I didn't recompute them.
- **The not-yet-mutation-tested feature and substrate builders.** You've noted this.
