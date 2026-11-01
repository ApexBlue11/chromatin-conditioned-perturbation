# REVIEW OF PACKET 102 (H-DD read once, §100.16: no claim)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 3db28ed (result 764b76a; scorer 6f9a527)

**"No claim" is correct, and the verdict is fixed by the stored slot means alone.**
- **(i) fails:** off seed 2 = 0.2242 is above the minimum on slot, 0.2198.
- **The rule needs both conditions,** so (ii) can't change it. The mirror fails too.

**The run is the registered computation on the registered inputs:**
- `model/results/v10c/hdd_reading.json` is byte-identical to the kernel's output.
- **The scorer** run on Kaggle has sha1 96921253…, equal to `hdd_test.py` at 6f9a527 (HEAD). The other source files are
  identical too.
- **Every pinned input I could hash matches the local file:**
  - the 6 checkpoints;
  - `compound_classes.json`;
  - `hdd_identity.npz`;
  - the substrate a276eda8…;
  - `drug_feature_index.json`.
- **`hdd_identity.npz`** holds plate 32's 54 test rows for each checkpoint. They equal the ledgered T-C predictions exactly
  (max |diff| 0), and those predictions are sha1-checked in review 100.
- **So the identity r of 0.9999998–0.9999999 ties each checkpoint to its ledgered T-C run.**

**The caveat is one closing sentence that goes beyond what was tested** (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | wording: "no chromatin effect specific to a drug class or a cell survives in v10-C" (Ask 2) | **The cell half of this was never tested.**<br>• T-C's strongest cell-level extreme is **CD34**: Δ_c −0.013; every off slot beats every on slot; seed z −4.1. That is the same extremeness as DNA damage, in the other direction. Review 100 C3 asked that the two be treated alike.<br>• **CD34 has 654 untouched `excluded_test_cell` rows.** Only DNA damage × MCF7 went to an untouched test. *"Survives"* implies a test it didn't get.<br>• **On classes, "none detected" is the supported form.** Small classes can't exclude effects of the size H-DD tested: T-C's epigenetic class CI reaches +0.016, sirtuin's rests on 4 compounds. | *"The one class signal tested on untouched rows (DNA damage in MCF7) was not confirmed. No other class or compound stood out in T-C's strata (0 BH discoveries), though small classes can't exclude effects of about 0.01. CD34's negative cell-level extreme (−0.013) is untested; 654 untouched rows exist."*<br>A CD34 test on those rows, with the same checkpoints and the same rule mirrored, would settle the cell half. It must be registered first. |
| 2 | MINOR | wording: "keeps its sign at half the size" (Ask 2) | **"Half the size" is a point estimate inside a wide interval.**<br>• **Compound-cluster one-sided bounds** −0.0034 to +0.0304: compatible with 0 and with T-C's +0.026.<br>• **Seed level:** Welch t 1.57 on ≈ 4 df, one-sided p ≈ 0.10.<br>• **Some regression toward 0 is expected** for a stratum chosen as T-C's most extreme.<br>• **Run noise is large in this 78-row stratum.** Slot SD ≈ 0.01 within each arm, and both arms' seed-2 runs are the highest in their arm (0.2363, 0.2242).<br>• **Only seed 2 is seed-matched** across arms, because of the guard retries (on 20 / 11 / 2, off 0 / 1 / 2). Its paired difference is +0.012. | *"The point estimate keeps its sign (+0.013 against T-C's +0.026) but is not distinguishable from 0 (one-sided bounds −0.003 to +0.030; seed t 1.6 on 4 df). Not confirmed."* |
| 3 | MINOR | reproducibility | **The kernel output doesn't hold the per-row values.**<br>So (ii)'s bound (−0.0034) and the reported strata's bounds can't be recomputed from it. Only Δ and (i) can. The verdict doesn't depend on (ii) here. | **For future read-once scorers, save the per-row r (6 × rows) beside the reading.** It is cheap, and it makes every bound reproducible without re-running the checkpoints. |

## Answers to the asks

**Ask 1 — yes, as far as the kernel output allows, and fully for the verdict.**
- **From the stored slot means I recomputed:** Δ (+0.012792), (i) (fails), the mirror (fails), and the other cells' and the
  pool's Δ and (i).
- **Integrity holds:**
  - code sha1 = the reviewed scorer;
  - inputs sha1-pinned and equal to the local files;
  - the identity file equals the ledgered predictions;
  - the identity r ties each checkpoint to its T-C run;
  - the pre-scoring counts equal the registration, as in my count in review 101.
- **(ii)'s bound can't be recomputed** without the per-row values (C3).
- **I didn't re-run the six checkpoints.** That is about 6 CPU-hours on Kaggle, too much for this laptop.

**Ask 2 — "not confirmed" is right. Two sentences need changing.**
- **"Half the size"** should carry its interval (C2).
- **"No class- or cell-specific chromatin effect survives"** overreaches. The cell half was never tested, and CD34 is testable
  (C1).

**On the pool:** its (i) "holds" by 0.00003 (on minimum 0.15233 against off maximum 0.15230). Reported, never read, so it
needs no reading. But don't quote it as a pass of (i).

## What I checked and found sound

- **The kernel output in full:** `hdd_reading.json` (byte-equal to the committed copy), `HDD_KERNEL_COMPLETE.json` (pins,
  returncode 0), and the log. That covers the command line, six "scored" lines with sha1 and identity r about 1 h apart, and
  the final verdict.
- **Source identity:** the downloaded `src` against HEAD (hdd_test, data, model, train, modules_v9 / v7).
- **The local sha1s** of the 6 checkpoints, the class map, the identity file, the substrate and the drug index, against the
  pins.
- **`hdd_identity.npz` against the six ledgered prediction files:** 54 rows of test plate 32 each, max |diff| 0.
- **The 6f9a527 diff of `hdd_test.py`:**
  - `heldout_counts` before any checkpoint;
  - `sha1_file`;
  - `identity_check` with T-C's own loader and tolerance 0.999;
  - neutral labels for non-primary strata;
  - provenance in the output.
- **Δ, (i), the mirror and Welch t / df for MCF7, the other cells and the pool,** from the stored slot means.
- **Untouched-row counts per test cell** (CD34 654, BJAB 207, HS578T 167, HT29 11,507, MCF7 14,309, MDAMB231 5,543, THP1
  5,701).

## What I could not assess, and why

- **The slot means themselves and (ii)'s bound.** Re-running the checkpoints (≈ 6 CPU-h) is beyond this laptop, and the
  per-row values weren't saved (C3).
