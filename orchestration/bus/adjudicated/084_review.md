# REVIEW OF PACKET 084 (Phase 1 code: C1–C4 rescoring, C2b / C3 inference, three CPU kernels)
verdict: SOUND
reviewed_commit: fbb6a3a

**The code implements §8 as amended, and the statements, preconditions and variants of record match the plan.**
- **Tests:** the 26 pass on rerun (`.venv-cuda`, CUDA hidden; 10 s).
- **Checked against the data:**
  - the fingerprint slice is right: `u` = Uni-Mol 512 | descriptors 20 (standardised) | ECFP4 2,048 **raw 0/1 bits**, so
    `u[:, 532:2580] > 0.5` is the true bit vector, the same file the C1 strata use;
  - the swap substitutes are deterministic and identical across seeds (fresh `default_rng(0)` per variant; the same test rows,
    cells and compounds);
  - the swap moves only `drug_row` and `u`, the model's only compound inputs (076).

**Five MINOR points.** C1 and C2 should land before anything runs, because every test runs once.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | correctness (C1's dose-only floor) | **Plate-mates without a floor enter the floor's contrast as zero vectors.**<br>• `plate_contrast(np.where(has_dose, D_dose, 0.0), …)` contrasts over *all* scored plate-mates, so a mate lacking a floor contributes **0**, not nothing. That biases the mates' mean toward 0.<br>• **Size, from metadata only** (no truth read): 233 of 13,364 P9 rows have no floor, and **303 of the 8,806 contrast-eligible rows** have at least one such mate. | Contrast the floor over floor-bearing mates only, e.g. `other_mean(D_dose, plate, pert, sel=has_dose, …)` with ≥ 2 such mates, else NaN. Add a test world with a floorless mate. |
| 2 | MINOR | C2b attribution (my 083 C5, refined) | **`x_cell := X_ctl` is the right inductive replacement, but the reading should state what it removes.**<br>• For an unseen cell, its own control is the only per-row inductive information, so the replacement removes exactly what transduction supplies: the cell mean, and with it the plate-deviation signal x_ctl − x_cell. C2a's F_tr prices that signal.<br>• **But the configuration sits at the edge of training:** the deviation is exactly 0, and `x_cell` becomes a single noisy control instead of a mean. So part of any drop is input shift. I should have said so in 083. | **Reported:** the same `ctl` / `global` passes on P9's final-snapshot checkpoints over P9's test rows (seen cells, where `x_cell` is training-derived). That cost is shift plus the deviation's legitimate value, an upper bound on pure shift.<br>**Reword the sentence:** *"… substantially carried by the cell-average input, which for unseen cells is available only transductively"*. Place the P9 drop beside it. The kernel already loads P9, so this adds two passes. |
| 3 | MINOR | provenance (kernels) | **The kernels pin the code, the checkpoints and the identity files, but not the data `XPertData` reads** through `find()`:<br>• the bundle;<br>• `drug_{fingerprints,unimol,descriptors,atom_reprs,atom_offsets}.npy`;<br>• the priors;<br>• the chromatin inputs.<br>Nor do they check uniqueness across the three mounted datasets.<br>The identity check (2,000 rows, r ≥ 0.9999) is a strong functional proof for those rows, but the swap also uses donors' drug inputs, and every other kernel in the project pins its data. | Add the data files to `PINS` and to `find_one`'s uniqueness check, at least the bundle and the five drug arrays. Record their sha1s in `AUDIT99_COMPLETE`. |
| 4 | MINOR | reader guard (C3) | **`swapped = np.all([donor ≥ 0 for each seed])` silently intersects.** The donors are identical by construction, so a seed-dependent difference would mean a defect, and it would be hidden. | In `_load_infer` for swap, require `donor_row_index_{tani,free}` equal across the three seeds (as for `row_index` / `y` / `ctl`). Then use them directly. |
| 5 | MINOR | description vs code (C3) | **The packet says unswapped rows are "excluded from both variants". The code excludes them per variant:** a tani-unswapped row still enters `free` if `free` found a substitute. | Either describe it as per-variant, or report `free` also on the tani rows, so the two variants are compared on one row set. The reading of record (tani) is unaffected. |

## Answers to the asks

**Ask 1 — the contrast code is right, and the 1e-4 relative zero rule is sound.**
- **Why 1e-4 works:** a control-only model gives identical predictions on a control group, so its contrast is float residue,
  about 1e-6 of the truth contrast. That is far below 1e-4, while any genuine compound-dependent variation is far above it.
- **No path remains for a plate-constant model:**
  - the mates are the same rows for truth and prediction;
  - the plate is the control group, the same unit that makes M constant;
  - rows with < 2 mates are NaN, not 0.
- **Counting the zeroed rows,** and reporting v9 against 0, are the right companions.

**Ask 2 — yes, that is the floor I meant, with C1's fix:**
- training rows only;
- the same (cell, time, dose), by exact float equality inside the bundle;
- leave-own-plate-out (`ctdp` nests the plate);
- then the same contrast.

**Ask 3 — yes, within the limits C2 states.** It makes the two control views identical, which is the point: it zeroes the
deviation that only the transductive mean makes computable. C2's shift control bounds how much of the drop is the edge-of-
distribution input.

**Ask 4 — mostly yes.**
- One substitute per (compound, cell), uniform among those allowed, consistent across the compound's rows: right.
- −1 rows excluded and counted: right.
- **"Unswapped under any seed":** equivalent here, because the donors are seed-invariant. Assert it instead (C4).
- The per-variant exclusion: C5.

**Ask 5 — acceptable as disclosed.**
- **The identity check** (2,000 rows of each checkpoint's own saved final-snapshot `deg_pred`, min r ≥ 0.9999) is an empirical
  equivalence proof of the forward pass.
- **The `c55ce257` changes are known to be inert here:**
  - B1's flag is off by default (076: the flag-off path equals the old code on 299 argument combinations);
  - the chromatin-encoding lift is byte-identical on the real `E_final` (070).
- **The modification paths** (`x_cell`, `drug_row` / `u` read in `batch()`) are `c55ce257`'s own and are what runs.

**Ask 6 — nothing blocks once C1 and C2 land.**
- **Locally:** C1 / C1b / C1c / C2a / C2c / C4 read saved predictions only. Run them after C1's fix, because C1 runs once.
- **The kernels:** add C2's P9 passes and C3's data pins first.
- **One practical note:** record the smoke's wall time per row. Each kernel does 3 × 13,364 + 3 × ~21,151 CPU forwards, and a
  12 h timeout writes no `AUDIT99_COMPLETE`, which the readers refuse.

## What I checked and found sound

- **`audit99_phase1.py` in full:**
  - `rowr_pred`, `plate_contrast`, `other_mean`, `loo_mean`, `mates_tanimoto`;
  - `_plates_for`'s identity guard;
  - C1 (primary, statements, per seed, against 0, M, the ratio, the floor, the strata), C1b, C1c, C2a, C2c, C4;
  - `read_c2b` (variant of record, precondition, statements), `read_c3` (variant of record, statements with CI / `ci_hi`);
  - `_load_infer` (manifest, sha1, mode / seed / limit, cross-seed equality);
  - run-once.
- **`audit99_infer.py` in full:**
  - the identity check, the `xcell` passes with restoration;
  - `substitutes` (within cell, compound-level, Tanimoto, determinism);
  - the swap's input moves.
- **`xpert_arm.py`'s `u` construction, and the drug array shapes / bit values.**
- **The three kernels and their metadata** (CPU, internet off, the pinned image; they differ only in seed and pins).
- **Plan §8 against the code.**
- **The floor's coverage,** from metadata only.

## What I could not assess, and why

- **Every Phase 1 value.** Nothing has run, as registered.
- **The `lincs-audit99` dataset's contents.** I relied on the kernels' pins.
- **The CPU runtime on Kaggle.** No timing was reported (Ask 6).
- **The mutation check** (0 of 20). I didn't rerun it.
