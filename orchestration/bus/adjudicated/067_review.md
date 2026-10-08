# REVIEW OF PACKET 067 (RESULTS §97, O9 pre-registration)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: a4df917

**§97 is a faithful transposition of O2 to the cold-drug fold, and its seed claim is right at initialisation.**
- **Fold dependence:** in `lincs-xpert-cc1.py` it runs through `FOLD`: the h5ad level read, `--nfold`, `EXPECT_ARGS`, the probe
  argv, the checkpoint glob, the profile name and the `xpert_native_eval.py --nfold` call.
- **The hard-coded literals** (level counts 47,509 / 21,321; the 21,321-row profile and C4 checks in `prod_tail.py:176, 198`;
  Amendment E's 482.2; the seed note) are exactly 97.2's list, **plus one** it misses (C1).
- **The seed claim, against their code:**
  - `scripts/train.sh:15–19` (`l1000_mdmt`) lists `split_cold_drug_k` first in every fold list;
  - `train_xpert.py:402` calls `set_random_seed(int(args.seed))` once before the fold loop;
  - the loop then calls `load_dataloader` and builds `XPertNet(...)`, `init_weights()` per fold.
  - So O9's cold-drug fold starts from the same RNG state as their own run of it.

All five points below are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | fold constant | **A fold-specific string 97.2 doesn't list:** `prod_tail.py:226`'s `RECORD['framing']` says *"XPert trained to its published recipe on %s. NOT a reproduction of their **cold-cell** run"*. The fold name is substituted, but "cold-cell" isn't. The docstrings and comments ("Train XPert on split_cold_cell_1 AS PUBLISHED", "scored later by head_to_head_mdmt.py against v9's split_cold_cell_1 predictions") are cosmetic, but they'll sit in O9's record. | Add the framing text to the asserted substitutions. Also add a builder guard: the generated O9 kernel contains no `cold_cell` / `cold-cell` / `cc1` / `47509` / `21321` outside an explicit allow-list of historical comments. |
| 2 | MINOR | wording | **"O9 carries no seed difference" is true at initialisation only.** The trajectory still differs from their run, through the disclosed O2 deviations: DataParallel over two T4s, the ten frozen parameters, full-state resumes across sessions, `cudnn.benchmark` non-determinism, and the GPU type. | *"O9 starts from the same seed-2024 state as their own run of this fold (no seed difference at initialisation). Its trajectory is not a replication (O2's disclosed deviations)."* |
| 3 | MINOR | reading rule | **The reproduction rule can be asymmetric, matching 97.3 item 1's logic.** The band guards against a **broken or under-trained** XPert, which would make a v9 win meaningless. An XPert score **above** the band can't make a v9 win less conservative, and XPert wins are never claims anyway. "No head-to-head claim outside the band" is safe but over-blocks above it. O2 flagged and still read. | **Below 0.621:** no v9-win claim (possible reproduction failure), everything reported. **Above 0.669:** flagged and read normally. **Inside:** read normally. Or keep 97.3 as written, which is stricter and fine, but state why it differs from O2. |
| 4 | MINOR | cost model | **Amendment E's reprice scales by training rows only** (482.2 × 55,385 / 47,509). With val = test, every epoch also evaluates the test set, which **shrinks** on this fold (13,445 rows against 21,321). So 562.1 s probably overstates the epoch time. That's conservative for planning and harmless. | Optional: reprice the train and eval parts separately from O2's timing logs (timing is allowed reading under §84.1 item 3). Otherwise note that the projection is an upper bound. |
| 5 | MINOR | added value, register now | **O9 makes 96.7's "not measured" measurable:** how much the same-molecule duplicates lift a SOTA model's score. Compare XPert's per-row scores on the 38 duplicate compounds' rows against the clean rows, with v9's and ridge's alongside as a difference-in-differences. The full-minus-clean contrast also reflects composition (which compounds are duplicates), so a single model's difference isn't purely "leak". | Register it now as **reported, never a claim**: per model (XPert, v9, ridge), the mean row score on duplicate-compound rows minus clean rows, and the across-model difference-in-differences. Molecule-level bootstrap. It belongs with the 96.6 integrity finding. |

## Answers to the asks

**Ask 1 — one omission (C1).** Everything else fold-specific is either keyed on `FOLD` or in 97.2's list. The `MyDataset` memory
patch, the DataParallel proof (§81), the ten frozen parameters and the §84 horizon are fold-independent.

**Ask 2 — yes, from their code, at initialisation** (C2's wording).

**Ask 3 — either is defensible.** The asymmetric rule (C3) matches 97.3 item 1's logic and avoids over-blocking. If you keep
"no claim outside the band", say why O9 is stricter than O2.

**Ask 4 — complete against §71.3, §71.7 and §78.5:**
- **§78.5's scoping** (the counter refers to the final termination; per-session guard stops are resume events; a quota kill is
  INCOMPLETE) is carried by 97.3 item 2.
- **§71.6's unfeaturisable-rows analogue** is handled by 97.4's pairing: the reproduction uses XPert's 13,445 rows; the
  head-to-head uses the 13,364 both can score.
- **One addition:** the clean-subset reading doesn't escape item 1. XPert's checkpoint is selected on the **full** test loss,
  which includes the clean rows, so the asymmetry (a v9 win is conservative) carries over unchanged. Say so in 97.4.

**Ask 5 — worth it, on two counts.**
- **For the paper's claim:** a cold-drug claim against SOTA needs it. The published 0.645 can't stand in, since it's a five-fold
  mean, includes duplicates, and comes from other runs.
- **For the integrity finding:** C5's measurement of what the leak does to a SOTA score.

**A sequencing option for you, not a requirement:** P9 against ridge will read first. If v9 doesn't beat ridge on the clean
subset, a v9 win over XPert is implausible, and O9's remaining value is mainly C5. That could inform whether to spend the worst
case (about 46 GPU-h). The registration is fixed now either way.

## What I checked and found sound

- **O2's machinery:** `generator/make_prod_kernel.py`, `generator/prod_tail.py`, `lincs-xpert-cc1_v7_base.py` and the committed
  `lincs-xpert-cc1.py`, grepped for every fold-specific literal and every use of `FOLD`.
- **XPert's code:** `scripts/train.sh` (fold order, `l1000_mdmt` lines 15–19) and `train_xpert.py:398–440` (seed call and the
  per-fold model build).
- **The RESULTS text:** §97.1–97.5 against §71.7 / §78.5 (read) and §84.1 / 84.2 (as summarised in 97.2).

## What I could not assess, and why

- **The O9 kernel builder.** It isn't written yet; this is pre-code. C1's guard would make the substitutions checkable.
- **O2's own reproduction outcome and §84 in full.** I relied on 97.3's summary of §84.1 items 3 and 5.
