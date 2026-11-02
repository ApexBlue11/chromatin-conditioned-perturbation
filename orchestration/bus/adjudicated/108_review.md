# REVIEW OF PACKET 108 (stage-1 code, `model/cellrep/stage1.py`, before its one run)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: aa7b8f5

**The code implements §103 / §103.1 / §103.2, gates included. 12 / 12 tests pass on my run** (0.2 s).

**The order and gates:**
1. **The leak test** on synthetic values over the real index structure. It must give fixed |r| < 0.02 and leaky < −0.9, or
   the run stops.
2. **P0**, which returns before anything else is computed if it fails.
3. **The tas strata, VD and the ceiling.**
4. **D1's positive control,** then D1 only if it passes.
5. **B1,** which is not gated on D1, as registered.

**The harness:** `fold_targets` builds per-fold references from R(g) \ {row cell, c₀} with ≥ 3, and training rows only in
core cells. The leaky §103 definition is rebuilt for the test only.

**The four challenges are small.** The one that matters before the run is how to switch D1's primary to hairpins if you
choose that (C1, Ask 2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | D1's primary (Ask 2) | **If the hairpin-level s_kd becomes primary,** as I recommend in Ask 2, the code doesn't yet implement it as primary:<br>• the hairpin variant is computed only **inside** the consensus positive control's pass branch;<br>• its own positive control is reported but **gates nothing**;<br>• it has no `passes` (CI and ≥ 5 of 6);<br>• the reported items (other features, rung 1, the s_ctl technical channel) are computed for the consensus version only. | **Make the hairpin version the primary path:** its positive control gates; its rule gives `passes`; the reported items are computed on it.<br>Keep the consensus version as the reported variant, with its positive control and rule values shown.<br>**Add one test** that the gate follows the primary. |
| 2 | MINOR | B1's own-coordinate asymmetry (Ask 3, choice 2) | **The target never carries an own-gene dip, but about 28 % of the competitors do.**<br>• The target's own coordinate is removed. Non-landmark targets have none anyway.<br>• The ~946 landmark-gene knockdowns in the competitor pool keep their deep own-coordinate dip.<br>• The dip adds variance that isn't shared with the drug, which shrinks those competitors' correlations toward 0.<br>• **If drug–knockdown correlations are generally positive** (shared generic responses), the target outranks dipped competitors for no specific reason. The bias is toward a pass.<br>• **The same asymmetry hits the hub control** when the hub gene is a landmark.<br>• By rough arithmetic the bias is small, about 0.005 in percentile. But it has a known direction, and the fix costs nothing.<br>• **The random-Y dry run can't show it:** an isotropic Y has no shared response. | **Compute every knockdown's correlation without its own coordinate** (and without t's), so the target, the competitors and the hub are treated alike.<br>**Add a test:** with a shared generic component and no specific link, the median percentile is ≈ 0.5. |
| 3 | MINOR | a registered covariate is missing | **§103 lists D1's covariates as** CCLE expression (primary) and *"the substrate's mean L1000 control profile of j in c (secondary)"*. The secondary analysis isn't computed. | **Report D1's rule also partialled on the L1000 control profile** (centred), or state why it's omitted. |
| 4 | MINOR | a fact in §103 | **The core set is described as "the six cells with chromatin on all three axes".** HT29 has ACC and ACT only (no PCG, no METH), per `v10c_chromatin.npz`'s mask.<br>D1's primary (ACT) is available in all six cells, so D1 is unaffected. The PCG and METH features in the reported list cover 5 cells. | **Correct the sentence** (*"chromatin with ACC and ACT in all six; PCG / METH in five, not HT29"*). Report each feature's cell count beside it. |

## Answers to the asks

**Ask 1 — yes, faithful**, apart from C3 (a missing secondary covariate) and C4 (a wrong description).

**Checked against the registration:**
- **P0:** the ascending own percentile, median ≤ 0.10, per cell; overexpression's descending percentile; per-gene values
  saved.
- **VD:** two-way shares per landmark on the 2,100 × 6 matrix, own gene excluded; MS_gxc against
  mean((y₉₆ − y_late)²) / 2. The mean square (not the variance) only loosens the upper bound.
- **The ceiling:** disjoint alternating halves, ≥ 2 cells each; ρ̂, √ρ̂ and the row-wise √ mean; PC3 / MCF7.
- **D1:** log(s_kd / s_ctl) over the 2,100 genes minus own; centring by landmark; the mean of within-cell (partial)
  Spearman; the same landmark resample in every cell; the axis mask.
- **B1:**
  - percentile 0 = most correlated;
  - the dose rule;
  - matched against mismatched, as mean(p_mis − p_match) with a drug-cluster CI and pairs without another cell dropped from
    that comparison only;
  - top-5 %, hub, agonists against overexpression, tas tertiles.
- **The calibrated dry runs** (VD ratio 1.0009, leaky −1.0000, B1 0.511) are what pure noise should give.

**Ask 2 — make the hairpin-level s_kd D1's primary, with the consensus version reported.** Decide it now, as you say.
- **Why hairpins:** they match `ctl_vector`'s construction (one perturbation, replicate MODZ). That removes the
  averaging-depth channel review 107 C3 identified, which is the technical threat D1 is most exposed to.
- **What it adds:** hairpin-level SD includes seed (off-target) effects. They are still perturbations of the cell, so they
  fit D1's question ("responsiveness to genetic perturbation"). Partialling on expression handles their expression-dependent
  part.
- **Implement it as in C1.**
- **Composition:** hairpin coverage per gene can differ slightly by cell. Report hairpins per gene per cell (13,000–15,000
  rows per cell).

**Ask 3:**
1. **B1 on Level 5:** acceptable, and better. Drug and knockdown signatures share the MODZ z normalisation.
   - **The compound filter** (no non-train cold-drug row anywhere) is stricter than row level, which is fine.
   - **I checked** that PC3, A375, A549 and HEPG2 have only `train` rows in `split_cold_cell_1`.
2. **One coordinate set per pair:** fix as in C2.
3. **The hub draw:** acceptable. Genes absent from the STRING graph get degree 0 (`gdeg.get(x, 0)`), so for low-degree
   targets the 10 nearest can include unannotated genes. Restrict the candidates to graph genes, or report how often it
   happens.
4. **D1's axis mask:** acceptable and needed (HT29, C4).
5. **120 h preferred:** acceptable. It is closer in time to 96 h, so it tightens both bounds (a smaller noise upper bound, a
   larger ceiling lower bound), and both stay valid.

## What I checked and found sound

- **`stage1.py` in full:**
  - the harness (`row_r`, `index_signatures`, `fold_targets`, `per_gene_model_scores`);
  - P0;
  - `two_way_shares`, `half_references`;
  - the rank / partial Spearman, `mean_within_cell`, `landmark_bootstrap`;
  - `dose_um`, `choose_rows`, `percentile_of`, `corr_rows`, the cluster bootstraps;
  - `load_sig_metrics`;
  - `run`'s order and gates;
  - `run_b1`'s filters, targets (Homo sapiens, sign −1), degree, rankings, matched / mismatched, agonists and strata;
  - `main`'s read-once guard.
- **`test_stage1.py`:** 12 passed (`CUDA_VISIBLE_DEVICES=-1`). The working tree outside the bus is unchanged by me.
- **Outcome-free checks:** the chromatin mask per core cell (HT29: ACC, ACT only); B1's four cells all `train`; the
  cold-drug label counts.

## What I could not assess, and why

- **Everything needing signature values.** That covers the size of C2's bias, whether drug–knockdown correlations are
  generally positive, and every rung's outcome. Not reading `Y`, the hairpin file or the Level-5 drug matrix keeps stage 1
  blind.
- **The new inputs' contents.** I didn't verify the sig_metrics file and the hairpin extract beyond the sha1s stated in the
  packet.
