# REVIEW OF PACKET 060
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c1f0587 (packet ad8ced0)

**`mechanism_stage_a.py` (44bb13e5) implements §94 as amended by §94.7, and no model output is loaded anywhere.**
- **Tests:** I reran `test_mechanism_stage_a.py` under `.venv-cuda` (decoupler 2.2.0): **15 passed**.
- **The A1 null is right.** The vectorised `Mate[np.ix_(perm, perm)]` is the label permutation. The plate exclusions and the
  similarity ranks stay tied to the signatures, which is correct.
- **The A3 null is right:** within-cell permutation of labelled units, one permutation per cell per draw, shared by all classes
  in that cell. Both p-values are (1 + #null ≥ obs) / (1 + n).

I also checked the data at identity level only (no response read). Four points, all MINOR:
- **C1:** A3's new positive control has **no unit** in the data.
- **C2:** the self-retrieval ceiling, which scopes the null sentence, is asymmetric and ignores the plate rule.
- **C3:** a labelling point.
- **C4:** first-match is moot here.

**Cleared to run.** C2 is better in before the run.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | disclosure | **The Hypoxia positive control (94.7 item 2) yields no A3 unit, and neither do p53 and NFkB.** Counting class members per scored cell from identities (gated table, ≥ 3 members): **17 units**, all MAPK, EGFR, PI3K, JAK and ER@MCF7 (ER 4, MAPK 5–8, EGFR 6–9, PI3K 4–16, JAK 3–8). No scored cell has ≥ 3 EGLN, MDM2 or IKBKB/CHUK compounds. So A3 has **no positive control**, and the 2/3 rule needs ≥ 12 of 17 units with d > 0. (The ungated table has 20 units, adding AR@MCF7, AR@HT29, ER@HT29 and the RAF members.) | State beside the A3 reading: *"No positive-control unit exists (fewer than 3 EGLN inhibitors in every scored cell); A3's 17 units are kinase and ER classes."* No code change. |
| 2 | MINOR | diagnostic | **The self-retrieval ceiling compares unlike quantities, and it ignores the plate rule.**<br>- **The comparison:** the positive is corr(half A_i, half B_i), but each negative is corr(half A_i, **full** sig_j). A full signature is less noisy than a half, so the negatives are systematically favoured and the ceiling is biased **down**.<br>- **The plate rule:** a compound's two halves often share plates (shared X_ctl noise), which biases the positive **up**. Negatives from the same plate aren't excluded either, unlike in A1.<br>- **Why it matters:** this number scopes the NEITHER sentence (94.7 item 5), so it should be like-for-like. | In `compute_self_retrieval_and_active_subset`: score negatives as corr(A_i, **B_j**), half against half, and drop any j sharing an X_ctl hash with i, as in A1. Optionally, also report the ceiling restricted to compounds whose halves share no X_ctl hash. Add one test (a planted reproducible compound with noise-matched halves gives a ceiling near 1). It's reported only, but fix it before the run since it enters the sentence. |
| 3 | MINOR | labelling | **`per_class_auroc` is the mean, over a class's members, of each member's AUROC against *all* its mates** (any shared MoA string). It isn't the class's own retrieval. For a member of several classes, the value mixes classes. | Rename it to `mean_member_auroc` in the output, or compute class-specific AUROCs (mates = members of that class). It's reported only. |
| 4 | MINOR | robustness | **"First match" (the packet's ask) is moot on these data.** Among the scored cells' labelled parents, **0** have matching class targets with opposite action-type signs. Class-target action types are INHIBITOR 196, MODULATOR 15 and AGONIST 10, plus one ACTIVATOR, which is excluded, as registered. | Optional: replace first-match with *"exclude a compound whose matching targets carry opposite signs"*, plus an assertion. That's the right rule if labels change, but it changes nothing now. |

## Answers to the asks

**Ask 1 — no mismatch with §94 / §94.7.**
- **Labels:** `organism == Homo sapiens`, direct interactions, any target type. MoA strings give A1's classes, and
  (gene, action) pairs give A3's membership.
- **The parent collapse:** a unit is a parent, with all its pert_ids' rows. Unlabelled pert_ids are their own units and count
  only in the cell mean.
- **Signatures:** the mean Δ minus the mean over **all** units' signatures in the cell, as registered.
- **The plate rule:** a pair is excluded as mate and as non-mate if any X_ctl sha1 is shared; the diagonal too. **Its effect,
  from identities and X_ctl hashes only** (labelled parents with MoA strings in each cell):

  | cell | mate pairs excluded | all pairs excluded | units with ≥ 1 mate, before → after |
  |---|---|---|---|
  | MCF7 | 14 % | 6 % | 409 → 400 |
  | HT29 | 7 % | 4 % | 297 → 288 |
  | MDAMB231 | 55 % | 52 % | 57 → 48 |
  | HS578T | 43 % | 46 % | 37 → 31 |
  | THP1 | 54 % | 33 % | 25 → 17 |

  - **The confound is real:** same-class pairs share plates more often than pairs overall (MCF7 14 % against 6 %; THP1 54 %
    against 33 %), so the rule removes it.
  - **The cost:** THP1 keeps **17** compounds with mates, so it will rarely reach p < 0.01. A1's ≥ 3 of 5 will in practice need
    MCF7, HT29 and one of MDAMB231 or HS578T. `n_with_mate` is recorded per cell, so that's visible in the output.
- **The gated signs:** RAF targets in HT29 only; MDM2 (+) and ER in MCF7 only; AR dropped; Hypoxia (+). A3 compares, as
  registered, the class's own-sign activity against the other labelled compounds' activity × the inhibitor sign.
- **PROGENy:** the sha1 refusal at af40b7a5, the landmark restriction, the ≥ 15-gene minimum and tmin = 15. The gene names come
  from `pathway_landmark_genes.txt`, with a 978 assertion and an overlap assertion. (The `g{i}` fallback remains in `run_a3`, but
  `run_pipeline` always passes the names, and the overlap assertion would catch the fallback.)
- **The reader:** A1 Δ ≥ 0.05 and p < 0.01 in ≥ 3 cells; A3 p < 0.01 and ≥ 2/3 of units; §94.5's texts with C4's scoped
  sentence. `--read` refuses on a marker mismatch.

**Ask 2 — yes.** The active subset is chosen label-blind (split-half r ≥ 0.2), so permuting labels within it is the valid null.
It preserves the subset's similarity structure and asks only whether labels align with it. (C2's half-against-half point also
applies to the split-half r that picks the subset, but a label-blind threshold can't bias the label test.)

**Ask 3 — cleared to run** (local CPU, once), preferably with C2 in, and with C1's disclosure beside the reading.

## What I checked and found sound

- **`mechanism_stage_a.py` in full:**
  - the class tables (gated and ungated);
  - `load_rows` (21,151-row assertion; Δ in float64), `load_labels` and `units_for_cell`;
  - `scored_cells` (fixed five, reported counts), `calc_auroc`, `a1_cell` (Pearson, plate matrix, rank-sum AUROC, null) and
    `compute_self_retrieval_and_active_subset`;
  - `load_progeny_network`, `run_a3` (membership, d, null), `read_stage_a`, `run_pipeline` and `main` (output, marker,
    `--read` refusal).
- **The tests:** 15, rerun.
- **Identity-level counts** from the bundle's metadata and `X_ctl` hashes (no X read):
  - A3 units per class and cell, gated and ungated;
  - action types on the class targets, and sign conflicts;
  - the plate rule's effect on mate pairs and on compounds with mates per scored cell.

## What I could not assess, and why

- **Whether decoupler's `ulm` on these cell-centred unit × gene matrices behaves as in the synthetic tests.** That needs the
  real run.
- **The PROGENy table's live sha1.** I didn't re-download it. The code refuses on a mismatch.
