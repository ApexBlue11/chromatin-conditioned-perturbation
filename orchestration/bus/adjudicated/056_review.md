# REVIEW OF PACKET 056
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 408ad05 (packet bfeb464)

**The reading is mechanical, and I reproduced it exactly.**
- **Provenance:** `chromatin_h1_93.json` = `6448b643`, matching the marker, the downloaded copy in
  `external/kaggle_out/chromatin93_h1/`, and the committed copy. The two markers are byte-identical. The marker's pins match the
  repo: `chromatin_h1.py` 5b5f3382 (review 055's fixed version), `c93_features.npz` 23e12b9d, `chromatin_funnel` c56bb6f1 and
  `chromatin_power` c501cba2.
- **The rerun:** `read_h1.py` on both directories gives readings **identical** to `chromatin93_h1_reading.json`
  (marker field aside).
- **The numbers:** every value in 93.13 and the packet matches the raw output, per slot, case and draw: the calibration table,
  ρ, faults, MDEs, `G_raw_delta`, the real conjuncts and per-cell values, Cp, LOCO, shrinkage, and the 3-cell counts.
- **The run log:** clean (exit 0, 5,002 s). Its one warning (`Mean of empty slice`, `chromatin_funnel.py:637`) comes from SKBR3
  and VCAP having no rows of record. They appear as `null` in `per_cell` and aren't counted.

The informative negative is licensed, with one caveat on how firm its MDE is (C3). The Stage 1 summary is not licensed as
written:
- **C1:** item 2 says more than the tests show and contradicts item 3.
- **C2:** item 3 lists two routes as untested that the record has partly tested.
- **C4:** item 1 needs scopes.

All four are MINOR, wording only.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **Item 2, *"carries no drug-conditioned response information that transfers to unseen cells in a gene-local form"*, is broader than §91 and §93 show, and item 3 contradicts it.** P2-type drug-specific interactions are both drug-conditioned and gene-local. They are bounded only at 5 % (F_reg here; v9's promoter marks in T1, §91.12) and **not at all** for F_enh (MDE_P2 none) or for the trees (H3: 0 of 3 at every π). Item 3 itself lists them as untested. What the record supports is the **gain form** at ≈ 0.5 % of the cell-specific residual, linear and trees, with ≤ 11 fitting cells. Three further scopes are missing:<br>(a) "reproducible" is **within-series** (93.12 C3);<br>(b) "unseen cells" means with **4–11 fitting cells** (H2's range);<br>(c) the second sentence's "what chromatin offers … is gene-generic" holds for the **promoter marks** only, and is small: T1 +0.0012, below its bar (§91.12); trees +0.0053 against a nothing-planted +0.003 (93.7). §91.12 also notes that a per-gene parameter could represent it, which hasn't been tested. **In H1, the gene means of the enhancer and regulon features add nothing either:** N1 − B = **−0.00002** on the rows of record and +0.00006 in LOCO. | **Item 2:** *"Across both learners and both feature granularities, the cell-specific part of accessibility (reproducible within series) carries no drug-conditioned response information of the gain form above ≈ 0.5 % of the cell-specific residual that transfers to the dev cells from 4–11 fitting cells. Drug-specific forms are bounded at 5 % (F_reg, v9 promoter marks; linear) or not at all (F_enh; trees). For the promoter marks, a small gene-generic gain remains: T1 +0.0012, below its bar; trees ≈ +0.002 above the null. A per-gene parameter could represent it, which is untested. Enhancer and regulon features add nothing in either form (N1 − B −0.00002)."* |
| 2 | MINOR | overreach | **Item 3's "not tested by any Stage 1 test" misplaces two of its three routes.**<br>(i) **Network propagation, one hop, is what F_reg is:** a target gene's feature built from its CollecTRI regulators' motif accessibility. It was null at MDE_P1 0.5 %, with P2 bounded at 5 %. What's untested is multi-hop or non-TF propagation (PPI, pathway), and the **458 of 978 genes** that have no regulator with a motif (F_reg = 0).<br>(ii) **The trained model's use of chromatin was measured** by T4 (inference, three seeds; §91.11) and by E2 (training-time ablation, three seeds; §92.10). Removing cell-specific chromatin raised the dev score **+0.0073 (centred +0.0045)**, with a per-cell split consistent across both: help in SKBR3 and HL60, harm in HEK293T, LNCAP and VCAP; U937 mixed ("suggests, not established"). What's untested is **why** that split arises. **H1 covers few of the split's cells:** SKBR3 fails `has` and VCAP is excluded; of the three it measures, it is null in HL60, the one cell where v9's chromatin appears to help (C − B +0.00008, C − N1 −0.0003). | **Item 3:** *"Not tested: (1) drug-specific interactions (P2-type), bounded at 5 % for F_reg and v9's promoter marks and unbounded for F_enh and the trees; (2) propagation beyond one TF→target hop (F_reg, null), including genes without a motif-bearing regulator (458 of 978); (3) the mechanism behind the trained model's per-cell chromatin pattern (T4, E2: help in SKBR3 and HL60, harm in HEK293T, LNCAP and VCAP). H1 measures only HL60 among the 'help' cells, and is null there."* |
| 3 | MINOR | caveat | **The informative-null label's "MDE_P1 ≤ 0.005 in both slots" is right as registered (3 of 3), but in the F_enh slot it sits right at the detection edge.** The weakest P1 0.5 % draw passed with C − B **+0.00426** (bar 0.004) and C − N1 **+0.00204** (bar 0.002), a margin of 0.00004 on the N1 conjunct. F_reg's weakest draw has room: +0.0072 and +0.0105. So for F_enh, a gain near 0.5 % would be detected only just, and gains somewhat below it are not excluded. A reader taking "≤ 0.5 %" as a comfortable bound for both features would be misled. | Beside the label, descriptively: *"F_enh's 0.5 % is at the detection edge (weakest draw: C − B +0.0043 against 0.004; C − N1 +0.0020 against 0.002); F_reg's has margin (+0.0072, +0.0105)."* No rule change: the MDE was registered as 3 of 3 on the {0.5, 2, 5} % grid. |
| 4 | MINOR | overreach | **Item 1 needs scopes that 93.7 and review 053 already fixed.**<br>**H2:** *"the curve was flat over 4–11 fitting cells with v9's marks; beyond 11 cells is untested"*. The data limit the range; the result doesn't.<br>**H3:** *"on v9's promoter features, gain form"*. The trees weren't run on the c93 features.<br>**H5's** wording is right (93.12 C3). | Add the two scopes to item 1 as written here. |

## Answers to the asks

**Ask 1 — yes, mechanical** (93.10, 93.12), and reproduced from the committed and downloaded outputs.
- **The reader's order:** NOT RUN → VOID (no null, P3 or G pass in either slot) → `real.reading.pass` false → DOES NOT ADVANCE.
- **The label:** MDE_P1 = 0.005 ≤ 0.02 in both slots.
- **The cell conjunct:** 2 of 4 against k = 3, over the measured cells only.
- **The slot-2 note:** "G uninformative" is correctly appended. It has no bearing on a negative reading, because G guards against
  a false *pass*.
- **Why it is uninformative there** (plausible, not verified on f\*): F_reg comes from MA standardised across cells per motif,
  so its gene-common component is weak. Within-cell r(own F_reg, mean of the other has-cells) has median **0.25**, range −0.76
  to 0.65, against F_enh's 0.76. A gene-mean plant is then something a cell's own F_reg column can't carry, and it can even
  carry it with the wrong sign (raw Δ −0.003 to −0.008).

**Ask 2 — licensed as written, with C3's caveat beside it.**
- **"Of a form linear in the encoded features"** and **"explaining ≥ 0.5 % of the cell-specific residual"** are the P1 gain
  form at its registered MDE.
- **"Beyond promoter accessibility from the same samples"** is right: B uses the c93 ±1 kb coverage F_prom, not v9's channel.
- **The P2 bounds** (5 % F_reg, none F_enh) match `faults_and_mde`.
- **The cell-sample caveat:** HME1 17 and HUH7 125 near-landmark peaks, and HEK293T, match 93.11.
- **"Inside the nothing-planted range":** real C − B −0.00008, against −0.00011 to +0.00049 across both slots' null draws.

**Ask 3 — not as written.** Items 1–3 need C4, C1 and C2.
- **Item 2's core** is supported **for the gain form**: no cell-specific transferable gain above ≈ 0.5 %.
- **Item 2 as worded** would also rule out the P2 forms the record bounds only weakly, which is what item 3 then lists as open.

**Ask 4 — an assessment of the options, not a choice of direction.**
- **Route (3), the trained model's use.** As a transfer question it's largely answered: E2 is the direct ablation, and T4 the
  inference-time one. An attribution study can describe use; it can't establish transfer value beyond those ablations. What
  remains open is the per-cell split's mechanism. E1 (H4, the clean encoding) is the scheduled test that bears on it, and its
  seeds 1–2 are pending.
- **Route (2), network propagation.** Its first step is already done (F_reg, null). Propagating further averages over more genes'
  accessibility, which tends to push a feature toward its gene-common part, the form already shown to add little. The design
  space is large (operator, depth, graph), so any such test needs **one** operator fixed and calibrated before data.
- **Route (1), P2-type.** It's the only form the record bounds weakly, so it's the only one where a new test could change what's
  known. **Its limit is cells per drug, not the learner:** a drug-specific weight is estimated from ≤ 9–11 fitting cells. An
  informative instrument would need a factorised drug × chromatin form, for example through drug features. **§88's null and the
  May-2026 benchmark** (L1000 models don't use their drug features; 81e7075) make that a low-prior route. **Whether any such
  test can be informative here** is settled cheaply and outcome-free: run the planned instrument's calibration alone (planted
  P2 at 0.5 / 2 / 5 %, no real response read). If MDE_P2 is not below 5 %, it can't improve the current bound.
- **Closing.** The record licenses closing the **gene-local gain form** for these data: both learners, both granularities,
  MDE ≈ 0.5 % (C3's edge for F_enh), ≤ 11 fitting cells. It doesn't license a general "chromatin does not transfer". P2 is
  weakly bounded, the cell range is data-limited, and E2/T4's consistent per-cell split shows the trained model's chromatin
  doing something cell-dependent.
- **Keeping the gene-generic finding for the paper.** It needs §91.12's own caveat (a per-gene parameter could represent it; a
  control with a non-chromatin per-gene covariate of matched shape would test whether it's chromatin-specific), its size
  against the null, and H1's result that the enhancer and regulon features carry none.
- **Which of these to do is the PI's and principal's call.**

## What I checked and found sound

- **Provenance:** sha1s of the output, both markers and the pinned code. The marker pins match the repo, and `read_h1.py` is
  unchanged since 52d1182, the reviewed version.
- **The reader:** `read_h1.py` reread in full, and rerun on `external/kaggle_out/chromatin93_h1` and `model/results/c93`. Both
  readings are identical to the committed one.
- **The raw output, item by item:**
  - real `reading`, `delta` and `vs_N1` (all, top, per cell, centred, cell counts), the conjuncts against `BARS['known']`, and
    `pass`;
  - scores for B, C, N1, Cp and S at λ = 0.25 / 0.5 / 0.75;
  - LOCO, which rises monotonically toward the gene mean (C < S0.25 < S0.5 < S0.75 < N1), with spans ≤ 0.00012;
  - shrinkage (best λ 0.75; against B and N1);
  - every calibration case × draw × slot (pass, raw Δ, top, centred, C − N1, cell counts), and `faults_and_mde`.
- **The builder:** `h1_builder`'s N1 = [b, P, mean E, mean R] over the fitting cells with `has`, the held-out cell excluded.
  This is the basis for C1(c)'s N1 − B.
- **The run log**, and the warning at `chromatin_funnel.py:637`, traced to dev cells without rows of record.
- **93.11's split-half table**, against `c93_split_half.json`'s medians keys (F_prom, F_enh, F_reg, MA).
- **§91.11–91.12, 92.10 and 93.7 text**, for the claims C1, C2 and C4 rest on.
- **From the frozen `c93_features.npz` and `E_final*`:** the F_reg / F_enh cross-cell structure (Ask 1), F_reg's coverage of
  520 of 978 genes, and v9's per-cell track availability for the dev cells.

## What I could not assess, and why

- **Whether the slot-2 G mechanism holds for f\* itself.** It depends on whether f\*'s permutation keeps F_reg's cross-cell
  structure. I checked it on the real F_reg only. It doesn't bear on the reading.
- **The source of v9's SKBR3 accessibility channel.** It is non-zero in `E_final` (sd 0.059) although SKBR3's ATAC files are
  empty in c93. I didn't trace it, so C2 says only that H1 doesn't measure SKBR3.
