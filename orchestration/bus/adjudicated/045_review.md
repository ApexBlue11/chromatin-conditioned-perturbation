# REVIEW OF PACKET 045
verdict: SOUND-WITH-CAVEATS
reviewed_commit: e8e78c7

**The numbers are right.**
- **The addendum:** every range in the score-unit table reproduces from `chromatin_power_91.json` (the five draws'
  `delta_all`, drug-known rows), including P1 at 1 % (+0.0122 to +0.0168). P3 never passes: its raw Δ is +0.0074 to
  +0.108, and its centred Δ stays within ±0.0003.
- **Figure 9's values:**
  - panel b matches the funnel JSON: FB +0.00262, FC +0.00208, FBC +0.00406, N1 +0.00383, N2 +0.00375;
  - panel c matches `v9_dev_T4_T4b_per_cell.json` per cell and per seed, and the n per cell sum to 4,043;
  - the failed-mark labels come from the two encodings' mark lists.
- **The manuscript's numbers:** all match RESULTS 91.10–91.12.
- **The hand pushes:** `kern_v9p7` and `kern_moa88_t1` are unchanged since 1cd614c and dc00cf8 (empty `git diff`, clean
  tree).

**Four of the sentences say more than the record licenses:**
- **C1 (MAJOR):** the abstract quotes the 0.5 % bound without its form, which is the same overreach review 043 C1 fixed in
  91.12's heading.
- **C2:** the addendum compares scores on different scales.
- **C3:** "reading chromatin costs accuracy" leaves out two dev cells where it helps on every seed.
- **C4:** the gene-generic paragraph overstates the increment, and part of that is my own wording in review 043.

**C5:** three small figure fixes.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | overreach | **The abstract's sentence states a bound ten times tighter than the record for one of the two forms tested.** *"Rules out a drug-conditioned, gene-local effect explaining ≥ 0.5 % of the cell-specific response"* drops "for the gain form". The drug-specific shift (P2) is also drug-conditioned and gene-local, and the record bounds it only at **5 %**: P2 at 2 % gave +0.0022 to +0.0034 and failed 0 of 5. The sentence also drops "on these dev cells". 91.12's heading pairs the two bounds since review 043 C1; the abstract, the most-quoted sentence, doesn't. §5.3 has a smaller version of the same problem. Its italic clause says *"no drug-conditioned gene-local chromatin effect … ≥ 0.5 %"*, then adds *"(drug-specific shifts are bounded at 5 %)"*, which contradicts the clause unless the clause names the gain form. | **Abstract:** *"… rules out a chromatin gain on the drug's mean response explaining ≥ 0.5 % of the cell-specific residual (drug-specific shifts: ≥ 5 %), of forms linear in the encoded tracks, on six dev cells"*. Use "residual", as in the body, not "response". **§5.3:** start the italic clause with *"no chromatin **gain** …"*, as 91.12 does ("For the gain form: …"). |
| 2 | MINOR | wrong-quantity | **The addendum ranks the funnel's floor against Δs in another score.** The funnel's Δ is a per-row Pearson Δ over B0 at r ≈ 0.145. The training ablation's 0.006–0.016, C7's 0.0042 and §91.10's "+0.002 to +0.01" are Δs in v9-type scores at r ≈ 0.44. For the same captured variance, a Pearson gain scales about as 1/r: adding an orthogonal component with r₂² gives Δr ≈ r₂²/2r. So the funnel's +0.004 bar at r 0.145 corresponds to about **+0.0013 at v9's 0.437**, assuming v9 would capture the same increment orthogonally to what it already explains. That's an assumption, and the true figure is unknown. So "comparable to C7, not an order of magnitude finer" and "§91.10's range is covered only from ≈ +0.004 up" are cross-scale claims. The crude conversion points the other way. (The direction is conservative for the negative, so no verdict changes.) One more scope point: "≤ 0.0003 of noise" is the spread over five synthetic features on the **same six cells**. It isn't sampling error over cells: the real T1 Δ per cell runs from −0.0008 to +0.0062. | **Addendum:** replace bullets 1 and 3 with *"the funnel's floor (≈ +0.004–0.005) is in its own score (B0 0.142) and is not ranked against Δs in v9's score (≈ 0.44)"*. If a ranking is wanted, give the explained-variance conversion with its assumption stated. Say *"spread over five synthetic features, same cells"* for the noise. **Lesson 8:** add *"(in its own score, B0 0.142, not directly comparable with v9's)"* after "≈ 0.005". |
| 3 | MINOR | overreach | **"On unseen cells reading it costs accuracy" (§5.3 heading) and the abstract's T4 clause generalise past the per-cell data.** (a) Reading chromatin **helps** in two dev cells, on every seed: **HL60 −0.0133** (seeds −0.0143 / −0.0149 / −0.0107) and **SKBR3 −0.0038** (−0.0067 / −0.0021 / −0.0025). SKBR3 is 41 % of the dev rows. U937 is ≈ 0 with seeds of mixed sign. The +0.006 is a net over cells that disagree. (b) T4 is a **mean** ablation. It removes cell-specific chromatin and keeps the gene-generic part, which §91.12 finds to be the informative part. So "removing chromatin" overstates what was removed (review 044 C2(b)). (c) "Held-out-cell score" in the abstract can be read as the test cells. It's the dev carve. | **Heading:** *"v9 reads its chromatin; on the dev cells, reading it costs accuracy on balance"*. **§5.3:** add *"…; it helps in HL60 (−0.013) and SKBR3 (−0.004), on every seed"*. **Abstract:** *"replacing each cell's chromatin with the training mean at inference raises the dev-cell score by 0.006 (a diagnostic)"*. |
| 4 | MINOR | overreach | **The gene-generic paragraph overstates the increment. Part of that is my review 043 C3, which I concede.** I wrote "exceeds all five π = 0 draws, **so the increment is real**", and §91.12 adopted it. The null draws rule out a fitting artefact from extra columns, not cell-to-cell variation. FBC − FB is positive in **3 of 6** cells: LNCAP +0.0062, HL60 +0.0020, SKBR3 +0.0006; HEK293T −0.0008, U937 −0.0002, VCAP −0.0001. Two cells carry it. Separately, *"This is the kind of per-gene content a gene embedding can represent"* drops review 043 C3(a)'s "not tested here". The capacity claim is true, but without the qualifier it reads as "v9 already has it". | **§91.12:** *"not a fitting artefact (it exceeds all five π = 0 draws); positive in 3 of 6 cells, carried by LNCAP and HL60"*. **Manuscript:** *"adds +0.0014 on the drug-known dev rows (3 of 6 cells)"*, and *"… a gene embedding can represent; whether v9's does is not tested"*. The gene-generic reading itself stands: N1 comes within +0.0002 of FBC. |
| 5 | MINOR | presentation | **Three figure details.** (a) **Panel a:** the real-data line runs across the planted axis and meets P2's curve near 1 %, which invites an implied effect size. There isn't one. The real increment is gene-generic, and the planted Δ (FBC\* − FB) isn't purely cell-specific either, so no point on the π axis is like-for-like. (b) **Panel b:** labels rounded to 4 decimals make FBC − N1 and FBC − N2 read 0.0003 and 0.0004, against the text's 0.0002 and 0.0003. (c) **Panel c:** "3 seeds averaged" can be read as a seed ensemble, which in this paper is a different quantity (+0.029). | (a) Keep the value, but as a short mark at the left edge, labelled *"real data: FBC − FB +0.0014 (gene-generic; no planted size implied)"*, with the same sentence in the caption. (b) Use 5 decimals in panel b. (c) Axis: *"mean of the three seeds' differences (dots: each seed)"*. |

## Answers to the asks

**Ask 1 — the table is right. Correction 1 is right. Correction 2 is cross-scale (C2).**
- **"MDE ≤ 0.5 %":** already a bound. Review 043 called it "the smallest π tested". "Ceiling" is the same statement, so the
  heading needn't change. Adding *"(the smallest size planted)"* is optional.
- **"Covered only from ≈ +0.004 up":** this mixes a v9-score range with a funnel-score Δ (C2).
- **"Above the π = 0 noise":** this holds against synthetic-feature noise, not cell-sampling noise (C2, C4).

**Ask 2 — three sentences need changes. The other two you asked about are licensed.**
- **The abstract sentence:** no. It needs the form, the 5 % bound and the dev-cell scope (C1), and its T4 clause needs C3.
- **"The cost is drug-specific":** yes. The manuscript defines the drug-specific component as the cell-centred one (line
  136), and centred +0.0056 against raw +0.0060 places the cost there. Write *"in the drug-specific (cell-centred)
  component"* to tie it to the definition. It means the cost varies with the drug within a cell. A gain-type
  modulation of all drugs would also qualify, so it doesn't mean "specific to particular drugs".
- **"The kind of per-gene content a gene embedding can represent":** licensed as a capacity statement only. Restore "not
  tested" (C4).
- **The rest is licensed:** the §5.3 training-ablation restatement (0.006–0.016, §91.10), the T4b sentences, and the
  limitation (12 of 26 is v9's input coverage; the funnel used 11 after dropping PHH's failed track). Lesson 8 needs C2's
  scale clause.

**Ask 3 — the figure doesn't mislead apart from C5.**
- **The log–log axes:** right for three decades. Both forms run at slope ≈ 1, and the ~10× gap between P1 and P2 is shown
  faithfully.
- **Filled and hollow markers:** they match the record (P1 5 of 5 at every π; P2 0 of 5 at ≤ 2 %, 5 of 5 at ≥ 5 %).
- **Panel c's ordering:** it shows the most-harmed cell (LNCAP) with no failed track, which is the honest presentation
  (review 042 C1).
- **The caption should say:**
  - filled means T1's **full** rule (all five conjuncts), not just the bar;
  - the band is |Δ|, since a log axis can't show negatives;
  - P3 is omitted because the centred conjunct always fails it, although its raw Δ lies above the bar.
- **The real-data line:** keep it, relabelled and shortened (C5a).

## What I checked and found sound

- **The addendum table:** all rows against the five records, plus P3 and the π = 0 draws on both row sets.
- **The panels:**
  - panel a plots `delta_all` on known rows, the median lines, the bar from the funnel JSON (asserted at 0.004), and the
    band from the null draws' max |Δ| (0.00025);
  - panel b's five bars against the funnel's scores;
  - panel c's bars and dots against the per-cell, per-seed JSON;
  - the failed marks are v9 marks minus rank-normal marks.
- **`t4_per_cell.py`:** it reads saved predictions only, asserts the row-order identity, and asserts the 91.11 values to
  6e-5.
- **Manuscript §5.3:** the calibration ranges as rounded (+0.006 to +0.009 for 0.5 %, from +0.0055 to +0.0090) and the
  N1 / N2 figures.
- **The kernels:** `kern_v9p7` and `kern_moa88_t1` are byte-unchanged in the repo since their cleared commits.

## What I could not assess, and why

- **What Kaggle actually runs for P7 and t1.** I checked the repo copies, not the pushed versions or their run status.
- **The SVG.** I reviewed the PNG only.
