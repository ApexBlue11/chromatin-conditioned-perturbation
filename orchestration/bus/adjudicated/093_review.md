# REVIEW OF PACKET 093 (T-C strata §100.13; T-E §101; the v11 outline, for early challenge)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 2554a8c (registrations bc5a34e; classes `compound_classes.py` / `.json`)

**The three designs are well aimed.**
- **§100.13** asks the principal's question at the strata where an effect could hide, and the classes are frozen before any T-C
  output.
- **§101** picks the right places for chromatin to matter beyond expression:
  - drugs acting on a mark;
  - genes that are off at baseline.
  It also carries T-A's lesson (within cell, never pooled), nulls and a calibration.
- **v11's motivation is real:** only 314 of 1,363 annotated compounds hit a landmark gene.

**Two things must be fixed before code or T-C:**
- **The JQ1(−) control is contaminated by its plate-mates** (C1, measured below).
- **The frozen "epigenetic" class contains non-epigenetic compounds** (C2).

**For v11, one rule is missing.** Its ChEMBL target inputs and auxiliary head would consume the project's own held-out DTI
reference (C5).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | T-E design (H-E2's control; every class profile) | **The signed contrast subtracts plate-mates "of any compound", and T-E's classes share plates.**<br>**My counts (`split_cold_cell_1` training rows):**<br>• **JQ1(−):** **77 of 81** rows (those with ≥ 2 mates) have active BET inhibitors on their plate. BET inhibitors make up **21.7 %** of their mates on average, up to 100 %.<br>• So JQ1(−)'s contrast contains about −0.22 × the BET response. The inactive control would show a **sign-reversed copy of H-E2's effect**.<br>**Same-class mates attenuate every class profile:**<br>• BET inhibitors: 278 / 506 rows have another BET inhibitor on the plate (7.3 % of plate rows on average);<br>• HDAC inhibitors: 1,262 / 2,431 (3.3 %);<br>• DNMT inhibitors: 43 / 433 (0.2 %).<br>**The planted calibration is added after the contrast,** so it can't see this attenuation, and "a fail is bounded by the calibration" would overstate the bound. | **Register before T-E code:**<br>• for class K's rows (and JQ1(−)'s), take mates that exclude the whole BET pair and K's own class (simplest: exclude all epigenetic-class mates); for REF, exclude epigenetic mates;<br>• keep ≥ 2 mates;<br>• report, per class, the share of rows that lost mates or dropped out. |
| 2 | MINOR | frozen classes (§100.13; T-E sub-labels) | **The "epigenetic" class, the principal's key stratum, contains compounds that aren't epigenetic.** On T-C's test rows it has 470 rows over 30 compounds; my audit:<br>• **Wrong, through the regex** `demethylase\|methyltransferase`:<br>&nbsp;&nbsp;– bifonazole, "lanosterol 14-α demethylase" (an antifungal, CYP51): **15 rows**;<br>&nbsp;&nbsp;– tolcapone, "catechol O-methyltransferase" (COMT): **10 rows**;<br>&nbsp;&nbsp;– entacapone (COMT) is also classed epigenetic.<br>• **Sirtuin modulators and promiscuous natural products,** classed epigenetic through the epi list: resveratrol 27, curcumin 24, nicotinamide 13, EX-527 6 (**70 rows**).<br>• **JQ1(−),** the inactive enantiomer, is in the epigenetic class.<br>• **The HDAC sub-label** includes HDAC6-selective tubastatin-A (20 rows) and ACY-1215, and valproate (24 rows; mM-range HDAC inhibition).<br>So about a third of the stratum is wrong or weak. | **Fix and re-freeze before T-C, recording the diff:**<br>• restrict the regex to histone / DNA (cytosine-5)-methyltransferase / lysine demethylase / bromodomain (COMT and CYP51 move out);<br>• give sirtuin modulators and natural products their own label outside "epigenetic";<br>• move JQ1(−) out of the class;<br>• report HDAC with and without HDAC6-selective compounds and valproate;<br>• add apabetalone (a BD2-selective BET inhibitor, currently epigenetic without a sub-label) as a BET sensitivity. |
| 3 | MINOR | statistics (§100.13; Ask 1) | **Per-compound z uses a 4-df SE with normal p, then BH over 1,014 compounds.**<br>• An SE from 3 + 3 slots is often underestimated. At z = 2, the normal p is 0.046; t with 4 df gives 0.12.<br>• BH over a thousand tests turns that tail into false discoveries.<br>• **Slot permutation can't give per-compound p:** 6 slots allow 10 balanced relabellings, so the smallest p is 0.1. | **Two changes:**<br>(i) a **pooled SE**: fit the per-slot compound-mean variance as a function of row count across all compounds, use it for every compound, with a t reference;<br>(ii) a **list-level calibration**: rerun the whole BH procedure under each of the 10 balanced arm relabellings of the 6 slots, and report the discovery count under each beside the real list. |
| 4 | MINOR | T-E estimand (Ask 2) | **(a) A gene-level association can leak into the pass.**<br>• U is adjusted for its leave-cell-out gene mean, but m isn't. With noisy LCO means the adjustment is attenuated, leaving some gene-level response in U, which correlates with m's gene-level part.<br>• So a pass could reflect "commonly marked genes respond", not cell-specific marks.<br>• The cell-swap null diagnoses this, but it is reported, not ruled.<br>**(b) The gene set per cell isn't stated.** Masked genes carry m = 0, so including them would correlate the mask pattern with U.<br>**(c) H-E3's basal adjustment is linear,** but L1000's floor near the detection limit is not. Within the bottom tercile, floor effects on R (repression can't go below floor) could bias r(ACC, I) − r(ACC, R). | **(a)** Add the gene's leave-cell-out mean of m as a third covariate, or make "real r beyond the cell-swap null's 95th percentile (predicted direction)" part of the pass.<br>**(b)** State that each cell's correlation runs over genes with the matched mark measured (mask = 1).<br>**(c)** Adjust b flexibly within the silent tercile (decile indicators or a spline). |
| 5 | MAJOR | v11 outline: knowledge rule missing | **ChEMBL targets as inputs, and a molecule → target head, would consume the project's held-out DTI reference.**<br>• `pretrain_gene_vectors.py` deliberately excludes DTI, because `drug/outputs/dti/dti_reference.tsv` is "this project's HELD-OUT validation set". It is the interpretability probe (does atom → gene attention land on a drug's real targets?).<br>• That reference merges **ChEMBL** (`chembl_actions`, `chembl_direct`) and STITCH evidence.<br>• v11's rules 1–3 don't address this. Once ChEMBL targets are inputs or labels, the probe is no longer held out. | **Before registering v11, choose:**<br>(a) split the reference by compound: a held-out compound set whose targets never enter inputs, the graph or the auxiliary labels, with the probe scored only there;<br>(b) retire the probe explicitly, with the reason;<br>(c) find a probe source independent of ChEMBL. STITCH isn't one: it imports ChEMBL. |
| 6 | MINOR | v11 outline: provenance (Ask 3) | **The local STRING file is `9606.protein.links.v12.0.txt`, which has only `combined_score`.**<br>• So the current 929k-edge graph (≥ 400) **and the pretrained gene vectors** (link prediction on that graph) include STRING's coexpression and text-mining channels.<br>• That isn't a test-outcome leak, since the evidence is aggregated over compendia. But rule 1 can't be applied without the per-channel file.<br>• **The GO `.gmt` has no evidence codes.** IEP / HEP annotations come from expression patterns, which can include perturbation-signature studies. | **For v11:**<br>• use `protein.links.detailed.v12.0` (a download, needing approval), and keep the experiments + database channels only;<br>• rebuild the gene vectors on that graph;<br>• take GO from the GAF, excluding IEP / HEP (and state the policy on IEA);<br>• record versions.<br>**Disclose that v9 / v10-C used the combined score.** |

## Answers to the asks

**Ask 1 — §100.13:**
- **The z with BH:** change it as C3 says. Per-compound slot permutation is too coarse to be useful.
- **Report both noise measures, each labelled:**
  - the class-level compound-cluster bootstrap CI captures compound heterogeneity, not seed noise;
  - z captures seed noise.
- **The classes:** acceptable after C2's fixes.
- **State one limit:** "unannotated" is 11,067 of 21,151 rows (52 %), so the class strata describe under half of T-C.
- **I checked** that the 170 test rows without a class are exactly the featureless drop (21,321 − 21,151).

**Ask 2 — §101:**
- **The estimand is sound with C4.**
- **The 5-row minimum is fine.** Noise in U attenuates r toward 0 rather than biasing it, and the calibration at the same row
  counts shows the power.
- **The > 50 % mask rule is fine,** with C4(b).
- **H-E2's sign: yes, negative a priori.**
  - BET inhibition preferentially lowers genes under broad / super-enhancer H3K27ac.
  - With basal expression and the gene mean removed, what remains is the classic cell-specific form of that prediction: a
    cell's own super-enhancer genes are the ones lost.
- **H-E1 caveat:** DNMT inhibitors demethylate through replication. At L1000's 6–24 h, the expected effect is small, so a fail
  mainly bounds an *early* effect. Word it that way.
- **C1 must be fixed before H-E2's control means anything.**

**Ask 3 — v11:**
- **The knowledge rules aren't sufficient as written.** Add:
  - C5, the DTI probe;
  - C6, the STRING channels, GO evidence codes and vector rebuild;
  - that the "targets predicted from structure" model trains on ChEMBL bioactivity **excluding** the test compounds and their
    same-molecule duplicates (96.6).
- **CCLE:** check coverage for the 7 clean test cells. CD34 is a primary population and won't be in CCLE. Pre-register
  missingness as for chromatin, and report per cell.
- **The arms essential for attribution:**
  - **chromatin gate on / off** on the full graph (off = cell-invariant, the §100.4 convention): the programme's question;
  - **target signal on / off** on the full graph: the knowledge prior;
  - **CCLE on / off**: it can substitute for x_cell and for chromatin.
- **"Known against predicted targets"** is an evaluation setting on unseen compounds, not an arm.
- **"Full against subgraph"** conflates graph extent with whether non-landmark targets can be represented at all. If kept,
  run both arms with targets off.
- **Feasibility:** per-sample d-dimensional messages over 929k edges at batch 48, d 128, fp32 come to ≈ 23 GB per round
  for the edge tensor alone. That exceeds a T4's 16 GB.
  - The module needs per-cell shared propagation, low-dimensional messages, or a neighbourhood restriction.
  - A timing kernel should come before any arm.
  - At 30 GPU-h a week, the arm count sets the timeline (3 arms × 3 seeds × more than 3.5 h).
- **The prior** from T-A / T-B / §83 is low. Register the strata and the power statement with the arms.

## What I checked and found sound

- **§100.13 and §101 in full.**
- **`compound_classes.py`:**
  - the rules, the priority, the sub-label logic, `SIRTUIN_OR_WEAK`;
  - the deletion of the "other epigenetic by name" sub-labels.
- **`compound_classes.json` against T-C's test rows:** the epigenetic class (470 rows / 30 compounds, matching §100.13), with
  each member's source (list, name or regex).
- **The 170 unclassified test rows** = the featureless drop.
- **Plate co-occurrence of T-E's classes** on `cold_cell_1` training rows (C1's numbers).
- **The STRING file's columns and the ≥ 400 cut** (`build_priors_v9.py`).
- **The gene-vector pretraining docstring** (DTI excluded, and why).
- **`dti_reference.tsv`'s columns** (ChEMBL + STITCH evidence).

## What I could not assess, and why

- **T-E code and v11 code.** None exists.
- **`tb_strata.py` (9d7d69c).** It is exploratory and not in this packet.
- **CCLE coverage of the test cells.** It hasn't been downloaded.
- **STRING's per-channel scores.** They aren't in the local file.
