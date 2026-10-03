# What transfers to unseen cell lines in drug-response prediction: a pre-registered head-to-head with a state-of-the-art model, and a dissection of what does not

*Draft v2, 2026-09-30. Every number cites its record ([§n] = `model/results/RESULTS.md`, [C n] = `model/results/CLAIMS.md`).
Sections marked ⏳ wait for results that are pre-registered but not yet run (P7, V2, §88). Wording marked "permitted" is fixed by
the adversarial review cited beside it and is not to be strengthened.*

---

## Abstract

Predicting the transcriptional response of an **unseen cell line** to a drug is the hardest and most useful split of the LINCS L1000
problem: published cold-cell correlations range from 0.195 to 0.383 [§46.2]. We report four things. **(i)** We trained a published
state-of-the-art model, XPert (Nature Machine Intelligence 2025), to its published recipe on its own cold-cell split; it reproduces
scores 0.386 on the test rows, inside its published 0.383 ± 0.027 [§87]. **(ii)** Against it, on 21,151 identical test rows, our
model v9 is higher on 5 of 8 held-out cell lines and lower on 3 (two beyond row-level noise); row-pooled it scores 0.473 against 0.387,
a figure dominated by MCF7 (51 % of rows). The pre-registered cell-level criterion was not met, so we make no claim that v9 generalises
better [§87; C 1.11]. A registered second comparison, with v9 selected on held-out training cells (three seeds), gives the same
reading: higher on 6 of 8 cell lines and lower on 2 (CD34, H1975), against the 7 the criterion needs [§85.14].
**(iii)** A dissection and a pre-registered development protocol on held-out training cells show that several components the field
treats as load-bearing — chromatin features, protein-interaction message passing, a named pathway layer, and ranking and reweighted
losses at their pre-registered weights — add nothing detectable or harm; atom-level drug tokens are the exception (a model trained
without them is worse, −0.0086, although removing them from one trained model at inference helped). One auxiliary objective (a
per-gene direction head) is accepted, with no detectable change in the drug-specific component. A seed-specific prediction component
(+0.029 from averaging three seeds, at three times the training compute) is larger than any architectural effect we measured;
a snapshot ensemble within one run recovers 0.57 of it (+0.017, accepted), inference-time dropout averaging 3–4 %
[§37, §55, §85.8, §85.9, §90.7, §90.8]. For chromatin, a test calibrated by planting effects of known size into
the real data rules out a chromatin gain on the drug's mean response explaining ≥ 0.5 % of the cell-specific residual
(drug-specific shifts: ≥ 5 %), of forms linear in the encoded tracks, on six dev cells; replacing each cell's chromatin with
the training mean at inference raises the dev-cell score by 0.006 on balance (a diagnostic) [§91].
**(iv)** For
interpretability, attention and gradient readouts do not recover annotated drug mechanism beyond calibrated nulls [C 4.1a, §86.4];
a shared linear readout of the named pathway nodes, trained to predict each pathway's response magnitude, ranks which pathways move
in unseen test cell lines (ρ 0.295 against 0.241 for a cell-agnostic training-row prior, on every seed, with no permutation nulls;
above the readout for other cells in all 8 test cells) [§85.10, §85.14, C 4.16]. ⏳ *[§88: a drug-dependent pathway readout against annotated mechanism.]* We also document how the benchmark's own numbers
are produced — a released checkpoint scored on folds it was trained on, and checkpoint selection on the test fold — and seven
methods lessons, each with its measurement.

## 1. Introduction

The use case for a drug-response model is a new patient-derived or clinical line: a cell the model has never seen. L1000 benchmarks
report this "cold-cell" split alongside warm splits, and it is consistently the weakest [§46.2]. Two recent results make a plain
leaderboard comparison insufficient. Linear baselines match deep models on several perturbation tasks (Ahlmann-Eltze et al. 2025),
and seven L1000 models were shown to ignore their own drug features (Bai et al. 2026) [§50]. A reported gain therefore needs two
controls: a like-for-like comparison with the competitor on the same rows, and a dissection showing which components carry it.

Most comparisons quote a competitor's published number against a re-implementation. We instead **trained the competitor ourselves**,
to its published recipe, on its own split, and compared per row under rules committed before training. The head-to-head (§71, §87),
the development protocol (§85, including P7) and the interpretability tests (§86, §88) were pre-registered in a public log and
reviewed by a blinded adversarial reviewer; the dissection (§37), the warm split (§43), the benchmark forensics (§42, §46) and the EMA
results (§21, §38.4) predate that protocol and are reported as exploratory [§4].

## 2. Data and benchmark

**Data.** XPert's released L1000 benchmark (`l1000_mdmt`, 68,830 signatures) with its splits; the target is the Level-3 delta
(treated minus matched control) on the 978 landmark genes, in XPert's own convention [§23–24, C 6.6]. The metric is the mean over
rows of the per-row Pearson correlation between predicted and measured delta (XPert's `metrics.py`). Our substrate matches theirs on
99.65 % of rows [§27].

**The cold-cell split.** `split_cold_cell_1` holds out 8 cell lines (21,321 test rows); 170 test rows whose compound cannot be
featurised are dropped for both models, leaving 21,151 paired rows [§87]. MCF7 is 51 % of them, so a row-pooled number is mostly a
statement about one cell line; our estimand is per cell line, with a cluster bootstrap over cells [§71].

**Benchmark forensics** *(reported as usage notes, not criticism).* Three properties of the benchmark shape any number measured on it.
- *The released warm checkpoint belongs to one fold.* Scored on all five warm folds, it gives delta Pearson 0.694 on `split_2` and
  0.738–0.744 on the other four, which cluster within ≈ 0.005 — the signature of one genuinely held-out fold and four whose test rows
  it largely trained on (each fold's training set is 80 % of the corpus) [§42]. Scores of that checkpoint on folds other than
  `split_2` are partly in-sample.
- *The published recipe selects its checkpoint on the test fold.* No split in the released data has a validation level, so the code's
  fallback sets the validation set to the test set, and early stopping and checkpoint selection monitor test loss [§69]. We ran the
  recipe as published and disclose this asymmetry; it favours XPert.
- *The published numbers exist.* XPert's per-scenario means are in its Supplementary Table R8 [§46]; our ridge baseline reproduces
  the scale of their TranSiGen column (0.296 against 0.293) [§46.4].

## 3. Models

**v9** (`model/v9/`; architecture in `MODEL_MATH.md`): gene tokens carrying the control profile and gene identity, control encoders,
FiLM conditioning on dose and time, base transformer blocks, a STRING message-passing step, a named pathway readout (800 Reactome /
GO:BP nodes; pre-drug, hence drug-invariant), perturbation blocks in which the gene tokens cross-attend to drug tokens (a global
drug vector plus Uni-Mol atom tokens), and delta / absolute heads; Huber + correlation losses with auxiliary pathway and chromatin
heads [§37, §85.7].

**XPert as published**, run by us with the deviations needed to train it on Kaggle T4s, each proven equivalent where it touches
numerics: a flash-attention shim, one tensor per drug instead of per row in their dataset (values proven identical), the Uni-Mol
array rebuilt from their release, DataParallel over two GPUs (gradients equal to one GPU to 5e-16; replica dropout masks were
duplicated in epoch 0), the ten parameters their loss never uses frozen, and full-state checkpoint/resume across sessions (bitwise) [§87, run record]. Activation checkpointing was **not**
applied in the final run [§87 corrected].

## 4. Pre-registration and statistics

For the pre-registered parts (§71, §85–§90) every reading was committed before its data, with the git order as witness; amendments
made after a baseline was measured but before any variant are marked as such (§85.10), and one reader was committed seconds after its
outputs arrived (review 023). A blinded adversarial reviewer — a separate model session given the objective, code and results but never
the author's reasoning — reviewed every pre-registered design and result: 161 challenges across 36 reviews at this draft, each adjudicated in the log
(`orchestration/bus/adjudicated/`, counted as table rows). The dissection, warm split, forensics and EMA results predate the protocol
and are exploratory. **The cold-cell test cells had been read before the protocol existed** (§44–§46.4); from the protocol on, model
development used a **dev carve** of six training cells [§85], and the one later test-cell comparison (P7) is labelled *"dev-selected
increments on a baseline partly chosen with test-cell knowledge"*. The cell line is the unit for any claim about unseen cells: we report
per-cell medians of row differences, their unweighted mean with a cluster bootstrap over cells, and a sign count [§71.2].
Interpretability readouts were calibrated against untrained initialisations (§86, §88) or against permutation and cell-agnostic
references (§85.10); the exact trained permuted-drug null was not run.

## 5. Results

### 5.1 The cold-cell head-to-head (Figure 1)
XPert trained to its recipe reached its best checkpoint at epoch 40 and early-stopped at epoch 90 (patience 50); on all its test rows it
scores **0.386**, inside its published 0.383 ± 0.027 [§87]. *Permitted wording (review 022):* **"v9 had higher per-row delta
correlation on 5 of 8 cell lines, including the five with the most test rows, and lower on 3 (CD34 and H1975 beyond row-level noise;
BJAB tied). The pre-registered criterion — a cluster mean above zero with a cluster confidence interval excluding zero and at least
7 of 8 cell lines favouring v9 — was not met, so we make no claim that v9 generalises to unseen cell lines better than XPert.
Row-pooled, v9 scores 0.473 against 0.387, a figure dominated by MCF7 (51 % of rows)."** The cluster mean of per-cell differences is
+0.0465 (percentile cluster bootstrap over 8 cells [0.0048, 0.0861], which undercovers at this n; t-interval
[−0.0064, +0.0994]). Disclosures: XPert's checkpoint was selected on test loss (§2), and its best checkpoint (epoch 40) came before
the recipe's objective switch at epoch 70; v9 trained a fixed 12 epochs; the compared v9 is one of two cold-cell variants whose test
scores had been seen (bound 0.0042 row-pooled, ≈ 0.0004 on the cluster estimand); the earlier v9 runs used identical dropout
masks on both GPUs, a defect found and fixed during this work [C 6.13].

**5.1b P7 — the registered second comparison (Figure 8)** (§85.12, §85.14). The dev-selected v9 (P2 + sign head + snapshot
ensemble) was trained on all 32 training cells with 3 seeds and scored once, blinded, on the same 21,151 rows, with readings fixed
in advance. *Proposed wording:* **"A dev-selected v9 is not shown to generalise better than XPert at the cell level: it is higher
on 6 of 8 unseen cell lines (BJAB's interval includes 0) and lower on 2 (CD34 and H1975, beyond row-level noise), and the
pre-registered criterion needs 7."**
- **The cluster mean** of per-cell differences is +0.0503: percentile cluster bootstrap over 8 cells [+0.0107, +0.0883], which
  undercovers at this n; t-interval [−0.0002, +0.1009]. §87's is +0.0465 (bootstrap [+0.0048, +0.0861]; t [−0.0064, +0.0994]).
  P7 differs from §87, one run of the pre-development v9, by +0.004, within the spread of P7's own seeds (+0.048 to +0.054).
  The same two cells favour XPert in both comparisons, both against the one XPert run.
- **Secondary rows (never in the verdict):**
  - per seed +0.054 / +0.048 / +0.049, each 6 of 8;
  - row-pooled 0.481 against 0.387, dominated by MCF7;
  - the final snapshot alone, +0.042 and 5 of 8.
- **The cell-centred score** (§85.10) is v9 0.496 against XPert 0.463. On the cell-centred score, which removes each cell's mean response (the part a model must predict for an unseen
  cell) from truth and prediction alike, v9 is above XPert in all 8 cells, CD34 and H1975 included. This is consistent with
  v9's deficit in those two cells lying in its predicted mean response for the cell. It is a registered secondary, is not a
  measure of cell-level generalisation, and licenses no claim.

### 5.2 The warm split
On the fold XPert's released checkpoint was trained on (`split_2`), v9 is ahead on all 8 metrics (delta Pearson +0.012 [0.011, 0.013],
identical rows; gaps from +0.0007 to +0.023), three v9 seeds against XPert's released checkpoint [§43]; exploratory (it predates the
protocol), and those v9 runs used the dropout-mask defect above [C 6.13].

### 5.3 What does not transfer (Figure 3)
**Dissection of the committed v9** (exploratory): STRING message passing and the named pathway layer contribute ≈ 0 to accuracy when
ablated to the mean in a trained model [§37]. Chromatin was tested by retraining: an arm without cell-specific chromatin (one run
each, scored on the cold-cell test cells, §45) differs by +0.0004 on the cluster estimand over the five covered cells; with one run per arm
that rules out only effects larger than ≈ 0.006–0.016 [§91.10], after an earlier row-bootstrap +0.0042 was retracted [§51–55]. **The development screens** (dev
carve, rules committed first; Figure 3) [§85.8]: removing atom tokens during training (−0.0086; one seed, dropped at the screen),
removing the control encoder (+0.0016, below the advance threshold; one seed), a ListNet ranking loss and a DEG-reweighted loss at their
pre-registered weights (−0.0205 and −0.0042; one seed each), chromatin-gated union-graph edges (+0.0020 at three seeds, below its threshold 0.0042) and a post-drug pathway layer (+0.0026 at three seeds, *"indistinguishable
from the baseline"*, review 027) are not accepted. **One candidate is accepted:** a per-gene direction (sign) head, +0.0048 at three
seeds, 4 of 6 dev cells, with the interpretability gate held (aux alignment 0.262 against a floor of 0.253; a decline of 0.011
from the baseline, within the gate's margin) [§85.8, §85.11]. *Permitted wording (§85.10):* **"improves the per-row score, with
no detectable change in the drug-specific (cell-centred) component (Δ_c = +0.0002)"** — it is never described as improving
drug-specific prediction. For atom tokens an inference ablation (one fold-0 model, drug self-attention off: removing them helped, the model lost 0.007 with
them on unseen cells) and a retraining ablation (one dev-carve seed: a model trained without them is 0.0086 worse) disagree in sign;
they differ in split, estimand and seed count, so we report this as an observation, not a finding [§37, §85.2 rule 9, §85.8].

**Chromatin, with a test calibrated before it was read (Figure 9).** The earlier training-ablation null (one run per arm) could
not have seen an effect smaller than ≈ 0.006–0.016, and the sign head we accepted moved the score by +0.0048 [§91.10]. We
therefore pre-registered a closed-form funnel on the dev carve and calibrated it before reading it [§91.2–§91.11].
- **The tests.** The main test (T1) asks whether a gene's chromatin in a new cell predicts how that cell's response to a drug
  departs from the drug's mean response in other cells. It is a drug-conditioned, gene-local hierarchical ridge, fitted only on
  dev-training cells. Two cheaper strategies were tested beside it: retrieving responses from cells with similar chromatin (T2),
  and predicting which genes can move in a cell (T3).
- **The calibration.** Effects of known size were planted into the real data, carried by tracks with the real ones' availability
  and shape. A chromatin gain explaining 0.5 % of the cell-specific residual, the smallest size planted, moved T1's score by
  +0.006 to +0.009 and was detected in 5 of 5 draws (the bar is +0.004). These are Δs in the funnel's own score (B0 0.142),
  not directly comparable with v9's (≈ 0.44). With nothing planted T1 never passed (|Δ| ≤ 0.0003, the spread over five
  synthetic features on the same cells).
  Drug-specific shifts were detected at 5 % of the residual (+0.005 to +0.007), not at 2 % (+0.002 to +0.003).
- **The reading: nothing advances.** T1's null is informative for the gain form: *no chromatin gain on the drug's mean
  response (drug-conditioned, gene-local) explaining ≥ 0.5 % of the cell-specific residual, of a form linear in the encoded
  tracks, on these dev cells*. Drug-specific shifts are bounded only at 5 %. T2's null is uninformative, and T3's positive control failed, so neither is interpreted [§91.12].
- **What chromatin does carry is gene-generic.** Beyond basal expression it adds +0.0014 on the drug-known dev rows, positive in
  3 of 6 cells and carried by LNCAP and HL60. Giving every cell the same mean chromatin keeps all but 0.0002 of that, and giving
  each dev cell another dev cell's chromatin costs 0.0003. This is the kind of per-gene content a gene embedding can represent;
  whether v9's does is not tested [§91.12].

**v9 reads its chromatin; on the dev cells, reading it costs accuracy on balance.** In the three dev baseline models,
replacing each cell's chromatin with the dev-training mean at inference (which keeps the gene-generic part) raises the dev
score by +0.0060, on every seed. The cell-centred gain is +0.0056, so the cost sits in the drug-specific (cell-centred)
component rather than in a per-cell offset. It is a net over cells that disagree: most of it sits in HEK293T +0.023, LNCAP
+0.027 and VCAP +0.014, while chromatin helps in HL60 (−0.013) and SKBR3 (−0.004), on every seed. For
VCAP the cost is its failed H3K27me3 track: ablating that track alone recovers 93 % of the gain. For HEK293T and LNCAP no cause
is identified [§91.11]. An inference-time ablation puts a model off its training distribution, so this is a diagnostic, not
a method, and the final model keeps chromatin as registered. ⏳ *[§92: one-seed dev screens retraining without cell-specific
chromatin (E2) and with a cleaned encoding (E1).]*

**Variance, not architecture (Figure 6).** Averaging the predictions of three seeds raises the dev score from 0.4369 to 0.4662
(+0.029), about 1.4× the largest candidate movement (C3, −0.0205) and 6× the largest gain (C6, +0.0048), and the gain survives
cell-centring (+0.027) [§85.9]. Seed predictions correlate
0.81 per row; an equicorrelated fit puts the infinite-ensemble ceiling at ≈ 0.48 (an explanation-level estimate, review 029). EMA was null twice [§21, §38.4], but under the WSD schedule its window lies in the annealed tail, so it does not test averaging along
the trajectory. **A snapshot ensemble within one run** (V2: three 4-epoch annealed cycles in place of one 12-epoch schedule, the
three end-of-cycle predictions averaged) scores +0.0167 over the baseline at three seeds, 6 of 6 dev cells, cell-centred +0.0170:
0.57 of the three-seed gain at one run's cost, accepted [§90.7]. Part of it is the schedule, not the averaging: the final snapshot
alone is +0.0077, and one 4-epoch annealed cycle already gives ≈ +0.005, so we claim no benefit from warm restarts (review 037).
**Averaging at inference does not substitute:** eight passes with dropout (V1-drop, +0.0009) or dropout plus stochastic depth
(V1-full, +0.0010), paired against the same checkpoint's deterministic pass, recover 3–4 % of the seed-ensemble gain and are not
accepted [§90.8].

**The final dev-selected model (P6).** The sign head and the snapshot ensemble together (three seeds) score +0.0184 over the
baseline, 6 of 6 dev cells, cell-centred +0.0133; pathway alignment 0.265, read on the final-snapshot weights (whose own accuracy is +0.0087); this passes the pre-registered stack rule, so its recipe,
retrained on all training cells, is what the second comparison scores against XPert [§85.11, §85.13]. It is **not shown to beat the snapshot ensemble alone**: seed-paired
differences are +0.004, −0.006 and +0.007, and on the cell-centred score the stack is 0.004 below V2 alone. The stack's per-row
edge over V2 (+0.0018) comes almost entirely from one dev cell (U937). With the sign head's Δ_c of +0.0002 when added alone,
this is consistent with its gain lying in each cell's mean delta profile rather than in each row's departure from it; the sign
head is in the final model by the pre-registered rule, not because it was shown to help on top of the snapshot ensemble [§85.13].

### 5.4 Interpretability (Figures 4, 7)
- **Atom→gene attention does not recover drug targets** (median rank percentile 0.560) [C 4.1a].
- **A gradient readout of the named pathway layer** is NULL [§86.4]; untrained models' readouts (the output projection) cleared the
  −0.02 floor with p < 0.05 on 2 of 3 initialisations, so the within-model permutation p is not calibrated across initialisations
  (review 023).
- **The named pathway readout ranks which pathways move in held-out cells** — stated with its controls (Figure 7; §85.10, C 4.16):
  *"a shared linear readout of the named pathway nodes, trained to predict each pathway's response magnitude, ranks which pathways move
  in held-out cells at ρ = 0.273, +0.044 above a cell-agnostic training-row prior (0.229)"*; its own readout beats the same model's
  readout for other cells in all 6 dev cells (licensed "in this cell"). These are the **dev baseline's** numbers (P2, §85.10);
  the final model, read on its final-snapshot weights, gives 0.265 and 5 of 6 dev cells (HL60 −0.0001) [§85.13].
  **On the unseen test cells** (P7's three checkpoints, read once, §85.14):
  - 0.292 / 0.304 / 0.290 against a training-row prior of 0.241 (all 32 training cells), so it beats the prior on every seed;
  - above the other cells' readout in **all 8 test cells**, licensed "in this cell" (a consistency filter, sign p 0.004);
  - no permutation nulls were run on the test rows. It is cell-level, drug-independent and supervised on the target
  it is scored against. An earlier "8–12 sd over a permutation null" for a channel-mean readout [§37] does not reproduce in these models
  (its sign is arbitrary) and its null did not control for a generic ranking; it is withdrawn as evidence of mechanism.
- ⏳ **A drug-dependent pathway readout** (C8b's post-drug layer, trained on fold 0, 3 seeds, 5 untrained calibrations) against
  annotated mechanism, SIGNAL / SEEN-ONLY / NULL by §88.3 with §88.6's wording [§88].

## 6. Methods lessons (each with its measurement)
1. **Chance for pathway alignment is not 0.5**: an untrained model scores 0.218 against a label-permutation null of 0.229 [C 4.15].
2. **A column-permutation null is not enough either**: a cell-agnostic prior reaches 0.229 of a readout's 0.273 [§85.10].
3. **Calibrate interpretability against untrained initialisations**: untrained models' output projections cleared the −0.02 floor
   with p < 0.05 on 2 of 3 initialisations [§86.4, review 023].
4. **Score the drug-specific component beside the raw score**: the accepted sign head moved the raw score +0.0048 and the cell-centred
   score +0.0002 [§85.10]; the seed ensemble moved both [§85.9]; adding the sign head to the snapshot ensemble left the raw
   score within seed noise of V2 alone and lowered the centred one by 0.004 [§85.13].
5. **DataParallel with a shared seed duplicates dropout masks** across replicas for a whole run [§85.5, C 6.13].
6. **A row bootstrap on a cell-level question licenses trivial effects**: the retracted chromatin claim was +0.0042 row-pooled against
   +0.00036 by cluster [§51–55].
7. **Read the supplementary before reverse-engineering a published number** [§46.1].
8. **Plant an effect of known size before reading a null**: the earlier training-ablation null could not have seen effects below
   ≈ 0.006–0.016, larger than the sign head we accepted (+0.0048). The calibrated funnel detects planted effects that move
   its score by ≈ 0.005 (in its own score, B0 0.142, not directly comparable with v9's), and with nothing planted it moves by
   ≤ 0.0003 [§91.10, §91.12].

## 7. Limitations
One cold-cell fold and one XPert run, so the intervals carry no XPert run-to-run variance (§87 is one v9 run; P7 adds three v9
seeds, whose per-seed cluster means span +0.048 to +0.054). Three of eight test cells have no chromatin, and 12 of the 26 dev-training cells have it. The chromatin tracks are gene-level
ATAC, H3K27ac and H3K27me3 at the landmark genes; the funnel's negative covers forms linear in those tracks on six dev cells. Landmark genes only. Most L1000 signatures are close to inert
(~75 %) [C 6.1], which bounds any per-row correlation. Every §85 increment was measured on six dev cells, against a baseline whose own selection saw test scores; with that few cells,
≥ 4 of 6 is a consistency filter, not a test. The test-cell pathway alignment is one measurement on P7's three seeds, without
permutation nulls.

## Figures
F1 per-cell head-to-head · F2 XPert reproduction · F3 dev screens · F4 gradient MoA probe with untrained calibration · F5 input
coverage (supplement) · F6 variance: seed, snapshot and inference-time ensembles · F7 pathway alignment against its references · F8 P7 head-to-head *(caption: per-cell bars are row bootstraps, row-level noise only, against one XPert run; the cluster
interval is a percentile cluster bootstrap over 8 cells, which undercovers (t-interval [−0.0002, +0.1009]); the criterion
needs ≥ 7 of 8 cells)* · F9 chromatin: the calibration, the closed-form gains, and T4/T4b *(caption: filled = T1's full rule, all five conjuncts;
the band is |Δ| because a log axis cannot show negatives; P3 is omitted because the centred conjunct always fails it,
although its raw Δ lies above the bar; the real-data mark is FBC − FB, gene-generic, and no planted size is implied)* · ⏳ dissection and
forensics figures.
