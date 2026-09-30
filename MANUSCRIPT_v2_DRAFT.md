# What transfers to unseen cell lines in drug-response prediction: a pre-registered head-to-head with a state-of-the-art model, and a dissection of what does not

*Draft v2, 2026-09-30. Every number cites its record ([§n] = `model/results/RESULTS.md`, [C n] = `model/results/CLAIMS.md`).
Sections marked ⏳ wait for results that are pre-registered but not yet run (P7, V2, §88). Wording marked "permitted" is fixed by
the adversarial review cited beside it and is not to be strengthened.*

---

## Abstract

Predicting the transcriptional response of an **unseen cell line** to a drug is the hardest and most useful split of the LINCS L1000
problem: published cold-cell correlations range from 0.195 to 0.383 [§46.2]. We report four things. **(i)** We trained a published
state-of-the-art model, XPert (Nature Machine Intelligence 2025), to its published recipe on its own cold-cell split; it reproduces
its published score (0.386 on the test rows against 0.383 ± 0.027) [§87]. **(ii)** Against it, on 21,151 identical test rows, our model
v9 is higher on 5 of 8 held-out cell lines and row-pooled (0.473 against 0.387), but the pre-registered cell-level criterion was not
met, so we make no claim that v9 generalises better [§87; C 1.11]. ⏳ *[P7: the registered second comparison with a dev-selected v9.]*
**(iii)** A pre-registered development protocol on held-out training cells shows that components the field treats as load-bearing —
chromatin features, protein-interaction message passing, a named pathway layer, atom-level drug tokens, ranking and reweighted losses —
add nothing or harm, while one auxiliary objective (a per-gene direction head) is accepted with no detectable change in the
drug-specific component; the largest lever we found is seed ensembling (+0.029), not architecture [§85.8, §85.9]. **(iv)** For
interpretability, attention and gradient readouts do not recover annotated drug mechanism beyond calibrated nulls [C 4.1a, §86.4];
a named pathway readout ranks which pathways move in held-out cells at ρ 0.273, of which only +0.044 exceeds a cell-agnostic prior
[§85.10]. ⏳ *[§88: a drug-dependent pathway readout against annotated mechanism.]* We also document how the benchmark's own numbers
are produced — a released checkpoint scored on folds it was trained on, and checkpoint selection on the test fold — and three
methods lessons for interpretability claims, each with its measurement.

## 1. Introduction

The use case for a drug-response model is a new patient-derived or clinical line: a cell the model has never seen. L1000 benchmarks
report this "cold-cell" split alongside warm splits, and it is consistently the weakest [§46.2]. Two recent results make a plain
leaderboard comparison insufficient. Linear baselines match deep models on several perturbation tasks (Ahlmann-Eltze et al. 2025),
and seven L1000 models were shown to ignore their own drug features (Bai et al. 2026) [§50]. A reported gain therefore needs two
controls: a like-for-like comparison with the competitor on the same rows, and a dissection showing which components carry it.

Most comparisons quote a competitor's published number against a re-implementation. We instead **trained the competitor ourselves**,
to its published recipe, on its own split, and compared per row under rules committed before training. Every reading in this paper
was pre-registered in a public log before its data existed, and each was reviewed by a blinded adversarial reviewer (160 of 163
challenges upheld at writing) [§4].

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
  0.738–0.744 on the other four, which cluster within 0.005 — the signature of one genuinely held-out fold and four whose test rows
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
array rebuilt from their release, DataParallel over two GPUs (gradients equal to one GPU to 5e-16), the ten parameters their loss
never uses frozen, and full-state checkpoint/resume across sessions (bitwise) [§87, run record]. Activation checkpointing was **not**
applied in the final run [§87 corrected].

## 4. Pre-registration and statistics

Every reading was committed before its data, with the git order as witness. A blinded adversarial reviewer — a separate model
session given the objective, code and results but never the author's reasoning — reviewed every design and result; challenges it
raised were adjudicated in the log (160 of 163 upheld at writing). The cell line is the unit for any claim about unseen cells: we
report per-cell medians of row differences, their unweighted mean with a cluster bootstrap over cells, and a sign count [§71.2].
Model development used a **dev carve** of six training cells, so the test cells were touched only by pre-registered comparisons
[§85]. Interpretability readouts were calibrated against untrained models, not against 0.5 or a trained null alone [§86, §88].

## 5. Results

### 5.1 The cold-cell head-to-head (Figure 1)
XPert trained to its recipe reached its best checkpoint at epoch 40 and early-stopped at epoch 90 (patience 50); on all its test rows it
scores **0.386**, inside its published 0.383 ± 0.027 [§87]. *Permitted wording (review 022):* **"v9 had higher per-row delta
correlation on 5 of 8 cell lines, including the five with the most test rows, and lower on 3 (CD34 and H1975 beyond row-level noise;
BJAB tied). The pre-registered criterion — a cluster mean above zero with a cluster confidence interval excluding zero and at least
7 of 8 cell lines favouring v9 — was not met, so we make no claim that v9 generalises to unseen cell lines better than XPert.
Row-pooled, v9 scores 0.473 against 0.387, a figure dominated by MCF7 (51 % of rows)."** The cluster mean of per-cell differences is
+0.0465 [0.0048, 0.0861]. Disclosures: XPert's checkpoint was selected on test loss (§2); v9 trained a fixed 12 epochs; the compared v9
is one of two cold-cell variants whose test scores had been seen (bound 0.0042 row-pooled); the earlier v9 runs used identical dropout
masks on both GPUs, a defect found and fixed during this work [C 6.13].

⏳ **5.1b P7 — the registered second comparison** (§85.12). The dev-selected v9 (P2 + C6 [+ V2]) on all 32 training cells, 3 seeds,
scored once on the same 21,151 rows; readings fixed in advance: v9 wins → *"a dev-selected v9 generalises better than XPert as
published"* (secondary, labelled *"dev-selected increments on a baseline partly chosen with test-cell knowledge"*); XPert wins →
*"uninterpretable as a model comparison"*; otherwise no claim. Secondary rows: the cell-centred score for both models, v9's own seed
ensemble (never set beside XPert's single run), and a V2-last row if V2 is used [§90.6].

### 5.2 The warm split
On the fold XPert's released checkpoint was trained on (`split_2`), v9 is higher by +0.012 on all 8 metrics [§43]; the earlier v9
used the dropout-mask defect above [C 6.13].

### 5.3 What does not transfer (Figure 3)
**Dissection of the committed v9** (ablations of a trained model to the mean) [§37, §55]: chromatin summed into gene tokens changes
the cold-cell score by +0.0004 on the right (cluster) estimand — a clean null, after an earlier row-bootstrap +0.0042 was retracted
[§51–55]; STRING message passing and the named pathway layer contribute ≈ 0 to accuracy [§37]. **The development screens** (dev
carve, rules committed first; Figure 3) [§85.8]: removing atom tokens during training (−0.0086), removing the control encoder
(+0.0016, below the advance threshold), a ListNet ranking loss (−0.0205), a DEG-reweighted loss (−0.0042), chromatin-gated union-graph
edges (+0.0020 at three seeds, below its threshold 0.0042) and a post-drug pathway layer (+0.0026 at three seeds, *"indistinguishable
from the baseline"*, review 027) are not accepted. **One candidate is accepted:** a per-gene direction (sign) head, +0.0048 at three
seeds, 4 of 6 dev cells, with the interpretability gate held (aux alignment 0.262 against a floor of 0.253; a decline of 0.011
from the baseline, within the gate's margin) [§85.8, §85.11]. *Permitted wording (§85.10):* **"improves the per-row score, with
no detectable change in the drug-specific (cell-centred) component (Δ_c = +0.0002)"** — it is never described as improving
drug-specific prediction. Atom tokens show an **inference-versus-training reversal**: in the committed model, removing them at inference
*helped* on unseen cells (the model lost 0.007 with them), but a model trained without them is 0.0086 worse [§37, §85.2 rule 9 vs
§85.8].

**Variance, not architecture (Figure 6).** Averaging the predictions of three seeds raises the dev score from 0.4369 to 0.4662
(+0.029), about 3.4× the largest candidate effect, and the gain survives cell-centring (+0.027) [§85.9]. Seed predictions correlate
0.81 per row; an equicorrelated fit puts the infinite-ensemble ceiling at ≈ 0.48 (an explanation-level estimate, review 029). Weight averaging along one trajectory (EMA) was
null twice [§21, §38.4], and averaging dropout masks at inference recovers ~3 % of the gain (+0.0009) [§90.3]. ⏳ *[V2 — a snapshot
ensemble within one run — and V1 with stochastic depth, §90.]*

### 5.4 Interpretability (Figures 4, 7)
- **Atom→gene attention does not recover drug targets** (median rank percentile 0.560) [C 4.1a].
- **A gradient readout of the named pathway layer** is NULL: calibrated against untrained models, untrained inits reach the same floor
  on 2 of 3 initialisations, so the within-model permutation p is not calibrated across initialisations [§86.4].
- **The named pathway readout ranks which pathways move in held-out cells** — stated with its controls (Figure 7; §85.10, C 4.16):
  *"a shared linear readout of the named pathway nodes, trained to predict each pathway's response magnitude, ranks which pathways move
  in held-out cells at ρ = 0.273, +0.044 above a cell-agnostic training-row prior (0.229)"*; its own readout beats the same model's
  readout for other cells in all 6 dev cells (licensed "in this cell"). It is cell-level, drug-independent and supervised on the target
  it is scored against. An earlier "8–12 sd over a permutation null" for a channel-mean readout [§37] does not reproduce in these models
  (its sign is arbitrary) and its null did not control for a generic ranking; it is withdrawn as evidence of mechanism.
- ⏳ **A drug-dependent pathway readout** (C8b's post-drug layer, trained on fold 0, 3 seeds, 5 untrained calibrations) against
  annotated mechanism, SIGNAL / SEEN-ONLY / NULL by §88.3 with §88.6's wording [§88].

## 6. Methods lessons (each with its measurement)
1. **Chance for pathway alignment is not 0.5**: an untrained model scores 0.218 against a label-permutation null of 0.229 [C 4.15].
2. **A column-permutation null is not enough either**: a cell-agnostic prior reaches 0.229 of a readout's 0.273 [§85.10].
3. **Calibrate interpretability against untrained initialisations**: untrained models reached the gradient readout's floor on 2 of 3
   inits [§86.4].
4. **Score the drug-specific component beside the raw score**: the accepted sign head moved the raw score +0.0048 and the cell-centred
   score +0.0002 [§85.10]; the seed ensemble moved both [§85.9].
5. **DataParallel with a shared seed duplicates dropout masks** across replicas for a whole run [§85.5, C 6.13].
6. **A row bootstrap on a cell-level question licenses trivial effects**: the retracted chromatin claim was +0.0042 row-pooled against
   +0.00036 by cluster [§51–55].
7. **Read the supplementary before reverse-engineering a published number** [§46.1].

## 7. Limitations
One cold-cell fold; one XPert run and, in §87, one v9 run, so the row-bootstrap intervals carry no run-to-run variance (P7 adds three
v9 seeds, not XPert seeds). Three of eight test cells have no chromatin. Landmark genes only. Most L1000 signatures are close to inert
(~75 %) [C 6.1], which bounds any per-row correlation. Every candidate was screened on six dev cells; with that few, ≥ 4 of 6 is a
consistency filter, not a test.

## Figures
F1 per-cell head-to-head · F2 XPert reproduction · F3 dev screens · F4 gradient MoA probe with untrained calibration · F5 input
coverage (supplement) · F6 seed ensemble · F7 pathway alignment against its references · ⏳ F8 P7 head-to-head · ⏳ dissection and
forensics figures.
