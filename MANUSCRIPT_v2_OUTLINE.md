# Manuscript v2 — outline (2026-09-25). Replaces the v5-era MANUSCRIPT.md framing; every claim cites its record.

**Working title:** *What transfers to unseen cell lines in drug-response prediction: a pre-registered head-to-head against
a state-of-the-art model, and a dissection of what does not.*

**The contribution, as measured (not as hoped):** a benchmarking-and-dissection paper. (i) The first head-to-head in which
a published SOTA model (XPert, Nat. Mach. Intell. 2025) is trained by independent investigators to its published recipe
on its own cold-cell split and compared on identical rows under rules committed before training; (ii) forensics of how
that benchmark's numbers are produced; (iii) pre-registered nulls for components the field treats as load-bearing; (iv)
methods lessons for interpretability claims. The mechanism line (chromatin gating edges, a drug-dependent pathway
readout) enters only if §89 / §88 read positive.

---

## Abstract (skeleton — numbers fixed by the record)
- Task, data (LINCS L1000, Level 3 / XPert's L1000_mdmt), split (8 held-out cell lines).
- XPert trained by us to its recipe: 0.386 on the test rows vs its published 0.383 ± 0.027 [RESULTS §87, §71.4].
- v9 vs XPert on 21,151 identical rows: row-pooled 0.473 vs 0.387; higher on 5 of 8 cell lines; the pre-registered
  cell-level criterion (≥ 7 of 8 with a cluster CI excluding 0) was not met — **no claim of better generalisation**
  [§87; CLAIMS 1.11; wording fixed by review 022 ask 4].
- Dissection: chromatin summed into gene tokens, STRING message passing and a named pathway layer contribute nothing
  to accuracy; atom tokens hurt when removed at inference but help when the model is trained without them removed
  [§55, §37, §85.8 C1].
- Interpretability: neither atom→gene attention nor a gradient readout of the pathway layer recovers annotated drug
  mechanism beyond calibrated nulls [CLAIMS 4.1a, §86.4]; [C8b result — pending, §88].

## 1. Introduction
- Unseen-cell generalisation is the use case (a new patient line) and the hardest split (published cold-cell PCCs
  0.195–0.383, Supp. Table R8 [§46.2]).
- Most comparisons quote published numbers against re-implementations; we retrain the competitor.
- Linear baselines match deep models on several perturbation tasks (Ahlmann-Eltze et al. 2025); seven L1000 models
  ignore their drug features (Bai et al. 2026) [§50] — so dissection is required, not optional.

## 2. Data and benchmark
- XPert's L1000_mdmt, their splits; our Level-3 substrate (99.65 % coverage) [§27]; the metric (mean per-row Pearson on
  the delta, their convention) [§23–24, CLAIMS 6.6].
- **Benchmark forensics (a result section of its own):** the released warm checkpoint belongs to `split_2`, so scoring it
  on other folds inflates 0.738–0.745 [§42, §46.3 — framed as a usage note, not a criticism]; their recipe selects its
  checkpoint on the TEST fold via the val = test fallback [§69]; their published numbers were in Supplementary Table R8
  [§46]. Our ridge reproduces their TranSiGen scale (0.296 vs 0.293) [§46.4].

## 3. Models
- v9 (architecture, `MODEL_MATH.md`); XPert as published with the declared deviations [§87 list: flash shim,
  MyDataset per-drug tensors, the unimol array, DataParallel with epoch-0 mask duplication, the frozen ten,
  full-state resume; activation checkpointing NOT applied — corrected].

## 4. Pre-registration and statistics
- Every reading committed before its data (git order witnessed); the critic loop (136 challenges, 133 upheld at writing);
  the cluster estimand over cell lines [§71.2]; the dev-cell protocol for model development [§85] (test cells touched
  once); calibration against untrained models for interpretability [§86, §88].

## 5. Results
### 5.1 The cold-cell head-to-head [§87] — Figure 1 (F1), Figure 2 (F2)
Permitted sentence (review 022): *"v9 had higher per-row delta correlation on 5 of 8 cell lines, including the five with
the most test rows, and lower on 3 (CD34 and H1975 beyond row-level noise; BJAB tied). The pre-registered criterion …
was not met, so we make no claim that v9 generalises to unseen cell lines better than XPert. Row-pooled … 0.473 against
0.387, dominated by MCF7 (51 % of rows)."* Must disclose: XPert's test-loss checkpoint selection and best epoch 40 before
its objective switch; v9's fixed 12 epochs and lockstep masks [CLAIMS 6.13]; the compared v9 was one of two cc1 variants
with test scores seen (bound: 0.0042 row-pooled).
### 5.2 The warm split [§43] — v9 +0.012 on 8/8 metrics on their released checkpoint's own fold; lockstep disclosure.
### 5.3 What does not transfer — Figure 3 (dissection; to be built)
chromatin (gene tokens) +0.0004 [§55]; STRING MP ~0 [§37]; pathway layer ~0 for accuracy [§37]; atom tokens: the
inference-vs-training reversal [§37 vs §85.8 C1]; the dev screens [§85.8, F3] — C1, C2, C3, C4 dropped; C8b not accepted
(*"indistinguishable from the baseline"*, review 027); C6 advanced at seed 0 with no detectable cell-centred gain (wording
fixed by §85.10); [C6 3-seed, C7 pending]. **Variance, not architecture:** a 3-seed prediction average gains +0.029 (≈ 3.4× the
largest candidate effect; survives cell-centring, +0.027) [§85.9]; V1/V2 pre-registered [§90]; no v9-ensemble row beside a
single XPert run [§90.6].
### 5.4 Interpretability — Figure 4 (F4)
atom→gene attention does not recover targets (median rank percentile 0.560) [CLAIMS 4.1a]; the pathway layer's alignment
is cell-level by construction [CLAIMS 4.16] and, stated honestly, modest: the aux readout ranks which pathways move in held-out
cells at ρ 0.273, **+0.044 above a cell-agnostic training-row prior (0.229)**, far above another cell's readout (≈ 0.10)
[§85.10]; §37's "8–12 sd" is caveated (channel mean sign not reproduced; its null did not control for a generic ranking); the gradient readout is NULL on 1 of 3 seeds, and untrained models reach the
floor on 2 of 3 inits — the within-model permutation p is not calibrated across initialisations [§86.4].
[C8b §88 — pending: SIGNAL / SEEN-ONLY / NULL, worded by §88.3's qualifiers.]
### 5.5 [Pending] the dev-selected v9 (P7), the registered second comparison, reported alongside 5.1 [§85.2 rule 10].

## 6. Methods lessons (short, each with its measurement)
0.5 is not chance for pathway alignment (0.218 vs 0.229 null) [CLAIMS 4.15]; **a column-permutation null is not enough either —
test against a cell-agnostic prior** (0.229 of 0.273) [§85.10, review 032]; **score the drug-specific component** (cell-centred)
beside the raw score [§90.2]; calibrate against untrained inits [§86.4];
DataParallel with shared seeds duplicates dropout masks [§85.6, CLAIMS 6.13]; a row-bootstrap CI on a cell-level
question licenses trivial effects (the retracted chromatin claim: +0.0042 row-pooled vs +0.00036 cluster) [§51–55].

## 7. Limitations
One fold, one XPert run and one v9 run in §87 (row-bootstrap CIs carry no run-to-run variance); 3 of 8 test cells have
no chromatin; landmark genes only; L1000 noise (~75 % inert signatures) [CLAIMS 6.1].

## Figures
F1 per-cell head-to-head · F2 XPert reproduction · F3 dev screens (grows) · F4 MoA probe with untrained calibration ·
F5 input coverage (supplement) · planned F6 dissection bar chart from §37/§55 values · planned F7 the forensics
(fold contamination of the released checkpoint, §42). Polish list: `model/figures/POLISH_TODO.md`.

## Gates before writing prose
§85 screens complete (C3, C6, C7, C8b) → P6/P7; §88 executed or skipped by its gate; each result packet reviewed.
