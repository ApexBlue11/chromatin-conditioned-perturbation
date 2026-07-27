# LINCS Model — Handoff for the next session (2026-07-25)

> **READ THIS FIRST, AND READ IT ADVERSARIALLY.** The previous session reached several *confident,
> wrong* conclusions that the user overturned by insisting on more careful tests. Treat every claim
> below as a hypothesis with evidence attached, not settled fact. Where a number is given, a better
> measurement can change it. Your job is to test, not to trust.

---

## 0. Operating rules earned the hard way (violate these and you WILL repeat my mistakes)
1. **Never judge anything on ALL signatures.** ~75% of LINCS perturbations are biologically INERT
   (no response to predict; replicate r≈0.07–0.14). Aggregate R²/correlations are dominated by that
   noise. **Always stratify by signature strength (mean|Y|≥1 ⇒ reproducible, replicate r≈0.5–0.62).**
   Diluting with inert sigs once **INVERTED the sign** of the epigenetics effect (+0.073 → −0.011).
2. **Compute the noise/reliability ceiling BEFORE chasing any gap.** Split-half replicate reliability
   tells you how much is even predictable. Example: the drug×cell "interaction" is 47.9% of raw
   variance but only ~38% reproducible ⇒ predictable target ≈18%, not 47.9%. Chasing 47.9% = fitting noise.
3. **Verify a measurement tests what it claims.** My "epi ceiling" measured a static per-(cell,gene)
   offset — the wrong quantity. My magnitude tests used |Y| — destroys the *signed* effect. Both gave
   false nulls.
4. **The data is GOOD.** Do NOT call LINCS L5 "bad data" (I did — wrong). It reproduces at r≈0.5–0.6 for
   real responses, matches CMap's ~40% gold rate, cross-phase agreement r=0.561. Low aggregate = inert
   compounds, not corruption. Pipeline verified bug-free (Y↔GCTX corr=1.000000, both phases).
5. **An improvement that only moves aggregate R² is not an improvement.** Check reproducible-subset
   metrics AND interpretability, not the lumped number.
6. **Interpretability is the objective, accuracy second (user).** A small, correctly-signed, highly
   significant mechanistic effect is a legitimate deliverable even if ΔR² is tiny.

---

## 1. Current state (v3, as of handoff)
Best model = v3 (reliability-weighted). Evaluated on REPRODUCIBLE signatures, cold (unseen) cells:
- **Pearson (median, per-signature) = 0.529 ; R² = 0.331.** Noise ceiling √0.51–0.62 ≈ 0.71–0.79 ⇒
  **~70% of achievable.** Still undertrained (budget-capped 7/12 epochs; a **resume run is in flight** —
  check `apexblue/lincs-train` status and collect `metrics.json`+`ckpt.pt`).
- corr(model, truth) overall = **0.746**; on the drug×cell interaction = **0.449** = **53% of the
  noise-corrected ceiling (0.842)** ⇒ real but moderate headroom on cell-specificity. DO NOT chase the
  raw 24.9%-vs-47.9% variance-share gap — ~62% of that 47.9% is noise.
- **Epigenetics CONTRIBUTES +0.073 R² (+28% rel.), fair test on reproducible sigs.** KEEP the branch.
- Baselines beaten (cold-cell MSE 1.55 vs Mean/Meancell 1.82, Meandrug 1.80).
- Per-cell cold R² spans 10× (HUVEC .22 → NPC .02): common cell types learnable, rare ones not.
  This is what the "phase gap" (P1 vs P2) actually is — composition, not batch effect.

Model = deterministic cross-attention predictor (NO VAE). Flow: EpiGate + **additive signed epi head**
→ gene tokens (+dose/time FiLM) → base encoder (gene↔gene, prior-biased) → perturb encoder
(atom→gene cross-attn + gene↔gene) → per-gene head. Full spec: `model/MODEL_MATH.md`. 27/27 unit tests.

---

## 2. What's built (all verified)
- **Data branches:** baseline `X_base` (83×978 CCLE) + 1719-cell background; epigenetics `E_final`
  (83×978×3 ATAC/H3K27ac/H3K27me3) + mask + reliability; network priors `A_copathway`/`STRING_adj`
  (978×978); target `Y` (312,438×978 L5, 0 NaN).
- **Drug branch (`drug/outputs/`):** 20 descriptors, 2048 ECFP4, UniMol CLS (512) + **per-atom tokens**
  (703,851×512 ragged + offsets), ChemBERTa (384). DTI reference `drug/outputs/dti/dti_reference.tsv`
  (ChEMBL+STITCH, 19,174 drug×gene edges) — **for validation, never in the loss.**
- **Model (`model/`):** config, modules, model, losses, data, train, diagnose, tests, MODEL_MATH.md.
- **sig_strength.npy** (mean|Y| per row) drives reliability weighting + reproducible eval.

---

## 3. Kaggle infrastructure (no local GPU/torch; everything runs there)
- Auth: `KAGGLE_CONFIG_DIR=C:/Users/Surya/.kaggle`, user `apexblue`. 30 GPU-h/week (was near cap 2026-07-24;
  check quota). **CPU kernels are FREE** (use for tests/diagnostics).
- Datasets: `lincs-train-bundle` (1.5GB, fp16 Y+atoms, all model inputs flat), `lincs-model-src`
  (code + sig_strength; re-version to ship code changes), `lincs-v2-ckpt` (holds the LATEST ckpt.pt for resume).
- Kernels: `lincs-train` (GPU train, resumes from ckpt-in-input), `lincs-diagnose` (CPU, the
  interaction/epi-ablation analysis), `lincs-model-tests` (CPU unit tests).
- **CLI GOTCHAS (all real, all cost me time):**
  - Pass **backslash** `-p` paths (forward slash corrupts upload staging on Windows).
  - Private datasets mount at `/kaggle/input/datasets/<user>/<slug>/` — use a **recursive glob finder**.
  - **P100 is UNUSABLE** (Kaggle torch = sm_70+, P100=sm_60). Use `machine_shape=NvidiaTeslaT4`.
  - `torch.amp.GradScaler("cuda")` not `torch.cuda.amp.*`.
  - Poll status by matching only `COMPLETE`/`ERROR`; treat network errors as "keep waiting" (a DNS blip
    once looked like completion). One epoch ≈ 80 min on T4 @ batch 64.
  - Log download can hit a Windows charmap error — parse the `.log` JSON yourself with utf-8.

---

## 4. Improvement strategies — HYPOTHESES TO TEST (not a plan; user asked only to "form up strategies")
Ranked by expected value / cost. **For each, the TEST and the SUSPICION are the point.** Do the test
before believing the strategy; actively look for a strategy this list MISSED.

**A. Finish training v3 (cheapest, in flight).** Test: does reproducible Pearson keep rising past epoch 7
or plateau? Suspicion: OneCycle anneals LR→0 by epoch 12, so gains may taper; a longer/cosine-restart
schedule might beat "just finish". Verify convergence, don't assume it.

**B. Run the interpretability deliverables — UNRUN and load-bearing.** DTI recall@k: does aggregated
atom→gene attention localize to known target genes (`dti_reference.tsv`)? Attention/gate/epi-contrib maps.
Suspicion: **the whole interpretability claim is untested.** If atom→gene attention does NOT recover DTI,
R² is irrelevant — the project's stated objective is unmet. This may also be a diagnostic (is the drug
branch even using atoms?). Stratify DTI validation by signature strength from the start.

**C. Drug-feature ablations (never run).** Ablate descriptors / fingerprint / UniMol-atoms / ChemBERTa
one at a time on reproducible data. Test: which actually contribute? Suspicion: heavy redundancy
(ChemBERTa vs fingerprint vs UniMol all encode structure); the 512-d atom tokens may be underused
relative to their cost. Drug-signature is 42% of reproducible variance — the biggest single lever — so
knowing which drug features matter is high-value and cheap (CPU-scale inference from the ckpt).

**D. Richer cell representation for the RARE-cell gap.** Per-cell R² 0.02–0.22; rare/unusual cell lines
are the failure. Test: add CCLE mutations/CNV or tissue-type embedding; does rare-cell cold R² rise?
Suspicion: cold-cell generalization may be fundamentally limited by having only 83 training cells —
more cell *features* won't help if the problem is too few cell *examples*. Measure cell-similarity vs
per-cell R² first to see if it's a representation or a coverage problem.

**E. Push cell-specificity (interaction) — but bounded by the ceiling.** Model at 53% of the 0.842
interaction ceiling. Test: deeper cross-attention / stronger cell-conditioning → corr(interaction) up?
Suspicion: reproducible interaction is only ~18% of variance; easy to fit noise. Gate any gain on the
reproducible-subset AND a re-measured ceiling, not the raw share.

**F. Epigenetics, now that it works (+0.073). Can it do more or mainly give mechanism?** Test:
drug-conditional epi (interaction was stronger for HDAC/EZH2 modifiers: H3K27ac r=−0.23 vs −0.16
controls); quantitative bigWig vs peak-at-TSS (higher resolution — but bigWig was deemed intractable on
this connection, see epigenetics report; maybe feasible on Kaggle). Suspicion: diminishing returns on
R²; its real value is likely mechanistic (correctly-signed, ~50-SE effects). Report effect sizes, don't
oversell.

**G. Loss/target refinements.** Test: per-gene weighting (most genes in a signature don't move);
Huber-δ tuning; predicting rank vs z-score. Suspicion: could improve strong-gene fit while degrading
"predict nothing" calibration for inactive compounds — check the WEAK stratum too, not just strong.

**H. Things this list may have MISSED (actively hunt for these):** batch/plate structure beyond `phase`;
dose-response shape fidelity (never checked the model reproduces inverted-U); whether MSE-shrinkage is
partly *correct* under noise (under-dispersion is optimal — don't "fix" it into overfitting); whether the
prior-bias attention (co-pathway/STRING) actually helps or is decoration (ablate λ); whether the
2 held-out cold folds generalize to the other 3 (only fold 0 has been run).

---

## 5. Immediate next actions (in order)
1. Collect the v3-resume result (`apexblue/lincs-train`): reproducible Pearson/R², epi ablation, per-cell.
2. **Strategy B (DTI recall@k)** — highest priority: it tests the project's actual objective and is unrun.
3. Strategy C (drug-feature ablations) — cheap, high-info, CPU-scale from the ckpt.
4. Then decide D/E/F with the user, gated on measurements.

## 6. Pointers
- Design + results + rejected ideas (with reasoning): `model/MODEL_MATH.md`
- Epigenetics saga (2 wrong "drop it" calls, final +0.073): `drug/outputs/dti/` reports + memory
  `lincs-epigenetic-assays`, `lincs-phase2-status`.
- Per-branch reports: `{baseline,epigenetics,network,drug,phase2_assembly}/report.md`.
- The scratchpad Kaggle staging dirs are session-local and will be GONE next chat — re-stage from
  `model/*.py` (they are the source of truth; the Kaggle datasets mirror them).
