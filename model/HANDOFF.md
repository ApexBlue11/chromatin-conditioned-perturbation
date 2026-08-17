# HANDOFF — read this first (updated 2026-07-27)

> **READ ADVERSARIALLY.** This project has repeatedly produced confident-but-wrong conclusions that were
> only caught by insisting on a proper control. Four claims were falsified in one day. Treat everything
> below as a hypothesis with evidence attached, and check the evidence before building on it.

## 0. State in one paragraph
v5 is trained and fully analysed. **Its interpretability claims were falsified by our own tests** —
atom→gene attention does not recover drug targets, and the pathway prior barely steers attention. The
causes were diagnosed against the literature (soft bias instead of a hard mask; early instead of late
modality fusion), and **v6 is a complete rebuild** fixing both. v6 is built, documented, unit-tested
(16/16) and **launched on Kaggle TPU** — but **unmeasured**. The one claim that has survived every test is
the chromatin (epigenetics) effect.

## 0. THE TWO THINGS THAT MATTER MOST (read before anything else)

**1. Seed variance exceeds every architecture effect this project has reported [M.10].** Three seeds of one
identical config: sd **0.0052 / 0.0150 / 0.0232** on unseen cell / compound / both ⇒ 2-sd = **±0.046**. v5,
v6 and v7 are *statistically indistinguishable* on two of three splits. Every "vN beats vN−1" number in the
older sections of RESULTS was inside noise. **Report mean ± range over ≥3 seeds, or report no difference.**
Ablations are exempt — they are within-run on identical signatures and carry no seed variance, which is why
ablate-to-mean kept giving clean answers while headline deltas were noise.

**2. Best config measured to date is `v7 --no_aux`** (RESULTS §22): 0.4452 / **0.4900** / **0.4772** pearson,
with R² **+0.3135** and **+0.2878** on compound/both — roughly double v6's. It beat the maximum of three
aux seeds 6/6. Seeds 1 and 2 are running to confirm. `use_aux` now defaults to **False**; the auxiliary-
supervision idea is untested, but Kendall uncertainty weighting is the wrong tool for it (see config_v7).

## 1. What is running right now
| Kernel | What | How to collect |
|---|---|---|
| `apexblue/lincs-v7-noaux-s1`, `-s2` | v7 no-aux seeds 1 & 2, T4 x2, 12 epochs, ~4.6 h each | §2, then `model/v7/eval_v7.py` |

```bash
export KAGGLE_CONFIG_DIR="C:/Users/Surya/.kaggle"
kaggle kernels status apexblue/lincs-v6-gpu
kaggle kernels output apexblue/lincs-v6-gpu -p <dir>     # -> ckpt_v6_fold0.pt, metrics_v6_fold0.json
```
**TPU is abandoned for this model** (`apexblue/lincs-v6-tpu`, do not restart it). Two full queue cycles
produced **zero** training — `xmp.spawn` nprocs rejected under PJRT, then v5e multi-process init failing
with *"Expected 8 worker addresses, got 1"* (TPU_NOTES #13/#15) — and the TPU was already measured **slower
per core than a T4** [8]. The GPU kernel started **within seconds, no queue**.
Kaggle publishes a kernel's log only once it ends, so a RUNNING kernel cannot be inspected mid-flight; the
fail-fast guards in `train_v6_gpu.py` fire in the first seconds, so "still RUNNING" already means the GPU is
real and is not a P100.
**Check status DIRECTLY and try pulling output — do not trust a poller.** A finished run has output even
when a stale poller still says QUEUED (this cost real time; see TPU_NOTES.md #10).
**Never hard-cancel**: a cancelled Kaggle kernel's `/kaggle/working` is discarded and the checkpoint is lost.

## 2. When v6 finishes — exactly what to do
**The evaluation is already written and tested (29/29 checks). Two commands, both local CPU, no Kaggle.**

1. Pull `ckpt_v6_fold0.pt` + `metrics_v6_fold0.json`; copy the ckpt to `model/results/`.
2. Accuracy + attribution — steps 2 and 3 below are both built into this one command (~2 h on CPU):
   ```bash
   python model/v6/eval_v6.py --ckpt model/results/ckpt_v6_fold0.pt
   ```
   - reproducible stratum only (mean|Y| ≥ 1), all three splits, v5's own metric functions **and v5's own
     subsampling caps**, so the comparison is signature-for-signature. v5 reference: **0.440 / 0.471 /
     0.451** Pearson. It **refuses to claim comparability** if the split sizes don't match v5's
     (7296 / ≥6000 / 2998).
   - every component **ablated to its MEAN, never to 0 or 1** — ablating a learned multiplicative component
     to 1 destroys its learned scale and produced a **30× artefact** last time (+0.103 vs the true +0.006).
   - each mode also reports `|dY|max`, so **"delta = 0.000" can be told apart from "the ablation never took
     effect"** — the failure mode this project keeps hitting is a valid computation of the wrong quantity.
3. Pathway readout vs independent annotation (~10 min):
   ```bash
   python model/v6/probe_pathways_v6.py --ckpt model/results/ckpt_v6_fold0.pt
   ```
   **Read §2a first — the obvious version of this test is invalid.**
4. **MoA probe — does the pathway layer carry drug mechanism?** (~20 min)
   ```bash
   python model/v6/probe_moa_v6.py --ckpt model/results/ckpt_v6_fold0.pt
   ```
   Activations are drug-invariant (§2a) but the **gradient** `∂Ŷ/∂a_p` is not [4.14], so a per-drug
   pathway attribution exists without a retrain. It checks, in order: (a) is the branch **inert**
   (cross-check against `mean_pathway`); (b) does the ranking vary across drugs **at all** — on an
   untrained model it does not; (c) only then, alignment to curated ChEMBL targets.
   ⚠️ **0.5 is not the chance level** — target-containing pathways are large and central and score 0.218
   on an *untrained* model. Judge against the permutation null only [4.15].
4. Judge v6 against v5 **on the same splits and stratum**, and report negatives as prominently as positives.

## 2a. ⚠️ The pathway readout is DRUG-INVARIANT (verified 2026-07-30, claim 4.12)
`pathway_activations` is produced *before* the drug enters the forward pass. Swapping in a completely
different molecule changes it by **exactly 0.0**; baseline (2.54), chromatin (1.28) and dose/time (1.59) all
move it. It is a **cell × dose/time** readout: "which named pathways are chromatin-permitted in this cell".

- **v6 does NOT repair the falsified drug-target claim** (§5, atom→gene attention). That was a *drug*-level
  claim; this is a *cell*-level readout. v6 sidesteps that failure rather than fixing it.
- **Do not score this readout against `dti_reference.tsv`.** Structurally guaranteed null, means nothing.
- The valid annotation is the **measured LINCS response** (never an input to the readout): does pathway *p*'s
  activation in cell *c* track how much *p*'s member genes actually move in *c*? With a gene-permutation
  null, a **cell-shuffle null** (cell-specific, or just a global pathway prior?), and train/test-cell
  stratification. That is what `probe_pathways_v6.py` runs.

## 3. Documentation map (all in-repo, nothing external)
| File | Contents |
|---|---|
| `model/v6/ARCHITECTURE.md` | **v6: the task, provenance of every input tensor, data flow, per-component rationale + evidence, what the pathway readout is and is NOT, what v6 is expected NOT to fix, how it must be judged** |
| `model/v6/eval_v6.py` | accuracy on the reproducible stratum (protocol-matched to v5) + ablate-to-mean for all 7 components |
| `model/v6/probe_pathways_v6.py` | pathway readout vs measured response, with gene-permutation + cell-shuffle nulls |
| `model/v6/test_v6.py` | 29 checks incl. the positive **and** negative control for the ablation mechanism |
| `model/v6/TPU_NOTES.md` | 12 TPU/XLA failure modes and the fix for each |
| `model/ARCHITECTURE_LESSONS.md` | why v5's interpretability failed; what P-NET/DCell/MOLI do instead |
| `model/LITERATURE_PRACTICE.md` | **methodology audit vs the field** — where we're ahead, 6 ranked gaps with fixes and costs, what to take from other papers' future-work sections |
| `model/results/CLAIMS.md` | ~60 claims: evidence, A/B/C/✗ strength, falsifiers, **retractions kept deliberately** |
| `model/results/RESULTS.md` | detailed measurement log, 17+ numbered experiments |
| `METHODOLOGY.md` | script-by-script workflow, environments, reproduction order |
| `MANUSCRIPT.md` | draft paper — unrun sections marked 🔲, never filled with placeholder text |
| `model/MODEL_MATH.md` | v5 equations |

GitHub: `ApexBlue11/chromatin-conditioned-perturbation` (private, MIT).

## 4. What is TRUE (survived testing)
- **Chromatin contributes**: ΔR² **+0.089** in-distribution, mechanistically correct sign (low activation /
  high Polycomb ⇒ larger response), and it **survives baseline-expression stratification in all four
  quartiles** — the floor-effect confound is excluded. Literature-consistent.
- **…but it tracks cell familiarity**: +0.089 in-dist → +0.035 unseen compound → **≈0 unseen cell**. It is
  an in-distribution refinement, *not* a cold-cell generalisation mechanism.
- v5 accuracy: unseen cell 0.440 / unseen compound 0.471 / unseen both 0.451.
- **The model's value is CHEMICAL generalisation, not cellular** (measured 2026-07-30, `baseline_linear.py`):
  on unseen COMPOUND it beats the best linear baseline by **+0.089** pearson and the best mean by **+0.106**;
  on unseen BOTH by **+0.120 / +0.135**.
- Drug features: ECFP4 and atom tokens are the pillars; **ChemBERTa is dead weight** (ΔR² +0.001, removed).
- Interaction under-expression (26.5 % vs 47.9 %) is **near-optimal under noise** — a correlation loss
  designed to fix it changed nothing.

## 5. What is FALSE (retracted — do not resurrect)
- ~~v5 beats all naive baselines~~ — **retracted 2026-07-30 [1.1/1.8]**. That was MSE on *all* cold-cell
  signatures. On the reproducible stratum with the metrics we report, **Meandrug (just predict the drug's
  training mean) beats v5 on unseen-cell on all three metrics**: pearson 0.4475 vs 0.440, R² +0.2846 vs
  +0.273, MSE 4.279 vs 4.348. A ridge on the same global inputs also edges it. **The model adds no
  cell-specificity.** It survives only for unseen-compound / unseen-both, where it wins by a lot (§4).
- ~~atom→gene attention localises to drug targets~~ — median target rank percentile **0.560 over 149 gold
  pairs, worse than chance**; the "target doesn't move" confound was tested and **rejected** (corr −0.045).
  The old "2.6× enrichment" was ~1–2 drugs of 111 in a top-5 ⇒ noise. **Top-k enrichment ≠ localisation.**
- ~~pathway conductance is the biggest contributor (+0.103)~~ — scale artefact; true structural effect
  **−0.003 / +0.006**.
- ~~our metric-convention critique is novel~~ — established as *control bias* / *signal dilution*; delta
  metrics are already the field standard. **Cite, don't claim.**
- ~~chromatin conditioning is novel, broadly~~ — chromatin is already used for drug **sensitivity**
  (GraOmicDRP). The surviving, narrower claim: conditioning **response-profile** prediction on chromatin.

## 6. Method rules (each learned by getting it wrong)
1. **Never evaluate on all signatures** — ~75 % of LINCS perturbations are inert; dilution once *inverted*
   the sign of the chromatin effect. Always stratify to mean|Y| ≥ 1.
2. **Ablate to the MEAN**, never to 0/1 — otherwise you measure scale destruction (30× artefact).
3. **A noisy reference masks real signal** — and a tail statistic is not localisation. Stratify the
   reference by evidence quality *and* check the per-item distribution, not just top-k.
4. **Compute the noise ceiling before chasing a gap.**
5. **Verify a measurement tests what it claims** — most errors here were valid computations of the wrong quantity.

## 7. Compute
- **GPU quota is available again (2026-08-14) — use it.** T4 x2 via `machine_shape: "NvidiaTeslaT4"` +
  `"enable_gpu": true` in `kernel-metadata.json` (this is Kaggle's T4 **x2** option; copied from the
  account's own working `lincs-train-v5` kernel rather than guessed). The `kaggle kernels push`
  `--accelerator` ids are: NvidiaTeslaP100, NvidiaTeslaT4, NvidiaTeslaT4Highmem, NvidiaL4, NvidiaTeslaA100,
  NvidiaH100, TpuV38, TpuV5E8, TpuV6E8, Tpu1VmV38, NvidiaL4X1, NvidiaRtxPro6000.
- **Do not use TPU for this model** — see §1. GPU also avoids the "1 concurrent TPU session" trap, which
  blocked pushing a corrected kernel while a broken one sat queued (the CLI has **no cancel**, only `delete`).
- Never P100 (sm_60 unusable): `train_v6_gpu.py` probes and exits in ~1 s if one is assigned.
- Kaggle caps: 5 concurrent CPU sessions, 2 GPU. CPU kernels are free.
- **torch 2.11 IS installed locally** — run unit tests and all CPU analyses locally; only send
  accelerator work and big-bundle jobs to Kaggle.

## 8. Next steps, in order
0. **CPU-only, nothing waits on the TPU** (from the methodology audit, `LITERATURE_PRACTICE.md` §5):
   fit the **ridge linear baseline** — we have never fitted one, and it is the baseline that beat every deep
   model in Nature Methods 2025, so it blocks every accuracy claim; then report **all mean|Y| strata** rather
   than only ≥1; add **Common-DEGs**; add an **unseen dose/time split**.
1. Collect v6 and run the two commands in §2 — the harness is written and tested, only the checkpoint is
   missing. The bottleneck is still **unmeasured** and may contribute nothing.
2. If the pathway readout works, that is the paper's mechanistic contribution; if not, report the null.
3. Protocol-matched SOTA comparison — needs LINCS **Level 3** (the comparison paper uses level 3
   quantile-normalised profiles) plus paired controls; we have Level 5 only.
4. k-fold across the remaining cell folds — every current number is fold 0 only.
