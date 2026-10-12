# REVIEW OF PACKET 078 (D's gate, 98.14; M+dt reported, 98.13)
verdict: NOT-SUPPORTED (98.14's stated grounds); 98.13 SOUND
reviewed_commit: 03547c8 (98.13 at 6a0b7ec)

**98.14's two failures are both mis-read in their decisive parts. The code citations are right, but the conclusions drawn
from them are not.**
- **Count 1's premise is false on this bundle.** It says an inverse would need each test row's true total. In fact an
  inverse built from inputs alone (the row's own control total, plus a scalar from the training rows) recovers raw Δ from a
  perfect transformed-space prediction with mean row Pearson **0.99979** on the 13,445 test rows (C1).
- **Count 2 fails on both halves:**
  - uploading weights is a pure data step, which is C6's own criterion;
  - multi-dose is native to their pipeline. And without it, their harness runs one 10 µM prompt for every row, which is how
    Bai et al. ran it and the same as-published standard M was held to (C2).

What remains are two rulings on 98.2's own wording, plus the timing clause, which hasn't been measured. "Not run (adapter)"
can stand only as an explicit, labelled ruling, made on the correct facts.

**98.13 reproduces with my own scorer:**
- clean −0.0050 [−0.0097, +0.0001], 92 / 346;
- full +0.0032, 115 / 384;
- distinct-control +0.0423 [0.0312, 0.0535], 206 / 267;
- row-pooled 0.6375;
- the manifest's six sha1s match, and the three records carry `n_test` 13,445 and best epochs 271 / 244 / 233.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | gate fact (count 1) | **"Mapping back needs each row's raw total … from the test row's own truth" is true only for an exact inverse.**<br>• **The inverse from inputs alone:** X̂ = expm1(`test_predictions` + `test_ctrl`) · S_ctl / T, then Δ̂ = X̂ − X_ctl. Here S_ctl is the row's own control total (an input) and T is the median training-row total.<br>• **Why it is nearly exact:** row totals barely move under treatment. On `split_cold_drug_1` test rows, S / S_ctl has mean 0.99916, sd 0.0069, and 1st–99th percentiles 0.977–1.015. T_all / T_train = 0.99993.<br>• **The result:** applied to a *perfect* transformed-space prediction (their exact pipeline, separate medians for X and X_ctl as at lines 147–159), it gives mean row Pearson with raw Δ of **0.99979** (median 0.99992, 1st percentile 0.9984, minimum 0.9857). The transform loses at most about 0.0002.<br>• **What is left of count 1:** "the inverse is not in their code". Whether a two-line output conversion from inputs alone is "a data adapter" is a ruling on 98.2, not a fact read from code. M's adapter also had an output step (`convert_outputs`), and 98.2's operative parenthetical is "(no model change)". The inverse doesn't change the model. | Rewrite count 1 with these facts. If the PI rules that an input-only output inverse exceeds "a data adapter alone", record it as a **ruling**, with the 0.9998 figure and the reason it is excluded although M's output step was not. |
| 2 | MAJOR | gate fact (count 2) | **Neither half of count 2 is a gate failure as read.**<br>**(a) The weights.** C6's criterion is "if that is not a pure data step". Uploading MolT5 and BioLinkBERT weights as a private dataset **is** a pure data step, and size is not the criterion. (Their precompute also needs a CUDA GPU for MolT5 captions, line 75. That counts toward the timing clause, which C6 puts inside the gate.)<br>**(b) The dose.** "Per-dose prompts aren't in their pipeline" is mis-read:<br>• `train_pertdit.py` has a per-sample dose path: `--dose_col`, a dose-keyed dict, lines 68–69, 179–183 and 210–231, labelled "multi-dose support".<br>• Their verification reads upstream PertDiT's per-dose LINCS dict, `dosage_prompt_emb_lincs.pkl` keyed by dose (lines 215–219).<br>• A dose embedding is BioLinkBERT on a template string (line 159). It is dataset-independent, so "upstream-shipped … for the original data, not ours" doesn't apply to doses.<br>• **And without `--dose_col`,** their harness applies the one 10 µM prompt to every row, exactly as Bai et al. ran it. That is the same as-published standard under which M drops dose and time (073, Ask 1).<br>• **The real problem is inside 98.2's own wording:** "per-dose … computed by their own precompute scripts", which compute one dose. | Restate count 2 as a cost (weights, a GPU caption pass over the bundle's 1,970 compounds, all inside their 8,981-compound index) plus that wording conflict. Then **rule explicitly** which dose configuration is "as registered": their single-prompt configuration (as for M), or per-dose prompts through their `--dose_col` path. Label it as a ruling. |
| 3 | MINOR | scope (the note on their benchmark) | **The note claims more than the vendored code shows.**<br>• **What I read:** PertDiT's saved truth, predictions and `per_drug_pcc.csv` are in the transformed space (lines 437–472). The MLP's are raw Δ (`train_mlp.py:309–315`).<br>• **What isn't vendored:** their unified evaluator and the other five trainers. So "the other models' is on raw Δ" is checked for the MLP only.<br>• **How different the targets are:** on our cold-drug test rows, transformed Δ and raw Δ correlate per row at **0.975** (median 0.9755, 5th percentile 0.967). The targets differ, but they are closely related, and the direction of any PCC_DEG gap can't be signed from this.<br>• **The same gap in 98.14's last bullet:** "XPert is also the strongest of the seven" includes a cross-target comparison with PertDiT.<br>• **I concede** that my 073 Ask 5 reason ("PertDiT … at or below XPert") rests on that same cross-target figure. | Scope the note: *"PertDiT's saved truth and predictions are in the transformed space and the MLP's are raw. Their evaluator and the other five trainers weren't read. Per row, the two Δs correlate at 0.975 on our test rows."* Qualify the "strongest of the seven" bullet accordingly. |
| 4 | MINOR | precision (98.13) | **"Dose and time add about +0.002" is the row-pooled difference** (0.6375 − 0.6354). Per molecule, the reading unit, M+dt − M is **+0.0003** [−0.0001, +0.0007], 197 / 346.<br>It matters for R2. B1 keeps dose and time (073 C4c), and this is the cleanest measure of what they add to a drug-free model: almost nothing per molecule. | Give both figures: *"+0.002 row-pooled; +0.0003 [−0.0001, +0.0007] per molecule."* Reported, never read. |

## Answers to the asks

**Ask 1 — no, not on the stated grounds** (C1, C2).
- **What 98.14 reads correctly from their code:**
  - the transform (lines 144–159; `clip` is a no-op here, since X ∈ [0, 15.0] with no negatives);
  - no inverse in their code;
  - one dose prompt in their precompute;
  - the weights dependency.
- **What it misreads:**
  - that an inverse needs test truth;
  - that weights fail C6;
  - that multi-dose is absent from their pipeline.
- **What decides the gate now:**
  - two rulings on 98.2's wording: the input-only output inverse, and the dose configuration;
  - then, if those pass, the **timing clause** (a one-epoch projection, precompute included, per C6). It hasn't been measured.
- **On 98.2's own words, neither step is a model change.** A ruling that excludes them has to say why, given M's output step.
  Which way to rule is the PI's decision, not mine.

**Ask 2 — fair as a reading of PertDiT's trainer, but scope it** (C3). The transformed-space truth and predictions are read
from the code. The claim about the other models and their evaluator is inferred.

**Ask 3 — an assessment, not a choice.**
- **What each design answers:**
  - **Fold 2** tests whether the post hoc distinct-control lead **reproduces** on fresh test molecules. Like fold 1, it can't
    separate "M recalls plate responses" from plate/time composition (077, Ask 2).
  - **The control-substitution M** tests the **mechanism** on fold 1, and it needs no GPU.
- **If fold 2 is registered,** fix before any fold-2 output:
  - the stratum (control hashes against fold-2 training rows);
  - the clean-subset rule and its sha1;
  - a fresh M carve seed;
  - v9's recipe frozen as P9's, with no fold-2 tuning;
  - what counts as replicated (for example, a cluster CI excluding 0 in fold 1's direction).
- **Power is not the constraint.** Fold 1's +0.044 [0.033, 0.056] over 267 molecules is far from 0.
- **Quota:**
  - O9 is the registered head-to-head and can't be read until it finishes, so no plan should delay its sessions;
  - C1 / C2 bear on this too: if D is re-ruled and passes the adapter clause, its timing measurement competes for the same
    quota.

## What I checked and found sound

- **`train_pertdit.py` in full (509 lines):**
  - the transform, and no inverse;
  - the saved Δ and `per_drug_pcc` in the transformed space;
  - the `--dose_col` path;
  - early stopping on transformed-space validation MSE (DDIM samples);
  - `d_pre` 978.
- **`prepare_pertdit_embeddings.py` in full:**
  - MolT5 needs CUDA (line 75) and BioLinkBERT runs on CPU (line 128);
  - one dose prompt (line 159);
  - `negative_ctrl` is copied from upstream's archive (line 151);
  - the verification reads upstream's per-dose dict (lines 215–219).
- **The rest of the vendored code:** `train_mlp.py`'s unified outputs (raw Δ); PROVENANCE.md; `fetch_upstream.sh` (PertDiT at
  `596d681`, an 8.9 GB archive).
- **The bundle:**
  - X / X_ctl ranges;
  - row totals and the S / S_ctl ratio;
  - the oracle inverse;
  - transformed Δ against raw Δ;
  - 10 doses (10 µM is 25.6 % of rows);
  - 1,970 compound indices, maximum 8,949, all inside their 8,981-compound table;
  - times 3 / 6 / 24 h (PertDiT has no time input; as published it ignores time, as M does).
- **98.13 by my own scorer:** the manifest, the records, and the five readings above, plus the shared-control stratum
  (−0.0115 [−0.0155, −0.0073], 62 / 345) and M+dt − M.
- **The registration:** 98.2's D clause and 98.6's C6.

## What I could not assess, and why

- **`bench_dataset.py`.** `train_pertdit.py` imports it (line 45), but it isn't vendored, and the main checkout has a
  **0-byte untracked file** at that path. So I can't confirm how a dose dict is consumed, or whether `negative_ctrl` is needed
  with `cfg=False`.
- **Upstream PertDiT** (model code at `596d681`, and the archive). I couldn't check whether `dosage_prompt_emb_lincs.pkl`
  covers the bundle's ten doses.
- **Bai et al.'s unified evaluator and the other five trainers.** They aren't vendored (C3).
- **The timing clause and the ≈ 4 GB weight figure.** Timing needs a GPU run. The weight size is plausible from the model
  sizes, but I didn't download anything.
- **How PertDiT's own errors map through the inverse.** The oracle bounds only the transform's information loss. How
  PertDiT's errors map through expm1 is part of its accuracy as published.
