# REVIEW OF PACKET 099 (v11 code, §102.2; readers §102.3; T-C strata §102.4), before any v11 GPU run
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 98bb828 (code), bca9e7b (readers), bfd7e1b (T-C strata)

**The code implements §102 / §102.1 faithfully, and it is careful about the base.**
- **`LincsV11.forward`** is v10-C's, op for op, with FG inserted after `h_basal`. The basal term stays drug-free.
- **FG matches §102.2:**
  - the node MLP;
  - a zero-weight gate at logit 0.95;
  - factorised gated rounds, √a ⊙ Â(√a ⊙ RMSNorm(z)) with W_k = 0;
  - the per-row 2-channel signal hopped twice over the row's cell's gated graph;
  - a zero-initialised readout.
- **The data layer:**
  - the base receives `V10CData(chrom_off=True)` exactly;
  - the new sources reach only FG;
  - "known" excludes probe compounds, unseen compounds and same-molecule duplicates of unseen compounds;
  - the vocabulary and hop scales come only from known compounds (outcome-free);
  - the arms switch only their own tables.
- **The trainer** is train_v10c's loop with the head's BCE. It adds the timing fields Ask 4 / C7 asked for.
- **The readers:**
  - R1 / R2 by `tb_read.rule`, with R2-CCLE's condition over the 4 CCLE cells;
  - ledger-checked records and predictions;
  - nothing read while any slot is unresolved;
  - strata per §101.1 C3.
- **Tests:** 49 / 49 pass on my run (v11 model 28, readers 8, probe 3, strata 7, T-C strata 3).

**One factual claim is false on GPU** (C1, demonstrated). It doesn't affect R1's validity, but the fix is one line and should
land before the arm kernels. C2 and C3 are design choices and wording.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | RNG claim on GPU (Asks 1, 8) | **The FG seed fork re-seeds every CUDA generator.**<br>• `LincsV11.__init__` calls `torch.manual_seed(seed + 7919)` inside `torch.random.fork_rng(devices=[])`.<br>• In this torch (2.13), `manual_seed` calls `torch.cuda.manual_seed_all`. My monkeypatched check shows one call with 7919 inside the fork.<br>• `fork_rng(devices=[])` saves and restores **only the CPU state.**<br>• So on Kaggle, the CUDA generators that draw training's dropout and stochastic-depth masks start from seed + 7919, not T-B off's seed.<br>**The tests can't see this:** the stream test uses CPU `torch.rand`, and the recipe test runs on CPU.<br>**What holds:** v11's base is seed-paired with T-B off **at init** (weights built on CPU).<br>**What doesn't hold on GPU:** pairing **in training**. §102.2's "the comparison with the reference is seed-paired" and Ask 1's "dropout stream" are false there.<br>**What it doesn't affect:** R1's validity, since §100.4's rule is unpaired. | **Inside the fork, seed only the CPU generator:** `torch.default_generator.manual_seed(...)`, or fork every CUDA device.<br>**Add a test:** monkeypatch `torch.cuda.manual_seed_all` and assert it isn't called while `LincsV11` is constructed.<br>**Land it before the arm kernels.** The timing kernel is unaffected. |
| 2 | MINOR | what R1 measures (Asks 1, 2, 11) | **§102.1 C1's "R1 against T-B off measures FG alone" isn't what's built.**<br>• In full, −chromatin and −CCLE, the head's BCE trains `w_u` / `ln_u` (the base's drug token).<br>• Gradient clipping uses the global norm, FG and head included.<br>• So R1 measures **FG plus auxiliary target supervision.** `v11_read`'s verdict already says this. | **Choose before the arms:**<br>**(a) Keep the design** and change §102.1 C1's sentence to *"FG with its auxiliary target supervision"*, consistent with the verdict and with C4's −targets wording;<br>**(b) Detach the drug token into the head** (`target_head(g.detach())`). Then R1 = FG alone and −targets = target input only, at some cost to the head's accuracy.<br>Either is defensible. |
| 3 | MINOR | the probe's interpretability (Ask 10) | **FG injects the head's predicted targets into the landmark tokens before the perturb blocks.**<br>So atom → gene attention on probe compounds can **echo the structure head's predictions**, for example through analogues, rather than show an atom-level mechanism. The Tanimoto bins help but don't separate the two. | **Report the head's own recall@k on the probe compounds** (its predicted probabilities ranked over landmark targets) beside the attention recall, per bin and setting.<br>If attention ≈ head, the attention restates the head. |

## Answers to the asks

**Ask 1 — the base: the evidence is strong for inputs, bytes, init weights and the forward pass.**
- **Not seed-paired in training on GPU** (C1).
- **R1 measures FG plus auxiliary supervision** (C2).
- **Hardware:** the same docker image is right. GPU atomics make same-seed runs differ anyway, which is one more reason the
  rule is unpaired.

**Ask 2 — yes, detach the probabilities into FG.**
- That keeps the head a target predictor, keeps "predicted targets" interpretable, and keeps −targets clean.
- Whether the head's own gradient should reach the base is C2's choice.

**Ask 3 — yes.**
- Excluding duplicates of unseen compounds from the **input**, as well as from the head's labels, applies §96.6's
  same-molecule rule consistently. The model never sees that molecule's targets anywhere.

**Ask 4 — yes.**
- Human genes in the graph only (864 genes, 1,149 compounds; vocabulary 835 on the dev carve).
- A compound with only non-human targets counts as unannotated and is predicted.
- Record the dev carve's vocabulary and status counts in each run record, as the trainer does.

**Ask 5 — sound and outcome-free.** It uses annotations and the graph only.
- **Hop 0** is already 1 for both channels: the RMS of nonzero entries of a binary or ±1 vector is 1. In effect only hops 1–2
  are rescaled.
- `s_scale` is computed per split and is written to the record.

**Ask 6 — acceptable.**
- z in the basal term couldn't move the primary (the contrast cancels the basal term within a plate). It could only move delta
  r.
- Keeping FG out of basal preserves F6 and the equivalence with T-B off.

**Ask 7 — sound.**
- **Sharing:** `replicate()` shallow-copies `__dict__`, so every replica shares one `DeviceTables`. Each device has one replica
  thread, so a device's copy is built once. The dict writes for different devices are safe under the GIL.
- **`replace()`** is used only in `fg_permutations`, through `core` and outside DataParallel.
- The timing kernel is the first real two-GPU exercise. Check `cells_per_replica_max` and the memory figures it records.

**Ask 8 — nothing must change before the timing kernel.**
- C1–C3 don't affect speed or memory.
- **C1, and C2's choice, must land before the 12 arm kernels.** That means re-pinning `lincs-v11-src` after the timing
  kernel, as §102.2 already plans for the epochs and retry budget.

**Ask 9 — acceptable.**
- The weights 1 / v̂² are the inverse variance of a sample variance up to a constant, the right choice for equal-df variance
  estimates.
- **A note:** with an SE pooled over about 1,000 compounds, a t reference on 4 df is **conservative** (it loses some power).
  It is what §101.1 registered; keep it, and say so in the output.

**Ask 10 — yes, the training-row setting is a legitimate addition** (responses fitted, targets never seen).
- "Descriptive" is the right label: ChEMBL subset 12 / 20, landmark-only reference.
- Add C3's head comparator.

**Ask 11 — the wording is right on the supervision.** Add two things:
- *"among runs whose drug path trained (100.8)"*;
- that the reference's 3 / 3 runs trained without a guard.
- It's an unpaired comparison (C1), so nothing more is needed.

**Ask 12 — yes.**
- **Class, compound and epigenetic × cell strata on the 7 clean cells match T-C's rule.**
- **H1975 is clean for v10-C** (NCIH1975 was removed from training), so an all-8-cell class stratum could also be reported,
  labelled. It is optional.

## What I checked and found sound

- **`model_v11.py` in full:**
  - `DeviceTables`, `spmm` (fp32 outside autocast);
  - FG (node input dims 128 + 2 + 20; gate 22 → 1, zero weight, bias logit 0.95; rounds with W_k = 0; RMSNorm; P;
    zero readout);
  - `target_signal` (scatter with a pad node, known vs head probabilities, sign 0 for predicted);
  - per-row gated hops;
  - `LincsV11.__init__` (base first; forked init; head bias = logit of the prior rate);
  - the forward pass against `LincsV10C.forward`, line by line;
  - `v11_loss` (BCE on known rows, off in −targets).
- **`data_v11.py` in full:**
  - `ccle_ranks`, `sym_norm` (isolated nodes zero), `combine_signs`, `hop_scale`;
  - `V11Data` (base via `super(chrom_off=True)`; graph and landmark one-to-one check; gene-vector standardisation;
    chromatin and CCLE order checks; −chromatin as the training-row-weighted mean of F and M; −CCLE zeroed);
  - `_targets`:
    - Homo sapiens ∧ in-graph;
    - status = probe / unseen-or-duplicate / known;
    - probe targets never written;
    - vocabulary and prior from known compounds;
    - `known_input` off in −targets;
  - `batch()` (`head_y` only for known rows).
- **`train_v11.py` in full:**
  - the recipe and guard as train_v10c;
  - timing fields;
  - the final guard check;
  - the known-target setting for status-2 rows;
  - `fg_permutations` (restores in `finally`);
  - the record and predictions (`target_status`).
- **`make_v11_kernels.py`:** template substitution with count checks; `lincs-v10c-src` not mounted; pins; the timing kernel's
  args.
- **The readers:**
  - `v11_read.py`: slot resolution, first completed attempt, ledger checks, the reference via `tb_read` + pred sha1, `rule`
    with the CCLE cell condition, row alignment, strata;
  - `v11_ledger.py` (`add`, dead-kernel handling);
  - `strata.py`: group stats, cluster CI, IRLS `fit_variance` with the non-negativity fallback, BH, `compound_test`'s
    SE = √(2v / 3), the 9 relabellings;
  - `tc_strata.py`: clean-7 class / compound / epigenetic strata, H1975 apart;
  - `v11_probe.py`'s header, reference parsing, recall and Tanimoto bins.
- **The CUDA re-seed:** torch 2.13's `_manual_seed_impl` source, and a monkeypatched demonstration (C1).
- **Tests** (`CUDA_VISIBLE_DEVICES=-1`, `.venv-cuda`): `model/v11/` plus `test_v10c_strata.py` and
  `test_v10c_tc_strata.py`, **49 / 49 passed** (28 model, 8 reader, 3 probe, 7 strata, 3 T-C strata), in 7 min 12 s.

## What I could not assess, and why

- **Anything on GPU.** That covers timing, d = 256 memory, two-GPU DataParallel, and C1's effect in practice. Nothing has run
  there, and this laptop can't run the full-width base (§102.2).
- **The outcome-free data builds.** I didn't re-derive the exp + db graph, the v11 gene vectors, the probe split or the
  action-sign map.
- **The rest of `v11_probe.py`'s body.** I read its header and helpers, not every line.
