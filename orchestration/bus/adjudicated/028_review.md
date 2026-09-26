# REVIEW OF PACKET 028
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 02c2c26

**No MAJOR problem. Nothing here should hold the seed-0 push.** The trainer and kernel are safe to launch:
- **Architecture:** a model built from the kernel's exact argv under the current code has the **same 190 state_dict
  keys and shapes** as the screened C8b checkpoint (`v9dev_c8b_dev6s0_seed0.pt`, 14.86 M float parameters). So GUARD 3
  can't fire spuriously after 6 h.
- **Code:** the four staged source files match 2c0bd5d byte for byte.
- **Implementation:** the probe and reader implement §88.1–88.3 and §88.6.
- **Tests:** I re-ran both suites, 9/9 and 8/8.

Three MINOR fixes belong in the probe and reader, which run on CPU after training, so none of them touches GPU time.
One is a gap in my own 025 C3 wording rule that the reader now makes visible (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **The "ANY seed" rule is conservative for the "beyond …" qualifier, but not for the sentence it substitutes.** If the output projection clears −0.02 with p < 0.05 on **one** seed, `read_moa_88.py:95-101` writes *"a named readout of a predicted signature that is itself target-aligned"*. That is a positive claim about the predicted signature on 1-of-3 evidence. 023 C2 showed untrained output projections clearing exactly that bar on 2 of 3 inits (u1 −0.0234, p 0.023; u2 −0.0211, p 0.039). The binary form is from my 025 C3, and it didn't anticipate the seeds disagreeing. (The data projection is model-free, so ANY and ALL coincide for it.) | Make the output-projection qualifier three-way, and record it as an amendment to §88.3's wording before any output exists. **Not aligned on any seed** → *"… beyond the model's predicted signature"*. **Aligned on all three** → *"… a named readout of a predicted signature that is itself target-aligned"*. **Mixed** → no qualifier, plus *"whether this goes beyond the model's predicted signature is not determined (output projection aligned on k of 3 seeds)"*. Keep ANY for the data projection, and assert its `S` is identical in all 8 JSONs (§86's identity check, S 0.5519). |
| 2 | MINOR | code-vs-intent | **`k̄` truncates where §88.2 says it rounds.** `probe_moa_88.py:239` has `int(np.median(...))`. §88.2 fixes `k̄ = int(round(median atom count))`. With D = 496 (even), the median is a half-integer whenever the two middle counts differ, and then the two forms give different mean drugs. | `k_med = int(round(float(np.median(...))))`, and record `k_med` and the raw median in the JSON (`k_med` is already written). |
| 3 | MINOR | provenance | **The reader can't tell which checkpoint produced which record, or whether two records are copies.** The probe JSON carries only basenames. `read()` checks epoch, the `cfg_from` basename, rows and quintiles, but not that the three trained records are seeds 0, 1 and 2, nor that the five untrained records are seeds 0–4 and distinct. A duplicated untrained record changes `sd_u`, and can hide the true minimum, so it can loosen both thresholds without any check failing. The training kernels print each checkpoint's sha1, and the probe kernels will pin it, but the chain stops before the JSON. | Write `sha1(checkpoint)` into each trained JSON, and `sha1(cfg_from)` plus the init seed into each untrained one. In `read()`, assert the trained sha1s equal the three printed by `kern_moa88_t{0,1,2}` (a pinned list in the reader, filled in before the probe kernels run). Assert every untrained `cfg_from` sha1 equals t0's, and the untrained seeds are exactly {0,1,2,3,4}. Also assert `n_compounds` = 496 and unseen `n` = 156, identity with §86. |

## Answers to the asks

**Ask 1 — yes.**
- **`score_within_strata`:** positives are permuted only among compounds sharing a stratum label. `pct` ranks are
  identical to `score()`'s. `p_s = mean(perm ≤ S)` is one-sided in the same direction.
- **Quintiles:** computing them once over all D compounds and reusing them for the subsets is the right choice. It keeps
  "response strength" one fixed definition across strata. Recomputing inside the unseen stratum would redefine it per
  subset.
- **The strength value:** it's computed from `b['y_delta']`, the measured delta, so it's identical across all 8 models,
  as §88.6 item 4 requires, and the sha1 check enforces that.
- **The reader's conjunctions match §88.3 term by term:**
  - All rows: `diff ≤ min_u`, `≤ m_u − 2·sd_u`, `p < 0.05`, `p_s < 0.05`, `S < null2_mean`.
  - Unseen: `diff ≤ min_u,unseen`, `≤ m_u,unseen − 2·sd_u,unseen`, `p < 0.05`, `diff ≤ −0.02`.
  - Both on all three seeds, with the gate on every seed. There's no −0.02 floor on all rows, correctly, since §88
    replaced it with the untrained calibration.
  - Precedence: INVALID, then VOID, then NULL on any gate failure, then SIGNAL, then SEEN-ONLY, then NULL. ddof 1 for
    both sds.
- **The key names the reader expects exist in what the probe writes.** `score()` returns `S, null1_mean, null1_sd,
  diff, p, null2_mean, n`, and a void record carries `row_sha1, epoch, untrained, cfg_from`. So no key mismatch will
  surface only when the real outputs arrive.

**Ask 2 — for the "beyond" qualifiers, yes. For the substituted sentence, no (C1).**

**Ask 3 — nothing I found would waste seed 0.** Specifically:
- **GUARD 3's WANT:** I rebuilt the trainer's cfg from the kernel argv. WANT matches it field for field, and it matches
  the screened C8b's cfg on every shared field. The only differences are fields absent from one side (`chromatin_edges`,
  `union_edges` False; `use_cell_ctl` True; `predict_l5` False in the dev cfg). The state_dicts are identical in keys and
  shapes. It would still be cleaner to assert WANT *before* `train_v9_gpu.main()` in t1/t2 (build the cfg from
  `build_parser()`), so the check can only ever cost seconds.
- **Batch and fold:** `tcfg.batch` 48 and `tcfg.fold` 0 come from `V9TrainConfig` defaults, and `a.batch` is None.
- **Seeding:** `probe_gpu` initialises CUDA (`torch.randn(..., device="cuda:i")`) before `seed_devices`, so
  `torch.cuda.default_generators` is populated and the reseed isn't skipped. If it ever were, the epoch-0 guard would
  catch it at the cost of one epoch.
- **Evaluation:** `evaluate()` makes no torch RNG call and runs in eval mode, so it can't re-synchronise the devices
  between epochs.
- **Budget:** 8.5 h against about 6.2 h for 12 epochs, and the probe refuses a budget-cut checkpoint.
- **The quantiser:** its edges are registered buffers, so they're in the saved state_dict, and trained models are
  probed with their own fitted edges.

## What I checked and found sound

- **§88.6's statement on u0–u2.** `train_v9_gpu.main()` calls `torch.manual_seed(seed)`, and `probe_gpu`'s only draws
  are on CUDA devices. `data_v9` makes no torch random call. The probe seeds immediately before construction. So u0–u2
  start from trained seeds 0–2's initial weights, and their quantisers differ (20k vs 40k rows), as stated.
- **The mean drug.** `atom_mask` True means *valid* (`key_mask = ~…atom_valid`, `model_v9.py:178`). So an all-ones mask over
  `k̄` rows is exactly "k̄ copies, mask exactly k̄". Only `u_feats` and `atoms` are replaced. `dose` and `time` are kept,
  which is right, since they're conditions, not drug identity.
- **The readout.** Δa is taken in eval mode under `no_grad`, the median over ≤ 4 rows. The degeneracy check uses max Δa
  > 1e-12, and the gate is the median cross-compound Spearman < 0.95 over §86's pair scheme.
- **The guards fire as stated:** `tcfg.fold == 0` on `--cfg_from`, `check_arch_match` before `fit_bins` (valid because
  the edges are registered at init), epoch ≠ 11 refused, and `post_pathway` required.
- **The DP seeding move** into `dp_seeding.py` is verbatim, apart from the stated CPU-test skip.

## What I could not assess, and why

- **Whether `apexblue/lincs-v9-src`'s current Kaggle version is the 2c0bd5d upload.** The staged directory matches, and
  GUARD 2 checks the trainer flags and the C8b layer in the mount, but not `dp_seeding.py`'s contents. A stale
  `dp_seeding.py` would fail on import, cheaply.
