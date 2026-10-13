# PACKET 082 — CODE: review 081 adopted; the real timing kernel; D's scorer (W41, written by the PI); the caption-collision stratum (RESULTS §98.20)
packet_id: 082
created: 2026-10-13
repo_commit: (this commit)
type: **CODE REVIEW** (nothing has run on a GPU. No D output exists, and the scorer is committed before any.)

## 1. Review 081, adopted (diffs in this commit)
- **C1 (BLOCKING):**
  - the trainers run with `env=dict(os.environ, PYTHONUNBUFFERED="1")`;
  - `parse_timing` refuses markers that aren't strictly increasing, and t_epoch or t_test under `MIN_TIMING_S = 1.0`;
  - `test_pi_kernel_blinding_on_a_real_timing_run` now also asserts, on the real tiny CPU timing run, t_epoch > 1 s and
    t_test > 1 s per seed.
  - **The mutant without the variable is KILLED** by that test on this machine.
- **C2:** `build_kernels(timing_json_path=…)`, which replaces the free `t_test_s`. It:
  - re-derives the decision with `timing_decision` and refuses a mismatch;
  - refuses non-positive timings;
  - fatals on `not_run_cost`;
  - writes `T_TEST_S`, `TIMING_SHA1` and `TIMING_DECISION` into the run kernel, and both run records carry the last two.
- **C3, (v3b):** `test_ground_truth.npy` against log1p(X · T_all / S) − `ctrl_t_ref` for the test rows, to 1e-4, with T_all the
  median positive total over all h5ad rows.
- **C4:**
  - `PINNED_GPU_VERSIONS` is asserted in the GPU kernel's install block;
  - the timing JSON and the run records carry the version line, `torch.__version__` and the GPU names;
  - the real line is checked on Kaggle CPU by `lincs-p10d-verline` (the same image, the same wheels, internet off): pip rc 0 / 0; the version line 'scanpy=1.12.4 anndata=0.13.3.post0 diffusers=0.30.2 einops=0.8.0 huggingface_hub=0.24.6 rdkit=2023.09.1' equals the pin, and the trainer's imports (diffusers schedulers, rdkit logger, scanpy transform) work.
- **The mutation check** on these guards: **0 of 9 survive**. The mutants were:
  - buffered child;
  - order check off;
  - floor off;
  - v3b off;
  - v3b missing-ok;
  - not_run_cost builds;
  - redo-mismatch ok;
  - non-positive ok;
  - version assert off.
- **Tests:** 22 pass on CPU (`test_p10_pertdit.py`).

## 2. The real timing kernel (`external/kaggle_kernels/kern_p10d_timing/`), built from the merge's `emb_record`
- **Its pins:**
  - bundle `1444b253`, carves `d6fc7844`, `p10_mlp` `97f84874`, `p10_pertdit` `929ce1fe`;
  - trainer `a431f53b`, PertDiT source zip `06bf66bd`, compound map `918e08f6`, dose list `7941bcd3`;
  - drug embeddings `eba75fdc`, dose embeddings `c42bce8e`, cap trainer `e910a855`.
- **A local dry run** of the kernel's prefix, everything up to the trainer launch, against the staged datasets, on CPU:
  - all 11 sha1s pass;
  - the cap-trainer rebuild matches;
  - the cfg check passes;
  - the timing h5ad is built from the real bundle (no test row);
  - `check_embeddings` passes on the real merged embeddings.
  - **Result:** "every guard up to the trainer launch passed". Nothing was written to `/kaggle/working`.
- **`apexblue/lincs-p10d-src` is uploaded** (the five pinned files plus licences).
- **The run kernel isn't built yet.** It needs the timing kernel's `p10d_timing.json` (C2).

## 3. D's scorer: `score_p10.py` (W41; agy was blocked by a broken plugin hook, so the PI wrote it, disclosed)
- **Refs:** `pertdit` (seed 117, the D reading of record) and `pertdit_s118` (reported only), both labelled "PertDiT (our run of
  their harness)".
- **`check_d_provenance`, (d1)–(d6):**
  - (d1) the exact 5-file manifest, with sha1s;
  - (d2) `sha1s == PIN_D` (the run kernel's 11 pins);
  - (d3) `n_test` 13,445 and the seed, in both the record and the npz;
  - (d4) of_record is 117 True and 118 False;
  - (d5) the record's `timing_sha1` is the sha1 of the downloaded `p10d_timing.json`, and its `timing_decision` is in {run,
    run_capped} and equals `p10_pertdit.timing_decision` re-derived from that file;
  - (d6) the npz's arm and `stop_reason` agree with the record.
- **The cap rule:**
  - `pertdit` with `stop_reason == 'cap'` → FATAL "not run (cost)";
  - with `--d_truncated_reported`, every block is labelled truncated;
  - s118 is always reported.
- **Reported-only blocks under `d_reported`:**
  - (a) the own-target score: their transform recomputed from the bundle over all 68,830 rows, with zero-total rows left at zero
    as scanpy does;
  - (b) the caption-collision parts: `molecule_reading` on the collision molecules against the rest, with row-pooled means. The
    collision JSON's sha1 and its `rows_sha1` are pinned, and its rows must be a subset of the clean rows;
  - (c) the oracle bound, 0.99979.
- **`read_p10_d` / `--read_d`:**
  - of-record only, with the verdict re-derived (via `_decide`);
  - truncated and s118 JSONs refused;
  - four statements mapped from `verdict()`'s real strings.
- **Tests:** 90 pass (62 existing, unchanged, plus 28 D). The suite's own-target test computes the truth independently with
  scanpy.
- **Mutation:** 27 mutants. Two survived the first pass:
  - **d1's file-set check** was masked by the per-file sha1 loop. A missing-file world was added, and it is now killed;
  - **s118's relabel** is equivalent, because every s118 block is already reported.

## 4. The caption-collision stratum (review 081 C5), fixed before any D output
- **The definition:** clean rows whose compound's MolT5 caption equals any training compound's caption.
- **The size:** **46 of 346 clean molecules; 1,577 of 11,983 clean rows.** No molecule has mixed rows. The training compounds have
  1,448 distinct captions among 1,578.
- **The file:** `model/results/p10/p10d_caption_collision.json` (file sha1 `bb3d6fb3`, rows sha1 `6bed581e`, `-text`), written by
  `model/v9/p10_d_caption_strata.py`. It reads P9's `row_index` only, plus the bundle's split / pert columns, the units and clean
  JSONs, and the caption caches.

## ASKS
1. **Is C1's fix complete?** Is there any other path where the stamps aren't event times, e.g. the trainer's own stderr
   (`stderr=STDOUT`, also unbuffered now)?
2. **C2's binding:** is re-deriving the decision in the builder, plus (d5) in the scorer, enough to tie the run to the measured
   timing?
3. **(v3b)'s tolerance and its T_all:** they follow your formula. Any objection?
4. **The scorer:**
   - is (d1)–(d6) the right set?
   - is the cap rule the 98.16 C2 enforcement?
   - is the own-target transform their transform?
   - is the reader's statement set the registered "same rule"?
5. **The collision stratum's definition** (any training compound's caption, exact string equality): within bounds as reported
   only?
6. **Anything that should block pushing the timing kernel at Saturday's reset,** after O9 session 3 and B1?
