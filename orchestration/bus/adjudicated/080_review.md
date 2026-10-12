# REVIEW OF PACKET 080 (D's embedding precompute: code, inputs, kernels)
verdict: SOUND
reviewed_commit: a78792d

**The code does what 98.16 pinned, and the real inputs reproduce from my own code.**
- **The generated script:** I regenerated it from the vendored source. The source is `2bea3057`, and the output is **`dd41c66a`**,
  byte-identical to the committed one. Without the header, its diff against theirs is 13 lines removed and 13 added.
- **The inputs:** I rebuilt them independently from the bundle's `meta_pert_idx` / `meta_dose` and XPert's table (`682fd4af`,
  both local copies). They are equal by content to the committed files:
  - the compound map, including its order;
  - `idx2smi_all`;
  - the five slices (disjoint; their union is all 1,970 SMILES; 394 each; keys 0..393);
  - the dose list;
  - the NaN placeholder ((2, 1024), float32, all NaN).
- **The pins:** every on-disk sha1 equals the kernels' pins.
- **The tests:** 25 pass on rerun (141 s). The suite includes the PI's guard-world tests for both kernels.

The three points below are MINOR provenance additions for the merge and the records.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance (merge) | **The merge trusts each slice record beyond its cache sha1.** It checks only that each cache matches its own record's `cache_sha1`, plus (m1)–(m3).<br>**It doesn't check that a record:**<br>• is for slice *i* (`rec['slice'] == i`);<br>• carries the pinned slice, script and placeholder sha1s;<br>• carries the two weight sha256s;<br>• carries the pinned version line.<br>**Why it matters:** `kernel_sources` mounts each caption kernel's latest output. An edited or re-run caption kernel (a debug version, or a re-split) would therefore be accepted, as long as its captions cover the SMILES. | In the merge, for each *i*, require:<br>• `rec['slice'] == i`;<br>• `rec['verified_sha1s']` equal to the pins for `idx2smi_slice{i}.npy`, the script, the placeholder and `p10_pertdit_emb.py`;<br>• `rec['verified_sha256s']` equal to both weight pins;<br>• `rec['versions']` equal to the pinned line.<br>Add a guard-world test. |
| 2 | MINOR | provenance (records) | **`emb_record.json` keeps only `compound_map_sha1` of the inputs it verified.**<br>• **Missing from it:** the script, `idx2smi_all`, placeholder, dose list, manifest and `p10_pertdit_emb.py` sha1s, which the GPU kernel's and scorer's chain should carry (98.16 Ask 6).<br>• **Missing from both records:** the **torch version** and the **CPU model**. Caption numerics, and therefore beam ties (Ask 5), depend on them. The `versions` line has no torch. | Add `verified_sha1s` to `emb_record`, and the five slice records' sha1s. Add `torch.__version__` and `platform.processor()` (or `/proc/cpuinfo`'s model name) to both records. |
| 3 | MINOR | guard | **The version line is recorded but not asserted.** pip with `==` pins and rc 0 makes a mismatch unlikely, but the assertion costs one comparison. It also covers the `P10D_SKIP_PIP_INSTALL` test hook ever reaching a real run. | Assert `installed_versions` equals the pinned string, in both kernels. |

## Answers to the asks

**Ask 1 — yes, exactly 98.16 C1** (two code changes and a header).
- **The dose loop:** their three lines are kept verbatim, one indent deeper:
  - `tokenizer_bert(...)`;
  - `model_bert(**inputs)`;
  - `.last_hidden_state.detach().squeeze(0).cpu()`.
  The only change is that the result is stored as `dose_emb[d]`.
- **The prompts:** each is `f"The dosage is {d} micromoles."` on `float(d)` from the JSON list. That gives '0.1' … '40.0', and
  '10.0' equals their literal (`dose_values` asserts it).
- **`DOSE_LIST_PATH`** belongs to change (ii).
- **The gate's replacement `print`** uses `device`, which is defined earlier in `main()`.
- **`sys` stays imported but unused.** Harmless.

**Ask 2 — yes, the compact index is handled on the data side without loss.**
- **Why compact keys are forced:** their sanity check asserts `range(len(idx2smi))` (their lines 182–184).
- **The compound map** is the sorted bijection of the bundle's 1,970 XPert indices onto 0..1969, and `idx2smi_all` is its
  composition with XPert's table. Both are equal to my rebuild.
- **For packet 081, pin:** the h5ad's `pert_idx` = `compound_map[meta_pert_idx]` row by row, asserted ∈ range(1970), with the
  map's sha1 checked in the GPU kernel.
- **For the record:** 7 XPert indices carry two BRD IDs each (XPert indexes by SMILES). D's compound unit is therefore XPert's
  index, the same index v9's compound inputs use.

**Ask 3 — the timeout handling is right, and a partial checkpoint isn't needed.**
- **Why the handling is right:**
  - `TimeoutExpired` triggers `fatal` before anything persists. Kaggle's own 12 h kill also persists nothing.
  - The merge's `find_one` then fails loudly, so no partial cache can reach it.
- **Why no checkpoint:**
  - **Margin:** at the probe's slowest rate, 394 × 72 s = 7.9 h, against 11.5 h. To time out, the mean would have to exceed
    105 s per SMILES.
  - **Balance:** the slices are balanced (total SMILES characters 21,447–21,774; longest 389–457).
  - **Cost of the alternative:** a mid-run checkpoint would need a third code change (their interval of 500) plus plumbing for
    a partial cache.
- **If a slice ever times out:** record the recovery now. Re-split that slice's SMILES into sub-slices (data files, sha1s
  pinned), re-run them, and make the merge accept the sub-slice caches by a listed manifest (a code change, reviewed then).

**Ask 4 — a weight guard after loading is acceptable.**
- **The revision guard runs first.** Both repos' `main` is at the pinned commit, which I checked on HuggingFace today (unchanged
  since 2022). So the window between check and download is seconds.
- **The weight guard hashes the file the run actually loaded:** the one under `snapshots/<pin>/`. Had `main` moved, the glob
  would find nothing and the kernel would fail.
- **The cost of a mismatch** is a free CPU session, never a persisted output.
- **Tokenizer and config files** are covered by the commit-level pin, not by hash. That's acceptable.
- **Pre-downloading and running offline** is optional. Done naively it fails: `snapshot_download` with a commit sha doesn't write
  `refs/main`, so an offline `from_pretrained(repo)` can't resolve. Download `revision="main"` and check the snapshot's sha
  instead.

**Ask 5 — nothing structural would differ from their pipeline.**
- **What carries over exactly:**
  - captions and BERT embeddings are per compound at batch size 1, so slicing and merging are exact;
  - the merge uses their own cache-resume path;
  - the SMILES come from the very table their script reads;
  - fp32, `eval()`, `no_grad`, beam 5, `max_length` 512, and BERT on CPU, all as in theirs.
- **What differs:**
  - (a) CPU beam ties (disclosed);
  - (b) the image's torch 2.10, against PertDiT's environment at 1.13.1. That is float-rounding-level, the same class as (a). It
    is disclosed for the GPU kernel in 98.16; disclose it for the caption kernels too, and record it (C2);
  - (c) a caption longer than BioLinkBERT's 512 positions would crash, as theirs would. It fails loudly in the merge, not
    silently.

## What I checked and found sound

- **The code:**
  - `p10_pertdit_emb.py` in full: the guards g1–g3, m1–m3 and o1–o4, and the in-memory refusal;
  - the five caption kernels (they differ only in `K` and the slice pin) and the merge kernel, both in full, including the guard
    order, the subprocess timeout, the snapshot-path weight hashing and atomic persistence;
  - the metadata: CPU, internet on, the pinned image, and only `p10d-emb-inputs` mounted, with no expression data.
- **Reproduced:**
  - the script, by regeneration (`dd41c66a`) and by diff;
  - all input files, by my own independent rebuild;
  - all pinned sha1s;
  - the 25 tests.
- **Their pipeline** at the substituted lines: the sanity check that forces compact keys, the cache-resume branch, and the
  `negative_ctrl` copy.
- **HuggingFace:** both revisions and weight sizes.

## What I could not assess, and why

- **The runs themselves** (Kaggle), and the uploaded dataset's contents. I relied on the kernels' mounted-file sha1 guards.
- **The mutation check** (11 → 1 of 31). I didn't rerun it.
- **MolT5's throughput on the bundle's own SMILES.** I used the probe's rate.
- **The untracked `model/v9/p10_pertdit.py`, `test_p10_pertdit.py` and `orchestration/make_p10_d_kernels.py`** in the main
  checkout. They were written while I reviewed (W40, packet 081), and they aren't part of this packet.
