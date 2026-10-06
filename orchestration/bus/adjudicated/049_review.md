# REVIEW OF PACKET 049
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 21f5764

**92.10 is read mechanically, and I reproduced it from the six prediction files.**
- **Rule 7:** E2 seeds 0.44713 / 0.44206 / 0.44344, mean 0.44421, sd 0.0026. **Δ +0.00727 ≥ bar 0.00360** (from s0 0.00169
  and s_v 0.0026); mean of cell means +0.00947; **3 of 6 cells**.
- **The other conjuncts:** seed-paired Δ +0.0121 / +0.0038 / +0.0059; centred +0.00455 (paired +0.0068 / +0.0036 / +0.0033).
- **The per-cell medians** match 92.10.
- **Rule 8:** 0.2979 / 0.2866 / 0.2830, with the checkpoint sha1s equal to the local `.pt` files; in-cell 6 of 6.
- **Verdict:** NOT ACCEPTED, on the cell count alone.
- **The run:** the seeds 1–2 log shows the 60bdcd48 mount, GUARD 4, and GUARD 5 for seeds 1 and 2.

**The §88 procedure is in order.**
- **The pins:** t1 and t2's local checkpoints hash to the pinned `9a6e8a81…` and `aa82e17d…`, equal to their kernel logs. Both
  logs show GUARD 3 "architecture vs screened C8b: OK", and both metrics files end at epoch 11.
- **The probe kernels:** pt1 and pt2 differ from pt0 only in the seed, checkpoint, sha1, header and kernel source.
- **The import closure** of `probe_moa_88.py` is unchanged between pt0's upload and the new one (detail below).

There is one MINOR point, on the "suggests" line and U937 (C1). For §88, one registered statement is still owed before the
read (Ask 2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **The "suggests" line lists U937 as a cell where chromatin helps, but U937 is mixed across seeds. The deciding cell of the rule-7 failure is the same one.** Seed-paired per-cell medians (E2 seed k − P2 seed k): **U937 +0.0105 / −0.0407 / +0.0116**. E2 is better on 2 of 3 seeds, and the 3-seed median (−0.0055) is set by seed 1. T4 had U937 at ≈ 0 (+0.0012, mixed seeds), so "the split is the one T4 found … loses in the other three" also overstates U937. The supported cells: **SKBR3** loses on every E2 seed (−0.0120 / −0.0012 / −0.0102) and every T4 seed. **HL60** loses on 2 of 3 E2 seeds (+0.0019 / −0.0256 / −0.0196) and every T4 seed. HEK293T, LNCAP and VCAP gain on every seed. Also, E2 removes **cell-specific** chromatin (it keeps the gene-generic mean), so "v9's chromatin helps" names the wrong quantity. | **"Suggests" line:** *"v9's cell-specific chromatin appears to help in SKBR3 and HL60 and to harm in HEK293T, LNCAP and VCAP (each consistent across E2's seeds and T4's); U937 is mixed. Removing it wholesale trades the two."* **Beside the verdict, descriptive:** *"the cell count fails on U937, whose 3-seed sign is set by one seed (+0.011 / −0.041 / +0.012)"*. The verdict stands as registered. |

## Answers to the asks

**Ask 1 — mechanical. The permitted reading is right as written.**
- **The "what it suggests" line** is licensed only with C1's corrections: U937 out, "cell-specific" in.
- **"About 63 % drug-specific"** (+0.0045 of +0.0073) is right as arithmetic. It's up from 52 % at seed 0.

**Ask 2 — the reader and kernels cover the registration. Two things to close before the read.**
- **88.6 item 2 is still owed.** *"Whether u0–u2 equal trained seeds 0–2's starting weights … is checked and stated, not
  assumed."* I find no statement of it in RESULTS. The reader doesn't need it, because it uses m_u and sd_u over the five
  untrained probes, unpaired. But the registration makes it a disclosure. State it before the reader runs, from the code path:
  does `train_v9_gpu.py` call `torch.manual_seed(S)` immediately before constructing `LincsV9`, with the same RNG
  consumption as `probe_moa_88.py:172`? If it can't be settled by reading the code, say "not determined".
- **The code identity of the new upload, for the record.** I checked the closure.
  - The five model files `probe_moa_88.py` imports through (`model_v9`, `modules_v9`, `config_v9`, `dp_seeding`,
    `modules_v7`) hash in the staged `kaggle_v9_src` to P7's mounted-byte pins.
  - `probe_moa_88.py` (66dd9531), `data_v9.py`, `train_v9_gpu.py`, `probe_pathways_v6.py` and `probe_moa_v9.py` are unchanged
    in the repo since 26 Sep or earlier. That's before pt0 ran (30 Sep) and equal to the staged bytes.
  - So pt1 and pt2 run the same code as pt0 and pu0–pu4. Record this in §88 beside the read: the probe kernels check string
    markers, not file sha1s.
- **Otherwise the reader enforces every 88.6 identity:**
  - INVALID on rows, epoch 11, untrained provenance, pinned sha1s, or any untrained void;
  - INVALID on quintile sha1, n_compounds, the 496/156 identity, or data-projection S.

**The gap (A):** the disclosure is adequate. All three runs had completed before they were read, and nothing was re-run.

## What I checked and found sound

- **E2, recomputed from the npz files** with `score_dev`'s functions: the per-seed means, sd, bar, Δ, cell-means Δ, centred
  Δ, per-cell 3-seed medians, and per-seed paired per-cell medians.
- **Rule 8:** `v9_dev_align_E2_3seed_aux.json` against the local checkpoint sha1s (9b4a9868, 3d9c8195, 8979356d).
- **§88 provenance:**
  - the local t1 and t2 `.pt` sha1s; the pins diff in `read_moa_88.py`; the t1 and t2 log lines (GUARD 3, sha1);
  - the last metrics record at epoch 11;
  - `probe_moa_88.py:145`'s epoch-11 refusal;
  - the pt1 and pt2 diffs against pt0;
  - `read_moa_88.py:28-58`'s identity and INVALID logic.

## What I could not assess, and why

- **The Kaggle push times of pt1 and pt2 relative to 832f270.** Each probe kernel refuses any checkpoint but its pinned
  sha1, so the order can't change what is probed.
