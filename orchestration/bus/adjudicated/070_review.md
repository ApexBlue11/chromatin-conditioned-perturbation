# REVIEW OF PACKET 070 (E1 at three seeds, 92.12; E3 code and kernel, 92.11)
verdict: SOUND
reviewed_commit: 77a7c37 (E1 read ad96ce5; E3 code 18262e3)

**A. The E1 reading is mechanical and right.**
- **The score:** I reran `score_dev.py --centred` on E1's three seed files against P2's (`v9dev_base2`, seeds 0–2) to scratch.
  The output JSON is **identical** to `v9_dev_score_E1_3seed.json`.
- **Rule 7 fails on the floor:** Δ = +0.002866 < 0.003.
- **The provenance checks out:**
  - the seed 1 and seed 2 checkpoints hash to `bfdb2d34…` and `e84afc78…`;
  - those sha1s are the ones in `ENCODING_SHA1` and in the rule-8 JSON;
  - the log shows the 60bdcd48 pin, GUARD 4 OK, and GUARD 5 unequal on all 12 epochs for both seeds.

**B. `encode_chromatin` is faithful to 92.11 + review 059 C8, and the v9 / clean identity holds on the real data.**
- **The identity:** on the real `E_final`, the 3495ada inline code and the new function give **byte-identical** values and masks
  for both `v9` and `clean`.
- **The tie identification:** on all **86** logged present channels, the per-channel non-tied count equals
  `E_peaks_log.txt`'s `nonzero_genes`, so the tie identification is validated channel by channel against the log.
- **The tests:** 7 pass on rerun.

There are two MINOR notes below; neither changes anything that was run or decided.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | arithmetic (92.12) | **The seed-noise term is 0.0026, not 0.0018.** 2√(s0²/3 + s_v²/3) = 2√((0.001689² + 0.001439²)/3) = **0.00256** (the 0.0018 looks like a division by 6). The floor of 0.003 still binds, so the verdict stands. The same formula gives E2's 0.0036 in 92.10 correctly. | Correct the figure in 92.12 (and the packet): *"max(0.003, 0.0026) = 0.003"*. |
| 2 | MINOR | guard wording | **The block-size guard checks N, not Z.** In `E_final`, every mark's covered values are an exact permutation grid r/(N−1): all distinct, max deviation 3e-8. So #{value ≤ (Z − 0.5)/(N − 1)} = Z for **any** Z. I passed Z ± 1 and Z ± 1,000: the guard accepts all four.<br>• Only the `TIE_EXPECTED_MASKED` check catches a gross Z error (−1,000 masks HME1 alone; +1,000 adds four H3K27me3 channels). Z ± 1 passes everything.<br>• This is the same property review 059 noted for `chromatin_genegeneric`'s assertion.<br>• Z is still right: ATAC and H3K27ac exactly, and H3K27me3 for 19 of 26 cells, from 93.16. My log check above confirms it channel by channel. | Describe the guard as *"N per mark (Z is not checkable from a joint-rank grid)"*. Optionally make `test_real_E_final_masks_exactly_the_registered_channels` assert, for every logged present channel, that the non-tied count equals the log's `nonzero_genes` (86 of 86 today). That is the check that actually validates Z. |

## Answers to the asks

**Ask 1 — yes, mechanical and right.**
- **Rule 7:** Δ +0.002866 against max(0.003, 0.0026) = 0.003 ✗ (C1's figure aside). Mean-of-cell-means +0.0028 ✓; 4 of 6 cells ✓.
- **The other conjuncts:**
  - centred +0.0021 ✓;
  - rule 8: 0.2731 / 0.2591 / 0.2807, mean 0.2709 ≥ 0.2532 and > 0.2292 ✓;
  - in-cell 6 of 6, seed means +0.081 / +0.065 / +0.068 ✓ (all from the JSONs, whose per-checkpoint sha1s I matched).
- **The decision row** is 92.9 item 4's "no / no", so **no change to the recipe**. The E1-vs-E2 comparator applies only when both
  are accepted, so it isn't run, correctly.
- **The permitted reading** is within the numbers (+0.0029, every seed positive, 4 of 6 cells, short of the floor). The 93.16
  caveat is carried.
- **E3's row is "E1 not accepted",** so it is read against P2, as 92.11's table says.

**Ask 2 — yes, faithful, and the identity proof is sufficient.** `tie` does what 92.11 registered:
- failed H3K27me3 tracks become missing first;
- in each remaining present (cell, mark), entries ≤ (Z − 0.5)/(N − 1) are set to 0;
- a channel with fewer than 10 present non-tied genes is masked entirely (values and mask);
- rank-normal with average ranks follows.
- **The masked set** on the real data is exactly [(HME1, ATAC), (SKBR3, ATAC)], and `XPertData` refuses anything else.

The identity proof for the model-of-record file is sufficient on four counts:
1. the verbatim-copy tests;
2. my independent check on the real `E_final` (values and masks, `v9` and `clean`);
3. the diff touches nothing else except the argparse choices;
4. the `v9` path still doesn't read `E_final_provenance.json`, so it gains no new dependency.

P7 and P9 pin 60bdcd48, so they can't mount the new file.

**Ask 3 — yes, cleared to push once `lincs-v9-src` carries 50ff83da**, which must be after P9 completes, as you state.
- **The diff** between `kern_v9dev_e3_s0` and E1's kernels is exactly: the arm flag (`tie`), seeds 0–2, the `_chromtie` glob,
  `ARM_NEEDS`, and the 50ff83da pin.
- **Guards 4 and 5** are unchanged.
- **The glob** without a seed tag can't pick up a stale file, because no `chromtie` JSON exists anywhere yet.
- **The timing:** three seeds in one session is about 5.1 h of training, judging by E1's 2-seed log (12,225 s), well under the
  session limit.
- **If P9 needs a rerun after the upload,** its pin will refuse it. Plan that before uploading, as 92.2 did for P7.

## What I checked and found sound

- **E1:**
  - the scratch rescore, byte-identical to the JSON;
  - the rule-7 threshold recomputed;
  - the seed 1 and 2 `.pt` sha1s against `ENCODING_SHA1` and the rule-8 JSON;
  - the log's pin and guard lines;
  - 92.3, 92.9 items 1–5, and the 92.12 text.
- **E3 code:**
  - the diff ad96ce5 → 18262e3 of `xpert_arm.py` (base = 3495ada's 60bdcd48), `align_dev.py` and `make_e_kernels.py`;
  - `test_chromatin_encoding.py`, rerun (7 pass, including the real-`E_final` test);
  - my old-vs-new identity check on the real `E_final`;
  - the permutation-grid property and the Z probe;
  - the 86-channel check against `E_peaks_log.txt`.
- **The boundary channels.** HCC15 and WSUDLCL2 H3K27me3 have **exactly 10** non-tied genes, which equals the log's
  `nonzero_genes`, so it isn't estimate-dependent. They stay present under "fewer than 10", as registered. Below 60: HUVEC 14,
  MCF10A 43, HCT116 56 (H3K27me3); HUH7 ATAC 33; SKMEL1 H3K27ac 19.

## What I could not assess, and why

- **The 7 H3K27me3 cells absent from the log** (A549, A673, HCT116, HEPG2, HL60, MCF7, PC3). Their tie split rests on 93.16's
  global-boundary estimate, and Z can't be checked from `E_final` (C2).
- **Rule 8, which I didn't rerun.** It needs the checkpoints on the dev rows (heavy on this laptop). I matched its JSON's
  checkpoint sha1s instead.
- **The 8-mutant run.** I didn't rerun it.
