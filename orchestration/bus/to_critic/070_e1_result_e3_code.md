# PACKET 070 — RESULT (E1 at three seeds, RESULTS 92.12) + CODE REVIEW (E3, 92.11: `encode_chromatin` + `--chromatin_encoding tie` + kernel)
packet_id: 070
created: 2026-10-10
repo_commit: 18262e3 (E1 read ad96ce5)
type: **RESULT + CODE REVIEW** (no E3 run yet; P9 RUNNING, O9 session 1 RUNNING since 18:12 IST)

## A. E1 at three seeds (92.12), read mechanically by 92.3 / 92.9
- **The runs:**
  - **Seeds 1–2:** `lincs-v9dev-e1-s1` v1 (pushed by hand 13:18, COMPLETE 18:11).
  - **Its log:** `mounted code verified for e1 (xpert_arm.py 60bdcd48039a)`; GUARD 4 `51e7e4ab…` OK; GUARD 5 distinct for seeds 1
    and 2.
  - **The checkpoints:** `bfdb2d34…` (s1) and `e84afc78…` (s2), added to `align_dev.ENCODING_SHA1` as `clean`.
- **Rule 7:**
  - Δ **+0.00287** (E1 0.43980, sd 0.0014; P2 0.43693, sd 0.0017) against max(0.003, 2√(…) = 0.0018) = **0.003**, so **✗**;
  - the mean-of-cell-means Δ is +0.0028 ✓;
  - 4 of 6 cells ✓;
  - the seed-paired Δ is +0.0063 / +0.0003 / +0.0020.
- **The other conjuncts:**
  - centred Δ +0.0021 ✓;
  - **rule 8:** aux alignment 0.2731 / 0.2591 / 0.2807, mean 0.2709 ≥ 0.2532 and > 0.2292 ✓;
  - **in-cell:** 6 of 6 ✓.
- **The verdict:** **NOT ACCEPTED** (only rule 7's 0.003 floor fails). 92.9's table, with E2 not accepted: **no change to the
  recipe**.
- **The permitted reading,** and the carried 93.16 caveat, are in 92.12.
- **Commands:**
  - `score_dev.py --centred --preds <E1 ×3> --baseline <P2 ×3>` → `v9_dev_score_E1_3seed.json`;
  - `align_dev.py --readout aux --no_nulls --chromatin_encoding clean` → `v9_dev_align_E1_3seed_aux.json`.

## B. E3 code (92.11; PI-written, disclosed)
- **The refactor:**
  - `xpert_arm.py`'s inline encoding (the `v9` and `clean` branches) is moved into a pure
    `encode_chromatin(E, Em, cidx, failed, encoding, tie=TIE)` that `XPertData` calls.
  - **The `v9` and `clean` paths are the same operations:** `test_v9_and_clean_are_the_lifted_code_exactly` and the
    average-rank duplicate test compare against **verbatim copies** of the old inline code, `np.array_equal`.
  - **The `v9` path is the model of record** (P2 / P7 / P9). P7 and P9 pin the old file (`60bdcd48`), so they are untouched.
    Any later `v9` run computes the same thing on the new file.
- **`tie`:**
  1. **The guard:** the pinned `TIE` block sizes must describe the loaded `E_final`, using `chromatin_genegeneric`'s assertion
     (mask sum == N and #≤ thr == Z per mark), else FATAL.
  2. **Failed H3K27me3 tracks** become missing, as in `clean`.
  3. **Per present (cell, mark):**
     - values ≤ (Z − 0.5) / (N − 1) are set to 0;
     - if fewer than 10 present non-tied genes remain, the channel becomes missing, mask and values (review 059 C8).
  4. **Rank-normal** (average ranks) per present (cell, mark).
  - **`XPertData` refuses** unless the newly masked channels equal `TIE_EXPECTED_MASKED = [('HME1', 0), ('SKBR3', 0)]`, the two
    channels 92.11 named before the code.
- **Tests (`test_chromatin_encoding.py`, 7):**
  - v9 / clean identical to the lifted code (with and without duplicates);
  - tie: the block shares one value below every non-tied entry; clean keeps its 12 distinct codes; non-tied order is
    preserved; the thin channel is missing; untied channels equal `clean`;
  - a wrong block size refuses;
  - the boundary entry of rank exactly Z stays outside the block;
  - unknown encoding;
  - **on the real `E_final`** (identity level): the pinned blocks hold, and the masked set is exactly HME1 and SKBR3 ATAC.
- **The mutation check:** 0 of 8 survive (block not zeroed, threshold 10 → 5, guard off, thin values kept, v9 std ddof, failed
  not masked, min-rank, threshold Z instead of Z − 0.5).
- **`align_dev.py`** accepts `--chromatin_encoding tie`. E3's checkpoint sha1s join `ENCODING_SHA1` before its read.
- **The kernel:** `orchestration/make_e_kernels.py e3 --seed_start 0 --seeds 3` → `kern_v9dev_e3_s0`.
  - It is E1's kernel with `--chromatin_encoding tie`, `_chromtie` and a pin of `xpert_arm.py` `50ff83da08c5…`.
  - Its `ARM_NEEDS`: `def encode_chromatin`, `TIE_EXPECTED_MASKED`, `TIE_MIN_NONTIED`, `E_final_provenance.json`.
- **The upload:** `lincs-v9-src` gets the new `xpert_arm.py` **only after P9 completes** (P9 pins the current version). Then E3
  is pushed into the first free slot: one session, three seeds, ≈ 5 GPU-h.

## ASKS
1. Is the E1 reading mechanical and right (92.12), including the decision-table row?
2. Is `encode_chromatin` faithful to 92.11 + review 059 C8, and is the `v9` / `clean` identity proof sufficient for a change to the
   model-of-record file?
3. Is the kernel cleared to push once `lincs-v9-src` carries `50ff83da`?
