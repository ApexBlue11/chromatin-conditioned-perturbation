# REVIEW OF PACKET 048
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 40dbbcc

**92.8 is read mechanically. I reproduced it from the prediction files** (`score_dev` functions on E1 s0, E2 s0 and P2's three
seeds):
- **The read:** E1 s0 0.44138, **Δ +0.00444 ≥ 0.0034, so ADVANCE**. Centred +0.00436.
- **Per cell:** all six medians against P2 as tabled; 4 of 6 cells.
- **E1 s0 − E2 s0:** raw −0.00575, centred −0.00099, and all six per-cell values as stated.
- **The log:** the 60bdcd48 mount line, `CHROMATIN ENCODING clean` (failed list includes HEK293T and VCAP), and GUARDs 4 and 5.
- **The seeds 1–2 kernel:** the header, seed flags, glob and slug only.
- **E2 seeds 1–2 on the new upload:** `--ablate_epi` with the v9 encoding is unchanged by 3495ada. The diff only moves the
  z-score loop into an `elif`, and the ablation code is untouched. So E2's seeds 1–2 on 60bdcd48 compute the same thing as seed
  0 on 75c58f52.

There are two MINOR points: the `align_dev` flag is unverified operator input (C1), and two one-seed sentences need their
caveats (C2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance | **`--chromatin_encoding` is operator-supplied, and nothing checks it against the checkpoint.** A wrong value gives a silently wrong rule-8 readout, the very failure the flag exists to prevent. That's E1's checkpoints with the default `v9`, or a P2-recipe checkpoint with `clean`. Unlike `ablate_epi`, which `align_dev` reads from the `.pt`, the encoding comes only from the command line. | Make `align_dev` refuse a mismatch. Either pin a map of checkpoint sha1 → encoding (as `P7_SHA1` pins P7: E1 s0 `4c95150f…`, then seeds 1–2 when they land), or require the arm JSON next to the checkpoint and assert its `candidate_flags.chromatin_encoding` (absent = v9) equals the flag. Either is outcome-free. Do **not** change `xpert_arm.py` now (Ask 2). |
| 2 | MINOR | overreach | **Two one-seed sentences are missing their caveats.** (a) *"Like T4 (94 %), and unlike E2 s0 (52 %), E1's gain sits in the drug-specific (cell-centred) component"* states a one-seed split as a property of E1, without "at one seed". The commit message, *"(drug-specific, like T4)"*, has the same problem. (b) **VCAP:** E1 changes two things for VCAP, dropping its failed H3K27me3 **and** rank-normalising its K27ac. "Consistent with the cleaned encoding removing VCAP's failed-track harm" can't separate the two. E1 +0.0165, T4b +0.0136 and E2 +0.0234 are also within single-seed per-cell noise (±0.01–0.03), so no ordering among them is resolvable. | (a) Add *"at one seed"* to the centred sentence. (b) Add *"(E1 also rank-normalises VCAP's K27ac, so the two changes are not separated here)"* to the VCAP bullet. The HEK293T, LNCAP and U937 bullets are fine as written: descriptive, one seed. |

## Answers to the asks

**Ask 1 — mechanical; C2 aside, the per-cell sentences are right.**
- **HEK293T:** "consistent with §91.11: HEK293T's harm sits in its good marks" follows from T4 +0.0225 against T4b −0.0005. E1
  (−0.0069) against E2 (+0.0174) agrees at one seed.
- **LNCAP:** "most of LNCAP's harm is not removed by cleaning" is a fair one-seed description.

**Ask 2 — the flag is the right minimal fix, with C1's check.**
- **Recording the encoding in checkpoints:** yes, eventually, but **not before E1's seeds 1–2 have run**. Any `xpert_arm.py`
  change alters its sha1. `kern_v9dev_e1_s1` pins 60bdcd48, so a new upload would make it refuse. Running seeds 1–2 on re-pinned
  code would need its own review, and the code would differ from seed 0's. Keep `lincs-v9-src` frozen at 60bdcd48 until E1's
  three-seed read, then make the metadata change with its own pin update.
- **Rule 8 at acceptance:** yes, use `--chromatin_encoding clean` for all three E1 checkpoints, under C1's verification.

**Ask 3 — four things to fix before next week's reads.**
- **The E1-vs-E2 comparator's per-cell convention.** 92.3 says "seed-paired Δ(E1 − E2) … and ≥ 4 of 6 cells" but doesn't
  define the per-cell quantity. Fix it now, before any seed 1–2 data: run `score_dev.py --centred --preds <E1 ×3> --baseline <E2
  ×3>`. That gives the seed-mean difference and the per-cell median of (E1 seed-mean row r − E2 seed-mean row r), the same
  convention as every other read. Then Δ is compared against max(0.003, 2√(s_E1²/3 + s_E2²/3)) as written.
- **The decision table, stated before the reads:**
  - E2 accepted, E1 accepted: the comparator decides;
  - E2 accepted, E1 not: E2's reading;
  - E2 not, E1 accepted: E1 against P2;
  - neither: no change.
  - E2's acceptance read comes a week before E1's seeds exist. That's fine, because each read is mechanical.
- **Code identity:** record in §92 that E2's seeds 1–2 ran on 60bdcd48 and seed 0 on 75c58f52, with the same computation (the
  3495ada diff). Keep the upload frozen until E1's seeds 1–2 (Ask 2).
- **Rule 8 for both arms at acceptance:** E2 takes `ablate_epi` from the `.pt`; E1 takes `clean` from C1's verified flag.

## What I checked and found sound

- **Recomputed from the npz files:** E1 s0's Δ (raw, centred, per cell) against P2, and E1 s0 − E2 s0 (raw, centred, per
  cell). All match 92.8.
- **The run:** the E1 log's mount, encoding and guard lines.
- **Code:**
  - the 3495ada diff of `xpert_arm.py` (the v9 branch and the ablation path are unchanged);
  - the `align_dev.py` diff, where the flag passes through to `XPertData` and the default stays `v9`;
  - the `kern_v9dev_e1_s1` diff against s0.
- **The readout JSON:** `v9_dev_align_E1_s0_aux.json` gives alignment 0.27306, training prior 0.22925, in-cell 6 of 6, and
  the checkpoint sha1.

## What I could not assess, and why

- **Whether reading E1's checkpoint with the v9 encoding would have changed its rule-8 value.** That's the size of the
  instrument gap. I didn't run the readout locally.
