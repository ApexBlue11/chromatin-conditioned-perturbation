# REVIEW OF PACKET 068, ADDENDUM: the O9 scorer entry (cdb78ec)
verdict: SOUND
reviewed_commit: cdb78ec (adjudication bb71fe4)

**The O9 `REFS` entry and the comparator rules in `score_p9.py` implement 97.3 / 97.4 / 97.6.**
- **Tests:** the working tree is clean and equals the commit (`score_p9.py` f0652a55). `test_score_p9.py` reran under
  `.venv-cuda`: **35 passed**.
- **Review 068 C1** is adopted as described.

## Challenges

No challenge, from the files I could read; the unknown is in what I could not assess below.

## Answers to the asks (the PI's check request)

**`comparator_verdict`:**
- **A v9 win becomes "NO v9-WIN CLAIM"** if the run isn't admissible for a v9 win (`run_record['admissible_for_v9_win']`, 97.3
  item 2) or the reproduction is **below** the band (97.6 item 3). Above the band, it's flagged only (`above_band` recorded,
  verdict unchanged).
- **The comparator's own win becomes "UNINTERPRETABLE as a model comparison"** (97.3 item 1).
- **References without these keys** (ridge) pass through unchanged.

**Reproduction:** the row-pooled mean of per-row Δ Pearson over **all** comparator rows (O9's 13,445, the full split, their
convention), with n recorded and the band [0.621, 0.669].

**Admissibility guard:** a missing `run_record`, or one without `admissible_for_v9_win`, is FATAL. O2's machinery writes that
key only at FINAL termination, so an INCOMPLETE O9 run can't be scored (97.3 item 2, §78.5). The run record's sha1 goes into
the output.

**What's unchanged:** the profile is loaded through `coldcell_h2h.load_profile` (O2's format), and the 1e-4 target agreement,
row-set and units guards are as at 0f3d0ae. The unadjusted verdict is kept beside the adjusted one.

## What I checked and found sound

- **The code:** the diff 0f3d0ae → cdb78ec of `score_p9.py` (the `REFS['xpert_o9']`, `comparator_verdict`, reproduction and
  admissibility blocks, and the output fields). The 35 tests, rerun.

## What I could not assess, and why

- **Whether O2's `finalize` sets `admissible_for_v9_win` exactly per §84.1 item 5** (best epoch ≤ 252 at the horizon;
  counter == 50 at an early stop). I took it as reviewed with O2. The scorer trusts that field.
