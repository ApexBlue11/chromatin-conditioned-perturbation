# REVIEW OF PACKET 069a (amendment 96.12 and the code, answering review 069)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 011700c

**96.12 and the code settle review 069's C1–C4.**
- **The tests:** 17 pass on rerun.
- **The corrected texts** say what the numbers support:
  - 96.8 item 1's "what it removes";
  - 96.10's tautomer and physchem-statistics record;
  - 96.11, with the excluding-set table equal to my numbers and the ceiling clause withdrawn with its reason.

**One wrong-input path is left (C1, confirmed by running it).** `check_v9_specs` requires one **shared** key but not **which**
key. Every P9 seed file also holds `y_true` and `ctl_true`, so `v9p9_seed{k}.npz:y_true-ctl_true` (the **measured** Δ) passes.
Two further gaps are content pins. All three are MINOR, and one small change closes the first.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | guard (Ask 2) | **The prediction key isn't pinned.** P7-format seed files hold `y_pred`, `deg_pred`, `y_true`, `ctl_true` and `row_index`. I called `check_v9_specs` directly, and it **accepts** specs keyed `y_true-ctl_true`, `y_pred` or `ctl_true`. The reader applies the same check to the B3 sources and requires the comparisons' `v9_specs` to equal them.<br>So one wrong key, used consistently, reads the **measured** Δ as "v9":<br>• item 1 passes, since measured passes B3 (96.11);<br>• item 2 then compares measured data with the references. On the excluding set, measured (1.08) is above physchem (0.91), one of the averaging references.<br>That is a route to `LICENSED` with no v9 prediction read. | Pin the key in `check_v9_specs` to exactly the registered predicted-Δ key: `deg_pred`, which your tests already use (or `y_pred-ctl_true` if you prefer that). Refuse any other. Optionally, in `compare`, assert `deg_pred` ≈ `y_pred − ctl_true` on the loaded rows. |
| 2 | MINOR | guard (Ask 2) | **The references file's content isn't pinned at reading time.**<br>• `compare` checks a references file against its marker **only if a marker exists**, so a copy or a regenerated file without one is used unverified.<br>• The reader checks the reference **key** but never `input_sha1s['ref_file']`. | In the reader, require `input_sha1s['ref_file']` to equal the registered sha1 for that key: `c27e405545f2…` for `nn1`, `nn5` and `physchem`, and the baselines file's sha1 for ridge. In `compare`, make the marker mandatory for the three chemistry references. |
| 3 | MINOR | guard (Ask 2) | **The v9 files are pinned by path, not by content.**<br>• The regex checks only the basename, so any directory is accepted.<br>• The B3 JSONs carry no sha1 of their delta source.<br>• Each comparison JSON records the v9 files' sha1s in `input_sha1s`, but the reader doesn't compare them across the four comparisons or with the files P9 was scored on. A re-download or rerun between the B3 runs and the comparisons wouldn't be noticed. | In the reader, require the four comparison JSONs' v9 sha1s to be identical, and equal to the seed-file sha1s that `score_p9.py`'s output records. Optionally record the sha1s in `mechanism_stage_a`'s B3 JSON too, and check them. |

## Answers to the asks

**Ask 1 — yes, C1–C4 are settled.**
- **C1 (MAJOR), the reader's decision logic:**
  - **`LICENSED`** needs item 1 **and** at least one of {`nn5`, `physchem`, `ridge_pred-ctl_true`} passed. The 1-NN note is
    appended when `nn1` is among the passes.
  - **`B3_ONLY_1NN`** when exactly `nn1` passes.
  - **`B3_ONLY`** otherwise when item 1 holds, and **`NO_CLAIM`** when item 1 fails.
  - **96.8 item 1's correction** states the residual and where it is largest. This is the stricter of the two options, and the
    mechanism text matches.
- **C2:** the 96.11 wording now matches the numbers:
  - the excluding-set table, which matches my figures to the second decimal;
  - the twins disclosed;
  - the SNR sentence limited to 5-NN and ridge, as untested;
  - the ceiling clause withdrawn, with the formula.
- **C3, the code:**
  - the four B3 sources go through `check_v9_specs` (distinct, P9 seed basenames in order, spec 4 = specs 1–3 joined, one key);
  - exactly four comparison JSONs, keyed by `ref_spec`, each with `v9_specs` equal to item 1's sources;
  - canonical labels from the key;
  - `beyond_R` needs exactly the four variant names;
  - the references sha1 is recorded.
  - Except for C1–C3 above, every path I traced refuses.
- **C4:** recorded in 96.10. "Training compounds" is defined as the usable molecules, and the cause is given as the tautomer
  pair.

**Ask 2 — yes, one path (C1).** C2 and C3 are content pins that close the remaining ways a stale or substituted file could
reach the reader.

## What I checked and found sound

- **The code:** the diff cd7d068 → 011700c of `stage_b_prime.py` (`REF_KEYS`, `AVERAGING_KEYS`, `check_v9_specs`, `compare`'s
  input block and outputs, `read_b_prime` in full).
- **The tests:** `test_stage_b_prime.py`, 17 passed on rerun.
- **The key probe:** `check_v9_specs` called directly with three wrong keys, all accepted.
- **The P7-format output keys,** read from `v9p7_seed0.npz`, and `score_p9.py`'s required keys.
- **The RESULTS diff:** 96.8 item 1, 96.10, 96.11 and 96.12.

## What I could not assess, and why

- **The 36-mutant run.** I didn't rerun it. The probe above is a mutant-equivalent input that the suite doesn't cover.
- **P9's outputs.** They don't exist yet, so whether the real seed files carry `deg_pred` is inferred from P7's format and the
  shared kernel.
