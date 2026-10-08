# REVIEW OF PACKET 063, ADDENDUM: clearance check on ff91d11 (95.5)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ff91d11

**Cleared to run Stage A′ once.**
- **Code and tests:** `mechanism_stage_a.py` = 39a85084. I reran `test_mechanism_stage_a.py` under `.venv-cuda`: **29 passed**.
- **The fixes:** review 063's C1–C3 are implemented as 95.5 records.

There is one MINOR code-to-registration gap below. It's moot on these data, so it doesn't block the run.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | code vs registration | **95.5 item 2 makes A3 decision-bearing on cold-drug only if its units span ≥ 2 classes,** but `read_stage_a`'s main `a3_signal` (line 948) doesn't apply that rule. Only the P9 gate's `a3_multi` does. If a cold-drug A3 had units from one class, the §94.5 decision text (BOTH / A3) would still count it. On these data it's **moot**: EGFR gives 6 units, and the new DNA row gives 3 (4 "DNA inhibitor" parents in each of MCF7, A549 and A375), so the units span 2 classes. | Optional before the run: on splits with `noncns_a1_min`, set `a3_signal = a3_signal and len(a3_classes) >= 2`, with a test. Otherwise record in 95.5 that the class-count rule was satisfied by construction (9 units, 2 classes) and isn't enforced in code. |

## Answers to the asks

**Clearance: yes.**

**C1(i), the DNA → p53 (+) row:**
- **Membership:** by **exact** MoA string (`'DNA inhibitor' in unit_moas`, set membership, not a substring), gated to MCF7, A549
  and A375. The sign is fixed at +1.
- **Its units** enter the same within-cell permutation null and `d_std` computation as every other unit.

**C1(ii), `A1_noncns`:**
- **What it does:** removes the regex-matched CNS / neurotransmitter / ion-channel MoA strings from each unit's labels and
  drops units left with no label. That keeps A1's convention that only labelled units are candidates. It runs `a1_cell` with
  rng 9400 + k + 50.
- **The regex:** catches the classes I named in 063 (D2, 5-HT2a, Kir6.2, Na and L-type Ca channels, NET, SERT, β1, M1,
  histamine). It keeps COX, PDE, HSP90, DNA and kinase classes.
- **One borderline omission:** monoamine-oxidase inhibitors aren't matched, if any are present. That's acceptable, since the
  regex was fixed before data.

**C1(iii), the P9 gate:** `p9_gate.open` = (`A1_noncns` signal in ≥ 3 of 6 cells) or (A3 signal with ≥ 2 classes). The reader
appends the OPEN or CLOSED sentence verbatim from 95.5.

**C2 and C3:** A375 is in the MDM2 gate, and `read_stage_b`'s B1 threshold is `SPLITS[measured['split']]['a1_min']`.

## What I checked and found sound

- **The code:** the diff 7000c59 → ff91d11 of `mechanism_stage_a.py` (`SPLITS` cold-drug gates and `extra_classes`,
  `CNS_CHANNEL_RE`, `run_a3`'s MoA-string row, `read_stage_a`'s `p9_gate`, `run_pipeline`'s `A1_noncns`, `read_stage_b`'s B1).
  The 29 tests, rerun.
- **The RESULTS text:** 95.5, against the code.
- **Identity level, from 063's counts:** 4 "DNA inhibitor" parents in each TP53-wild-type scored cell, so 3 DNA units, plus 6
  EGFR units.

## What I could not assess, and why

- **Stage A′'s outcome.** It hasn't run.
