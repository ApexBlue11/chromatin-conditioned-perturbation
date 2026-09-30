# REVIEW OF PACKET 033
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ef74b6d

**The readings in A–C are right, and I reproduce every number from the npz files and the rule-8 JSON:** C6 is ACCEPTED,
with §85.10's "no detectable drug-specific change" wording and the "in this cell" licence. C7 and V1-drop are NOT
ACCEPTED.

P6 needs one conflict resolved before V2 lands, and I introduced it. If V2 is accepted and stacked, two §90 clauses from
my review 029 contradict each other at P7's headline (C1). Two smaller additions to F follow (C2, C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **If V2 enters the stack, §90.4 and §90.6 contradict each other at P7.** §90.4: *"If V2 is adopted, §85.2 rule 10's 'never the ensemble' is amended: a within-run snapshot average counts as that run's prediction."* §90.6: *"No row or sentence sets a v9 ensemble (across seeds **or snapshots**) beside XPert's single run as a comparison."* P7's headline is read under §71.3 against XPert's single run. With V2 stacked, each v9 run's prediction *is* a snapshot average, so §90.6 forbids the very comparison §90.4 makes routine. Both clauses are from my review 029, and the ban was over-broad. Its reason was compute: a 3-seed ensemble costs 3× the training. That doesn't apply to within-run snapshots, which cost one run, the same as P2. | Decide before V2's data exists (~15:00), and write it into §90. My assessment: **amend §90.6 to treat a within-run snapshot average like V1.** If accepted and stacked, it's part of v9's method, the headline compares v9 (with V2) against XPert as published, and it's disclosed as such. Add a required labelled row with **V2-last in place of the snapshot average**, so the ensembling contribution is visible beside the headline. The seed-ensemble ban stays as written. The alternative, V2 in P7 only as a labelled secondary, is also coherent. What isn't coherent is leaving both clauses as they are. |
| 2 | MINOR | stats | **F's confirmation rule drops §90.2's co-criterion for the stack.** §90.2 requires Δ_centred > 0 of *every* variance-reduction arm. A stack containing V2 is still one, and rule 7 plus rule 8 don't check it. | Add **Δ_centred(stack) > 0** to F's confirmation conditions whenever the stack contains V2, or V1. |
| 3 | MINOR | code-vs-intent | **F doesn't cover V1-full, which is still pending.** V1-full can still be accepted when `lincs-v1mc-s0f` lands (§90.3: accepted **and** Δ_centred ≥ V1-drop's +0.00096). §90.3 also requires replication on another architecture before stacking. | Extend F: if V1-full is accepted, P6 applies it to **C6's three checkpoints**, inference only with no GPU, using the same confirmation rule, C2 included. A same-sign paired Δ there doubles as §90.3's replication, since C6 is another architecture. |

## Answers to the asks

**Ask 1 — yes, all three.**
- **C6:** seeds 0.44179 / 0.44007 / 0.44327 (sd 0.00160).
  - **Rule 7:** Δ **+0.00477** ≥ 0.003, since `2·√(s0²/3 + s_v²/3)` = 0.00269, so the floor binds. Cell means +0.00746,
    cells 4/6 (SKBR3 −0.0004 counts against it).
  - **Rule 8:** alignment 0.2673 / 0.2628 / 0.2572, mean 0.2624 ≥ 0.2532 and > 0.2292. The checkpoint sha1s in the JSON
    (`103b48c1`, `481c0610`, `4fbe4bd0`) match the local files.
  - **Centred:** 0.47534 / 0.47366 / 0.47603, Δ_c **+0.00015** against thr_c 0.003, so the middle wording applies.
  - **In-cell:** 6/6, with seed means +0.076 / +0.071 / +0.059.
  - One note for the record: C6's alignment is **0.011 below P2's**. That's within rule 8's 0.02 margin, but it's a
    decline, not "keeps".
- **C7:** 0.44103 / 0.44049 / 0.43518, sd 0.00323, threshold **0.00421** (the doubled seed variance raises it). Δ
  +0.00197 also fails the 0.003 floor alone, so NOT ACCEPTED doesn't depend on s_v. Cells 5/6, centred +0.00162
  (reported). C7u and attribution don't run, per §89.
- **V1-drop:** paired Δ +0.00089 / +0.00099 / +0.00088 (mean **+0.00092** < 0.003). Centred +0.00096, cells 4/6, cell
  means +0.00118. The recomputed deterministic arm scores 0.43955, exactly C8b's 3-seed mean. "About 3 % of the +0.029"
  is right: 0.00092 / 0.0293.

**Ask 2 — F is acceptable with C1–C3.** On the "not undo" tolerance, the difference Δ(stack) − Δ(best component) has an
sd of about √(s_stack²/3 + s_best²/3) ≈ 0.0013–0.0019. And max(Δ(C6), Δ(V2)) is biased upward, because both components
were selected for passing. So 0.003 is about 1.6–2.3 sd before that bias, and a stack whose true effect equals its best
component will occasionally be "not confirmed". The consequence is mild, since P7 then uses the better single
component. I'd keep the rule as proposed, and state that operating characteristic beside it.

**Ask 3 — yes, and the reason for deferring is sound.** A §88 seed needs about 7.2 GPU-h (t0 took 7.18). After V2's 5.2
h, about 5.8 h remain before the reset, and a quota-killed kernel discards `/kaggle/working`, so starting one now
would likely waste it. The order is validity-neutral: §88 reads nothing from §85, and the reader refuses to run until
t1 and t2 are pinned.

## What I checked and found sound

- **All C6 and C7 seeds use the same code path.** Seeds 1–2 ran on the 684e6de-era `xpert_arm.py`, whose `N = 1` path
  is identical to the earlier one (review 030). Rows and targets are identical to P2's across all nine files.
- **§88 t0:** its sha1 `efd0e1cf…` matches the local `c8b_ckpt_v9_fold0_seed0.pt` and is pinned in `read_moa_88.py:20`
  before any probe output exists, as review 028 C3 asked.
- **The GPU arithmetic in E:** 19.0 h used, ~11 h left. V2 (5.2 h) plus a P6 run (5.1 h) is 10.3 h, which fits only
  just. If V2 overruns, P6 slips to Saturday as stated.

## What I could not assess, and why

- **Why V1-full s0 was cancelled at 6.2 h.** It isn't visible in the repo. The rerun's identity check will confirm the
  checkpoint and rows again.
