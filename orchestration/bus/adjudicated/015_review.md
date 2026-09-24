# REVIEW OF PACKET 015
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 4f6ccfc

**GO on O2, once C1 is committed before session 1.** I reproduce every v7 number from `cc1_v7/run_record.json`:

| check | measured | bar |
|---|---|---|
| float64 (81.1a) | 5.4e-16 / 4.8e-16, loss relative difference 0 | 1e-10 |
| DP against same-GPU split (81.1b) | 2.4e-9 / 2.7e-9 | 1e-5 |
| grad-None set | the ten, in every mode and tag | — |
| autocast on in every replica call | yes | — |
| memory per GPU | 7.37 GiB | 13 GiB |
| state round-trip (81.3a) | no field differs | — |
| three straight runs and the resumed run (81.3b) | bitwise identical | — |

The criteria (§81.5 at ccfcaee, 06:58; §81.6 at da3c5ac, 07:11) predate the run. The code does what §81 says.

This is now the strongest piece of engineering evidence on the bus. **DataParallel computes the recipe's function**
(float64 rules out everything but rounding). **The checkpoint is complete, and the resume is exact** (bitwise, with DP
and dropout on).

What remains is launch design, not measurement. **The integrity point is C1: pin down, before launch, which events
can end the run.** C2–C4 are cheap hardening.

**One correction to my 014:** the 1e-9 "finding" line assumed DP against split would be bit-exact because "the
two-term gradient sum is commutative". That holds only for a leaf that gets one gradient contribution per half.
`cell_emb` feeds both encoders, so per half it receives at least two contributions. The split path accumulates all of
them into one `.grad` in engine order. DP sums within each replica, then across replicas. Floating-point addition
isn't associative, so a small fraction of elements differ by one ulp. That gives rel_L2 of order 1e-9, which is what
you measured. Your hypothesis is right in substance, and the "finding" is my mis-set line, not a property of DP.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **Fix which events can end the run before it starts, or the stopping point becomes a choice.** The packet names two final events: their early stopping, and the session cap. But §78.5 (27f7738) gives *"the quota runs out"* as an example of a final guard stop that §71.7 reads. The packet instead treats a quota kill as a resume event. Crashes and abandonment aren't covered at all. That matters here. Their stopper monitors **test** loss4, which is logged every epoch, and you push each session after seeing that log. If a mid-run guard stop *can* be declared final, the PI chooses where the comparison is read. At a stop with `counter_at_end ≥ 45`, §71.7 would then admit a v9-win claim that waiting a few epochs might have overturned. Extending past the cap is the conservative direction for v9 claims. Stopping early is not. **A cap in sessions is also soft**: a crashed or re-run session makes "5" a judgment call. And slower epochs quietly shrink the horizon, since 25 % slower means ~20 % fewer epochs under the same cap. | Commit before session 1: **(i)** the only final terminations are their early stopping and reaching the cap. A quota kill, crash, platform failure or abandonment is **incomplete**: resumed if possible, and otherwise reported as no result, never read through §71.7. Amend §78.5's quota example to match. **(ii)** State the cap as a **cumulative completed-epoch horizon** (≈ 297, which is 5 × 59.4 at the v7 rate), not a session count, so crashes don't consume it. **(iii)** Between sessions, the only inputs to any decision are timing (Amendment E), state integrity and quota, never the logged loss. **(iv)** At early stopping, assert `counter_at_end == 50 == marker.counter` (patience 50). |
| 2 | MINOR | code-vs-intent | **A guard kill between the three saves leaves an inconsistent state set, and nothing checks it.** `save_state()` writes `full_state.pt`, then `resume_from.pt`, then copies `best.pth`, each atomic but not atomic *together*. Suppose a kill lands after `full_state.pt` is written but before `best.pth` is copied, at an epoch that set a new best. The next session then restores `best_score` for epoch *e* alongside the checkpoint from an earlier best. If nothing later improves, the final evaluation uses the wrong checkpoint. `restore_state()` copies `best.pth` without comparing it to `st['best_sha1']`, which was recorded for exactly this. The end-of-session record's sha1s describe the files as they are, so they won't catch it either. A `resume_from.pt` one epoch behind *is* caught by the first-epoch assert. The window is about a second per epoch, so it's rare, but it produces a silently wrong checkpoint. | In `restore_state()`, assert `sha1(best.pth) == st['best_sha1']` and `resume_from['epoch'] == st['epoch']`. Better still, **stop only at epoch boundaries**: in the save hook, after saving, exit cleanly if `elapsed + one epoch > deadline`. A time-based twin of `LINCS_STOP_AFTER_EPOCH`, already proven, which makes the set always consistent and stops losing the partial epoch. |
| 3 | MINOR | provenance | **Ask 4: verifying sha1s against a record carried in the same upload checks integrity, not provenance.** If `run_record.json` of session k−1 comes in with the state, attaching the wrong version (k−2 with its own record) passes. Separately, the run will span two weeks. v7's equivalence proofs ran on `torch 2.10.0+cu128`, and an image update between sessions would move the stack mid-run. | Paste session k−1's three sha1s and its final epoch as **literals into session k's kernel**, and commit them to git before the push. Chain of custody then runs through git. At restore, re-run 81.3a **in process** (seconds, and no test variable needed): compare `collect_state()` against the loaded `full_state` field by field, and hard-fail on any difference. Assert `torch.__version__` and the CUDA version equal session 1's, and pin the image if the API allows. A changed stack is a stop-and-return, not a silent continue. A one-epoch round-trip each session isn't needed: the mechanism is proven, and the per-session risks are provenance and environment. |
| 4 | MINOR | code-vs-intent | **The one new code path, the early-stop marker and termination, hasn't been exercised, and its failure would be quiet.** If the marker isn't written, their `main()` runs its post-loop testing and exits with return code 0. Unless the kernel treats "rc 0 without a marker" as an error, that end state could be classified as finished, or not at all. | Before session 1, run one short test-mode pass on CPU or a single GPU, with `LINCS_TRUNCATE_BATCHES` and a test-only patience override, where early stopping fires. Show the marker → termination → `counter_at_end == patience` → prediction chain end to end. In production, make "trainer exited without a marker and without a guard fire" fatal. |

## Answers to the asks

**Ask 1 — GO, conditional on C1 being committed before session 1.** C2–C4 are strongly advised, and none needs GPU time.
The equivalence evidence is complete for what O2 needs. Float64 settles semantics, DP-against-split settles fp32 parity
on the real devices, the in-replica assertion settles autocast, and 81.3a/b settles resume. The frozen ten are a
declared deviation with provably zero effect on one GPU. **Add** "DataParallel over both T4s (runtime patch inside
`forward`)", "ten unused parameters frozen" and "full-state resume hooks" to `RECORD['deviations']`. The v7 record
lists only the earlier five, and that list is what the paper will cite.

**Ask 2 — terminating at the marker is acceptable.** After `stopper.step` returns True, their code only reloads the
best checkpoint into memory, saves a loss figure and runs their test pass, and `decisions_locked` never reads that
pass. The best checkpoint was written to disk at the best epoch, 50 epochs earlier, and our prediction reloads it from
there. Two conditions: take `last_epoch_index` from the **marker**, not from the state directory, which holds one epoch
fewer because no LambdaLR step follows an early stop. And apply C1 (iv) and C4.

**Ask 3 — the rule is right, and the unit should be epochs.** Final termination by the cap, read through §71.7 with 45
unchanged, is §78.5 applied as written. At the v7 rate, five sessions is ≈ 297 epochs. That admits a v9-win claim if
XPert's best epoch falls at or before ≈ 252, comfortably past the warm-split anchor (≥ 214 epochs run, best at 164).
Whether that margin is worth ~40 GPU-h is your call. The rule's integrity is what C1 protects.

**Ask 4 — see C3:** git-committed literal sha1s, plus the in-process round-trip at restore, plus a version assert. No
per-session training check.

**Ask 5 — the 20 % timing difference doesn't need explaining before launch.** The price uses the slower figure, and
Amendment E measures the real epochs against it. It's plausibly host variance on Kaggle's shared T4s. A five-step
window is ~5 s of wall time, and a 4-vCPU host drives 10 loader workers plus DP's threads. The v7 code additions
(the localise check and the replica counter) cost microseconds. One refinement ties into C1 (ii): with the cap in
epochs, a slower real epoch costs quota rather than horizon, and Amendment E's > 25 % trigger then bounds that cost.

## What I checked and found sound

- **Every §81 check against its committed bar,** read directly from `run_record.json`. That includes the split
  reference sharing single-GPU's batch-shape sensitivity exactly (5.4e-4 both), which is what makes DP-against-split
  the right comparison. The fp16 scales were found by halving (64, 16), and `replica_calls_autocast_on` equals
  `replica_calls`.
- **The float64 gradients were kept in float64** (014 C2): `xpert_dp_probe.py:148` keeps `torch.float64` for the
  f64 tags. The 5.4e-16 is a genuine float64 difference.
- **The hardened localiser:** it moves only the set `{'drug_HG_embed'}` and raises on any other off-device tensor, so
  replica parameter copies can never be moved. The cache holds the source and checks it with `is`.
- **`xpert_resume_patch.py` does what 81.2 says, with the 014 fixes:**
  - The restore loads with `map_location='cpu'`.
  - The load is strict, with the key set asserted equal.
  - `--resume_from` without `LINCS_RESUME_DIR` is refused at `apply()`.
  - The first `train()` hard-fails if the restore didn't run.
  - The first epoch must equal the saved epoch + 1. That also catches `LINCS_RESUME_DIR` set without `--resume_from`,
    since `start_epoch` would be 0.
  - The save hook wraps `LambdaLR.step` on the class, so `state_dict()` stays picklable.
  - `epoch = last_epoch − 1` correctly accounts for the constructor's own step.
  - The `train` and `validate` patches replace module globals that `main()` resolves at call time.
- **81.3b is the test 014 asked for.** It ran in deterministic mode, with DP and published dropout: three straight
  runs bitwise equal, and the resumed run equal to them. Truncated epochs don't weaken it, since the boundary logic
  and the RNG draw count per iterator don't depend on epoch length. The LambdaLR halving at 40 and the switch at 70
  are functions of restored fields that 81.3a proves equal.
- **§78.5 was committed at 27f7738 on 09-23**, before any multi-session run, with the threshold unchanged.

## What I could not assess, and why

- **The launch code itself** (`run_train_dp.py`, the marker, the session-k kernel). It's to be built after GO, and C4
  asks for its one new path to be exercised.
- **Whether the Kaggle API allows pinning the image.** C3's version assert is the fallback that works regardless.
- **XPert's cold-cell convergence epoch.** Nothing released bounds it, so the ≈ 252 margin in ask 3 is a prior.
