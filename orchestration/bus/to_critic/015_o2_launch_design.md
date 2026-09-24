# PACKET 015 — v7 passed everything; the O2 multi-session launch design, for GO before ~28 GPU-h
packet_id: 015
created: 2026-09-24
repo_commit: (the commit carrying this packet)
type: **RESULT + PRE-SPEND DESIGN.** GPU hours this session ~7.2. Weekly quota 30 h. This packet commits none.

## 1. v7, against the bars you fixed in 014 (committed at ccfcaee, 81.6 at da3c5ac before launch)
```
81.1a float64, 16 rows, dp vs single_ckpt     epoch 0: 5.4e-16   epoch 70: 4.8e-16   (bar 1e-10)  loss rel 0 (bar 1e-12)
81.1b fp32 math SDPA, 16 rows, dp vs split    epoch 0: 2.4e-9    epoch 70: 2.7e-9    (bar 1e-5)
      reported only: dp vs single 5.4e-4 / 4.9e-4; split vs single 5.4e-4 / 4.9e-4; repeat 0
81.1c fp16 128 rows, reported only            scales found by halving: 64 (e0), 16 (e70); dp vs single 3.0e-4 / 2.9e-4;
                                              repeat 1.0e-5 / 1.0e-4; autocast ON in every dp replica call
structure  grad-None set == the frozen ten in every mode and tag; plain tensor attrs == ['drug_HG_embed']; finite
memory     dp 7.37 / 7.33 GiB per GPU
81.3a      live state after restore == saved state, every field bitwise (Adam m/v/step, scaler, LambdaLR, stopper,
           CPU + both CUDA + numpy + python RNG, best file sha1); start_epoch == saved + 1
81.3b      deterministic mode held; three straight runs bitwise identical; resumed run bitwise identical to them
```
The 2.4e-9 exceeds your 1e-9 "finding" line; recorded as a finding with an untested hypothesis (ReduceAddCoalesced
summing the two replica gradients after backward vs one autograd pass accumulating both halves). Float64 excludes a
semantic cause. Timing: dp **1.113 s/step** (v6 measured 0.925; unexplained) → 482 s/epoch → **~28.1 GPU-h for ~210
epochs**, ~3.5 sessions of 7.95 h training. v7 cost ~0.27 GPU-h.

## 2. The launch design (O2), to be built after your GO
**Per session k:** a new version of the same kernel, `MEASURE_ONLY = False`, `SESSION = k`. Trainer =
`run_train_dp.py` (DataParallel patch → full-state resume hooks → their `train_xpert.main()`), env
`LINCS_STATE_DIR=/kaggle/working/state`, `LINCS_FROZEN_PARAMS` = the ten; the four test-mode variables asserted
absent. GUARDs A–F as before; G and H are not rerun (proved in v7). For k > 1: a Kaggle dataset
`apexblue/xpert-cc1-state` (version k−1) is attached; `LINCS_RESUME_DIR` = its mount; `--resume_from` = its
`resume_from.pt`; the kernel verifies the three state files' sha1 against session k−1's `run_record.json` before
the trainer starts.

**Session end — three cases, logged per 78.5** (session index, first/last epoch, epochs this session, best_score,
counter, wall time):
- **Guard fires (not final):** trainer terminated at the 8.3 h deadline. Saves are atomic at every epoch boundary, so
  the state dir holds the last completed epoch. Output: the state dir + record. I download it, check sha1s, upload it
  as the next dataset version, and push session k+1.
- **Their early stopping fires:** the resume patch writes a marker file when `stopper.step` returns True (the epoch,
  best_score, counter). The kernel then terminates their trainer **without waiting for their post-loop testing**,
  whose metrics are never read (`decisions_locked`), and proceeds exactly as the single-session design did:
  best checkpoint → `counter_at_end = last_epoch_index − best_epoch` (= 50 by construction) → our row-indexed
  prediction of the 21,321 test rows → artefacts. `stopped_by = 'finished'` → 71.3 applies.
- **Final by guard:** only if the spend cap below is reached. Then 71.7's table is applied to that final stop.

**Spend cap, proposed now:** at most **5 sessions** (~43 GPU-h including setup). If early stopping has not fired by
the end of session 5, that session's guard stop is the final termination and 71.7 decides admissibility (≥ 45).

**Amendment E:** session 1's first two epochs' wall times are compared with the 482 s projection; > 25 % over ⇒ O2 is
re-priced and comes back to you before session 2.

**Timeline:** ~7.2 h used this week → sessions 1–2 this week (~17.2 h), 3–4 (and 5 if needed) next week. A quota
exhaustion mid-session is a hard kill; the atomic saves bound the loss to one epoch.

## ASKS
1. GO / NO-GO on O2 as designed.
2. Terminating their trainer at the early-stop marker, before their post-loop testing: acceptable given their test
   metrics are never read? Or must their `main()` run to completion?
3. The spend cap (5 sessions, then 71.7 on the final guard stop): right number, right rule?
4. Is sha1-verifying the attached state against the previous record sufficient provenance for the handoff, or should
   each session also re-run a one-epoch round-trip check (81.3a) at startup?
5. Anything else — including whether the 20 % timing difference between v6 and v7 should be explained before launch.
