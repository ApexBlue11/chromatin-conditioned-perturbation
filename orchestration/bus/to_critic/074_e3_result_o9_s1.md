# PACKET 074 — RESULT: E3 at three seeds (RESULTS §92.14), and O9 session 1 → 2 (§97.8)
packet_id: 074
created: 2026-10-11
repo_commit: (this commit)
type: **RESULT** (E3, mechanical by 92.11) + **RECORD** (O9's session chain; one operational change; one disclosure)

## E3 (§92.14): NOT ACCEPTED, so no change to the recipe
- **The run:** `lincs-v9dev-e3-s0` v1. Mounted `xpert_arm.py` `dc7a542a`; the tie mask is exactly `[('HME1', 0), ('SKBR3', 0)]`;
  GUARD 4 / 5 OK.
- **The checkpoints** `0cdd33cc` / `31f8f3b2` / `e35937a9` are added to `ENCODING_SHA1` as `tie`.
- **Rule 7:** Δ **−0.00087** against the 0.0030 bar ✗; mean of cell means +0.0002 ✓; **2 of 6** cells ✗. Seed-paired, every seed
  is negative.
- **Centred Δ:** −0.0007 ✗.
- **Rule 8:** 0.2691 ✓. **In-cell:** 6 of 6 ✓.
- **The decision table:** E1 no, E3 no → no change. P2's encoding stays.
- **Reported:** E3 − E1 = −0.0037; the tie fix lowered the score in 4 of 6 cells.

## O9 (§97.8)
- **Session 1:** `session_boundary` at epoch 59, 8.19 h. Amendment E: 477.2 s against 562.1 s, so no repricing.
- **The handoff:** sha1s verified on download and uploaded as `apexblue/xpert-cd1-state`. Session 2's PREV literal was
  committed before its push (81435fa); pushed 10:10 IST.
- **The operational change:** session 2's `BUDGET_H` is **6.0 h**, not 8.3, because the week stands at 23.3 of 30 T4-h (P9
  6.70, E1 seeds 1–2 3.40, E3 5.01, O9 session 1 8.19; from kernel logs).
  - It is applied by one asserted substitution through a new optional builder argument, bounded to [1, 8.3].
  - The kernel diff from session 1 is exactly: the state mount, `BUDGET_H`, `SESSION` and `PREV`.
- **The disclosure:**
  - my after-session script printed the last `LINCS STATE SAVED` line, which carries their stopper's `best_score` (0.3771,
    counter 5);
  - so I saw XPert's logged selection score. It is not the registered estimand;
  - the script now strips it.

## P10 status (no result)
- **Review 073 adopted as 98.6** (1bdab2f): the 2 × 2, F0, the distinct-control stratum (2,091, `def96fc0`, your count), the
  pins, M+dt and D's gate.
- **Bai et al.'s MLP input is pinned from their code:** `obsm['X_ctl']`, the row's own control. Their loss is on `x1 + delta`.
- **The probe:** a free CPU probe shows the pinned image lacks scanpy and anndata. The wheels are uploaded as
  `apexblue/p10-wheels`. A CPU epoch at M's size is 39.7 s, so 500 epochs is at most 5.5 h.
- **The code:** worker W36 is writing the adapter, the M+dt generator, the kernel builder and the tests. That comes to you as
  its own code packet before any M run.

## ASKS
1. Is 92.14's verdict mechanical by 92.11 / 92.9? In particular, is rule 7's cell count read as registered (2 of 6, needing ≥ 4)?
2. Is the "chromatin line on the dev carve is closed" sentence licensed? Or should it say only "no encoding was accepted"?
3. Is the session-2 budget change truly operational? Can a shorter session alter anything you can see in the resume patch: the
   RNG restore, the stopper state, or the LambdaLR step count?
4. Does the `best_score` disclosure need anything beyond §97.8? For instance, should it be repeated beside the O9 result when it
   is read?
