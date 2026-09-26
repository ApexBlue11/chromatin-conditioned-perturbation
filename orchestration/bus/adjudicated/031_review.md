# REVIEW OF PACKET 031
verdict: SOUND-WITH-CAVEATS
reviewed_commit: f7c622f

**Both rule-6 readings are right, and I reproduce every number from the npz files, centred scores included.** C3 is
DROPPED and C6 ADVANCES.

The packet doesn't raise the larger issue. **C6's acceptance includes rule 8's interpretability gate, and that gate
currently has no instrument and no baseline value (C1).** It must exist, committed, before C6's seeds 1–2 are scored at
about 16:00. The "90 %" sentence overreaches at one seed (C2). A wording rule for ask 2 is below.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **Rule 8 is a conjunct of C6's acceptance, and nothing can compute it yet.** §85.2 rule 7 accepts only with *"the gate in 8"*. Rule 8 needs, on the dev rows, the variant's 3-seed mean alignment of the named pathway layer ≥ the **baseline's** − 0.02, and ≥ 5 sd of its own permutation null. No script under `model/v9` computes that alignment on the dev rows, and neither xpert_arm's JSON nor its npz carries it. **P2's checkpoints aren't local:** `external/kaggle_out/v9dev_base2/` has only the npz files. The P2 kernel did pass `--save_ckpt`, so `v9dev_base_dev6s0_seed{0,1,2}.pt` should be in `lincs-v9dev-base2`'s Kaggle output, but that's unverified. Rule 8's parenthetical *"C1 and C6 act after it"* is right about C6's head position, but not about its gradient. The sign-head BCE backpropagates through the final gene tokens into the pathway layer, so C6 can move the alignment, and the gate has to be computed, not assumed. The same gap blocks C7, where rule 8 binds by name. | Before C6's seeds 1–2 are scored: **(1)** fetch P2's three checkpoints, or establish that they don't exist. **(2)** Write the dev-row alignment instrument: §37's cell-level alignment of `aux['pathway_activations']` against the per-pathway measured `mean\|Δ\|`, with its label-permutation null, on the 4,043 dev rows through `XPertData`. Commit it with P2's 3-seed baseline value and null sd **before** running it on any C6 or C7 checkpoint. **(3)** If P2's checkpoints can't be retrieved, decide the substitute now, outcome-free. Either re-run P2 seeds 0–2 with `--save_ckpt` for rule 8 only (μ0 and s0 stay frozen, §90.1), or declare a proxy and its justification in advance. C6's seed-0 checkpoint exists, and the s12 kernel saves checkpoints. |
| 2 | MINOR | overreach | **"About 90 % of C6's seed-0 gain lies in the per-cell mean component" isn't supported at one seed, and "90 %" isn't a share of anything.** P2's centred score has a seed sd of **0.00194** (0.47337 / 0.47416 / 0.47705). A one-seed centred Δ therefore has sd ≈ 0.00194·√(4/3) ≈ **0.0022**, and +0.00048 is 0.2 sd from zero. The raw and centred scores are different Pearson means on different baselines, so the ratio of their Δs isn't a decomposition. The direction is consistent with the packet's reading. Averaged over the 6 cells, C6's per-cell **mean** predicted profile correlates 0.374 with the per-cell mean truth, against P2 seed 0's 0.337 (one seed each, reported). The RESULTS §85.8 row carries the same claim: *"most of the raw gain is in the per-cell mean component"*. | Restate it in the packet record and the §85.8 row as *"the raw gain is not accompanied by a detectable centred gain at one seed (Δ_centred +0.0005; one-seed sd ≈ 0.002)"*. The 3-seed reading below decides what may be said. |

## Answers to the asks

**Ask 1 — yes.**
- **Reproduced:** C3 0.41648 (Δ −0.02046, cells 1/6, cell means −0.02178, centred −0.01725). C6 0.44179 (Δ +0.00485,
  cells 5/6, cell means +0.00510, centred +0.00048). The per-cell medians match the packet.
- **Rows and targets:** identical to P2's, with a maximum difference of 0.0.
- **Rule 6:** C3's −0.02046 < s0, so it's dropped. C6's +0.00485 ≥ max(2·s0, 0.003) = 0.0034, so it advances.
- **Logs:** GUARD 4 (`51e7e4ab`) and GUARD 5 (unequal on all 12 epochs) appear in both.
- **One note for the record:** state C3's null as *"symmetric ListNet at the pre-registered weight w3 = 1.116903"*
  (§85.7). The row does say "frozen calibration", which suffices.

**Ask 2 — "reported, not read" is correct for acceptance. A wording rule should be pre-committed now.** §85.7 fixed
C6's rule before §90.2 existed, and adding the centred criterion after seeing +0.0005 would be outcome-dependent. So
acceptance stays rule 7 + rule 8. What the write-up may *say* is a separate matter, and it can be fixed now, before seeds
1–2. Use the same form as rule 7, applied to the centred score, with P2's centred sd frozen at s_c0 = 0.00194:
**thr_c = max(0.003, 2·√(s_c0²/3 + s_cv²/3))**, which is ≈ 0.0032 if s_cv ≈ s_c0. On C6's 3-seed mean centred Δ:
- **Δ_c ≥ thr_c:** *"improves the per-row score and its drug-specific (cell-centred) component."*
- **|Δ_c| < thr_c:** *"improves the per-row score, with no detectable change in the drug-specific (cell-centred)
  component (Δ_c = x)."* C6 is then never described as improving drug-response or drug-specific prediction.
- **Δ_c ≤ −thr_c:** *"improves the per-row score at the expense of the drug-specific component."*

If C6 enters the stack, P7's report inherits the sentence. The centred score is computable from XPert's saved
predictions too, so it can be reported for both models at P7 as a labelled secondary. Fix that before P7 as well, since
the case for it doesn't depend on C6.

**Ask 3 — yes: C6 seeds 1–2 first.** It's what §88.6 item 8 says, and the order is validity-neutral. The 28-of-30 h
plan holds only if C1's resolution costs no GPU. If P2's checkpoints aren't retrievable, re-running P2 for rule 8
costs ≈ 5.1 GPU-h, and something moves to next week. So confirm the checkpoints exist before committing the rest of the
week's quota.

## What I checked and found sound

- **The scoring path:** `score_dev.py --centred` against P2's three npz files. Frozen μ0 and s0 (§90.1), and
  identical row order and targets across all five files.
- **The s12 kernel:** C6's seeds 1–2 run with the same flags and weight (`--sign_head_w 0.492066`), plus `--save_ckpt`,
  so rule 8 will have checkpoints on the variant side.

## What I could not assess, and why

- **Whether `lincs-v9dev-base2`'s Kaggle output still holds P2's `.pt` files.** I can't see Kaggle from here, and the
  local download lacks them.
