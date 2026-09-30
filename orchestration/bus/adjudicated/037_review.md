# REVIEW OF PACKET 037
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 8034ad1

**The §90.4 reading is right, and I reproduce every number from the 15 V2 prediction files.**
- **V2:** 0.45196 / 0.45852 / 0.45039, Δ **+0.01669** ≥ 0.00535, cell means +0.01796, **6/6**, Δ_centred **+0.01697**.
  So it's ACCEPTED.
- **V2-last:** +0.00773 (5/6, centred +0.00717).
- **Recovery:** 0.570 of the 3-seed ensemble recovered.
- **File integrity:** each main file is the mean of its three snapshots (max |d| ≤ 1e-6), and `_last` = `_snap2`
  exactly.

There's one gap in the fallback P6 would trigger (C1), and one confound in the "reported, not read" interpretation that
matters for the write-up (C2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **V2 was accepted without rule 8, and §85.11's fallback can carry it into P7 unchecked.** §90.4's reading (rule 7, cell means, cells, Δ_centred) omits §85.2 rule 8. My review 029 didn't ask for it either. §85.11 applies rule 8 to the *stack*, but if P6 is not confirmed, *"P7 uses the single accepted component with the larger Δ"*, which is V2, whose interpretability gate has never been evaluated. That's incoherent in exactly the case that matters. If P6 fails **on rule 8**, for instance because C6's −0.011 decline and some decline from V2's schedule add up below 0.2532, the fallback would promote V2 without asking whether V2 alone passes. V2 retrains every parameter under a different schedule, so rule 8 plausibly binds on it. | Before P6 lands: run `align_dev.py --readout aux` on V2's three checkpoints (`v9dev_v2_dev6s0_seed{0,1,2}.pt`, the final-snapshot weights, all local; inference only), and commit the result. Amend the fallback: *"not confirmed → P7 uses the single accepted component with the larger Δ **that passes rule 8**; if V2 fails rule 8, P7 = P2 + C6."* It's outcome-free, since P6 hasn't been read. |
| 2 | MINOR | confound | **"The cyclic schedule helps the single final model" isn't the only reading, and the data point to a simpler one.** Each **first** snapshot is just a single 4-epoch WSD run: warm-up, stable, decay to about 0, then evaluation. It already beats P2's 12-epoch run on every seed: snap0 0.4398 / 0.4468 / 0.4395 against P2's 0.4350 / 0.4383 / 0.4375, i.e. **+0.0048 / +0.0085 / +0.0020**, mean ≈ +0.0051, against V2-last's +0.0077. So most of the "schedule effect" is available from a *shorter* annealed run, with no restarts at all. That's consistent with 12 epochs over-fitting the training cells for cold-cell transfer. It's a confound in the interpretation, not in the accepted reading. | Word §90.7's reported line as *"V2-last − P2 = +0.0077; a single 4-epoch cycle (snapshot 0) already gives ≈ +0.0051, so the schedule effect is not attributable to warm restarts."* Don't claim a restart benefit in the manuscript without a pre-registered test, such as 1 × 4 vs 3 × 4 epochs. |

## Answers to the asks

**Ask 1 — the reading is right. One interpretation line overreaches (C2).**
- **Threshold:** max(0.003, 2·√(0.00169²/3 + 0.00431²/3)) = 0.00535.
- **Per-cell medians:** HEK293T +0.0155, HL60 +0.0176, LNCAP +0.0154, SKBR3 +0.0103, U937 +0.0178, VCAP +0.0206.
- **Guards:** GUARD 4 and GUARD 5 are in the log (unequal on all 12 epochs, all three seeds).
- **A strength to state:** V2 seed *s* and P2 seed *s* share their initialisation (`torch.manual_seed(s)`), their data
  order (`np.random.seed(s)`, and `D.batch` makes no draws, per review 030) and their dropout and stochastic-depth mask
  stream. The number of CUDA draws per step doesn't depend on the learning rate, and the cycle-end evaluations run in
  eval mode. So V2 − P2 differs **only** in the LR schedule and the snapshot averaging. The per-seed paired Δs (+0.0169 /
  +0.0203 / +0.0129) are cleaner than rule 7's independent-samples threshold assumes, so the acceptance is conservative.

**Ask 2 — yes, V2 alone is the right reading of "the single accepted component with the larger Δ", provided it passes
rule 8 (C1).** Adding the `v2` branch to `make_p7_kernel.py` before P6 lands is right. Its GUARD 6 must also cover
`_snap{k}` and `_last`, as the `c6_v2` branch does.

**Ask 3 — nothing to change beyond C1.** V2 took 18,040 s (5.0 h) for three seeds, and P6 adds only a linear head,
so ≈ 5.1 h is realistic against ≈ 6 h left. If the quota cuts it, the loss is time, not validity: a killed kernel keeps
nothing, and the rerun after the reset precedes P7's generation. Next week then holds P6 (≈ 5.1 h, if rerun) + P7
(≈ 5.7 h) + §88 seeds 1–2 (≈ 14.4 h) ≈ 25 h of 30.

## What I checked and found sound

- **§90.4's conjunction** against the recomputed numbers, and the reported quantities: schedule effect +0.0077,
  ensembling gain V2 − V2-last = +0.0090, and fraction recovered (0.45363 − 0.43693) / (0.4662 − 0.43693) = 0.570.
- **The kernel:** it's the P2 command plus `--snapshot_cycles 3`. Output naming follows review 030's note, and all 15
  prediction files (9 snapshots, 3 `_last`, 3 main) are present, with identical row order and targets to P2's.

## What I could not assess, and why

- **Whether V2 passes rule 8.** That needs the aux readout on its checkpoints (C1), which is an inference job I won't
  run on the laptop at this size.
