# REVIEW OF PACKET 006
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 5076f9c

**Part 1: your gate call is correct. It is a fail, and I reach that independently.** The execution is the
best on this bus so far — pre-committed rule, null key run under identical code, endpoints anchored to
packet 005's cells, three defects found in delegated code before use. One operator defect below that
neither of us caught, which also answers ask 2.

**Part 2: I would not buy this run, and ask 8 is the most important question in the packet.** Not because
the design is unsound — it is fixable — but because a different 5.8 h settles the objective in
`state.json` and this does not.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **PART 1 — the operator does not mask what the hypothesis is about, and this answers ask 2.** `model_v9.py:139` builds `D = cat([w_u(u).unsqueeze(1), w_a(atoms)], dim=1)`, so the **global token is position 0 of the drug sequence**, and `modules_v9.py:245` applies `A_blend = alpha*A + (1-alpha)*I` to the **whole** matrix. At `alpha=0` the global token — which carries the Uni-Mol CLS plus descriptors and fingerprints — can no longer read the atoms, and the atoms can no longer read it. So the operator removes **global↔atom flow together with atom↔atom flow**, while the hypothesis (§56.1, as restated) is only about the latter. That is very likely where most of the 0.09–0.12 `S11−S10` degradation comes from, which in turn is why every estimand built on this operator has drowned in it. | Blend only the atom–atom submatrix: apply `alpha*A + (1-alpha)*I` to `A[..., 1:, 1:]` and leave row 0 and column 0 at their learned values. One line, inference-only, reuses everything. That isolates intramolecular contextualisation from the global↔atom pathway and should collapse `S11−S10` toward the size of the effect being measured. **So the answer to ask 2 is: retire this operator, yes — but the family is not exhausted, and the untried member is the one that matches the hypothesis.** |
| 2 | MAJOR | overreach | **PART 2 — this is not a three-source union; it is one co-annotation graph plus a 4 % increment, which answers ask 6.** From your own provenance counts: Reactome ∪ GO = 14,922 + 17,902 + 36,021 + 6,753 + 2,164 + 619 = **78,381 of 81,846 edges, 95.8 %**. STRING contributes **3,465 unique edges, 4.2 %**. With Reactome∩GO Jaccard at 0.546 the two co-annotation sources are largely one source counted twice. TxPert's monotone-improvement-per-graph result was obtained over sources that are not 55 % overlapping, so it should not be assumed to transfer, and "unioned biological graph" overstates the diversity in the record. | Either report it as "co-annotation graph + STRING increment" and drop the three-source framing, or show the per-source ablation that would justify the framing. If the edge gate is built, the provenance bit `p_ij` makes it free to report the gate's behaviour separately on the 3,465 STRING-only edges — which is the only subset where the graph is not co-annotation. |
| 3 | MAJOR | stats | **PART 2 — ask 7: the estimand you propose is row-level, and the claim is cell-level.** You name `unseen_cell` as primary, which is right — it is the regime the claim is about — and then propose "median of the per-row contrast plus a sign test", which is the estimand I recommended for the *drug* question where the unit is the signature. For a chromatin claim on unseen cell lines the unit of generalisation is the **cell line**, and this project has already been burned once by exactly that substitution: the retracted +0.0042 had a row-bootstrap CI of [+0.0036, +0.0049] and a cluster estimate of +0.00036 [−0.0054, +0.0062]. Do not re-import the row-level estimand into a cell-level question. | Pre-commit: per-cell-line paired delta, reported individually, summarised over cells; restricted to cells that **have** chromatin tracks (§55's five, not all eight — the no-track cells cannot carry the treatment); cluster bootstrap plus per-cell signs. Accept in advance that with ~5 clusters the interval will be wide, and say what width would count as informative **before** the run. |
| 4 | MAJOR | confound | **PART 2 — ask 5: the free null answers a weaker question than the claim, so the real price is ~11.6 h, not 5.8 h.** Mean-ablating `E` inside the gate at inference is within-run and free, but it asks "does this trained gate use cell-specific chromatin?" The claim is "chromatin selects which edges conduct", whose null is "a *static* learned edge weight does just as well" — and a 17 %-dense graph has a great deal to gain from any sparsification, chromatin-driven or not. Only a second run, trained with `E` held constant or shuffled across cell lines, separates those. A confound you have not named: **the gate's benefit may be sparsification per se**, which is your own stated reasoning turned against the design — if a dense graph is near-uniform smoothing, then *any* learned gate helps and chromatin is incidental. | Budget both runs or neither. Primary null: train the identical gate with `E` replaced by its training mean (static gate, same capacity, same architecture). Shuffled-`E` is the second-line control and only worth buying if the primary separates. The inference-time mean-ablation is worth reporting but must not be presented as the null for the claim. |
| 5 | MAJOR | overreach | **PART 2 — ask 8: §46.5 is the better buy, and it is not close.** `state.json` sets the objective as "establish whether v9 generalises to unseen cell lines better than published SOTA". Review 001 C8 left that claim **inadmissible**: v9's 0.4734 is fold 1, one seed, 21,151 rows; XPert's 0.383 ± 0.027 is a five-fold mean on 21,321 rows; no XPert run exists on any cold-cell fold. §46.5 converts the project's headline claim from inadmissible to admissible on identical rows, using `head_to_head_mdmt.py`, which already exists and already has its guards. It has **no null-key problem** — it is a direct paired comparison, not an ablation contrast — which matters when the last two probes both died on their null keys. §50.5 (drug-blind retrain) is second: it is clean, interpretable, and packet 005 has already made the drug branch the live question. The edge gate is third — a new hypothesis stacked on a measured null (+0.000360), on a graph that is 96 % co-annotation. | Spend the next 5.8 h on §46.5. If the cold-cell head-to-head holds, the paper has its central claim; if it does not, everything downstream of it changes and the edge gate would have been built on sand. |
| 6 | MINOR | stats | **PART 1 — the §63.3 threshold is not met in any meaningful sense.** After the correction, the primary split's endpoint span of 0.00492 exceeds the estimand's own CI width of 0.00487 **by 0.00005**, a 1 % margin on a bootstrap statistic. That is a tie, not a pass, and it should be recorded as one — particularly since the threshold was itself re-specified after the run when §62.3 named the wrong interval. | Record it as "met to within bootstrap resolution", or re-draw the bootstrap with a different seed and see whether the sign of the margin is stable. Either way it cannot bear weight. |

## What I checked and found sound

- **The gate call is right, and I reach it independently.** On `unseen_compound` the null key runs
  +0.01140 → +0.01303 → +0.01404 → +0.01502 → +0.01444: rising across four of five alphas, span 0.00304,
  **62 % of the hypothesis key's span**. A null key that moves by nearly two-thirds of the hypothesis
  effect has established that the operator shifts ablation effects generically. On the wording ambiguity:
  §62.2 says "monotone trend", §62.3 says "strictly monotone", and the only thing distinguishing them is
  a single dip at the last point. **Resolving a pre-committed ambiguity in the direction that lets you
  read your hypothesis, after seeing which reading does that, is the degree of freedom pre-commitment
  exists to remove.** It fails. Your reasoning reaches the same verdict; I would drop the 1.6:1 magnitude
  ratio from the justification, since that ratio was not part of the pre-committed rule and does not need
  to be — the trend alone is disqualifying.
- **Ask 3: no, `unseen_cell` cannot be read.** It was removed as primary in §58.3 *before* this run for
  seed-instability with a sign flip; its null key is quieter but not flat (21 %, non-monotone); and its
  hypothesis-key effect at alpha=1 is +0.00290, i.e. there is almost nothing there to rescue. Promoting
  the split where the result looks best, after the pre-committed split failed its gate, is the textbook
  case the pre-commitment forbids. Leave it.
- **Ask 4: the operator verification is strong apart from C1.** `alpha=1.0` bit-identical to the default
  and `alpha=0.0` bit-identical to `diagonal=True`, both at 0.00e+00, is the right pair of anchors;
  cross-atom flow 0 / 0.858 / 1.711 / 2.562 / 3.409 is linear in alpha as the blend implies; parameter
  count asserted at all five; padding and NaN checked. I verified the blend line and the sequence
  construction myself — C1 is about *what* is blended, not whether the blend works.
- **Anchoring verified.** Endpoints reproduce packet 005's 2×2 cells exactly on all three splits
  (−0.01012 / +0.00530, −0.01697 / −0.01814, −0.01843 / −0.02637), and `rows_sha` matching across the
  2×2, the hypothesis sweep and the gate sweep is the right provenance mechanism.
- **Running the null key under code that was not modified between the two runs**, and deferring the
  method-rule-18 fix until both had finished, is the correct discipline and worth more than the fix.
- **Finding three defects in delegated code before using it** — a capacity test that asserted nothing, a
  shared RNG that made the row sample depend on how many alphas were requested, and missing row
  provenance — is the reason this packet's numbers are trustworthy. The second of those would have been
  invisible in the output.
- **Part 2's self-corrections.** The `max_term` reasoning being wrong, the density being intrinsic to
  co-membership over 978 genes, and §47.5's parameter pair being a shape never trained are all surfaced
  against your own interest. The density arithmetic checks: 81,846 / (978·977/2 = 477,753) = 17.13 %.

## One observation that must NOT be used to rescue this run

The alpha JSON carries `score_full` per alpha, so the model's own quality can be divided out. On
`unseen_compound` it runs 0.4520 → 0.5688, +26 % — the operator is a large model-quality dial. Dividing
each effect by it:

| alpha | null key, normalised | hypothesis key, normalised |
|---|---|---|
| 0.00 | 0.02522 | −0.01410 |
| 0.25 | 0.02770 | −0.01601 |
| 0.50 | 0.02838 | −0.01680 |
| 0.75 | 0.02769 | −0.01649 |
| 1.00 | **0.02540** | **−0.01985** |

The null key's trend very nearly vanishes — it returns to its starting value (0.02522 → 0.02540) and is
non-monotone — while the hypothesis key's does not (34 % span, most extreme at alpha=1). So "the operator
is a quality dial and both curves ride it" explains the null key well and the hypothesis key poorly.

**I am not offering this as a reason to read the hypothesis curve.** I constructed this normalisation
after seeing that the pre-committed gate failed; dividing by `score_full` is one arbitrary choice among
several (headroom `1−score`, or `score − chance` would each give a different answer, and I deliberately
did not compute them to avoid selecting among them). That is precisely the move I objected to in review
005 C2, and I will not make it while criticising it. It is hypothesis-generating for a *future*
pre-commitment, alongside C1's atom-only mask, and nothing more. This run's answer is the gate's answer:
fail.

## What I could not assess, and why

- **Whether the atom-only mask in C1 would survive its own null key.** It has not been run. I expect
  `S11−S10` to shrink substantially, but that is a prediction, not a measurement, and it should be
  pre-committed before it is tested.
- **Whether the edge gate would work.** No result exists and I am not speculating; C2–C4 are about what
  the design would license, not about the outcome.
- **The novelty position on cell-conditioned graphs.** You state it as a claim about a literature search
  and I am not able to check the negative. The TxPert quotation about randomly-initialised node
  embeddings is consistent with a static graph, but absence of a published cell-conditioned graph is not
  something I can verify from here.
- **Whether §50.5 or §46.5 is the better second buy.** I have ranked §46.5 first on the grounds that it
  settles the stated objective; between §50.5 and the edge gate I have a weaker preference and would not
  defend the ordering strongly.
