# REVIEW OF PACKET 105 (v11's structure head against popularity and a nearest-neighbour lookup; §102.9)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 6272a3d (code 598b614 + b70f56b; registration §102.8 at 4dfeac5)

**The comparison is the one §102.8 registered, and its numbers reproduce exactly from the saved per-compound values.**
- **From `per_compound`, with my own code:**
  - means;
  - paired compound bootstraps (10,000, `default_rng(0)`);
  - every setting and bin.
- **Held-out compounds, all 114:** head 0.2178, popularity 0.1669, NN 0.1742.
  - head − popularity **+0.0508 [+0.0068, +0.0966]**;
  - head − NN **+0.0435 [+0.0103, +0.0796]**.
  - Both equal the JSON, so **both registered verdicts fire as written.**
- **The code:**
  - the head is computed from `u_feats` alone, as the forward pass computes it;
  - popularity is `vocab_prior` on the landmarks;
  - NN is the highest Tanimoto of a known (status 0) compound whose head labels (`tgt_node`, ChEMBL) include the landmark,
    ties broken by popularity;
  - the expected-recall tie rule and the paired bootstrap match §102.8.

**The caveats are three wording points** (C1–C3). The first is a non-independence the text doesn't state.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | "the training-row setting agrees" (Ask 3) | **For the head, popularity and NN the two settings aren't different experiments.**<br>• All three are functions of the compound alone. The head uses only `u_feats`; popularity is global; NN uses fingerprints. The rows play no part.<br>• **100 of the 172 training-row compounds are also among the 114 held-out ones**, with identical per-compound values. Only 72 are new.<br>• So the training-row result largely repeats the held-out one on overlapping compounds. It isn't confirmation.<br>• The row setting matters only for the attention probe. | *"The head comparison doesn't depend on rows. The training-row compound set overlaps the held-out set (100 of 172) and gives similar numbers; it is not independent."*<br>If a second look is wanted, report the 72 training-only compounds separately. |
| 2 | MINOR | "the gain is concentrated in close analogues" (Ask 3) | **No difference between bins was tested, and the bins' CIs overlap.**<br>**Head − NN by bin:**<br>• < 0.3: +0.021 [−0.034, +0.077];<br>• 0.3–0.5: +0.044 [−0.005, +0.098];<br>• ≥ 0.5: +0.096 [+0.022, +0.183], on 20 compounds.<br>**Every method scores far higher on analogues** (NN 0.403 against 0.110 for < 0.3). In relative terms the head's gain over NN is similar across bins (≈ 24 % against ≈ 19 %).<br>So "largest in absolute terms on analogues, the only bin whose CI excludes 0" is supported, but *"concentrated"* is not. | *"The gain over NN is largest on close analogues (≥ 0.5: +0.096 [+0.022, +0.183], 20 compounds), the only bin whose CI excludes 0. The bins are not shown to differ."* |
| 3 | MINOR | naming the comparators (Ask 3) | **Two phrases claim more than the comparison shows.**<br>• ***"Beyond nearest-neighbour lookup"*** reads as beyond similarity methods in general. The registered NN is one specific lookup: the maximum Tanimoto per target, popularity tie-break. A similarity-weighted k-NN vote might do better.<br>• **The popularity margin is narrow:** the CI's lower bound is +0.007.<br>• ***"Does not turn it into better responses (R2-targets)":*** R2-targets is a no-claim (Δ +0.0002, 3 / 6). It shows no **detectable** gain, and it removes the target input and the head's supervision together. | **Verdict 2:** *"the head transfers beyond a nearest-neighbour lookup (the most similar known compound targeting each gene)"*.<br>**The closing sentence:** *"… the response model does not turn it into detectably better responses (R2-targets: input and supervision removed together) or into its attention."*<br>**Optionally,** give the popularity CI beside verdict 1. |

## Answers to the asks

**Ask 1 — yes, from the per-compound values, and the NN is right as registered.**
- **Every reported mean and CI reproduces exactly,** in both settings and every bin (above).
- **Only known compounds feed the NN.** `known_target_matrix` takes status-0 compounds' `tgt_node`, which are exactly the
  labels the head trains on (`head_y` from `tgt_vocab`). Probe compounds' targets are never written (`status != 1`).
- **Ties:** NN ties go to popularity (`lexi_score`). Out-of-vocabulary landmarks rank last for all methods, so no method can
  find a target outside the vocabulary.
- **I didn't re-run the script.** It rebuilds `V11Data` twice and loads 6 checkpoints, and the checkpoints don't carry the
  vocabulary needed for an outside recomputation of popularity. So the per-compound popularity and NN values themselves are
  checked by code reading, not recomputed.
- **The head values are pinned by the self-check:** they reproduce `v11_probe.json` with max |diff| 0.0.

**Ask 2 — yes, the correction is acceptable.**
- **§102.8 registers ranking equality** (*"the popularity ranking equals the ranking by each −targets checkpoint's bias"*).
  The first version's value test at atol 1e-5 was stricter than registered.
- **The order of computation, in both versions:**
  1. the count and bin checks;
  2. the NN scores, which involve no targets;
  3. the head logits;
  4. the head self-check against the already-published probe values;
  5. the raise.

  No popularity or NN recall is computed before that point. So the first stop revealed nothing.
- **The recorded deviation** (max 0.009 logits, all three seeds) is consistent with weight decay on a head that receives no
  loss. Its weights stay at init up to the same small shrinkage (review 104 C1).

**Ask 3 — the two verdicts are correctly applied.**
- **"Present for dissimilar compounds too remains not established":** right. Both dissimilar bins' CIs include 0, against
  NN and against popularity.
- **Edits:** C1 (the training-row setting), C2 ("concentrated") and C3 (naming the NN, "detectably").

## What I checked and found sound

- **`v11_probe_head.py` in full:**
  - `expected_recall` (the registered tie rule);
  - `lexi_score`, `landmark_prior`, `known_target_matrix`, `tanimoto_matrix`, `nn_scores`, `head_logits`;
  - `same_ranking`, `paired_bootstrap`, the verdict functions;
  - `main`'s order of computation, the self-checks and the summaries.
- **`data_v11.py`'s target construction:** status, vocabulary, `tgt_node` / `tgt_vocab`, `vocab_prior`, `head_y`.
- **The b70f56b diff,** and the self-check record (counts and bins equal, ranking equality in 3 seeds, head
  reproduction 0.0).
- **An exact re-derivation** of every summary and bootstrap CI from `per_compound`. The setting overlap: 100 compounds in
  both settings, 14 held-out only, 72 training-only.

## What I could not assess, and why

- **The per-compound popularity and NN values themselves.** That would need `V11Data`'s vocabulary and labels (the script's
  rebuild); they are checked by code reading only.
- **Whether a stronger similarity baseline** (k-NN vote) would match the head. It isn't registered (C3).
