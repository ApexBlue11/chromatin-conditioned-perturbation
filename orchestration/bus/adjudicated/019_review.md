# REVIEW OF PACKET 019
verdict: SOUND-WITH-CAVEATS
reviewed_commit: a05dba1

**The protocol's architecture is right.** It develops on a carve of training cells, never predicts test cells in dev
mode, touches the test once, and registers that final look as a second comparison with §71 untouched. That is the
design that avoids repeating the test-guided selection we disclose in XPert's recipe.

Two things must change before P1 runs, because neither can be fixed after the carve exists:
- **The carve must be size-banded, not uniform (C2).** XPert's training cells are far too uneven for a uniform random
  draw.
- **P4's gate protects less than the packet says (C1).** The one surviving interpretability readout is computed before
  the drug enters the model.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | overreach | **P4 protects a drug-independent property, and it's framed as keeping "the mechanistic interpretability".** In `LincsV9.forward`, `path_delta, pathways = self.pathway(h)` runs after the base blocks and PPI, **before any perturb block**, and the perturb blocks are where the drug first enters: atoms and `u` go only into cross-attention. `pathway_activations` are therefore identical for any two drugs sharing cell, control profile, dose and time. `probe_v9`'s alignment is a per-row Spearman between those activations and `mean|Δ|` over each term's genes (`interp_v9.py:80-100`). So what §37 established, and what P4 protects, is: *the named pathway nodes rank which pathways move **in this cell**, whatever the drug.* That's real, and it beats a label-permutation null, but it isn't a drug-mechanism association. When `use_aux` is on, it is also partly **supervised**: `aux_path` is trained on exactly that `mean|Δ|` target. The principal asked for mechanistic interpretability to be kept, and would read P4 as guaranteeing it. Separately, an absolute 5 sd floor lets a variant drop from the baseline's 8–12 sd to 5 sd and still count as "kept". | Scope P4's wording: *"keeps the named pathway layer's cell-level alignment (drug-independent; partly supervised by the aux loss)."* Make it a **non-inferiority** gate against the baseline on the same dev rows: variant alignment ≥ baseline alignment − a margin fixed now (e.g. 0.02), **and** ≥ 5 sd of its own null. Use the 3-seed mean, or say it's seed 0 only. State plainly that no drug-specific interpretability readout currently survives (§4.1a retracted atom → gene), so the gate can't protect one. |
| 2 | MAJOR | stats | **A uniform random 6-of-32 carve is a gamble on these cells.** I counted rows per training cell in `split_cold_cell_1`: **3, 32, 35, 37, 52, 54, 54, 54, 56, 56, 75,** 135 … 1,986, **3,001, 3,877, 5,881, 6,794, 6,887, 7,972**. Eleven of 32 have fewer than 100 rows, and four hold 58 % of all rows. A uniform draw of 6 likely brings in 2 or more tiny cells, whose per-cell medians are noise, and it's the ≥ 4/6 rule that reads those. It could also take a 7,972-row cell out of training, 17 % of the data, and let it dominate the dev per-row mean. The test side has the same shape: one of its 8 cells holds 10,969 of 21,321 rows (51 %). | Commit the carve **rule** before drawing: e.g. pick 6 by fixed seed from the 12 cells with 200–2,000 rows (224 … 1,986). That's ~6k dev rows and costs training ~12 %, with the giants kept for training and the tiny cells kept out of the per-cell rule. Record the cells, lineages and row counts before any training. Score dev with **both** XPert's per-row mean, which is what §71.3's final reading uses, and the mean of per-cell means, so a single cell can't carry acceptance. |
| 3 | MINOR | stats | **Ask 2: fix the accept threshold's floor now. `s0` from 3 seeds has 2 degrees of freedom.** Its 95 % interval spans roughly [0.52, 6.3] × σ, and the accept rule `2·√(s0²/3 + s_v²/3)` has no floor. A chance-low `s0` makes acceptance lenient, and it would be set after `s0` is seen. | Accept iff mean Δ ≥ **max(0.003, 2·√(s0²/3 + s_v²/3))**, the same floor as P3's advance rule, committed now. Optionally pool the seed sd across every 3-seed condition as the program proceeds. It's variance, not effect, so pooling is outcome-neutral if fixed now. Fix the seeds (0, 1, 2) in advance, and log **every** dev run, dropped ones included, with no silent reruns. |
| 4 | MINOR | overreach | **C1's prior omits the contrary measurement on the objective's own regime.** "§37: ablating atoms at inference already helps on all three of our splits" came from an earlier model. On the `sa0` checkpoint the atoms **help** on `unseen_cell`: median per row +0.00290 [+0.00176, +0.00378], sign p 2.7e−8 (§74; reproduced in 017). They hurt only on the compound splits. For a dev regime of unseen *cells*, C1's prior is mixed at best, and it depends on whether §45's model has drug self-attention on. | State both measurements in C1's prior. Name whether P2's baseline has drug self-attention on or off, since the atom effect on unseen cells has only been measured with it on. |
| 5 | MINOR | provenance | **P7's history needs disclosing, and its estimand fixing.** The packet says every design choice so far was read on test splits. Any that were read on `split_cold_cell_1`'s 8 test cells (the §45 model is a cc1 model) make P7's test cells not fresh with respect to the **baseline**. Only the dev-selected *increments* are clean. P7 also compares 3 v9 seeds against one XPert run, and "3 seeds" could mean averaging per-row scores or averaging predictions, which is an ensemble. | Pre-register P7's estimand: the per-row score averaged over the 3 seeds, with the ensemble reported separately (as you wrote) and never used as the primary. Pre-register the hierarchy: §71 primary, P7 secondary, and which claim each combination of outcomes licenses. Write down which earlier choices were read on cc1 test cells, and label P7 *"dev-selected increments on a baseline partly chosen with test-cell knowledge"*. |

## Answers to the asks

**Ask 1 — yes, it's sufficient for new selection, given C2's carve.** Two conditions:
- **In dev mode, every fitting step must exclude the dev cells**, not only the test cells: quantiser bins (`fit_bins`
  on training rows only), normalisation statistics, anything else fitted from data. The quantiser already fits on
  "training rows only". Assert that the dev cells are absent from the rows it's fed.
- **The protocol can't launder earlier test-guided choices** (C5). What it guarantees is that nothing *added from now
  on* was selected on test cells.

K = 6 is reasonable under C2's banded pool. With 6 cells, the "≥ 4 of 6" rule has a one-sided sign-test p of 0.34: it's a sanity
filter, not evidence of cell-level generalisation. Say so. That's what P7 is for.

**Ask 2 — see C3. Fix the numbers now:** the 0.003 floor on acceptance, the seeds, and the logging rule. The advance
rule is sound as screening. If `s0` is about half the floor, a variant whose true effect equals the floor is dropped on its first seed about 20 %
of the time.

**Ask 3 — a gate is right, but not this one as worded** (C1). It should be a non-inferiority gate against the baseline
plus the absolute floor, scoped to what the readout is. In practice it binds only on variants that change the pathway
layer's input or its supervision:
- **C2** changes `ctl_mix`, which feeds the readout.
- **C5**, if CCLE enters through `cell_ctx` → FiLM, which runs before the readout.
- **C7** gates the PPI message passing, which also runs before the readout.
- **C3 and C4** reweight the loss the aux term shares.

C1 (atoms enter only in the perturb blocks) and C6 (an output head) act after the readout and will pass almost by
construction.

**Ask 4 — yes, a second registered comparison with §71 untouched is the right use,** with C5's estimand, hierarchy and
history written down now. §71.7's XPert admissibility applies to P7 unchanged, since it's the same O2 run.

**Ask 5 — two coverage items to fix before building.**
- **C5 (CCLE):** check basal-expression coverage for all 40 cells of `split_cold_cell_1`, and pre-register how a
  missing cell is handled. Disclose in P7 that XPert doesn't take this input.
- **C7 (chromatin):** coverage for these 40 cells is likely partial. The earlier chromatin tracks covered 32–35 of our
  own cells. Pre-register the missing-cell handling and the mismatched-chromatin null key review 016 asked for.

Nothing needs dropping. Keeping C7 last is right given §82/§83.

## What I checked and found sound

- **The split's shape:** 32 training cells, 47,509 rows; 8 test cells, 21,321 rows. Counted from the h5ad
  `split_cold_cell_1` and `cell_iname` columns.
- **The pathway-alignment machinery:** per-row Spearman against `mean|Δ|` over member genes, and a null that permutes
  which activation carries which pathway label, with its own p-value and a 2 sd "beats" flag. It's a sound test of
  label-specific alignment, and it's what C1 scopes.
- **The sequencing:** noise before variants, one factor at a time, stacking checked against the best single variant,
  and the test touched once. Keeping §71's comparison on the committed model is right.

## What I could not assess, and why

- **Which earlier design choices were read on cc1's test cells.** Only the record can say (C5).
- **Seed noise on a cell-level dev carve.** P2 measures it, which is correct.
- **CCLE and chromatin coverage for XPert's 40 cells.**
