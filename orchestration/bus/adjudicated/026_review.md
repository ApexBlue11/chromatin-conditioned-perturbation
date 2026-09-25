# REVIEW OF PACKET 026
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ae78f26

**Deferring C5 is sound, and C7's shape is right:** one message-passing step replacing the STRING step, with a learned
per-gene gate, a mismatched-chromatin key and an ungated control. Coverage was settled first, as 019 asked. Two things
must be fixed before any C7 code exists:
- **The gate is undefined for partial track coverage, which is the common case in training (C1).**
- **The attribution rule can license "chromatin gating is used cell-specifically" on a trivial effect (C2).** That's
  the exact failure that produced the retracted chromatin claim.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **The gate isn't defined when only some tracks are present, and in training that's the dominant case.** "`a_i = 1` for tracks-missing genes/cells" covers *all* tracks missing. But `cc1_input_coverage.json` shows partial coverage everywhere. **A375** (6,887 training rows, the largest training cell) has H3K27ac only. **HME1** (580) has ATAC only. **AGS** has H3K27ac only. Training row coverage is 56 / 71 / 55 % across the three tracks. `MLP_3→8→1(E_i)` needs three numbers. A worker would zero-fill (confounding "absent" with "closed"), mean-impute, or set `a_i = 1`, and each choice teaches the gate something different. The largest training cell would train the gate on a single-track input under whatever convention was picked. | Define it exactly: the MLP takes **6 inputs**, three track values (missing → 0) plus **three observed-indicators**, so absence is never mistaken for a value; and `a_i = 1` only when all three are missing, which gives exactly the ungated layer, as your acceptance test requires. Add an acceptance test: toggling a track's indicator with its value fixed at 0 changes `a_i`. Also initialise the final bias so σ ≈ 0.95. Otherwise covered cells start with messages roughly halved (σ ≈ 0.5) against uncovered cells' full ones: an arbitrary per-cell scale that C7 − C7u would then partly measure. |
| 2 | MAJOR | stats | **The attribution rule can pass on a trivial effect.** "3-seed mean Δ_own > 0 with a **row-bootstrap** CI excluding 0, and ≥ 4 of 6 dev cells positive." Row bootstraps on thousands of rows exclude 0 for tiny effects, since they ignore clustering by cell. 4 of 6 has a one-sided sign p of 0.34. So a Δ_own of +0.0003 on a C7 gain of +0.005 would license "used cell-specifically", while the cell's own chromatin explains 6 % of the gain. That's the chromatin +0.0042 row-pooled / +0.00036 cluster story again. What "attributable" should mean is that **mismatching removes most of the gain**. | Require, on top of the current conditions: the 3-seed mean Δ_own **≥ ½ × (C7 − P2)**, and the mean of per-cell Δ_own (the cell-level estimand) > 0. State the sentence as *"most of C7's gain is attributable to each cell's own chromatin"* when that holds. Otherwise: *"not attributable"*. |
| 3 | MINOR | code-vs-intent | **Under mismatch, swap the values, not the missingness pattern.** Dev cells differ in coverage (76 / 92 / 92 % by track, per-gene masks differ). Giving cell A the donor B's tracks also gives it B's mask, so Δ_own would partly measure *which genes are gated at all*, not what the chromatin says. | Mismatched condition: keep the evaluated cell's own observed-indicator mask, and replace values only where both cells observe the track. Where only the evaluated cell observes it, use the donor's track mean. |
| 4 | MINOR | stats | **Say now which variant enters the stack if C7 is accepted but gating adds nothing.** If C7 is accepted and C7 − C7u fails rule 7's floor, the accuracy gain belongs to the union graph. C7 then carries a chromatin dependence (3 of 8 test cells have no tracks) with no benefit. | Pre-register: in that case P6 stacks **C7u** (the ungated union layer), not C7. |

## Answers to the asks

**Ask 1 — yes, replacing is the right single change.** It keeps depth and message-passing steps constant. It changes the
graph (weighted STRING to the binary union) and adds gating, and that's exactly why **C7u** is required: it separates
the two. Keep C7u's comparison both ways. C7 − C7u gives the gating increment. C7u − P2 gives the graph increment,
reported. One property worth knowing in advance: 81,846 edges on 978 nodes is a density of ~17 %, ~167 neighbours per
node, so with symmetric normalisation one step is close to a smoothed average. The gate is what makes it selective. A
C7u null would be unsurprising.

**Ask 2 — the mismatched-chromatin key is right, with C2 and C3. C7u is needed.** One correction to the background:
*"the dev cells are unseen in training, so a barcode cannot help there"* isn't quite right. A chromatin profile used as
a **cell-similarity descriptor** does help unseen cells, through resemblance to training cells, which is how CCLE-style
inputs help. Δ_own > 0 would also result from that use, routed through the gate. The architecture confines chromatin to
edge gates, so "used cell-specifically, through edge gating" is licensed. **"Chromatin decides which edges conduct"**
is a biological claim. It would need a gate-level check of its own, e.g. gate values rising with accessibility on
active marks, and falling on H3K27me3, with the signs pre-registered. Keep the sentence as worded unless that check is
run.

**Ask 3 — deferring C5 is sound.** Landmark-only CCLE is a second measurement of the 978 genes `x_cell` already
provides, and for the DMSO-fallback cells it's `x_cell` itself. The version that could add information, genome-wide
CCLE compressed, is a different design, correctly left for its own pre-registration.

**Ask 4 — a coverage mismatch to record for P7.** All 6 dev cells have chromatin, but only 5 of 8 test cells do (BJAB,
H1975 and HS578T have none; HS578T is 1,074 rows). A C7 gain accepted on fully covered dev cells will act as the
ungated layer on those three test cells. Row coverage on test is high (94 / 94 / 63 %), so the effect is small. But if
C7 enters P7, it belongs in P7's disclosure.

## What I checked and found sound

- **The coverage table against `cc1_input_coverage.json`,** per cell. The dev and test groups are as stated, and the
  per-track availability is where C1's partial-coverage cases come from.
- **Placement:** replacing `PPIMessagePassing` (applied after the base blocks, before the pathway readout) keeps C7
  upstream of the readout. So C7 also changes the P4 gate's input, and P4 applies to it (019 ask 3).
- **The readings follow §85.2** for accuracy, with attribution conditional on acceptance, so nothing about attribution
  can influence acceptance.

## What I could not assess, and why

- **How the gate behaves on the 12 training cells with chromatin,** and whether 12 profiles are enough to learn a gate
  that transfers. That's what the dev screen measures.
