# REVIEW OF PACKET 098 (v11 design, §102, before any v11 model code)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 1690fbf

**The design carries the record forward honestly.**
- It states the low prior.
- It moves CCLE to the centre, following §101.6 / §101.8.
- It applies reviews 093 C5 / C6: the probe split, the exp + db graph, GO evidence filtering.
- It answers the 23 GB problem with per-cell shared propagation and 6-wide per-row signals.
- Starting every new term at zero is the right pattern.

**Two points must be settled before code, because they decide which runs are needed:**
- **C1:** R1's reference is not like-for-like.
- **C2:** no arm removes chromatin's *information*.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | R1 reference (Asks 3, 5) | **v11 changes the base's priors as well as adding FG,** so T-B's off arm isn't a like-for-like reference.<br>• §102 re-pretrains the **gene vectors** on the exp + db graph.<br>• It rebuilds **`M_pathway_v11`**.<br>• Its knowledge rule ("STRING experimental / database only") also covers the base's 978 × 978 **`STRING_adj_978_v9`**, which uses the combined score and so includes coexpression and text-mining.<br>• If any of these reach the base (gene tokens, GatedPPI, the pathway readout), **R1 conflates FG with prior changes.** §102 doesn't say which do. | **Choose before code:**<br>**(a) Keep the base byte-identical to T-B's off arm** (v9 gene vectors, v9 978 adjacency, v9 `M_pathway`), confine the new sources to FG, and disclose that the base's priors predate the knowledge rules. Add a test that v11 with FG at zero and the same base weights reproduces v10-C-off's forward output.<br>**(b) Add a reference "v11-base, FG off"** with the new priors (3 seeds, ≈ 10–14 GPU-h). It is then R1's reference, and it is also the "pure" reference Ask 3 asks about. |
| 2 | MAJOR | attribution design (Ask 3) | **"−gate" doesn't remove chromatin's information.**<br>• In §102, chromatin (16 + 4 mask) and CCLE enter the **node input x_v** as well as the gate.<br>• "−gate" makes only the *gate* cell-invariant. Cell-specific chromatin still reaches z_v.<br>• So R2-gate tests the gating *mechanism*. No arm answers the principal's question: does genome-wide chromatin contribute in v11?<br>• §102 also doesn't say whether "−CCLE (masked)" masks CCLE in both x_v and the gate. | **Define a "−chromatin" arm:** training-mean, cell-invariant chromatin in **both** x_v and the gate (§100.4's convention), with CCLE kept. Make it R2's chromatin attribution, replacing or alongside "−gate".<br>**State** that "−CCLE" masks CCLE in x_v and in the gate. |
| 3 | MINOR | R2-CCLE cell rule | **Only 4 of the 6 dev cells have CCLE** (HUVEC and NPC don't).<br>Those two can't show a CCLE effect, so §100.4's "≥ 4 of 6 cells" effectively requires all four CCLE cells. That is a stricter rule than for the other levers. | **Register R2-CCLE's cell condition over the 4 CCLE dev cells** (e.g. ≥ 3 of 4). Report HUVEC and NPC separately. Keep the Δ and threshold over all rows, or state otherwise. |
| 4 | MINOR | what "−targets" removes | **"t_d ≡ 0, the head off" removes two things:** the target input and the auxiliary supervision of the drug token. | Word R2-targets as *"target knowledge (input + auxiliary supervision)"*. Split it only if you want to attribute the two separately. |
| 5 | MINOR | target encoding (Ask 2) | **Three unstated pieces:**<br>• **The sign map:** ChEMBL `action_type` → {+1, −1, 0} isn't frozen. There are modulators, PAMs, partial agonists, and so on.<br>• **The sign channel for *predicted* targets:** the head predicts presence only.<br>• **Annotation status is confounded with compound class:** known targets go to well-studied drugs, predicted ones to the rest (52 % unannotated). | **Freeze:**<br>• the `action_type` → sign table (and commit it);<br>• sign = 0 for predicted targets, or a separate sign head.<br>**Report** strata by annotation status (known / predicted / none). |
| 6 | MINOR | the DTI probe (Ask 1) | **Grouping by same molecule doesn't stop near-analogue transfer through the structure head.** A probe compound whose analogue's targets are in training can "hit" through structural similarity, which isn't what the atom → gene probe is meant to show. | **Report probe scores by maximum Tanimoto to annotated training compounds,** in bins such as < 0.4 / 0.4–0.7 / > 0.7, as in audit99.<br>**State the probe's n and power.** If 20 % is still well powered, it withholds fewer targets from training than 30 %. |
| 7 | MINOR | the no-op at init and its implementation (Ask 4) | **Three details:**<br>• With W_k = 0 and U = 0 the forward pass is unchanged. But creating FG's parameters shifts the RNG stream for the base's initialisation unless FG is built after the base or uses its own generator. So "starts exactly as v10" must be tested with identical base weights loaded.<br>• From step 1, the auxiliary loss changes the trajectory (expected; state it).<br>• **The unique-cell count per batch** can exceed "a handful". Plates are cell-specific and packed whole, up to 16 rows each, so small plates mean more cells. | Test the no-op with identical base weights.<br>Implement the gated propagation factorised as **√a ⊙ (Â (√a ⊙ z))**, as v10-C's GatedPPI does, so no per-cell edge tensor is built.<br>Have the timing kernel record the distribution of unique cells per batch. |

## Answers to the asks

**Ask 1 — the probe split is sound, with C6.**
- **Excluding both splits' test compounds from the probe pool is right.** Their annotations must stay usable in the
  "known targets" setting, so they can't be held out from the model.
- **Grouping by InChIKey's first block with a fixed seed is right.**

**Ask 2 — two channels are adequate for a first version, with C5's frozen sign map.**
- **Predicted probabilities in place of missing annotations are acceptable.** Training covers soft inputs, because
  unannotated and probe compounds are about half the rows.
- Report by annotation status.

**Ask 3 — not as specified.**
- **Attribution:** it needs a chromatin-information arm (C2).
- **R1:** it needs a like-for-like reference (C1).
- **A minimal set:** full, −chromatin, −targets, −CCLE, with "−gate" optional, plus the reference of C1(b) if the base's
  priors change.
- **A pure-v10 reference isn't needed for R2,** since every R2 comparison is within v11. For R1 it is C1's choice.

**Ask 4 — yes, in the forward pass, with C7's test and factorised gating.**
- Per-cell shared propagation with 6-wide per-row signals is the right answer to the memory problem.
- Run the timing kernel first, as planned.

**Ask 5 — this follows from C1.**
- **If R1's reference is T-B's off arm,** keep v9's `M_pathway` (and v9's vectors and 978 adjacency) in the base.
- **If a new reference is run,** rebuild `M_pathway_v11` for both it and the arms.

**Ask 6 — register before code:**
- **C1's choice and C2's arms.**
- **Nothing read until every arm × seed exists,** as with T-B, with a seed-balanced run order.
- **The guard and retries** (§100.8), with collapse counts per arm and seed.
- **Epochs** fixed by §100.4's timing rule, for all arms.
- **A power statement:** with T-B's seed SDs (0.002–0.0035), §100.4's threshold is ≈ 0.005 contrast r. Gains below that
  can't be detected. For scale, T-B's trained-runs gap was +0.0017.
- **C3's rule for CCLE.**
- **The probe's metric, n and Tanimoto bins** (C6).
- **The strata and permutations as listed.**

## What I checked and found sound

- **§102 in full:**
  - the data (graph, chromatin, CCLE, the pathways to build);
  - the knowledge rules;
  - the model (node input, gate, gated K-round propagation, the per-row two-channel signal, readout, auxiliary head);
  - the arms, the reference, the cost;
  - the readings.
- **Consistency with reviews 093 (C5 / C6), 095 / 097 and §101.6 / §101.8.**
- **The memory argument:** per-cell z is 19,496 × 64 with sparse propagation, and the per-row signal is 6 × 19,496.

## What I could not assess, and why

- **v11 code and the timing kernel.** Neither exists.
- **Probe size and target coverage after the split.** They aren't computed in §102.
