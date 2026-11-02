# REVIEW OF PACKET 110 (stage 2 design: candidate cell representations on the knockdown proxy; §103.6)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: b76b7af

**The plan is the right shape:**
- the rung-3 positive control before any real comparison;
- ablation twins within each family (M2 / M1, M3 / M4);
- a structural null (M5);
- every score against all baselines and relative to the ceiling;
- s_ctl only as a covariate, never an input.

**No input leaks outcomes** (Ask 1).

**Three design points should be settled before the code is final, since it is being written now:**
- C1: the graph family's positive control has the wrong shape;
- C2: M3's perturbation encoding is unspecified, and that decides both cold-gene and M5;
- C3: cold-gene's uncertainty ignores gene sampling.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | rung 3 for the graph family (Asks 2, 4) | **The injection s · φ(c, j) · ψ(g) is dense and independent of graph distance.** It puts the same gene factor on every landmark.<br>• M3 / M4 can only produce effects that propagate from g's node within K = 3 gated steps, attenuated by Â.<br>• **Outcome-free, on the expdb graph:** for the median knocked-down gene, only **77 %** of landmarks are within 3 hops (13.5 % within 2; 96 % within 4). **6.5 %** of (gene, landmark) pairs are unreachable, and **27 landmarks aren't graph nodes.**<br>• So the graph family can fail this control for **structural** reasons. "Undetectable" for M3 − M4 would then be uninformative about chromatin gating.<br>• It is also the M2 form: φ multiplies a gene factor, which is exactly M2's head. So the control flatters M2 against M3. | **Give the graph family a graph-shaped injection:** e.g. s · φ(c, j) · ψ(g) · [Â³e_g]_j, rescaled to the same variance share, i.e. an effect that propagates from g and is modulated by j's chromatin.<br>**Report both injections for both families,** so each null is read against the shape its family can express.<br>**State** that the dense injection measures sensitivity to an M2-shaped effect. |
| 2 | MAJOR | M3's perturbation encoding; what M5 and R-graph-cold test (Ask 4) | ***"A learned perturbation vector at the knocked-down node"* is ambiguous:**<br>• **If it is a free per-gene parameter,** cold-gene is impossible for M3: a held-out gene's vector is never trained.<br>• **If it is f(gene vector),** the 128-d vectors were pretrained on the **real** expdb graph. So M5, the rewired graph, still receives the real graph's structure through its inputs, and **R-graph-real (M3 − M5) understates** what the network contributes.<br>• **R-graph-cold (M3 − M1) compares** "propagation over STRING" with "a STRING-pretrained embedding in an MLP". Both use the graph, so it isn't "graph against no graph". | **Register the encoding:**<br>• either a **shared** learned start vector placed at g's node, where identity comes from location only (this works for cold genes);<br>• or f(gene vector), with **M5 using vectors retrained on the rewired graph**.<br>**Word R-graph-cold as "propagation against embedding".** |
| 3 | MAJOR | cold-gene's uncertainty (Ask 3) | **The cold-gene rule uses the seed SE, but generalisation to new genes depends on which 20 % of genes were held out.**<br>• Seeds capture training noise only. One draw of genes is fixed, and gene-to-gene variation is the main uncertainty.<br>• **"≥ 5 of 6 cells positive"** is a consistency check, not evidence. All cells share one model and the same genes, so the six per-cell values are strongly correlated. | **The rule:** a **paired gene-cluster bootstrap** CI of Δ. Resample held-out genes; each gene's Δ is its 3-seed mean averaged over core cells. Require the CI to exclude 0.<br>Keep ≥ 5 of 6 cells as an extra condition, or as reported.<br>Report the seed SD beside it. |
| 4 | MINOR | the injection's content (Ask 2) | **φ is ACT:prom_1kb, which correlates with expression.** M1 can partly recover the injected term through expression, so M2 − M1's "detectable level" mixes chromatin with expression.<br>**The range is sensible** (2 / 5 / 10 % of Var(d)). With d's reliability ≈ 0.18, that is ≈ 11–55 % of the true cell-specific variance. | **Inject the part of ACT orthogonal to expression** (within cell, residualised on centred CCLE), since the question is chromatin beyond expression. Report the raw-ACT injection too. |
| 5 | MINOR | rules and the stopping set (Ask 3) | **(a) R-understand compares against a baseline fixed in advance** (k = 3 nearest cells). It may not be the strongest; for d it could lose to ridge.<br>**(b) In LOCO, stopping on 10 % of training genes** removes those genes from training entirely. Their held-out-cell rows are then scored as cold genes, which mixes regimes in the LOCO score. | **(a)** Compare each candidate against the **per-fold maximum over all baselines**, the strongest and conservative, keeping k-NN as reported.<br>**(b)** In LOCO, stop on held-out training **rows** (gene, cell pairs, genes still seen elsewhere), or exclude the stopping genes from the LOCO score. |

## Answers to the asks

**Ask 1 — no outcome leak.**
- **Gene vectors:** pretrained on the expdb graph without L1000 or DTI. They are outcome-free; their graph content matters for
  M5 (C2).
- **CCLE PCs:** fit on all 131 LINCS CCLE cells, the held-out cell included. That is transductive in the feature space only
  (unsupervised, no targets), which is acceptable. State it.
- **Masked chromatin:** fine.
- **Early stopping:** on training genes only, never test rows. Fix the LOCO regime mix (C5b).
- **The held-out cell's L1000:** excluded, as registered, and s_ctl isn't an input.

**Ask 2 — valid for the MLP family, with residualised ACT (C4). Not yet for the graph family** (C1).
- **What a null where the injection is detectable says:** "no chromatin effect of this form and at least this size". Word it
  that way.

**Ask 3:**
- **The LOCO fold rule** (mean Δ ≥ max(0.003, 2 · SD / √6), ≥ 5 of 6) is a t ≥ 2 criterion on 5 df: one-sided p ≈ 0.05 at
  the margin. It is appropriate, because the folds are the unit of generalisation to new cells.
- **Cold-gene:** use a gene bootstrap (C3).
- **Six rules are registered.** Say they are separate questions, read without a family-wise correction.

**Ask 4:**
- **Reach:** K = 3 covers about three quarters of landmarks for the median gene. K = 4 reaches 96 %.
  - Register K = 4 as a sensitivity run, or report M3's gains stratified by graph distance d(g, j) ≤ 2 / 3 / ≥ 4 (outcome-free
    strata).
  - Specify the readout for the 27 landmarks that aren't graph nodes (bias only, or excluded from M3's score and every
    comparator's).
- **The null:** degree-preserving rewiring is the right null for "does the real topology matter", provided M5's inputs don't
  carry the real graph (C2). Report the rewired graph's fraction of edges kept (a degree-preserving swap leaves some).

**Ask 5 — what's missing for "why":**
- **Reliability strata:** report every comparison by the knockdown signature's `tas` tertile (outcome-free, from sig_metrics).
  d is noisy, so gains may sit in the reliable knockdowns.
- **Distance strata** for the graph family (Ask 4).
- **Both injection shapes** (C1).
- **Rung 4's usage checks for M3:**
  - the gate distribution (do gates vary across cells, or saturate?);
  - cell-swap of the gate inputs at inference, as for chromatin.

## What I checked and found sound

- **§103.6 in full:** the settings, inputs, candidates, baselines, rungs 3–5, reporting, compute.
- **Outcome-free, on `string_graph_v11_expdb.npz`:**
  - 19,496 nodes, 345,962 edges;
  - 3,734 of 3,927 core knockdown genes in the graph;
  - 951 of 978 landmarks in the graph;
  - shortest-path distances from knockdown genes to landmarks: the hop fractions and the 6.5 % unreachable;
  - median degree 35 / 38.
- **The input rule against §103.2 C1,** and s_ctl's role.

## What I could not assess, and why

- **The code.** It is being written, and will come in its own packet.
- **The CCLE PC fit and the gene vectors' pretraining details.** I took them as described.
