# PACKET 110 — DESIGN REVIEW: stage 2, candidate cell representations on the knockdown proxy (RESULTS §103.6)
packet_id: 110
created: 2026-11-02
type: **DESIGN REVIEW** (before any stage-2 model is trained). The principal asked to proceed on all three Kaggle accounts,
so the code is being written in parallel, and the runs start after this review.

**Review 109 adopted** (§103.5): ceiling wording, D1's signal-to-noise decomposition, "no detectable cell specificity", and
the stage-2 advice.

**What §103.6 registers:**
- **Settings:**
  - LOCO (6 folds; d with per-fold references; primary r on d);
  - cold-gene (20 % genes; primary r on y in the core cells);
  - early stopping on 10 % of training genes, never on test rows.
- **Inputs:**
  - g: STRING gene vector (128-d) and its node. 3,734 of 3,927 core genes; the other 193 are excluded everywhere.
  - c: CCLE landmarks + 50 PCs, chromatin per landmark (16, masked), global chromatin (42).
  - No L1000-derived input from the held-out cell.
- **The candidates:**
  - M1: expression MLP;
  - M2: + a perturbation-conditioned chromatin term per landmark;
  - M3: recurrent gated propagation on the full STRING graph (K = 3, gates from each node's expression and chromatin);
  - M4: M3 with expression-only gates;
  - M5: M3 on a degree-preserving rewired graph.
  - 3 seeds each.
- **Rung 3 first:**
  - injection d′ = d + s · φ(ACT:prom_1kb) · ψ(g) at 2 / 5 / 10 % of Var(d), for M1 / M2;
  - at 5 % for M3 / M4;
  - the detectable level is set before any real comparison is read.
- **Rung 5:**
  - R-chrom-MLP (M2 − M1), R-chrom-graph (M3 − M4) and R-understand on LOCO d (fold-level rule, ≥ 5 of 6);
  - R-graph-cold (M3 − M1), R-graph-real (M3 − M5) and R-understand-cold on cold-gene y.
- **Reporting:** against all baselines, with 6-cell CIs, as a fraction of the ceiling; s_ctl as a covariate wherever chromatin
  is read.
- **The compute:** ≈ 25–30 GPU-h over three accounts.

## ASKS
1. **Is any input a leak?**
   - the gene vectors (pretrained on the STRING graph without L1000 or DTI);
   - CCLE PCs fit on the 131 LINCS cells;
   - masked chromatin;
   - early stopping on training genes.
2. **Is the injection design a valid positive control** for both families? Is 2 / 5 / 10 % of Var(d) the right range, given
   VD's true gene × cell variance of ≥ 0.055 / 0.380 ≈ 14 % of the interaction?
3. **The rules:**
   - a fold-level SD for LOCO with 6 folds;
   - seed SE for cold-gene;
   - is ≥ 5 of 6 per-cell positivity right for cold-gene, where all cells are in one model?
4. **M3:**
   - is K = 3 enough to reach the landmarks from most knocked-down genes? (Path lengths on the graph can be reported,
     outcome-free.)
   - is the degree-preserving rewiring the right null?
5. **What is missing** before stage-2 results can say why something works or doesn't?
