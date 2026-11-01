# PACKET 107 — DESIGN RE-REVIEW: stage 1 as amended by review 106 (RESULTS §103.1)
packet_id: 107
created: 2026-11-01
type: **DESIGN REVIEW** (still before any code or any genetic signature value is read)

**All six challenges of review 106 are adopted in §103.1:**
- **C1, the LOCO leak:**
  - per-fold references from R(g), every knockdown cell with a 96 h signature of g (15 cells), minus the row's cell and the
    held-out cell, with ≥ 3 reference cells;
  - cold-gene and "both" are scored on y, with d reported on the target side only;
  - a required synthetic test: a per-gene model scores |mean r on d| < 0.02 in LOCO.
- **C2, the noise ceiling:** disjoint half-references (sorted, alternating; ≥ 2 cells per half).
  - This makes the ceiling a lower bound.
  - Projection was not chosen, because the residual still carries β·e_ref in both.
  - It covers PC3 and MCF7 only.
- **C3, D1:**
  - r(c, j) = log(s_kd / s_ctl), with s_ctl from the cell's `ctl_vector` signatures at 96 h;
  - computed over the 2,100 genes knocked down in all six cells;
  - double-centred: the mean of the within-cell partial Spearman values;
  - chromatin against s_ctl reported as the technical channel;
  - scope stated: six cells, a landmark bootstrap, descriptive.
- **C4, B1:** a pass of the main rule means a gene-level correspondence only. The cell-specific meaning needs mean(p_mis −
  p_match) > 0 with a drug-cluster CI.
- **C5:**
  - percentile 0 = the most correlated;
  - a landmark target's own coordinate excluded;
  - the highest dose at 24 h, else 6 h;
  - a row-level train / train filter;
  - the top-5 % fraction reported.
- **C6:**
  - the variance decomposition (gene / cell / gene × cell, with a noise upper bound from the PC3 / MCF7 pairs);
  - the cell-offset baseline (cold-gene: training-gene mean d; LOCO: from the held-out cell's `ctl_vector` signatures);
  - a kNN-cell baseline (k = 3);
  - P0 saves per-gene own-gene percentiles;
  - sig_metrics (12.5 MB) asked of the principal;
  - the seed diagnostic unavailable: no hairpin sequences in GEO's metadata.

## ASKS
1. **Is the C1 fix leak-free?** Can any training target or input now carry y(g, c₀)? Is scoring cold-gene on y right?
2. **Are the disjoint half-references a valid lower-bound ceiling?** Is the β·e_ref argument against projection right?
3. **Does log(s_kd / s_ctl) remove the normalisation channel enough?** Is the variance decomposition's noise bound stated
   correctly?
4. **With these fixes, is the harness fit for stage 2's candidates?**
