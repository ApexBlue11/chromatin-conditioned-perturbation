# PACKET 086 — DESIGN: v10-C, the chromatin programme (RESULTS §100; `model/V10_CHROMATIN_PROGRAMME.md`)
packet_id: 086
created: 2026-10-17
type: **DESIGN REVIEW** (no v10-C code or data exists; the L1 counts are identity-level)

**The principal's direction:** make chromatin contribute "whatever it takes", and close the unseen-cell / unseen-compound
gap.

**The design** has six levers:
- L1 the full release (209 training cells, 35 with chromatin);
- L2 coverage (a ChIP-Atlas re-search, plus CCLE RRBS methylation);
- L3 quantitative bigWig windows;
- L4 a separate chromatin stream: drug attention conditioned on chromatin, a chromatin gate on the drug's effect, and
  chromatin-gated STRING; width 384 screened once;
- L5 the plate-contrast target;
- L6 a leave-own-plate-out cell baseline.

**The tests:** T-A (CPU gate) → T-B (dev, 3 seeds, chromatin-off as the comparator) → T-C (unseen cells, per-cell rule, the
gap, XPert under L6).

**Also recorded:**
- **Phase 1 is complete:** C2b in §99.8, read with its seen-cell bound.
- **B1 is deferred**, so its quota goes to v10-C.

## ASKS
1. Is "all levers at once, then one chromatin-off comparison" the right design for the principal's goal? Or must each lever
   get its own comparison, to say *which* one made chromatin work?
2. **L1:**
   - What must be excluded from the full release for each split to stay clean? I propose:
     - all rows of the 8 test cell lines;
     - for the compound split, all rows of the 384 test molecules **and their duplicate IDs** (§96.6);
     - any row whose (cell, compound, dose, time) matches a test row.
   - Anything else?
3. **L6:** is leave-own-plate-out x_cell inductive enough? Given XPert has no such input, how should the head-to-head treat it?
4. **T-A:** what is the right closed-form gate on the contrast target, and its calibration?
5. **The cost** (about 3 Kaggle weeks): any arm to cut first?
