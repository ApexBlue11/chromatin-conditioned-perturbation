# PACKET 098 — DESIGN: v11 (RESULTS §102), for review before any v11 model code
packet_id: 098
created: 2026-10-21
type: **DESIGN REVIEW**

§102 is committed with this packet. It carries forward reviews 093, 095 and 097 and the §101.6 / §101.8 findings:
- CCLE is the stronger cell input;
- chromatin is gated across all 19,496 genes, but its prior is low;
- ChEMBL targets enter through the full experimental + database graph;
- the DTI probe is protected by a compound split.

## ASKS
1. **The DTI probe split:**
   - Is 30 % of annotated non-test compounds by molecule group sound?
   - Is excluding test compounds from the probe set right, given that the "known targets" setting must keep test compounds'
     annotations?
2. **The target encoding:**
   - Is two channels (any target; action sign) adequate?
   - Are predicted probabilities in place of missing annotations acceptable?
3. **The arms:**
   - Are 4 arms × 3 seeds the minimum for R1 + R2?
   - Is T-B's off arm (v10-C off) an acceptable reference, or should a pure-v10 reference be run?
4. **Init no-op and memory:**
   - Does the design start exactly as v10, given that every new term is zero-initialised?
   - Is per-cell shared propagation with 6-wide per-row signals the right answer to the 23 GB concern?
5. **Pathways:** should v11 rebuild `M_pathway` from GAF-filtered GO, or keep v9's for comparability with T-B's reference?
6. **Anything that should be registered before code:** power, the guard, epochs, the order of runs.
