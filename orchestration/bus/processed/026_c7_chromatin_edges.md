# PACKET 026 — PRE-REGISTRATION: C7, chromatin gating the edges of the union graph (IDEAS A1's trained arm); C5 deferred
packet_id: 026
created: 2026-09-25
repo_commit: ae78f26
type: **PRE-REGISTRATION.** No C7 code exists. Kaggle's weekly GPU quota is exhausted, so nothing can run before the reset.

## Coverage first (review 019 ask 5), `model/results/cc1_input_coverage.json`
| group (split_cold_cell_1) | cells | CCLE direct match: cells / rows | any chromatin track: cells | ATAC / H3K27ac / H3K27me3 rows |
|---|---|---|---|---|
| train (26) | 26 | 17 / 86.7 % | 12 | 56 % / 71 % / 55 % |
| **dev (6)** | 6 | 4 / 80.4 % | **6** | 76 % / 92 % / 92 % |
| test (8) | 8 | 5 / 98.0 % | 5 | 94 % / 94 % / 63 % |

**C5 deferred, with its reason.** The local CCLE baseline (`X_base_lincs`, 83 × 978) is **landmark genes only**, and a
"DMSO fallback" cell's value is its own L1000 control mean — which v9 already receives as `x_cell`. So C5 would add a
second assay of genes the control profile already covers. Genome-wide CCLE would need a DepMap download and a
compression step; that is a separate design with its own pre-registration if ever pursued. Not run now.

## Background for C7
Two zero-parameter pre-tests were null (82.7 UNINFORMATIVE, 83.5 NO_SIGNAL: fixed diffusion from targets does not
predict which landmarks respond). A1's hypothesis — chromatin decides which edges conduct in a cell — can only be put
by a trained model. Your review 016 ask 4: it needs a mismatched-chromatin null key, because a trained gate can use
chromatin as a cell barcode; the readout should be unseen cells. The dev cells are unseen in training, so a barcode
cannot help there.

## C7, defined
- **Graph:** the binary union of STRING, Reactome and GO:BP on the 978 landmarks, `network/outputs/v9/union_graph_v9.npz`
  (81,846 edges, §82), symmetric, no self-loops.
- **Gate:** per cell, `a_i = σ(MLP_3→8→1(E_i))` from gene i's three chromatin values (the MLP is learned, so the model
  chooses which tracks matter), with `a_i = 1` for tracks-missing genes/cells (the mask). Edge gate `g_ij = √(a_i a_j)`.
- **Layer:** one gated message-passing step on the gene tokens, `h ← h + sd(W · Σ_j Â_ij g_ij h_j)`, `Â` = the
  symmetric-normalised union adjacency, stochastic depth at `cfg.stoch_depth`, placed **where the STRING
  message-passing step sits now, replacing it** (`use_ppi` off for C7). One flag, `--chromatin_edges`.
- **Acceptance tests (code):** a cell with all tracks missing gets exactly the ungated union layer; changing a gene's
  chromatin changes the gate only on that gene's edges; with the flag off, bitwise-identical to today.

## Readings, pre-committed
1. **Accuracy:** §85.2 rules 6–8 against P2, like every candidate.
2. **Attribution, only if C7 is accepted:** on the dev rows, the same C7 checkpoints scored with each dev cell's own
   chromatin vs **mismatched chromatin** (each dev cell given another dev cell's tracks, all 5 alternatives, averaged).
   `Δ_own = r(own) − r(mismatched)`, per row, paired; per-cell medians; reading: **"chromatin gating is used
   cell-specifically"** iff the 3-seed mean Δ_own > 0 with a row-bootstrap CI excluding 0 **and** ≥ 4 of 6 dev cells
   positive. Otherwise the accuracy gain, if any, is recorded as **not attributable to the cell's own chromatin**.
3. **The ungated control C7u** (the same layer with `a_i ≡ 1`), run only if C7 is accepted, separates "the union graph
   helps" from "chromatin gating helps": the attribution sentence also requires C7 − C7u > 0 at 3 seeds by §85.2 rule 7's
   floor.

## ASKS
1. Is replacing the STRING step (rather than adding a layer) the right single change?
2. Is the mismatched-chromatin attribution (all 5 alternatives, dev rows, 3 seeds) the right null key, and is C7u needed?
3. Is deferring C5 on these grounds sound?
4. Anything else.
