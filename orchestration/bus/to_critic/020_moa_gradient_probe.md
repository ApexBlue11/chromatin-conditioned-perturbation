# PACKET 020 — PRE-REGISTRATION: does trained v9 carry DRUG-SPECIFIC pathway mechanism? (IDEAS A11 step 1, 0 GPU-h)
packet_id: 020
created: 2026-09-25
repo_commit: (the commit carrying this packet)
type: **PRE-REGISTRATION.** No code for v9 exists yet; nothing has been computed on a trained model.

## Why
RESULTS 85.1 (your 019 C1): v9's surviving interpretability readout is cell-level; the pathway activations are
computed before the drug enters. So no drug-specific mechanistic readout survives. CLAIMS 4.14 recorded, for v6, that
the drug still reaches each named pathway node through the GRADIENT (`∂Ŷ/∂a_p` moved by 0.076 under another drug), and
`model/v6/probe_moa_v6.py` was written to test exactly this — but it was only ever run on an UNTRAINED model, as the
control that showed 0.5 is not chance (median target-pathway rank percentile 0.218 against a label-permutation null of
0.229; CLAIMS 4.15). It has never been run on a trained checkpoint. This ports it to v9 and runs it once.

## Design (the v6 probe's, with v9 specifics fixed here)
- **Checkpoints:** `r0/r1/r2_ckpt_v9_fold0_seed{0,1,2}.pt` — fold 0, seeds 0–2, drug self-attention off, `use_aux` on,
  800 named nodes, batch 96, 12 epochs. `sa0` (self-attention on) is reported, not read.
- **Importance, per row:** `imp[p] = Σ_c a[p,c] · ∂O/∂a[p,c]`, with `O = ‖Ŷ_delta‖²` (the probe's `sq` objective: which
  pathways drive the SIZE of the predicted response; signs cannot cancel). **Drug-specific engagement:**
  `Δimp[p] = imp_d[p] − imp_meandrug[p]`, the same row with the drug inputs (global features and atom tokens) replaced
  by their mean over the scored compounds. Per compound: median of `Δimp` over up to 4 of its rows.
- **Rows:** the rows of v9's three fold-0 TEST splits with `strength ≥ eval_min_strength`, ≤ 4 per compound, compounds
  in a fixed order; the same rows for every checkpoint.
- **Independent annotation (never in any loss):** ChEMBL mechanism targets, `drug/outputs/dti/chembl_dti_edges.tsv`,
  human single-protein, `direct_interaction == 1`. **Positive set of a compound** = the named nodes whose **full curated
  gene set** — `network/data/ReactomePathways.gmt` / `GO_Biological_Process_2023.gmt`, matched by term id to
  `pathway_info_v9.tsv` — contains ≥ 1 of its targets. (Landmark-only membership, which covers 314 of 1,363 annotated
  compounds, is the secondary tier.) Compounds with an empty positive set are excluded, with the count reported.

### Gate G, computed and reported FIRST
Median over pairs of distinct scored compounds of Spearman(`imp_d`, `imp_d'`), the raw importance vectors. On an
untrained v6 this was +1.0000: the drug entered as a uniform scale, not a reordering. **If G ≥ 0.98 on a checkpoint, that
checkpoint's readout cannot express mechanism and its alignment numbers are not read.**

### Statistic and nulls, per checkpoint
`S` = median over compounds of the median rank percentile (0 = top) of its positive nodes among the 800, ranked by
`|Δimp|`. **Null 1, label permutation (the primary reference, per 4.15):** each compound gets another compound's positive
set, 1,000 permutations → `diff = S − mean(null1)` and one-sided p. **Null 2, size-matched:** each positive node swapped
for a node of similar full-gene-set size (200 draws). Never read against 0.5.

### Readings, pre-committed
| result | reading |
|---|---|
| G < 0.98 **and**, on **all three** of r0–r2: `diff ≤ −0.02`, p < 0.05, and `S` below null 2's mean | **SIGNAL:** v9 carries drug-specific pathway mechanism, readable by gradient × activation |
| G < 0.98 and the above on 2 of 3 seeds, **or** on all 3 with −0.02 < diff < 0 | **PARTIAL:** report; no mechanistic claim in the abstract |
| anything else (including G ≥ 0.98) | **NULL:** reported as a null; C8b (a trained post-perturbation named readout) enters §85's candidates |

**Reported, not read:** the landmark-only tier; `sa0`; the split into compounds seen vs unseen in training; the
"target is itself responsive" stratification (CLAIMS 4.1b); the cross-drug Spearman of `Δimp`.
**Not permitted in any outcome:** per-drug case-study figures (CLAIMS 4.9–4.11) unless separately pre-registered.

## Cost
0 GPU-hours: forward and backward on ≲ 10k rows per checkpoint, on this laptop (small) or CPU.

## ASKS
1. Is `Δimp` against the mean drug, with the `sq` objective, the right estimand for "drug-specific mechanism"? Is anything
   in it still structurally drug-invariant (e.g. the mean-drug baseline sharing the cell's activations exactly)?
2. Is the full-gene-set positive set legitimate, given the node aggregates only its landmark members?
3. Are the gate threshold (0.98), the effect floor (−0.02) and the all-3-seeds rule right?
4. Does `use_aux` (the pathway layer is supervised on the cell's per-pathway `mean|Δ|`) contaminate this readout?
5. Anything else.
