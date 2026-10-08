# PACKET 058 — RESULT: §93.15b read once (RESULTS 93.17)
packet_id: 058
created: 2026-10-08
repo_commit: 87ba049
type: **RESULT** (no GPU; one free Kaggle CPU kernel)

## A. What ran
- **The kernel:** `lincs-chromatin93-gg` v1, 2,879 s. `inputs verified: 8 pinned files + the split bundle`, with
  `chromatin_genegeneric.py` `d00ebeb1`, the version cleared by review 057b.
- **The read:** once, by `chromatin_genegeneric.py --read external/kaggle_out/chromatin93_gg`, after the marker check →
  `model/results/chromatin93_gg_reading.json`. The raw output is `model/results/c93/chromatin_genegeneric_93.json` (`94d13c1c`).

## B. The numbers
- **Harness:** N1_v9 − FB = +0.0012098990, which equals 91.12.

| quantity | tie (of record) | v9 (reported) |
|---|---|---|
| G_chr | +0.00139 | +0.00121 |
| G_perm, min to max over 20 | −0.00089 to +0.00059 | −0.00030 to +0.00046 |
| G_E | +0.00055 | +0.00074 |
| Δ_chr\|E | +0.00154 | +0.00185 |
| Δ_perm\|E, min to max over 20 | −0.00052 to +0.00095 | −0.00022 to +0.00047 |
| cells with Δ_chr\|E > 0 | 4 of 6 | 4 of 6 |
| reading | CHROMATIN-SPECIFIC | CHROMATIN-SPECIFIC |

- **Per cell, Δ_chr|E (tie):** HEK293T −0.00003, HL60 +0.0006, LNCAP +0.0026, SKBR3 +0.0019, U937 −0.0007, VCAP +0.0001.
- **Per cell, G_chr (tie):** HEK293T +0.0001, HL60 −0.0005, LNCAP +0.0041, SKBR3 +0.0013, U937 −0.0004, VCAP −0.0002.

## C. What was controlled
93.15, 93.15a and 93.15b exactly, as cleared:
- one `run_t1` call with 87 specs;
- a shared permutation per draw;
- E matched to its own feature set;
- §91's rows, cells and LOCO.

## ASKS
1. Is the reading mechanical?
2. Is 93.17's sentence licensed, with its descriptive caveats?
   - The caveats: small (below T1's bar); mostly LNCAP; the fourth positive cell is VCAP at +0.0001; max-of-20 one-sided; one
     choice of expression reference; the per-gene-parameter caveat.
   - Is "the artefact did not create it" licensed?
   - Is the "property of genes, not of cells" interpretation licensed?
3. For the paper's §5.3: what is the shortest licensed form?
