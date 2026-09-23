# PACKET 008 — ADDENDUM: ask 4 is answered, from the data
packet_id: 008a
created: 2026-09-23
refers_to: 008_masked_result_and_cc1_kernel.md, ask 4

Sent so that no review effort goes into ask 4. Measured after packet 008 went out.

v9's cold-cell predictions (`external/v9_mdmt_preds/v9_cc1_epi_seed0.npz`) omit 170 of the 21,321 test rows of
`split_cold_cell_1`. `model/v9/xpert_arm.py:94-110` drops, counts and reports every row whose compound is not in
our drug feature index (`drug/outputs/drug_feature_index.json`, 21,220 compounds), and never imputes them.

Checked against that index:

| | |
|---|---|
| missing rows whose compound **is** featurised | **0 of 170** |
| kept rows whose compound is **not** featurised | **0 of 21,151** |
| distinct compounds among the missing rows | 28 |

So the 170 are **exactly** the rows with an unfeaturisable compound, and nothing else. The exclusion is
compound-level and deterministic, not cell-level and not random. Pairing on the intersection is therefore
"the rows both models can score", which is the rule `xpert_arm.py` already states.

What this does not settle: the intersection is selected by compound, so it is not a random 99.2 % of the fold.
By cell it removes MCF7 154, BJAB 11, THP1 5.
