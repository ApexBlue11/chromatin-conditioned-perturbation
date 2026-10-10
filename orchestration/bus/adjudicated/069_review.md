# REVIEW OF PACKET 069 (Stage B′ code, 96.10 identity-level outputs, and the 96.11 addition)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: cd7d068 (code 8a89a89; 96.10 0942769; 96.11 cd7d068)

**The code implements 96.8 / 96.9, and both of its outputs reproduce from my own independent code.**
- **The chemistry file:** I re-featurised all 1,977 compounds myself, using RDKit's generator API rather than the module's
  call. Fingerprints, descriptors, molecule keys and parsability match `chem_cold_drug_1.npz` with **0 mismatches**.
- **The references:** I rebuilt 1-NN, 5-NN and physchem for all 13,364 rows with my own code. The maximum absolute difference
  from `refs_cold_drug_1.npz` is **≤ 7.7e-7**, which is float32 storage. The counts match too: 1,776 (compound, cell) pairs,
  1 fallback, 1,445 molecules, 63 multi-`pert_id` molecules, and max Tanimoto 389 / 389 against the sidecar.
- **The tests:** 14 + 5 pass on rerun.
- **96.11's table** matches its JSONs, and I recomputed every mean d_std in it independently.

**The caveat is about interpretation, not code (C1).** 96.11 states that d_std still rises with signal-to-noise. That is right,
and it means 96.8 item 1's standardisation removes the **scale** of the t-values but **not** their **reliability attenuation**.
So the v9-against-R comparison keeps a smoothness term wherever the two differ in noise. That term is largest against 1-NN.

**A concession of my own.** Review 066 said that per-model standardisation leaves "the direction structure". That was
incomplete for the same reason.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | interpretation, pre-output | **What 96.11's SNR claim implies for item 2.** Z-scoring within the cell removes the common scale of a model's t-values. It doesn't remove the attenuation of the member-vs-others contrast by per-compound estimation noise.<br>• **Toy check** (n 100, m 4, ULM-style regression t, per-gene noise σ from 0 to 8): d_std falls 1.90, 1.88, 1.86, 1.86, 1.80, 1.55.<br>• **The package's own test** (`test_standardisation_removes_smoothness_inflation`) asserts only a reduction to < 25 % of the raw inflation, not removal.<br>• **So T_v9 − T_R still carries a smoothness term.** It is largest against **1-NN**, whose signature is one training neighbour's **measured** mean response, not a denoised prediction. It is smaller against the averaging references (5-NN, physchem 5-NN) and smallest against ridge.<br>• **The risk:** a v9 pass against 1-NN alone could come from v9 being denoised rather than from anything beyond chemistry. The licensed sentence ("more strongly than chemistry-only references (those R passed)") would still be literally true. | Before any P9 output:<br>• correct 96.8 item 1 so it says the standardisation **reduces** the smoothness term (removing the scale only), consistent with 96.11;<br>• fix now how a 1-NN pass is read. Either the licensed sentence needs at least one averaging reference (5-NN, physchem or ridge), or a 1-NN pass carries a beside-text saying 1-NN is not denoised and the pass may reflect v9's smoothness. Your choice. |
| 2 | MINOR | wording (96.11) | **Three statements in 96.11 go beyond the numbers.**<br>(a) **"near the ceiling the standardisation allows" is false.** The largest attainable d_std for a unit is √(n(n−1) / (m(n−m))). Here that is **4.6–5.8** (n 81–105 labelled compounds, m 3–4 members). Ridge's EGFR units at 1.8–2.2 are about 35–40 % of it, so "below saturation" adds nothing.<br>(b) **"reproduces it at least as strongly as the measured data does"** holds on the 9-unit set. But that set includes doxorubicin and afatinib, whose 1-NN and 5-NN references are training measurements of the **same molecule** (Tanimoto 1.0). On the excluding set of record (4 units, same standardisation), the mean d_std is: **measured 1.08, 1-NN 0.92, 5-NN 1.15, physchem 0.91, ridge 1.30**. Only 5-NN and ridge match or beat measured there; 1-NN's DNA@A375 is −0.08 and physchem's EGFR@MCF7 is −0.56.<br>(c) **"Why the references exceed the measured data" isn't uniform.** Physchem is below measured even on 9 units (0.88 against 0.95). And 1-NN's signatures are measurements, so "model predictions are smoother" doesn't cover it. | Drop the ceiling clause. Report the excluding-set means beside the 9-unit table. Limit the SNR sentence to the averaging and fitted references (5-NN, ridge), say it is consistent with SNR but untested, and say the 9-unit excess partly reflects the two twin members. The conclusion for Stage B′ (item 1 is uninformative; item 2 carries the claim) stands as written. |
| 3 | MINOR | guards | **The reader can be given the wrong inputs without refusing.**<br>(a) **Item 1:** the provenance guard accepts any `delta_source` other than `measured`. The four reference B3 JSONs now sit in the same directory with valid markers and `split_cold_drug_1`, so `b3_ridge.json` would pass, and **all four references pass B3**. Nothing ties `--b3` to P9's seed files, or checks that the four are distinct and in seed0, seed1, seed2, seed-mean order.<br>(b) **Item 2:** any number of comparison JSONs is accepted. Nothing requires exactly the four registered references, or checks that each comparison's v9 inputs equal the B3 sources. Given a subset, the reader writes the licensed sentence over fewer references.<br>(c) **`compare`:** it records `ref_spec` as a string, not the references file's sha1, and doesn't check that file against its marker. The 157 MB npz isn't in git, so its sha1 is the only link to the reviewed `c27e4055…`. It also doesn't check that spec 4 is the mean of specs 1–3. | **Reader:** require 4 B3 JSONs whose `delta_source` paths are the P9 seed files, with the seed-mean as the three joined, all distinct. Require exactly {ridge, 1-NN, 5-NN, physchem} comparison JSONs, with v9 input sha1s matching the B3 inputs.<br>**`compare`:** verify the references file's marker, record its sha1, and assert spec 4 equals specs 1–3 joined. |
| 4 | MINOR | record text (96.10) | (a) **Physchem standardisation** uses the **training molecules'** mean and population sd (1,445, ddof 0). 96.8 item 4 says "training compounds' mean and sd". The choice is consistent with 96.9's collapse and with the packet's own description. Compared with compound-level, ddof-1 statistics, the means differ by ≤ 2.5 % (basic amines; the others ≤ 1.5 %) and the sds by ≤ 1.8 %, though every physchem row shifts slightly.<br>(b) **The 2-of-389 cause is a tautomer.** The two sildenafil `pert_id`s are the 4H and 6H lactam tautomers (`…[nH]c(nc2=O)…` against `…nc([nH]c2=O)…`). Standard InChI's mobile-H layer merges them; ECFP4 doesn't. The registered representative, `BRD-K50128260`, is the less common 4H drawing. | Record (a) in 96.10. Replace "its two `pert_id`s have different fingerprints" with the tautomer explanation (b). |

## Answers to the asks

**Ask 1 — yes, a faithful implementation.**
- **`elements`** mirrors `run_a3` and is guarded at runtime against it (≤ 1e-9 on d and d_std for every unit, plus the unit-key
  set). Activities are standardised **once** per model and cell, over that cell's labelled compounds (ddof 1).
- **`run_swap_test` is the registered swap:**
  - **The elements** are the labelled (compound, cell) pairs of the reading's cells, less the excluded parents.
  - **g_i** accumulates a_ui over every unit that shares an element. In MCF7 that's both the DNA and EGFR units, so one sign flip
    exchanges the element's whole (v9, R) pair, as 96.9 item 9 requires.
  - **Δ_obs** = Σ g_i = T_v9 − T_R. The draws come from `default_rng(9470)` per call, so every R and every variant uses the
    same signs.
  - **p** = (1 + #{Δ* ≥ Δ_obs}) / 10,001.
  - **The brute-force test** covers two units that share elements (50 masks, 1e-12).
- **The exclusion is right:**
  - the excluded parents are removed from members **and** others;
  - z is not recomputed (the standardisation stays over all labelled compounds);
  - units left with fewer than 3 members drop out;
  - `compare` refuses unless the surviving set is exactly {DNA@MCF7, DNA@A375, DNA@A549, EGFR@MCF7} for every model;
  - the pairing guard enforces identical labelled units and identical eval units for v9 and R.
- **The seed-mean** is `load_predicted_delta`'s mean over the comma-joined files. That's the prediction-averaged ensemble of 96.9
  item 4.

**Ask 2 — acceptable as the registered rule acting, and not worth an amendment.**
- **The cause** is a tautomer (C4b), not a different structure. It's the **only** one of the 63 multi-`pert_id` training
  molecules whose parsable forms have different fingerprints.
- **The effect is immaterial:**
  - **Of the two affected test compounds,** `BRD-K16542329` is unlabelled and profiled only in HELA, so it's outside B′.
  - **The other, `BRD-K13926615` (CHEMBL1520, vardenafil),** is labelled and is an "other" element in MCF7, A375 and A549.
    Sildenafil is its nearest neighbour under either tautomer (0.531 against 0.590). Only its weight in vardenafil's 5-NN
    changes, and vardenafil enters d with weight 1/|O| ≈ 1/100.
- **The alternatives are worse here:**
  - taking the maximum over a molecule's `pert_id` fingerprints would change nothing material;
  - tautomer canonicalisation would change every fingerprint relative to the 96.6 audit.

**Ask 3 — two things before the comparison runs on P9's predictions:**
- the C1 decision on how a 1-NN pass is read, which must be fixed before any P9 output;
- the C3 guards.

Nothing else is missing:
- 96.8 item 5's reporting (fallback counts, members' 1-NN, per-class direction, raw labelled) is in place;
- the excluding reading is the one of record;
- the rule "every variant passes, for that R" is implemented.

**96.11 (the PI's addition):**
- **The numbers are right.** I recomputed mean d_std independently: 1-NN 1.084, 5-NN 1.291, physchem 0.879, ridge 1.840,
  measured 0.946. The per-unit DNA values and the 6-cell EGFR means match the table.
- **Measured on 13,364 rows** equals the 13,445-row version to two decimals (9 units 0.946; excluding set 1.079 against 1.081),
  so the row-set difference in the table doesn't matter.
- **"Item 1 alone is uninformative; item 2 carries the claim" is licensed.**
- **The SNR framing is licensed only in part:**
  - **Licensed:** that d_std rises with SNR. It's real, and it's the reason v9 isn't compared with the measured data.
  - **Not licensed:** "near the ceiling", the uniform "references exceed measured", and smoothness as the explanation for 1-NN
    (C2).
  - **And it has a consequence 96.11 doesn't draw** for 96.8 item 1 and for reading a 1-NN pass (C1).

## What I checked and found sound

- **The code:**
  - `stage_b_prime_chem.py` in full;
  - `stage_b_prime.py` in full (`build_references`, `elements`, `run_swap_test`, `check_pairing`, `compare`, `read_b_prime`);
  - `mechanism_stage_a.load_rows` / `load_predicted_delta` (spec parsing, seed-mean, the `-ctl_true` form) and `read_stage_a`'s
    cold-drug A3 rule, against which the reader's item 1 matches.
- **The tests:** both suites rerun (14 + 5 pass); the sign-flip brute-force test and the smoothness test read.
- **The inputs:** of the 1,977 compounds, 884 are in both `pert_info` tables, with **0** SMILES or InChIKey-block conflicts, so
  the merge order doesn't matter.
- **Independent rebuilds:**
  - the chemistry features;
  - all three references;
  - the max-Tanimoto list;
  - excluding-set d_std per reference;
  - per-unit d_std ceilings;
  - the sildenafil impact.

## What I could not assess, and why

- **The comparison itself.** P9's predictions don't exist yet, and no swap test has run on v9.
- **The B3 p-values in 96.11.** 0.000999 is 1/1001, the floor for 1,000 permutations. I didn't rerun the permutations; T and
  d_std I did recompute.
- **How large the smoothness residual (C1) is in the real data.** The toy shows its direction, not its size for L1000 noise,
  which is correlated across genes.
