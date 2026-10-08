# REVIEW OF PACKET 063
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 7000c59 (registration 40ba31b; packet 527cc99)

**The split refactor is correct, and cold-cell behaviour is unchanged.**
- **Code and tests:** `mechanism_stage_a.py` = 2bf65909. I reran `test_mechanism_stage_a.py` under `.venv-cuda`: **25 passed**.
- **The gates** come only from the class rows' `targets_fn`, reading the `ACTIVE` dict at call time. The tables' `cells` fields
  are unused.
- **The split plumbing:** `scored_cells`, `load_rows` (13,445-row assertion), the A1 threshold and the result's `split` field
  follow `--split`. The default is §94's.

**But the cold-drug test compounds are a thin substrate for these readouts,** and the registration doesn't yet say what Stage A′
can establish on them, or what outcome justifies P9's GPU run (C1, MAJOR). Both must be fixed before the run, since they're
decision rules. C2–C3 are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | substrate / decision rule | **What the cold-drug test compounds are,** from identity-level counts on the 13,445 rows (no response read):<br>**MoA classes with ≥ 3 labelled parents per scored cell** are mostly CNS, ion-channel and anti-inflammatory ligands. Per cell (MCF7 / PC3 / A375 / HA1E / HT29 / A549): D2 antagonist 5 / 5 / 5 / 4 / 5 / 5; 5-HT2a antagonist 6 in each; Kir6.2 (sulfonylurea) 3–4; COX inhibitor 3–6; Na-channel blocker 3 (1 in A549). Plus **"DNA inhibitor" 4 in each cell**. Kinase classes have ≤ 3 per cell.<br>**A3's gated table gives only 6 units, all EGFR,** with 3–4 members each, drawn from **4 distinct compounds** across all six cells. There are **0 MDM2 inhibitors** and no MAPK, PI3K, JAK, ER, NFkB or Hypoxia unit.<br>**Consequences:**<br>(a) **A3 here is one class and 4 compounds, replicated across cells.** The 2/3 rule over 6 non-independent units isn't a mechanism-general readout.<br>(b) **A1's mates are mostly CNS-class pairs.** CMap shows antipsychotic-like compounds give shared, reproducible signatures largely through cationic-amphiphilic physicochemistry (the lysosomal / cholesterol-synthesis response), not their annotated receptor. An A1 pass could be carried by physicochemistry; a null could mean "mostly inert compounds".<br>(c) **P9 (≈ 6 GPU-h) is gated on "a readout with signal".** As written, 4 EGFR inhibitors or a CNS-physicochemistry retrieval could open it, though neither supports *"mechanism from chemistry"*. | **Fix in §95.2, before the run:**<br>(i) **A3 is decision-bearing on this split only if its units span ≥ 2 classes.** Otherwise it's reported only. Optionally, add one pre-specified class with a defensible sign: *DNA inhibitors → PROGENy p53 (+) in TP53-wild-type cells* (MCF7, A549, A375; C2).<br>(ii) **Record now** that an A1 pass carried by CNS classes may reflect shared physicochemistry. Report per-class AUROCs beside it (C3 of review 060 applies: member-mean AUROC), and require Stage B′ to include a physicochemical-descriptor reference (Ask 2).<br>(iii) **State P9's mechanism gate explicitly.** For example: A′ signal in a readout that isn't carried by a single class (A1 with member-mean AUROC above null in ≥ 2 non-CNS classes, or A3 per (i)). Otherwise: *"the cold-drug split's held-out compounds cannot test mechanism-from-chemistry at these readouts"*, and P9 is justified only on accuracy grounds, registered as such.<br>(iv) **Put the identity-level class composition above in §95.2 as context.** |
| 2 | MINOR | gating | **A375 is TP53-wild-type, so it belongs in the MDM2 gate** with MCF7 and A549. HT29 is R273H, PC3 is TP53-null, and HA1E's SV40 large T inactivates p53, so those exclusions are right. This is moot on these data (0 MDM2 inhibitors in every scored cell), but the table should be correct. | `'mdm2': {'MCF7', 'A549', 'A375'}` for `split_cold_drug_1`. Re-pin. |
| 3 | MINOR | code (future) | **`read_stage_b` hard-codes B1's cell threshold at ≥ 3** (`… >= 3`), so it isn't split-aware. For Stage B′ on cold-drug, the registered threshold is 4 of 6. | Use `SPLITS[measured['split']]['a1_min']` in `read_stage_b`, with a test, before Stage B′. |

## Answers to the asks

**Ask 1 — the scored cells (≥ 25 by identity) and A1 ≥ 4 of 6 are right.**
- **Gates:** RAF in HT29 and A375 (BRAF V600E) is right. A549's KRAS G12S is exactly why RAF stays out of A549, and MEK/ERK
  applies everywhere. ER in MCF7 only, and AR dropped, are right.
- **The gating gaps:** A375 for MDM2 (C2). PC3 is PTEN-null, which would *strengthen* PI3K-inhibitor effects; no gate is needed.
  HA1E is SV40-LT/hTERT-immortalised and RAS-wild-type, so MEK/ERK, EGFR and PI3K apply there.
- **The larger issue is substrate (C1),** not gating.

**Ask 2 — ridge plus a Tanimoto nearest neighbour is the right core.** It needs these additions, each registered with Stage B′:
1. **A structure-only retrieval floor for B1:** the AUROC of Tanimoto similarity itself, mates against non-mates, with no
   responses. If chemistry alone retrieves mates at, say, 0.65, v9 must beat that.
2. **k-NN** (k = 5, similarity-weighted) beside 1-NN, which is noisy.
3. **A physicochemical reference** (descriptors only: cLogP, basic pKa / charge, MW, TPSA), because of C1(b)'s
   cationic-amphiphilic route. Ridge's descriptor block partly covers this; state which descriptors.
4. **The NN rule:** the nearest **training** compound profiled in the same cell, with a stated fallback if none, and per-compound
   max Tanimoto to training reported. Stratify, or restrict the claim, by it (for example, max similarity < 0.5).
5. **Paired swap nulls for v9 − each reference,** registered as part of the reading. That's the lesson of 062.

**Ask 3 — the refactor has no defect for cold-cell.** Stage A′ is **not cleared until C1(i)–(iii) are recorded.** They're
decision rules, so they must precede the data. C2 should be in too. C3 is needed before Stage B′, not before A′.

## What I checked and found sound

- **The code:** `mechanism_stage_a.py` 2bf65909 against bbefd48 (`SPLITS`, `ACTIVE`, the gate lambdas, `load_rows`,
  `scored_cells`, `read_stage_a`'s threshold, `run_pipeline`, `--split`). The 25 tests, rerun.
- **The registration:** §95.1–95.4.
- **Identity level, on the cold-drug test rows** (13,445; no responses):
  - labelled parents per scored cell (81–105);
  - MoA classes with ≥ 3 parents per cell;
  - MDM2 inhibitors per cell (0);
  - A3 units under the gated table (6, all EGFR), and distinct EGFR compounds (4).

## What I could not assess, and why

- **Whether the CNS-class signatures in these cells are physicochemical or receptor-driven.** That needs responses, which are
  Stage A′'s to read. C1(ii) makes the interpretation explicit before the run.
