# Drug Branch (Phase 3) — Perturbation Featurization

Produces the model's **drug input**, keyed by **`pert_id`** (joined per signature at train time; ~21,220
usable drugs after the no-SMILES exclusion). Plan: SMILES + physicochemical descriptors + **Uni-Mol** (3D)
+ **MolBERTa** (structure LM) + biological/MoA grounding. Governed by the interpretability objective.

## Done
- **`step1_extract_drug_smiles.py`** → `outputs/drug_list.tsv`: **21,220 unique usable drugs**
  (pert_id, iname, #sigs, canonical_smiles). All have SMILES (the 79 no-SMILES drugs — incl. the 2
  proprietary, SMILES withheld — were already dropped upstream; 310,114 usable sigs).
- **`step2_compute_descriptors.py`** (`.venv-drug` = uv Python 3.12 + RDKit) →
  - `outputs/drug_descriptors.npy` (21,220 × **20** interpretable physchem descriptors: MW, LogP, TPSA,
    HBD/HBA, rotatable bonds, aromatic/aliphatic rings, FractionCSP3, QED, BertzCT complexity, …; names
    in `drug_descriptor_names.json`).
  - `outputs/drug_fingerprints.npy` (21,220 × **2048** ECFP4/Morgan bits, mean 58 set).
  - `outputs/drug_feature_index.json` (**pert_id → row**). 0 SMILES parse failures.
- **`step3_integrate_kaggle_embeddings.py`** — deep embeddings computed on **Kaggle GPU (T4×2)**, not
  locally (Python 3.14/Windows lacks torch/UniMol wheels; local CPU too slow). Kernel
  `apexblue/lincs-drug-embeddings-unimol-molberta` reads the SMILES dataset `apexblue/lincs-drug-smiles`,
  shards across both T4s (one worker/GPU, `CUDA_VISIBLE_DEVICES`-pinned), and writes:
  - `outputs/drug_unimol.npy` (21,220 × **512**, UniMol v1 84m 3D CLS) + `drug_unimol_mask.npy`
    (**21,220/21,220** conformers OK; 2 mols used RDKit 2D→3D fallback).
  - `outputs/drug_chemberta.npy` (21,220 × **384**, `DeepChem/ChemBERTa-77M-MLM` mean-pooled = our
    "MolBERTa"). 0 NaN / 0 all-zero rows for both; **row order == `drug_feature_index.json`** (verified
    against the kernel's `drug_pert_ids.json`). Sanity NN: vorinostat→pyroxamide/Fluoro-SAHA/HNHA,
    TSA→panobinostat/dacinostat, sirolimus→everolimus/tacrolimus (chemically/mechanistically coherent);
    duplicate pert_ids of one drug sit at ~1.0 cosine. Runtime ~40 min. (Kaggle workflow notes below.)
  - MoLFormer-XL was attempted as a guarded bonus in the same kernel but **failed and was skipped**
    (optional; ChemBERTa satisfies the SMILES-LM slot). `drug_embed_meta.json` records dims/coverage.

## DTI validation reference (Done) — for learn-then-validate interpretability
Ground-truth drug→gene targets over the 978 landmark genes, from **both open DBs** (user choice), kept
as scored edges so validation can threshold. Not a hardcoded prior — the model *learns* drug→gene and
we score it against this. Outputs in `outputs/dti/`.
- **`step4_dti_foundations.py`** → `dti/landmark_genes.tsv` (978 genes: symbol, **entrez 978/978**,
  **ensp 950/978** from network-branch STRING v12 = STITCH id space) + `dti/drug_inchikeys.tsv`
  (21,220 InChIKeys, the cross-DB key; 0 fail).
- **`step5_dti_chembl.py`** (ChEMBL REST) — InChIKey→molecule, curated **mechanisms keyed on the PARENT
  molecule** (salt-attached targets not missed; e.g. imatinib→ABL1/KIT/PDGFR), target→UniProt→gene.
  **6,640/21,179** drugs in ChEMBL; **497 curated landmark edges**, **314 drugs**, **82 landmark genes**
  (high precision). `dti/chembl_dti_edges.tsv` (+ non-landmark targets, flagged). Cache `_chembl_ik2mol.json`.
- **`step6_dti_stitch.py`** (STITCH v5 human, fully local) — LINCS `pubchem_cid` (**19,329/21,220**) →
  STITCH `CIDm/CIDs`; stream the 74 MB links file, keep landmark `9606.ENSP` proteins. **28,387 edges**,
  709/950 proteins hit; **high-conf ≥700 = 4,865 edges / 687 drugs / 340 genes**. Adds *downstream*
  associations (sirolimus→EIF4EBP1/RHEB/RPS6) that ChEMBL's direct-binding view omits. Input in `../data/`.
- **`step7_dti_merge.py`** → **`dti/dti_reference.tsv`** (one row/drug×gene w/ `chembl_actions`,
  `chembl_direct`, `stitch_score`, `evidence`). **19,174 pairs, 1,718 drugs, 729 genes**; **156 pairs in
  BOTH sources, 128 with ChEMBL-curated + STITCH≥700** (cross-source agreement = gold validation core:
  β2-agonists/antagonists→ADRB2, TZDs→PPARG, NSAIDs→PTGS2 all recovered correctly).

- **`step8_integrate_atom_tokens.py`** — per-**atom** UniMol tokens (Kaggle `apexblue/lincs-atom-tokens`,
  `remove_hs=True`, `return_atomic_reprs=True`, T4×2). These are what the model attends over (atom→gene),
  distinct from the pooled CLS. `drug_atom_reprs.npy` (**703,851 × 512** ragged) + `drug_atom_offsets.npy`
  (21,221; drug j = reprs[off[j]:off[j+1]]) + `drug_atom_counts.npy` + `drug_atom_cls.npy` (21,220×512).
  All 21,220 mols OK, aligned. **~+1 token/mol vs RDKit heavy (mode +1, 21,192/21,220) = UniMol's virtual
  global node** (kept as an extra token; strip token-0 later for pure per-atom SAR). Store 1.4 GB.

## Next → the MODEL (built in `../model/`, see `../model/MODEL_MATH.md`)
Deterministic cross-attention predictor consuming this branch's {atom tokens, descriptors, fingerprint,
UniMol CLS, ChemBERTa} as drug token(s), wired to atom→gene attention; learned drug→gene scored vs
`dti/dti_reference.tsv`. Model + data pipeline built & verified (20/20 unit tests + real-data checks);
next = assemble the Kaggle training bundle and train on GPU. Optional: MoLFormer-XL second SMILES-LM.

## Env
`.venv-drug` (uv Python 3.12 + RDKit) — Python 3.14 lacks RDKit/UniMol wheels. Outputs are pert_id-keyed
lookups (NORMALIZED; never per-signature-denormalized).
**Kaggle**: creds at `C:\Users\Surya\.kaggle\kaggle.json` (user `apexblue`); 30 GPU-h/week. CLI gotchas —
(1) pass **backslash** folder paths to `kaggle datasets/kernels ... -p` (a forward-slash path corrupts the
upload staging filename on Windows); (2) first `kernels push` slugs the kernel from the **title**, so set
`id` = title-slug to update in place; (3) accelerator via `machine_shape` = `NvidiaTeslaT4` (the only T4
option = T4×2) / `NvidiaTeslaP100` / `Tpu1VmV38`; (4) log download can hit a Windows charmap error — read
`drug_embed_meta.json` instead. Kernel/dataset dirs: `scratchpad/kaggle/`.
