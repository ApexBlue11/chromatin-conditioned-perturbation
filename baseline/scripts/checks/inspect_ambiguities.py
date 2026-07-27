import os
import re
import pandas as pd

base = r"c:\Users\apexb\Downloads\LINCS Project"
md = os.path.join(base, "LINCS L1000 MetaData")
ccle = os.path.join(base, "CCLE")

sig = pd.read_csv(os.path.join(md, "GSE92742_Broad_LINCS_sig_info.txt"), sep="\t", low_memory=False)
model = pd.read_csv(os.path.join(ccle, "Model.csv"))

cells = sorted(sig.loc[sig["pert_type"].astype(str) == "trt_cp", "cell_id"].dropna().astype(str).unique().tolist())


def norm(v):
    return re.sub(r"[^A-Z0-9]", "", str(v).upper())


def row_candidates(row, key):
    hits = []
    for field in ["StrippedCellLineName", "CellLineName", "CCLEName"]:
        raw = row.get(field)
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        raw = str(raw).strip()
        if norm(raw) == key:
            hits.append((str(row["ModelID"]), field, raw, row.get("EngineeredModel"), row.get("EngineeredModelDetails"), row.get("ModelType"), row.get("IsDefaultEntryForModel"), row.get("IsDefaultEntryForMC"), row.get("OncotreeLineage"), row.get("CCLEName"), row.get("CellLineName"), row.get("StrippedCellLineName")))
    aliases = row.get("ModelIDAlias")
    if pd.notna(aliases):
        for alias in str(aliases).split(","):
            alias = alias.strip()
            if alias and norm(alias) == key:
                hits.append((str(row["ModelID"]), "ModelIDAlias", alias, row.get("EngineeredModel"), row.get("EngineeredModelDetails"), row.get("ModelType"), row.get("IsDefaultEntryForModel"), row.get("IsDefaultEntryForMC"), row.get("OncotreeLineage"), row.get("CCLEName"), row.get("CellLineName"), row.get("StrippedCellLineName")))
    return hits


ambiguous = []
for cell in cells:
    key = norm(cell)
    rows = []
    for _, row in model.iterrows():
        rows.extend(row_candidates(row, key))
    unique = sorted({r[0] for r in rows})
    if len(unique) > 1:
        ambiguous.append((cell, unique, rows))

print(f"AMBIGUOUS_COUNT {len(ambiguous)}")
for cell, unique, rows in ambiguous:
    print(f"\nCELL {cell} -> {len(unique)} ACHs {unique}")
    for r in rows:
        print(r)
