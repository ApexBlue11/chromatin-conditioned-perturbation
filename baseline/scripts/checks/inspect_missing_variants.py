import os
import re
import pandas as pd

base = r"c:\Users\apexb\Downloads\LINCS Project"
md = os.path.join(base, "LINCS L1000 MetaData")
ccle = os.path.join(base, "CCLE")

sig = pd.read_csv(os.path.join(md, "GSE92742_Broad_LINCS_sig_info.txt"), sep="\t", low_memory=False)
model = pd.read_csv(os.path.join(ccle, "Model.csv"))

cells = sorted(sig.loc[sig["pert_type"].astype(str) == "trt_cp", "cell_id"].dropna().astype(str).unique().tolist())
missing = ["ASC", "CD34", "FIBRNPC", "H1299", "HEK293T", "HS27A", "MCH58", "NEU", "NKDBA", "NPC", "PHH", "SKB", "U266"]


def norm(v):
    return re.sub(r"[^A-Z0-9]", "", str(v).upper())

for cell in missing:
    key = norm(cell)
    print("\n===", cell, "===")
    for _, row in model.iterrows():
        ach = str(row["ModelID"])
        values = []
        for field in ["StrippedCellLineName", "CellLineName", "CCLEName", "ModelIDAlias"]:
            raw = row.get(field)
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            if field == "ModelIDAlias":
                for alias in str(raw).split(","):
                    alias = alias.strip()
                    if alias and (norm(alias) == key or key in norm(alias) or norm(alias) in key):
                        values.append((field, alias, norm(alias)))
            else:
                raw = str(raw).strip()
                if norm(raw) == key or key in norm(raw) or norm(raw) in key:
                    values.append((field, raw, norm(raw)))
        if values:
            print(ach, values, row.get("OncotreeLineage"), row.get("EngineeredModel"), row.get("CCLEName"), row.get("CellLineName"), row.get("StrippedCellLineName"), row.get("ModelIDAlias"))
