import os
import pandas as pd

base = r"c:\Users\apexb\Downloads\LINCS Project"
p = os.path.join(base, "CCLE", "Model.csv")

cols = ["ModelID", "StrippedCellLineName", "CCLEName", "CellLineName", "ModelIDAlias", "OncotreeLineage"]
df = pd.read_csv(p, usecols=cols)

wanted = ["MCF10A", "NPC", "A375", "MCF7", "VCAP"]
for w in wanted:
    print("\n===", w, "===")
    hits = df[
        df["StrippedCellLineName"].astype(str).str.upper().eq(w)
        | df["CCLEName"].astype(str).str.upper().eq(w)
        | df["CellLineName"].astype(str).str.upper().eq(w)
        | df["ModelIDAlias"].astype(str).str.upper().str.contains(w, na=False)
    ]
    print("matches", len(hits))
    if len(hits):
        print(hits.to_string(index=False))
