import os
import pandas as pd

base = r"c:\Users\apexb\Downloads\LINCS Project"
path = os.path.join(base, "CCLE", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

targets = {"ACH-001357", "ACH-000219", "ACH-000019", "ACH-000115"}
found = set()
for chunk in pd.read_csv(path, usecols=["ModelID"], chunksize=500):
    vals = set(chunk["ModelID"].astype(str).tolist())
    found.update(targets & vals)
    if found == targets:
        break

print("FOUND", sorted(found))
print("MISSING", sorted(targets - found))
