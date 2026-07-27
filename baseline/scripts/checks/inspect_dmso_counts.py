import os
import pandas as pd

base = r"c:\Users\apexb\Downloads\LINCS Project"
md = os.path.join(base, "LINCS L1000 MetaData")
inst_path = os.path.join(md, "GSE92742_Broad_LINCS_inst_info.txt")

ii = pd.read_csv(inst_path, sep="\t", low_memory=False)
print("INST_CELL_COUNT", ii['cell_id'].dropna().astype(str).nunique())

for target in ["MCF10A", "NPC", "A375", "MCF7", "VCAP"]:
    sub = ii[(ii['cell_id'].astype(str) == target) & (ii['pert_type'].isin(['ctl_vehicle', 'ctl_untrt']))]
    print("\n", target, "dmso_count", len(sub))
    if len(sub):
        print(sub[["inst_id", "pert_type", "cell_id", "pert_iname"]].head(10).to_string(index=False))
