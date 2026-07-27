import os
import pandas as pd

base = r"c:\Users\apexb\Downloads\LINCS Project"
md = os.path.join(base, "LINCS L1000 MetaData")
ii = pd.read_csv(os.path.join(md, "GSE92742_Broad_LINCS_inst_info.txt"), sep="\t", low_memory=False)
missing = ['ASC','CD34','FIBRNPC','H1299','HEK293T','HS27A','MCH58','NEU','NKDBA','NPC','PHH','SKB','U266']
for cell in missing:
    sub = ii[(ii['cell_id'].astype(str) == cell) & (ii['pert_type'].astype(str) == 'ctl_vehicle')]
    source = 'ctl_vehicle'
    if sub.empty:
        sub = ii[(ii['cell_id'].astype(str) == cell) & (ii['pert_type'].astype(str) == 'ctl_untrt')]
        source = 'ctl_untrt'
    print(cell, len(sub), source)
