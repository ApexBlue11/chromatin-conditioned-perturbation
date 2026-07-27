# -*- coding: utf-8 -*-
"""
step15_rebuild_reliability.py -- rebuild E_reliability.tsv cleanly (step14's <100 threshold wrongly
penalized biologically-normal sparse H3K27me3). Base weight = coverage resolution tier; then
down-weight ONLY failed-ChIP H3K27me3 (raw nonzero peaks < 10, parsed from the step14 log).
H3K27me3 being sparse on the mostly-active 978 landmark genes is correct biology, not low quality.
"""
import sys, re, csv
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
OUT="../outputs"
RELI={"direct_measurement":1.0,"substitute_assay":0.6,"tissue_type_match":0.4,
      "related_line_inheritance":0.7,"imputed":0.5,"unresolved":0.0}
MARKS=["ATAC-seq","H3K27ac","H3K27me3"]

cov=pd.read_csv(f"{OUT}/coverage_report_phase2.tsv",sep="\t",dtype=str,keep_default_na=False)
cells=sorted(cov.lincs_cell_id.unique())
# base reliability per (cell,mark) from resolution tier
base={c:{m:0.0 for m in MARKS} for c in cells}
for _,r in cov.iterrows():
    if r.status=="resolved":
        base[r.lincs_cell_id][r.assay_target]=RELI.get(r.resolution_tier,0.5)

# parse H3K27me3 raw nonzero-peak counts from step14 log
import glob
logtxt=""
for p in glob.glob("C:/Users/Surya/AppData/Local/Temp/claude/*/*/tasks/*.output"):
    try:
        t=open(p,encoding="utf-8",errors="replace").read()
        if "beds, nonzero=" in t and "H3K27me3 slots" in t: logtxt=t
    except: pass
nz={}
for m in re.finditer(r"^\s*(\S+)\s+\S+:\s+\d+ beds, nonzero=(\d+)", logtxt, re.M):
    nz[m.group(1)]=int(m.group(2))
failed=[c for c,z in nz.items() if z<10]
print(f"parsed H3K27me3 nonzero for {len(nz)} cells; FAILED-ChIP (nonzero<10): {sorted(failed)}")
for c in failed:
    if c in base: base[c]["H3K27me3"]*=0.3   # failed ChIP -> hard down-weight

with open(f"{OUT}/E_reliability.tsv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["cell_id"]+MARKS)
    for c in cells: w.writerow([c]+[f"{base[c][m]:.2f}" for m in MARKS])
print("rebuilt E_reliability.tsv")
# summary
import numpy as np
arr={m:[base[c][m] for c in cells if base[c][m]>0] for m in MARKS}
for m in MARKS: print(f"  {m:9}: {len(arr[m])} cells weighted, mean weight={np.mean(arr[m]):.2f}")
print(f"  H3K27me3 failed-ChIP down-weighted: {sorted(failed)}")
