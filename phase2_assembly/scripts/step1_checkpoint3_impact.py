# -*- coding: utf-8 -*-
"""
step1_checkpoint3_impact.py -- Phase-2 CHECKPOINT 3 analysis.
How many trt_cp signatures are affected by missing epigenetic coverage?
Breaks the 83 cells into full(3/3)/partial(1-2)/zero(0/3) and counts signatures per tier,
flags designated test/val cells. NOTHING is dropped here -- reporting only.
"""
import sys, csv, glob
from collections import Counter, defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

COV="../../epigenetics/outputs/coverage_report_final.tsv"
P1=glob.glob("../../Data Info/GSE92742_Broad_LINCS_sig_info.txt/*.txt")[0]
P2=glob.glob("../../Data Info/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt/*.txt")[0]
TEST_LINES={"MCF10A","NPC"}   # designated cell-split test lines (baseline provenance)

# --- coverage tier per cell ---
cov=pd.read_csv(COV,sep="\t",dtype=str,keep_default_na=False)
res=cov[cov.status=="resolved"].groupby("lincs_cell_id").size()
allcells=cov["lincs_cell_id"].unique()
tier={}; marks_have=defaultdict(set)
for _,r in cov.iterrows():
    if r.status=="resolved": marks_have[r.lincs_cell_id].add(r.assay_target)
for c in allcells:
    n=len(marks_have[c])
    tier[c]= "full" if n==3 else ("partial" if n>0 else "zero")

# --- trt_cp signatures per cell (both phases) ---
def sig_per_cell(path):
    cnt=Counter()
    with open(path,encoding="utf-8",newline="") as f:
        r=csv.reader(f,delimiter="\t"); h=next(r)
        ci=h.index("cell_id"); ti=h.index("pert_type")
        for row in r:
            if len(row)>max(ci,ti) and row[ti]=="trt_cp": cnt[row[ci]]+=1
    return cnt
c1=sig_per_cell(P1); c2=sig_per_cell(P2)
sig=Counter();
for k,v in c1.items(): sig[k]+=v
for k,v in c2.items(): sig[k]+=v
TOTAL=sum(sig.values())

# --- aggregate by tier ---
bytier=defaultdict(lambda:[0,0])  # tier -> [n_cells, n_sigs]
for c in allcells:
    bytier[tier[c]][0]+=1; bytier[tier[c]][1]+=sig.get(c,0)

print("="*72); print("CHECKPOINT 3 -- EPIGENETIC-COVERAGE IMPACT ON trt_cp SIGNATURES"); print("="*72)
print(f"Total trt_cp signatures (both phases, pre restricted-SMILES): {TOTAL:,}")
print(f"\n{'tier':8} {'cells':>6} {'signatures':>12} {'% of sigs':>10}")
for t in ["full","partial","zero"]:
    nc,ns=bytier[t]; print(f"{t:8} {nc:6d} {ns:12,} {100*ns/TOTAL:9.2f}%")
comp = bytier['full'][1]
print(f"\nComplete gate (3/3 marks) available for: {comp:,} sigs ({100*comp/TOTAL:.2f}%)")
print(f"Missing >=1 mark (partial+zero):          {TOTAL-comp:,} sigs ({100*(TOTAL-comp)/TOTAL:.2f}%)")
print(f"  of which ZERO epigenetics (0/3):        {bytier['zero'][1]:,} sigs ({100*bytier['zero'][1]/TOTAL:.2f}%)")
print(f"  of which PARTIAL (1-2/3):               {bytier['partial'][1]:,} sigs ({100*bytier['partial'][1]/TOTAL:.2f}%)")

# --- per-cell detail for zero + partial ---
def dump(t):
    rows=sorted([(c,sig.get(c,0),sorted(marks_have[c])) for c in allcells if tier[c]==t],
                key=lambda x:-x[1])
    print(f"\n--- {t.upper()} cells ({len(rows)}) : cell, sigs, marks_present ---")
    for c,n,mk in rows:
        star=" <<< TEST/VAL LINE" if c in TEST_LINES else ""
        print(f"   {c:10} {n:7,}  {mk}{star}")
    return sum(n for _,n,_ in rows)
dump("zero"); dump("partial")

print("\n--- designated TEST/VAL lines (cell-split) coverage status ---")
for c in sorted(TEST_LINES):
    print(f"   {c:10} tier={tier.get(c,'NOT_IN_ROSTER')}  sigs={sig.get(c,0):,}  marks={sorted(marks_have[c])}")

# save a per-cell table
out=pd.DataFrame([(c,tier[c],sig.get(c,0),"|".join(sorted(marks_have[c])),c in TEST_LINES) for c in sorted(allcells)],
                 columns=["cell_id","epi_tier","trt_cp_sigs","marks_present","is_test_line"])
out.to_csv("../outputs/checkpoint3_per_cell_impact.tsv",sep="\t",index=False)
print("\nWrote ../outputs/checkpoint3_per_cell_impact.tsv")
