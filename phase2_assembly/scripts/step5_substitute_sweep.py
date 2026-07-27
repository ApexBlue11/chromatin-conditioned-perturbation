# -*- coding: utf-8 -*-
"""
step5_substitute_sweep.py -- apply the property-matched substitute menu (Cistrome, CVCL-verified)
to EVERY unresolved (cell, mark) pair in the final coverage report. This is the biologically-
principled substitute tier: for each empty slot, fill from an assay measuring the SAME regulatory
property (accessibility / activation / Polycomb-repression), never opposite-sign.
Reports total recovery. Writes substitute_sweep_candidates.tsv (for review; not folded in yet).
"""
import sys, json, re
from collections import defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

S=json.load(open("../../epigenetics/data/cistrome_human_samples.json"))
gap=json.load(open("../../epigenetics/data/gap_cells_cellosaurus.json"))
PATCH_CVCL={"A375":"CVCL:0132","HUH7":"CVCL:0336","THP1":"CVCL:0006","U937":"CVCL:0007"}
cov=pd.read_csv("../../epigenetics/outputs/coverage_report_final.tsv",sep="\t",dtype=str,keep_default_na=False)
unres=[(r.lincs_cell_id,r.assay_target) for _,r in cov.iterrows() if r.status=="unresolved"]

MENU={
 "ATAC-seq":[(0,"ATAC",("ET","ATAC")),(1,"DNase-seq",("ET","DNASE")),(2,"H3K4me3",("F","h3k4me3")),(2,"H3K4me1",("F","h3k4me1"))],
 "H3K27ac":[(0,"H3K27ac",("F","h3k27ac")),(1,"H3K9ac",("F","h3k9ac")),(1,"p300",("F","ep300")),(1,"p300",("F","p300")),(2,"H3K4me1",("F","h3k4me1")),(2,"H3K4me3",("F","h3k4me3"))],
 "H3K27me3":[(0,"H3K27me3",("F","h3k27me3")),(1,"EZH2",("F","ezh2")),(1,"SUZ12",("F","suz12")),(1,"EED",("F","eed")),(1,"H2AK119ub",("F","h2ak119")),(1,"RNF2",("F","rnf2")),(2,"H3K9me3",("F","h3k9me3"))],
}
def wb(p,t):
    try: return re.search(rf"\b{re.escape(p)}\b",str(t),re.I) is not None
    except re.error: return False
def cvcls(s): return {o["ontology_accession"] for o in s.get("ontologies",[]) if o.get("ontology_type")=="CVCL"}
def factors(s): return [(f.get("name","") or "").lower() for f in (s.get("factors") or []) if isinstance(f,dict)]
def assay_hit(s,kind,key):
    if kind=="ET": return s.get("experiment_type","")==key
    return (key in factors(s)) or (key in s.get("title","").lower())

def cell_cvcl(c): return (gap.get(c,{}) or {}).get("cvcl") or PATCH_CVCL.get(c)
def cell_syn(c):
    s=set((gap.get(c,{}) or {}).get("synonyms",[]) or []); s.add(c); return sorted(s)

rows=[]; filled=defaultdict(dict)
for cell,mk in unres:
    tgt=cell_cvcl(cell); syn=cell_syn(cell)
    if not tgt:   # primary lines with no CVCL -> can't CVCL-verify; skip (avoid wrong-cell fills)
        continue
    cellsamp=[s for s in S if tgt in cvcls(s) and any(wb(a,s.get("title","")) or
              any(wb(a,o.get("term","")) for o in s.get("ontologies",[])) for a in syn)]
    if not cellsamp: continue
    for tier,label,(kind,key) in MENU[mk]:
        hits=[s for s in cellsamp if assay_hit(s,kind,key)]
        if hits:
            rows.append(dict(cell=cell,mark=mk,substitute=label,tier=tier,
                             tier_name={0:"DIRECT",1:"substitute_strong",2:"proxy_weak"}[tier],
                             n=len(hits),ids=",".join(str(s["id"]) for s in hits[:15])))
            filled[cell][mk]=(label,tier); break

df=pd.DataFrame(rows)
df.to_csv("../outputs/substitute_sweep_candidates.tsv",sep="\t",index=False)
print("="*70); print("SUBSTITUTE SWEEP over all",len(unres),"unresolved (cell,mark) pairs"); print("="*70)
if len(df):
    print(df[["cell","mark","substitute","tier_name","n"]].to_string(index=False))
    print("\nby tier:", dict(df.tier_name.value_counts()))
    # note: tier 0 = DIRECT means an exact-mark match the earlier passes missed (worth flagging)
    newfills=df[df.tier>0]
    print(f"\nNEW property-matched substitute fills (tier>0): {len(newfills)} slots across {newfills.cell.nunique()} cells")
    direct=df[df.tier==0]
    if len(direct): print(f"DIRECT-mark matches earlier passes missed: {len(direct)} -> {list(zip(direct.cell,direct.mark,direct.substitute))}")
else:
    print("No CVCL-verified substitute matches found for any unresolved pair.")
print("\nwrote ../outputs/substitute_sweep_candidates.tsv")
