# -*- coding: utf-8 -*-
"""
chipatlas_match.py -- S4. Match the 32 zero-coverage (+ any gap) cells against the
ChIP-Atlas human subset. ChIP-Atlas carries NO CVCL, so matching is name-based:
a Cellosaurus synonym must word-boundary-match the STRUCTURED Cell_type field
(col 5) -- not just the free-text title -- to count. Every match is surfaced for
operator review (name_verified_chipatlas confidence, weaker than S1's CVCL match).
Writes chipatlas_candidates.tsv. Nothing folded into the final report here.
"""
import sys, json, re
from collections import defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

def norm(x): return "".join(c for c in str(x).lower() if c.isalnum())
def wb(pat,text):
    if not text or not pat: return False
    try: return re.search(rf"\b{re.escape(pat)}\b",str(text),re.IGNORECASE) is not None
    except re.error: return False

gap=json.load(open("gap_cells_cellosaurus.json"))
PATCH={"A375":["A375","A-375"],"HUH7":["Huh-7","Huh7","HuH-7","HUH7"],
       "THP1":["THP-1","THP1"],"U937":["U-937","U937"]}
for c,syn in PATCH.items():
    if c in gap: gap[c]["synonyms"]=sorted(set(gap[c]["synonyms"])|set(syn)|{c})

rows=[]
with open("chip_atlas_human_epi.tab",encoding="utf-8") as f:
    for ln in f:
        p=ln.rstrip("\n").split("\t")
        if len(p)<8: continue
        rows.append(p)   # id,genome,acl,antigen,ctclass,ct,ctdesc,title
print(f"loaded {len(rows)} ChIP-Atlas human epi rows")

def antigens_for(mark):
    if mark=="ATAC-seq": return ("ATAC-Seq","DNase-seq")   # class or antigen
    return (mark,)

# only the still-unresolved (cell,mark) after EpiMap fold-in
f=pd.read_csv("coverage_report_final.tsv",sep="\t",dtype=str,keep_default_na=False)
unres={(r.lincs_cell_id,r.assay_target) for _,r in f.iterrows() if r.status=="unresolved"}

out=[]
for c in sorted({cell for cell,_ in unres}):
    if c not in gap: continue
    syn=gap[c]["synonyms"] or [c]
    for mark in ["ATAC-seq","H3K27ac","H3K27me3"]:
        if (c,mark) not in unres: continue
        ants=antigens_for(mark)
        hits=[]
        for p in rows:
            acl,antigen,ct,ctdesc=p[2],p[3],p[5],p[6]
            mark_ok = (antigen in ants) or (mark=="ATAC-seq" and acl in ("ATAC-Seq","DNase-seq"))
            if not mark_ok: continue
            # require synonym word-boundary match in the STRUCTURED cell_type field
            if any(wb(a,ct) for a in syn):
                hits.append(p)
        if hits:
            cts=sorted({h[5] for h in hits})
            out.append(dict(cell=c,mark=mark,n_exp=len(hits),
                            matched_cell_types=" | ".join(cts)[:80],
                            example_ids=",".join(h[0] for h in hits[:6]),
                            confidence="name_verified_chipatlas"))

df=pd.DataFrame(out)
df.to_csv("chipatlas_candidates.tsv",sep="\t",index=False)
print("="*70); print("S4 ChIP-Atlas CANDIDATES (for review; nothing folded in)"); print("="*70)
if len(df)==0:
    print("No ChIP-Atlas matches for any zero-coverage cell in cell_type field.")
else:
    print(f"{len(df)} (cell,mark) gaps could be filled, covering {df['cell'].nunique()} cells:\n")
    for _,r in df.iterrows():
        print(f"  {r.cell:9} {r.mark:9} n_exp={r.n_exp:3}  cell_types='{r.matched_cell_types}'  ids={r.example_ids}")
    print(f"\ncells gaining >=1 mark: {sorted(df['cell'].unique())}")
print("\nWrote chipatlas_candidates.tsv")
