# -*- coding: utf-8 -*-
"""
step4_substitute_recover.py -- property-matched substitute-assay recovery (Cistrome, offline,
CVCL-verified) for the high-impact partial/zero cells. For each MISSING mark we search, in
priority order, assays that measure the SAME regulatory property (accessibility / activation /
Polycomb-repression). Reports the best available substitute per slot. Nothing folded in yet.
"""
import sys, json, re
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

with open("../../epigenetics/data/cistrome_human_samples.json") as f:
    S=json.load(f)
gap=json.load(open("../../epigenetics/data/gap_cells_cellosaurus.json"))

# cell -> (missing marks, CVCL, extra synonyms)
TARGETS={
 "A375":(["H3K27me3"],"CVCL:0132",["A375","A-375"]),
 "HA1E":(["ATAC-seq","H3K27ac","H3K27me3"],"CVCL:VU89",["HA1E","HA-1E"]),
 "HCC515":(["ATAC-seq","H3K27ac","H3K27me3"],"CVCL:5136",["HCC515","HCC-515"]),
 "YAPC":(["ATAC-seq","H3K27ac","H3K27me3"],"CVCL:1794",["YAPC","YAP-C"]),
 "BT20":(["ATAC-seq","H3K27ac","H3K27me3"],"CVCL:0178",["BT20","BT-20"]),
 "HS578T":(["ATAC-seq","H3K27ac","H3K27me3"],"CVCL:0332",["HS578T","Hs 578T","Hs-578T"]),
}
# property-matched substitute menu per slot: (tier, label, matcher)
#   tier: 0 direct, 1 strong same-property substitute, 2 weaker proxy
MENU={
 "ATAC-seq":[(0,"ATAC",("ET","ATAC")),(1,"DNase-seq",("ET","DNASE")),
             (1,"FAIRE",("F","faire")),(2,"H3K4me3",("F","h3k4me3")),(2,"H3K4me1",("F","h3k4me1"))],
 "H3K27ac":[(0,"H3K27ac",("F","h3k27ac")),(1,"H3K9ac",("F","h3k9ac")),(1,"p300/EP300",("F","ep300")),
            (1,"p300",("F","p300")),(1,"CBP/CREBBP",("F","crebbp")),(2,"H3K4me1",("F","h3k4me1")),
            (2,"H3K4me3",("F","h3k4me3"))],
 "H3K27me3":[(0,"H3K27me3",("F","h3k27me3")),(1,"EZH2",("F","ezh2")),(1,"SUZ12",("F","suz12")),
             (1,"EED",("F","eed")),(1,"H2AK119ub",("F","h2ak119")),(1,"RNF2/RING1B",("F","rnf2")),
             (2,"H3K9me3",("F","h3k9me3"))],
}
def wb(p,t):
    try: return re.search(rf"\b{re.escape(p)}\b",str(t),re.I) is not None
    except re.error: return False
def cvcls(s): return {o["ontology_accession"] for o in s.get("ontologies",[]) if o.get("ontology_type")=="CVCL"}
def factors(s): return [ (f.get("name","") or "").lower() for f in (s.get("factors") or []) if isinstance(f,dict)]
def assay_hit(s, kind, key):
    if kind=="ET": return s.get("experiment_type","")==key
    # factor match: factor-name equals key OR key substring in title (histone/factor named in title)
    return (key in factors(s)) or (key in s.get("title","").lower())

print("="*74); print("PROPERTY-MATCHED SUBSTITUTE RECOVERY (Cistrome, CVCL-verified)"); print("="*74)
result=defaultdict(dict)
for cell,(marks,tgt,extra) in TARGETS.items():
    syn=sorted(set(gap.get(cell,{}).get("synonyms",[]) or [])|set(extra)|{cell})
    # pre-filter to samples whose title/ontology name-matches this cell AND carries the target CVCL
    cellsamp=[s for s in S if tgt in cvcls(s) and any(wb(a,s.get("title","")) or
              any(wb(a,o.get("term","")) for o in s.get("ontologies",[])) for a in syn)]
    print(f"\n### {cell} (CVCL {tgt}): {len(cellsamp)} CVCL-verified Cistrome samples total")
    for mk in marks:
        best=None
        for tier,label,(kind,key) in MENU[mk]:
            hits=[s for s in cellsamp if assay_hit(s,kind,key)]
            if hits:
                best=(tier,label,[s["id"] for s in hits])
                break
        if best:
            tier,label,ids=best
            tag={0:"DIRECT",1:"substitute(strong)",2:"proxy(weak)"}[tier]
            print(f"   {mk:9} <- {label:12} [{tag}]  ids={ids[:8]}{' +'+str(len(ids)-8) if len(ids)>8 else ''}")
            result[cell][mk]={"assay":label,"tier":tier,"ids":ids}
        else:
            print(f"   {mk:9} <- none (no direct or property-matched substitute in Cistrome)")
print("\n=== SUMMARY ===")
for cell,(marks,_,_) in TARGETS.items():
    r=result.get(cell,{});
    print(f"  {cell:8} filled {len(r)}/{len(marks)}: "+", ".join(f"{m}<-{r[m]['assay']}(t{r[m]['tier']})" for m in r) if r else f"  {cell:8} filled 0/{len(marks)}")
json.dump(result,open("../outputs/substitute_recovery.json","w"),indent=1)
print("wrote ../outputs/substitute_recovery.json")
