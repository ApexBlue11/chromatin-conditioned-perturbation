# -*- coding: utf-8 -*-
"""
step3_chipatlas_recover.py -- targeted ChIP-Atlas recovery for the high-impact cells,
WITH per-sample GEO treatment-filtering (keep only baseline/untreated; drop perturbed --
lesson from the earlier THP-1/HSV-1 finding). Name-verified via structured Cell_type.
Reports clean baseline experiment counts per (cell, mark). Nothing folded in yet.
"""
import sys, re, json, time, urllib.request
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

SUBSET="../outputs/chip_atlas_human_epi.tab"
TARGETS = {
 "A375":  (["H3K27me3"], ["A375","A-375"]),
 "HA1E":  (["ATAC-seq","H3K27ac","H3K27me3"], ["HA1E","HA-1E"]),
 "HCC515":(["ATAC-seq","H3K27ac","H3K27me3"], ["HCC515","HCC-515"]),
 "YAPC":  (["ATAC-seq","H3K27ac","H3K27me3"], ["YAPC","YAP-C","YAP C"]),
 "BT20":  (["ATAC-seq","H3K27ac","H3K27me3"], ["BT20","BT-20"]),
 "HS578T":(["ATAC-seq","H3K27ac","H3K27me3"], ["HS578T","Hs 578T","Hs-578T","Hs578T"]),
}
def wb(p,t):
    try: return re.search(rf"\b{re.escape(p)}\b",str(t),re.I) is not None
    except re.error: return False
def antigens_for(mark): return ("ATAC-Seq","DNase-seq") if mark=="ATAC-seq" else (mark,)

PERTURB_HINT=re.compile(r"\b(knock|sh|si|crispr|cas9|dox|doxycyclin|infect|virus|hsv|treat|inhibitor|"
                        r"stimul|induc|\dh\b|hour|drug|shRNA|siRNA|KO|OE|overexpress|mutant|resist)\b",re.I)
BASELINE_OK=re.compile(r"\b(untreated|control|wild.?type|WT|none|DMSO|vehicle|parental|normal|naive)\b",re.I)

rows=[l.rstrip("\n").split("\t") for l in open(SUBSET,encoding="utf-8") if l.count("\t")>=7]
print(f"loaded {len(rows)} ChIP-Atlas rows")

def geo_treatment(gsm):
    url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}&targ=self&form=text&view=quick"
    try:
        with urllib.request.urlopen(url,timeout=20) as r: t=r.read().decode("utf-8","replace")
    except Exception: return None,"(geo fetch failed)"
    fields=[ln.split("=",1)[1].strip() for ln in t.split("\n")
            if ln.startswith(("!Sample_title","!Sample_source_name_ch1","!Sample_characteristics_ch1"))]
    blob=" ; ".join(fields)
    if BASELINE_OK.search(blob) and not PERTURB_HINT.search(blob): return True, blob[:120]
    if PERTURB_HINT.search(blob): return False, blob[:120]
    return True, blob[:120]   # no perturbation hint -> treat as baseline

print("="*72); print("CHIP-ATLAS RECOVERY (name-verified + GEO baseline-filtered)"); print("="*72)
summary=defaultdict(dict)
for cell,(marks,syn) in TARGETS.items():
    print(f"\n### {cell}  synonyms={syn}")
    for mk in marks:
        ants=antigens_for(mk)
        gsms=[]
        for p in rows:
            acl,ant,ct,title=p[2],p[3],p[5],p[7]
            ok=(ant in ants) or (mk=="ATAC-seq" and acl in ("ATAC-Seq","DNase-seq"))
            if ok and any(wb(a,ct) for a in syn):
                m=re.search(r"(GSM\d+)",title)
                gsms.append(m.group(1) if m else p[0])
        gsms=list(dict.fromkeys(gsms))
        if not gsms:
            print(f"   {mk:10}: no name match"); continue
        # GEO-filter (cap to first 12 to bound network)
        baseline=[]; perturbed=[]
        for g in gsms[:12]:
            if not g.startswith("GSM"): baseline.append(g); continue
            ok,info=geo_treatment(g); time.sleep(0.25)
            (baseline if ok else perturbed).append(g)
        print(f"   {mk:10}: {len(gsms)} match(es); baseline={len(baseline)} {baseline[:6]} | perturbed-dropped={len(perturbed)}")
        if baseline: summary[cell][mk]=baseline
print("\n=== RECOVERY SUMMARY (baseline-only) ===")
for cell,(marks,_) in TARGETS.items():
    rec=summary.get(cell,{})
    print(f"  {cell:8} recovered {len(rec)}/{len(marks)}: {list(rec.keys()) or 'none'}")
json.dump(summary, open("../outputs/chipatlas_recovery.json","w"), indent=1)
print("wrote ../outputs/chipatlas_recovery.json")
