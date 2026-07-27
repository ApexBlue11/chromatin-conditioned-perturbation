# -*- coding: utf-8 -*-
"""
step2_recover_high_impact.py -- targeted epigenetic recovery for the highest-signature
partial/zero cells (operator: recover the big ones before finalizing the split).
Sources: Cistrome (offline, CVCL-verified) + ENCODE (live, exact ontology). CVCL-verified only.
Reports candidates; nothing is folded into the final coverage report yet.
"""
import sys, json, re, urllib.request, urllib.parse
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

# high-impact cells and the marks they are MISSING (from checkpoint3)
TARGETS = {
 "A375":  (["H3K27me3"], "CVCL:0132", ["A375","A-375"]),
 "HA1E":  (["ATAC-seq","H3K27ac","H3K27me3"], "CVCL:VU89", None),
 "HCC515":(["ATAC-seq","H3K27ac","H3K27me3"], "CVCL:5136", None),
 "YAPC":  (["ATAC-seq","H3K27ac","H3K27me3"], "CVCL:1794", None),
 "BT20":  (["ATAC-seq","H3K27ac","H3K27me3"], "CVCL:0178", None),
 "HS578T":(["ATAC-seq","H3K27ac","H3K27me3"], "CVCL:0332", None),
}
gap=json.load(open("../../epigenetics/data/gap_cells_cellosaurus.json"))
def synonyms(cell, fallback):
    s=set(gap.get(cell,{}).get("synonyms",[]) or [])
    if fallback: s|=set(fallback)
    s.add(cell)
    return sorted(s)

with open("../../epigenetics/data/cistrome_human_samples.json") as f:
    samples=json.load(f)
def wb(p,t):
    if not t or not p: return False
    try: return re.search(rf"\b{re.escape(p)}\b",str(t),re.I) is not None
    except re.error: return False
def scvcls(s): return {o["ontology_accession"] for o in s.get("ontologies",[]) if o.get("ontology_type")=="CVCL"}
def smark(s):
    et=s.get("experiment_type",""); t=s.get("title","").lower()
    if et in("ATAC","DNASE"): return "ATAC-seq"
    if et=="IP":
        for m in("h3k27ac","h3k27me3"):
            if m in t: return {"h3k27ac":"H3K27ac","h3k27me3":"H3K27me3"}[m]
    return None

def encode_all(cell):
    params=[("type","Experiment"),("status","released"),("searchTerm",cell),("format","json"),
            ("limit","200"),("field","accession"),("field","biosample_ontology.term_name"),
            ("field","assay_title"),("field","target.label")]
    url="https://www.encodeproject.org/search/?"+urllib.parse.urlencode(params)
    try:
        req=urllib.request.Request(url,headers={"Accept":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r:
            return json.loads(r.read().decode()).get("@graph",[])
    except Exception: return []

def norm(x): return "".join(c for c in str(x).lower() if c.isalnum())

print("="*72); print("TARGETED RECOVERY (Cistrome CVCL-verified + ENCODE live)"); print("="*72)
found=defaultdict(dict)
for cell,(marks,tgt,fb) in TARGETS.items():
    syn=synonyms(cell,fb); synn={norm(s) for s in syn}
    print(f"\n### {cell}  (CVCL target {tgt}) missing={marks}  synonyms={syn}")
    # ENCODE (one call, bucket)
    enc=encode_all(cell); ebuck=defaultdict(list)
    for e in enc:
        term=e.get("biosample_ontology",{}).get("term_name","")
        if norm(term) not in synn: continue
        at=e.get("assay_title",""); tl=(e.get("target") or {}).get("label","")
        if at in ("ATAC-seq","DNase-seq"): ebuck["ATAC-seq"].append(e.get("accession"))
        elif at=="Histone ChIP-seq" and tl in("H3K27ac","H3K27me3"): ebuck[tl].append(e.get("accession"))
    for mk in marks:
        # Cistrome verified
        cis=[s for s in samples if smark(s)==mk and any(wb(a,s.get("title","")) or
             any(wb(a,o.get("term","")) for o in s.get("ontologies",[])) for a in syn)]
        cis_ok=[s for s in cis if tgt in scvcls(s)]
        enc_ids=ebuck.get(mk,[])
        status=[]
        if enc_ids: status.append(f"ENCODE={enc_ids[:6]}")
        if cis_ok:  status.append(f"Cistrome(CVCL-verified {len(cis_ok)}): {[s['id'] for s in cis_ok][:8]}")
        if cis and not cis_ok: status.append(f"Cistrome name-hit but CVCL!=target ({len(cis)} samples, dominant={sorted({a for s in cis for a in scvcls(s)})[:3]})")
        print(f"   {mk:10}: {' | '.join(status) if status else 'NONE (Cistrome/ENCODE)'}")
        if enc_ids or cis_ok: found[cell][mk]=("encode" if enc_ids else "cistrome")

print("\n=== RECOVERY SUMMARY ===")
for cell,(marks,_,_) in TARGETS.items():
    rec=found.get(cell,{})
    print(f"  {cell:8} recovered {len(rec)}/{len(marks)} missing marks: {list(rec.keys()) or 'none'}")
