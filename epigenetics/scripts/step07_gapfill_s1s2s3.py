# -*- coding: utf-8 -*-
"""
gapfill_s1s2s3.py -- Execute S1 (Cistrome synonym re-search, CVCL-VERIFIED),
S2 (live ENCODE re-query), S3 (EpiMap imputed) for every unresolved (cell,mark).
Writes gapfill_candidates.tsv for operator review BEFORE folding into the final report.

Verification rule (operator-mandated for S1): a Cistrome synonym hit is accepted only
if the matched sample's CVCL set contains the target cell's CVCL (VERIFIED_CVCL_MATCH).
Hits whose CVCL differs, or where target has no CVCL, are recorded as UNVERIFIED and NOT accepted.
"""
import sys, json, re, time, urllib.request, urllib.parse
from collections import defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

def norm(x): return "".join(c for c in str(x).lower() if c.isalnum())
def wb(pat,text):
    if not text or not pat: return False
    try: return re.search(rf"\b{re.escape(pat)}\b",str(text),re.IGNORECASE) is not None
    except re.error: return False

gap=json.load(open("gap_cells_cellosaurus.json"))
# patch 4 well-known lines my exact-match disambiguation skipped (CVCLs from classified audit)
PATCH={"A375":("CVCL:0132",["A375","A-375"]),"HUH7":("CVCL:0336",["Huh-7","Huh7","HuH-7","HUH7"]),
       "THP1":("CVCL:0006",["THP-1","THP1"]),"U937":("CVCL:0007",["U-937","U937"])}
for c,(cv,syn) in PATCH.items():
    if c in gap and not gap[c]["cvcl"]:
        gap[c]["cvcl"]=cv; gap[c]["synonyms"]=sorted(set(gap[c]["synonyms"])|set(syn)|{c})

with open("cistrome_human_samples.json") as fjs: samples=json.load(fjs)
epi=pd.read_csv("epimap_metadata.tsv",sep="\t",dtype=str,keep_default_na=False)

def sample_cvcls(s):
    return {o["ontology_accession"] for o in s.get("ontologies",[]) if o.get("ontology_type")=="CVCL"}
def sample_mark(s):
    et=s.get("experiment_type",""); t=s.get("title","").lower()
    if et in("ATAC","DNASE"): return "ATAC-seq"
    if et=="IP":
        for m in("h3k27ac","h3k27me3"):
            if m in t: return {"h3k27ac":"H3K27ac","h3k27me3":"H3K27me3"}[m]
    return None

# ---- ENCODE: one call per cell, bucket by assay client-side ----
def encode_all(cell):
    params=[("type","Experiment"),("status","released"),("searchTerm",cell),("format","json"),
            ("limit","200"),("field","accession"),("field","biosample_ontology.term_name"),
            ("field","assay_title"),("field","target.label"),("field","files.assembly"),("field","files.status")]
    url="https://www.encodeproject.org/search/?"+urllib.parse.urlencode(params)
    try:
        req=urllib.request.Request(url,headers={"Accept":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r:
            return json.loads(r.read().decode()).get("@graph",[])
    except Exception as e:
        return []

# EpiMap name index (perturb null)
epi_idx=defaultdict(list)
for _,r in epi.iterrows():
    if r["perturb"].strip(): continue
    for k in (r["ct"],r["name"]):
        if k.strip(): epi_idx[norm(k)].append(r["id"])

rows=[]
cells=sorted(gap)
for c in cells:
    g=gap[c]; syn=g["synonyms"] or [c]; tgt=g["cvcl"]; miss=g["missing"]
    syn_norm={norm(s) for s in syn}|{norm(c)}
    # ---- S2 ENCODE ----
    enc=encode_all(c); time.sleep(0.15)
    enc_by_mark=defaultdict(list)
    for exp in enc:
        term=exp.get("biosample_ontology",{}).get("term_name","")
        if norm(term) not in syn_norm: continue        # exact ontology == a synonym
        at=exp.get("assay_title",""); tl=(exp.get("target") or {}).get("label","")
        asm=set(fx.get("assembly") for fx in exp.get("files",[]) if fx.get("status")=="released" and fx.get("assembly"))
        build="GRCh38" if "GRCh38" in asm else ("hg38" if "hg38" in asm else (sorted(asm)[0] if asm else "GRCh38"))
        if at in("ATAC-seq","DNase-seq"): enc_by_mark[("ATAC-seq",at)].append((exp.get("accession"),build))
        elif at=="Histone ChIP-seq" and tl in("H3K27ac","H3K27me3"): enc_by_mark[(tl,at)].append((exp.get("accession"),build))
    for (mk,at),lst in enc_by_mark.items():
        if mk not in miss: continue
        accs=[a for a,_ in lst]; build=lst[0][1]
        rows.append(dict(cell=c,mark=mk,strategy="S2",source="encode",assay_used=at,
            matched_via=f"ontology==synonym", target_cvcl=tgt or "", matched_cvcl="(ENCODE ontology exact)",
            verification="VERIFIED_ONTOLOGY_EXACT",contamination="clean_by_design",build=build,
            sample_ids=",".join(accs),is_primary=str(at==mk)))
    # ---- S1 Cistrome synonym re-search + CVCL verification ----
    for mk in miss:
        hits=[]
        for s in samples:
            if sample_mark(s)!=mk: continue
            title=s.get("title",""); terms=[o.get("term","") for o in s.get("ontologies",[])]
            if not any(wb(a,title) or any(wb(a,t) for t in terms) for a in syn): continue
            hits.append(s)
        if not hits: continue
        # verify by CVCL
        if tgt:
            verified=[s for s in hits if tgt in sample_cvcls(s)]
            unver   =[s for s in hits if tgt not in sample_cvcls(s)]
        else:
            verified=[]; unver=hits
        if verified:
            asm=set(s.get("assembly") for s in verified if s.get("assembly"))
            build="hg38" if ("hg38" in asm or "GRCh38" in asm) else ("hg19" if ("hg19" in asm or not asm) else sorted(asm)[0])
            rows.append(dict(cell=c,mark=mk,strategy="S1",source="cistrome",assay_used=mk,
                matched_via="synonym+CVCL", target_cvcl=tgt, matched_cvcl=tgt,
                verification="VERIFIED_CVCL_MATCH",contamination="clean",build=build,
                sample_ids=",".join(str(s["id"]) for s in verified),is_primary="True"))
        if unver and not verified:
            # record but DO NOT accept
            other=sorted({a for s in unver for a in sample_cvcls(s)})[:4]
            rows.append(dict(cell=c,mark=mk,strategy="S1",source="cistrome",assay_used=mk,
                matched_via="synonym", target_cvcl=tgt or "(none)", matched_cvcl=",".join(other) or "(no CVCL)",
                verification="UNVERIFIED_CVCL_MISMATCH",contamination="n/a",build="",
                sample_ids=",".join(str(s["id"]) for s in unver[:10]),is_primary="True"))
    # ---- S3 EpiMap (imputed) ----
    epi_hits=set()
    for a in syn+[c]:
        for bid in epi_idx.get(norm(a),[]): epi_hits.add(bid)
    if epi_hits:
        for mk in miss:
            rows.append(dict(cell=c,mark=mk,strategy="S3",source="epimap",assay_used=mk,
                matched_via="biosample name==synonym", target_cvcl=tgt or "", matched_cvcl="(EpiMap biosample)",
                verification="IMPUTED",contamination="imputed_track",build="hg19",
                sample_ids=",".join(sorted(epi_hits)),is_primary="True"))

df=pd.DataFrame(rows)
if len(df):
    df=df[["cell","mark","strategy","source","assay_used","is_primary","verification",
           "target_cvcl","matched_cvcl","contamination","build","sample_ids","matched_via"]]
df.to_csv("gapfill_candidates.tsv",sep="\t",index=False)

print("="*70); print("GAPFILL CANDIDATES (for review; nothing folded into final yet)"); print("="*70)
print("total candidate rows:",len(df))
if len(df):
    print("\nby strategy x verification:")
    print(df.groupby(["strategy","verification"]).size().to_string())
    acc=df[df.verification.isin(["VERIFIED_CVCL_MATCH","VERIFIED_ONTOLOGY_EXACT","IMPUTED"])]
    print(f"\nACCEPTABLE candidates: {len(acc)} rows covering {acc['cell'].nunique()} cells, "
          f"{len(acc.groupby(['cell','mark']))} distinct (cell,mark) gaps")
    print("\n--- ACCEPTABLE (VERIFIED / IMPUTED) rows ---")
    for _,r in acc.iterrows():
        print(f"  {r.cell:9} {r.mark:9} {r.strategy} {r.source:7} {r.verification:22} cvcl={r.target_cvcl:10} ids={r.sample_ids[:40]}")
    rej=df[df.verification=="UNVERIFIED_CVCL_MISMATCH"]
    print(f"\n--- REJECTED (synonym matched but CVCL != target): {len(rej)} rows (NOT accepted) ---")
    for _,r in rej.iterrows():
        print(f"  {r.cell:9} {r.mark:9} target={r.target_cvcl:10} matched={r.matched_cvcl}  ids={r.sample_ids[:30]}")
print("\nWrote gapfill_candidates.tsv")
