# -*- coding: utf-8 -*-
"""
resolve_gap_cvcls.py -- For every cell with an unresolved mark, resolve its
authoritative Cellosaurus CVCL + full synonym list, disambiguating by requiring
an exact normalized-name match to the LINCS cell_id. Caches to gap_cells_cellosaurus.json.
This is the verified-matching backbone for S1 (synonym re-search), S2 (ENCODE re-query),
S3 (EpiMap). No coverage decision is made here -- identity resolution only.
"""
import sys, json, time, urllib.request, urllib.parse
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

def norm(x): return "".join(c for c in str(x).lower() if c.isalnum())

f=pd.read_csv("coverage_report_final.tsv",sep="\t",dtype=str,keep_default_na=False)
per=f.groupby("lincs_cell_id").apply(lambda g:(g.status=="resolved").sum(),include_groups=False)
gap_cells=sorted(per[per<3].index)
missing={c:sorted(f[(f.lincs_cell_id==c)&(f.status=="unresolved")]["assay_target"]) for c in gap_cells}

def search(name):
    q=urllib.parse.urlencode({"q":name,"format":"json","fields":"id,ac,sy,misspelling","rows":"8"})
    url=f"https://api.cellosaurus.org/search/cell-line?{q}"
    req=urllib.request.Request(url,headers={"Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=25) as r:
        return json.loads(r.read().decode()).get("Cellosaurus",{}).get("cell-line-list",[])

out={}
for c in gap_cells:
    rec={"cell_id":c,"missing":missing[c],"cvcl":None,"synonyms":[],"category":None,"note":""}
    # dot-suffix / primary lines won't have a CVCL; skip network for the known ones
    base=c.split(".")[0]
    try:
        hits=search(c) or search(base)
        chosen=None
        for cl in hits:
            names=[n.get("value") for n in cl.get("name-list",[])]
            if norm(c) in {norm(n) for n in names} or norm(base) in {norm(n) for n in names}:
                chosen=cl; break
        if chosen is None and hits:
            rec["note"]="no exact-name hit; ambiguous -> left unmatched"
        if chosen:
            acc=[a.get("value") for a in chosen.get("accession-list",[]) if a.get("type")=="primary"]
            rec["cvcl"]=acc[0].replace("_",":") if acc else None
            rec["synonyms"]=sorted({n.get("value") for n in chosen.get("name-list",[])})
            rec["category"]=chosen.get("category")
    except Exception as e:
        rec["note"]=f"ERR {type(e).__name__}: {e}"
    out[c]=rec
    print(f"{c:10} cvcl={rec['cvcl']!s:12} syn={len(rec['synonyms'])} missing={rec['missing']} {rec['note']}")
    time.sleep(0.25)

json.dump(out, open("gap_cells_cellosaurus.json","w"), indent=1)
resolved=sum(1 for r in out.values() if r["cvcl"])
print(f"\nResolved a CVCL for {resolved}/{len(gap_cells)} gap cells. Cached gap_cells_cellosaurus.json")
