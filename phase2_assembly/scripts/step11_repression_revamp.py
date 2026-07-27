# -*- coding: utf-8 -*-
"""
step11_repression_revamp.py -- targeted epigenetic revamp (analysis phase).
(1) Repression: for cells lacking H3K27me3, search Cistrome (CVCL-verified) for property-matched
    PRC2/PRC1 substitutes (EZH2, SUZ12, EED, H2AK119ub) -- better than H3K9me3.
(2) Imputed-vs-tissue (prefer imputed): for the current tissue-matched cells, check EpiMap
    imputed-track availability (would replace the tissue match).
Reports the gain; nothing folded in yet.
"""
import sys, json, re
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT="../.."
cov=pd.read_csv("../outputs/coverage_report_phase2.tsv",sep="\t",dtype=str,keep_default_na=False)
S=json.load(open(f"{ROOT}/epigenetics/data/cistrome_human_samples.json"))
gap=json.load(open(f"{ROOT}/epigenetics/data/gap_cells_cellosaurus.json"))
epi=pd.read_csv(f"{ROOT}/epigenetics/data/epimap_metadata.tsv",sep="\t",dtype=str,keep_default_na=False)
PATCH={"A375":"CVCL:0132","HUH7":"CVCL:0336","THP1":"CVCL:0006","U937":"CVCL:0007"}
def cvcl(c): return (gap.get(c,{}) or {}).get("cvcl") or PATCH.get(c)
def syn(c):
    s=set((gap.get(c,{}) or {}).get("synonyms",[]) or []); s.add(c); return sorted(s)
def scv(s): return {o["ontology_accession"] for o in s.get("ontologies",[]) if o.get("ontology_type")=="CVCL"}
def factors(s): return [(f.get("name","") or "").lower() for f in (s.get("factors") or []) if isinstance(f,dict)]
def wb(p,t):
    try: return re.search(rf"\b{re.escape(p)}\b",str(t),re.I) is not None
    except re.error: return False

REPRESSION_SUBS=[("EZH2","ezh2"),("SUZ12","suz12"),("EED","eed"),("H2AK119ub","h2ak119"),("H3K9me3","h3k9me3")]

# cells lacking H3K27me3
missing_me3=sorted(cov[(cov.assay_target=="H3K27me3")&(cov.status=="unresolved")]["lincs_cell_id"])
print(f"cells lacking H3K27me3 (repression): {len(missing_me3)}")
gained=[]
for c in missing_me3:
    tgt=cvcl(c)
    if not tgt: continue
    cs=[s for s in S if tgt in scv(s) and any(wb(a,s.get("title","")) or
        any(wb(a,o.get("term","")) for o in s.get("ontologies",[])) for a in syn(c))]
    for label,key in REPRESSION_SUBS:
        hits=[s for s in cs if s.get("experiment_type")=="IP" and (key in factors(s) or key in s.get("title","").lower())]
        if hits:
            gained.append((c,label,[s["id"] for s in hits][:8]))
            print(f"   {c:10} repression via {label}: ids={[s['id'] for s in hits][:6]}")
            break
print(f"\nREPRESSION GAIN: {len(gained)} cells gain a repression feature via PRC2/PRC1 substitutes")
print(f"  -> H3K27me3 property coverage would go from {83-len(missing_me3)} to {83-len(missing_me3)+len(gained)} cells")

# imputed availability for current tissue-matched cells
print("\n--- imputed-vs-tissue (prefer imputed): EpiMap availability for tissue-matched cells ---")
tissue_cells=sorted(cov[cov.identity_confidence=="tissue_type_only"]["lincs_cell_id"].unique())
def norm(x): return "".join(ch for ch in str(x).lower() if ch.isalnum())
epi_names={}
for _,r in epi.iterrows():
    if r["perturb"].strip(): continue
    for k in (r["ct"],r["name"]):
        if k.strip(): epi_names.setdefault(norm(k),r["id"])
for c in tissue_cells:
    hit=None
    for a in syn(c)+[c]:
        if norm(a) in epi_names: hit=epi_names[norm(a)]; break
    print(f"   {c:8} tissue-matched -> EpiMap imputed biosample: {hit or 'NONE (keep tissue match)'}")
