# -*- coding: utf-8 -*-
"""
classify_v6_matches.py -- Apply the item-8 dominant-identity screen (offline)
to every NEW v6 resolution pool and every substitute-assay pool.

Dominant identity = SUM of CVCL ontology 'value' per accession (item 6 rule),
NOT raw sample count. Reports distinct-CVCL spread so tissue-type sweeps are
visible (a real cell-line match collapses to ~1 dominant CVCL; a tissue-type
sweep scatters across many). Cellosaurus network ancestry walk is NOT done here
(no network dependency); rows needing it are marked NEEDS_ANCESTRY_WALK.
"""
import sys, json, re
from collections import defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

with open("cistrome_human_samples.json") as f:
    samples = json.load(f)
by_id = {str(s["id"]): s for s in samples}

def pool_stats(ids):
    ids = [str(i) for i in ids]
    val = defaultdict(float); cnt = defaultdict(int); term = {}
    zero_cvcl = 0; found = 0
    for sid in ids:
        s = by_id.get(sid)
        if s is None: continue
        found += 1
        cvcls = [(o.get("ontology_accession"), o.get("term",""), float(o.get("value",0)))
                 for o in s.get("ontologies",[]) if isinstance(o,dict)
                 and str(o.get("ontology_type"))=="CVCL"]
        if not cvcls: zero_cvcl += 1
        for acc,t,v in cvcls:
            val[acc]+=v; cnt[acc]+=1; term[acc]=t
    ranked = sorted(val.items(), key=lambda kv: kv[1], reverse=True)
    return found, zero_cvcl, ranked, term, cnt

def show(label, ids, target_cvcl=None):
    found, zero, ranked, term, cnt = pool_stats(ids)
    n = len(list(ids))
    distinct = len(ranked)
    print(f"\n{label}")
    print(f"    samples in note: {n} | found in cache: {found} | zero-CVCL samples: {zero} | distinct CVCLs: {distinct}")
    for acc, v in ranked[:6]:
        star = "  <-- TARGET" if (target_cvcl and acc==target_cvcl) else ""
        print(f"      {acc:12} {term.get(acc,''):24} sum_value={v:7.3f}  n={cnt[acc]}{star}")
    if distinct>6: print(f"      ... +{distinct-6} more distinct CVCLs")
    dom = ranked[0][0] if ranked else None
    return dom, distinct, zero, found, n

def ids_from_note(note):
    m = re.search(r"sample ID\(s\):\s*([0-9,]+)", note)
    return m.group(1).split(",") if m else []

print("#"*78)
print("# PART A:  24 NEW v6 PRIMARY-ASSAY RESOLUTIONS  (item-8 dominant-identity screen)")
print("#"*78)
df6 = pd.read_csv("coverage_report_v6.tsv", sep="\t", dtype=str, keep_default_na=False)
new6 = df6[df6["v6_change"].str.startswith("RESOLVED")]
# known target CVCLs where derivable (from classified audit / known biology)
TARGET = {}  # these lines have no established single CVCL for a generic pool; left None
for _,r in new6.iterrows():
    cell, mark = r["lincs_cell_id"], r["mark"]
    if "Cistrome sample" not in r["notes"]:
        print(f"\n{cell}/{mark}: source={r['source_used']} (non-Cistrome, e.g. EpiMap/pre-auth) -- {r['notes'][:80]}")
        continue
    ids = ids_from_note(r["notes"])
    alias = re.search(r"\[alias='([^']+)'", r["notes"])
    show(f"{cell}/{mark}  (alias='{alias.group(1) if alias else '-'}')", ids)

print("\n\n"+"#"*78)
print("# PART B:  10 SUBSTITUTE-ASSAY POOLS  (item-8 screen + target-CVCL check)")
print("#"*78)
# target CVCL per cell, from cistrome_subline_audit_classified.tsv dominant of primary marks
SUB_TARGET = {"AGS":"CVCL:0139","H1299":"CVCL:0060","NPC":"CVCL:9771",
              "NPC.CAS9":"CVCL:9771","NPC.TAK":"CVCL:9771","PHH":None,"SKMEL28":"CVCL:0526"}
dfsub = pd.read_csv("substitute_assay_results.tsv", sep="\t", dtype=str, keep_default_na=False)
for _,r in dfsub.iterrows():
    cell, mark, src = r["lincs_cell_id"], r["mark"], r["source_used"]
    if src=="encode":
        print(f"\n{cell}/{mark}: ENCODE {r['notes']}")
        print(f"    -> CLEAN by design (ENCODE exact biosample_ontology equality). class in file={r['contamination_classification']}")
        continue
    ids = ids_from_note(r["notes"])
    tgt = SUB_TARGET.get(cell)
    dom, distinct, zero, found, n = show(f"{cell}/{mark}  (Cistrome substitute; file class={r['contamination_classification']})", ids, tgt)
    if tgt is None:
        verdict = "primary/no-CVCL target -- ancestry n/a; zero-CVCL pool is consistent" if zero==found else "NEEDS_ANCESTRY_WALK"
    elif dom==tgt:
        verdict = f"DOMINANT MATCHES TARGET {tgt} -> screen=CLEAN (name-verified)"
    else:
        verdict = f"DOMINANT {dom} != target {tgt} -> NEEDS_ANCESTRY_WALK"
    print(f"    VERDICT: {verdict}")

print("\n\n"+"#"*78)
print("# PART C:  RE-CHECK the 3 rows the BRIEFING claimed are MULTIPLEXED_EXPERIMENT")
print("#          (file classifies all 3 as GENUINE_CONTAMINATION). MULTIPLEXED rule =")
print("#          two named lines with NEAR-EQUAL summed confidence.")
print("#"*78)
df5 = pd.read_csv("coverage_report.tsv", sep="\t", dtype=str, keep_default_na=False)
for cell,mark in [("HEK293T","ATAC-seq"),("MDAMB231","H3K27ac"),("MDAMB231","H3K27me3")]:
    row = df5[(df5["lincs_cell_id"]==cell)&(df5["mark"]==mark)].iloc[0]
    ids = ids_from_note(row["notes"])
    dom, distinct, zero, found, n = show(f"{cell}/{mark}", ids)
    _,_,ranked,term,_ = pool_stats(ids)
    if len(ranked)>=2:
        top,second = ranked[0][1], ranked[1][1]
        ratio = second/top if top else 0
        near = ratio > 0.5
        print(f"    top2 summed values: {ranked[0][0]}={top:.3f} vs {ranked[1][0]}={second:.3f} | ratio={ratio:.3f}")
        print(f"    near-equal (MULTIPLEXED rule)? {near}  -> {'MULTIPLEXED candidate' if near else 'NOT multiplexed; GENUINE_CONTAMINATION consistent with file'}")
