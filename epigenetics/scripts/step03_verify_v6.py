# -*- coding: utf-8 -*-
"""
verify_v6_partA.py  --  Independent re-verification of run_epigenetics_audit_v6.py
Read-only against v5 + v6 outputs. Prints raw evidence. No file is modified.
Every reported number is produced by an assertion or an explicit print here.
"""
import sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

V5 = "coverage_report.tsv"
V6 = "coverage_report_v6.tsv"

df5 = pd.read_csv(V5, sep="\t", dtype=str, keep_default_na=False)
df6 = pd.read_csv(V6, sep="\t", dtype=str, keep_default_na=False)

print("="*78)
print("SECTION 1  --  ROW COUNT / STRUCTURE")
print("="*78)
print("v5 rows:", len(df5), "| v6 rows:", len(df6))
assert len(df6) == 249, "v6 is not 249 rows"
# 83 cells x 3 marks, no dup
cells = sorted(df6["lincs_cell_id"].unique())
marks = sorted(df6["mark"].unique())
print("distinct cells:", len(cells), "| distinct marks:", marks)
assert len(cells) == 83, "not 83 cells"
dup = df6.duplicated(subset=["lincs_cell_id","mark"]).sum()
print("duplicate (cell,mark) pairs:", dup)
assert dup == 0
# every cell has all 3 marks
per = df6.groupby("lincs_cell_id")["mark"].nunique()
print("cells lacking all 3 marks:", list(per[per!=3].index) or "NONE")
assert (per==3).all()
# cell_id present on every row
blank_cell = (df6["lincs_cell_id"].str.strip()=="").sum()
print("rows with blank cell_id:", blank_cell)
assert blank_cell == 0
print("PASS: structure OK\n")

print("="*78)
print("SECTION 2  --  NO COMMA-JOINED AMBIGUOUS GENOME BUILDS IN v6")
print("="*78)
bad = df6[df6["genome_build"].str.contains(",", na=False)]
print("rows with comma in genome_build:", len(bad))
if len(bad): print(bad[["lincs_cell_id","mark","genome_build"]].to_string())
assert len(bad)==0
print("PASS: no ambiguous comma-joined builds\n")

print("="*78)
print("SECTION 3  --  ARE THE 12 GENUINE_CONTAMINATION ROWS UNTOUCHED (v5==v6)?")
print("="*78)
TWELVE = [
 ("HEK293T","ATAC-seq"),("HEK293T","H3K27ac"),("HL60","ATAC-seq"),
 ("JURKAT","H3K27ac"),("JURKAT","H3K27me3"),("LNCAP","ATAC-seq"),
 ("MCF10A","H3K27me3"),("MDAMB231","H3K27ac"),("MDAMB231","H3K27me3"),
 ("NPC","H3K27ac"),("SKMEL28","H3K27me3"),("THP1","H3K27ac"),
]
def getrow(df,c,m):
    r = df[(df["lincs_cell_id"]==c)&(df["mark"]==m)]
    return r.iloc[0] if len(r) else None
allmatch=True
for c,m in TWELVE:
    r5,r6 = getrow(df5,c,m), getrow(df6,c,m)
    same = (r5["source_used"]==r6["source_used"] and r5["genome_build"]==r6["genome_build"]
            and r5["notes"]==r6["notes"])
    if not same: allmatch=False
    print(f"  {c}/{m}: v5==v6 core fields? {same} | v6_change='{r6['v6_change']}' | src={r6['source_used']}")
assert allmatch, "a flagged-contaminated row changed between v5 and v6!"
print("PASS: all 12 GENUINE_CONTAMINATION rows byte-identical v5->v6\n")

print("="*78)
print("SECTION 4  --  ALL v5-RESOLVED ROWS CARRIED FORWARD IDENTICALLY?")
print("="*78)
# Every row that was resolved in v5 must be identical in v6 (v6 only adds, never edits resolved rows)
changed = []
for _,r5 in df5.iterrows():
    if r5["source_used"]=="unresolved": continue
    if r5["lincs_cell_id"]=="NPC.CAS9": continue  # v6 legitimately overrides
    r6 = getrow(df6, r5["lincs_cell_id"], r5["mark"])
    if not (r5["source_used"]==r6["source_used"] and r5["genome_build"]==r6["genome_build"] and r5["notes"]==r6["notes"]):
        changed.append((r5["lincs_cell_id"], r5["mark"]))
print("v5-resolved rows altered by v6 (excl NPC.CAS9):", changed or "NONE")
assert not changed
print("PASS: v6 did not silently rewrite any v5-resolved row\n")

print("="*78)
print("SECTION 5  --  CATEGORIZE v6 STATUS OF EVERY ROW")
print("="*78)
cat = {"unchanged_resolved":0,"new_resolution":0,"still_unresolved":0,"npc_cas9_resolved":0}
new_rows=[]
for _,r in df6.iterrows():
    ch = r["v6_change"]; src=r["source_used"]
    if src=="unresolved":
        cat["still_unresolved"]+=1
    elif ch.startswith("unchanged"):
        cat["unchanged_resolved"]+=1
    elif "pre-authorized" in ch or "NPC pre-authorized" in ch:
        cat["npc_cas9_resolved"]+=1; new_rows.append(r)
    elif ch.startswith("RESOLVED"):
        cat["new_resolution"]+=1; new_rows.append(r)
    else:
        cat.setdefault("OTHER:"+ch,0); cat["OTHER:"+ch]+=1
for k,v in cat.items(): print(f"  {k}: {v}")
print("  TOTAL new (incl NPC.CAS9):", len(new_rows))
print()

print("="*78)
print("SECTION 6  --  CLASSIFY EACH NEW v6 RESOLUTION (mechanism)")
print("="*78)
GENERIC_TERMS = {"hesc","fibroblast","motor neuron","kidney epithelial","myoblast",
                 "skeletal muscle","myocyte","adipocyte stem","adipose stem"}
# dot-suffix lines whose resolution was flagged pending/forbidden by v6's own dot_suffix table
PENDING = {"ASC.C":"NEEDS_OVERSEER_DECISION","NPC.TAK":"NEEDS_OVERSEER_DECISION",
           "SKL.C":"NEEDS_OVERSEER_DECISION","MNEU.E":"NOT_DEFENSIBLE"}
import re
rows_out=[]
for r in new_rows:
    cell=r["lincs_cell_id"]; mark=r["mark"]; notes=r["notes"]
    m = re.search(r"\[alias='([^']+)'", notes)
    alias = m.group(1) if m else ("NPC(pre-auth)" if "pre-authorized" in notes else "-")
    is_generic = alias.lower() in GENERIC_TERMS
    if "pre-authorized" in r["v6_change"]:
        mech="PRE_AUTHORIZED(locked)"
    elif is_generic:
        mech="TISSUE_TYPE_MATCH_ONLY"
    else:
        mech="NAME_BASED"
    pend = PENDING.get(cell,"")
    rows_out.append((cell,mark,alias,mech,pend,r["source_used"]))
print(f"{'cell':10} {'mark':9} {'alias_used':16} {'mechanism':24} {'dot_suffix_status'}")
for cell,mark,alias,mech,pend,src in rows_out:
    print(f"{cell:10} {mark:9} {alias:16} {mech:24} {pend}")

from collections import Counter
print("\n  Mechanism tally over the", len(rows_out), "new resolutions:")
for k,v in Counter(x[3] for x in rows_out).items(): print(f"    {k}: {v}")
print("\n  New resolutions on dot-suffix lines the v6 dot_suffix table said NOT to auto-resolve:")
for cell,mark,alias,mech,pend,src in rows_out:
    if pend: print(f"    {cell}/{mark}  (v6 documented: {pend})  -> but coverage_report_v6 marks it RESOLVED via '{alias}'")
