# -*- coding: utf-8 -*-
"""
build_coverage_final.py -- Part C. Implements the locked Checkpoint-1 decisions.
Produces coverage_report_final.tsv (one row per cell x primary-mark = 249 rows)
plus dropped_contaminant_samples.tsv (audit of every sample removed).
Every decision is applied explicitly; consistency checks assert at the end.
"""
import sys, json, re
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

df6  = pd.read_csv("coverage_report_v6.tsv", sep="\t", dtype=str, keep_default_na=False)
df5  = pd.read_csv("coverage_report.tsv",    sep="\t", dtype=str, keep_default_na=False)
sub  = pd.read_csv("substitute_assay_results.tsv", sep="\t", dtype=str, keep_default_na=False)
cls  = pd.read_csv("cistrome_subline_audit_classified.tsv", sep="\t", dtype=str, keep_default_na=False)
with open("cistrome_human_samples.json") as f: samples=json.load(f)
by_id={str(s["id"]):s for s in samples}

def ids_from_note(n):
    m=re.search(r"sample ID\(s\):\s*([0-9,]+)",n); return m.group(1).split(",") if m else []
def cvcls(sid):
    return {o["ontology_accession"] for o in by_id.get(sid,{}).get("ontologies",[]) if o.get("ontology_type")=="CVCL"}

# ---- genuine independent-contaminant CVCLs per (cell,mark) from classified audit ----
CONTAM={}
for _,r in cls[cls["classification"]=="GENUINE_CONTAMINATION"].iterrows():
    indep=set(re.findall(r"CVCL:[0-9A-Z]+", " ".join(
        seg for seg in r["evidence"].split(";") if "independently established" in seg)))
    CONTAM[(r["cell_line"],r["mark"])]=indep
TWELVE=set(CONTAM)

def clean_pool(cell, mark, ids):
    """Drop samples carrying a genuine independent contaminant; return (kept, dropped)."""
    bad=CONTAM.get((cell,mark),set())
    dropped=[s for s in ids if (cvcls(s)&bad)]
    kept=[s for s in ids if s not in dropped]
    return kept,dropped

# ---- rebuilt CLEAN skeletal-muscle tissue pools (AT6BR LCL + DUX4-perturbed excluded) ----
MUSCLE={
 "ATAC-seq":  (["7921","78636","78637","78689","78690","91617","91618","91619"],"hg38"),
 "H3K27ac":   (["38011","68382"],"hg19"),
 "H3K27me3":  (["3438","3439","3444"],"hg19"),
}

# ---- accepted substitute assays (fold into the ATAC-seq target row) ----
sub_ok={}
for _,r in sub.iterrows():
    cell=r["lincs_cell_id"]
    if cell=="SKMEL28":   # operator decision: drop -> leave unresolved
        continue
    sub_ok[cell]=dict(assay=r["substitute_assay_used"], source=r["source_used"],
                      build=r["genome_build"], notes=r["notes"])

CELLS=sorted(df6["lincs_cell_id"].unique())
MARKS=["ATAC-seq","H3K27ac","H3K27me3"]
v6map={(r["lincs_cell_id"],r["mark"]):r for _,r in df6.iterrows()}

# operator-approved EpiMap imputed gap-fills (S3), verified to correct biosample
EPIMAP_IMPUTED={("HS27A","H3K27ac"):"BSS00704",("HS27A","H3K27me3"):"BSS00704",
                ("HT29","H3K27me3"):"BSS00708",("HUH7","H3K27me3"):"BSS00718"}
TISSUE_KEEP={"HUES3","MCH58","NKDBA"}      # tissue-appropriate, keep v6 pool
MUSCLE_LINES={"SKL","SKB","SKL.C"}
drop_audit=[]
rows=[]

def emit(cell,mark,**k):
    base=dict(lincs_cell_id=cell, assay_target=mark, status="unresolved",
              resolution_tier="unresolved", assay_used="", is_primary_assay="",
              source_used="", genome_build="", identity_confidence="",
              contamination_status="", model_confidence_flag="",
              sample_ids_used="", sample_ids_dropped="", notes="")
    base.update(k); rows.append(base)

def npc_final(mark):
    """NPC's own cleaned final state for a mark (used by NPC/NPC.CAS9/NPC.TAK)."""
    if mark=="ATAC-seq":
        ids=ids_from_note(sub.loc[sub.lincs_cell_id=="NPC","notes"].iloc[0])
        kept=[s for s in ids if not (cvcls(s)&{"CVCL:7047","CVCL:7084"})]
        drop=[s for s in ids if s not in kept]
        return dict(assay="H3K4me3 ChIP-seq",src="cistrome",build="hg38",kept=kept,drop=drop,primary=False)
    row=v6map[("NPC",mark)]; ids=ids_from_note(row["notes"])
    kept,drop=clean_pool("NPC",mark,ids)
    return dict(assay=mark,src="cistrome",build=row["genome_build"],kept=kept,drop=drop,primary=True)

for cell in CELLS:
    for mark in MARKS:
        r6=v6map[(cell,mark)]; src=r6["source_used"]; notes=r6["notes"]

        # ---- MNEU.E locked unresolved ----
        if cell=="MNEU.E":
            emit(cell,mark,status="unresolved",resolution_tier="unresolved",
                 notes="MNEU.E terminally-differentiated motor neurons: NOT_DEFENSIBLE to inherit (locked).")
            continue

        # ---- SKMEL28 REHABILITATED (MEL-745A tag = Cistrome parser artifact; GEO/ChIP-Atlas
        #      confirm GSM838934/GSM838933 are cleanly SK-MEL-28). Operator-approved reversal. ----
        if cell=="SKMEL28":
            if mark=="H3K27me3":
                emit(cell,mark,status="resolved",resolution_tier="direct_measurement",
                     assay_used="H3K27me3",is_primary_assay="True",source_used="cistrome",genome_build="hg38",
                     identity_confidence="name_verified",contamination_status="clean",
                     model_confidence_flag="rehabilitated: MEL-745A CVCL tag was Cistrome parser artifact; GEO GSM838934='SK-MEL-28'",
                     sample_ids_used="43882",notes="SK-MEL-28 H3K27me3 (GSM838934); MEL-745A tag disregarded as parser artifact.")
            elif mark=="ATAC-seq":
                emit(cell,mark,status="resolved",resolution_tier="substitute_assay",
                     assay_used="H3K4me3 ChIP-seq",is_primary_assay="False",source_used="cistrome",genome_build="hg38",
                     identity_confidence="name_verified",contamination_status="clean",
                     model_confidence_flag="substitute: H3K4me3 for missing ATAC; rehabilitated (GSM838933='SK-MEL-28', MEL-745A tag artifact)",
                     sample_ids_used="43881",notes="SK-MEL-28 H3K4me3 (GSM838933) as ATAC proxy; MEL-745A tag disregarded as parser artifact.")
            else:  # H3K27ac -- no data
                emit(cell,mark,status="unresolved",resolution_tier="unresolved",
                     notes="No SK-MEL-28 H3K27ac data in ENCODE/Cistrome/EpiMap/ChIP-Atlas.")
            continue

        # ---- NPC family inheritance ----
        if cell in ("NPC.CAS9","NPC.TAK"):
            fin=npc_final(mark)
            flag=("possible_donor_difference (NPC.TAK maintained as distinct line from NPC; TAK not a documented perturbation)"
                  if cell=="NPC.TAK" else "pre_authorized (Cas9 transgene, chromatin-neutral)")
            if fin["drop"]:
                drop_audit.append(dict(cell=cell,mark=mark,dropped=",".join(fin["drop"]),reason="inherited-from-NPC contaminant removal (HK-1/NPC-HK1)"))
            emit(cell,mark,status="resolved",resolution_tier="related_line_inheritance",
                 assay_used=fin["assay"],is_primary_assay=str(fin["primary"]),source_used=fin["src"],
                 genome_build=fin["build"],identity_confidence="related_line",
                 contamination_status=("contaminant_samples_removed" if fin["drop"] else "clean"),
                 model_confidence_flag=flag,
                 sample_ids_used=",".join(fin["kept"]),sample_ids_dropped=",".join(fin["drop"]),
                 notes=f"Inherited from NPC/{mark} (cleaned).")
            continue

        # ---- ASC.C inheritance ----
        if cell=="ASC.C":
            if mark=="ATAC-seq":
                ids=ids_from_note(v6map[("ASC","ATAC-seq")]["notes"])
                emit(cell,mark,status="resolved",resolution_tier="related_line_inheritance",
                     assay_used=mark,is_primary_assay="True",source_used="cistrome",
                     genome_build=v6map[("ASC","ATAC-seq")]["genome_build"],identity_confidence="related_line",
                     contamination_status="clean",model_confidence_flag="lot_variant ('.C' = different lot of ASC; not a perturbation)",
                     sample_ids_used=",".join(ids),notes="Inherited from ASC/ATAC-seq (clean primary adipocyte, zero-CVCL).")
            else:
                emit(cell,mark,status="unresolved",notes="ASC has no resolved H3K27ac/H3K27me3 to inherit.")
            continue

        # ---- muscle tissue-type lines (rebuilt clean pool) ----
        if cell in MUSCLE_LINES:
            ids,build=MUSCLE[mark]
            via="SKL rebuilt clean skeletal-muscle pool (AT6BR LCL lymphoblastoid mis-tags excluded)"
            flag="tissue_type_only; '.C' lot variant" if cell=="SKL.C" else "tissue_type_only"
            emit(cell,mark,status="resolved",resolution_tier="tissue_type_match",
                 assay_used=mark,is_primary_assay="True",source_used="cistrome",genome_build=build,
                 identity_confidence="tissue_type_only",contamination_status="clean",
                 model_confidence_flag=flag,sample_ids_used=",".join(ids),
                 notes=f"Skeletal-muscle tissue-type match ({via}).")
            continue

        # ---- tissue-appropriate keep (HUES3/MCH58/NKDBA) ----
        if cell in TISSUE_KEEP and r6["v6_change"].startswith("RESOLVED"):
            ids=ids_from_note(notes)
            emit(cell,mark,status="resolved",resolution_tier="tissue_type_match",
                 assay_used=mark,is_primary_assay="True",source_used=src,genome_build=r6["genome_build"],
                 identity_confidence="tissue_type_only",contamination_status="clean",
                 model_confidence_flag="tissue_type_only",
                 sample_ids_used=",".join(ids) if ids else "",
                 notes=notes)
            continue

        # ---- ATAC substitute assay (fold in) for still-unresolved ATAC ----
        if mark=="ATAC-seq" and src=="unresolved" and cell in sub_ok:
            sd=sub_ok[cell]; ids=ids_from_note(sd["notes"])
            # decontaminate cistrome substitute pools using this cell's known genuine
            # contaminants across any mark (NPC -> HK-1/NPC-HK1); ENCODE = exact match, clean.
            known=set().union(*[CONTAM.get((cell,m),set()) for m in MARKS]) if sd["source"]=="cistrome" else set()
            dropped=[s for s in ids if (cvcls(s)&known)]
            kept=[s for s in ids if s not in dropped]
            if dropped:
                drop_audit.append(dict(cell=cell,mark=mark,dropped=",".join(dropped),
                                       reason="substitute-assay pool contaminant removal"))
            emit(cell,mark,status="resolved",resolution_tier="substitute_assay",
                 assay_used=sd["assay"],is_primary_assay="False",source_used=sd["source"],
                 genome_build=sd["build"],identity_confidence="name_verified",
                 contamination_status=("contaminant_samples_removed" if dropped else "clean"),
                 model_confidence_flag=f"substitute: {sd['assay']} used for missing ATAC-seq",
                 sample_ids_used=(",".join(kept) if kept else sd["notes"]),
                 sample_ids_dropped=",".join(dropped),notes=sd["notes"])
            continue

        # ---- the 12 GENUINE_CONTAMINATION: drop + re-aggregate ----
        if (cell,mark) in TWELVE:
            ids=ids_from_note(notes); kept,dropped=clean_pool(cell,mark,ids)
            if dropped:
                drop_audit.append(dict(cell=cell,mark=mark,dropped=",".join(dropped),
                                       reason="GENUINE_CONTAMINATION independent-line sample removal"))
            if not kept:
                emit(cell,mark,status="unresolved",notes="All samples contaminated (single dual-tagged sample); dropped -> unresolved.")
            else:
                mp=" MULTIPLEXED sample(s) among dropped (title names >1 line)." if (cell,mark) in {("HEK293T","ATAC-seq"),("MDAMB231","H3K27ac"),("MDAMB231","H3K27me3")} else ""
                lc=" LOW_CONFIDENCE_single_clean_sample" if len(kept)<=1 else ""
                emit(cell,mark,status="resolved",resolution_tier="direct_measurement",
                     assay_used=mark,is_primary_assay="True",source_used=src,genome_build=r6["genome_build"],
                     identity_confidence=("name_verified_low_confidence" if len(kept)<=1 else "name_verified"),
                     contamination_status="contaminant_samples_removed",
                     model_confidence_flag="decontaminated"+lc,sample_ids_used=",".join(kept),
                     sample_ids_dropped=",".join(dropped),
                     notes=f"Dominant identity correct; {len(dropped)} contaminant sample(s) dropped, {len(kept)} clean kept.{mp}")
            continue

        # ---- clean v5-carried resolved ----
        if src!="unresolved" and r6["v6_change"].startswith("unchanged"):
            emit(cell,mark,status="resolved",resolution_tier="direct_measurement",
                 assay_used=mark,is_primary_assay="True",source_used=src,genome_build=r6["genome_build"],
                 identity_confidence="name_verified",contamination_status="clean",
                 sample_ids_used=",".join(ids_from_note(notes)) if "sample ID" in notes else "",notes=notes)
            continue

        # ---- accepted EpiMap-imputed gap-fills (S3, operator-approved, name-verified biosample) ----
        if (cell,mark) in EPIMAP_IMPUTED:
            bid=EPIMAP_IMPUTED[(cell,mark)]
            emit(cell,mark,status="resolved",resolution_tier="imputed",
                 assay_used=mark,is_primary_assay="True",source_used="epimap",genome_build="hg19",
                 identity_confidence="imputed",contamination_status="imputed_track",
                 model_confidence_flag="imputed_track (EpiMap predicted, not measured; biosample name-verified)",
                 sample_ids_used=bid,notes=f"EpiMap imputed track, biosample {bid} [perturb=null, exact-name verified].")
            continue

        # ---- everything else: unresolved ----
        emit(cell,mark,status="unresolved",notes="No verified data via direct, substitute, or tissue-type matching.")

final=pd.DataFrame(rows)[["lincs_cell_id","assay_target","status","resolution_tier","assay_used",
    "is_primary_assay","source_used","genome_build","identity_confidence","contamination_status",
    "model_confidence_flag","sample_ids_used","sample_ids_dropped","notes"]]
final.to_csv("coverage_report_final.tsv",sep="\t",index=False)
pd.DataFrame(drop_audit).to_csv("dropped_contaminant_samples.tsv",sep="\t",index=False)

# ================= CONSISTENCY CHECKS =================
print("="*70); print("CONSISTENCY CHECKS"); print("="*70)
print("row count:",len(final)); assert len(final)==249,"not 249 rows"
assert final.duplicated(["lincs_cell_id","assay_target"]).sum()==0,"dup (cell,mark)"
print("duplicate (cell,mark):",int(final.duplicated(['lincs_cell_id','assay_target']).sum()))
assert (final["lincs_cell_id"].str.strip()!="").all(),"blank cell_id"
print("blank cell_id rows:",int((final['lincs_cell_id'].str.strip()=='').sum()))
badb=final[final["genome_build"].str.contains(",",na=False)]
print("comma-joined ambiguous builds:",len(badb)); assert len(badb)==0
assert final.groupby("lincs_cell_id")["assay_target"].nunique().eq(3).all(),"a cell lacks 3 marks"
print("all 83 cells have exactly 3 marks: True")
print("\n--- status counts ---");           print(final["status"].value_counts().to_string())
print("\n--- resolution_tier counts ---");  print(final["resolution_tier"].value_counts().to_string())
print("\n--- identity_confidence counts ---");print(final["identity_confidence"].value_counts().to_string())
print("\n--- contamination_status counts ---");print(final["contamination_status"].value_counts().to_string())
print("\nTotal contaminant samples dropped across all rows:",
      sum(len(d['dropped'].split(',')) for d in drop_audit))
print("Rows with a dropped-sample audit entry:",len(drop_audit))
print("\nWrote coverage_report_final.tsv (249 rows) + dropped_contaminant_samples.tsv")
