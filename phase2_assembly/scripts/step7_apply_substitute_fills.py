# -*- coding: utf-8 -*-
"""
step7_apply_substitute_fills.py -- fold the 4 property-matched substitute fills (step5) into a
Phase-2 coverage file. Keeps the epigenetics branch untouched; writes coverage_report_phase2.tsv.
Confidence: NOMO1/H3K9ac = substitute_strong; the H3K4me3 fills = substitute_weak_proxy.
Flags SKMEL28/H3K27ac as REDUNDANT (same H3K4me3 sample already used for its ATAC slot).
"""
import sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

cov=pd.read_csv("../../epigenetics/outputs/coverage_report_final.tsv",sep="\t",dtype=str,keep_default_na=False)
sub=pd.read_csv("../outputs/substitute_sweep_candidates.tsv",sep="\t",dtype=str,keep_default_na=False)

TIER={"substitute_strong":"substitute (same-property, strong)","proxy_weak":"substitute (weak proxy)"}
applied=[]
for _,s in sub.iterrows():
    cell,mk=s.cell,s.mark
    m=(cov.lincs_cell_id==cell)&(cov.assay_target==mk)
    if not m.any() or cov.loc[m,"status"].iloc[0]!="unresolved": continue
    redundant = (cell=="SKMEL28" and mk=="H3K27ac")  # same H3K4me3 sample as its ATAC substitute
    flag=f"substitute:{s.substitute} for missing {mk} ({s.tier_name})"
    if redundant: flag+=" [REDUNDANT: same H3K4me3 sample as ATAC slot -> low independent value]"
    cov.loc[m,"status"]="resolved"
    cov.loc[m,"resolution_tier"]="substitute_assay"
    cov.loc[m,"assay_used"]=s.substitute
    cov.loc[m,"is_primary_assay"]="False"
    cov.loc[m,"source_used"]="cistrome"
    cov.loc[m,"identity_confidence"]="name_verified" if s.tier_name=="substitute_strong" else "substitute_weak_proxy"
    cov.loc[m,"model_confidence_flag"]=flag
    cov.loc[m,"sample_ids_used"]=s.ids
    cov.loc[m,"notes"]=f"Property-matched substitute ({s.substitute}) for {mk}; CVCL-verified."
    applied.append((cell,mk,s.substitute,s.tier_name,"REDUNDANT" if redundant else "ok"))

cov.to_csv("../outputs/coverage_report_phase2.tsv",sep="\t",index=False)
res=(cov.status=="resolved").sum()
print("Applied substitute fills:")
for a in applied: print("  ",a)
print(f"\ncoverage_report_phase2.tsv: resolved {res}/249, unresolved {249-res}")
per=cov.groupby("lincs_cell_id").apply(lambda g:(g.status=='resolved').sum(),include_groups=False)
print(f"per-cell: full={int((per==3).sum())} partial={int(((per>0)&(per<3)).sum())} zero={int((per==0).sum())}")
