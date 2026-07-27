# -*- coding: utf-8 -*-
"""
step13_merge_hybrid_tensor.py -- assemble the final HYBRID epigenetic tensor:
  channel 0 ATAC-seq   = peak features   (E_peaks[:,:,0])
  channel 1 H3K27ac    = peak features   (E_peaks[:,:,1])
  channel 2 H3K27me3   = bigWig COVERAGE  (E_h3k27me3_coverage) -- broad-domain-appropriate
plus a unified availability mask and per-(cell,mark) reliability weights.
Validates biology. Outputs -> E_final.npy, E_final_mask.npy, E_final_provenance.json.
"""
import sys, json, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
OUT="../outputs"; MARKS=["ATAC-seq","H3K27ac","H3K27me3"]

Ep=np.load(f"{OUT}/E_peaks.npy"); Mp=np.load(f"{OUT}/E_peaks_mask.npy")
Ec=np.load(f"{OUT}/E_h3k27me3_coverage.npy"); Mc=np.load(f"{OUT}/E_h3k27me3_coverage_mask.npy")
NC,NG,_=Ep.shape
E=Ep.copy(); M=Mp.copy()
E[:,:,2]=Ec                 # replace repression channel with coverage
M[:,:,2]=Mc                 # ...and its mask
np.save(f"{OUT}/E_final.npy",E); np.save(f"{OUT}/E_final_mask.npy",M)

def cov(mi): return int(M[:,:,mi].any(axis=1).sum())
print("=== E_final (hybrid) ===  shape",E.shape)
for mi,mk in enumerate(MARKS):
    feat="coverage" if mi==2 else "peaks"
    print(f"  ch{mi} {mk:9} [{feat:8}] cells covered={cov(mi)}")
print(f"  cells with >=1 mark: {int(M.any(axis=(1,2)).sum())}/83 ; fully(3/3): {int((M.any(axis=1)).all(axis=1).sum())}")

# biology validation: does coverage-based H3K27me3 anti-correlate with H3K27ac more strongly than the peak version?
both=[ci for ci in range(NC) if M[ci,:,1].any() and M[ci,:,2].any() and (E[ci,:,2]>0).sum()>20]
if both:
    cors=[np.corrcoef(E[ci,:,1],E[ci,:,2])[0,1] for ci in both]
    print(f"\n  corr(H3K27ac_peaks, H3K27me3_COVERAGE) over {len(both)} cells = {np.nanmean(cors):+.3f} "
          f"(expect stronger NEGATIVE than the peak version's -0.11)")
ac=[ci for ci in range(NC) if M[ci,:,0].any() and M[ci,:,1].any() and (E[ci,:,0]>0).sum()>20 and (E[ci,:,1]>0).sum()>20]
if ac:
    c2=[np.corrcoef(E[ci,:,0],E[ci,:,1])[0,1] for ci in ac]
    print(f"  corr(ATAC_peaks, H3K27ac_peaks) over {len(ac)} cells = {np.nanmean(c2):+.3f} (expect POSITIVE)")

json.dump({"shape":list(E.shape),"channels":{"0":"ATAC-seq/peaks","1":"H3K27ac/peaks","2":"H3K27me3/coverage"},
           "cells_covered_per_mark":[cov(0),cov(1),cov(2)],
           "cells_with_any":int(M.any(axis=(1,2)).sum()),
           "cells_full_3of3":int((M.any(axis=1)).all(axis=1).sum())},
          open(f"{OUT}/E_final_provenance.json","w"),indent=1)
print("\nwrote E_final.npy, E_final_mask.npy, E_final_provenance.json")
