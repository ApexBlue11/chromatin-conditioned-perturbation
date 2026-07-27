# -*- coding: utf-8 -*-
"""
step13_finalize_tensor.py -- build E_final from E_peaks with an honest H3K27me3 quality floor:
keep H3K27me3 ONLY where it came from ENCODE broadPeak (broad-domain-appropriate); MASK the
ChIP-Atlas narrowPeak H3K27me3 slots (fragmented, unreliable for a broad mark). ATAC/H3K27ac
peaks unchanged. Outputs E_final.npy, E_final_mask.npy, E_final_provenance.json + validation.
Decision: bigWig coverage H3K27me3 deferred (bandwidth-bound ~3-5h on this connection).
"""
import sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
OUT="../outputs"; ROOT="../.."; MARKS=["ATAC-seq","H3K27ac","H3K27me3"]

E=np.load(f"{OUT}/E_peaks.npy").copy(); M=np.load(f"{OUT}/E_peaks_mask.npy").copy()
cell_index=json.load(open(f"{ROOT}/baseline/outputs/ccle_baseline_lincs_v5/lincs_cell_index.json"))["cell_id_to_row"]
cov=pd.read_csv(f"{OUT}/coverage_report_phase2.tsv",sep="\t",dtype=str,keep_default_na=False)

# H3K27me3 slots that are NOT ENCODE -> mask out of the repression channel (channel 2)
me3=cov[(cov.assay_target=="H3K27me3")&(cov.status=="resolved")]
dropped=[]
for _,r in me3.iterrows():
    if r.source_used!="encode" and r.lincs_cell_id in cell_index:
        ci=cell_index[r.lincs_cell_id]
        if M[ci,:,2].any():
            E[ci,:,2]=0.0; M[ci,:,2]=False; dropped.append(r.lincs_cell_id)
kept_me3=[c for c in cell_index if M[cell_index[c],:,2].any()]
print(f"H3K27me3 quality floor: kept {len(kept_me3)} ENCODE-broadPeak cells {sorted(kept_me3)}")
print(f"  masked {len(dropped)} fragmented ChIP-Atlas narrowPeak H3K27me3 cells")

np.save(f"{OUT}/E_final.npy",E); np.save(f"{OUT}/E_final_mask.npy",M)
def cov_cells(mi): return int(M[:,:,mi].any(axis=1).sum())
print("\n=== E_final ===", E.shape)
for mi,mk in enumerate(MARKS):
    feat="peaks(broadPeak)" if mi==2 else "peaks"
    print(f"  ch{mi} {mk:9} [{feat:16}] cells={cov_cells(mi)}")
print(f"  cells with >=1 mark: {int(M.any(axis=(1,2)).sum())}/83")

# biology validation
NC=E.shape[0]
ac=[ci for ci in range(NC) if M[ci,:,0].any() and M[ci,:,1].any() and (E[ci,:,0]>0).sum()>20 and (E[ci,:,1]>0).sum()>20]
if ac: print(f"\n  corr(ATAC, H3K27ac) over {len(ac)} cells = {np.nanmean([np.corrcoef(E[ci,:,0],E[ci,:,1])[0,1] for ci in ac]):+.3f} (expect +)")
both=[ci for ci in range(NC) if M[ci,:,1].any() and M[ci,:,2].any() and (E[ci,:,2]>0).sum()>20]
if both: print(f"  corr(H3K27ac, H3K27me3-broadPeak) over {len(both)} cells = {np.nanmean([np.corrcoef(E[ci,:,1],E[ci,:,2])[0,1] for ci in both]):+.3f} (expect -)")

json.dump({"shape":list(E.shape),"channels":{"0":"ATAC-seq/peaks","1":"H3K27ac/peaks","2":"H3K27me3/ENCODE-broadPeak-only"},
           "cells_per_mark":[cov_cells(0),cov_cells(1),cov_cells(2)],"cells_any":int(M.any(axis=(1,2)).sum()),
           "h3k27me3_masked_chipatlas_narrowpeak":sorted(dropped),
           "note":"bigWig-coverage H3K27me3 deferred (bandwidth-bound on this connection); reliability weights in E_reliability.tsv"},
          open(f"{OUT}/E_final_provenance.json","w"),indent=1)
print("\nwrote E_final.npy, E_final_mask.npy, E_final_provenance.json")
