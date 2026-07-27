# -*- coding: utf-8 -*-
"""
step10_extract_peak_tensor.py -- build the PEAK-based epigenetic feature tensor.
E_peaks[cell, gene, mark] = max peak signal overlapping the mark-specific TSS window,
averaged across the slot's samples. Sources: ENCODE (narrow/broadPeak, GRCh38) +
Cistrome-via-ChIP-Atlas SRX bed (hg38). EpiMap (imputed, no peaks) -> masked in the peak tensor.
Windows: ATAC/H3K27ac +/-2kb, H3K27me3 +/-10kb. Also emits mask + reliability weights.
Run with .venv-epi python. Outputs -> phase2_assembly/outputs/.
"""
import sys, os, re, json, csv, gzip, io, time, urllib.request
from bisect import bisect_left
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding="utf-8"); t0=time.time()
ROOT="../.."; OUT="../outputs"

MARKS=["ATAC-seq","H3K27ac","H3K27me3"]
WIN={"ATAC-seq":2000,"H3K27ac":2000,"H3K27me3":10000}
RELI={"direct_measurement":1.0,"substitute_assay":0.6,"tissue_type_match":0.4,
      "related_line_inheritance":0.7,"imputed":0.5}

# --- cells (83, baseline order for cross-branch alignment) ---
cell_index=json.load(open(f"{ROOT}/baseline/outputs/ccle_baseline_lincs_v5/lincs_cell_index.json"))["cell_id_to_row"]
cells=[c for c,_ in sorted(cell_index.items(), key=lambda kv: kv[1])]
cidx={c:i for i,c in enumerate(cells)}; NC=len(cells)
# --- genes (978 canonical order); TSS in hg38 ---
canon=[l.strip() for l in open(f"{ROOT}/baseline/Network Data/pathway_landmark_genes.txt",encoding="utf-8") if l.strip()]
gidx={g:i for i,g in enumerate(canon)}; NG=len(canon)
tss={}
with open(f"{OUT}/tss_hg38.tsv",encoding="utf-8") as f:
    for r in csv.DictReader(f,delimiter="\t"):
        if r["symbol"] in gidx: tss[r["symbol"]]=(r["chrom"],int(r["tss_hg38"]))
print(f"cells={NC} genes={NG} tss_resolved={len(tss)}")

# --- coverage report + Cistrome id->GSM + ChIP-Atlas GSM->SRX ---
import pandas as pd
cov=pd.read_csv(f"{OUT}/coverage_report_phase2.tsv",sep="\t",dtype=str,keep_default_na=False)
S=json.load(open(f"{ROOT}/epigenetics/data/cistrome_human_samples.json"))
id2gsm={str(s["id"]):s.get("external_id") for s in S if s.get("external_id_type")=="GEO"}
gsm2srx={}
for l in open(f"{OUT}/chip_atlas_human_epi.tab",encoding="utf-8"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=8:
        m=re.search(r"(GSM\d+)",p[7])
        if m: gsm2srx.setdefault(m.group(1),p[0])

def http_get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"lincs"}),timeout=timeout).read()

def encode_peak_urls(exp_acc, mark):
    ptype="broadPeak" if mark=="H3K27me3" else "narrowPeak"
    q=(f"https://www.encodeproject.org/search/?type=File&dataset=/experiments/{exp_acc}/"
       f"&file_format=bed&file_format_type={ptype}&assembly=GRCh38&status=released"
       f"&output_type=replicated+peaks&output_type=peaks&output_type=pseudoreplicated+peaks"
       f"&format=json&field=accession&limit=3")
    try:
        d=json.loads(http_get(q,30)); accs=[g["accession"] for g in d.get("@graph",[])][:1]
        return [f"https://www.encodeproject.org/files/{a}/@@download/{a}.bed.gz" for a in accs]
    except Exception: return []

def parse_bed(raw):
    """narrowPeak/broadPeak (ENCODE + ChIP-Atlas): signalValue = col 7 (index 6), fallback col 5.
    return {chrom: (starts_array, records[(start,end,sig)])} sorted by start."""
    txt=gzip.decompress(raw).decode("utf-8","replace") if raw[:2]==b"\x1f\x8b" else raw.decode("utf-8","replace")
    by=defaultdict(list)
    for ln in txt.splitlines():
        if not ln or ln[0]=="#": continue
        f=ln.split("\t")
        if len(f)<5: continue
        try:
            st,en=int(f[1]),int(f[2])
            sig=float(f[6]) if (len(f)>6 and f[6] not in ("",".")) else float(f[4])
        except: continue
        by[f[0]].append((st,en,sig))
    out={}
    for c,ivs in by.items():
        ivs.sort(); out[c]=([x[0] for x in ivs], ivs)
    return out

def window_signal(bed, chrom, tss_pos, half):
    """max signalValue among peaks overlapping [tss-half, tss+half]."""
    ent=bed.get(chrom)
    if not ent: return 0.0
    starts,ivs=ent; lo,hi=tss_pos-half,tss_pos+half
    i0=bisect_left(starts, lo-1_000_000); i1=bisect_left(starts, hi+1)
    best=0.0
    for j in range(max(0,i0), i1):
        st,en,sig=ivs[j]
        if en>=lo and st<=hi and sig>best: best=sig
    return best

E=np.zeros((NC,NG,3),dtype=np.float32); mask=np.zeros((NC,NG,3),dtype=bool); reli=np.zeros((NC,3),dtype=np.float32)
log=[]
resolved=cov[cov.status=="resolved"]
for _,r in resolved.iterrows():
    cell,mark=r.lincs_cell_id,r.assay_target
    if cell not in cidx or mark not in MARKS: continue
    ci,mi=cidx[cell],MARKS.index(mark); half=WIN[mark]
    src=r.source_used; beds=[]
    try:
        if src=="encode":
            for acc in re.findall(r"ENCSR\w+", r.notes)[:4]:
                for u in encode_peak_urls(acc,mark):
                    try: beds.append(parse_bed(http_get(u)))
                    except Exception: pass
        elif src=="cistrome":
            sids=[x for x in re.split(r"[,\s]+", r.sample_ids_used) if x.isdigit()]
            srxs=[gsm2srx.get(id2gsm.get(s)) for s in sids]
            srxs=[x for x in srxs if x][:6]
            for srx in srxs:
                u=f"https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{srx}.05.bed"
                try: beds.append(parse_bed(http_get(u)))
                except Exception: pass
        # epimap -> no peaks; leave masked
    except Exception as e:
        log.append(f"{cell}/{mark} ERR {e}")
    if not beds:
        log.append(f"{cell}/{mark} src={src}: 0 bed tracks (masked)"); continue
    # aggregate: per gene, avg over tracks of (max peak signal in window)
    vec=np.zeros(NG,dtype=np.float32); cnt=0
    for bed in beds:
        col=np.array([window_signal(bed,*tss[g],half) if g in tss else 0.0 for g in canon],dtype=np.float32)
        vec+=col; cnt+=1
    vec/= max(cnt,1)
    E[ci,:,mi]=vec
    for gi,g in enumerate(canon):
        if g in tss: mask[ci,gi,mi]=True
    reli[ci,mi]=RELI.get(r.resolution_tier,0.5)
    log.append(f"{cell}/{mark} src={src}: {cnt} bed track(s), nonzero_genes={(vec>0).sum()}  ({time.time()-t0:.0f}s)")
    print(log[-1])

# per-mark quantile-ish normalization (rank-based to [0,1]) over covered entries only
for mi in range(3):
    m=mask[:,:,mi]; vals=E[:,:,mi][m]
    if vals.size:
        order=vals.argsort().argsort().astype(np.float32)
        E[:,:,mi][m]=(order/max(len(vals)-1,1))
np.save(f"{OUT}/E_peaks.npy",E); np.save(f"{OUT}/E_peaks_mask.npy",mask)
with open(f"{OUT}/E_reliability.tsv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["cell_id"]+MARKS)
    for c in cells: w.writerow([c]+[f"{reli[cidx[c],mi]:.2f}" for mi in range(3)])
json.dump({"cells":NC,"genes":NG,"marks":MARKS,"windows":WIN,
           "slots_with_peaks":int((mask.any(axis=1)).sum()),
           "coverage_per_mark":[int(mask[:,:,mi].any(axis=1).sum()) for mi in range(3)]},
          open(f"{OUT}/E_peaks_provenance.json","w"),indent=1)
open(f"{OUT}/E_peaks_log.txt","w",encoding="utf-8").write("\n".join(log))
print(f"\nE_peaks.npy {E.shape}; slots with peaks (cell,mark): {(mask.any(axis=1)).sum()}  ({time.time()-t0:.0f}s)")
print("per-mark cells covered:", [int(mask[:,:,mi].any(axis=1).sum()) for mi in range(3)])
print("wrote E_peaks.npy, E_peaks_mask.npy, E_reliability.tsv, E_peaks_provenance.json, E_peaks_log.txt")
