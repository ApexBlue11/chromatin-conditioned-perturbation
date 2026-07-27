# -*- coding: utf-8 -*-
"""
step14_finalize_h3k27me3.py -- final repression channel: re-extract ALL H3K27me3 (ENCODE +
Cistrome-via-ChIP-Atlas, both narrowPeak) RAW at +/-10kb TSS windows, normalize the whole channel
together, merge into E_final = [ATAC peaks, H3K27ac peaks, H3K27me3 narrowPeak]. Quality-weight:
down-weight cells whose H3K27me3 signal is near-zero (poor ChIP). Updates E_reliability.tsv.
"""
import sys, re, json, csv, gzip, urllib.request, time
from bisect import bisect_left
from collections import defaultdict
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); t0=time.time()
ROOT="../.."; OUT="../outputs"; HALF=10000

cell_index=json.load(open(f"{ROOT}/baseline/outputs/ccle_baseline_lincs_v5/lincs_cell_index.json"))["cell_id_to_row"]
cells=[c for c,_ in sorted(cell_index.items(),key=lambda kv:kv[1])]; cidx=cell_index
canon=[l.strip() for l in open(f"{ROOT}/baseline/Network Data/pathway_landmark_genes.txt",encoding="utf-8") if l.strip()]
gidx={g:i for i,g in enumerate(canon)}; NG=len(canon)
tss={}
for r in csv.DictReader(open(f"{OUT}/tss_hg38.tsv",encoding="utf-8"),delimiter="\t"): tss[r["symbol"]]=(r["chrom"],int(r["tss_hg38"]))
cov=pd.read_csv(f"{OUT}/coverage_report_phase2.tsv",sep="\t",dtype=str,keep_default_na=False)
S=json.load(open(f"{ROOT}/epigenetics/data/cistrome_human_samples.json"))
id2gsm={str(s["id"]):s.get("external_id") for s in S if s.get("external_id_type")=="GEO"}
gsm2srx={}
for l in open(f"{OUT}/chip_atlas_human_epi.tab",encoding="utf-8"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=8:
        m=re.search(r"(GSM\d+)",p[7])
        if m: gsm2srx.setdefault(m.group(1),p[0])

def http(u,t=60): return urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"lincs"}),timeout=t).read()
def encode_narrowpeak(acc):
    q=(f"https://www.encodeproject.org/search/?type=File&dataset=/experiments/{acc}/&file_format=bed"
       f"&file_format_type=narrowPeak&assembly=GRCh38&status=released&output_type=replicated+peaks"
       f"&output_type=peaks&output_type=pseudoreplicated+peaks&format=json&field=accession&limit=1")
    try:
        g=json.loads(http(q,30))["@graph"]
        return [f"https://www.encodeproject.org/files/{g[0]['accession']}/@@download/{g[0]['accession']}.bed.gz"] if g else []
    except Exception: return []
def parse_bed(raw):
    txt=gzip.decompress(raw).decode("utf-8","replace") if raw[:2]==b"\x1f\x8b" else raw.decode("utf-8","replace")
    by=defaultdict(list)
    for ln in txt.splitlines():
        if not ln or ln[0]=="#": continue
        f=ln.split("\t")
        if len(f)<5: continue
        try:
            st,en=int(f[1]),int(f[2]); sig=float(f[6]) if (len(f)>6 and f[6] not in ("",".")) else float(f[4])
        except: continue
        by[f[0]].append((st,en,sig))
    return {c:([x[0] for x in sorted(v)],sorted(v)) for c,v in by.items()}
def wsig(bed,chrom,pos):
    ent=bed.get(chrom)
    if not ent: return 0.0
    starts,ivs=ent; lo,hi=pos-HALF,pos+HALF; best=0.0
    i0=bisect_left(starts,lo-1_000_000); i1=bisect_left(starts,hi+1)
    for j in range(max(0,i0),i1):
        st,en,s=ivs[j]
        if en>=lo and st<=hi and s>best: best=s
    return best

me3=cov[(cov.assay_target=="H3K27me3")&(cov.status=="resolved")]
raw=np.zeros((len(cells),NG),dtype=np.float32); mask=np.zeros((len(cells),NG),dtype=bool); nz={}
print(f"H3K27me3 slots: {len(me3)}")
for _,r in me3.iterrows():
    cell=r.lincs_cell_id
    if cell not in cidx: continue
    ci=cidx[cell]; beds=[]
    if r.source_used=="encode":
        for acc in re.findall(r"ENCSR\w+",r.notes)[:2]:
            for u in encode_narrowpeak(acc):
                try: beds.append(parse_bed(http(u)))
                except Exception: pass
    elif r.source_used=="cistrome":
        sids=[x for x in re.split(r"[,\s]+",r.sample_ids_used) if x.isdigit()]
        srx=[gsm2srx.get(id2gsm.get(s)) for s in sids]; srx=[x for x in srx if x][:6]
        for x in srx:
            try: beds.append(parse_bed(http(f"https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{x}.05.bed")))
            except Exception: pass
    if not beds: print(f"  {cell:9} {r.source_used}: 0 beds"); continue
    vec=np.zeros(NG,dtype=np.float32); n=0
    for bed in beds:
        vec+=np.array([wsig(bed,*tss[g]) if g in tss else 0.0 for g in canon],dtype=np.float32); n+=1
    vec/=max(n,1); raw[ci]=vec
    for gi,g in enumerate(canon):
        if g in tss: mask[ci,gi]=True
    nz[cell]=int((vec>0).sum())
    print(f"  {cell:9} {r.source_used}: {n} beds, nonzero={nz[cell]} ({time.time()-t0:.0f}s)")

# normalize whole channel together -> [0,1]
me3n=raw.copy(); vals=me3n[mask]
if vals.size: me3n[mask]=(vals.argsort().argsort().astype(np.float32))/max(len(vals)-1,1)

# merge into E_final
E=np.load(f"{OUT}/E_peaks.npy").copy(); M=np.load(f"{OUT}/E_peaks_mask.npy").copy()
E[:,:,2]=me3n; M[:,:,2]=mask
np.save(f"{OUT}/E_final.npy",E); np.save(f"{OUT}/E_final_mask.npy",M)

# reliability: base from coverage tier, but down-weight poor-quality H3K27me3 (nonzero<100)
rel=pd.read_csv(f"{OUT}/E_reliability.tsv",sep="\t",dtype=str,keep_default_na=False).set_index("cell_id")
for cell,z in nz.items():
    if z<100: rel.loc[cell,"H3K27me3"]=f"{float(rel.loc[cell,'H3K27me3'])*0.3:.2f}"  # poor ChIP -> down-weight
rel.reset_index().to_csv(f"{OUT}/E_reliability.tsv",sep="\t",index=False)

def cc(mi): return int(M[:,:,mi].any(axis=1).sum())
print(f"\n=== E_final {E.shape} ===  ATAC={cc(0)} H3K27ac={cc(1)} H3K27me3={cc(2)}  any={int(M.any(axis=(1,2)).sum())}/83")
both=[ci for ci in range(len(cells)) if M[ci,:,1].any() and M[ci,:,2].any() and (E[ci,:,2]>0).sum()>20]
if both: print(f"  corr(H3K27ac,H3K27me3) over {len(both)} = {np.nanmean([np.corrcoef(E[ci,:,1],E[ci,:,2])[0,1] for ci in both]):+.3f} (expect -)")
poor=[c for c,z in nz.items() if z<100]
print(f"  poor-quality H3K27me3 down-weighted (nonzero<100): {sorted(poor)}")
json.dump({"shape":list(E.shape),"channels":{"0":"ATAC/peaks","1":"H3K27ac/peaks","2":"H3K27me3/narrowPeak(TSS+/-10kb)"},
           "cells_per_mark":[cc(0),cc(1),cc(2)],"h3k27me3_poor_downweighted":sorted(poor)},
          open(f"{OUT}/E_final_provenance.json","w"),indent=1)
print("wrote E_final.npy, E_final_mask.npy, updated E_reliability.tsv, E_final_provenance.json")
