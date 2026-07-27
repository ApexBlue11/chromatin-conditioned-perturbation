# -*- coding: utf-8 -*-
"""
step12_h3k27me3_coverage.py -- REVAMP: rebuild the repression (H3K27me3) channel using bigWig
COVERAGE (broad-domain-appropriate), replacing the poor narrowPeak features. Sources:
ENCODE fold-change (GRCh38), Cistrome->ChIP-Atlas SRX bw (hg38), EpiMap imputed (hg19->hg19 TSS).
Window = TSS +/-10kb, binned mean (zoom, fast). Caps 3 samples/slot. Run with .venv-epi python.
Outputs -> E_h3k27me3_coverage.npy, mask, log.
"""
import sys, re, json, csv, urllib.request, time, os, tempfile
import numpy as np, pybigtools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True); t0=time.time()
ROOT="../.."; OUT="../outputs"; HALF=10000; MAXSAMP=1

def download(url):
    """download bigWig to a temp file; return local path or None."""
    try:
        fd,p=tempfile.mkstemp(suffix=".bw"); os.close(fd)
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"lincs"}),timeout=180) as r, open(p,"wb") as f:
            while True:
                b=r.read(1<<20)
                if not b: break
                f.write(b)
        return p
    except Exception:
        return None

cell_index=json.load(open(f"{ROOT}/baseline/outputs/ccle_baseline_lincs_v5/lincs_cell_index.json"))["cell_id_to_row"]
cells=[c for c,_ in sorted(cell_index.items(),key=lambda kv:kv[1])]; cidx={c:i for i,c in enumerate(cells)}
canon=[l.strip() for l in open(f"{ROOT}/baseline/Network Data/pathway_landmark_genes.txt",encoding="utf-8") if l.strip()]
NG=len(canon)
def load_tss(path,col):
    d={}
    for r in csv.DictReader(open(path,encoding="utf-8"),delimiter="\t"): d[r["symbol"]]=(r["chrom"],int(r[col]))
    return d
tss38=load_tss(f"{OUT}/tss_hg38.tsv","tss_hg38"); tss19=load_tss(f"{OUT}/tss_hg19.tsv","tss_hg19")

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

def http_json(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={"Accept":"application/json"}),timeout=30).read())
def encode_bw(acc):
    q=(f"https://www.encodeproject.org/search/?type=File&dataset=/experiments/{acc}/&file_format=bigWig"
       f"&output_type=fold+change+over+control&assembly=GRCh38&status=released&format=json&field=accession&limit=1")
    try:
        g=http_json(q)["@graph"]
        return [f"https://www.encodeproject.org/files/{g[0]['accession']}/@@download/{g[0]['accession']}.bigWig"] if g else []
    except Exception: return []

def read_cov(url, tssmap):
    """Per-gene small-window mean (one tiny read per gene; robust for long resumable runs)."""
    try: bw=pybigtools.open(url); chroms=set(bw.chroms().keys())
    except Exception: return None
    out=np.zeros(NG,dtype=np.float32)
    for gi,g in enumerate(canon):
        if g not in tssmap: continue
        ch,pos=tssmap[g]
        if ch not in chroms: continue
        try:
            v=bw.values(ch,max(0,pos-HALF),pos+HALF,bins=1,summary="mean",exact=False,fillna=0)
            out[gi]=float(np.asarray(v)[0])
        except Exception: pass
    return out

RAW=f"{OUT}/E_h3k27me3_coverage_RAW.npy"; RAWM=f"{OUT}/E_h3k27me3_coverage_mask.npy"
# --- resume: load partial RAW (pre-normalization) if present ---
if os.path.exists(RAW) and os.path.exists(RAWM):
    E=np.load(RAW); mask=np.load(RAWM); print(f"RESUMING from partial ({mask.any(axis=1).sum()} cells done)",flush=True)
else:
    E=np.zeros((len(cells),NG),dtype=np.float32); mask=np.zeros((len(cells),NG),dtype=bool)
log=[]
me3=cov[(cov.assay_target=="H3K27me3")&(cov.status=="resolved")]
print(f"H3K27me3 resolved slots to coverage-extract: {len(me3)}",flush=True)
done=0
for _,r in me3.iterrows():
    cell=r.lincs_cell_id
    if cell not in cidx: continue
    ci=cidx[cell]
    if mask[ci].any(): continue          # already done (resume)
    src=r.source_used; urls=[]; tssmap=tss38
    try:
        if src=="encode":
            for acc in re.findall(r"ENCSR\w+",r.notes)[:MAXSAMP]:
                urls+=encode_bw(acc)
        elif src=="cistrome":
            sids=[x for x in re.split(r"[,\s]+",r.sample_ids_used) if x.isdigit()]
            srx=[gsm2srx.get(id2gsm.get(s)) for s in sids]; srx=[x for x in srx if x][:MAXSAMP]
            urls=[f"https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/{x}.bw" for x in srx]
        elif src=="epimap":
            bss=re.search(r"(BSS\d+)",r.notes)
            if bss: urls=[f"https://epigenome.wustl.edu/epimap/data/imputed/impute_{bss.group(1)}_H3K27me3.bigWig"]; tssmap=tss19
    except Exception as e:
        log.append(f"{cell} url-resolve ERR {e}")
    cols=[]
    for u in urls[:MAXSAMP]:
        c=read_cov(u,tssmap)              # coarse per-chrom remote read (small data, few requests)
        if c is not None: cols.append(c)
    if not cols:
        log.append(f"{cell} src={src}: 0 bigWig ({time.time()-t0:.0f}s)"); print(log[-1],flush=True); continue
    v=np.mean(cols,axis=0); E[ci]=v
    for gi,g in enumerate(canon):
        if g in tssmap: mask[ci,gi]=True
    log.append(f"{cell} src={src}: {len(cols)} bw, nonzero={(v>0).sum()} ({time.time()-t0:.0f}s)")
    print(log[-1],flush=True)
    done+=1
    if done%3==0:                          # periodic checkpoint (resumable)
        np.save(RAW,E); np.save(RAWM,mask)

np.save(RAW,E); np.save(RAWM,mask)
# rank-normalize covered entries to [0,1] -> final tensor
Efinal=E.copy(); vals=Efinal[mask]
if vals.size:
    order=vals.argsort().argsort().astype(np.float32); Efinal[mask]=order/max(len(vals)-1,1)
np.save(f"{OUT}/E_h3k27me3_coverage.npy",Efinal)
open(f"{OUT}/E_h3k27me3_coverage_log.txt","w",encoding="utf-8").write("\n".join(log))
print(f"\nH3K27me3 coverage: cells covered={mask.any(axis=1).sum()} ({time.time()-t0:.0f}s)",flush=True)
print("wrote E_h3k27me3_coverage.npy, mask, log",flush=True)
