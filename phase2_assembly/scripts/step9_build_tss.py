# -*- coding: utf-8 -*-
"""
step9_build_tss.py -- build TSS coordinates for the 978 landmark genes.
MANE Select (GRCh38) primary, matched by Entrez. TSS = transcript 5' end (start if +, end if -).
Lifts to hg19 (pyliftover) for EpiMap tracks. Run with the .venv-epi python (has pyliftover).
Outputs -> phase2_assembly/outputs/tss_hg38.tsv, tss_hg19.tsv.
"""
import sys, io, gzip, csv, json, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
ROOT="../.."
LANDMARK=f"{ROOT}/baseline/Network Data/pathway_landmark_genes.txt"
GENE_INFO=f"{ROOT}/Data Info/GSE92742_Broad_LINCS_gene_info.txt/GSE92742_Broad_LINCS_gene_info.txt"
MANE="https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz"

syms=[l.strip() for l in open(LANDMARK,encoding="utf-8") if l.strip()]
sym2ent={}
with open(GENE_INFO,encoding="utf-8",newline="") as f:
    for row in csv.DictReader(f,delimiter="\t"): sym2ent[row["pr_gene_symbol"]]=row["pr_gene_id"]
ent=[sym2ent.get(s) for s in syms]
print(f"landmark genes: {len(syms)}; with entrez: {sum(e is not None for e in ent)}")

print("downloading MANE summary ...")
raw=urllib.request.urlopen(MANE,timeout=60).read()
txt=gzip.decompress(raw).decode("utf-8","replace")
rd=csv.DictReader(io.StringIO(txt),delimiter="\t")
cols=rd.fieldnames
def col(*cands):
    for c in cands:
        for f in cols:
            if f.lower().lstrip("#")==c.lower(): return f
    return None
c_gene=col("NCBI_GeneID","GeneID"); c_chr=col("GRCh38_chr","chr"); c_start=col("chr_start")
c_end=col("chr_end"); c_strand=col("chr_strand"); c_status=col("MANE_status")
print("MANE columns used:",c_gene,c_chr,c_start,c_end,c_strand,c_status)
def nc_to_chr(acc):
    # NC_000001.11 -> chr1 ; 23->chrX 24->chrY 12920->chrM
    m=acc.split(".")[0].replace("NC_","").lstrip("0") or "0"
    try: n=int(m)
    except: return acc
    if n==23: return "chrX"
    if n==24: return "chrY"
    if n==12920: return "chrM"
    return f"chr{n}"
ent2tss={}
for row in rd:
    gid=row[c_gene].replace("GeneID:","").strip()
    strand=row[c_strand]; chrom=nc_to_chr(row[c_chr])
    try: start=int(row[c_start]); end=int(row[c_end])
    except: continue
    tss = start if strand=="+" else end
    ent2tss[gid]=(chrom,tss,strand,row.get(c_status,""))

rows=[]; miss=[]
for s,e in zip(syms,ent):
    if e and e in ent2tss:
        chrom,tss,strand,st=ent2tss[e]
        rows.append((s,e,chrom,tss,strand))
    else:
        miss.append(s)
print(f"TSS resolved (MANE): {len(rows)}/{len(syms)}; unmapped={len(miss)} {miss[:15]}")
with open("../outputs/tss_hg38.tsv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["symbol","entrez","chrom","tss_hg38","strand"])
    for r in rows: w.writerow(r)

# liftover hg38 -> hg19 for EpiMap
from pyliftover import LiftOver
lo=LiftOver("hg38","hg19")
lifted=[]; lo_fail=0
for s,e,chrom,tss,strand in rows:
    r=lo.convert_coordinate(chrom, tss)
    if r: lifted.append((s,e,r[0][0],r[0][1],strand))
    else: lo_fail+=1
with open("../outputs/tss_hg19.tsv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["symbol","entrez","chrom","tss_hg19","strand"])
    for r in lifted: w.writerow(r)
print(f"lifted to hg19: {len(lifted)}; liftover failures: {lo_fail}")
print("wrote tss_hg38.tsv, tss_hg19.tsv")
