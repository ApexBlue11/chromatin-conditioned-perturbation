# -*- coding: utf-8 -*-
"""
download_chipatlas.py -- Stream ChIP-Atlas experimentList.tab (199 MB, public, kyushu-u
mirror), filter to human (hg19/hg38) ATAC-Seq / DNase-seq / H3K27ac / H3K27me3 experiments,
and write a compact subset chip_atlas_human_epi.tab. The full 199MB file is never stored.
"""
import sys, urllib.request, io
sys.stdout.reconfigure(encoding="utf-8")
URL="https://dbarchive.biosciencedbc.jp/kyushu-u/metadata/experimentList.tab"
KEEP_GENOME={"hg19","hg38"}
KEEP_ANTIGEN={"H3K27ac","H3K27me3","ATAC-Seq","DNase-seq"}
KEEP_CLASS={"ATAC-Seq","DNase-seq"}

req=urllib.request.Request(URL,headers={"Accept-Encoding":"identity"})
kept=0; total=0
with urllib.request.urlopen(req,timeout=120) as r, open("chip_atlas_human_epi.tab","w",encoding="utf-8") as out:
    buf=b""
    while True:
        chunk=r.read(1<<20)
        if not chunk: break
        buf+=chunk
        *lines,buf=buf.split(b"\n")
        for lb in lines:
            total+=1
            p=lb.decode("utf-8","replace").split("\t")
            if len(p)<6: continue
            genome,acl,ant=p[1],p[2],p[3]
            if genome not in KEEP_GENOME: continue
            if not (ant in KEEP_ANTIGEN or acl in KEEP_CLASS): continue
            # keep: id, genome, antigen_class, antigen, celltype_class, celltype, celltype_desc, title
            row=[p[0],p[1],p[2],p[3],p[4] if len(p)>4 else "",p[5] if len(p)>5 else "",
                 p[6] if len(p)>6 else "",p[8] if len(p)>8 else ""]
            out.write("\t".join(row)+"\n"); kept+=1
print(f"scanned {total} rows; kept {kept} human ATAC/DNase/H3K27ac/H3K27me3 rows")
print("wrote chip_atlas_human_epi.tab")
