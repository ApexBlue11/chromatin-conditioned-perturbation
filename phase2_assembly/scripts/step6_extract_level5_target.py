# -*- coding: utf-8 -*-
"""
step6_extract_level5_target.py -- Phase-2 TASK A. Build the Y target matrix:
both-phases trt_cp signatures x 978 landmark genes (canonical order), from the Level 5 COMPZ GCTX.
Also emits the per-signature metadata table (sig_id, cell_id, pert_id, dose, time) and flags
restricted-SMILES signatures (candidate 252) for the operator to confirm before final exclusion.
NOTHING is excluded here beyond trt_cp filtering -- restricted rows are flagged, not dropped.
Outputs -> phase2_assembly/outputs/.
"""
import sys, os, json, glob, csv, time
import numpy as np, h5py
sys.stdout.reconfigure(encoding="utf-8")
t0=time.time()
ROOT="../.."
def di(pat): return glob.glob(f"{ROOT}/Data Info/{pat}/*.txt")[0]
GENE_INFO=f"{ROOT}/Data Info/GSE92742_Broad_LINCS_gene_info.txt/GSE92742_Broad_LINCS_gene_info.txt"
LANDMARK=f"{ROOT}/baseline/Network Data/pathway_landmark_genes.txt"
P1_SIG=di("GSE92742_Broad_LINCS_sig_info.txt"); P2_SIG=di("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt")
P1_PERT=di("GSE92742_Broad_LINCS_pert_info.txt"); P2_PERT=di("GSE70138_Broad_LINCS_pert_info_2017-03-06.txt")
P1_GCTX=f"{ROOT}/level5 data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
P2_GCTX=f"{ROOT}/level5 data/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx"
OUT="../outputs"

# --- 978 canonical landmark symbols (order matters; last line may lack newline) ---
syms=[l.strip() for l in open(LANDMARK,encoding="utf-8") if l.strip()]
print(f"canonical landmark symbols: {len(syms)} ({syms[0]}..{syms[-1]})")
# --- gene_info: symbol -> entrez ---
sym2ent={}
with open(GENE_INFO,encoding="utf-8",newline="") as f:
    r=csv.DictReader(f,delimiter="\t")
    for row in r: sym2ent[row["pr_gene_symbol"]]=row["pr_gene_id"]
ent=[sym2ent.get(s) for s in syms]
missing=[s for s,e in zip(syms,ent) if e is None]
print(f"symbols mapped to entrez: {sum(e is not None for e in ent)}/{len(syms)}; unmapped={missing}")

# --- restricted-SMILES drugs (missing/blank canonical_smiles) ---
def load_smiles(path):
    m={}
    with open(path,encoding="utf-8",newline="") as f:
        r=csv.DictReader(f,delimiter="\t")
        for row in r:
            sm=row.get("canonical_smiles","")
            m[row["pert_id"]]=sm
    return m
smiles={}; smiles.update(load_smiles(P1_PERT)); smiles.update(load_smiles(P2_PERT))
def is_restricted(pid):
    sm=smiles.get(pid,"")
    return (sm=="" or sm in ("-666","restricted","NA","null","-666.0"))

# --- collect trt_cp signatures per phase from sig_info ---
def load_sigs(path, phase):
    out=[]
    with open(path,encoding="utf-8",newline="") as f:
        r=csv.DictReader(f,delimiter="\t")
        for row in r:
            if row.get("pert_type")!="trt_cp": continue
            out.append(dict(sig_id=row["sig_id"],cell_id=row["cell_id"],pert_id=row["pert_id"],
                            pert_iname=row.get("pert_iname",""),
                            dose=row.get("pert_idose",row.get("pert_dose","")),
                            time=row.get("pert_itime",row.get("pert_time","")),phase=phase))
    return out
sigs=load_sigs(P1_SIG,"P1")+load_sigs(P2_SIG,"P2")
print(f"trt_cp signatures: {len(sigs):,}")
n_restr=sum(is_restricted(s['pert_id']) for s in sigs)
restr_drugs=sorted({s['pert_id'] for s in sigs if is_restricted(s['pert_id'])})
print(f"restricted-SMILES (missing canonical_smiles): {n_restr:,} sigs across {len(restr_drugs)} drugs")

# --- extract from a GCTX: map sig_id->col idx, entrez->row idx, pull 978 in canonical order ---
def extract(gctx, want_sigs):
    with h5py.File(gctx,"r") as h:
        col=[x.decode() if isinstance(x,bytes) else str(x) for x in h["0/META/COL/id"][:]]
        rowids=[x.decode() if isinstance(x,bytes) else str(x) for x in h["0/META/ROW/id"][:]]
        col_ix={c:i for i,c in enumerate(col)}
        row_ix={e:i for i,e in enumerate(rowids)}
        gene_rows=np.array([row_ix[e] for e in ent if e in row_ix])
        # signatures present in THIS gctx (sorted by col index for efficient h5 read)
        pres=[(s,col_ix[s["sig_id"]]) for s in want_sigs if s["sig_id"] in col_ix]
        pres.sort(key=lambda x:x[1])
        idx=np.array([ci for _,ci in pres])
        M=h["0/DATA/0/matrix"]
        Y=np.empty((len(idx),len(gene_rows)),dtype=np.float32)
        CH=4000
        for a in range(0,len(idx),CH):
            b=min(a+CH,len(idx))
            block=M[idx[a:b],:]            # (chunk, 12328)
            Y[a:b,:]=block[:,gene_rows]
        return [s for s,_ in pres], Y, len(gene_rows)
print("extracting Phase 1 ...")
s1,Y1,ng1=extract(P1_GCTX,[s for s in sigs if s["phase"]=="P1"]); print(f"  P1: {Y1.shape}  ({time.time()-t0:.0f}s)")
print("extracting Phase 2 ...")
s2,Y2,ng2=extract(P2_GCTX,[s for s in sigs if s["phase"]=="P2"]); print(f"  P2: {Y2.shape}  ({time.time()-t0:.0f}s)")
assert ng1==ng2==sum(e is not None for e in ent), "gene-row count mismatch between phases"

Y=np.concatenate([Y1,Y2],axis=0)
meta=s1+s2
os.makedirs(OUT,exist_ok=True)
np.save(f"{OUT}/Y_target_level5_978.npy", Y)
with open(f"{OUT}/signatures.tsv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["row","sig_id","cell_id","pert_id","pert_iname","dose","time","phase","restricted_smiles"])
    for i,s in enumerate(meta):
        w.writerow([i,s["sig_id"],s["cell_id"],s["pert_id"],s["pert_iname"],s["dose"],s["time"],s["phase"],int(is_restricted(s["pert_id"]))])
prov=dict(built=time.strftime("%Y-%m-%dT%H:%M:%S"), Y_shape=list(Y.shape), n_landmark=ng1,
          gene_order_source="pathway_landmark_genes.txt", unmapped_symbols=missing,
          n_trt_cp=len(meta), n_restricted_sigs=n_restr, restricted_drugs=restr_drugs,
          n_p1=len(s1), n_p2=len(s2), nan_count=int(np.isnan(Y).sum()))
json.dump(prov, open(f"{OUT}/target_provenance.json","w"), indent=1)
print(f"\nY_target_level5_978.npy: {Y.shape}  NaN={prov['nan_count']}  ({time.time()-t0:.0f}s)")
print(f"restricted-SMILES drugs ({len(restr_drugs)}): {restr_drugs}  -> {n_restr} sigs (FLAGGED not dropped)")
print("wrote Y_target_level5_978.npy, signatures.tsv, target_provenance.json")
