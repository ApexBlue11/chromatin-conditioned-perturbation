# lincs-audit99-probe: sha1 and mounted-copy count of every data file XPertData reads via find(), plus the audit99 dataset.
# Outcome-free: hashes files only; reads no row, no prediction.
import glob, hashlib, json, os
NAMES = ['xpert_mdmt_splits.npz', 'drug_feature_index.json', 'drug_descriptors.npy', 'drug_fingerprints.npy',
         'drug_unimol.npy', 'drug_atom_reprs.npy', 'drug_atom_offsets.npy', 'lincs_cell_index.json', 'E_final.npy',
         'E_final_mask.npy', 'cell_lineage.npy', 'M_pathway_v9.npy', 'STRING_adj_978_v9.npy', 'gene_vectors_978.npy']
def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()
out = {}
for n in NAMES:
    hits = sorted(glob.glob('/kaggle/input/**/' + n, recursive=True))
    out[n] = {'copies': len(hits), 'paths': hits, 'sha1': [sha1(p) for p in hits]}
os.makedirs('/kaggle/working', exist_ok=True)
json.dump(out, open('/kaggle/working/audit99_data_pins.json', 'w'), indent=1)
print(json.dumps({n: (v['copies'], v['sha1']) for n, v in out.items()}))
