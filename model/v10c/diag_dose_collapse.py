"""Post-run diagnosis of a collapsed T-B arm (RESULTS 100.7): dose scale s, gate gamma and the drug output spread.

    python model/v10c/diag_dose_collapse.py ON_S0.pt ON_S1.pt OFF_S0.pt OUT.json
"""
import sys, json
import numpy as np, torch
sys.path.insert(0, 'model/v10c'); sys.path.insert(0, 'model/v9'); sys.path.insert(0, 'model')
torch.set_num_threads(4)
from data_v10c import V10CData, find
from model_v10c import LincsV10C, V10CConfig
roots = ['external/kaggle_v10c_src', 'external/v10c_full', 'external/lightning/v9payload/data']
out = {}
for tag, path, off in [('on_s0', sys.argv[1], False), ('on_s1', sys.argv[2], False), ('off_s0', sys.argv[3], True)]:
    D = V10CData(roots, split='cold_cell_1', dev=True, chrom_off=off)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['cfg'] if isinstance(ck['cfg'], V10CConfig) else V10CConfig(**ck['cfg'])
    m = LincsV10C(cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
                  np.load(find('gene_vectors_978.npy', roots)))
    Xc, Xm = D.fit_rows(); m.fit_bins(Xc, Xm)
    sd = {k.replace('module.', ''): v for k, v in ck['model'].items()}
    print(tag, m.load_state_dict(sd, strict=False))
    m.eval()
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(D.ev, 480, replace=False))
    rows_tr = np.sort(rng.choice(D.tr, 480, replace=False))
    rec = {}
    for nm, rr in (('dev', rows), ('train', rows_tr)):
        S, Gm, Dr = [], [], []
        with torch.no_grad():
            for i in range(0, len(rr), 48):
                o, aux = m(D.batch(rr[i:i + 48], 'cpu'), return_aux=True)
                S.append(aux['dose_scale'].numpy()); Gm.append(aux['gamma'].mean((1, 2)).numpy()); Dr.append(o['drug'].numpy())
        S, Gm, Dr = np.concatenate(S), np.concatenate(Gm), np.concatenate(Dr)
        rec[nm] = {'s_quantiles': np.quantile(S, [0, .1, .5, .9, 1]).round(6).tolist(), 's_eq_0_frac': float((S < 1e-6).mean()),
                   'gamma_mean': float(Gm.mean()), 'drug_out_sd_across_rows_per_gene_mean': float(Dr.std(0).mean())}
    out[tag] = rec
    print(tag, json.dumps(rec))
json.dump(out, open(sys.argv[4], 'w'), indent=1)
