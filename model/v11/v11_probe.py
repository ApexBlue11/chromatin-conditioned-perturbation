"""v11's DTI probe (RESULTS 102 item 5, 102.1 C6): does atom -> gene attention localise to a compound's known targets when
those targets were never shown to the model? Reported, never a claim; run after every arm's runs exist.

  compounds  the probe compounds of dti_probe_split.json (their targets never entered v11's inputs, graph or labels) that
             have a target in dti_reference.tsv, in two row settings: 'heldout' (their dev-cell rows) and 'train' (their
             training rows: responses the model fitted, targets it never saw)
  signal     per landmark gene, the attention mass on the compound's ATOM tokens (the global token excluded), recomputed
             from each perturb block's cross-attention by a forward hook, averaged over heads, blocks and up to 6 of the
             compound's rows of the setting (as eval.py's atom_attn)
  metric     recall@k (k = 5, 10, 20, 50) of the compound's dti_reference targets among the 978 genes ranked by the signal,
             against chance k / 978; on all edges, the ChEMBL subset and the ChEMBL + STITCH subset (eval.py's subsets)
  bins       each probe compound's maximum ECFP4 Tanimoto to the known (annotated, allowed) training compounds:
             < 0.3, 0.3-0.5, >= 0.5 (102.1 C6), so analogue transfer through the structure head is visible
  models     every completed slot of v11-full and v11-targets-off, and T-B's off arm (v10-C) for the same compounds

    python model/v11/v11_probe.py --ckpts external/kaggle_out/v11_ckpt --tb_ckpts external/kaggle_out/v10c_tb \
        --out model/results/v11/v11_probe.json
"""
import argparse, csv, glob, json, math, os, sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
from data_v10c import V10CData, find  # noqa: E402
from model_v10c import LincsV10C, V10CConfig  # noqa: E402
from data_v11 import V11Data  # noqa: E402
from model_v11 import LincsV11  # noqa: E402

KS = (5, 10, 20, 50)
SUBSETS = ('all', 'chembl', 'both')
BINS = ((0.0, 0.3, '<0.3'), (0.3, 0.5, '0.3-0.5'), (0.5, 1.01, '>=0.5'))
MAX_ROWS = 6


def load_reference_targets(path, n_genes=978):
    """eval.py's parsing: pert_id -> set of landmark gene_idx, per evidence subset."""
    ref = {s: defaultdict(set) for s in SUBSETS}
    for row in csv.DictReader(open(path, encoding='utf-8'), delimiter='\t'):
        try:
            gi = int(row['gene_idx'])
        except (KeyError, ValueError):
            continue
        if not 0 <= gi < n_genes:
            continue
        ev = (row.get('evidence') or '').lower()
        chembl = ('chembl' in ev) or int(float(row.get('chembl_direct') or 0)) == 1 or bool((row.get('chembl_actions') or '').strip())
        ref['all'][row['pert_id']].add(gi)
        if chembl:
            ref['chembl'][row['pert_id']].add(gi)
        if ev.startswith('chembl+stitch'):
            ref['both'][row['pert_id']].add(gi)
    return ref


def recall_at_k(score, targets, ks=KS):
    order = np.argsort(-score, kind='stable')
    return {k: len(set(order[:k].tolist()) & targets) / len(targets) for k in ks}


def tanimoto_max(fp, query, pool):
    """Max Tanimoto of each query fingerprint row to the pool rows (binary fingerprints)."""
    Q, P = (fp[query] > 0).astype(np.float32), (fp[pool] > 0).astype(np.float32)
    inter = Q @ P.T
    union = Q.sum(1)[:, None] + P.sum(1)[None, :] - inter
    return (inter / np.maximum(union, 1)).max(1) if len(pool) else np.zeros(len(query))


def bin_of(t):
    for lo, hi, name in BINS:
        if lo <= t < hi:
            return name
    raise ValueError(t)


class AtomMass:
    """Forward hooks on every perturb block's CrossAttention: the attention of each gene query on the atom keys, summed
    over atoms, averaged over heads, recomputed exactly as CrossAttention.forward computes its weights."""

    def __init__(self, model):
        self.maps, self.handles = [], []
        for blk in model.perturb:
            self.handles.append(blk.cross.register_forward_hook(self.hook))

    def hook(self, mod, inp, out):
        x, mem, mem_mask = inp
        B, L, _ = x.shape
        M = mem.shape[1]
        k, _ = mod.kv(mem).chunk(2, dim=-1)
        q = mod.q_norm(mod.q(x).view(B, L, mod.h, mod.dh).transpose(1, 2))
        k = mod.k_norm(k.view(B, M, mod.h, mod.dh).transpose(1, 2))
        s = (q @ k.transpose(-1, -2)) / math.sqrt(mod.dh)
        s = s.masked_fill(mem_mask[:, None, None, :], float('-inf'))
        a = torch.softmax(s.float(), -1)                              # [B, H, L, M]
        self.maps.append(a[..., 1:].sum(-1).mean(1))                  # atom mass per gene, [B, L]

    def take(self):
        m = torch.stack(self.maps, 0).mean(0)                         # over blocks
        self.maps = []
        return m

    def close(self):
        for h in self.handles:
            h.remove()


def signals(model, D, rows_by_comp, device='cpu'):
    model.eval()
    am = AtomMass(model)
    out = {}
    try:
        with torch.no_grad():
            for p, rows in rows_by_comp.items():
                model(D.batch(rows, device))
                out[p] = am.take().mean(0).float().cpu().numpy()
    finally:
        am.close()
    return out


def score_model(sig, ref, bins):
    res = {}
    for sub in SUBSETS:
        per = defaultdict(lambda: {k: [] for k in KS})
        for p, s in sig.items():
            t = ref[sub].get(p)
            if not t:
                continue
            r = recall_at_k(s, t)
            for key in ('all_bins', bins[p]):
                for k in KS:
                    per[key][k].append(r[k])
        res[sub] = {b: {'n_compounds': len(v[KS[0]]), **{f'recall@{k}': float(np.mean(v[k])) for k in KS},
                        **{f'chance@{k}': k / 978 for k in KS}} for b, v in per.items()}
    return res


def probe_rows(D, probe, setting='heldout', max_rows=MAX_ROWS):
    """Each probe compound's rows of the setting (held-out or training; up to max_rows, in row order)."""
    pool = {'heldout': D.ev, 'train': D.tr}[setting]
    out = {}
    for p in sorted(probe):
        r = pool[D.pert[pool] == p]
        if len(r):
            out[p] = r[:max_rows]
    return out


def build_model(kind, ck, D, roots):
    cfg = ck['cfg'] if isinstance(ck['cfg'], V10CConfig) else V10CConfig(**ck['cfg'])     # as tb_read.mechanism
    args = (cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
            np.load(find('gene_vectors_978.npy', roots)))
    m = LincsV11(*args, D.fg_inputs()) if kind == 'v11' else LincsV10C(*args)
    m.fit_bins(*D.fit_rows())
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True, help='directory holding v11_<tag>.pt files (searched recursively)')
    ap.add_argument('--tb_ckpts', required=True, help='directory holding T-B off v10c_<tag>.pt files')
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- the probe is read once')
    torch.set_num_threads(4)
    roots = a.roots
    probe = set(json.load(open(find('dti_probe_split.json', roots), encoding='utf-8'))['probe'])
    ref = load_reference_targets(find('dti_reference.tsv', roots))
    D = V11Data(roots, dev=True, arm='full')
    settings = {s: {p: r for p, r in probe_rows(D, probe, s).items() if ref['all'].get(p)} for s in ('heldout', 'train')}
    scored = sorted(set(settings['heldout']) | set(settings['train']))
    di = json.load(open(find('drug_feature_index.json', roots), encoding='utf-8'))
    fp = np.load(find('drug_fingerprints.npy', roots))
    uq = np.unique(D.pert)
    known_rows = sorted({di[p] for p in uq[D.status == 0]})
    tmax = dict(zip(scored, tanimoto_max(fp, [di[p] for p in scored], known_rows)))
    bins = {p: bin_of(float(t)) for p, t in tmax.items()}
    rec = {'rule': 'RESULTS 102 item 5 / 102.1 C6: reported, never a claim', 'n_probe_compounds': len(probe),
           'n_scored': {s: len(r) for s, r in settings.items()},
           'n_by_bin': {s: {b: sum(bins[p] == b for p in r) for _, _, b in BINS} for s, r in settings.items()},
           'n_by_subset': {s: {sub: sum(1 for p in r if ref[sub].get(p)) for sub in SUBSETS} for s, r in settings.items()},
           'max_tanimoto': {p: float(t) for p, t in tmax.items()}, 'models': {}}

    def score_all(m, Dm):
        return {s: score_model(signals(m, Dm, r), ref, bins) for s, r in settings.items()}
    cache = {'full': D}
    for path in sorted(glob.glob(os.path.join(a.ckpts, '**', 'v11_*.pt'), recursive=True)):
        tag = os.path.basename(path)[4:-3]
        arm = tag.split('_v11')[1].split('_d')[0]
        if arm not in ('full', 'no_targets'):
            continue
        if arm not in cache:
            cache[arm] = V11Data(roots, dev=True, arm=arm)
        Da = cache[arm]
        m = build_model('v11', torch.load(path, map_location='cpu', weights_only=False), Da, roots)
        rec['models'][tag] = score_all(m, Da)
    Dref = V10CData(roots, dev=True, chrom_off=True)
    if not (np.array_equal(Dref.ev, D.ev) and np.array_equal(Dref.tr, D.tr) and np.array_equal(Dref.pert, D.pert)):
        raise SystemExit('FATAL: the reference data are not on v11\'s rows')
    for path in sorted(glob.glob(os.path.join(a.tb_ckpts, '**', 'v10c_cold_cell_1_dev_off_d*_s*.pt'), recursive=True)):
        tag = os.path.basename(path)[5:-3]
        m = build_model('v10c', torch.load(path, map_location='cpu', weights_only=False), Dref, roots)
        rec['models']['reference_' + tag] = score_all(m, Dref)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({t: {s: v[s]['all'].get('all_bins') for s in v} for t, v in rec['models'].items()}, indent=1))
    return rec


if __name__ == '__main__':
    main()
