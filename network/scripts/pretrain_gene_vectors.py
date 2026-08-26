# -*- coding: utf-8 -*-
"""
Pretrain gene vectors by LINK PREDICTION on the full-proteome STRING graph, then read off the 978
landmark rows -- the element V9_HANDOFF.md §C identifies as the substantive difference between our gene
representation (learned from scratch, 978 nodes) and the field's (pretrained, 19k nodes).

Why pretraining is the point. The 978 landmarks are 5 % of the proteome. A vector learned only from
landmark-to-landmark edges cannot encode that two landmarks are two steps apart THROUGH a non-landmark
gene, which is most of what STRING knows. Pretraining on all 19,496 nodes and then slicing gives each
landmark a position in the whole interactome.

DTI IS DELIBERATELY EXCLUDED. XPert's heterogeneous graph carries 12,890 drug-target edges. Ours does not,
because `drug/outputs/dti/dti_reference.tsv` is this project's HELD-OUT validation set: the one
interpretability probe we run -- does atom->gene attention land on a drug's real targets -- is only honest
while the model has never seen those edges. `--use_dti` exists so the trade can be measured rather than
argued about, and it writes to a separate file so the two can never be confused.

Method:
  * scoring: sigmoid(<z_u, z_v>) with edge confidence as the positive weight;
  * negatives: BOTH a uniform and a degree-matched sampler are scored, per method rule 6 (a readout's
    chance level is measured, not assumed). The expectation going in was that uniform would be the easier,
    inflated null; MEASURED IT IS THE HARDER ONE -- 0.8934 uniform against 0.9198 degree-matched. Uniform
    negatives are mostly low-degree nodes whose vectors are barely constrained by any edge, so their scores
    are high-variance rather than reliably low. Both numbers are recorded and neither is called "the"
    number;
  * held-out edges: 5 % never seen in training, scored against the same negative sampler.

Also reported, because graph reconstruction is not the thing we actually care about: whether the learned
vectors recover REACTOME/GO CO-MEMBERSHIP, which was never in the training objective. That is the check
that the vectors carry biology rather than topology alone.

    python network/scripts/pretrain_gene_vectors.py
"""
import os, json, argparse, time

import numpy as np
import torch

ROOT = r'C:\Projects\LINCS'
V9 = os.path.join(ROOT, 'network', 'outputs', 'v9')


def auc_ap(pos, neg):
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1); fpr = fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr, fpr))
    prec = tp / np.maximum(tp + fp, 1)
    ap = float((prec * y).sum() / max(y.sum(), 1))
    return auc, ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dim', type=int, default=128, help='XPert ships PPI_gene_vector_128d.npy')
    ap_.add_argument('--epochs', type=int, default=30)
    ap_.add_argument('--batch', type=int, default=200000)
    ap_.add_argument('--lr', type=float, default=0.02)
    ap_.add_argument('--neg', type=int, default=5)
    ap_.add_argument('--holdout', type=float, default=0.05)
    ap_.add_argument('--use_dti', action='store_true')
    ap_.add_argument('--seed', type=int, default=0)
    a = ap_.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    g = np.load(os.path.join(V9, 'string_graph_v9.npz'), allow_pickle=True)
    nodes = [str(x) for x in g['nodes']]
    E = g['edge_index'].astype(np.int64)
    W = g['weight'].astype(np.float32)
    lm_idx = g['landmark_idx'].astype(np.int64)
    N = len(nodes)
    print('graph: %d nodes, %d undirected edges, %d landmarks' % (N, E.shape[1], len(lm_idx)))
    tag = '_dti' if a.use_dti else ''
    if a.use_dti:
        raise SystemExit('--use_dti is a deliberate, separate experiment: DTI is this project\'s held-out '
                         'validation set. Wire it only with the probe consequences written down first.')

    perm = rng.permutation(E.shape[1])
    n_hold = int(a.holdout * E.shape[1])
    hold, train = perm[:n_hold], perm[n_hold:]
    Etr, Wtr = E[:, train], W[train]
    Eho = E[:, hold]
    print('train edges %d | held out %d' % (Etr.shape[1], Eho.shape[1]))

    # degree distribution of the TRAINING graph drives the degree-matched negative sampler
    deg = np.zeros(N, np.float64)
    np.add.at(deg, Etr[0], 1.0); np.add.at(deg, Etr[1], 1.0)
    p_deg = deg ** 0.75
    p_deg = p_deg / p_deg.sum()
    edge_set = set(map(int, Etr[0].astype(np.int64) * N + Etr[1]))
    edge_set |= set(map(int, Eho[0].astype(np.int64) * N + Eho[1]))

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z = torch.nn.Parameter(torch.randn(N, a.dim, device=dev) * 0.01)
    opt = torch.optim.Adam([Z], lr=a.lr)
    src = torch.from_numpy(Etr[0]).to(dev)
    dst = torch.from_numpy(Etr[1]).to(dev)
    wt = torch.from_numpy(Wtr).to(dev)
    p_t = torch.from_numpy(p_deg).float().to(dev)
    M = Etr.shape[1]
    for ep in range(a.epochs):
        idx = torch.randperm(M, device=dev)
        tot = 0.0
        for lo in range(0, M, a.batch):
            b = idx[lo:lo + a.batch]
            u, v, w = src[b], dst[b], wt[b]
            neg = torch.multinomial(p_t, len(b) * a.neg, replacement=True).view(len(b), a.neg)
            zu, zv, zn = Z[u], Z[v], Z[neg]
            pos = (zu * zv).sum(-1)
            negs = torch.einsum('bd,bkd->bk', zu, zn)
            loss = (w * torch.nn.functional.softplus(-pos)).mean() + \
                   torch.nn.functional.softplus(negs).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print('  epoch %2d  loss %.4f  %4.0fs' % (ep, tot / M, time.time() - t0), flush=True)

    Zn = Z.detach().cpu().numpy()

    # ---- link prediction on HELD-OUT edges, against two nulls ----
    def score(pairs):
        return (Zn[pairs[0]] * Zn[pairs[1]]).sum(1)

    n = Eho.shape[1]
    uni = np.stack([rng.integers(0, N, n), rng.integers(0, N, n)])
    dm = np.stack([rng.choice(N, n, p=p_deg), rng.choice(N, n, p=p_deg)])
    for name, neg in [('uniform', uni), ('degree-matched', dm)]:
        real = np.isin(neg[0].astype(np.int64) * N + neg[1], np.fromiter(edge_set, np.int64))
        neg = neg[:, ~real]                      # a sampled "negative" that is a real edge is not a negative
        auc, apv = auc_ap(score(Eho), score(neg))
        print('held-out link prediction vs %-14s AUC %.4f  AP %.4f' % (name, auc, apv))
        if name == 'uniform':
            auc_u, ap_u = auc, apv
        else:
            auc_d, ap_d = auc, apv

    # ---- does it recover pathway co-membership, which was never in the objective? ----
    M_path = np.load(os.path.join(V9, 'M_pathway_v9.npy'))
    Zl = Zn[lm_idx]
    Zl = Zl / (np.linalg.norm(Zl, axis=1, keepdims=True) + 1e-9)
    S = Zl @ Zl.T
    co = (M_path.T.astype(np.float32) @ M_path.astype(np.float32)) > 0
    iu = np.triu_indices(len(lm_idx), 1)
    same, diff = S[iu][co[iu]], S[iu][~co[iu]]
    auc_co, _ = auc_ap(same, diff[rng.choice(len(diff), min(len(diff), 400000), replace=False)])
    print('co-membership recovery (never trained on): cos same-pathway %.4f vs different %.4f, AUC %.4f'
          % (float(same.mean()), float(diff.mean()), auc_co))

    np.save(os.path.join(V9, 'gene_vectors_full%s.npy' % tag), Zn.astype(np.float32))
    np.save(os.path.join(V9, 'gene_vectors_978%s.npy' % tag), Zn[lm_idx].astype(np.float32))
    json.dump(dict(dim=a.dim, epochs=a.epochs, nodes=N, train_edges=int(Etr.shape[1]),
                   holdout_edges=int(Eho.shape[1]), use_dti=a.use_dti, seed=a.seed,
                   link_auc_uniform=round(auc_u, 4), link_ap_uniform=round(ap_u, 4),
                   link_auc_degree_matched=round(auc_d, 4), link_ap_degree_matched=round(ap_d, 4),
                   comembership_auc=round(auc_co, 4),
                   note='both nulls reported; measured uniform 0.8934 < degree-matched 0.9198, the opposite of the prior expectation, because uniform negatives are undertrained low-degree nodes',
                   seconds=round(time.time() - t0, 1)),
              open(os.path.join(V9, 'gene_vectors_provenance%s.json' % tag), 'w'), indent=2)
    print('\nwrote gene_vectors_978%s.npy [%d, %d] -> %s' % (tag, len(lm_idx), a.dim, V9))


if __name__ == '__main__':
    main()
