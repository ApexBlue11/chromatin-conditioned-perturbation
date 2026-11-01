"""v11's structure head against a pure popularity ranking and a nearest-neighbour lookup (RESULTS 102.8, review 104 C1).
Reported, never a claim; registered in 102.8 before any of it was computed.

  compounds   v11_probe's probe compounds with a dti_reference.tsv target, in its two settings (held-out, training rows)
  head        v11-full's structure head per seed, from the compound's own features (u_feats -> w_u -> ln_u + type_drug ->
              target_head, as the forward pass computes it); sigmoid on the vocabulary landmarks, the rest last
              (v11_probe.head_on_landmarks); a compound's head recall is the mean over seeds
  popularity  one ranking for every compound, by each landmark's prior rate among the known compounds (vocab_prior)
  NN          each landmark scores the highest ECFP4 Tanimoto to a known compound whose ChEMBL targets (the head's own
              labels) include it; ties by popularity; landmarks no known compound targets rank last
  untrained   v11-targets-off's head, 102.7's comparator, reported
  recall@k    expected recall under uniformly random tie-breaking
  statistic   per-compound recall@10 (5 / 20 / 50 reported); head - popularity and head - NN with a paired compound
              bootstrap 95 % percentile CI (10,000 resamples, default_rng(0)); overall and per Tanimoto bin, both settings;
              the 'all' subset of record, 'chembl' descriptive
  self-checks the head ranked as the probe ranks reproduces v11_probe.json; popularity equals each -targets checkpoint's
              bias ranking; the compound sets and bins equal v11_probe.json's counts

    python model/v11/v11_probe_head.py --ckpts external/kaggle_out/v11 --roots <v11 roots> \\
        --probe_json model/results/v11/v11_probe.json --out model/results/v11/v11_probe_head.json
"""
import argparse, glob, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
from data_v10c import find  # noqa: E402
from data_v11 import V11Data  # noqa: E402
import v11_probe as vp  # noqa: E402

KS = vp.KS
N_BOOT = 10000
REF_TOL = 1e-9


def expected_recall(score, targets, ks=KS):
    """Expected recall@k under uniformly random tie-breaking: a target with a genes strictly above it and m genes tied with
    it (itself included) counts min(1, max(0, (k - a) / m))."""
    s = np.asarray(score, np.float64)
    t = np.array(sorted(targets), np.int64)
    above = np.array([(s > s[i]).sum() for i in t], np.float64)
    tied = np.array([(s == s[i]).sum() for i in t], np.float64)
    return {k: float(np.clip((k - above) / tied, 0.0, 1.0).mean()) for k in ks}


def lexi_score(primary, secondary):
    """One score ordered by primary, then secondary, whose ties are exactly the ties of both."""
    pairs = np.stack([np.asarray(primary, np.float64), np.asarray(secondary, np.float64)], 1)
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    return inv.reshape(-1).astype(np.float64)


def landmark_prior(D):
    """vocab_prior on the 978 landmarks; -inf outside the head's vocabulary."""
    vpos = {int(n): i for i, n in enumerate(D.vocab)}
    return np.array([float(D.vocab_prior[vpos[int(n)]]) if int(n) in vpos else -np.inf for n in D.landmark_idx])


def known_target_matrix(D, di, known_rows):
    """[len(known_rows), 978] 1 where a known compound on that fingerprint row has the landmark as a ChEMBL target (its
    head labels; several known perts on one row are unioned)."""
    lm = {int(n): i for i, n in enumerate(D.landmark_idx)}
    rpos = {r: i for i, r in enumerate(known_rows)}
    K = np.zeros((len(known_rows), len(D.landmark_idx)), bool)
    uq = np.unique(D.pert)
    for k in np.flatnonzero(D.status == 0):
        for n in D.tgt_node[k][:D.tgt_count[k]]:
            if int(n) in lm:
                K[rpos[di[uq[k]]], lm[int(n)]] = True
    return K


def tanimoto_matrix(fp, query, pool):
    """Tanimoto of each query fingerprint row to each pool row (binary fingerprints), as v11_probe.tanimoto_max."""
    Q, P = (fp[query] > 0).astype(np.float32), (fp[pool] > 0).astype(np.float32)
    inter = Q @ P.T
    union = Q.sum(1)[:, None] + P.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


def nn_scores(sims, K, prior_lm):
    """Each landmark: the highest Tanimoto of a known compound targeting it (-1 if none), then popularity."""
    best = np.where(K, np.asarray(sims, np.float64)[:, None], -1.0).max(0)
    return lexi_score(best, np.where(np.isfinite(prior_lm), prior_lm, 0.0))


def head_logits(model, D, rows_by_comp):
    """Per compound: target_head(ln_u(w_u(u_feats)) + type_drug), exactly the forward pass's head (it uses no row input
    other than the compound's u_feats)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for p, rows in rows_by_comp.items():
            u = torch.as_tensor(np.ascontiguousarray(D.u_tab[D.u_inv[np.asarray(rows)[:1]]]))
            out[p] = model.target_head(model.ln_u(model.w_u(u)) + model.type_drug)[0].float().cpu().numpy()
    return out


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=0):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    m = d[np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {'n': int(len(d)), 'mean_diff': float(d.mean()),
            'ci95': [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}


def verdict_popularity(cmp):
    return ("the head ranks held-out compounds' targets above popularity" if cmp['ci95'][0] > 0
            else 'not established: the head is not shown above popularity')


def verdict_nn(cmp):
    if cmp['ci95'][0] > 0:
        return 'the head transfers beyond nearest-neighbour lookup'
    if cmp['ci95'][1] < 0:
        return 'a nearest-neighbour lookup does better than the head'
    return 'the head and a nearest-neighbour lookup are not distinguished'


def arm_of(path):
    tag = os.path.basename(path)[4:-3]
    return tag, tag.split('_v11')[1].split('_d')[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--probe_json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_boot', type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- read once')
    torch.set_num_threads(4)
    roots = a.roots
    probe_ref = json.load(open(a.probe_json, encoding='utf-8'))
    probe = set(json.load(open(find('dti_probe_split.json', roots), encoding='utf-8'))['probe'])
    ref = vp.load_reference_targets(find('dti_reference.tsv', roots))
    D = V11Data(roots, dev=True, arm='full')
    settings = {s: {p: r for p, r in vp.probe_rows(D, probe, s).items() if ref['all'].get(p)} for s in ('heldout', 'train')}
    scored = sorted(set(settings['heldout']) | set(settings['train']))
    di = json.load(open(find('drug_feature_index.json', roots), encoding='utf-8'))
    fp = np.load(find('drug_fingerprints.npy', roots))
    uq = np.unique(D.pert)
    known_rows = sorted({di[p] for p in uq[D.status == 0]})
    tmax = dict(zip(scored, vp.tanimoto_max(fp, [di[p] for p in scored], known_rows)))
    bins = {p: vp.bin_of(float(t)) for p, t in tmax.items()}
    checks = {'n_scored': {s: len(r) for s, r in settings.items()} == probe_ref['n_scored'],
              'n_by_bin': {s: {b: sum(bins[p] == b for p in r) for _, _, b in vp.BINS} for s, r in settings.items()}
              == probe_ref['n_by_bin']}
    prior_lm = landmark_prior(D)
    K = known_target_matrix(D, di, known_rows)
    sims = dict(zip(scored, tanimoto_matrix(fp, [di[p] for p in scored], known_rows)))
    pop = prior_lm
    nn = {p: nn_scores(sims[p], K, prior_lm) for p in scored}

    every = {p: settings['heldout'].get(p, settings['train'].get(p)) for p in scored}
    heads = {'full': {}, 'no_targets': {}}
    q = np.clip(D.vocab_prior.astype(np.float64), 1e-4, 1 - 1e-4)
    prior_logit = np.log(q / (1 - q))
    Dnt = None
    for path in sorted(glob.glob(os.path.join(a.ckpts, '**', 'v11_*.pt'), recursive=True)):
        tag, arm = arm_of(path)
        if arm not in heads:
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if arm == 'no_targets':
            if Dnt is None:
                Dnt = V11Data(roots, dev=True, arm='no_targets')
            bias = ck['model'].get('target_head.bias', ck['model'].get('module.target_head.bias')).double().numpy()
            checks[f'popularity_equals_bias_{tag}'] = bool(np.allclose(bias, prior_logit, atol=1e-5))
            m = vp.build_model('v11', ck, Dnt, roots)
            heads[arm][tag] = {p: vp.head_on_landmarks(lg, Dnt) for p, lg in head_logits(m, Dnt, every).items()}
        else:
            m = vp.build_model('v11', ck, D, roots)
            heads[arm][tag] = {p: vp.head_on_landmarks(lg, D) for p, lg in head_logits(m, D, every).items()}

    # self-check: ranked as the probe ranks, every head reproduces v11_probe.json
    worst = 0.0
    for arm, by_tag in heads.items():
        for tag, sc in by_tag.items():
            for s, rows in settings.items():
                got = vp.score_model({p: sc[p] for p in rows}, ref, bins)
                want = probe_ref['models'][tag][s + '_head']
                for sub, bb in want.items():
                    for b, d in bb.items():
                        for k in KS:
                            worst = max(worst, abs(got[sub][b][f'recall@{k}'] - d[f'recall@{k}']))
    checks['head_reproduces_probe_max_abs_diff'] = worst
    checks['head_reproduces_probe'] = worst <= REF_TOL
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise SystemExit(f'FATAL: a self-check failed {checks}')

    # per-compound recall
    per = {}
    for p in scored:
        rec = {'bin': bins[p], 'max_tanimoto': float(tmax[p]), 'settings': [s for s in settings if p in settings[s]]}
        for sub in vp.SUBSETS:
            t = ref[sub].get(p)
            if not t:
                continue
            r = {'n_targets': len(t), 'popularity': expected_recall(pop, t), 'nn': expected_recall(nn[p], t)}
            for arm, by_tag in heads.items():
                r[arm] = {tag: expected_recall(sc[p], t) for tag, sc in by_tag.items()}
            r['head'] = {k: float(np.mean([v[k] for v in r['full'].values()])) for k in KS}
            r['untrained'] = {k: float(np.mean([v[k] for v in r['no_targets'].values()])) for k in KS}
            rec[sub] = r
        per[p] = rec

    def summary(s, sub, b=None):
        ps = [p for p in settings[s] if sub in per[p] and (b is None or per[p]['bin'] == b)]
        if not ps:
            return None
        out = {'n_compounds': len(ps)}
        for meth in ('head', 'popularity', 'nn', 'untrained'):
            out[meth] = {f'recall@{k}': float(np.mean([per[p][sub][meth][k] for p in ps])) for k in KS}
        out['head_per_seed@10'] = {tag: float(np.mean([per[p][sub]['full'][tag][10] for p in ps])) for tag in heads['full']}
        out['untrained_per_seed@10'] = {tag: float(np.mean([per[p][sub]['no_targets'][tag][10] for p in ps]))
                                        for tag in heads['no_targets']}
        for base in ('popularity', 'nn'):
            out[f'head_minus_{base}@10'] = paired_bootstrap([per[p][sub]['head'][10] for p in ps],
                                                            [per[p][sub][base][10] for p in ps], a.n_boot)
        return out

    res = {s: {sub: {'all_bins': summary(s, sub), **{b: summary(s, sub, b) for _, _, b in vp.BINS}}
               for sub in ('all', 'chembl')} for s in settings}
    of_record = res['heldout']['all']['all_bins']
    rec = {'rule': 'RESULTS 102.8 (review 104 C1): reported, never a claim; registered before computing',
           'self_checks': checks, 'n_known_rows': len(known_rows),
           'verdicts': {'popularity': verdict_popularity(of_record['head_minus_popularity@10']),
                        'nn': verdict_nn(of_record['head_minus_nn@10'])},
           'results': res, 'per_compound': per}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'self_checks': checks, 'verdicts': rec['verdicts'],
                      'heldout_all': {k: v for k, v in of_record.items() if k != 'n_compounds'}}, indent=1))
    return rec


if __name__ == '__main__':
    main()
