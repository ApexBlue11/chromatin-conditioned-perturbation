"""Tests for RESULTS 88 (W20 + PI glue): the within-strata null, the post-pathway readout, the guards, the strata, the
trainer's defaults. Toy data and tiny models only; CPU."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))

from probe_moa_88 import (score_within_strata, strength_quintiles, responsiveness_groups, check_arch_match,
                          post_delta)
from probe_moa_v9 import score
from model_v9 import LincsV9
from config_v9 import V9Config
from dp_seeding import seed_devices


def _two_strata_toy(D=60, P=20):
    """Stratum 0 ranks nodes 0-9 on top and has its positive there; stratum 1 the mirror. A within-stratum permutation
    can only ever hand a compound a top-half node (pct <= 0.45); an unrestricted one hands half of them a bottom node."""
    rng = np.random.default_rng(0)
    scores, allpos, strata = np.zeros((D, P)), [], np.zeros(D, int)
    for i in range(D):
        top = np.arange(10) if i < D // 2 else np.arange(10, 20)
        scores[i, top] = 10 + rng.random(10)
        scores[i, np.setdiff1d(np.arange(P), top)] = rng.random(10)
        allpos.append([int(rng.choice(top))])
        strata[i] = 0 if i < D // 2 else 1
    return scores, allpos, strata


def test_a_within_strata_stays_in_stratum():
    scores, allpos, strata = _two_strata_toy()
    r = score_within_strata(scores, allpos, strata, rng_seed=0, n_perm=200)
    assert r['null1s_mean'] <= 0.45 + 1e-12, r          # every permuted positive stayed in its own top half
    u = score(scores, allpos, np.ones(20, int), rng_seed=0, n_perm=200, n_size=2)
    assert u['null1_mean'] > 0.45, u                     # the unrestricted null crosses strata
    assert r['S'] == u['S']


def test_a_one_stratum_equals_unrestricted():
    rng = np.random.default_rng(42)
    scores = rng.random((60, 10))
    allpos = [[int(rng.integers(0, 10))] for _ in range(60)]
    r = score_within_strata(scores, allpos, np.zeros(60, int), rng_seed=7, n_perm=200)
    u = score(scores, allpos, np.ones(10, int), rng_seed=7, n_perm=200, n_size=2)
    assert r['null1s_mean'] == u['null1_mean'] and r['p_s'] == u['p']


def _tiny():
    torch.manual_seed(0)
    cfg = V9Config(d_model=16, d_pathway=8, post_pathway=True, expr_encoder='raw')
    M = np.zeros((5, 978), np.float32)
    for k in range(5):
        M[k, k * 50:(k + 1) * 50] = 1
    return LincsV9(cfg, M, None, None).eval(), cfg


def _batch(cfg, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {'x_ctl': torch.randn(B, 978, generator=g), 'x_cell': torch.randn(B, 978, generator=g),
            'E': torch.randn(B, 978, 3, generator=g), 'r': torch.ones(B, 978), 'E_mask': torch.ones(B, 978, 3),
            'cell_ctx': torch.randn(B, 16, generator=g), 'atoms': torch.randn(B, 4, cfg.d_atom, generator=g),
            'atom_mask': torch.ones(B, 4, dtype=torch.bool), 'u_feats': torch.randn(B, cfg.d_global, generator=g),
            'dose': torch.randn(B, generator=g), 'time': torch.randn(B, generator=g)}


def test_b_post_delta_readout():
    model, cfg = _tiny()
    b = _batch(cfg)
    with torch.no_grad():
        d_same, _, _ = post_delta(model, b, {k: v.clone() for k, v in b.items()})
        assert torch.equal(d_same, torch.zeros_like(d_same))          # drug inputs equal -> exactly 0
        b0 = {k: v.clone() for k, v in b.items()}
        b0['atoms'] = torch.randn_like(b['atoms']); b0['u_feats'] = torch.randn_like(b['u_feats'])
        d, _, _ = post_delta(model, b, b0)
        _, aux = model(b, return_aux=True)
        _, aux0 = model(b0, return_aux=True)
        hand = ((aux['post_pathway_activations'] - aux0['post_pathway_activations']) ** 2).sum(-1).sqrt()
        assert d.shape == (2, 5) and torch.allclose(d, hand, atol=1e-6)
        assert float(d.max()) > 1e-6                                   # the post layer IS drug-dependent
        pre = (aux['pathway_activations'] - aux0['pathway_activations']).abs().max()
        assert float(pre) == 0.0                                       # the pre-drug layer is drug-invariant


def _probe(*args):
    return subprocess.run([sys.executable, os.path.join(HERE, 'probe_moa_88.py')] + list(args),
                          capture_output=True, text=True, cwd=REPO)


def test_c_guards():
    r = _probe('--untrained', '--seed', '0')
    assert r.returncode != 0 and 'give --cfg_from with --untrained' in r.stderr
    tmp = tempfile.mkdtemp()
    p1 = os.path.join(tmp, 'nopost.pt')
    torch.save({'cfg': vars(V9Config(post_pathway=False)), 'tcfg': {'fold': 0}}, p1)
    r = _probe('--untrained', '--seed', '0', '--cfg_from', p1)
    assert r.returncode != 0 and 'post_pathway' in r.stderr, r.stderr[-400:]
    p2 = os.path.join(tmp, 'fold1.pt')
    torch.save({'cfg': vars(V9Config(post_pathway=True)), 'tcfg': {'fold': 1}}, p2)
    r = _probe('--untrained', '--seed', '0', '--cfg_from', p2)
    assert r.returncode != 0 and 'fold-0' in r.stderr, r.stderr[-400:]
    p3 = os.path.join(tmp, 'cut.pt')
    torch.save({'cfg': vars(V9Config(post_pathway=True)), 'tcfg': {'fold': 0}, 'epoch': 8}, p3)
    r = _probe('--ckpt', p3)
    assert r.returncode != 0 and '!= 11' in r.stderr, r.stderr[-400:]


def test_c_arch_match():
    a = {'w': torch.zeros(2, 3), 'b': torch.zeros(3)}
    check_arch_match(a, {'w': torch.ones(2, 3), 'b': torch.ones(3)})
    with pytest.raises(SystemExit):
        check_arch_match(a, {'w': torch.zeros(2, 3)})
    with pytest.raises(SystemExit):
        check_arch_match(a, {'w': torch.zeros(3, 2), 'b': torch.zeros(3)})


def test_d_responsiveness_groups():
    G = 978
    deltas = []
    for c in range(5):
        d = np.full((2, G), 0.01)
        d[:, 0] = 5.0                     # gene 0 is every compound's most responsive gene -> percentile 0
        deltas.append(d)
    target_idx = [[], [0], [1], [2], [3]]  # compound 0 has no landmark target
    vals, no_lm, resp, notresp = responsiveness_groups(deltas, target_idx)
    assert no_lm.tolist() == [True, False, False, False, False]
    assert vals[1] == 0.0
    assert not resp[0] and not notresp[0]
    assert abs(int(resp.sum()) - int(notresp.sum())) <= 1 and int(resp.sum() + notresp.sum()) == 4
    assert resp[1]                        # the most-responsive target is on the responsive side


def test_d_strength_quintiles():
    deltas = [np.full((3, 978), float(k + 1)) for k in range(10)]
    strength, strata, sha = strength_quintiles(deltas)
    assert strength.tolist() == [float(k + 1) for k in range(10)]
    assert np.bincount(strata, minlength=5).tolist() == [2, 2, 2, 2, 2]
    assert sha == strength_quintiles([d.copy() for d in deltas])[2]


def test_e_trainer_defaults_and_seed_list():
    from train_v9_gpu import build_parser
    a = build_parser().parse_args([])
    assert (a.dp_seed_mode, a.tf32, a.post_pathway) == ('lockstep', 'on', False)
    a = build_parser().parse_args(['--post_pathway', '--dp_seed_mode', 'distinct', '--tf32', 'off'])
    assert (a.dp_seed_mode, a.tf32, a.post_pathway) == ('distinct', 'off', True)
    assert seed_devices(42, 2, 'lockstep') == [42, 42]
    assert seed_devices(42, 2, 'distinct') == [42, 1042]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
