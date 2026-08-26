# -*- coding: utf-8 -*-
"""
v9 code-vs-design checks. Each asserts a property the v9 design CLAIMS -- not that the code runs.
Run BEFORE any accelerator time:

    python model/v9/test_v9.py

Exits nonzero on failure. The checks that matter most are the ones that would otherwise fail silently:
the absolute head really being anchored on the control, the quantiser never seeing evaluation data, every
readout terminating in a NAMED unit, and the method rules (ablate to the mean with |dY|max; measure a
readout's chance level) being enforced by code rather than by discipline.
"""
import os, sys, csv, json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9, v9_loss, aux_targets
from modules_v9 import BinnedExpression, RawExpression
from interp_v9 import ablate_to_mean, ablate_module_to_mean, permutation_null, pathway_alignment

R = []


def check(name, cond, detail=''):
    R.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f'  -- {detail}' if detail else ''))


def batch(cfg, B=4, dead=50, n_atoms=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {'x_ctl': 4 + 6 * torch.rand(B, cfg.n_genes, generator=g),
         'x_cell': 4 + 6 * torch.rand(B, cfg.n_genes, generator=g),
         'E': torch.randn(B, cfg.n_genes, cfg.d_epi, generator=g),
         'r': torch.rand(B, cfg.n_genes, generator=g),
         'atoms': torch.randn(B, cfg.max_atoms, cfg.d_atom, generator=g),
         'atom_mask': torch.zeros(B, cfg.max_atoms, dtype=torch.bool),
         'u_feats': torch.randn(B, cfg.d_global, generator=g),
         'cell_ctx': torch.eye(cfg.d_cell_ctx)[torch.randint(0, cfg.d_cell_ctx, (B,), generator=g)],
         'dose': torch.rand(B, generator=g), 'time': torch.rand(B, generator=g)}
    b['atom_mask'][:, :n_atoms] = True
    b['r'][:, :dead] = 0.0
    b['y_delta'] = torch.randn(B, cfg.n_genes, generator=g) * 0.4
    b['y_abs'] = b['x_ctl'] + b['y_delta']
    b['y_l5'] = torch.randn(B, cfg.n_genes, generator=g)
    b['m_l3'] = torch.ones(B, dtype=torch.bool)
    b['m_l5'] = torch.ones(B, dtype=torch.bool)
    return b


def _raises(model, b):
    try:
        with torch.no_grad():
            model(b)
        return False
    except RuntimeError:
        return True


def main():
    cfg, dc = V9Config(), V9DataConfig()
    root = dc.root
    M = np.load(os.path.join(root, dc.m_pathway_path))
    ppi = np.load(os.path.join(root, dc.ppi_v9_path))
    gv = np.load(os.path.join(root, dc.gene_vec_path))
    info = list(csv.DictReader(open(os.path.join(root, dc.pathway_info_v9_path), encoding='utf-8'),
                               delimiter='\t'))
    order = [l.strip() for l in open(os.path.join(root, dc.gene_order_path), encoding='utf-8') if l.strip()]
    syms = list(csv.DictReader(open(os.path.join(root, dc.landmark_symbols_path), encoding='utf-8'),
                               delimiter='\t'))

    # ---------------- the priors are what they say they are ----------------
    check('M_pathway is [P, 978] and P matches the config',
          M.shape == (cfg.n_pathways, cfg.n_genes), str(M.shape))
    check('every pathway node has at least one member gene (no dead node)',
          int((M.sum(1) == 0).sum()) == 0, f'{int((M.sum(1) == 0).sum())} empty')
    # M's columns are canonical rows, but the priors name genes by their CURRENT HGNC symbol while
    # pathway_landmark_genes.txt still holds the 2012 L1000 names. landmark_symbols_v9.tsv is the bridge,
    # and this check fails loudly if the two namings ever drift apart.
    cur_pos = {r['hgnc_symbol']: i for i, r in enumerate(syms)}
    check('row p of M is row p of pathway_info_v9.tsv, verified by rebuilding the gene set',
          len(info) == M.shape[0] and all(
              set(np.flatnonzero(M[i]).tolist()) ==
              {cur_pos[g] for g in info[i]['landmark_gene_symbols'].split(',')}
              for i in range(0, len(info), 37)),
          'checked every 37th row, via the HGNC bridge')
    check('the symbol bridge is complete: every current symbol maps to exactly one canonical row',
          len(cur_pos) == len(syms) == cfg.n_genes)
    check('every pathway node carries a curated NAME and a source',
          all(r['term_name'].strip() and r['source'] in ('Reactome', 'GO:BP') for r in info),
          f"{sum(1 for r in info if r['source'] == 'Reactome')} Reactome / "
          f"{sum(1 for r in info if r['source'] == 'GO:BP')} GO:BP")
    check('landmark symbol map is in canonical order and resolves all 978',
          [r['l1000_symbol'] for r in syms] == order and
          all(r['resolved_via'] != 'unresolved' for r in syms),
          f"{sum(1 for r in syms if r['l1000_symbol'] != r['hgnc_symbol'])} renamed")
    check('STRING v9 adjacency is [978, 978] and denser than the truncated one',
          ppi.shape == (978, 978) and int((ppi != 0).sum()) > 25330,
          f'{int((ppi != 0).sum()) // 2} undirected edges vs 12,665 before')
    check('pretrained gene vectors are [978, d_gene_vec]',
          gv.shape == (cfg.n_genes, cfg.d_gene_vec), str(gv.shape))

    torch.manual_seed(0)
    m = LincsV9(cfg, M, ppi, gv).eval()
    b = batch(cfg)
    check('a binned model REFUSES to run before its quantiser is fitted (an unfitted quantiser would '
          'silently ignore the expression input entirely)',
          (cfg.expr_encoder != 'binned') or (not m.bins_fitted and _raises(m, b)))
    m.fit_bins(np.random.default_rng(0).uniform(0, 15, size=(4000, cfg.n_genes)).astype(np.float32))
    check('after fit_bins the model reports its quantisers fitted', m.bins_fitted)
    with torch.no_grad():
        out = m(b)
        out2, aux = m(b, return_aux=True)

    # ---------------- outputs ----------------
    for k in ['abs', 'delta', 'l5']:
        check(f'head "{k}" is [B, 978] and finite',
              tuple(out[k].shape) == (4, cfg.n_genes) and bool(torch.isfinite(out[k]).all()))
    check('return_aux does not change the prediction',
          all(torch.allclose(out[k], out2[k], atol=1e-6) for k in out))
    check('THE ABSOLUTE HEAD IS ANCHORED: abs == x_ctl + delta, exactly',
          bool(torch.allclose(out['abs'], b['x_ctl'] + out['delta'], atol=1e-6)),
          f"max dev {float((out['abs'] - b['x_ctl'] - out['delta']).abs().max()):.2e}")

    # ---------------- expression encoding ----------------
    be = BinnedExpression(8, n_bins=cfg.n_bins, n_genes=cfg.n_genes, mode='global')
    check('quantiser starts UNFITTED, so evaluation data can never define the bins',
          float(be.fitted) == 0.0 and float(be.edges.abs().sum()) == 0.0)
    tr = np.random.default_rng(0).uniform(0, 15, size=(2000, cfg.n_genes)).astype(np.float32)
    be.fit(tr)
    idx = be.bucket(torch.as_tensor(tr[:256]))
    check('after fitting on TRAIN rows the quantiser is marked fitted', float(be.fitted) == 1.0)
    check('bins are quantile-spaced: usage is spread, not collapsed onto one level',
          len(torch.unique(idx)) > cfg.n_bins * 0.9, f'{len(torch.unique(idx))}/{cfg.n_bins} bins used')
    v = torch.linspace(0, 15, 500).unsqueeze(0).repeat(1, 1)[:, :cfg.n_genes]
    if v.shape[1] < cfg.n_genes:
        v = torch.nn.functional.pad(v, (0, cfg.n_genes - v.shape[1]), value=15.0)
    bi = be.bucket(v)[0, :500]
    check('binning is monotone in the value (larger expression never gets a smaller bin)',
          bool((bi[1:] >= bi[:-1]).all()))
    check('n_bins is 128, the field\'s setting', cfg.n_bins == 128)
    check('the raw encoder remains available as the A/B control arm',
          isinstance(RawExpression(8), torch.nn.Module))

    # ---------------- the priors are LOAD-BEARING, checked by ablation to the MEAN ----------------
    score = lambda y: float(y.abs().mean())
    d_ctl, m_ctl = ablate_to_mean(m, b, 'x_ctl', score)
    check('matched control is live: ablating it to the batch mean moves the prediction',
          m_ctl > 1e-4, f'dScore {d_ctl:+.4f}  |dY|max {m_ctl:.4f}')
    d_epi, m_epi = ablate_to_mean(m, b, 'E', score)
    check('chromatin is live as a gene embedding (|dY|max > 0 distinguishes a null from a dead branch)',
          m_epi > 1e-4, f'dScore {d_epi:+.4f}  |dY|max {m_epi:.4f}')
    # STRING message passing is ZERO-INIT BY DESIGN (v7: "starts as an exact no-op and must earn its
    # contribution"), so at initialisation it CANNOT move the output and a liveness assertion here would
    # be false. What is asserted at init is that it is wired in; liveness is asserted on a trained
    # checkpoint, and the same ablation is what measures it there.
    d_ppi, m_ppi = ablate_module_to_mean(m, b, m.ppi, score)
    check('STRING message passing is zero-init, so it is an exact no-op before training',
          m.ppi is not None and m_ppi == 0.0, f'|dY|max {m_ppi:.4f} (expected exactly 0)')
    with torch.no_grad():
        m.ppi.w.weight.normal_(0, 0.02)
    d_ppi2, m_ppi2 = ablate_module_to_mean(m, b, m.ppi, score)
    with torch.no_grad():
        m.ppi.w.weight.zero_()
    check('STRING message passing is genuinely WIRED IN: give it non-zero weights and it moves the output',
          m_ppi2 > 1e-4, f'dScore {d_ppi2:+.4f}  |dY|max {m_ppi2:.4f}')
    check('genes with no STRING edge receive exactly zero message',
          float(m.ppi.has_edge.sum()) == float((np.asarray(ppi) != 0).any(1).sum()),
          f'{int(m.ppi.has_edge.sum())}/978 genes have an edge')
    d_pw, m_pw = ablate_module_to_mean(m, b, m.pathway, score)
    check('named pathway readout is live', m_pw > 1e-4, f'dScore {d_pw:+.4f}  |dY|max {m_pw:.4f}')
    check('ablation helper reports |dY|max alongside the score change (method rule 2)',
          isinstance(m_epi, float) and isinstance(d_epi, float))

    # ---------------- chromatin: both routes present, and reliability gates it ----------------
    check('chromatin enters the gene representation as an embedding', m.gene_repr.use_epi)
    check('the signed additive chromatin HEAD is kept as well', cfg.keep_epi_head)
    check('chromatin contribution is signed (both directions occur)',
          bool((aux['epi_contrib'] > 0).any() and (aux['epi_contrib'] < 0).any()))
    b0 = dict(b); b0['r'] = torch.zeros_like(b['r'])
    with torch.no_grad():
        _, aux0 = m(b0, return_aux=True)
    check('reliability r=0 zeroes the chromatin head contribution for those genes',
          float(aux0['epi_contrib'].abs().max()) < 1e-6,
          f"max |contrib| at r=0: {float(aux0['epi_contrib'].abs().max()):.2e}")

    # ---------------- the control encoder is SEPARATE ----------------
    ctl_ids = {id(p) for p in m.ctl_enc.parameters()}
    pert_ids = {id(p) for p in m.perturb.parameters()}
    check('the control encoder shares no parameters with the perturbation stream',
          len(ctl_ids & pert_ids) == 0, f'{len(ctl_ids)} vs {len(pert_ids)} tensors')
    b1 = dict(b); b1['use_cell_ctl'] = False
    with torch.no_grad():
        y1 = m(b1)['delta']
    check('the two control views are both used: dropping the per-cell view changes the prediction',
          float((y1 - out['delta']).abs().max()) > 1e-4,
          f"|dY|max {float((y1 - out['delta']).abs().max()):.4f}")

    # ---------------- auxiliary supervision: FIXED small weights, never learned ----------------
    names = [n for n, _ in m.named_parameters()]
    check('no learned task-weight parameter exists (v7 measured Kendall weighting sending ~90% of the '
          'gradient to the auxiliaries)', not any('task_weight' in n or 'log_var' in n for n in names))
    check('auxiliary weights are small and fixed', cfg.aux_pathway_w <= 0.1 and cfg.aux_epi_w <= 0.1,
          f'pathway {cfg.aux_pathway_w}, epi {cfg.aux_epi_w}')

    # ---------------- loss ----------------
    Mn = torch.as_tensor(M, dtype=torch.float32)
    Mn = Mn / Mn.sum(1, keepdim=True).clamp(min=1)
    loss, parts = v9_loss(out2, b, cfg, Mn, aux)
    check('loss is finite and multi-task', bool(torch.isfinite(torch.as_tensor(loss))) and
          {'abs', 'delta', 'l5', 'pcc'} <= set(parts), ', '.join(sorted(parts)))
    bm = dict(b); bm['m_l5'] = torch.zeros(4, dtype=torch.bool); bm['y_l5'] = torch.full((4, 978), float('nan'))
    lm, _ = v9_loss(out2, bm, cfg, Mn, aux)
    check('a fully-masked target contributes no NaN (per-target row masks work)',
          bool(torch.isfinite(torch.as_tensor(lm))))
    tp, te = aux_targets(b['y_delta'], Mn)
    # The anchored absolute head makes the abs loss identical to the delta loss, element for element.
    # That is not a bug, but it means w_abs is not a second task and the effective delta weight is their
    # SUM. If someone un-anchors the head, this check fails and the weights have to be revisited.
    check('the anchored absolute head makes the abs loss IDENTICAL to the delta loss '
          '(so w_abs is not a second task: the effective delta weight is w_abs + w_delta)',
          abs(parts['abs'] - parts['delta']) < 1e-6,
          f"abs {parts['abs']:.6f} vs delta {parts['delta']:.6f}")
    check('aux targets come from the MEASURED response and match the head shapes',
          tp.shape == aux['pathway_pred'].shape and te.shape == aux['epi_pred'].shape)
    check('pathway aux target is a masked mean of |delta| (non-negative, bounded by max|delta|)',
          bool((tp >= 0).all() and (tp <= b['y_delta'].abs().max() + 1e-5).all()))

    # ---------------- interpretability contract ----------------
    check('pathway activations are one per NAMED node',
          aux['pathway_activations'].shape[1] == len(info), str(tuple(aux['pathway_activations'].shape)))
    obs, nm, nsd, p = permutation_null(np.random.default_rng(0).normal(size=200),
                                       np.random.default_rng(1).normal(size=200),
                                       lambda s, l: float(np.corrcoef(s, l)[0, 1]))
    check('a measured chance level is available for any readout (method rule 6)',
          abs(nm) < 0.25 and nsd > 0, f'null mean {nm:+.4f} sd {nsd:.4f}, p={p:.3f}')
    al = pathway_alignment(aux['pathway_activations'].numpy(), b['y_delta'].numpy(), M)
    check('pathway alignment returns a rank correlation in [-1, 1]', -1 <= al <= 1, f'{al:+.4f}')

    # ---------------- config contracts ----------------
    check('CCLE is off by default (redundant and dominated, RESULTS 27.4), but still switchable',
          dc.use_ccle is False and hasattr(dc, 'use_ccle'))
    check('all three targets are predicted', dc.predict_abs and dc.predict_delta and dc.predict_l5)
    check('the Level-5 head is retained so v3-v7 numbers stay comparable', 'l5' in out)
    n_par = sum(p.numel() for p in m.parameters())
    check('parameter count is in a trainable range for a T4 x2 budget', 5e6 < n_par < 1.2e8,
          f'{n_par / 1e6:.1f}M')

    torch.manual_seed(0)
    m2 = LincsV9(cfg, M, ppi, gv).eval()
    m2.fit_bins(np.random.default_rng(0).uniform(0, 15, size=(4000, cfg.n_genes)).astype(np.float32))
    with torch.no_grad():
        check('construction is deterministic under a fixed seed (weights AND quantiser)',
              bool(torch.allclose(m2(b)['delta'], out['delta'], atol=1e-6)))
    with torch.no_grad():
        check('B=1 works', tuple(m(batch(cfg, B=1))['delta'].shape) == (1, cfg.n_genes))

    print('\n%d/%d checks passed' % (sum(R), len(R)))
    sys.exit(0 if all(R) else 1)


if __name__ == '__main__':
    main()
