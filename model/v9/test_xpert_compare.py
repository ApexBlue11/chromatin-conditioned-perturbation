# -*- coding: utf-8 -*-
"""
Design checks for the XPert comparison apparatus. Each asserts a property the comparison CLAIMS, not that
the code runs. Run before any number from this apparatus is written down:

    python model/v9/test_xpert_compare.py

The checks that matter most are the ones that would otherwise fail silently and still produce a plausible
number: the gene axis being theirs, the dose reaching v9 at their resolution and no finer, the released
checkpoint's non-default architecture flag, the flash-attention stand-in computing the same function as
the kernel it replaces, and the paired comparison refusing rows that are not actually the same rows.

Checks whose inputs are absent (a bundle not yet built, features not yet fetched) SKIP loudly rather than
pass vacuously.
"""
import os, sys, csv, json, types, math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = r'C:\Projects\LINCS'
XPERT = os.path.join(ROOT, 'external', 'xpert', 'code', 'XPert')
PROC = os.path.join(XPERT, 'processed_data')
BUNDLE = os.path.join(ROOT, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')
CKPT = os.path.join(ROOT, 'external', 'xpert', 'saved_model_extracted', 'saved_model',
                    'l1000_mdmt_warm_split.pth')

R = []


def check(name, cond, detail=''):
    R.append(bool(cond))
    print('[%s] %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail else ''))


def skip(name, why):
    print('[SKIP] %s  -- %s' % (name, why))


def main():
    # ---------------- 1. the gene axis is THEIRS, entry by entry ----------------
    gi = os.path.join(PROC, 'l1000_gene_info_978.csv')
    order_p = os.path.join(ROOT, 'Data Info', 'pathway_landmark_genes.txt')
    if os.path.exists(gi) and os.path.exists(order_p):
        import pandas as pd
        theirs = pd.read_csv(gi).sort_values('gene_idx')['gene_name'].astype(str).tolist()
        ours = [l.strip() for l in open(order_p, encoding='utf-8') if l.strip()]
        check('gene axis: their 978 gene order equals ours entry by entry',
              theirs == ours, '%d genes' % len(ours))
    else:
        skip('gene axis', 'l1000_gene_info_978.csv not fetched')

    # ---------------- 2. the bundle is a faithful, self-consistent copy of their benchmark ----------
    if os.path.exists(BUNDLE):
        z = np.load(BUNDLE, allow_pickle=True)
        X, C = z['X'], z['X_ctl']
        check('bundle: 68,830 conditions x 978 genes', X.shape == (68830, 978), str(X.shape))
        check('bundle: expression and control are finite everywhere',
              np.isfinite(X).all() and np.isfinite(C).all())
        check('bundle: values are Level-3 log expression, inside their config range [0, 15.002]',
              X.min() >= 0 and X.max() <= 15.002 and C.min() >= 0 and C.max() <= 15.002,
              '[%.2f, %.2f]' % (float(X.min()), float(X.max())))
        lab = z['split_split_1'].astype(str)
        check('bundle: split_1 train and test are disjoint and cover every row',
              set(np.unique(lab).tolist()) == {'train', 'test'})
        # the five warm folds should be a partition: every row is test exactly once
        cnt = np.zeros(len(lab), int)
        for k in range(1, 6):
            cnt += (z['split_split_%d' % k].astype(str) == 'test').astype(int)
        check('bundle: the five warm folds are a partition (each row is test exactly once)',
              bool((cnt == 1).all()), 'counts seen: %s' % np.unique(cnt).tolist())

        # ---------------- 3. dose reaches v9 at THEIR resolution and no finer ----------------
        di, dose = z['meta_dose_idx'], z['meta_dose']
        by_bin = {int(b): np.unique(dose[di == b]) for b in np.unique(di)}
        check('dose: each bin maps to exactly one representative value (no extra information leaks in)',
              all(len(v) == 1 for v in by_bin.values()))
        reps = [float(v[0]) for v in by_bin.values()]
        check('dose: distinct bins map to distinct values (the map is invertible)',
              len(set(reps)) == len(reps))
        check('dose: representatives increase with the bin index',
              all(reps[i] < reps[i + 1] for i in range(len(reps) - 1)),
              '%s' % [round(r, 3) for r in reps])

        # ---------------- 4. the pooled-dose caveat is carried, not lost ----------------
        pooled = z['meta_dose_pooled'].astype(bool)
        check('dose: the pooled-dose flag is present and matches the recorded 18.9%',
              abs(float(pooled.mean()) - 0.1895) < 0.002, '%.4f' % float(pooled.mean()))
    else:
        skip('bundle checks', 'run model/v9/xpert_mdmt_extract.py first')

    # ---------------- 5. the released checkpoint's architecture is NOT the argparse default ----------
    if os.path.exists(CKPT):
        import torch
        sd = torch.load(CKPT, map_location='cpu', weights_only=False)['model_state_dict']
        check('checkpoint: contains cls_token and class_fc, so include_cell_idx was True (a NON-default '
              'flag that changes the forward pass)',
              'cls_token' in sd and 'class_fc.weight' in sd)
        check('checkpoint: the cell classifier width matches their config num_cell_id=240',
              tuple(sd['class_fc.weight'].shape) == (240, 256), str(tuple(sd['class_fc.weight'].shape)))
        check('checkpoint: drug branch takes 512-d atom features, i.e. unimol (514 minus mask and symbol)',
              tuple(sd['drug_emb.linear.weight'].shape) == (256, 512))
        check('checkpoint: the encoder stacks match config (trt CA+SA+SA+CA, ctl SA x4)',
              len({k.split('.')[2] for k in sd if k.startswith('attnEncoder_trt.crossEncoders')}) == 2 and
              len({k.split('.')[2] for k in sd if k.startswith('attnEncoder_ctl.selfEncoders')}) == 4)
    else:
        skip('checkpoint checks', 'l1000_mdmt_warm_split.pth not extracted')

    # ---------------- 6. the flash-attention stand-in computes the SAME function as the kernel ---------
    try:
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, os.path.join(HERE, '_shims'))
        from flash_attn.flash_attn_interface import flash_attn_func
        g = torch.Generator().manual_seed(0)
        B, S, H, D = 3, 17, 4, 16
        q, k, v = (torch.randn(B, S, H, D, generator=g) for _ in range(3))
        got = flash_attn_func(q, k, v, dropout_p=0.0)
        # hand-written reference: exactly what flash attention computes, unmasked, scale 1/sqrt(D)
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
        s = (qt @ kt.transpose(-1, -2)) / math.sqrt(D)
        ref = (F.softmax(s, dim=-1) @ vt).transpose(1, 2)
        check('flash shim: matches a hand-written unmasked attention to 1e-5',
              bool(torch.allclose(got, ref, atol=1e-5)),
              'max|diff| %.2e' % float((got - ref).abs().max()))
        check('flash shim: preserves the (batch, seqlen, nheads, headdim) layout flash_attn_func returns',
              tuple(got.shape) == (B, S, H, D))
        # the branch difference this shim exists to preserve: the flash path IGNORES the mask
        mask = torch.zeros(B, 1, 1, S)
        mask[:, :, :, S // 2:] = -10000.0
        masked = (F.softmax(s + mask, dim=-1) @ vt).transpose(1, 2)
        check('flash shim: differs from the MASKED dense branch -- confirming the two branches in their '
              'model_utils.py are not interchangeable, which is why the shim is needed',
              not bool(torch.allclose(got, masked, atol=1e-3)))
        bad = False
        try:
            flash_attn_func(q, k, v, window_size=(4, 4))
        except NotImplementedError:
            bad = True
        check('flash shim: refuses options it does not implement instead of returning a different quantity',
              bad)
    except ImportError as e:
        skip('flash shim checks', 'torch unavailable (%s)' % e)

    # ---------------- 7. the fetched unimol rows are correctly strided ----------------
    smi_p = os.path.join(PROC, 'all_drugs_idx2smi_8981.npy')
    smi_p = smi_p if os.path.exists(smi_p) else smi_p + '.hold'
    got_any = False
    for name in ['unimol_mdmt_1970.npz', 'unimol_hdaci_30.npz']:
        p = os.path.join(PROC, name)
        if not os.path.exists(p) or not os.path.exists(smi_p):
            continue
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            break
        got_any = True
        z = np.load(p)
        smi = np.load(smi_p, allow_pickle=True).item()
        A = z['feat'].shape[1]
        bad, n_ok, n_trunc = [], 0, 0
        for i, f in list(zip(z['idx'], z['feat']))[:300]:
            s_i = smi.get(int(i))
            m = Chem.MolFromSmiles(s_i) if s_i else None
            if m is None:
                continue
            want = min(Chem.AddHs(m).GetNumAtoms() + 2, A)
            got = int(f[:, 0].sum())
            if got != want:
                bad.append((int(i), got, want))
            elif got == A:
                n_trunc += 1
            else:
                n_ok += 1
        check('unimol %s: mask_len == min(n_atoms + 2, %d) for every molecule (stride is right, and '
              'truncation at the cap is accounted for)' % (name, A), not bad,
              '%d exact, %d truncated, %d bad %s' % (n_ok, n_trunc, len(bad), bad[:2]))
        masks = z['feat'][:, :, 0]
        check('unimol %s: every atom mask is a prefix run of 1s' % name,
              bool(np.all(np.diff(masks, axis=1) <= 0)))
    if not got_any:
        skip('unimol stride checks', 'features not fetched, or rdkit/SMILES unavailable')

    # ---------------- 8. the paired comparison refuses rows that are not the same rows ----------------
    import subprocess, tempfile
    td = tempfile.mkdtemp()
    n, G = 40, 978
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, G)).astype(np.float32)
    c = rng.normal(size=(n, G)).astype(np.float32)
    np.save(os.path.join(td, 'theirs.npy'),
            {'y_true': y, 'y_pred': y + 0.1, 'ctl_true': c, 'row_index': np.arange(n)})
    np.savez(os.path.join(td, 'ours.npz'), y_pred=y + 0.2, y_true=y, ctl_true=c,
             row_index=np.arange(n) + 1000)                       # deliberately different rows
    r = subprocess.run([sys.executable, os.path.join(HERE, 'head_to_head_mdmt.py'),
                        '--theirs', os.path.join(td, 'theirs.npy'),
                        '--ours', os.path.join(td, 'ours.npz'),
                        '--bundle', 'does_not_exist.npz',
                        '--out', os.path.join(td, 'o.json')], capture_output=True, text=True)
    check('paired comparison: REFUSES to compare two runs covering different rows',
          r.returncode != 0 and 'different rows' in (r.stdout + r.stderr))
    np.savez(os.path.join(td, 'ours2.npz'), y_pred=y + 0.2, y_true=y + 5.0, ctl_true=c,
             row_index=np.arange(n))                              # same rows, different TARGET
    r2 = subprocess.run([sys.executable, os.path.join(HERE, 'head_to_head_mdmt.py'),
                         '--theirs', os.path.join(td, 'theirs.npy'),
                         '--ours', os.path.join(td, 'ours2.npz'),
                         '--bundle', 'does_not_exist.npz',
                         '--out', os.path.join(td, 'o2.json')], capture_output=True, text=True)
    check('paired comparison: REFUSES when the two runs disagree about the target on aligned rows',
          r2.returncode != 0 and 'TARGET' in (r2.stdout + r2.stderr))

    # ---------------- 9. their metric is the MEAN of per-row Pearson, and we report that ----------
    try:
        sys.path.insert(0, XPERT)
        import importlib.util
        spec = importlib.util.spec_from_file_location('their_metrics', os.path.join(XPERT, 'metrics.py'))
        tm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tm)
        a = rng.normal(size=(50, 200))
        b = a + rng.normal(size=(50, 200))
        from head_to_head_mdmt import per_row_pearson
        mine = per_row_pearson(a, b)
        check('metric: our per-row Pearson equals theirs row by row',
              bool(np.allclose(np.array(tm.pearson(a, b, flag=True)[1]), mine, atol=1e-9)))
        check('metric: their headline is the MEAN, which differs from the median we have always reported',
              abs(float(np.mean(mine)) - tm.pearson(a, b)) < 1e-9
              and abs(float(np.median(mine)) - tm.pearson(a, b)) > 1e-6,
              'mean %.4f vs median %.4f' % (float(np.mean(mine)), float(np.median(mine))))
    except Exception as e:
        skip('metric equivalence', '%s: %s' % (type(e).__name__, e))

    print('\n%d/%d checks passed' % (sum(R), len(R)))
    sys.exit(0 if all(R) else 1)


if __name__ == '__main__':
    main()
