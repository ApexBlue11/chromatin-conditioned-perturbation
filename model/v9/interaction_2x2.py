# -*- coding: utf-8 -*-
"""2x2 interaction harness for the atom / contextualisation experiment [TASK W5].

Why this harness exists:
`model/v9` predicts drug-induced gene expression. Ablating the per-atom drug tokens improves accuracy
by ~0.0146 on unseen compounds (3 seeds, intervals exclude zero; measured in atom_ablation_ci.py).
The leading hypothesis is that uncontextualised per-atom vectors act as noise: in v9, genes attended over
a static bag of independent Uni-Mol vectors with no intramolecular structure.

A `_DrugBlock` (drug self-attention) was added behind `config_v9.drug_self_attn` [RESULTS 47.2], along with
an identity-masked diagonal ablation mode (`diagonal=True`) [TASK W4] where each atom attends only to itself.
This preserves all parameters, normalisations, and the SwiGLU feed-forward branch, eliminating cross-token
information flow without altering model capacity.

Why a 2x2 factorial rather than two separate main effects:
Adversarial review established that measuring "ablate atoms" and "ablate contextualisation" separately
yields only two marginal main effects. The decisive scientific quantity is the INTERACTION: does the effect
of atom tokens depend on whether contextualisation is active?

The 2x2 design:
  Cell | Atoms   | Contextualisation | Implementation
  S11  | Present | Present           | model(batch)
  S01  | Ablated | Present           | model(batch_with_atoms_meaned)
  S10  | Present | Ablated           | model(batch, diagonal=True)
  S00  | Ablated | Ablated           | model(batch_with_atoms_meaned, diagonal=True)

Quantities:
  atom_effect_with_context    = score(S11) - score(S01)
  atom_effect_without_context = score(S10) - score(S00)
  INTERACTION                 = atom_effect_with_context - atom_effect_without_context

Requirements enforced in code:
  1. Score function: per-row Pearson correlation on delta against y_delta, aggregated as the MEDIAN.
     `pearson_rows` is copied byte-for-byte from `probe_v9.py`.
  2. Atom ablation: matches `interp_v9.py::ablate_to_mean` exactly (batch['atoms'] replaced with its
     dim-0 chunk mean, broadcast).
  3. Paired evaluation: all 4 cells scored on identical rows, chunked identically.
  4. Diagnostics: report per split (unseen_cell, unseen_compound, unseen_both): all 4 cell scores,
     both main effects, interaction, and |dY|max for each ablation to distinguish true nulls from
     unfired branches.
  5. Bootstrap: 20000 resamples over rows (seed 0), resampling row indices once per draw and
     recomputing all 4 medians on the same resampled rows to emit a 95% CI.
  6. Failure mode prevention: checkpoints without drug_self_attn=True (such as r0_ckpt_v9_fold0_seed0.pt)
     are detected explicitly and refused with "diagonal mode unavailable for this checkpoint", avoiding
     a silent S10 == S11 false null.
"""
import argparse
import copy
import io
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
os.chdir(HERE)

from config_v9 import V9Config, V9DataConfig                     # noqa: E402
from model_v9 import LincsV9                                       # noqa: E402
from data_v9 import LincsV9Dataset, collate_v9, build_splits       # noqa: E402
from train_v9_gpu import resolve_v9                                # noqa: E402


class DiagonalUnavailableError(RuntimeError):
    """Raised when diagonal ablation is attempted on a model/checkpoint lacking drug self-attention."""
    pass


def pearson_rows(a, b):
    """Byte-for-byte the definition in probe_v9.py, so numbers attach to that one."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def make_atom_ablated_batch(batch, key='atoms'):
    """Matches interp_v9.py::ablate_to_mean: replace batch[key] with its chunk mean over dim 0."""
    ref = batch[key]
    patched = dict(batch)
    patched[key] = ref.mean(dim=0, keepdim=True).expand_as(ref)
    return patched


def check_diagonal_available(model):
    """Verify whether model has drug self-attention enabled and can run diagonal ablation.

    Adversarial review requirement:
    Checkpoints trained with `drug_self_attn=False` lack `_DrugBlock`. If evaluated under diagonal
    mode without a guard, PerturbBlock ignores diagonal=True and produces S10 == S11 and S00 == S01.
    That would falsely appear as an interaction of 0.0 (a false null).
    We detect this explicitly and report 'diagonal mode unavailable for this checkpoint'.
    """
    cfg = getattr(model, 'cfg', None)
    has_cfg = bool(getattr(cfg, 'drug_self_attn', False)) if cfg is not None else False
    perturb_blocks = getattr(model, 'perturb', [])
    n_drug_sa = sum(1 for blk in perturb_blocks if getattr(blk, 'drug_sa', None) is not None)
    if not has_cfg or n_drug_sa == 0:
        return False, (
            "diagonal mode unavailable for this checkpoint: "
            f"cfg.drug_self_attn={has_cfg}, perturb blocks with drug_sa={n_drug_sa}/{len(perturb_blocks)}"
        )
    return True, ""


def evaluate_chunks_2x2(model, chunks, key='atoms'):
    """Evaluate S11, S01, S10, S00 on identical chunks and track |dY|max.

    Returns:
        r11, r01, r10, r00: 1D numpy arrays of per-row Pearson correlations
        dy_stats: dict of maximum absolute output differences (|dY|max) across all chunks
    """
    r11_list, r01_list, r10_list, r00_list = [], [], [], []
    dymax_atom_with_ctx = 0.0
    dymax_atom_no_ctx = 0.0
    dymax_ctx_with_atoms = 0.0
    dymax_ctx_no_atoms = 0.0

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for c in chunks:
            tgt = c['y_delta'].float().cpu().numpy()
            c_ablated = make_atom_ablated_batch(c, key=key)

            # S11: atoms present, contextualisation present
            out11 = model(c)
            y11 = out11['delta'] if isinstance(out11, dict) else out11

            # S01: atoms ablated, contextualisation present
            out01 = model(c_ablated)
            y01 = out01['delta'] if isinstance(out01, dict) else out01

            # S10: atoms present, contextualisation ablated (diagonal=True)
            out10 = model(c, diagonal=True)
            y10 = out10['delta'] if isinstance(out10, dict) else out10

            # S00: atoms ablated, contextualisation ablated (diagonal=True)
            out00 = model(c_ablated, diagonal=True)
            y00 = out00['delta'] if isinstance(out00, dict) else out00

            # |dY|max tracking across all chunks
            dy_a_wc = float((y11 - y01).abs().max().item())
            dy_a_nc = float((y10 - y00).abs().max().item())
            dy_c_wa = float((y11 - y10).abs().max().item())
            dy_c_na = float((y01 - y00).abs().max().item())

            dymax_atom_with_ctx = max(dymax_atom_with_ctx, dy_a_wc)
            dymax_atom_no_ctx = max(dymax_atom_no_ctx, dy_a_nc)
            dymax_ctx_with_atoms = max(dymax_ctx_with_atoms, dy_c_wa)
            dymax_ctx_no_atoms = max(dymax_ctx_no_atoms, dy_c_na)

            r11_list.append(pearson_rows(y11.float().cpu().numpy(), tgt))
            r01_list.append(pearson_rows(y01.float().cpu().numpy(), tgt))
            r10_list.append(pearson_rows(y10.float().cpu().numpy(), tgt))
            r00_list.append(pearson_rows(y00.float().cpu().numpy(), tgt))

    model.train(was_training)

    r11 = np.concatenate(r11_list) if r11_list else np.empty(0, dtype=np.float32)
    r01 = np.concatenate(r01_list) if r01_list else np.empty(0, dtype=np.float32)
    r10 = np.concatenate(r10_list) if r10_list else np.empty(0, dtype=np.float32)
    r00 = np.concatenate(r00_list) if r00_list else np.empty(0, dtype=np.float32)

    dy_stats = {
        'atom_with_context': round(dymax_atom_with_ctx, 4),
        'atom_without_context': round(dymax_atom_no_ctx, 4),
        'context_with_atoms': round(dymax_ctx_with_atoms, 4),
        'context_without_atoms': round(dymax_ctx_no_atoms, 4),
        'dY_max_atom': round(max(dymax_atom_with_ctx, dymax_atom_no_ctx), 4),
        'dY_max_context': round(max(dymax_ctx_with_atoms, dymax_ctx_no_atoms), 4),
    }

    return r11, r01, r10, r00, dy_stats


def boot_2x2(r11, r01, r10, r00, n_boot=20000, seed=0):
    """Compute 2x2 cell scores, main effects, interaction, and bootstrap intervals.

    Row filtering:
    All 4 cells are evaluated on IDENTICAL rows. A row non-finite in ANY of the
    4 cells is dropped across all 4 to preserve row pairing.

    Aggregation:
    Scores are median per-row Pearson correlation.
    Interaction is the difference of differences of medians:
      (median(S11) - median(S01)) - (median(S10) - median(S00))

    Bootstrap:
    Resample row indices once per draw and recompute all 4 medians on the
    identical resampled rows. Emit a 95% interval (2.5% and 97.5% percentiles).
    """
    ok = np.isfinite(r11) & np.isfinite(r01) & np.isfinite(r10) & np.isfinite(r00)
    a11, a01, a10, a00 = r11[ok], r01[ok], r10[ok], r00[ok]
    n_rows = int(len(a11))
    if n_rows == 0:
        raise ValueError("No valid rows remaining after filtering non-finite values across 2x2 cells.")

    s11 = float(np.median(a11))
    s01 = float(np.median(a01))
    s10 = float(np.median(a10))
    s00 = float(np.median(a00))

    atom_with_ctx = s11 - s01
    atom_no_ctx = s10 - s00
    ctx_with_atoms = s11 - s10
    ctx_no_atoms = s01 - s00
    interaction = atom_with_ctx - atom_no_ctx

    # Paired row differences
    diff_atom_with_ctx = a11 - a01
    diff_atom_no_ctx = a10 - a00
    diff_interaction = diff_atom_with_ctx - diff_atom_no_ctx
    mean_interaction = float(diff_interaction.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(n_boot, n_rows))

    boot_m11 = np.median(a11[idx], axis=1)
    boot_m01 = np.median(a01[idx], axis=1)
    boot_m10 = np.median(a10[idx], axis=1)
    boot_m00 = np.median(a00[idx], axis=1)

    boot_atom_wc = boot_m11 - boot_m01
    boot_atom_nc = boot_m10 - boot_m00
    boot_inter = boot_atom_wc - boot_atom_nc

    boot_mean_inter = diff_interaction[idx].mean(axis=1)

    ci95_inter = [round(float(np.percentile(boot_inter, 2.5)), 5),
                  round(float(np.percentile(boot_inter, 97.5)), 5)]
    ci95_atom_wc = [round(float(np.percentile(boot_atom_wc, 2.5)), 5),
                    round(float(np.percentile(boot_atom_wc, 97.5)), 5)]
    ci95_atom_nc = [round(float(np.percentile(boot_atom_nc, 2.5)), 5),
                    round(float(np.percentile(boot_atom_nc, 97.5)), 5)]
    ci95_mean_inter = [round(float(np.percentile(boot_mean_inter, 2.5)), 5),
                       round(float(np.percentile(boot_mean_inter, 97.5)), 5)]

    return {
        'n_rows': n_rows,
        'scores': {
            'S11': round(s11, 5),
            'S01': round(s01, 5),
            'S10': round(s10, 5),
            'S00': round(s00, 5),
        },
        'atom_effect_with_context': round(atom_with_ctx, 5),
        'atom_effect_with_context_ci95': ci95_atom_wc,
        'atom_effect_without_context': round(atom_no_ctx, 5),
        'atom_effect_without_context_ci95': ci95_atom_nc,
        'context_effect_with_atoms': round(ctx_with_atoms, 5),
        'context_effect_without_atoms': round(ctx_no_atoms, 5),
        'interaction': round(interaction, 5),
        'interaction_ci95': ci95_inter,
        'interaction_paired_mean': round(mean_interaction, 5),
        'interaction_paired_mean_ci95': ci95_mean_inter,
    }


def run_interaction_2x2(model, chunks, n_boot=20000, seed=0, key='atoms', raise_if_unavailable=True):
    """Full 2x2 harness entry point for a model and a list of chunks.

    Checks diagonal mode availability, scores the 4 cells, tracks |dY|max,
    and runs the row bootstrap.
    """
    avail, reason = check_diagonal_available(model)
    if not avail:
        if raise_if_unavailable:
            raise DiagonalUnavailableError(reason)
        return {
            'diagonal_available': False,
            'error': reason,
        }

    r11, r01, r10, r00, dy_stats = evaluate_chunks_2x2(model, chunks, key=key)
    res = boot_2x2(r11, r01, r10, r00, n_boot=n_boot, seed=seed)
    res['diagonal_available'] = True
    res['dY_max'] = dy_stats
    res['fired_atom'] = bool(dy_stats['dY_max_atom'] > 1e-5)
    res['fired_context'] = bool(dy_stats['dY_max_context'] > 1e-5)
    return res


def main():
    ap = argparse.ArgumentParser(
        description="2x2 interaction harness for atom tokens x contextualisation."
    )
    ap.add_argument('--ckpt', required=True, help="Path to checkpoint (.pt file)")
    ap.add_argument('--batch', type=int, default=48, help="Chunk size (must match probe_v9 / atom_ablation_ci)")
    ap.add_argument('--n_eval', type=int, default=1500, help="Maximum rows to evaluate per split")
    ap.add_argument('--n_boot', type=int, default=20000, help="Number of bootstrap resamples for interaction CI")
    ap.add_argument('--seed', type=int, default=0, help="RNG seed for row selection and bootstrap")
    ap.add_argument('--key', default='atoms', help="Batch key to ablate (default: atoms)")
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = a.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(os.getcwd(), ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_fn = os.path.basename(ckpt_path)
    ckpt_stem = os.path.splitext(ckpt_fn)[0]
    dst_json = os.path.join(ROOT, 'model', 'results', f'v9_interaction_2x2_{ckpt_stem}.json')
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)

    print(f"=== v9 2x2 Interaction Harness [TASK W5] ===", flush=True)
    print(f"Checkpoint: {ckpt_fn} | Device: {dev}", flush=True)
    print(f"Config: batch={a.batch}, n_eval={a.n_eval}, n_boot={a.n_boot}, seed={a.seed}\n", flush=True)

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_raw = ck.get('cfg', {})
    cfg_kwargs = {k: v for k, v in cfg_raw.items() if k in V9Config.__dataclass_fields__ and v is not None}
    cfg = V9Config(**cfg_kwargs)

    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_pathway_path))
    ppi = np.load(R(dc.ppi_v9_path))
    gv = np.load(R(dc.gene_vec_path))

    torch.manual_seed(0)
    model = LincsV9(cfg, M, ppi, gv)
    if 'model' in ck:
        model.load_state_dict(ck['model'])
    model = model.to(dev).eval()

    avail, reason = check_diagonal_available(model)
    if not avail:
        print(f"[REFUSED] {reason}", flush=True)
        print(f"Checkpoint '{ckpt_fn}' does not have drug_self_attn=True.", flush=True)
        print("Evaluating diagonal ablation without _DrugBlock would silently return S10 == S11,", flush=True)
        print("fabricating a false null interaction (0.0). Refusing evaluation rather than producing false numbers.\n", flush=True)
        out = {
            'ckpt': ckpt_fn,
            'epoch': int(ck.get('epoch', -1)),
            'diagonal_available': False,
            'error': reason,
            'n_eval': a.n_eval,
            'batch': a.batch,
            'n_boot': a.n_boot,
            'seed': a.seed,
            'splits': {},
        }
        with io.open(dst_json, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(json.dumps(out, indent=2) + '\n')
        print(f"Wrote refusal record to {dst_json}", flush=True)
        return

    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    if not model.bins_fitted and cfg.expr_encoder == 'binned':
        tr = sp['train'][ds.has_l3[sp['train']]]
        rows = np.sort(ds.ds_to_l3[tr])
        sample_rows = rows[np.linspace(0, len(rows) - 1, min(20000, len(rows))).astype(int)]
        model.fit_bins(np.asarray(ds.Xctl[sample_rows], np.float32))
        model = model.to(dev)
        if not model.bins_fitted:
            raise RuntimeError("FATAL: quantiser did not fit.")

    out = {
        'ckpt': ckpt_fn,
        'epoch': int(ck.get('epoch', -1)),
        'diagonal_available': True,
        'device': dev,
        'n_eval': a.n_eval,
        'batch': a.batch,
        'n_boot': a.n_boot,
        'seed': a.seed,
        'key': a.key,
        'splits': {},
    }

    rng = np.random.default_rng(a.seed)
    for name, key in [('unseen_cell', 'test_coldcell'),
                      ('unseen_compound', 'test_colddrug'),
                      ('unseen_both', 'test_coldboth')]:
        idx = sp[key][ds.strength[sp[key]] >= dc.eval_min_strength]
        idx = idx[ds.has_l3[idx]]
        n_eligible = int(len(idx))
        if n_eligible == 0:
            print(f"Split {name} has 0 eligible rows; skipping.", flush=True)
            continue

        n_eval_bound = bool(a.n_eval < n_eligible)
        n_sample = min(a.n_eval, n_eligible)
        idx_eval = np.sort(rng.choice(idx, n_sample, replace=False))

        chunks = [collate_v9([ds[i] for i in idx_eval[s:s + a.batch]])
                  for s in range(0, len(idx_eval), a.batch)]
        chunks = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in c.items()} for c in chunks]

        r11, r01, r10, r00, dy_stats = evaluate_chunks_2x2(model, chunks, key=a.key)
        st = boot_2x2(r11, r01, r10, r00, n_boot=a.n_boot, seed=a.seed)
        st['n_eligible'] = n_eligible
        st['n_eval_requested'] = a.n_eval
        st['n_eval_bound'] = n_eval_bound
        st['dY_max'] = dy_stats
        st['fired_atom'] = bool(dy_stats['dY_max_atom'] > 1e-5)
        st['fired_context'] = bool(dy_stats['dY_max_context'] > 1e-5)
        out['splits'][name] = st

        sign_inter = 'SPANS ZERO' if st['interaction_ci95'][0] <= 0 <= st['interaction_ci95'][1] else 'excludes 0'
        print(f"--- {name} (eligible={n_eligible}, requested={a.n_eval}, bound={n_eval_bound}, evaluated={st['n_rows']}) ---", flush=True)
        print(f"  Scores (median Pearson) : S11={st['scores']['S11']:+.5f}  S01={st['scores']['S01']:+.5f}  S10={st['scores']['S10']:+.5f}  S00={st['scores']['S00']:+.5f}", flush=True)
        print(f"  Atom effect (with ctx)  : {st['atom_effect_with_context']:+.5f}  CI95 [{st['atom_effect_with_context_ci95'][0]:+.5f}, {st['atom_effect_with_context_ci95'][1]:+.5f}]", flush=True)
        print(f"  Atom effect (no ctx)    : {st['atom_effect_without_context']:+.5f}  CI95 [{st['atom_effect_without_context_ci95'][0]:+.5f}, {st['atom_effect_without_context_ci95'][1]:+.5f}]", flush=True)
        print(f"  Context effect (w/ atoms: {st['context_effect_with_atoms']:+.5f} | w/o atoms: {st['context_effect_without_atoms']:+.5f}", flush=True)
        print(f"  INTERACTION             : {st['interaction']:+.5f}  CI95 [{st['interaction_ci95'][0]:+.5f}, {st['interaction_ci95'][1]:+.5f}]  ({sign_inter})", flush=True)
        print(f"  Paired-mean interaction : {st['interaction_paired_mean']:+.5f}  CI95 [{st['interaction_paired_mean_ci95'][0]:+.5f}, {st['interaction_paired_mean_ci95'][1]:+.5f}]", flush=True)
        print(f"  |dY|max                 : atom={st['dY_max']['dY_max_atom']:.4f}  context={st['dY_max']['dY_max_context']:.4f}\n", flush=True)

    with io.open(dst_json, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(out, indent=2) + '\n')
    print(f"Wrote 2x2 interaction results to {dst_json}", flush=True)


if __name__ == '__main__':
    main()
