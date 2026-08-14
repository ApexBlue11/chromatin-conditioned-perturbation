# -*- coding: utf-8 -*-
"""
MoA probe: does v6's NAMED pathway layer carry drug-specific mechanism? Local CPU, no retrain.

THE POINT. v6's pathway *activations* are drug-invariant by construction -- a completely different molecule
changes them by exactly 0.0 [4.12] -- because the layer sits above where the drug enters. But P-NET reads
node **importance**, not raw activation, and the drug DOES reach the node through the gradient: measured
`d(Yhat)/da_p` changes by 0.076 under a different drug [4.14]. So a per-drug pathway attribution is already
available from any v6 checkpoint. This script computes it and tests it properly.

WHY NOT INTEGRATED GRADIENTS, despite ARCHITECTURE_LESSONS asking for an axiomatic method. Layer-IG
attributes `(a_p - a_p^baseline) * integral(grad)`. With a mean-drug baseline the activation term is
**identically zero** here (the activation does not depend on the drug), so Layer-IG returns 0 for every
pathway and every drug. It is the wrong tool for an intermediate node the input cannot move. The meaningful
first-order quantity is gradient x activation, contrasted against the mean drug:

    imp_d[p]      = sum_c a[p,c] * dObj/da[p,c]   under drug d          (first-order share of the output)
    delta_imp[p]  = imp_d[p] - imp_meandrug[p]    <- drug-SPECIFIC pathway engagement

THE GATE, CHECKED FIRST. On an untrained model the delta-free ranking is identical across unrelated drugs
(Spearman +1.0000, top-5 overlap 5/5): the drug enters as a near-uniform SCALE, not a reordering. If that
also holds for the trained checkpoint, the readout cannot express MoA and every alignment number below is
meaningless -- so cross-drug ranking agreement is reported BEFORE any target test, and a near-1 value is
declared a null result rather than worked around.

THE ALIGNMENT TEST, if the gate passes. Independent annotation = curated ChEMBL drug->target edges (never in
the loss). Positive set for drug d = the pathways CONTAINING one of d's target genes. Rank all 360 pathways
by |delta_imp| and report the median rank percentile of the positives (0 = top, 0.5 = chance) -- the same
statistic, on the same reference, that falsified the atom->gene attention claim at 0.560 [4.1a]. Three
controls, all of which the earlier failure taught us to run:
  * label-permutation null -- give each drug another drug's positive set (preserves set sizes).
    **This is the only valid reference point, and 0.5 is NOT it.** Target-containing pathways are
    systematically large and central, so they rank high under any scoring whatsoever: run this on an
    UNTRAINED model and the statistic reads **0.218**, which against "chance = 0.5" would look like a
    striking discovery, while the permutation null sits at **0.229** (p = 0.13, i.e. nothing). Comparing to
    0.5 would let you "find" mechanism in a randomly-initialised network.
  * size-matched null      -- random pathways matched on gene count (big pathways are hit by chance more)
  * "target doesn't move" stratification -- split by whether the target gene is actually responsive in the
    measured data, which is the confound that was tested and REJECTED for attention [4.1b]

Run:  python model/v6/probe_moa_v6.py --ckpt model/results/ckpt_v6_fold0.pt
"""
import os, sys, csv, json, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

from config_v6 import V6Config, V6DataConfig
from data import LincsDataset, collate
from eval_v6 import load_pathway_names, load_v6
from probe_pathways_v6 import spearman, rank


def load_dti(path, n_genes):
    """Curated drug->target edges, split by evidence quality. 'both' is the gold core (ChEMBL AND STITCH)
    and is the tier the attention claim was falsified on."""
    ref = {"chembl": defaultdict(set), "both": defaultdict(set)}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                gi = int(row["gene_idx"])
            except (KeyError, ValueError):
                continue
            if not 0 <= gi < n_genes:
                continue
            ev = (row.get("evidence") or "").lower()
            is_chembl = ("chembl" in ev) or (int(float(row.get("chembl_direct") or 0)) == 1) \
                        or bool((row.get("chembl_actions") or "").strip())
            if is_chembl:
                ref["chembl"][row["pert_id"]].add(gi)
            if ev.startswith("chembl+stitch"):
                ref["both"][row["pert_id"]].add(gi)
    return ref


def objective(y, kind):
    """What we attribute. 'sq' = predicted response ENERGY ||Yhat||^2 -- for MoA we care which pathways
    drive the size of the response, and a plain sum would let opposite-signed genes cancel."""
    return (y ** 2).sum() if kind == "sq" else (y.abs().sum() if kind == "abs" else y.sum())


def importance(model, batch, kind):
    """[P] first-order share of the objective carried by each named pathway node."""
    y, aux = model(batch, return_interp=True)
    a = aux["pathway_activations"]                                   # [B,P,dp]
    g, = torch.autograd.grad(objective(y, kind), a)
    return (a.detach() * g).sum(-1).mean(0)                          # [P]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--subset", default="both", choices=["both", "chembl"])
    ap.add_argument("--max_sig_per_drug", type=int, default=4)
    ap.add_argument("--max_drugs", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--objective", default="sq", choices=["sq", "abs", "signed"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--random_init", action="store_true")
    a = ap.parse_args()

    dc, cfg = V6DataConfig(), V6Config()
    dc.cell_fold = a.fold
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_reactome_path))
    if M.shape[0] > M.shape[1]:
        M = M.T
    pathways = load_pathway_names(dc, M)
    P = M.shape[0]
    sizes = M.sum(1).astype(int)
    print(f"pathway_info.tsv <-> M_reactome row order VERIFIED ({P} named nodes)", flush=True)

    model, ckpt_name = load_v6(dc, cfg, M, a.ckpt, a.fold, a.random_init)
    model.eval()
    if a.random_init:
        # The zero-init output layers make the pathway branch an exact no-op, which blocks the gradient
        # path entirely and makes every importance exactly 0 -- so an untrained model cannot exercise this
        # probe at all. De-zero them, exactly as test_v6 does for the ablation controls.
        with torch.no_grad():
            for lin in [model.pathway.w_out, model.pathway.epi_gate[-1], model.film.net[-1],
                        model.head.epi_head[-1]]:
                lin.weight.normal_(0, 0.1); lin.bias.normal_(0, 0.1)
        print("   (output layers de-zeroed so the branch is live; numbers remain meaningless)", flush=True)

    ds = LincsDataset(dc)
    dindex = json.load(open(R(dc.drug_index_path)))
    ref = load_dti(R("drug/outputs/dti/dti_reference.tsv"), cfg.n_genes)[a.subset]
    rep_mask = ds.strength >= dc.eval_min_strength
    drug_sigs = defaultdict(list)
    for j in np.where(rep_mask)[0]:
        drug_sigs[int(ds.drug_row[j])].append(int(j))

    # ---- mean-drug baseline, over the drugs we will actually score ---------------------------------
    eligible = []
    for pid, tg in ref.items():
        d = dindex.get(pid)
        if d is None or not drug_sigs.get(d):
            continue
        pos = np.unique(np.where(M[:, sorted(tg)].sum(1) > 0)[0])     # pathways containing a target
        if len(pos) == 0 or len(pos) >= P:
            continue                                                   # 243 landmarks are in no pathway
        eligible.append((pid, d, sorted(tg), pos))
    if a.max_drugs:
        eligible = eligible[:a.max_drugs]
    print(f"drugs scored: {len(eligible)}  (subset='{a.subset}', >=1 reproducible signature, "
          f">=1 target in a pathway)", flush=True)
    if len(eligible) < 10:
        print("too few drugs -- aborting"); return

    u_mean = torch.from_numpy(ds.u_feats[[d for _, d, _, _ in eligible]].mean(0))
    atom_sum, atom_n = torch.zeros(cfg.d_atom), 0
    for _, d, _, _ in eligible:
        a0, a1 = int(ds.atom_off[d]), int(ds.atom_off[d + 1])
        at = torch.from_numpy(np.array(ds.atom_reprs[a0:a1], np.float32))
        atom_sum += at.sum(0); atom_n += at.shape[0]
    atom_mean = atom_sum / max(atom_n, 1)

    # ---- per-drug pathway engagement --------------------------------------------------------------
    raw, delta, pids = [], [], []
    with torch.enable_grad():
        for i, (pid, d, tg, pos) in enumerate(eligible):
            rows = drug_sigs[d][:a.max_sig_per_drug]
            b = collate([ds[j] for j in rows], cfg.max_atoms)
            imp = importance(model, b, a.objective)
            b0 = {k: v.clone() for k, v in b.items()}                  # same cells/doses, AVERAGE drug
            b0["u_feats"] = u_mean.unsqueeze(0).expand_as(b0["u_feats"]).clone()
            b0["atoms"] = atom_mean.view(1, 1, -1).expand_as(b0["atoms"]).clone()
            imp0 = importance(model, b0, a.objective)
            raw.append(imp.numpy()); delta.append((imp - imp0).numpy()); pids.append(pid)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(eligible)} drugs", flush=True)
    raw, delta = np.stack(raw), np.stack(delta)                        # [D,P]

    # ---- DEGENERACY CHECK, before the gate ---------------------------------------------------------
    # If the branch contributes nothing, every importance is ~0 and every downstream statistic is a
    # meaningless tie. That is a DIFFERENT finding from "the readout is not drug-specific", and conflating
    # the two is exactly the failure mode eval_v6's |dY|max exists to prevent. It is also, directly, the
    # `mean_pathway` ablation answer.
    scale = float(np.abs(raw).max())
    if scale < 1e-12:
        print(f"\n{'='*80}\nPATHWAY BRANCH IS INERT in this checkpoint: max |importance| = {scale:.2e}.\n"
              f"Every pathway node contributes nothing to the output, so there is no MoA readout to test "
              f"and no ranking to compare. This is a RESULT -- report it, and cross-check it against "
              f"eval_v6's `mean_pathway` ablation, which should show ~0.\n{'='*80}", flush=True)
        json.dump({"ckpt": ckpt_name, "fold": a.fold, "random_init": bool(a.random_init),
                   "branch_inert": True, "max_abs_importance": scale, "n_drugs": len(pids)},
                  open(os.path.join(os.path.dirname(HERE), "results",
                                    f"probe_moa_v6_fold{a.fold}.json"), "w"), indent=2)
        return

    # ---- THE GATE: is the ranking drug-specific at all? -------------------------------------------
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(len(pids)) for j in rng.choice(len(pids), 3) if i != j][:600]
    nanmed = lambda v: float(np.median([x for x in v if np.isfinite(x)])) if any(np.isfinite(v)) else float("nan")
    rho_raw = nanmed([spearman(raw[i], raw[j]) for i, j in pairs])
    rho_del = nanmed([spearman(np.abs(delta[i]), np.abs(delta[j])) for i, j in pairs])
    print(f"\n{'='*80}\nGATE -- cross-drug ranking agreement (1.0 = the readout is the SAME for every drug)")
    print(f"  raw importance          median Spearman = {rho_raw:+.4f}")
    print(f"  drug-specific |delta|   median Spearman = {rho_del:+.4f}")
    gate_ok = rho_del < 0.95
    verdict = ("drug-specific ranking EXISTS; the alignment test below is meaningful" if gate_ok else
               "NULL RESULT: the ranking is drug-INVARIANT. The pathway layer cannot express MoA in this "
               "architecture and every number below is meaningless. Fix the ARCHITECTURE (add a pathway "
               "layer AFTER the perturbation encoder) -- do not reinterpret this away.")
    print(f"  => {verdict}\n{'='*80}", flush=True)

    # ---- alignment against curated targets ---------------------------------------------------------
    # Rank percentiles depend only on the SCORES, which never change across the nulls -- only the positive
    # SET does. Precompute once per drug (0 = ranked top) so 1000 permutations are lookups, not argsorts.
    pct = np.empty_like(delta)
    for i in range(len(eligible)):
        order = np.argsort(-np.abs(delta[i]))
        pct[i, order] = np.arange(P) / P
    allpos = [pos for _, _, _, pos in eligible]
    obs = [float(np.median(pct[i][allpos[i]])) for i in range(len(allpos))]
    med = float(np.median(obs))

    perm = np.empty(a.n_perm)                       # give each drug ANOTHER drug's positive set
    for t in range(a.n_perm):
        sh = rng.permutation(len(allpos))
        perm[t] = np.median([np.median(pct[i][allpos[sh[i]]]) for i in range(len(allpos))])

    # size-matched null: swap each positive pathway for another of similar gene count (large pathways are
    # hit by chance more often, so an unmatched null would flatter us)
    order_by_size = np.argsort(sizes)
    rank_in_size = np.empty(P, int); rank_in_size[order_by_size] = np.arange(P)
    size_null = np.empty(min(a.n_perm, 200))
    for t in range(len(size_null)):
        vals = []
        for i, pos in enumerate(allpos):
            k = rank_in_size[pos]
            lo = np.clip(k - 15, 0, P - 1); hi = np.clip(k + 15, 1, P)
            picked = order_by_size[rng.integers(lo, hi)]
            vals.append(np.median(pct[i][np.unique(picked)]))
        size_null[t] = np.median(vals)

    pval = float((perm <= med).mean())
    print(f"\nTARGET ALIGNMENT (0 = target pathways ranked top)")
    print(f"  observed median rank percentile     = {med:.4f}   over {len(obs)} drugs")
    print(f"  label-permutation null              = {perm.mean():.4f} +- {perm.std():.4f}"
          f"   p(one-sided) = {pval:.4f}")
    print(f"  size-matched null                   = {size_null.mean():.4f} +- {size_null.std():.4f}")
    print(f"\n  !! 0.5 IS NOT THE CHANCE LEVEL HERE. Target-containing pathways are systematically large "
          f"and central,\n     so they rank high under ANY scoring: on an UNTRAINED model this statistic "
          f"reads 0.218 against a\n     permutation null of 0.229. Judge only against the null.")
    print(f"  VERDICT: signal vs null = {med - perm.mean():+.4f} (negative = better than null), p = {pval:.4f}"
          f"  ->  {'SIGNAL' if pval < 0.05 else 'NO SIGNAL - report it as a null, per 4.1a'}")
    print(f"  reference: atom->gene attention scored 0.560 on this same reference and was RETRACTED [4.1a]",
          flush=True)

    # ---- the confound that killed the attention claim: do the targets actually move? ---------------
    strat = {}
    tg_pct = []
    for i, (pid, d, tg, pos) in enumerate(eligible):
        rows = drug_sigs[d][:a.max_sig_per_drug]
        absY = np.abs(np.asarray(ds.Y[ds.y_row[rows]], np.float32)).mean(0)     # [978] measured
        r = rank(absY) / len(absY)
        tg_pct.append(float(np.mean([r[g] for g in tg])))                        # 1.0 = most-moved
    tg_pct = np.array(tg_pct)
    hi, lo = tg_pct >= np.quantile(tg_pct, 0.75), tg_pct <= np.quantile(tg_pct, 0.25)
    for nm, sel in [("target IS responsive (top quartile)", hi), ("target barely moves (bottom)", lo)]:
        if sel.sum() >= 5:
            strat[nm] = float(np.median(np.array(obs)[sel]))
            print(f"  {nm:38s} median pct = {strat[nm]:.4f}  (n={int(sel.sum())})", flush=True)
    corr_conf = spearman(tg_pct, np.array(obs))
    print(f"  corr(target responsiveness, attribution rank) = {corr_conf:+.4f}"
          f"   <- near 0 means the failure is NOT explained by untestable targets [4.1b]", flush=True)

    top = np.argsort(-np.abs(delta).mean(0))[:15]
    out = {"ckpt": ckpt_name, "fold": a.fold, "random_init": bool(a.random_init), "subset": a.subset,
           "objective": a.objective, "n_drugs": len(obs), "n_pathways": P,
           "gate_cross_drug_spearman_raw": rho_raw, "gate_cross_drug_spearman_delta": rho_del,
           "gate_passed": bool(gate_ok),
           "median_rank_percentile": med, "chance": 0.5,
           "permutation_null_mean": float(perm.mean()), "permutation_null_sd": float(perm.std()),
           "p_one_sided": float((perm <= med).mean()),
           "size_matched_null_mean": float(size_null.mean()),
           "stratified_by_target_responsiveness": strat,
           "corr_responsiveness_vs_attribution": corr_conf,
           "attention_reference_4_1a": 0.560,
           "top15_pathways_by_mean_abs_engagement": [
               {"pathway_id": pathways[p][0], "pathway_name": pathways[p][1], "n_genes": int(sizes[p]),
                "mean_abs_delta_importance": float(np.abs(delta).mean(0)[p])} for p in top]}
    dest = os.path.join(os.path.dirname(HERE), "results", f"probe_moa_v6_fold{a.fold}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    print("If this comes out at chance, say so as plainly as 4.1a did. A second falsified interpretability "
          "claim is a result; a quietly reinterpreted one is not.", flush=True)


if __name__ == "__main__":
    main()
