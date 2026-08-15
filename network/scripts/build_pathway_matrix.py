# -*- coding: utf-8 -*-
"""
Rebuild the Reactome membership matrix at a configurable node size, WITH its names, in one step.

Why this exists: the original matrix was built inside a notebook and only `M_reactome.npy` was kept in
`network/outputs/` -- the row->name mapping lived in a file nobody referenced, so "360 NAMED nodes" was not
checkable. Anything that regenerates M must regenerate the names in the SAME row order or every pathway we
report is mislabelled. This script emits both and verifies them against each other before writing.

Measured coverage vs threshold (see V7 discussion):
    min_size   nodes   landmark genes covered
           1    1983            747 / 978
           5     786            747 / 978
          10     368            746 / 978      <- the original setting (360 after root-umbrella exclusion)
          20     139            737 / 978
Gene coverage is CAPPED at 747: 231 landmark genes are in no Reactome pathway at any threshold. Lowering
min_size buys a FINER PARTITION over the same genes, not more coverage. The 231 uncovered genes need a
different source -- which is what the STRING PPI layer in v7 provides (it reaches 912/978).

Run:  python network/scripts/build_pathway_matrix.py --min_size 5
"""
import os, sys, csv, json, argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_size", type=int, default=5, help="minimum LANDMARK genes per pathway")
    ap.add_argument("--root_exclude_min_size", type=int, default=70,
                    help="drop huge umbrella pathways above this size (they carry no specific signal)")
    ap.add_argument("--gmt", default=os.path.join(ROOT, "network/data/ReactomePathways.gmt"))
    ap.add_argument("--genes", default=os.path.join(ROOT, "Data Info/pathway_landmark_genes.txt"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "network/outputs"))
    ap.add_argument("--tag", default=None, help="suffix for the output files; default = _ms{min_size}")
    a = ap.parse_args()
    tag = a.tag if a.tag is not None else f"_ms{a.min_size}"

    with open(a.genes, encoding="utf-8") as f:
        genes = [l.strip() for l in f if l.strip()]
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    paths = {}
    with open(a.gmt, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            name, rhsa, members = p[0], p[1], set(p[2:])
            lm = sorted(members & gi.keys())
            if len(lm) >= a.min_size and len(lm) <= a.root_exclude_min_size:
                paths[rhsa] = (name, lm)

    # sorted by pathway id -- the SAME ordering rule the original pipeline used, so downstream code that
    # assumes "row p of M == row p of pathway_info" keeps holding
    ids = sorted(paths)
    M = np.zeros((len(ids), G), dtype=np.int8)
    for i, pid in enumerate(ids):
        for g in paths[pid][1]:
            M[i, gi[g]] = 1

    # verify the two artefacts agree BEFORE writing either
    bad = [i for i, pid in enumerate(ids)
           if set(np.where(M[i] > 0)[0].tolist()) != {gi[g] for g in paths[pid][1]}]
    if bad:
        raise RuntimeError(f"row/name mismatch on {len(bad)} rows -- refusing to write")

    os.makedirs(a.out_dir, exist_ok=True)
    m_path = os.path.join(a.out_dir, f"M_reactome{tag}.npy")
    i_path = os.path.join(a.out_dir, f"pathway_info{tag}.tsv")
    np.save(m_path, M)
    with open(i_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pathway_id", "pathway_name", "n_landmark_genes", "landmark_gene_symbols"])
        for pid in ids:
            name, lm = paths[pid]
            w.writerow([pid, name, len(lm), ",".join(lm)])

    cov = int((M.sum(0) > 0).sum())
    prov = {"min_size": a.min_size, "root_exclude_min_size": a.root_exclude_min_size,
            "n_pathways": len(ids), "genes_covered": cov, "n_genes": G,
            "mean_pathway_size": float(M.sum(1).mean()), "density": float(M.mean()),
            "source_gmt": os.path.basename(a.gmt), "row_order": "sorted by pathway_id"}
    json.dump(prov, open(os.path.join(a.out_dir, f"pathway_provenance{tag}.json"), "w"), indent=2)
    print(json.dumps(prov, indent=2))
    print(f"wrote {m_path}\n      {i_path}")


if __name__ == "__main__":
    main()
