# -*- coding: utf-8 -*-
"""
v9 priors, rebuilt over the FULL PROTEOME instead of the 978 landmarks.

Three defects are fixed here, and only one of them is the one V9_HANDOFF.md §C names.

1. STRING WAS TRULY TRUNCATED (the handoff is right about this one). We kept only landmark-to-landmark
   edges: 12,665 edges, 66 landmarks with no edge at all. Rebuilt over the whole proteome the same
   confidence threshold gives ~929k edges over ~19.5k genes and the isolated count falls to the number of
   landmarks STRING has never heard of.

2. REACTOME WAS NOT TRUNCATED, AND THE HANDOFF'S DIAGNOSIS IS WRONG. Measured: ReactomePathways.gmt
   contains 11,963 genes in total, and 231 of our landmarks appear in NO pathway at ANY filter setting
   (checked at min_size=1 with the umbrella exclusion off: coverage caps at 747/978). It is an annotation
   gap in the source, not something our filtering deleted, so "231 -> ~0" is unreachable from Reactome
   alone. GO Biological Process is therefore added as a SECOND named source, which takes the orphan count
   to 45. Every node still carries a curated human-readable name, and every node records which source it
   came from, so no pathway claim can silently mix them.

3. OUR GENE SYMBOLS ARE FROM 2012 AND THE ANNOTATION SOURCES ARE NOT. 34 of the 978 L1000 landmark symbols
   have since been renamed by HGNC (AARS -> AARS1, IKBKAP -> ELP1, KIAA0196 -> WASHC5, ...). Joining on the
   stale string silently drops those genes from every external source. Resolution goes through the Entrez
   id that L1000's own gene_info carries, against the local HGNC complete set; all 978 resolve, with no
   collisions. This alone moves STRING-absent 28 -> 8 and Reactome orphans 231 -> 213.

Outputs (network/outputs/v9/):
    landmark_symbols_v9.tsv     canonical order -> l1000 symbol, entrez, current HGNC symbol
    STRING_adj_978_v9.npy       [978, 978] float32, the induced landmark subgraph (comparable to the old)
    string_graph_v9.npz         full-proteome graph: node symbols, edge_index, weights -- for pretraining
    M_pathway_v9.npy            [P, 978] int8 membership, named nodes
    pathway_info_v9.tsv         row p <-> (source, term_id, name, landmark members) -- VERIFIED against M
    priors_v9_provenance.json   source hashes, every count above, and the gate outcomes

    python network/scripts/build_priors_v9.py
"""
import os, csv, json, hashlib, argparse, time
from collections import defaultdict

import numpy as np

csv.field_size_limit(10 ** 9)
ROOT = r'C:\Projects\LINCS'
DATA = os.path.join(ROOT, 'network', 'data')
OUT = os.path.join(ROOT, 'network', 'outputs', 'v9')
GENE_INFO = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_gene_info.txt',
                         'GSE92742_Broad_LINCS_gene_info.txt')
GENE_ORDER = os.path.join(ROOT, 'Data Info', 'pathway_landmark_genes.txt')
HGNC = os.path.join(ROOT, 'baseline', 'hgnc_complete_set.txt')


def sha256(path, cap=64 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b or h.block_size and f.tell() > cap:
                break
            h.update(b)
    return h.hexdigest()


def resolve_symbols():
    """L1000 landmark symbols -> current HGNC symbols, via the Entrez id L1000 itself provides."""
    sym2ent = {}
    with open(GENE_INFO, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pr_is_lm'] == '1':
                sym2ent[r['pr_gene_symbol']] = r['pr_gene_id']
    ent2cur, prev2cur = {}, {}
    with open(HGNC, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['status'] != 'Approved':
                continue
            if r['entrez_id']:
                ent2cur[r['entrez_id']] = r['symbol']
            for p in (r['prev_symbol'] or '').split('|'):
                if p:
                    prev2cur.setdefault(p, r['symbol'])
    order = [l.strip() for l in open(GENE_ORDER, encoding='utf-8') if l.strip()]
    cur, ent, how = [], [], []
    for g in order:
        e = sym2ent.get(g, '')
        if e in ent2cur:
            c, w = ent2cur[e], ('same' if ent2cur[e] == g else 'entrez')
        elif g in prev2cur:
            c, w = prev2cur[g], 'prev_symbol'
        else:
            c, w = g, 'unresolved'
        cur.append(c); ent.append(e); how.append(w)
    assert len(set(cur)) == len(cur), 'symbol resolution collapsed two landmarks onto one symbol'
    return order, cur, ent, how


def build_string(cur, thresh):
    ensp2sym = {}
    with open(os.path.join(DATA, '9606.protein.info.v12.0.txt'), encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip('\n').split('\t')
            ensp2sym[p[0]] = p[1]
    nodes, edges, w = {}, [], []
    with open(os.path.join(DATA, '9606.protein.links.v12.0.txt'), encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            x, y, s = line.split()
            s = int(s)
            if s < thresh:
                continue
            sx, sy = ensp2sym.get(x), ensp2sym.get(y)
            if sx is None or sy is None or sx == sy:
                continue
            if sx > sy:                                  # keep one direction; the graph is undirected
                continue
            for t in (sx, sy):
                if t not in nodes:
                    nodes[t] = len(nodes)
            edges.append((nodes[sx], nodes[sy])); w.append(s / 1000.0)
    # every landmark must be a node even if isolated, so the 978 rows always exist
    for g in cur:
        if g not in nodes:
            nodes[g] = len(nodes)
    names = [None] * len(nodes)
    for g, i in nodes.items():
        names[i] = g
    E = np.array(edges, np.int32).T
    W = np.array(w, np.float32)
    gi = {g: i for i, g in enumerate(cur)}
    A = np.zeros((len(cur), len(cur)), np.float32)
    for (u, v), ww in zip(E.T, W):
        a, b = names[u], names[v]
        if a in gi and b in gi:
            A[gi[a], gi[b]] = A[gi[b], gi[a]] = ww
    return names, E, W, A


def load_gmt(path, name_of, id_of, gene_from):
    sets = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 3:
                continue
            genes = {x for x in f[gene_from:] if x}
            if genes:
                sets[id_of(f)] = (name_of(f), genes)
    return sets


def select_nodes(cand, cur, max_nodes, min_landmarks):
    """Greedy set cover over the landmarks first -- so coverage is guaranteed rather than hoped for --
    then fill the remaining budget with the highest-landmark-count terms."""
    lm = set(cur)
    usable = {k: (src, nm, g & lm) for k, (src, nm, g) in cand.items() if len(g & lm) >= min_landmarks}
    chosen, covered = [], set()
    pool = dict(usable)
    while pool and len(chosen) < max_nodes:
        k = max(pool, key=lambda k: (len(pool[k][2] - covered), len(pool[k][2])))
        gain = len(pool[k][2] - covered)
        if gain == 0:
            break
        chosen.append(k); covered |= pool[k][2]; pool.pop(k)
    rest = sorted(pool, key=lambda k: -len(pool[k][2]))
    chosen += rest[:max(0, max_nodes - len(chosen))]
    return chosen, usable, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--string_thresh', type=int, default=400,
                    help='STRING combined_score cut. 400 reproduces the field default: it yields ~929k '
                         'edges over ~19.5k genes, against XPert\'s released 901,260 over 19,392.')
    ap.add_argument('--max_nodes', type=int, default=800)
    ap.add_argument('--min_landmarks', type=int, default=1,
                    help='greedy set cover runs first, so a 1-landmark term is only ever chosen '
                         'when it is the ONLY thing covering an otherwise orphan gene')
    ap.add_argument('--pathway_size', default='5,500', help='min,max TOTAL genes per term. Measured '
                    'orphan floor over both sources is 45; [5,500] gives 50, [5,300] gives 59, and widening past 500 only admits contentless umbrella terms.')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    lo, hi = [int(x) for x in a.pathway_size.split(',')]

    # ---------------- 1. symbols ----------------
    order, cur, ent, how = resolve_symbols()
    n_changed = sum(1 for x, y in zip(order, cur) if x != y)
    print('symbols: %d/978 renamed since L1000 (%s)' % (n_changed, ', '.join(
        f'{x}->{y}' for x, y in list(zip(order, cur)) if x != y)[:110] + '...'))
    with open(os.path.join(OUT, 'landmark_symbols_v9.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['canonical_row', 'l1000_symbol', 'entrez_id', 'hgnc_symbol', 'resolved_via'])
        for i, (o, c, e, h) in enumerate(zip(order, cur, ent, how)):
            w.writerow([i, o, e, c, h])

    # ---------------- 2. STRING, full proteome ----------------
    names, E, W, A = build_string(cur, a.string_thresh)
    deg_new = (A != 0).sum(1)
    old = np.load(os.path.join(ROOT, 'network', 'outputs', 'STRING_adj_978.npy'))
    iso_old = int(((old != 0).sum(1) == 0).sum())
    iso_new = int((deg_new == 0).sum())
    full_deg = defaultdict(int)
    for u, v in E.T:
        full_deg[u] += 1; full_deg[v] += 1
    idx = {g: i for i, g in enumerate(names)}
    iso_full = int(sum(1 for g in cur if full_deg[idx[g]] == 0))
    print('STRING @%d: %d nodes, %d undirected edges (XPert released 19,392 / 901,260)'
          % (a.string_thresh, len(names), E.shape[1]))
    print('  isolated landmarks -- old landmark-truncated graph: %d | induced 978x978 now: %d | '
          'in the FULL graph: %d' % (iso_old, iso_new, iso_full))
    np.save(os.path.join(OUT, 'STRING_adj_978_v9.npy'), A)
    np.savez_compressed(os.path.join(OUT, 'string_graph_v9.npz'), nodes=np.array(names),
                        edge_index=E, weight=W,
                        landmark_idx=np.array([idx[g] for g in cur], np.int32))

    # ---------------- 3. pathways: Reactome + GO-BP, full gene sets ----------------
    rea = load_gmt(os.path.join(DATA, 'ReactomePathways.gmt'), lambda f: f[0], lambda f: f[1], 2)
    gop = os.path.join(DATA, 'GO_Biological_Process_2023.gmt')
    go = load_gmt(gop, lambda f: f[0].rsplit(' (GO:', 1)[0],
                  lambda f: 'GO:' + f[0].rsplit(' (GO:', 1)[1].rstrip(')'), 2) if os.path.exists(gop) else {}
    lm = set(cur)
    cov_r = len(set().union(*[g for _, g in rea.values()]) & lm)
    cov_g = len(set().union(*[g for _, g in go.values()]) & lm) if go else 0
    print('Reactome: %d terms, covers %d/978 landmarks | GO-BP: %d terms, covers %d/978'
          % (len(rea), cov_r, len(go), cov_g))

    cand = {}
    for src, sets in [('Reactome', rea), ('GO:BP', go)]:
        for k, (nm, g) in sets.items():
            if lo <= len(g) <= hi:
                cand[(src, k)] = (src, nm, g)
    chosen, usable, covered = select_nodes(cand, cur, a.max_nodes, a.min_landmarks)
    chosen.sort()
    gi = {g: i for i, g in enumerate(cur)}
    M = np.zeros((len(chosen), len(cur)), np.int8)
    rows = []
    for i, k in enumerate(chosen):
        src, nm, mem = usable[k]
        for g in mem:
            M[i, gi[g]] = 1
        rows.append([src, k[1], nm, len(mem), ','.join(sorted(mem))])
    bad = [i for i, k in enumerate(chosen)
           if set(np.flatnonzero(M[i]).tolist()) != {gi[g] for g in usable[k][2]}]
    assert not bad, 'row/name mismatch on %d rows -- refusing to write' % len(bad)

    orph = int((M.sum(0) == 0).sum())
    orph_rea = 978 - cov_r
    print('pathway nodes: %d (%d Reactome, %d GO:BP) | landmarks with no node: %d  '
          '(Reactome alone would leave %d)' % (len(chosen), sum(1 for k in chosen if k[0] == 'Reactome'),
                                               sum(1 for k in chosen if k[0] == 'GO:BP'), orph, orph_rea))
    np.save(os.path.join(OUT, 'M_pathway_v9.npy'), M)
    with open(os.path.join(OUT, 'pathway_info_v9.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['source', 'term_id', 'term_name', 'n_landmark_genes', 'landmark_gene_symbols'])
        w.writerows(rows)

    # ---------------- 4. provenance + the gate outcomes ----------------
    prov = dict(
        created=time.strftime('%Y-%m-%d %H:%M'), seconds=round(time.time() - t0, 1),
        sources={p: dict(sha256_first64MB=sha256(os.path.join(DATA, p)),
                         bytes=os.path.getsize(os.path.join(DATA, p)))
                 for p in ['9606.protein.links.v12.0.txt', '9606.protein.info.v12.0.txt',
                           'ReactomePathways.gmt'] + (['GO_Biological_Process_2023.gmt'] if go else [])},
        hgnc=dict(path=HGNC, sha256_first64MB=sha256(HGNC)),
        symbols=dict(renamed=n_changed, unresolved=how.count('unresolved')),
        string=dict(threshold=a.string_thresh, nodes=len(names), undirected_edges=int(E.shape[1]),
                    xpert_reference=dict(nodes=19392, edges=901260),
                    isolated_landmarks=dict(old_truncated=iso_old, induced_978=iso_new, full_graph=iso_full)),
        pathways=dict(nodes=len(chosen), reactome=sum(1 for k in chosen if k[0] == 'Reactome'),
                      go_bp=sum(1 for k in chosen if k[0] == 'GO:BP'),
                      size_filter=[lo, hi], min_landmarks=a.min_landmarks,
                      landmarks_covered=int((M.sum(0) > 0).sum()), orphans=orph,
                      reactome_only_orphans=orph_rea),
        gates=dict(
            string_isolated=dict(handoff='66 -> ~0', measured_from=iso_old, measured_to=iso_full,
                                 met=iso_full <= 10,
                                 note='residual are landmarks with no STRING entry under any symbol'),
            reactome_orphans=dict(handoff='231 -> ~0', reactome_only=orph_rea, with_go_bp=orph,
                                  met=False,
                                  note='NOT reachable from Reactome: only 11,963 genes are annotated at '
                                       'all, so 213 landmarks are absent at any filter. GO:BP added as a '
                                       'second NAMED source takes it to %d.' % orph)))
    json.dump(prov, open(os.path.join(OUT, 'priors_v9_provenance.json'), 'w'), indent=2)
    print('\n-> %s' % OUT)


if __name__ == '__main__':
    main()
