# TASK W7 — build the unioned gene-gene graph with multi-hot provenance

Precisely specified. **Do not redesign it, and do not touch the model.** This task builds a data artefact
and nothing else. Repo root `C:\Projects\LINCS`.

## RUN EVERY COMMAND IN THE FOREGROUND AND BLOCK ON IT
A previous worker on this codebase launched its verification in the background, went idle, and its session
terminated it on exit — exit code 0, nothing accomplished. **Do not background anything.** Write your
report last, only after your verification has actually returned output.

## Why this exists
`model/v9` uses two biological graphs in two completely separate places: STRING as one message-passing
step over a 978x978 adjacency (`model_v9.py:99,145`), and Reactome only as an auxiliary-loss target
(`model_v9.py:180-186`). They have never been combined. A third source, GO:BP, sits unused on disk.
TxPert (arXiv:2505.14919) unions its graphs and attaches a **multi-hot edge feature** saying which source
each edge came from, and reports that combining helps monotonically. This task produces that union for our
978 landmark genes so a later arm can consume it. **That later arm is not part of this task.**

## Inputs, all already on disk — download nothing
```
network/outputs/v9/STRING_adj_978_v9.npy    (978,978) float32, weights in [0,0.999], 2.72% nonzero
network/outputs/v9/M_pathway_v9.npy         (800,978)  int8   Reactome membership, 1.78% nonzero
network/outputs/v9/pathway_info_v9.tsv              Reactome pathway names/ids for those 800 rows
network/outputs/v9/landmark_symbols_v9.tsv          the 978 landmark gene symbols IN MODEL ORDER
network/data/GO_Biological_Process_2023.gmt         5,406 GO:BP terms, gene SYMBOL gmt
```
**The 978 row order in `landmark_symbols_v9.tsv` is authoritative and must be preserved exactly.** Every
matrix v9 loads is in that order; an artefact in any other order would pair gene i with gene j and produce
plausible, wrong numbers. Assert the order rather than assuming it.

**Do NOT use** `drug/outputs/dti/dti_reference.tsv`. It is bipartite drug-to-gene, not gene-to-gene, so it
does not belong in this union — and it is a held-out validation set.

## What to build: `network/scripts/build_union_graph.py`

Three gene-gene edge sets over the 978 landmarks, then their union.

### Source 1 — STRING
Edges where `STRING_adj_978_v9[i,j] > --string_thresh` (default `0.0`, i.e. every nonzero). Keep the
weight. Symmetric; emit each undirected edge once with `i < j`.

### Source 2 — Reactome co-pathway
Two landmark genes share an edge if they co-occur in at least one of the 800 Reactome pathways.
**Filter terms by size first:** drop pathways with fewer than `--min_term` (default 5) or more than
`--max_term` (default 200) landmark members. A 400-gene pathway creates a 79,800-edge clique that encodes
almost no specificity, and `A_copathway.npy` from the v5 era is already 12.83 % dense, which is the
symptom of exactly this. Edge weight = number of shared surviving pathways.

### Source 3 — GO:BP co-annotation
Parse the GMT (tab-separated: `term_name<TAB><description><TAB>gene1<TAB>gene2...`; note the description
field is often empty, so **do not assume a fixed number of leading columns — split on tab and treat the
first field as the name, the second as description, the rest as genes**). Map symbols to the 978 landmark
order, discard non-landmark genes, then apply the same `--min_term` / `--max_term` size filter to the
**landmark-restricted** term, and build co-annotation edges the same way. Edge weight = number of shared
surviving terms.

### The union
Union of the three edge sets. For each edge emit:
- `edge_index` `(2, E)` int32, with `i < j` — undirected, each edge once
- `provenance` `(E, 3)` uint8 — multi-hot `[string, reactome, go]`
- `weights` `(E, 3)` float32 — each source's own weight, `0.0` where that source has no such edge.
  **Do not normalise the three weights onto one scale; they are different quantities and a consumer
  should decide.** Say so in the docstring.

Write `network/outputs/v9/union_graph_v9.npz` with those three arrays plus `nodes` (the 978 symbols in
model order) and `n_nodes`.

Write `network/outputs/v9/union_graph_v9_provenance.json` containing, and **print the same table**:
- for each source: raw term/edge count, terms surviving the size filter, edges contributed, density
- edges in the union; edges in each of the 7 non-empty provenance combinations
- the Jaccard overlap between each pair of sources
- every parameter used, and the sha1 of each input file
- `landmark_order_sha1` of `landmark_symbols_v9.tsv`

CLI: `--string_thresh` (0.0), `--min_term` (5), `--max_term` (200), `--out_dir`
(`network/outputs/v9`). Print progress with `flush=True`.

## Tests — `network/scripts/test_union_graph.py`
This codebase has been burned by tests that checked the wrong property: one asserted a quantiser's
`fitted == 1.0` while its bins were NaN and every input silently bucketed to zero. Another counted
parameters once, printed the number, and asserted nothing. **Every check below must be able to fail.**

1. **Node order is preserved.** `nodes` from the npz equals `landmark_symbols_v9.tsv` element-for-element.
2. **STRING round-trip.** For 200 randomly chosen `i<j` pairs, an edge with `provenance[:,0]==1` exists
   **iff** `STRING_adj_978_v9[i,j] > thresh`, and its `weights[:,0]` equals that entry. Assert both
   directions — presence AND absence.
3. **Reactome round-trip.** For 50 random surviving pathways, every co-member pair appears in the union
   with `provenance[:,1]==1`.
4. **No self-loops, no duplicates, `i < j` everywhere.**
5. **The size filter actually removed something**, and the count matches the provenance JSON. If nothing
   was filtered at the default settings, say so loudly rather than passing quietly.
6. **Union is a superset of each source**, and `provenance.sum(axis=1) >= 1` for every edge.
7. **Multi-hot is not degenerate:** at least one edge has provenance summing to 2 or more, and at least
   one edge is unique to each source. Print all seven combination counts.

## Verification you must run, in the foreground
```
cd C:\Projects\LINCS
python network\scripts\build_union_graph.py
python network\scripts\test_union_graph.py
```
Paste the full output of both. Then also paste the seven provenance-combination counts and the per-source
densities, because those numbers are the point of the artefact, not a by-product.

## Constraints
- Create only `network/scripts/build_union_graph.py` and `network/scripts/test_union_graph.py`, plus the
  two output files. **Modify no existing file.** **Do not touch anything under `model/`.**
- No git. No GPU. No training. No network access — every input is local.
- If a 978x978x3 dense array would be more convenient for you, do not build one as the artefact; the edge
  list is the deliverable. You may use dense intermediates internally.
- Match the codebase register: docstrings state *why* a choice was made and cite the measurement or review
  that forced it.

## Report
`C:\Projects\LINCS\research\W7_union_graph_REPORT.md`: the two files' key sections, the real output of both
commands, the provenance table, and a section `## What I was unsure about` naming anything you guessed at
— in particular any ambiguity in the GMT parsing or in symbol-to-index mapping, including how many GO
symbols failed to map and whether any mapped to more than one index.
