# -*- coding: utf-8 -*-
"""
step5_dti_chembl.py -- curated drug->target (DTI) from ChEMBL, for the learn-then-validate plan.
Maps our drugs (pert_id) to ChEMBL by InChIKey, pulls CURATED MECHANISMS keyed by the PARENT
molecule (so salt-attached mechanisms are not missed), resolves each target to a UniProt accession
-> gene symbol, and flags membership in the 978 landmark genes.

Outputs (outputs/dti/):
  chembl_dti_edges.tsv  pert_id, pert_iname, molecule_chembl_id, parent_chembl_id, target_chembl_id,
                        uniprot, gene_symbol, entrez, target_type, organism, action_type,
                        direct_interaction, is_landmark, mechanism_of_action
  chembl_dti_summary.json
Cache: outputs/dti/_chembl_ik2mol.json (inchikey -> [molecule_chembl_id, parent_chembl_id]).
Only stdlib + network. Run with any python (uses ../outputs/dti from step4).
"""
import urllib.request, json, csv, time, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

CB = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
OUT = "../outputs/dti"
CACHE = f"{OUT}/_chembl_ik2mol.json"
UA = {"User-Agent": "lincs-dti-research/1.0"}

def http_json(url, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if a == tries - 1:
                print("  ! give up:", url[:90], repr(e)); raise
            time.sleep(1.5 * (a + 1))

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chembl_list(endpoint, params, key, limit=1000):
    """Fetch a ChEMBL list endpoint, following pagination."""
    url = f"{CB}/{endpoint}.json?{params}&limit={limit}"
    out = []
    while url:
        d = http_json(url); out += d.get(key, [])
        nxt = d.get("page_meta", {}).get("next")
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        time.sleep(0.08)
    return out

# ---- inputs ----
ik_rows = list(csv.DictReader(open(f"{OUT}/drug_inchikeys.tsv", encoding="utf-8"), delimiter="\t"))
pid2ik = {r["pert_id"]: r["inchikey"] for r in ik_rows if r["inchikey"]}
pid2iname = {r["pert_id"]: r["pert_iname"] for r in
             csv.DictReader(open("../outputs/drug_list.tsv", encoding="utf-8"), delimiter="\t")}
lm = list(csv.DictReader(open(f"{OUT}/landmark_genes.tsv", encoding="utf-8"), delimiter="\t"))
lm_syms = {r["symbol"] for r in lm}
sym2entrez = {r["symbol"]: r["entrez"] for r in lm}
uniq_iks = sorted(set(pid2ik.values()))
print(f"drugs {len(pid2ik):,} | unique InChIKeys {len(uniq_iks):,} | landmark genes {len(lm_syms)}")

# ---- 1) InChIKey -> (molecule, parent) via ChEMBL (cached) ----
ik2mol = {}
if os.path.exists(CACHE):
    ik2mol = json.load(open(CACHE)); print(f"cache: {len(ik2mol):,} inchikeys mapped")
todo = [ik for ik in uniq_iks if ik not in ik2mol]
print(f"querying ChEMBL molecule for {len(todo):,} inchikeys ...")
for bi, batch in enumerate(chunks(todo, 45)):
    q = "molecule_structures__standard_inchi_key__in=" + ",".join(batch)
    mols = chembl_list("molecule", q + "&only=molecule_chembl_id,molecule_structures,molecule_hierarchy", "molecules")
    for m in mols:
        ik = (m.get("molecule_structures") or {}).get("standard_inchi_key")
        if not ik:
            continue
        h = m.get("molecule_hierarchy") or {}
        ik2mol[ik] = [m["molecule_chembl_id"], h.get("parent_chembl_id") or m["molecule_chembl_id"]]
    if bi % 20 == 0:
        json.dump(ik2mol, open(CACHE, "w")); print(f"  molecule batch {bi} | mapped {len(ik2mol):,}")
json.dump(ik2mol, open(CACHE, "w"))
matched_iks = [ik for ik in uniq_iks if ik in ik2mol]
print(f"InChIKeys found in ChEMBL: {len(matched_iks):,}/{len(uniq_iks):,}")

# ---- 2) mechanisms by parent molecule ----
parent2pids = defaultdict(list)
for pid, ik in pid2ik.items():
    if ik in ik2mol:
        parent2pids[ik2mol[ik][1]].append(pid)
parents = sorted(parent2pids)
print(f"unique parent molecules: {len(parents):,} ; querying mechanisms ...")
mechs = []
for batch in chunks(parents, 45):
    mechs += chembl_list("mechanism", "parent_molecule_chembl_id__in=" + ",".join(batch), "mechanisms")
print(f"mechanism records: {len(mechs):,}")

# ---- 3) targets -> UniProt accession ----
tgts = sorted({m["target_chembl_id"] for m in mechs if m.get("target_chembl_id")})
tgt_acc, tgt_meta = {}, {}
for batch in chunks(tgts, 45):
    for t in chembl_list("target", "target_chembl_id__in=" + ",".join(batch), "targets"):
        tgt_acc[t["target_chembl_id"]] = [c.get("accession") for c in (t.get("target_components") or []) if c.get("accession")]
        tgt_meta[t["target_chembl_id"]] = (t.get("target_type", ""), t.get("organism", ""))
print(f"unique targets: {len(tgts):,}")

# ---- 4) UniProt accession -> gene symbol (batch, per-acc fallback) ----
accs = sorted({a for al in tgt_acc.values() for a in al})
acc2sym = {}
for batch in chunks(accs, 90):
    try:
        d = http_json(f"{UNIPROT}/accessions?accessions=" + ",".join(batch) + "&fields=accession,gene_primary&format=json")
        for e in d.get("results", []):
            acc = e.get("primaryAccession")
            g = ((e.get("genes") or [{}])[0].get("geneName") or {}).get("value")
            if acc:
                acc2sym[acc] = g or ""
    except Exception:
        for a in batch:
            try:
                e = http_json(f"{UNIPROT}/{a}.json")
                acc2sym[a] = ((e.get("genes") or [{}])[0].get("geneName") or {}).get("value", "")
            except Exception:
                acc2sym[a] = ""
print(f"UniProt accessions resolved: {sum(1 for a in accs if acc2sym.get(a))}/{len(accs)}")

# ---- 5) build edges ----
edges = []
for m in mechs:
    par = m.get("parent_molecule_chembl_id"); tgt = m.get("target_chembl_id")
    if par not in parent2pids or not tgt:
        continue
    ttype, torg = tgt_meta.get(tgt, ("", ""))
    for pid in parent2pids[par]:
        mol = ik2mol[pid2ik[pid]][0]
        for acc in (tgt_acc.get(tgt) or [""]):
            sym = acc2sym.get(acc, "")
            edges.append([pid, pid2iname.get(pid, ""), mol, par, tgt, acc, sym,
                          sym2entrez.get(sym, ""), ttype, torg, m.get("action_type", ""),
                          m.get("direct_interaction", ""), int(sym in lm_syms),
                          (m.get("mechanism_of_action") or "").replace("\t", " ")])

hdr = ["pert_id", "pert_iname", "molecule_chembl_id", "parent_chembl_id", "target_chembl_id",
       "uniprot", "gene_symbol", "entrez", "target_type", "organism", "action_type",
       "direct_interaction", "is_landmark", "mechanism_of_action"]
with open(f"{OUT}/chembl_dti_edges.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(hdr); w.writerows(edges)

lm_edges = [e for e in edges if e[12] == 1]
drugs_any = {e[0] for e in edges}
drugs_lm = {e[0] for e in lm_edges}
summary = {
    "drugs_total": len(pid2ik), "unique_inchikeys": len(uniq_iks),
    "inchikeys_in_chembl": len(matched_iks), "mechanism_records": len(mechs),
    "unique_targets": len(tgts), "edges_total": len(edges), "edges_landmark": len(lm_edges),
    "drugs_with_any_target": len(drugs_any), "drugs_with_landmark_target": len(drugs_lm),
    "landmark_genes_hit": len({e[6] for e in lm_edges}),
}
json.dump(summary, open(f"{OUT}/chembl_dti_summary.json", "w"), indent=2)
print("SUMMARY:", json.dumps(summary, indent=2))
print("wrote chembl_dti_edges.tsv")
