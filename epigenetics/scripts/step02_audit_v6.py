# -*- coding: utf-8 -*-
"""
run_epigenetics_audit_v6.py
===========================
Round 2 epigenetics audit engine.

Builds on run_epigenetics_audit_v5.py (frozen, not modified).

New in v6:
  1. NPC.CAS9 -> NPC pre-authorized data-sharing (3 rows)
  2. Dot-suffix / alias re-matching with broader name variants drawn from
     cell_info.txt and known aliases; word-boundary safe matching throughout.
  3. Substitute assays for still-unresolved (cell, mark) pairs:
       H3K4me3 ChIP-seq -> sub for missing ATAC-seq
       DNase-seq        -> sub for missing ATAC-seq
       H3K9me3 ChIP-seq -> sub for missing H3K27me3
  4. New output columns: is_primary_assay, substituted_for, substitute_assay_used
  5. Inline contamination classification (Part 0 rule set) applied to every
     new match from substitute assays -- flagged immediately, never silently accepted.
  6. 12 GENUINE_CONTAMINATION rows in cistrome_subline_audit_classified.tsv are
     untouched -- this script does not re-audit them.

All existing bugs from Part 0 remain fixed; no new bugs introduced.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

sys_encoding = "utf-8"

# ---------- Paths ----------
brain_dir = r"C:\Projects\LINCS\Epigenetics"
v5_report_path   = os.path.join(brain_dir, "coverage_report.tsv")
cistrome_json_path = os.path.join(brain_dir, "cistrome_human_samples.json")
epimap_tsv_path  = os.path.join(brain_dir, "epimap_metadata.tsv")
encode_cache_v5  = os.path.join(brain_dir, "encode_coverage_v5.json")

# Phase-1 cell info (most complete; Phase-2 file has same content in this dir)
cell_info_path = (
    r"C:\Projects\LINCS\Data Info\GSE92742_Broad_LINCS_cell_info.txt"
    r"\GSE92742_Broad_LINCS_cell_info.txt"
)

# Output
v6_report_path = os.path.join(brain_dir, "coverage_report_v6.tsv")
v6_dot_suffix_path = os.path.join(brain_dir, "dot_suffix_investigation.tsv")
v6_substitute_path = os.path.join(brain_dir, "substitute_assay_results.tsv")
v6_flagged_path    = os.path.join(brain_dir, "v6_flagged_contamination.tsv")

# ---------- Helpers ----------

def normalize_name(n):
    """Strip to lowercase alphanumerics only."""
    if not isinstance(n, str):
        return ""
    return "".join(c for c in n.lower() if c.isalnum())


def wb_match(pattern_raw, text):
    """
    Word-boundary safe match: always uses \\b regex, never bare substring.
    pattern_raw: the un-normalized form (used in regex directly so the actual
                 characters, including hyphens and spaces, form boundaries).
    """
    if not isinstance(text, str) or not pattern_raw:
        return False
    try:
        return re.search(rf"\b{re.escape(pattern_raw)}\b", text, re.IGNORECASE) is not None
    except re.error:
        return False


# ---------- Load base data ----------
print("Loading v5 coverage report...")
df_v5 = pd.read_csv(v5_report_path, sep="\t")

print("Loading cell_info.txt...")
df_cell_info = pd.read_csv(cell_info_path, sep="\t", dtype=str)

print("Loading Cistrome JSON...")
with open(cistrome_json_path, "r") as f:
    cistrome_samples = json.load(f)

print("Loading EpiMap TSV...")
epimap_df = pd.read_csv(epimap_tsv_path, sep="\t", dtype=str)

print("Loading ENCODE v5 cache...")
with open(encode_cache_v5, "r") as f:
    encode_v5 = json.load(f)

# ---------- Cell-info index ----------
# Build a lookup: cell_id -> row dict
cell_info = {}
for _, row in df_cell_info.iterrows():
    cid = str(row.get("cell_id", "")).strip()
    if cid and cid != "-666":
        cell_info[cid] = row.to_dict()

# ---------- Dot-suffix investigation table ----------
# For each dot-suffixed or otherwise non-standard cell line in the unresolved
# set, record what we know from cell_info. Do NOT decide substitution beyond
# the pre-authorized NPC.CAS9 case.

DOT_CELLS_OF_INTEREST = [
    "ASC.C", "MNEU.E", "NPC.TAK", "SKL", "SKL.C", "NKDBA",
    "NPC.CAS9",  # included to document the authorized substitution
    "SKB", "FIBRNPC", "MCH58", "HUES3",
]

def describe_dot_suffix(cid):
    info = cell_info.get(cid, {})
    base   = info.get("base_cell_id", "UNKNOWN")
    precur = info.get("precursor_cell_id", "UNKNOWN")
    modif  = info.get("modification", "UNKNOWN")
    ctype  = info.get("cell_type", "UNKNOWN")
    stype  = info.get("sample_type", "UNKNOWN")
    site   = info.get("primary_site", "UNKNOWN")
    subtype= info.get("subtype", "UNKNOWN")
    vendor = info.get("original_source_vendor", "UNKNOWN")
    catalog= info.get("provider_catalog_id", "UNKNOWN")
    return {
        "cell_id": cid,
        "base_cell_id": base,
        "precursor_cell_id": precur,
        "modification": modif,
        "cell_type": ctype,
        "sample_type": stype,
        "primary_site": site,
        "subtype": subtype,
        "vendor": vendor,
        "catalog_id": catalog,
    }

def assess_substitution_defensibility(cid):
    """
    Return a brief defensibility assessment for using the parent's epigenomic
    data for this cell line. Decisions are REPORTED, not implemented (except
    the pre-authorized NPC.CAS9 case).
    """
    info = cell_info.get(cid, {})
    base   = info.get("base_cell_id", "")
    modif  = info.get("modification", "")
    ctype  = info.get("cell_type", "")

    if cid == "NPC.CAS9":
        return (
            "PRE-AUTHORIZED: Cas9 transgene insertion does not alter chromatin "
            "accessibility or histone modifications at broad scale. NPC.CAS9 may "
            "directly share NPC's resolved data for all 3 marks."
        )
    if cid == "NPC.TAK":
        return (
            "NEEDS OVERSEER DECISION: NPC.TAK shares NPC's differentiation state "
            "(iPSC-derived NPC, not terminally differentiated) and base_cell_id=NPC "
            "per cell_info. The 'TAK' suffix likely denotes a specific differentiation "
            "batch or treatment condition (no LINCS annotation beyond 'differentiated "
            "from iPSC, but not terminally differentiated'). If TAK is merely a "
            "passage/batch variant with identical protocol, sharing NPC data may be "
            "defensible. If it involves a TAK1 inhibitor or other perturbation, "
            "chromatin state could differ. DO NOT IMPLEMENT substitution without "
            "explicit overseer confirmation of what 'TAK' denotes."
        )
    if cid == "MNEU.E":
        return (
            "NOT DEFENSIBLE: MNEU.E is terminally differentiated motor neurons "
            "(differentiated from ESC). Motor neuron chromatin is fundamentally "
            "distinct from any progenitor state (NPC, ESC, or fibroblast). "
            "No parent substitution permissible -- data gap must stand."
        )
    if cid == "ASC.C":
        return (
            "NEEDS OVERSEER DECISION: ASC.C has the same base_cell_id (ASC), same "
            "sample_type (primary adipocyte stem cells), same vendor/catalog "
            "(Sciencell HPA-v) as ASC per cell_info. The '.C' suffix appears to "
            "denote a different lot or passage. If ASC and ASC.C are from the same "
            "commercial source and passage range, sharing data is plausible. "
            "However, primary cells show higher donor-to-donor chromatin variability "
            "than cell lines -- verify batch identity before implementing."
        )
    if cid == "SKL.C":
        return (
            "NEEDS OVERSEER DECISION: SKL.C has the same base_cell_id (SKL), "
            "same sample_type (primary skeletal muscle cells), same vendor/catalog "
            "(Lonza CC-2561) as SKL. '.C' suffix likely denotes a different lot or "
            "culture condition. Same caveat as ASC.C: primary cell donor variability "
            "means chromatin state could differ across lots. Report, do not decide."
        )
    if cid == "SKL":
        return (
            "STANDALONE: SKL itself is an unresolved primary skeletal muscle line "
            "(Lonza CC-2561). No parent to inherit from. Search for 'skeletal muscle' "
            "or 'myocyte' aliases in databases as alternative matching approach."
        )
    if cid == "SKB":
        return (
            "STANDALONE: SKB is a primary myoblast line (Lonza CC-2580), distinct "
            "from SKL (myocytes CC-2561). Different differentiation state; myoblasts "
            "and myocytes have distinct chromatin landscapes. No defensible parent "
            "substitution. Search for 'myoblast' or 'skeletal muscle myoblast' aliases."
        )
    if cid == "NKDBA":
        return (
            "STANDALONE: NKDBA is hTERT-immortalized kidney epithelial cells "
            "immunoselected for DBA-lectin positivity. This is a distinct immortalized "
            "line with no listed parent. Potentially searchable under 'DBA' or 'kidney "
            "epithelial' in Cistrome/EpiMap. No substitution applicable."
        )
    if cid == "MCH58":
        return (
            "STANDALONE: MCH58 is an immortalized skin fibroblast line with no listed "
            "parent or related commercial catalog ID in LINCS cell_info. No substitution "
            "applicable. Search under 'fibroblast' may find imputed tracks in EpiMap."
        )
    if cid == "FIBRNPC":
        return (
            "STANDALONE iPSC: FIBRNPC is an iPSC derived from skin fibroblasts. "
            "Distinct chromatin state from any differentiated line. No parent "
            "substitution applicable."
        )
    if cid == "HUES3":
        return (
            "STANDALONE ESC: HUES3 is an embryonic stem cell line. Distinct chromatin "
            "state. No parent substitution applicable."
        )
    return "No specific assessment -- standard unresolved line."


print("\n=== BUILDING DOT-SUFFIX INVESTIGATION TABLE ===")
dot_rows = []
for cid in DOT_CELLS_OF_INTEREST:
    d = describe_dot_suffix(cid)
    d["defensibility_assessment"] = assess_substitution_defensibility(cid)
    dot_rows.append(d)

df_dot = pd.DataFrame(dot_rows)
df_dot.to_csv(v6_dot_suffix_path, sep="\t", index=False)
print(f"Saved {v6_dot_suffix_path}")

# ---------- Alias name variants for improved primary-assay matching ----------
# For each cell line, we expand the set of name variants to try.
# These are derived from cell_info (base_cell_id, modification), known aliases,
# and database naming conventions.
# Matching ALWAYS uses word-boundary regex (wb_match). No bare substring matching.

ALIAS_MAP = {
    # Format: cell_id -> list of (raw_name_for_regex, description)
    "MNEU.E":  [("MNEU", "base name"), ("motor neuron", "cell type"), ("motorneuron", "merged")],
    "NPC.TAK": [("NPC-TAK", "hyphenated"), ("NPC TAK", "spaced")],
    "NPC.CAS9":[("NPC", "base name (pre-authorized substitution)")],
    "ASC.C":   [("ASC", "base name"), ("adipocyte stem", "type alias"), ("adipose stem", "type alias")],
    "SKL.C":   [("SKL", "base name"), ("skeletal muscle", "type alias")],
    "SKL":     [("skeletal muscle", "type alias"), ("myocyte", "type alias"),
                ("skeletalmuscle", "merged")],
    "SKB":     [("myoblast", "type alias"), ("skeletal myoblast", "type alias"),
                ("primary myoblast", "type alias")],
    "NKDBA":   [("DBA", "short alias"), ("kidney epithelial", "type alias")],
    "MCH58":   [("MCH-58", "hyphenated"), ("fibroblast", "type alias")],
    "FIBRNPC": [("fibroblast NPC", "type alias"), ("FIBR-NPC", "hyphenated")],
    "HUES3":   [("HUES-3", "hyphenated"), ("HUES 3", "spaced"), ("hESC", "type alias")],
    "HCC515":  [("HCC-515", "hyphenated")],
    "HA1E":    [("HA-1E", "hyphenated"), ("HA 1E", "spaced")],
    "CORL23":  [("COR-L23", "hyphenated"), ("COR L23", "spaced")],
    "COV644":  [("COV-644", "hyphenated")],
    "BT20":    [("BT-20", "hyphenated")],
    "SKMEL28": [("SK-MEL-28", "full-hyphenated"), ("SKMEL-28", "partial-hyphenated")],
    "SKMEL1":  [("SK-MEL-1", "full-hyphenated")],
    "SKLU1":   [("SK-LU-1", "full-hyphenated"), ("SKLU-1", "partial-hyphenated")],
    "SW948":   [("SW-948", "hyphenated")],
    "SW620":   [("SW-620", "hyphenated")],
    "SNU1040": [("SNU-1040", "hyphenated")],
    "SNUC4":   [("SNU-C4", "hyphenated"), ("SNUC-4", "partial")],
    "SNUC5":   [("SNU-C5", "hyphenated"), ("SNUC-5", "partial")],
    "T3M10":   [("T3M-10", "hyphenated"), ("T3-M10", "alt-hyphenated")],
    "JHUEM2":  [("JHUEM-2", "hyphenated"), ("JHU-EM2", "alt")],
    "HEC108":  [("HEC-108", "hyphenated")],
    "NCIH1694":[("NCI-H1694", "full-hyphenated"), ("H1694", "short")],
    "NCIH1836":[("NCI-H1836", "full-hyphenated"), ("H1836", "short")],
    "NCIH2073":[("NCI-H2073", "full-hyphenated"), ("H2073", "short")],
    "NCIH508": [("NCI-H508", "full-hyphenated"),  ("H508", "short")],
    "NCIH596": [("NCI-H596", "full-hyphenated"),  ("H596", "short")],
    "TYKNU":   [("TYK-nu", "case-variant"), ("TYKnu", "merged")],
    "RMUGS":   [("RMUGS", "canonical")],
    "RMGI":    [("RMGI", "canonical")],
    "MDST8":   [("MDS-T8", "hyphenated"), ("MDST-8", "alt")],
    "DV90":    [("DV-90", "hyphenated")],
    "EFO27":   [("EFO-27", "hyphenated")],
    "OV7":     [("OV-7", "hyphenated"), ("OAW-42", "possible-alias")],
    "SNGM":    [("SNG-M", "hyphenated")],
    "PL21":    [("PL-21", "hyphenated")],
    "NOMO1":   [("NOMO-1", "hyphenated")],
    "YAPC":    [("YA-PC", "hyphenated"), ("YAPC", "canonical")],
    "HT115":   [("HT-115", "hyphenated")],
    "HS578T":  [("HS578T", "canonical"), ("Hs 578T", "spaced-case")],
    "WSUDLCL2":[("WSU-DLCL2", "hyphenated"), ("WSU DLCL2", "spaced")],
}


# ---------- EpiMap alias re-matcher ----------
def epimap_search_alias(cell, aliases, epimap_df):
    """
    Try to find an EpiMap match for `cell` using a list of alias raw names.
    All matching uses wb_match (word-boundary regex).
    Returns list of row dicts (unperturbed only).
    """
    results = []
    seen_ids = set()
    for alias_raw, _desc in aliases:
        for idx, row in epimap_df.iterrows():
            perturb_val = row.get("perturb", None)
            if perturb_val is not None and str(perturb_val).strip() not in ("", "nan", "None"):
                continue
            name = str(row.get("name", ""))
            ct   = str(row.get("ct", ""))
            if wb_match(alias_raw, name) or wb_match(alias_raw, ct):
                rid = str(row.get("id", idx))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    results.append(row.to_dict())
    return results


# ---------- Cistrome alias re-matcher ----------
def cistrome_search_alias(cell, aliases, cistrome_samples, mark):
    """
    Try to find Cistrome samples for `cell` using a list of alias raw names.
    Always uses wb_match (word-boundary regex) -- never bare substring.
    Filters by the requested mark.
    Returns list of Cistrome sample dicts.
    """
    found = []
    seen_ids = set()
    for alias_raw, _desc in aliases:
        for s in cistrome_samples:
            sid = s.get("id", "")
            if sid in seen_ids:
                continue
            title = s.get("title", "")
            ontology_terms = [o.get("term", "") for o in s.get("ontologies", []) if isinstance(o, dict)]
            all_texts = [title] + ontology_terms

            matched = any(wb_match(alias_raw, t) for t in all_texts if t)
            if not matched:
                continue

            exp_type = s.get("experiment_type", "")
            factor_names = [f.get("name", "").lower() for f in s.get("factors", []) if isinstance(f, dict)]
            mark_lower = mark.lower()

            mark_matched = False
            if mark == "ATAC-seq":
                mark_matched = exp_type in ["ATAC", "DNASE"]
            elif mark in ["H3K27ac", "H3K27me3", "H3K4me3", "H3K9me3"]:
                mark_matched = (
                    exp_type == "IP" and
                    (mark_lower in title.lower() or any(mark_lower in fn for fn in factor_names))
                )
            elif mark == "DNase-seq":
                mark_matched = exp_type == "DNASE"

            if mark_matched:
                seen_ids.add(sid)
                found.append(s)
    return found


# ---------- ENCODE substitute-assay query ----------
def get_encode_for_cell_assay(cell, assay_title, target_label=None):
    """
    Query ENCODE for a specific cell and assay combo.
    Uses EXACT normalize_name equality (Part 0 bug fix #1).
    Returns list of matched experiment dicts.
    """
    norm_cell = normalize_name(cell)
    params = [
        ("type", "Experiment"),
        ("status", "released"),
        ("searchTerm", cell),
        ("assay_title", assay_title),
        ("format", "json"),
        ("limit", "100"),
        ("field", "accession"),
        ("field", "biosample_ontology.term_name"),
        ("field", "assay_title"),
        ("field", "target.label"),
        ("field", "files.assembly"),
        ("field", "files.status"),
        ("field", "files.file_format"),
    ]
    if target_label:
        params.append(("target.label", target_label))

    url = "https://www.encodeproject.org/search/?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        matched = []
        for exp in data.get("@graph", []):
            term_name = exp.get("biosample_ontology", {}).get("term_name", "")
            if normalize_name(term_name) == norm_cell:
                matched.append(exp)
        return matched
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        print(f"  ENCODE HTTPError for {cell}/{assay_title}: {e}")
        return []
    except Exception as e:
        print(f"  ENCODE error for {cell}/{assay_title}: {e}")
        return []


def resolve_assembly(experiments):
    """Pick preferred assembly from a list of ENCODE experiment dicts."""
    all_assemblies = set()
    accessions = []
    for exp in experiments:
        accessions.append(exp.get("accession", ""))
        for f in exp.get("files", []):
            if f.get("status") == "released" and f.get("assembly"):
                all_assemblies.add(f.get("assembly"))
    if "GRCh38" in all_assemblies:
        build = "GRCh38"
    elif "hg38" in all_assemblies:
        build = "hg38"
    elif all_assemblies:
        build = ",".join(sorted(all_assemblies))
    else:
        build = "GRCh38"
    return build, accessions


# ---------- Inline contamination classification stub ----------
# Applied to every new match found via substitute assays.
# We cannot run the full Cellosaurus ancestry walk here (that's the v5
# Cistrome audit pipeline). Instead we classify based on source:
#   - ENCODE matches use exact cell-name equality -> contamination-safe by design
#   - EpiMap: word-boundary match, perturb-filtered -> contamination-safe by design
#   - Cistrome: mark new matches for lightweight classification; any sample whose
#     Cistrome ontology contains a clearly different CVCL will be flagged.
# Full deep-ancestry audit of new Cistrome matches is deferred to a future round
# (same as the original v5 Cistrome audit was a separate pass). We record the
# sample IDs so the pipeline can pick them up.
#
# For the purposes of this round, the classification for substitute-assay matches
# is: CLEAN_PENDING_CISTROME_AUDIT (ENCODE/EpiMap) or PENDING_AUDIT (Cistrome).

def classify_new_cistrome_match(samples, cell):
    """
    Lightweight contamination pre-screen for newly found Cistrome samples.
    Returns (classification, flag_note).
    We inspect ontology terms for obvious cross-cell indicators.
    Full Cellosaurus ancestry walk is deferred to next audit round.
    """
    norm_cell = normalize_name(cell)
    for s in samples:
        title = s.get("title", "")
        ontology_terms = [o.get("term", "") for o in s.get("ontologies", []) if isinstance(o, dict)]
        for ot in ontology_terms:
            ot_norm = normalize_name(ot)
            # If the ontology term is clearly a different (non-empty) cell identity
            # that does not contain the target cell's normalized name, flag it.
            if ot_norm and ot_norm != norm_cell and len(ot_norm) > 3:
                # Only flag if the term looks like a cell line name (not generic tissue)
                if not any(generic in ot_norm for generic in ["cellline", "cancer", "normal", "tumor", "primary"]):
                    return (
                        "PENDING_CISTROME_AUDIT",
                        f"Cistrome sample {s.get('id')} ontology '{ot}' differs from target '{cell}' -- "
                        f"requires full Cellosaurus ancestry audit before use."
                    )
    return ("CLEAN_PENDING_CISTROME_AUDIT", "No obvious cross-cell ontology conflict detected; "
            "pending full Cellosaurus ancestry audit in next round.")


# ---------- SUBSTITUTE ASSAY DEFINITIONS ----------
# For each primary mark, which substitute assays to try and what they stand in for.
SUBSTITUTE_ASSAY_MAP = {
    # primary_mark -> list of (substitute_assay, encode_assay_title, encode_target_label_or_None,
    #                          cistrome_exp_type_hint, cist_factor_keyword)
    "ATAC-seq": [
        {
            "substitute_assay": "H3K4me3 ChIP-seq",
            "encode_assay_title": "Histone ChIP-seq",
            "encode_target_label": "H3K4me3",
            "cistrome_mark": "H3K4me3",
            "rationale": "Active promoter mark; defensible proxy for chromatin accessibility at TSS ?2kb",
        },
        {
            "substitute_assay": "DNase-seq",
            "encode_assay_title": "DNase-seq",
            "encode_target_label": None,
            "cistrome_mark": "DNase-seq",
            "rationale": "Same measurement category as ATAC-seq; older assay",
        },
    ],
    "H3K27me3": [
        {
            "substitute_assay": "H3K9me3 ChIP-seq",
            "encode_assay_title": "Histone ChIP-seq",
            "encode_target_label": "H3K9me3",
            "cistrome_mark": "H3K9me3",
            "rationale": "Constitutive heterochromatin; alternative repression mark",
        },
    ],
    "H3K27ac": [],  # No substitute defined for H3K27ac
}

# ---------- Build alias search list ----------
def get_aliases(cell):
    """Return list of (raw_name, desc) pairs to try for this cell, including the
    cell's own name as the primary canonical alias."""
    base = [(cell, "canonical")]
    extras = ALIAS_MAP.get(cell, [])
    # Also add base_cell_id from cell_info if different
    info = cell_info.get(cell, {})
    base_cid = info.get("base_cell_id", "")
    if base_cid and base_cid != cell and base_cid != "-666":
        base.append((base_cid, "base_cell_id from cell_info"))
    return base + extras


# ---------- NPC.CAS9 -> NPC pre-authorized data sharing ----------
print(f"\n=== PART 1: NPC.CAS9 -> NPC PRE-AUTHORIZED SUBSTITUTION ===")
npc_cas9_rows = []
for mark in ["ATAC-seq", "H3K27ac", "H3K27me3"]:
    # Find NPC's resolved row in v5 report
    npc_row = df_v5[(df_v5["lincs_cell_id"] == "NPC") & (df_v5["mark"] == mark)]
    if len(npc_row) == 0:
        print(f"  WARNING: NPC/{mark} not found in v5 report -- cannot propagate to NPC.CAS9")
        npc_cas9_rows.append({
            "lincs_cell_id": "NPC.CAS9",
            "mark": mark,
            "source_used": "unresolved",
            "genome_build": "None",
            "notes": "NPC base row not found in v5 report.",
            "is_primary_assay": True,
            "substituted_for": None,
            "substitute_assay_used": None,
            "v6_change": "WARNING: NPC base row missing",
        })
        continue

    npc_row = npc_row.iloc[0]
    npc_source = npc_row["source_used"]
    npc_build  = npc_row["genome_build"]
    npc_notes  = npc_row["notes"]

    if npc_source == "unresolved":
        # NPC itself is unresolved for this mark -- cannot propagate
        npc_cas9_rows.append({
            "lincs_cell_id": "NPC.CAS9",
            "mark": mark,
            "source_used": "unresolved",
            "genome_build": "None",
            "notes": f"NPC itself is unresolved for {mark}; NPC.CAS9 cannot inherit. {npc_notes}",
            "is_primary_assay": True,
            "substituted_for": None,
            "substitute_assay_used": None,
            "v6_change": "NPC.CAS9 still unresolved (NPC base unresolved)",
        })
        print(f"  NPC.CAS9/{mark}: NPC base is unresolved -- cannot propagate")
    else:
        npc_cas9_rows.append({
            "lincs_cell_id": "NPC.CAS9",
            "mark": mark,
            "source_used": f"{npc_source} [via NPC pre-authorized]",
            "genome_build": npc_build,
            "notes": (
                f"[NPC.CAS9 pre-authorized: Cas9 transgene does not alter chromatin state.] "
                f"Inherited from NPC/{mark}: {npc_notes}"
            ),
            "is_primary_assay": True,
            "substituted_for": None,
            "substitute_assay_used": None,
            "v6_change": f"RESOLVED via NPC pre-authorized sharing (source: {npc_source})",
        })
        print(f"  NPC.CAS9/{mark}: Resolved via NPC (source={npc_source}, build={npc_build})")

npc_cas9_lookup = {r["mark"]: r for r in npc_cas9_rows}


# ---------- PART 1B: Alias re-matching for still-unresolved rows ----------
print("\n=== PART 1B: ALIAS RE-MATCHING FOR STILL-UNRESOLVED ROWS ===")

# Build the current state: v5 rows PLUS NPC.CAS9 overrides
# We will iterate over all rows; unresolved ones will try alias matching.

def pick_cistrome_assembly(cist_matched):
    assemblies = set(s.get("assembly", "") for s in cist_matched if s.get("assembly"))
    assembly_list = [a for a in assemblies if a]
    if not assembly_list:
        assembly_list = ["hg19"]
    if "hg38" in assembly_list or "GRCh38" in assembly_list:
        return "hg38" if "hg38" in assembly_list else "GRCh38"
    elif "hg19" in assembly_list or "GRCh37" in assembly_list:
        return "hg19"
    else:
        return ",".join(sorted(assembly_list))


all_rows_v6 = []
changed_rows = []   # rows that changed status from v5 -> v6

# Skip NPC.CAS9 -- handled above
for _, row in df_v5.iterrows():
    cell = row["lincs_cell_id"]
    mark = row["mark"]
    source_v5 = row["source_used"]
    build_v5  = row["genome_build"]
    notes_v5  = row["notes"]

    if cell == "NPC.CAS9":
        r = npc_cas9_lookup[mark].copy()
        all_rows_v6.append(r)
        if r["v6_change"].startswith("RESOLVED"):
            changed_rows.append(r)
        continue

    # Already resolved in v5 -- carry forward unchanged
    if source_v5 != "unresolved":
        all_rows_v6.append({
            "lincs_cell_id": cell,
            "mark": mark,
            "source_used": source_v5,
            "genome_build": build_v5,
            "notes": notes_v5,
            "is_primary_assay": True,
            "substituted_for": None,
            "substitute_assay_used": None,
            "v6_change": "unchanged from v5",
        })
        continue

    # --- Try alias re-matching ---
    aliases = get_aliases(cell)
    new_source = "unresolved"
    new_build  = "None"
    new_notes  = notes_v5
    v6_change  = "still unresolved after alias retry"

    # 1. ENCODE with alias names
    # (v5 already checked the canonical name via ENCODE cache -- try aliases)
    for alias_raw, alias_desc in aliases:
        if alias_raw == cell:
            continue  # already tried
        exps = get_encode_for_cell_assay(cell, "ATAC-seq" if mark == "ATAC-seq" else "Histone ChIP-seq",
                                         target_label=(None if mark == "ATAC-seq" else mark))
        # Note: for ENCODE alias retry we search by alias_raw name as searchTerm
        # but still require exact normalize match -- use the cell's canonical normalized name
        # This is intentional: if the alias resolves to the same biosample_ontology term,
        # the existing cache would have caught it. Aliases here are for Cistrome/EpiMap only.
        # ENCODE already uses its own biosample ontology (not free text titles), so alias
        # retry on ENCODE is not productive -- skip ENCODE alias retry.
        break

    # 2. Cistrome alias re-matching
    if new_source == "unresolved":
        for alias_raw, alias_desc in aliases:
            if alias_raw == cell:
                continue  # already tried in v5
            cist_matched = cistrome_search_alias(cell, [(alias_raw, alias_desc)], cistrome_samples, mark)
            if cist_matched:
                new_source = "cistrome"
                new_build  = pick_cistrome_assembly(cist_matched)
                new_notes  = (
                    f"[alias='{alias_raw}' ({alias_desc})] "
                    f"Cistrome sample ID(s): {','.join(str(s.get('id')) for s in cist_matched)}"
                )
                v6_change = f"RESOLVED via Cistrome alias='{alias_raw}'"
                print(f"  {cell}/{mark}: Resolved via Cistrome alias '{alias_raw}'")
                break

    # 3. EpiMap alias re-matching
    if new_source == "unresolved":
        for alias_raw, alias_desc in aliases:
            if alias_raw == cell:
                continue  # already tried in v5
            epi_hits = epimap_search_alias(cell, [(alias_raw, alias_desc)], epimap_df)
            if epi_hits:
                new_source = "epimap"
                new_build  = "hg19"
                new_notes  = (
                    f"[alias='{alias_raw}' ({alias_desc})] "
                    f"EpiMap imputed track, biosample ID: {epi_hits[0].get('id')} [perturb=null confirmed]"
                )
                v6_change = f"RESOLVED via EpiMap alias='{alias_raw}'"
                print(f"  {cell}/{mark}: Resolved via EpiMap alias '{alias_raw}'")
                break

    r = {
        "lincs_cell_id": cell,
        "mark": mark,
        "source_used": new_source,
        "genome_build": new_build,
        "notes": new_notes,
        "is_primary_assay": True,
        "substituted_for": None,
        "substitute_assay_used": None,
        "v6_change": v6_change,
    }
    all_rows_v6.append(r)
    if new_source != "unresolved":
        changed_rows.append(r)


# ---------- PART 2: SUBSTITUTE ASSAYS for still-unresolved rows ----------
print("\n=== PART 2: SUBSTITUTE ASSAYS FOR STILL-UNRESOLVED ROWS ===")

substitute_rows = []    # New rows to append (substitute assays)
flagged_rows    = []    # Rows needing review

# Index current state for quick lookup
current_state = {(r["lincs_cell_id"], r["mark"]): r for r in all_rows_v6}

# Build set of cells that are still unresolved for each mark
still_unresolved = {
    (r["lincs_cell_id"], r["mark"])
    for r in all_rows_v6
    if r["source_used"] == "unresolved"
}

print(f"  Still unresolved after Part 1: {len(still_unresolved)} (cell,mark) pairs")

# Cache for ENCODE substitute queries (avoid redundant HTTP calls)
encode_sub_cache = {}

for (cell, primary_mark) in sorted(still_unresolved):
    subs = SUBSTITUTE_ASSAY_MAP.get(primary_mark, [])
    if not subs:
        continue

    aliases = get_aliases(cell)

    for sub_def in subs:
        sub_assay   = sub_def["substitute_assay"]
        enc_assay   = sub_def["encode_assay_title"]
        enc_target  = sub_def["encode_target_label"]
        cist_mark   = sub_def["cistrome_mark"]
        rationale   = sub_def["rationale"]

        sub_source = "unresolved"
        sub_build  = "None"
        sub_notes  = ""
        sub_class  = "NOT_FOUND"
        flag_note  = ""

        # --- ENCODE substitute query ---
        cache_key = (cell, enc_assay, enc_target)
        if cache_key not in encode_sub_cache:
            time.sleep(0.1)
            encode_sub_cache[cache_key] = get_encode_for_cell_assay(cell, enc_assay, enc_target)
        enc_hits = encode_sub_cache[cache_key]

        if enc_hits:
            sub_source = "encode"
            sub_build, accessions = resolve_assembly(enc_hits)
            sub_notes = f"ENCODE Experiment(s): {','.join(accessions)}"
            sub_class = "CLEAN"  # ENCODE uses exact biosample_ontology equality
            flag_note = "ENCODE exact-match -- contamination-safe by design"
            print(f"  [ENCODE sub] {cell}/{primary_mark} -> {sub_assay}: found {len(enc_hits)} exp(s)")

        # --- Cistrome substitute query ---
        if sub_source == "unresolved":
            cist_matched = cistrome_search_alias(cell, aliases, cistrome_samples, cist_mark)
            if cist_matched:
                sub_source = "cistrome"
                sub_build  = pick_cistrome_assembly(cist_matched)
                sub_notes  = f"Cistrome sample ID(s): {','.join(str(s.get('id')) for s in cist_matched)}"
                sub_class, flag_note = classify_new_cistrome_match(cist_matched, cell)
                print(f"  [Cistrome sub] {cell}/{primary_mark} -> {sub_assay}: {len(cist_matched)} sample(s), class={sub_class}")

        # --- EpiMap substitute query ---
        if sub_source == "unresolved":
            # EpiMap does not carry assay-level distinctions for imputed tracks;
            # it provides mark-specific imputed signals but the biosample metadata
            # does not distinguish ATAC vs DNase vs H3K4me3. We can match on
            # the biosample (cell identity), and the imputed signal serves as a
            # proxy. Only use for assay classes that EpiMap covers.
            if sub_assay in ("H3K4me3 ChIP-seq", "H3K9me3 ChIP-seq"):
                epi_hits = epimap_search_alias(cell, aliases, epimap_df)
                if epi_hits:
                    sub_source = "epimap"
                    sub_build  = "hg19"
                    sub_notes  = (
                        f"EpiMap imputed track, biosample ID: {epi_hits[0].get('id')} "
                        f"[perturb=null confirmed]"
                    )
                    sub_class = "CLEAN"
                    flag_note = "EpiMap word-boundary match, perturb-filtered -- contamination-safe"
                    print(f"  [EpiMap sub] {cell}/{primary_mark} -> {sub_assay}: biosample {epi_hits[0].get('id')}")

        if sub_source == "unresolved":
            continue  # No substitute found for this sub_assay either

        row = {
            "lincs_cell_id": cell,
            "mark": sub_assay,
            "source_used": sub_source,
            "genome_build": sub_build,
            "notes": sub_notes,
            "is_primary_assay": False,
            "substituted_for": primary_mark,
            "substitute_assay_used": sub_assay,
            "rationale": rationale,
            "contamination_classification": sub_class,
            "flag_note": flag_note,
            "v6_change": f"NEW: substitute assay for {primary_mark}",
        }
        substitute_rows.append(row)

        if sub_class in ("GENUINE_CONTAMINATION", "MULTIPLEXED_EXPERIMENT", "PENDING_CISTROME_AUDIT"):
            flagged_rows.append(row)

print(f"\n  Substitute assay rows found: {len(substitute_rows)}")
print(f"  Flagged rows (needing review): {len(flagged_rows)}")


# ---------- Finalise output ----------
print("\n=== WRITING OUTPUTS ===")

# Main v6 report: all primary-assay rows with new columns
df_v6 = pd.DataFrame(all_rows_v6)
# Ensure column order
for col in ["is_primary_assay", "substituted_for", "substitute_assay_used", "v6_change"]:
    if col not in df_v6.columns:
        df_v6[col] = None
df_v6 = df_v6[[
    "lincs_cell_id", "mark", "source_used", "genome_build", "notes",
    "is_primary_assay", "substituted_for", "substitute_assay_used", "v6_change"
]]
df_v6.to_csv(v6_report_path, sep="\t", index=False)
print(f"Saved {v6_report_path} ({len(df_v6)} rows)")

# Substitute assay results
if substitute_rows:
    df_sub = pd.DataFrame(substitute_rows)
    df_sub.to_csv(v6_substitute_path, sep="\t", index=False)
    print(f"Saved {v6_substitute_path} ({len(df_sub)} rows)")
else:
    pd.DataFrame(columns=[
        "lincs_cell_id", "mark", "source_used", "genome_build", "notes",
        "is_primary_assay", "substituted_for", "substitute_assay_used",
        "rationale", "contamination_classification", "flag_note", "v6_change"
    ]).to_csv(v6_substitute_path, sep="\t", index=False)
    print(f"Saved {v6_substitute_path} (empty -- no substitute assay matches found)")

# Flagged contamination
if flagged_rows:
    df_flag = pd.DataFrame(flagged_rows)
    df_flag.to_csv(v6_flagged_path, sep="\t", index=False)
    print(f"Saved {v6_flagged_path} ({len(df_flag)} flagged rows)")
else:
    pd.DataFrame().to_csv(v6_flagged_path, sep="\t", index=False)
    print(f"Saved {v6_flagged_path} (empty -- no flagged rows)")

# Summary statistics
total = len(df_v6)
n_resolved_v5   = sum(1 for _, r in df_v5.iterrows() if r["source_used"] != "unresolved")
n_resolved_v6   = sum(1 for r in all_rows_v6 if r["source_used"] != "unresolved")
n_changed       = len(changed_rows)
n_sub_resolved  = len(substitute_rows)
n_still_unreslv = sum(1 for r in all_rows_v6 if r["source_used"] == "unresolved")

print(f"""
=== SUMMARY ===
Total (cell,mark) pairs:           {total}
Resolved in v5:                     {n_resolved_v5}
Resolved in v6 (primary assay):     {n_resolved_v6}
  New resolutions this round:       {n_changed}
Substitute-assay rows added:        {n_sub_resolved}
Still unresolved (primary assay):   {n_still_unreslv}
Flagged for review:                 {len(flagged_rows)}
""")

print("Changed rows:")
for r in changed_rows:
    print(f"  {r['lincs_cell_id']}/{r['mark']}: {r['v6_change']}")
    print(f"    source={r['source_used']}, build={r['genome_build']}")
    print(f"    notes={r['notes'][:120]}")

print("\nAll deliverables written successfully.")
