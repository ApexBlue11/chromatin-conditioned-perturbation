# -*- coding: utf-8 -*-
"""
retroactive_classify_v6_part1b.py
==================================
Retroactively applies the FIXED contamination classifier to the 22 rows
that Part 1b resolved via alias matching, but never screened.

Two bugs fixed in the classifier here vs the original:

BUG 1 (missed call):
  classify_new_cistrome_match() was never called in the Part 1b block.
  This script applies it retroactively to every alias-resolved row.

BUG 2 (backwards exemption for type-alias searches):
  The original classifier skipped any ontology term containing
  "cellline", "cancer", "normal", "tumor", "primary" -- reasoning
  it was "generic tissue language, not a cell line name".
  But for type-alias matches (alias_desc in {"type alias", "cell type"}),
  the search term ITSELF was generic, so every hit was already found via
  a generic description. The "sounds generic, skip" exemption is therefore
  meaningless and backwards for this case -- it exempts exactly the
  category of hits that most need scrutiny.

  Fixed rules:
    - name-based alias (alias_desc NOT in TYPE_ALIAS_DESCS):
        Keep the original exemption. If the ontology term sounds like a
        generic tissue descriptor, skip it.
    - type-alias (alias_desc IN TYPE_ALIAS_DESCS):
        Flag PENDING_CISTROME_AUDIT for ANY ontology term that is non-empty,
        longer than 3 chars, and NOT a normalized match to the target cell's
        canonical name (or any of its known name-based aliases). No generic-
        term exemption whatsoever.

  Same stricter logic applies to EpiMap rows matched via type alias: if
  the matched row's own 'ct' or 'name' field does not contain the target
  cell's canonical name or any name-based alias, flag PENDING_EPIMAP_AUDIT.

Outputs:
  retroactive_classification_v6.tsv   -- per-row findings for all 22 rows
  flagged_raw_records_v6.json         -- full raw Cistrome/EpiMap record for
                                         every newly-flagged row
  coverage_report_v6_reclassified.tsv -- regenerated coverage report with
                                         new columns added to the 22 rows
"""

import json
import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# -- Paths -------------------------------------------------------------------
brain_dir = r"C:\Projects\LINCS\Epigenetics"

cistrome_json_path  = os.path.join(brain_dir, "cistrome_human_samples.json")
epimap_tsv_path     = os.path.join(brain_dir, "epimap_metadata.tsv")
v6_report_path      = os.path.join(brain_dir, "coverage_report_v6.tsv")

retro_class_path    = os.path.join(brain_dir, "retroactive_classification_v6.tsv")
flagged_raw_path    = os.path.join(brain_dir, "flagged_raw_records_v6.json")
v6_reclassed_path   = os.path.join(brain_dir, "coverage_report_v6_reclassified.tsv")

# -- Load data ---------------------------------------------------------------
print("Loading Cistrome JSON...")
with open(cistrome_json_path, "r") as f:
    cistrome_samples = json.load(f)

# Build ID -> sample dict for fast lookup
cistrome_by_id = {str(s.get("id", "")): s for s in cistrome_samples}

print("Loading EpiMap TSV...")
epimap_df = pd.read_csv(epimap_tsv_path, sep="\t", dtype=str)
epimap_by_id = {str(row.get("id", "")): row.to_dict()
                for _, row in epimap_df.iterrows()}

print("Loading v6 coverage report...")
df_v6 = pd.read_csv(v6_report_path, sep="\t", dtype=str)

# -- Helper ------------------------------------------------------------------
def normalize_name(n):
    if not isinstance(n, str):
        return ""
    return "".join(c for c in n.lower() if c.isalnum())


# alias_descs that signal a GENERIC TISSUE/TYPE search
TYPE_ALIAS_DESCS = {"type alias", "cell type"}

# For each alias-resolved row we need to know:
#   cell, mark, alias_raw, alias_desc, source, sample IDs (for Cistrome)
#   or biosample ID (for EpiMap)

# The 22 rows are those where v6_change starts with "RESOLVED via Cistrome alias="
# or "RESOLVED via EpiMap alias="

def parse_alias_info(v6_change, notes):
    """
    Extract alias_raw and alias_desc from the v6_change / notes strings.
    v6_change example: "RESOLVED via Cistrome alias='fibroblast'"
    notes example:     "[alias='fibroblast' (type alias)] Cistrome sample ID(s): ..."
    Returns (alias_raw, alias_desc, source, sample_id_list)
    """
    alias_raw = ""
    alias_desc = ""
    sample_ids = []
    epimap_id = ""

    # Extract alias name and desc from notes (most reliable)
    m = re.search(r"\[alias='([^']+)'\s+\(([^)]+)\)\]", str(notes))
    if m:
        alias_raw  = m.group(1)
        alias_desc = m.group(2).strip()

    # Determine source
    if "Cistrome sample ID(s):" in str(notes):
        src = "cistrome"
        m2 = re.search(r"Cistrome sample ID\(s\):\s*([0-9,]+)", str(notes))
        if m2:
            sample_ids = [x.strip() for x in m2.group(1).split(",") if x.strip()]
    elif "EpiMap imputed track" in str(notes):
        src = "epimap"
        m3 = re.search(r"biosample ID:\s*(\S+)\s", str(notes))
        if m3:
            epimap_id = m3.group(1).strip()
    else:
        src = "unknown"

    return alias_raw, alias_desc, src, sample_ids, epimap_id


# -- Fixed classifier --------------------------------------------------------

def name_based_aliases_for_cell(cell):
    """
    Return normalized forms of all NAME-BASED aliases for a cell.
    These are alias entries whose desc is NOT in TYPE_ALIAS_DESCS.
    Used to decide whether an ontology term is actually 'the same cell
    under a different name' and therefore not a contamination signal.
    """
    # ALIAS_MAP from v6 script (name-based entries only, inlined here)
    # Only entries relevant to the 22 alias-resolved rows are needed.
    NAME_ALIAS_MAP = {
        "NPC.CAS9":  ["npc"],
        "NPC.TAK":   ["npctak", "npc"],
        "ASC.C":     ["asc"],
        "HUES3":     ["hues3", "hues3"],
        "MCH58":     ["mch58"],
        "MNEU.E":    ["mneu", "mneue"],
        "NKDBA":     ["nkdba"],
        "SKB":       ["skb"],
        "SKL":       ["skl"],
        "SKL.C":     ["skl", "sklc"],
    }
    base = [normalize_name(cell)]
    for a in NAME_ALIAS_MAP.get(cell, []):
        base.append(normalize_name(a))
    return set(base)


def classify_cistrome_fixed(samples, cell, alias_desc):
    """
    Fixed classifier for Cistrome samples matched via alias.

    alias_desc: the description string from ALIAS_MAP for the alias used.
    Returns (classification, flag_note, flagged_sample_ids)
    """
    norm_cell     = normalize_name(cell)
    name_norms    = name_based_aliases_for_cell(cell)
    is_type_alias = alias_desc in TYPE_ALIAS_DESCS

    flagged_ids = []
    flag_notes  = []

    for s in samples:
        sid = str(s.get("id", ""))
        ontology_terms = [o.get("term", "")
                          for o in s.get("ontologies", [])
                          if isinstance(o, dict)]

        for ot in ontology_terms:
            ot_norm = normalize_name(ot)
            if not ot_norm or len(ot_norm) <= 3:
                continue

            # Is this term actually the target cell (under any name-based alias)?
            if ot_norm in name_norms:
                continue

            if is_type_alias:
                # TYPE-ALIAS path: NO generic exemption. Any non-matching
                # ontology term triggers PENDING_CISTROME_AUDIT.
                flagged_ids.append(sid)
                flag_notes.append(
                    f"[TYPE-ALIAS MATCH] Cistrome sample {sid} ontology '{ot}' "
                    f"does not match target '{cell}' (alias_desc='{alias_desc}'); "
                    f"no generic-term exemption applies to type-alias searches."
                )
                break  # one flag per sample is enough
            else:
                # NAME-ALIAS path: apply original generic exemption.
                generics = ["cellline", "cancer", "normal", "tumor", "primary"]
                if any(g in ot_norm for g in generics):
                    continue  # exempt -- generic tissue annotation
                flagged_ids.append(sid)
                flag_notes.append(
                    f"[NAME-ALIAS MATCH] Cistrome sample {sid} ontology '{ot}' "
                    f"differs from target '{cell}' (alias_desc='{alias_desc}') "
                    f"and is not a generic tissue term."
                )
                break

    if flagged_ids:
        return (
            "PENDING_CISTROME_AUDIT",
            "; ".join(flag_notes),
            flagged_ids,
        )
    return (
        "CLEAN_PENDING_CISTROME_AUDIT",
        "No cross-cell ontology conflict under fixed classifier; "
        "pending full Cellosaurus ancestry audit.",
        [],
    )


def classify_epimap_fixed(epimap_row, cell, alias_desc):
    """
    Fixed classifier for an EpiMap row matched via alias.

    For type-alias matches: if the row's 'ct' and 'name' fields do not
    contain the canonical cell name or a name-based alias, flag
    PENDING_EPIMAP_AUDIT.

    For name-based matches: accept as CLEAN_PENDING (EpiMap uses
    free-text names, so a name-based match implies identity).
    """
    norm_cell     = normalize_name(cell)
    name_norms    = name_based_aliases_for_cell(cell)
    is_type_alias = alias_desc in TYPE_ALIAS_DESCS

    ct   = str(epimap_row.get("ct",   ""))
    name = str(epimap_row.get("name", ""))
    ct_norm   = normalize_name(ct)
    name_norm = normalize_name(name)

    if not is_type_alias:
        return (
            "CLEAN_PENDING_EPIMAP_AUDIT",
            "Name-based alias EpiMap match -- identity inferred from alias.",
            [],
        )

    # For type-alias: check whether the row itself actually names the target cell
    matched_by_name = (
        any(nn in ct_norm   for nn in name_norms if nn) or
        any(nn in name_norm for nn in name_norms if nn)
    )

    if matched_by_name:
        return (
            "CLEAN_PENDING_EPIMAP_AUDIT",
            f"EpiMap row ct='{ct}' / name='{name}' contains a name-based alias "
            f"for target '{cell}'; identity confirmed.",
            [],
        )
    else:
        return (
            "PENDING_EPIMAP_AUDIT",
            f"[TYPE-ALIAS MATCH] EpiMap row ct='{ct}' / name='{name}' matched via "
            f"type alias '{alias_desc}' but does NOT contain the target cell name "
            f"'{cell}' or any name-based alias. Full manual review required before use.",
            [epimap_row.get("id", "?")],
        )


# -- Identify the 22 alias-resolved rows ------------------------------------
alias_resolved_mask = df_v6["v6_change"].str.startswith("RESOLVED via Cistrome alias=") | \
                      df_v6["v6_change"].str.startswith("RESOLVED via EpiMap alias=")

alias_rows = df_v6[alias_resolved_mask].copy()
print(f"\nFound {len(alias_rows)} alias-resolved rows to retroactively classify.")

# -- Run the fixed classifier on each ----------------------------------------
retro_results = []
flagged_raw   = []

for idx, row in alias_rows.iterrows():
    cell  = str(row["lincs_cell_id"])
    mark  = str(row["mark"])
    notes = str(row["notes"])
    v6_ch = str(row["v6_change"])

    alias_raw, alias_desc, src, sample_ids, epimap_id = parse_alias_info(v6_ch, notes)

    if src == "cistrome":
        # Load the actual sample records
        samples_found = []
        missing_ids   = []
        for sid in sample_ids:
            s = cistrome_by_id.get(sid)
            if s:
                samples_found.append(s)
            else:
                missing_ids.append(sid)

        classification, flag_note, flagged_ids = classify_cistrome_fixed(
            samples_found, cell, alias_desc
        )

        result = {
            "cell":            cell,
            "mark":            mark,
            "alias_raw":       alias_raw,
            "alias_desc":      alias_desc,
            "alias_category":  "type-alias" if alias_desc in TYPE_ALIAS_DESCS else "name-based",
            "source":          "cistrome",
            "n_samples":       len(sample_ids),
            "n_loaded":        len(samples_found),
            "missing_ids":     ",".join(missing_ids),
            "classification":  classification,
            "flag_note":       flag_note,
            "flagged_sample_ids": ",".join(flagged_ids),
        }
        retro_results.append(result)

        if flagged_ids:
            for sid in flagged_ids:
                s = cistrome_by_id.get(sid, {})
                flagged_raw.append({
                    "cell": cell, "mark": mark,
                    "alias_raw": alias_raw, "alias_desc": alias_desc,
                    "classification": classification,
                    "flag_note": flag_note,
                    "raw_record": s,
                })
            print(f"  FLAGGED: {cell}/{mark} (alias='{alias_raw}', {alias_desc}) "
                  f"-> {classification}")
            print(f"    Sample(s): {','.join(flagged_ids)}")
            print(f"    Note: {flag_note[:160]}")
        else:
            print(f"  CLEAN:   {cell}/{mark} (alias='{alias_raw}', {alias_desc}) "
                  f"-> {classification}")

    elif src == "epimap":
        epi_row = epimap_by_id.get(epimap_id, {})
        classification, flag_note, flagged_ids = classify_epimap_fixed(
            epi_row, cell, alias_desc
        )

        result = {
            "cell":            cell,
            "mark":            mark,
            "alias_raw":       alias_raw,
            "alias_desc":      alias_desc,
            "alias_category":  "type-alias" if alias_desc in TYPE_ALIAS_DESCS else "name-based",
            "source":          "epimap",
            "n_samples":       1,
            "n_loaded":        1 if epi_row else 0,
            "missing_ids":     "" if epi_row else epimap_id,
            "classification":  classification,
            "flag_note":       flag_note,
            "flagged_sample_ids": ",".join(str(x) for x in flagged_ids),
        }
        retro_results.append(result)

        if flagged_ids:
            flagged_raw.append({
                "cell": cell, "mark": mark,
                "alias_raw": alias_raw, "alias_desc": alias_desc,
                "classification": classification,
                "flag_note": flag_note,
                "raw_record": epi_row,
            })
            print(f"  FLAGGED: {cell}/{mark} (EpiMap, alias='{alias_raw}', {alias_desc}) "
                  f"-> {classification}")
            print(f"    ID: {epimap_id}, ct='{epi_row.get('ct','')}', "
                  f"name='{epi_row.get('name','')}'")
        else:
            print(f"  CLEAN:   {cell}/{mark} (EpiMap, alias='{alias_raw}', {alias_desc}) "
                  f"-> {classification}")
    else:
        result = {
            "cell": cell, "mark": mark,
            "alias_raw": alias_raw, "alias_desc": alias_desc,
            "alias_category": "unknown",
            "source": src,
            "n_samples": 0, "n_loaded": 0, "missing_ids": "",
            "classification": "ERROR_UNPARSEABLE",
            "flag_note": f"Could not parse source/IDs from notes: {notes[:200]}",
            "flagged_sample_ids": "",
        }
        retro_results.append(result)
        print(f"  ERROR: {cell}/{mark} -- unparseable row")

# -- Write retroactive classification table ----------------------------------
df_retro = pd.DataFrame(retro_results)
df_retro.to_csv(retro_class_path, sep="\t", index=False)
print(f"\nSaved {retro_class_path} ({len(df_retro)} rows)")

# -- Write flagged raw records -----------------------------------------------
with open(flagged_raw_path, "w", encoding="utf-8") as f:
    json.dump(flagged_raw, f, indent=2, ensure_ascii=False)
print(f"Saved {flagged_raw_path} ({len(flagged_raw)} flagged records)")

# -- Regenerate coverage_report_v6 with classification columns ---------------
# Add two new columns: alias_match_category, alias_contamination_classification
# for the 22 alias-resolved rows; leave blank for all others.

retro_lookup = {
    (r["cell"], r["mark"]): r for r in retro_results
}

alias_categories   = []
alias_class_col    = []
alias_flag_col     = []

for _, row in df_v6.iterrows():
    key = (str(row["lincs_cell_id"]), str(row["mark"]))
    if key in retro_lookup:
        rr = retro_lookup[key]
        alias_categories.append(rr["alias_category"])
        alias_class_col.append(rr["classification"])
        alias_flag_col.append(rr["flag_note"])
    else:
        alias_categories.append("")
        alias_class_col.append("")
        alias_flag_col.append("")

df_v6_out = df_v6.copy()
df_v6_out["alias_match_category"]       = alias_categories
df_v6_out["alias_contamination_class"]  = alias_class_col
df_v6_out["alias_contamination_note"]   = alias_flag_col

df_v6_out.to_csv(v6_reclassed_path, sep="\t", index=False)
print(f"Saved {v6_reclassed_path} ({len(df_v6_out)} rows)")

# -- Final summary -----------------------------------------------------------
n_flagged = sum(1 for r in retro_results if r["classification"].startswith("PENDING"))
n_clean   = sum(1 for r in retro_results if r["classification"].startswith("CLEAN"))
n_type    = sum(1 for r in retro_results if r["alias_category"] == "type-alias")
n_name    = sum(1 for r in retro_results if r["alias_category"] == "name-based")

print(f"""
=== RETROACTIVE CLASSIFICATION SUMMARY ===
Total alias-resolved rows classified:  {len(retro_results)}
  Type-alias rows:                      {n_type}
  Name-based alias rows:                {n_name}
Newly flagged (PENDING audit):          {n_flagged}
Clean (CLEAN_PENDING audit):            {n_clean}
Raw records written to flagged file:    {len(flagged_raw)}
""")
