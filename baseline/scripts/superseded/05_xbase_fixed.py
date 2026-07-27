"""
CCLE Baseline Matrix - Fixed version with proper Entrez ID matching
Rule 2: Gene matching is Entrez-ID-first
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import re
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"c:\Users\apexb\Downloads\LINCS Project"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "ccle_baseline")
NETWORK_DIR = os.path.join(BASE_DIR, "Network Data")
METADATA_DIR = os.path.join(BASE_DIR, "LINCS L1000 MetaData")

print("="*70)
print("X_BASE PIPELINE - FIXED VERSION")
print("="*70)

try:
    # Load 978 landmark genes
    print("\n[1] Loading landmark genes...")
    with open(os.path.join(NETWORK_DIR, "pathway_landmark_genes.txt")) as f:
        landmark_genes = [line.strip() for line in f if line.strip()]
    print(f"    OK: {len(landmark_genes)} genes")
    
    # Load gene_info to build Entrez mapping
    print("\n[2] Loading LINCS gene_info for Entrez mapping...")
    gene_info = pd.read_csv(
        os.path.join(METADATA_DIR, "GSE92742_Broad_LINCS_gene_info.txt"),
        sep="\t", low_memory=False
    )
    
    # Build landmark → Entrez mapping
    landmark_to_entrez = {}
    entrez_to_landmark = {}
    
    for idx, row in gene_info.iterrows():
        if row['pr_is_lm'] == 1:  # Landmark gene
            symbol = row['pr_gene_symbol'].strip().upper()
            try:
                entrez = int(row['pr_gene_id'])
                landmark_to_entrez[symbol] = entrez
                entrez_to_landmark[entrez] = symbol
            except:
                pass
    
    print(f"    OK: Built mapping for {len(landmark_to_entrez)} landmark genes")
    print(f"    Canonical Entrez IDs: {list(landmark_to_entrez.values())[:5]}")
    
    # Load CCLE expression
    print("\n[3] Loading CCLE expression data (~5 min)...")
    ccle_path = os.path.join(BASE_DIR, "CCLE", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
    ccle = pd.read_csv(ccle_path, index_col=0)
    
    # Drop metadata columns
    meta_cols = ['SequencingID', 'ModelConditionID', 'ModelID', 'IsDefaultEntryForMC', 'IsDefaultEntryForModel']
    gene_cols = [c for c in ccle.columns if c not in meta_cols]
    ccle = ccle[gene_cols]
    print(f"    OK: {ccle.shape[0]} DepMap cells x {ccle.shape[1]} genes")
    
    # Parse CCLE column headers for Entrez IDs
    print("\n[4] Parsing CCLE gene headers...")
    pattern = r'(.+?)\s*\((\d+)\)$'
    ccle_entrez_map = {}  # Entrez ID → column name
    ccle_symbol_map = {}  # Symbol (upper) → column name
    
    for col in ccle.columns:
        m = re.match(pattern, col)
        if m:
            symbol = m.group(1).strip().upper()
            try:
                entrez = int(m.group(2))
                ccle_entrez_map[entrez] = col
                ccle_symbol_map[symbol] = col
            except:
                pass
    
    print(f"    OK: {len(ccle_entrez_map)} genes with Entrez IDs")
    
    # Match landmark genes to CCLE (Priority 1: Entrez, Priority 2: Symbol)
    print("\n[5] Matching landmarks to CCLE (Entrez-first)...")
    landmark_matched = []
    landmark_missing = []
    
    for gene_symbol in landmark_genes:
        entrez_id = landmark_to_entrez.get(gene_symbol)
        matched = False
        
        # Priority 1: Entrez ID match
        if entrez_id and entrez_id in ccle_entrez_map:
            ccle_col = ccle_entrez_map[entrez_id]
            landmark_matched.append((gene_symbol, ccle_col, 'entrez'))
            matched = True
        
        # Priority 2: Symbol match
        elif gene_symbol in ccle_symbol_map:
            ccle_col = ccle_symbol_map[gene_symbol]
            landmark_matched.append((gene_symbol, ccle_col, 'symbol'))
            matched = True
        
        if not matched:
            landmark_missing.append(gene_symbol)
            landmark_matched.append((gene_symbol, None, 'missing'))
    
    print(f"    OK: Matched {len(landmark_matched) - len(landmark_missing)}/978")
    if landmark_missing:
        print(f"    Missing genes: {landmark_missing}")
        if len(landmark_missing) > 10:
            print(f"    [ESCALATION] More than 10 genes missing!")
            sys.exit(1)
    
    # Build landmark matrix from CCLE
    print("\n[6] Building landmark matrix...")
    landmark_data = []
    
    for gene_symbol, ccle_col, match_type in landmark_matched:
        if ccle_col:
            landmark_data.append(ccle[ccle_col].values.astype(np.float32))
        else:
            # Missing gene - will fill with mean later
            landmark_data.append(np.full(len(ccle), np.nan, dtype=np.float32))
    
    # Transpose to get (N_cells, 978)
    X_raw_temp = np.array(landmark_data, dtype=np.float32).T
    print(f"    Shape before imputation: {X_raw_temp.shape}")
    
    # Fill NaN with per-gene mean
    print("\n[7] Imputing missing values...")
    imputed_count = 0
    for gene_idx in range(X_raw_temp.shape[1]):
        if np.isnan(X_raw_temp[:, gene_idx]).any():
            gene_mean = np.nanmean(X_raw_temp[:, gene_idx])
            if np.isnan(gene_mean):
                # All NaN for this gene - use global mean
                gene_mean = np.nanmean(X_raw_temp)
            X_raw_temp[np.isnan(X_raw_temp[:, gene_idx]), gene_idx] = gene_mean
            imputed_count += 1
    
    X_raw = X_raw_temp.astype(np.float32)
    print(f"    Imputed {imputed_count} genes")
    print(f"    Final shape: {X_raw.shape}, NaN count: {np.isnan(X_raw).sum()}")
    
    if np.isnan(X_raw).any():
        print(f"    [ESCALATION] NaN still present in X_raw!")
        sys.exit(1)
    
    # Standardize
    print("\n[8] Standardizing...")
    scaler = StandardScaler()
    X_base = scaler.fit_transform(X_raw).astype(np.float32)
    
    print(f"    Shape: {X_base.shape}")
    print(f"    NaN count: {np.isnan(X_base).sum()}")
    print(f"    Gene means: min={X_base.mean(axis=0).min():.6f}, max={X_base.mean(axis=0).max():.6f}")
    print(f"    Gene stds: min={X_base.std(axis=0).min():.6f}, max={X_base.std(axis=0).max():.6f}")
    
    if np.isnan(X_base).any():
        print(f"    [ESCALATION] NaN in scaled X_base!")
        sys.exit(1)
    
    # Save outputs
    print("\n[9] Saving outputs...")
    os.makedirs(os.path.join(OUTPUT_DIR, "scalers"), exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, "X_base.npy"), X_base)
    np.save(os.path.join(OUTPUT_DIR, "X_base_raw.npy"), X_raw)
    print(f"    Saved: X_base.npy, X_base_raw.npy")
    
    # Save cell line names (from CCLE index)
    cell_ids = ccle.index.tolist()
    with open(os.path.join(OUTPUT_DIR, "cell_line_names.txt"), "w") as f:
        f.write("\n".join(cell_ids))
    
    # Save index
    cell_id_to_row = {cid: i for i, cid in enumerate(cell_ids)}
    with open(os.path.join(OUTPUT_DIR, "cell_line_index.json"), "w") as f:
        json.dump({
            "cell_id_to_row": cell_id_to_row,
            "row_to_cell_id": {str(i): cid for i, cid in enumerate(cell_ids)}
        }, f)
    print(f"    Saved: cell_line_names.txt ({len(cell_ids)} cells), cell_line_index.json")
    
    # Save scaler
    with open(os.path.join(OUTPUT_DIR, "scalers", "ccle_expression_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Saved: scaler")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    assert X_base.shape[1] == 978, f"Wrong gene count: {X_base.shape[1]}"
    assert X_base.shape[0] > 0, "No cells!"
    assert not np.isnan(X_base).any(), "NaN in X_base"
    assert X_base.dtype == np.float32, f"Wrong dtype: {X_base.dtype}"
    
    gm = X_base.mean(axis=0)
    gs = X_base.std(axis=0)
    assert np.abs(gm).max() < 0.5, f"Gene means not centered: {np.abs(gm).max()}"
    assert np.abs(gs - 1.0).max() < 0.3, f"Gene stds not unit: {np.abs(gs-1).max()}"
    
    print(f"\n[OK] ALL ASSERTIONS PASSED")
    print(f"\nFINAL RESULTS:")
    print(f"  X_base: {X_base.shape}")
    print(f"  Cells: {X_base.shape[0]}")
    print(f"  Genes: {X_base.shape[1]}")
    print(f"  Data type: {X_base.dtype}")
    print(f"  NaN count: {np.isnan(X_base).sum()}")
    print(f"  Gene means: [{gm.min():.4f}, {gm.max():.4f}]")
    print(f"  Gene stds: [{gs.min():.4f}, {gs.max():.4f}]")
    print(f"\n  Output dir: {OUTPUT_DIR}")
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
