"""
Simplified CCLE Baseline Expression Matrix Pipeline
- More robust memory management
- Better error reporting
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
import json
import pickle
import re
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"c:\Users\apexb\Downloads\LINCS Project"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "ccle_baseline")
NETWORK_DIR = os.path.join(BASE_DIR, "Network Data")

try:
    # Load genes
    print("Loading landmark genes...")
    with open(os.path.join(NETWORK_DIR, "pathway_landmark_genes.txt")) as f:
        landmark_genes = [line.strip() for line in f if line.strip()]
    print(f"OK: {len(landmark_genes)} landmark genes")
    
    # Load CCLE
    print("\nLoading CCLE expression (~5 min)...")
    ccle_path = os.path.join(BASE_DIR, "CCLE", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
    ccle = pd.read_csv(ccle_path, index_col=0)
    
    # Drop metadata columns
    meta_cols = ['SequencingID', 'ModelConditionID', 'ModelID', 'IsDefaultEntryForMC', 'IsDefaultEntryForModel']
    gene_cols = [c for c in ccle.columns if c not in meta_cols]
    ccle = ccle[gene_cols]
    print(f"OK: {ccle.shape}")
    
    # Parse genes
    print("\nParsing gene headers...")
    pattern = r'(.+?)\s*\((\d+)\)$'
    entrez_map = {}
    symbol_map = {}
    
    for col in ccle.columns:
        m = re.match(pattern, col)
        if m:
            sym = m.group(1).strip().upper()
            ent = int(m.group(2))
            entrez_map[ent] = col
            symbol_map[sym] = col
    
    print(f"OK: {len(entrez_map)} genes parsed")
    
    # Match landmarks
    print("\nMatching landmarks to CCLE...")
    matched_cols = []
    missing = []
    
    for gene in landmark_genes:
        # Try Entrez first (need gene_info for this)
        # For now, try symbol match
        if gene.upper() in symbol_map:
            matched_cols.append(ccle[symbol_map[gene.upper()]])
        else:
            missing.append(gene)
            matched_cols.append(pd.Series([np.nan] * len(ccle)))
    
    ccle_lm = pd.DataFrame({gene: matched_cols[i] for i, gene in enumerate(landmark_genes)})
    
    # Fill NaN
    for col in ccle_lm.columns:
        ccle_lm[col].fillna(ccle_lm[col].mean(), inplace=True)
    
    print(f"OK: Matched {len(landmark_genes) - len(missing)}/978, Missing: {missing[:3] if missing else 'none'}")
    
    # Build array
    print("\nBuilding X_base_raw...")
    X_raw = ccle_lm.values.astype(np.float32)
    print(f"OK: {X_raw.shape}")
    
    # Standardize
    print("\nStandardizing...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    print(f"OK: mean={X.mean(axis=0).mean():.3f}, std={X.std(axis=0).mean():.3f}")
    
    # Save
    print("\nSaving outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_base.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "X_base_raw.npy"), X_raw)
    print(f"OK: Saved X_base.npy and X_base_raw.npy")
    
    # Save scaler
    with open(os.path.join(OUTPUT_DIR, "scalers", "ccle_expression_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"OK: Saved scaler")
    
    # Index
    cell_ids = ccle_lm.index.tolist()
    with open(os.path.join(OUTPUT_DIR, "cell_line_names.txt"), "w") as f:
        f.write("\n".join(cell_ids))
    
    index_map = {cid: i for i, cid in enumerate(cell_ids)}
    with open(os.path.join(OUTPUT_DIR, "cell_line_index.json"), "w") as f:
        json.dump({"cell_id_to_row": index_map}, f)
    
    print(f"\nPIPELINE COMPLETE")
    print(f"Shape: {X.shape}")
    print(f"Saved to: {OUTPUT_DIR}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
