import h5py
import os

phase1_path = r"c:\Users\apexb\Downloads\LINCS Project\phase 1\GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
phase2_path = r"c:\Users\apexb\Downloads\LINCS Project\phase 2\GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx"

for label, path in [("Phase 1 L5", phase1_path), ("Phase 2 L5", phase2_path)]:
    if os.path.exists(path):
        try:
            with h5py.File(path, 'r') as f:
                n_rows = len(f['0/META/ROW/id'][:])
                n_cols = len(f['0/META/COL/id'][:])
                print(f"\n{label}: {n_rows:,} genes × {n_cols:,} instances")
                
                # Sample some row IDs (first few gene identifiers)
                row_ids = f['0/META/ROW/id'][:5]
                print(f"  Sample genes (first 5): {row_ids}")
                
                # Sample some column IDs (first few instance IDs)
                col_ids = f['0/META/COL/id'][:3]
                print(f"  Sample instances (first 3): {col_ids}")
                
                # Show available metadata
                print(f"  Available metadata keys: {list(f['0/META'].keys())}")
        except Exception as e:
            print(f"{label}: ERROR - {e}")
    else:
        print(f"{label}: File not found at {path}")
