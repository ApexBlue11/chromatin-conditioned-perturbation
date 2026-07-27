"""
Step 1: Analyze Phase 1 vs Phase 2 LINCS differences
- Decompress metadata files
- Load 978 landmark genes
- Compare Level 5 gctx structures
- Analyze inst_info for cell lines and conditions
"""

import os
import gzip
import pandas as pd
import h5py
import numpy as np

# ──── PATHS ────────────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\apexb\Downloads\LINCS Project"
METADATA_DIR = os.path.join(BASE_DIR, "LINCS L1000 MetaData")
NETWORK_DIR = os.path.join(BASE_DIR, "Network Data")
PHASE1_DIR = os.path.join(BASE_DIR, "phase 1")
PHASE2_DIR = os.path.join(BASE_DIR, "phase 2")
CCLE_DIR = os.path.join(BASE_DIR, "CCLE")

# ──── FUNCTION: Decompress gzip file ────────────────────────────────────
def decompress_gz(gz_path, output_path):
    """Decompress .gz file if not already decompressed."""
    if os.path.exists(output_path):
        print(f"  Already exists: {os.path.basename(output_path)}")
        return output_path
    
    print(f"  Decompressing {os.path.basename(gz_path)}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.writelines(f_in)
    print(f"    → {os.path.basename(output_path)}")
    return output_path

# ──── STEP 1: Load canonical 978 landmark genes ────────────────────────
print("\n" + "="*70)
print("STEP 1: Load canonical 978 landmark genes")
print("="*70)

gene_order_path = os.path.join(NETWORK_DIR, "pathway_landmark_genes.txt")
with open(gene_order_path) as f:
    landmark_genes = [line.strip() for line in f if line.strip()]

print(f"✓ Loaded {len(landmark_genes)} canonical landmark genes")
print(f"  First 5: {landmark_genes[:5]}")
print(f"  Last 5: {landmark_genes[-5:]}")
assert len(landmark_genes) == 978, f"Expected 978 genes, got {len(landmark_genes)}"

# ──── STEP 2: Decompress and load metadata files ────────────────────────
print("\n" + "="*70)
print("STEP 2: Decompress and load LINCS metadata")
print("="*70)

# Decompress files
print("Decompressing .gz files...")
gene_info_gz = os.path.join(METADATA_DIR, "GSE92742_Broad_LINCS_gene_info.txt.gz")
sig_info_gz = os.path.join(METADATA_DIR, "GSE92742_Broad_LINCS_sig_info.txt.gz")
inst_info_gz = os.path.join(METADATA_DIR, "GSE92742_Broad_LINCS_inst_info.txt.gz")
cell_info_gz = os.path.join(METADATA_DIR, "GSE92742_Broad_LINCS_cell_info.txt.gz")

gene_info_path = decompress_gz(gene_info_gz, gene_info_gz.replace('.gz', ''))
sig_info_path = decompress_gz(sig_info_gz, sig_info_gz.replace('.gz', ''))
inst_info_path = decompress_gz(inst_info_gz, inst_info_gz.replace('.gz', ''))
cell_info_path = decompress_gz(cell_info_gz, cell_info_gz.replace('.gz', ''))

# Load metadata
print("\nLoading metadata files...")
gene_info = pd.read_csv(gene_info_path, sep="\t")
sig_info = pd.read_csv(sig_info_path, sep="\t")
inst_info = pd.read_csv(inst_info_path, sep="\t")
cell_info = pd.read_csv(cell_info_path, sep="\t")

print(f"✓ gene_info: {gene_info.shape[0]:,} genes")
print(f"  Columns: {list(gene_info.columns)[:5]}...")
print(f"  pr_is_lm column sample: {gene_info['pr_is_lm'].unique()}")

print(f"\n✓ sig_info: {sig_info.shape[0]:,} signatures")
print(f"  Columns: {list(sig_info.columns)[:5]}...")
print(f"  pert_type values: {sig_info['pert_type'].unique()[:10]}")

print(f"\n✓ inst_info: {inst_info.shape[0]:,} instances")
print(f"  Columns: {list(inst_info.columns)[:5]}...")

print(f"\n✓ cell_info: {cell_info.shape[0]:,} cell lines")
print(f"  Columns: {list(cell_info.columns)[:5]}...")

# ──── STEP 3: Analyze Phase 1 and Phase 2 Level 5 gctx ──────────────────
print("\n" + "="*70)
print("STEP 3: Analyze Phase 1 vs Phase 2 Level 5 gctx")
print("="*70)

phase1_l5 = os.path.join(PHASE1_DIR, "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")
phase2_l5 = os.path.join(PHASE2_DIR, "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx")

def analyze_gctx(path, label):
    """Analyze a gctx file structure."""
    if not os.path.exists(path):
        print(f"✗ {label}: File not found")
        return None
    
    try:
        with h5py.File(path, 'r') as f:
            n_rows = len(f['0/META/ROW/id'][:])
            n_cols = len(f['0/META/COL/id'][:])
            
            # Sample rows (genes)
            row_sample = f['0/META/ROW/id'][:5]
            
            # Sample columns (instances)
            col_sample = f['0/META/COL/id'][:5]
            
            # Check if data is in the expected place
            try:
                data_shape = f['0/DATA/0/matrix'].shape
            except:
                data_shape = "Unable to read"
            
            print(f"✓ {label}:")
            print(f"    Dimensions: {n_rows:,} genes × {n_cols:,} instances")
            print(f"    File size: {os.path.getsize(path) / (1024**3):.2f} GB")
            print(f"    Data shape: {data_shape}")
            print(f"    Sample genes: {row_sample}")
            print(f"    Sample instances: {col_sample}")
            
            return {"n_genes": n_rows, "n_instances": n_cols, "file_size_gb": os.path.getsize(path) / (1024**3)}
    except Exception as e:
        print(f"✗ {label}: Error - {e}")
        return None

phase1_info = analyze_gctx(phase1_l5, "Phase 1 Level 5")
phase2_info = analyze_gctx(phase2_l5, "Phase 2 Level 5")

# ──── STEP 4: Extract LINCS cell lines from sig_info ────────────────────
print("\n" + "="*70)
print("STEP 4: Extract LINCS cell lines (from sig_info)")
print("="*70)

# Get treatment compounds
treatment_sigs = sig_info[sig_info['pert_type'] == 'trt_cp']
print(f"Treatment compound signatures (trt_cp): {len(treatment_sigs):,}")

lincs_cells = sorted(treatment_sigs['cell_id'].unique())
print(f"Unique cell lines in treatments: {len(lincs_cells):,}")
print(f"First 10 cell lines: {lincs_cells[:10]}")

# ──── STEP 5: Check CCLE data ────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5: Check CCLE data")
print("="*70)

ccle_expr_path = os.path.join(CCLE_DIR, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
if not os.path.exists(ccle_expr_path):
    # Try alternate name
    ccle_expr_path = os.path.join(CCLE_DIR, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    if not os.path.exists(ccle_expr_path):
        print("✗ CCLE expression file not found (checked both names)")
    else:
        print(f"✓ Found CCLE expression (alternate name)")
else:
    print(f"✓ Found CCLE expression (primary name)")

# Load a small sample of CCLE to check structure
ccle_expr = pd.read_csv(ccle_expr_path, index_col=0, nrows=10)
print(f"\nCCLE expression matrix (partial load):")
print(f"  Shape (first 10 rows): {ccle_expr.shape}")
print(f"  Sample column headers:\n    {list(ccle_expr.columns[:3])}")
print(f"  Sample DepMap IDs: {list(ccle_expr.index[:3])}")

# ──── SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: Phase 1 vs Phase 2")
print("="*70)

if phase1_info and phase2_info:
    print(f"\nPhase 1 Level 5: {phase1_info['n_instances']:,} instances")
    print(f"Phase 2 Level 5: {phase2_info['n_instances']:,} instances")
    print(f"Ratio (P1:P2): {phase1_info['n_instances']/phase2_info['n_instances']:.1f}x")
    
    print(f"\nDifferences:")
    print(f"  Extra instances in Phase 1: {phase1_info['n_instances'] - phase2_info['n_instances']:,}")
    print(f"  Phase 1 larger by: {(phase1_info['n_instances']/phase2_info['n_instances'] - 1)*100:.1f}%")
    
    print(f"\nBoth have same genes: {phase1_info['n_genes']} = {phase2_info['n_genes']}")
    print(f"  → Both measure the 978 landmark genes")
    
    print(f"\nPhase 1 question: More replicates, more drugs, or more cell lines?")
    print(f"  Answer: Need to analyze inst_info.txt (instances per compound per cell)")

print(f"\nNext steps:")
print(f"  1. Analyze inst_info to understand Phase 1 vs Phase 2 composition")
print(f"  2. Identify unmatched LINCS cells for Phase 1/2 Level 3 fallback")
print(f"  3. Build full X_base pipeline")
print(f"\n" + "="*70)
