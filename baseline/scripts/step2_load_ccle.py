"""
Step 2: Load CCLE data and parse gene information
- Properly load CCLE expression matrix
- Parse gene symbols and Entrez IDs from column headers
- Build lookup dictionaries
"""

import os
import pandas as pd
import numpy as np

# ──── PATHS ────────────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\apexb\Downloads\LINCS Project"
CCLE_DIR = os.path.join(BASE_DIR, "CCLE")

print("\n" + "="*70)
print("STEP 2: Load and parse CCLE data")
print("="*70)

ccle_expr_path = os.path.join(CCLE_DIR, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

print(f"\nReading CCLE expression from: {os.path.basename(ccle_expr_path)}")
print(f"File size: {os.path.getsize(ccle_expr_path) / (1024**2):.1f} MB")

# Read first few lines to understand structure
print("\nFirst 5 rows/cols to understand structure:")
ccle_test = pd.read_csv(ccle_expr_path, index_col=0, nrows=5)
print(f"Shape: {ccle_test.shape}")
print(f"\nFirst 5 column headers:")
for i, col in enumerate(ccle_test.columns[:5]):
    print(f"  {i}: {col}")

print(f"\nFirst 3 DepMap IDs (index):")
print(ccle_test.index[:3].tolist())

# Full load
print("\n" + "-"*70)
print("Full load of CCLE expression...")
ccle_expr = pd.read_csv(ccle_expr_path, index_col=0)

print(f"✓ Loaded CCLE expression:")
print(f"  Shape: {ccle_expr.shape}")
print(f"  DepMap cell lines (rows): {ccle_expr.shape[0]}")
print(f"  Genes (columns): {ccle_expr.shape[1]}")
print(f"  Data type: {ccle_expr.dtypes[0]}")
print(f"  Value range: [{ccle_expr.iloc[:, 0].min():.2f}, {ccle_expr.iloc[:, 0].max():.2f}]")

# Parse column headers: Format is likely "SYMBOL (ENTREZ_ID)"
print("\n" + "-"*70)
print("Parsing gene symbols and Entrez IDs from column headers...")

import re

# Pattern to match "SYMBOL (ENTREZ_ID)"
pattern = r'(.+?)\s*\((\d+)\)$'

gene_symbol_to_col = {}      # symbol (upper) → column name
gene_entrez_to_col = {}      # entrez_id (int) → column name
gene_symbol_to_entrez = {}   # symbol (upper) → entrez_id (int)

parse_errors = []

for col in ccle_expr.columns:
    match = re.match(pattern, col)
    if match:
        symbol = match.group(1).strip().upper()
        entrez_str = match.group(2).strip()
        try:
            entrez_id = int(entrez_str)
            gene_symbol_to_col[symbol] = col
            gene_entrez_to_col[entrez_id] = col
            gene_symbol_to_entrez[symbol] = entrez_id
        except ValueError:
            parse_errors.append(f"{col}: Could not parse Entrez ID '{entrez_str}'")
    else:
        parse_errors.append(f"{col}: Does not match pattern 'SYMBOL (ENTREZ_ID)'")

print(f"✓ Parsed {len(gene_symbol_to_col)} genes from column headers")
print(f"  Unique symbols: {len(gene_symbol_to_col)}")
print(f"  Unique Entrez IDs: {len(gene_entrez_to_col)}")

if parse_errors:
    print(f"\n⚠ Parse errors (first 10):")
    for err in parse_errors[:10]:
        print(f"  {err}")

# Sample parsed genes
print(f"\nSample parsed genes:")
sample_cols = list(gene_symbol_to_col.items())[:5]
for symbol, col in sample_cols:
    entrez = gene_symbol_to_entrez[symbol]
    print(f"  {symbol} (Entrez: {entrez})")

# Load CCLE Model.csv for cell line metadata
print("\n" + "-"*70)
print("Loading CCLE Model.csv for cell line names...")

model_path = os.path.join(CCLE_DIR, "Model.csv")
model_df = pd.read_csv(model_path)

print(f"✓ Loaded Model.csv:")
print(f"  Shape: {model_df.shape}")
print(f"  Columns: {list(model_df.columns)[:5]}...")

# Check for key columns
key_cols = ['ModelID', 'StrippedCellLineName', 'OncotreeLineage']
for col in key_cols:
    if col in model_df.columns:
        print(f"  ✓ Has {col}")
    else:
        print(f"  ✗ Missing {col}")

# Build DepMap ID → name mapping
depmap_to_name = {}
depmap_to_tissue = {}

if 'ModelID' in model_df.columns and 'StrippedCellLineName' in model_df.columns:
    for idx, row in model_df.iterrows():
        mid = row.get('ModelID')
        name = row.get('StrippedCellLineName')
        tissue = row.get('OncotreeLineage', 'unknown')
        if mid and name:
            depmap_to_name[mid] = name
            depmap_to_tissue[mid] = tissue

print(f"\n✓ Built DepMap ID → cell line name mapping:")
print(f"  {len(depmap_to_name)} mappings")
print(f"  Sample: {list(depmap_to_name.items())[:3]}")

# Summary
print("\n" + "="*70)
print("SUMMARY: CCLE Data Ready")
print("="*70)
print(f"CCLE expression matrix: {ccle_expr.shape[0]} cells × {ccle_expr.shape[1]} genes")
print(f"Parsed genes: {len(gene_symbol_to_col)} with valid Entrez IDs")
print(f"Cell line metadata: {len(depmap_to_name)} DepMap IDs")
print(f"\nReady to match with LINCS cell lines")
