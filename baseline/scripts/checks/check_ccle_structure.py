import pandas as pd
import os

BASE_DIR = r"c:\Users\apexb\Downloads\LINCS Project"
CCLE_DIR = os.path.join(BASE_DIR, "CCLE")
ccle_expr_path = os.path.join(CCLE_DIR, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

# Just read the header
df_header = pd.read_csv(ccle_expr_path, nrows=0)
print(f"Total columns: {len(df_header.columns)}")
print(f"\nFirst 20 column names:")
for i, col in enumerate(df_header.columns[:20]):
    print(f"  {i:2d}: {col}")

print(f"\nLast 10 column names:")
for i, col in enumerate(df_header.columns[-10:]):
    idx = len(df_header.columns) - 10 + i
    print(f"  {idx:4d}: {col}")

# Check if there's a gene section
print(f"\nLooking for gene columns (with parentheses for Entrez ID)...")
gene_cols = [col for col in df_header.columns if '(' in col and ')' in col]
print(f"Found {len(gene_cols)} gene columns with (Entrez) format")
if gene_cols:
    print(f"First 5: {gene_cols[:5]}")
