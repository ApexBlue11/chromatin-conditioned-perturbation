import numpy as np
import json
import os

OUTPUT_DIR = r"c:\Users\apexb\Downloads\LINCS Project\output\ccle_baseline"

print("="*70)
print("VALIDATION: X_BASE OUTPUTS")
print("="*70)

# Load arrays
X_base = np.load(os.path.join(OUTPUT_DIR, 'X_base.npy'))
X_base_raw = np.load(os.path.join(OUTPUT_DIR, 'X_base_raw.npy'))

print(f"\n1. ARRAY SHAPES")
print(f"   X_base: {X_base.shape}")
print(f"   X_base_raw: {X_base_raw.shape}")

print(f"\n2. DATA TYPES")
print(f"   X_base: {X_base.dtype}")
print(f"   X_base_raw: {X_base_raw.dtype}")

# Check for NaN
nan_scaled = np.isnan(X_base).sum()
nan_raw = np.isnan(X_base_raw).sum()
print(f"\n3. NaN COUNTS")
print(f"   X_base: {nan_scaled}")
print(f"   X_base_raw: {nan_raw}")

# Check standardization
gm = X_base.mean(axis=0)
gs = X_base.std(axis=0)
print(f"\n4. STANDARDIZATION STATISTICS")
print(f"   Gene means: min={gm.min():.4f}, max={gm.max():.4f}, mean={gm.mean():.4f}")
print(f"   Gene stds: min={gs.min():.4f}, max={gs.max():.4f}, mean={gs.mean():.4f}")

# Check value ranges
print(f"\n5. VALUE RANGES")
print(f"   X_base: [{X_base.min():.2f}, {X_base.max():.2f}]")
print(f"   X_base_raw: [{X_base_raw.min():.2f}, {X_base_raw.max():.2f}]")

# Check cell count
with open(os.path.join(OUTPUT_DIR, 'cell_line_names.txt')) as f:
    cells = [line.strip() for line in f if line.strip()]
print(f"\n6. CELL LINES: {len(cells)}")
print(f"   First 5: {cells[:5]}")
print(f"   Last 5: {cells[-5:]}")

# Check index
with open(os.path.join(OUTPUT_DIR, 'cell_line_index.json')) as f:
    idx = json.load(f)
print(f"\n7. INDEX INTEGRITY")
print(f"   Mappings: {len(idx['cell_id_to_row'])}")

# Check scaler
import pickle
scaler_path = os.path.join(OUTPUT_DIR, 'scalers', 'ccle_expression_scaler.pkl')
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"\n8. SCALER")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")
    print(f"   n_features: {len(scaler.mean_)}")

print(f"\n" + "="*70)

# Run assertions
try:
    assert X_base.shape[1] == 978, f"Expected 978 genes, got {X_base.shape[1]}"
    assert X_base.shape[0] >= 50, f"Expected >=50 cells, got {X_base.shape[0]}"
    assert X_base_raw.shape == X_base.shape, "Shape mismatch"
    assert X_base.dtype == np.float32, f"Wrong dtype"
    assert not np.isnan(X_base).any(), "NaN in X_base"
    assert not np.isnan(X_base_raw).any(), "NaN in X_base_raw"
    assert np.abs(gm).max() < 0.5, f"Gene means off: {np.abs(gm).max()}"
    assert np.abs(gs - 1.0).max() < 0.3, f"Gene stds off"
    print("[OK] ALL ASSERTIONS PASSED")
except AssertionError as e:
    print(f"[FAIL] {e}")
