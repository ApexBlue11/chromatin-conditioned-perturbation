import numpy as np
import os

BASE = r"c:\Users\apexb\Downloads\LINCS Project\output\ccle_baseline"

print("Checking X_base files...")
try:
    X = np.load(os.path.join(BASE, "X_base.npy"))
    print(f"X_base: {X.shape}, {X.dtype}, NaN={np.isnan(X).sum()}")
    print(f"  Values: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Mean per gene: {X.mean(axis=0)[:5]}")
    print(f"  Std per gene: {X.std(axis=0)[:5]}")
except Exception as e:
    print(f"ERROR reading X_base: {e}")

try:
    X_raw = np.load(os.path.join(BASE, "X_base_raw.npy"))
    print(f"\nX_base_raw: {X_raw.shape}, {X_raw.dtype}, NaN={np.isnan(X_raw).sum()}")
    print(f"  Values: [{X_raw.min():.3f}, {X_raw.max():.3f}]")
except Exception as e:
    print(f"ERROR reading X_base_raw: {e}")

print("\n[SUCCESS] Both files loaded successfully")
