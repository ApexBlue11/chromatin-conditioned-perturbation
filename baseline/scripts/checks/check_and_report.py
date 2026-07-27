#!/usr/bin/env python
import numpy as np
import json
import os
import sys

OUTPUT_DIR = r"c:\Users\apexb\Downloads\LINCS Project\output\ccle_baseline"

# Write results to file instead of stdout
with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w") as report:
    try:
        report.write("="*70 + "\n")
        report.write("X_BASE VALIDATION REPORT\n")
        report.write("="*70 + "\n\n")
        
        # Check X_base
        xbase_path = os.path.join(OUTPUT_DIR, "X_base.npy")
        if os.path.exists(xbase_path):
            try:
                X_base = np.load(xbase_path)
                report.write(f"X_base.npy:\n")
                report.write(f"  Shape: {X_base.shape}\n")
                report.write(f"  Dtype: {X_base.dtype}\n")
                report.write(f"  NaN count: {np.isnan(X_base).sum()}\n")
                report.write(f"  File size: {os.path.getsize(xbase_path) / (1024**2):.1f} MB\n")
                report.write(f"  Value range: [{X_base[~np.isnan(X_base)].min():.3f}, {X_base[~np.isnan(X_base)].max():.3f}]\n")
                
                # Check standardization
                valid_idx = ~np.isnan(X_base).any(axis=1)
                if valid_idx.any():
                    X_valid = X_base[valid_idx]
                    gm = X_valid.mean(axis=0)
                    gs = X_valid.std(axis=0)
                    report.write(f"  Gene means: [{gm.min():.4f}, {gm.max():.4f}]\n")
                    report.write(f"  Gene stds: [{gs.min():.4f}, {gs.max():.4f}]\n")
            except Exception as e:
                report.write(f"ERROR reading X_base: {e}\n")
        else:
            report.write("X_base.npy: NOT FOUND\n")
        
        # Check X_base_raw
        xraw_path = os.path.join(OUTPUT_DIR, "X_base_raw.npy")
        if os.path.exists(xraw_path):
            try:
                X_raw = np.load(xraw_path)
                report.write(f"\nX_base_raw.npy:\n")
                report.write(f"  Shape: {X_raw.shape}\n")
                report.write(f"  Dtype: {X_raw.dtype}\n")
                report.write(f"  NaN count: {np.isnan(X_raw).sum()}\n")
                report.write(f"  File size: {os.path.getsize(xraw_path) / (1024**2):.1f} MB\n")
            except Exception as e:
                report.write(f"ERROR reading X_base_raw: {e}\n")
        else:
            report.write("\nX_base_raw.npy: NOT FOUND\n")
        
        # Check other files
        report.write(f"\nOther output files:\n")
        for fname in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                report.write(f"  {fname}: {size_kb:.1f} KB\n")
        
        report.write("\n" + "="*70 + "\n")
        report.write("Validation report written successfully\n")
    
    except Exception as e:
        report.write(f"FATAL ERROR: {e}\n")
        import traceback
        report.write(traceback.format_exc())

# Also try to print to console
print("Validation report written to:", os.path.join(OUTPUT_DIR, "validation_report.txt"))
