import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Okabe-Ito color-blind safe palette
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

OUT_DIR = "model/figures/out"
os.makedirs(OUT_DIR, exist_ok=True)

def p_val(fig, name, val):
    print(f"{fig} | {name}: {val}")

def make_f1():
    with open("model/results/coldcell_h2h_split_cold_cell_1_O2.json", "r") as f:
        data = json.load(f)
    
    cells = data["per_cell"]
    # Sort by n_scored
    cells = sorted(cells, key=lambda x: x["n_scored"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    y_pos = np.arange(len(cells))
    labels = [f"{c['cell']} (n={c['n_scored']})" for c in cells]
    
    d_c_medians = [c["d_c_median"] for c in cells]
    d_c_err_low = [c["d_c_median"] - c["d_c_median_ci95"][0] for c in cells]
    d_c_err_high = [c["d_c_median_ci95"][1] - c["d_c_median"] for c in cells]
    
    ax1.errorbar(d_c_medians, y_pos, xerr=[d_c_err_low, d_c_err_high], fmt='o', color=COLORS[1])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.axvline(0, color='black', linestyle='--')
    
    cluster_mean = data["cluster"]["mean_of_d_c"]
    cluster_ci = data["cluster"]["cluster_ci95"]
    
    ax1.axvspan(cluster_ci[0], cluster_ci[1], alpha=0.2, color=COLORS[2])
    ax1.axvline(cluster_mean, color=COLORS[5], linestyle=':')
    ax1.set_xlabel("d_c_median")
    ax1.set_title("per held-out cell line")
    
    ours_means = [c["ours_mean"] for c in cells]
    theirs_means = [c["theirs_mean"] for c in cells]
    
    for i in range(len(cells)):
        ax2.plot([ours_means[i], theirs_means[i]], [y_pos[i], y_pos[i]], color='gray', zorder=1)
        ax2.scatter([ours_means[i]], [y_pos[i]], color=COLORS[5], label='ours' if i==0 else "", zorder=2)
        ax2.scatter([theirs_means[i]], [y_pos[i]], color=COLORS[6], label='theirs' if i==0 else "", zorder=2)
        
        p_val("F1", f"cell_{cells[i]['cell']}_n_scored", cells[i]["n_scored"])
        p_val("F1", f"cell_{cells[i]['cell']}_d_c_median", cells[i]["d_c_median"])
        p_val("F1", f"cell_{cells[i]['cell']}_d_c_median_ci95_0", cells[i]["d_c_median_ci95"][0])
        p_val("F1", f"cell_{cells[i]['cell']}_d_c_median_ci95_1", cells[i]["d_c_median_ci95"][1])
        p_val("F1", f"cell_{cells[i]['cell']}_ours_mean", cells[i]["ours_mean"])
        p_val("F1", f"cell_{cells[i]['cell']}_theirs_mean", cells[i]["theirs_mean"])
    
    p_val("F1", "cluster_mean_of_d_c", cluster_mean)
    p_val("F1", "cluster_ci95_0", cluster_ci[0])
    p_val("F1", "cluster_ci95_1", cluster_ci[1])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.legend()
    ax2.set_xlabel("mean score")
    ax2.set_title("model per-cell mean scores")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_coldcell_h2h.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f1_coldcell_h2h.svg"))
    plt.close()
    
    cells_favouring_ours = data["cluster"]["cells_favouring_ours"]
    n_cells = data["cluster"]["n_cells"]
    
    p_val("F1", "cells_favouring_ours", cells_favouring_ours)
    p_val("F1", "n_cells", n_cells)
    
    caption = f"v9 minus XPert (trained to its published recipe), per-row delta Pearson; {cells_favouring_ours} of {n_cells} cells favour v9; the pre-registered criterion (≥ 7 of 8 and a cluster CI excluding 0) was not met."
    with open(os.path.join(OUT_DIR, "f1_coldcell_h2h.txt"), "w", encoding="utf-8") as f:
        f.write(caption)

def make_f2():
    with open("model/results/coldcell_h2h_split_cold_cell_1_O2.json", "r") as f:
        data = json.load(f)
    
    # XPert's published cold-cell mean (Nature Machine Intelligence 2025, Supplementary Table R8)
    pub_mean = 0.383
    pub_sd = 0.027
    
    ours_mean = data["reproduction"]["theirs_all_rows_mean"]
    band = data["reproduction"]["band"]
    
    p_val("F2", "pub_mean", pub_mean)
    p_val("F2", "pub_sd", pub_sd)
    p_val("F2", "theirs_all_rows_mean", ours_mean)
    p_val("F2", "band_0", band[0])
    p_val("F2", "band_1", band[1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Published XPert", "Trained XPert"], [pub_mean, ours_mean], yerr=[pub_sd, 0], capsize=5, color=[COLORS[1], COLORS[5]])
    ax.axhspan(band[0], band[1], color='gray', alpha=0.2, label='reproduction.band')
    ax.set_ylabel("Mean Score")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f2_xpert_reproduction.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f2_xpert_reproduction.svg"))
    plt.close()

def make_f3():
    with open("model/results/v9_dev_score_baseline_kaggle.json", "r") as f:
        base_data = json.load(f)
    s0 = base_data["sd"]
    adv_thresh = max(2 * s0, 0.003)
    
    p_val("F3", "s0", s0)
    p_val("F3", "adv_thresh", adv_thresh)
    
    files = glob.glob("model/results/v9_dev_score_*.json")
    points = []
    for f in files:
        with open(f, "r") as fd:
            data = json.load(fd)
        if "vs_baseline" in data:
            label = data["label"]
            delta = data["vs_baseline"]["delta_per_row_mean"]
            points.append((label, delta))
            p_val("F3", f"cand_{label}_delta", delta)
            
    points = sorted(points, key=lambda x: x[1])
    labels = [p[0] for p in points]
    deltas = [p[1] for p in points]
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(points)*0.5)))
    y_pos = np.arange(len(points))
    ax.plot(deltas, y_pos, 'o', color=COLORS[1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(s0, color=COLORS[3], linestyle='--', label=f'Drop thresh (+{s0:.5f})')
    ax.axvline(adv_thresh, color=COLORS[5], linestyle='-.', label=f'Advance thresh (+{adv_thresh:.5f})')
    
    ax.set_xlabel("delta_per_row_mean")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f3_dev_screens.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f3_dev_screens.svg"))
    plt.close()

def make_f4():
    with open("model/results/probe_moa_v9_reading_86.json", "r") as f:
        data = json.load(f)
        
    floor = -0.02
    reading = data.get("reading", "NULL")
    
    p_val("F4", "floor", floor)
    p_val("F4", "reading", reading)
    
    trained_diffs = []
    untrained_diffs = []
    seeds = ["0", "1", "2"]
    
    for s in seeds:
        td = data["seeds"][s]["trained"]["diff"]
        ud = data["seeds"][s]["untrained"]["diff"]
        trained_diffs.append(td)
        untrained_diffs.append(ud)
        
        p_val("F4", f"seed_{s}_trained_diff", td)
        p_val("F4", f"seed_{s}_untrained_diff", ud)
        
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(seeds))
    width = 0.35
    
    ax.bar(x - width/2, trained_diffs, width, label='Trained', color=COLORS[5])
    ax.bar(x + width/2, untrained_diffs, width, label='Untrained', color=COLORS[6])
    
    ax.axhline(floor, color='black', linestyle='--', label=f'Floor ({floor})')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Diff")
    ax.set_title(f"MOA Probe (Reading: {reading})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f4_moa_probe.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f4_moa_probe.svg"))
    plt.close()

def make_f5():
    with open("model/results/cc1_input_coverage.json", "r") as f:
        data = json.load(f)
        
    summary = data["summary"]
    groups = ["train", "dev", "test"]
    
    metrics = ["ccle_direct_rows_frac", "ATAC_rows_frac", "H3K27ac_rows_frac", "H3K27me3_rows_frac"]
    metric_labels = ["Direct CCLE Match", "ATAC", "H3K27ac", "H3K27me3"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.2
    
    for i, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        vals = [summary[g][metric] for g in groups]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m_label, color=COLORS[i+1])
        for g in groups:
            p_val("F5", f"{g}_{metric}", summary[g][metric])
            
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel("Row Fraction")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f5_input_coverage.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f5_input_coverage.svg"))
    plt.close()

if __name__ == "__main__":
    make_f1()
    make_f2()
    make_f3()
    make_f4()
    make_f5()
