import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: the Tk backend cannot find tk.tcl on this machine
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import itertools

# v9 (blue), XPert/comparator (orange), third role/untrained/reference/null (aqua)
COLOR_V9 = "#2a78d6"
COLOR_COMP = "#eb6834"
COLOR_THIRD = "#1baf7a"
COLOR_TEXT = "#0b0b0b"
COLOR_SEC_TEXT = "#52514e"
COLOR_GRID = "#e6e6e3"
COLOR_REF = "#8a8985"
COLOR_GRAY = "#8a8985"

plt.rcParams['text.color'] = COLOR_TEXT
plt.rcParams['axes.labelcolor'] = COLOR_TEXT
plt.rcParams['xtick.color'] = COLOR_TEXT
plt.rcParams['ytick.color'] = COLOR_TEXT
plt.rcParams['axes.edgecolor'] = COLOR_TEXT
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6

def setup_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=0.5, zorder=0)

OUT_DIR = "model/figures/out"
os.makedirs(OUT_DIR, exist_ok=True)
P7_JSON = "model/results/coldcell_h2h_split_cold_cell_1_P7.json"
O2_JSON = "model/results/coldcell_h2h_split_cold_cell_1_O2.json"

def p_val(fig, name, val):
    print(f"{fig} | {name}: {val}")

# F1
def make_f1():
    with open("model/results/coldcell_h2h_split_cold_cell_1_O2.json", "r") as f:
        data = json.load(f)
    
    cells = sorted(data["per_cell"], key=lambda x: x["n_scored"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    setup_axes(ax1)
    setup_axes(ax2)
    
    y_pos = np.arange(len(cells))
    labels = [f"{c['cell']} (n={c['n_scored']})" for c in cells]
    
    d_c_medians = [c["d_c_median"] for c in cells]
    d_c_err_low = [c["d_c_median"] - c["d_c_median_ci95"][0] for c in cells]
    d_c_err_high = [c["d_c_median_ci95"][1] - c["d_c_median"] for c in cells]
    
    ax1.errorbar(d_c_medians, y_pos, xerr=[d_c_err_low, d_c_err_high], fmt='o', color=COLOR_V9, capsize=2, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.axvline(0, color=COLOR_REF, linestyle='--', zorder=1)
    
    cluster_mean = data["cluster"]["mean_of_d_c"]
    cluster_ci = data["cluster"]["cluster_ci95"]
    
    ax1.axvspan(cluster_ci[0], cluster_ci[1], alpha=0.2, color=COLOR_THIRD, zorder=1)
    ax1.axvline(cluster_mean, color=COLOR_V9, linestyle=':', zorder=2)
    ax1.set_xlabel("v9 − XPert: per-row Δ Pearson, median per cell")
    
    # Custom legend for ax1
    band_patch = Patch(color=COLOR_THIRD, alpha=0.2, label='Cluster mean 95% CI')
    mean_line = plt.Line2D([0], [0], color=COLOR_V9, linestyle=':', label='Cluster mean')
    ax1.legend(handles=[band_patch, mean_line], loc='best', frameon=False)
    
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    ours_means = [c["ours_mean"] for c in cells]
    theirs_means = [c["theirs_mean"] for c in cells]
    
    for i in range(len(cells)):
        ax2.plot([ours_means[i], theirs_means[i]], [y_pos[i], y_pos[i]], color=COLOR_GRAY, zorder=1)
        ax2.scatter([ours_means[i]], [y_pos[i]], color=COLOR_V9, zorder=3)
        ax2.scatter([theirs_means[i]], [y_pos[i]], color=COLOR_COMP, zorder=3)
        
    ax2.scatter([], [], color=COLOR_V9, label='v9')
    ax2.scatter([], [], color=COLOR_COMP, label='XPert (trained to its published recipe)')
        
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)  # out of the data rows (PI)
    ax2.set_xlabel("mean per-row Δ Pearson (per cell)")
    
    for i in range(len(cells)):
        p_val("F1", f"cell_{cells[i]['cell']}_n_scored", cells[i]["n_scored"])
        p_val("F1", f"cell_{cells[i]['cell']}_d_c_median", cells[i]["d_c_median"])
        p_val("F1", f"cell_{cells[i]['cell']}_d_c_median_ci95_0", cells[i]["d_c_median_ci95"][0])
        p_val("F1", f"cell_{cells[i]['cell']}_d_c_median_ci95_1", cells[i]["d_c_median_ci95"][1])
        p_val("F1", f"cell_{cells[i]['cell']}_ours_mean", cells[i]["ours_mean"])
        p_val("F1", f"cell_{cells[i]['cell']}_theirs_mean", cells[i]["theirs_mean"])
        
    p_val("F1", "cluster_mean_of_d_c", cluster_mean)
    p_val("F1", "cluster_ci95_0", cluster_ci[0])
    p_val("F1", "cluster_ci95_1", cluster_ci[1])
    p_val("F1", "cells_favouring_ours", data["cluster"]["cells_favouring_ours"])
    p_val("F1", "n_cells", data["cluster"]["n_cells"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_coldcell_h2h.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f1_coldcell_h2h.svg"))
    plt.close()
    
# F2
def make_f2():
    with open("model/results/coldcell_h2h_split_cold_cell_1_O2.json", "r") as f:
        data = json.load(f)
    
    pub_mean = 0.383
    pub_sd = 0.027
    ours_mean = data["reproduction"]["theirs_all_rows_mean"]
    band = data["reproduction"]["band"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    setup_axes(ax)
    ax.bar(["Published XPert", "Trained XPert"], [pub_mean, ours_mean], yerr=[pub_sd, 0], capsize=2, color=[COLOR_COMP, COLOR_V9], zorder=3)
    ax.axhspan(band[0], band[1], color=COLOR_THIRD, alpha=0.2, zorder=1)
    
    band_patch = Patch(color=COLOR_THIRD, alpha=0.2, label='Reproduction band')
    ax.legend(handles=[band_patch], frameon=False)
    
    ax.set_ylabel("Mean Score")
    
    p_val("F2", "pub_mean", pub_mean)
    p_val("F2", "pub_sd", pub_sd)
    p_val("F2", "theirs_all_rows_mean", ours_mean)
    p_val("F2", "band_0", band[0])
    p_val("F2", "band_1", band[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f2_xpert_reproduction.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f2_xpert_reproduction.svg"))
    plt.close()

# F3
def make_f3():
    with open("model/results/v9_dev_score_baseline_kaggle.json", "r") as f:
        base_data = json.load(f)
    s0 = base_data["sd"]
    
    rows = [
        ("C1_noatoms", "C1 no atom tokens (1 seed)", "dropped", "v9_dev_score_C1_noatoms.json"),
        ("C2_lctl0", "C2 no control encoder (1 seed)", "dropped", "v9_dev_score_C2_lctl0.json"),
        ("C3_listnet", "C3 ListNet ranking loss (1 seed)", "dropped", "v9_dev_score_C3_listnet.json"),
        ("C4_degk50", "C4 DEG-reweighted loss (1 seed)", "dropped", "v9_dev_score_C4_degk50.json"),
        ("C6_signhead_3seed", "C6 sign head (3 seeds)", "accepted", "v9_dev_score_C6_signhead_3seed.json"),
        ("C7_chromedges_3seed", "C7 chromatin-gated edges (3 seeds)", "not accepted", "v9_dev_score_C7_chromedges_3seed.json"),
        ("C8b_postpath", "C8b post-drug pathway layer (3 seeds)", "not accepted", "v9_dev_score_C8b_postpath.json"),
        ("V2_snap3", "V2 snapshot ensemble, 3 cycles (3 seeds)", "accepted", "v9_dev_score_V2_snap3.json"),
        ("P6_c6_v2", "P6 stack: C6 + V2 (3 seeds)", "confirmed (stack)", "v9_dev_score_P6_c6_v2.json")
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    setup_axes(ax1)
    setup_axes(ax2)
    
    y_pos = np.arange(len(rows))[::-1]
    
    # references are neutral (PI polish): the band is the baseline's seed spread, the lines are rule 6 / rule 7 thresholds
    ax1.axvspan(-s0, s0, color=COLOR_REF, alpha=0.12, zorder=1)
    ax1.axvline(0, color=COLOR_TEXT, linestyle='-', linewidth=1.5, zorder=2)
    ax1.axvline(0.00169, color=COLOR_REF, linestyle='-', linewidth=1, zorder=2)
    ax1.axvline(0.0034, color=COLOR_REF, linestyle='-', linewidth=1, zorder=2)
    ax1.axvline(0.003, color=COLOR_REF, linestyle='--', linewidth=1, zorder=2)
    import matplotlib.patches as _mp
    import matplotlib.lines as _ml
    ax1.legend(handles=[_mp.Patch(color=COLOR_REF, alpha=0.12, label='baseline seed spread (± s0)'),
                        _ml.Line2D([], [], color=COLOR_REF, lw=1, label='rule 6: drop s0 = 0.0017 / advance 0.0034'),
                        _ml.Line2D([], [], color=COLOR_REF, lw=1, ls='--', label='rule 7: accept floor 0.003')],
               loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)
    
    ax2.axvline(0, color=COLOR_TEXT, linestyle='-', linewidth=1.5, zorder=2)
    
    labels = []
    decision_texts = []
    
    c8b_y = None
    v2_y = None
    c1_y = None
    
    for i, (key, label, decision, fname) in enumerate(rows):
        labels.append(label)
        with open(os.path.join("model/results", fname), "r") as f:
            data = json.load(f)
        
        vb = data["vs_baseline"]
        delta = vb["delta_per_row_mean"]
        sd = data.get("sd", 0.0)
        
        color = COLOR_V9 if decision in ["accepted", "confirmed (stack)"] else COLOR_GRAY
        y = y_pos[i]
        
        if key == "C8b_postpath":
            c8b_y = y
        elif key == "V2_snap3":
            v2_y = y
        elif key == "C1_noatoms":
            c1_y = y
        
        if vb.get("n_seeds_variant", 1) == 3:
            err = 2 * np.sqrt((s0**2)/3 + (sd**2)/3)
            ax1.errorbar(delta, y, xerr=err, fmt='o', color=color, capsize=2, zorder=3)
            p_val("F3", f"{key}_err", err)
        else:
            ax1.plot(delta, y, 'o', color=color, zorder=3)
            
        p_val("F3", f"{key}_delta_per_row_mean", delta)
        
        decision_texts.append((y, decision, color))
        
        if "delta_centred" in vb:
            dc = vb["delta_centred"]
            ax2.plot(dc, y, 'o', color=color, zorder=3)
            p_val("F3", f"{key}_delta_centred", dc)
        else:
            ax2.text(0, y, "n/a", va='center', ha='center', color=COLOR_GRAY, fontsize=9, zorder=4,
                     bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
            
    # Draw separator between C8b and V2
    sep_y = (c8b_y + v2_y) / 2
    ax1.axhline(sep_y, color=COLOR_GRID, linewidth=1, zorder=1)
    ax2.axhline(sep_y, color=COLOR_GRID, linewidth=1, zorder=1)
    
    # Draw group labels
    ax1.text(ax1.get_xlim()[0], c1_y + 0.5, "architecture / loss screens", va='bottom', ha='left', color=COLOR_SEC_TEXT, fontsize=8)
    ax1.text(ax1.get_xlim()[0], sep_y - 0.06, "  variance and stack", va='top', ha='left', color=COLOR_SEC_TEXT, fontsize=8)
    
    # Draw decision texts after loop using final xlim
    final_xlim = ax1.get_xlim()[1]
    for y, decision, color in decision_texts:
        ax1.text(final_xlim, y, f"  {decision}", va='center', ha='left', color=COLOR_SEC_TEXT, fontsize=9)
        
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Δ vs baseline, per-row delta Pearson (dev cells)")
    ax2.set_xlabel("Δ vs baseline, per-row delta Pearson (dev cells)\ncell-centred")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    p_val("F3", "s0", s0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f3_dev_screens.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f3_dev_screens.svg"))
    plt.close()

# F4
def make_f4():
    with open("model/results/probe_moa_v9_reading_86.json", "r") as f:
        data = json.load(f)
        
    floor = -0.02
    reading = data.get("reading", "NULL")
    
    seeds = ["0", "1", "2"]
    trained_diffs = []
    untrained_diffs = []
    trained_ps = []
    untrained_ps = []
    
    for s in seeds:
        td = data["seeds"][s]["trained"]["diff"]
        ud = data["seeds"][s]["untrained"]["diff"]
        tp = data["seeds"][s]["trained"]["p"]
        up = data["seeds"][s]["untrained"]["p"]
        
        trained_diffs.append(td)
        untrained_diffs.append(ud)
        trained_ps.append(tp)
        untrained_ps.append(up)
        
        p_val("F4", f"seed_{s}_trained_diff", td)
        p_val("F4", f"seed_{s}_untrained_diff", ud)
        p_val("F4", f"seed_{s}_trained_p", tp)
        p_val("F4", f"seed_{s}_untrained_p", up)
        
    p_val("F4", "floor", floor)
    p_val("F4", "reading", reading)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    setup_axes(ax)
    
    x = np.arange(len(seeds))
    width = 0.35
    
    bars_t = ax.bar(x - width/2, trained_diffs, width, label='Trained', color=COLOR_V9, zorder=3)
    bars_u = ax.bar(x + width/2, untrained_diffs, width, label='Untrained', color=COLOR_THIRD, zorder=3)
    
    ax.axhline(floor, color=COLOR_REF, linestyle='--', zorder=2)
    
    for i, bar in enumerate(bars_t):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval - 0.005 if yval < 0 else yval + 0.005, f"p={trained_ps[i]}", ha='center', va='top' if yval < 0 else 'bottom', fontsize=9, color=COLOR_TEXT)
        
    for i, bar in enumerate(bars_u):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval - 0.005 if yval < 0 else yval + 0.005, f"p={untrained_ps[i]}", ha='center', va='top' if yval < 0 else 'bottom', fontsize=9, color=COLOR_TEXT)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("S − label-permutation null (rank percentile; lower = better)")
    
    ax.text(0.02, 0.98, f"Reading: {reading}", transform=ax.transAxes, va='top', ha='left', fontsize=9)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f4_moa_probe.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f4_moa_probe.svg"))
    plt.close()

# F5
def make_f5():
    with open("model/results/cc1_input_coverage.json", "r") as f:
        data = json.load(f)
        
    summary = data["summary"]
    groups = ["train", "dev", "test"]
    metrics = ["ccle_direct_rows_frac", "ATAC_rows_frac", "H3K27ac_rows_frac", "H3K27me3_rows_frac"]
    metric_labels = ["Direct CCLE Match", "ATAC", "H3K27ac", "H3K27me3"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    setup_axes(ax)
    
    x = np.arange(len(groups))
    width = 0.2
    colors = [COLOR_V9, COLOR_COMP, COLOR_THIRD, COLOR_GRAY]
    
    for i, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        vals = [summary[g][metric] for g in groups]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m_label, color=colors[i], zorder=3)
        for g in groups:
            p_val("F5", f"{g}_{metric}", summary[g][metric])
            
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel("Row Fraction")
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f5_input_coverage.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f5_input_coverage.svg"))
    plt.close()

# F6
def make_f6():
    import sys
    sys.path.append('model')
    from v9.score_dev import row_pearson, centred_r, cell_of_rows
    
    p2_files = [f'external/kaggle_out/v9dev_base2/v9dev_base_dev6s0_seed{i}.npz' for i in range(3)]
    v2_files = [f'external/kaggle_out/v9dev_v2/v9dev_v2_dev6s0_seed{i}.npz' for i in range(3)]
    c8b_files = [f'external/kaggle_out/v9dev_c8b/v9dev_c8b_dev6s0_seed{i}.npz' for i in range(3)]
    
    def evaluate_ensemble(files):
        runs = []
        for f in files:
            z = np.load(f)
            runs.append({'pred': z['deg_pred'], 'true': z['y_true'] - z['ctl_true'], 'row_index': z['row_index']})
        
        cells = cell_of_rows(runs[0]['row_index'])
        raw_scores = {1: [], 2: [], 3: []}
        centred_scores = {1: [], 2: [], 3: []}
        
        for K in [1, 2, 3]:
            for combo in itertools.combinations(runs, K):
                avg_pred = np.mean([r['pred'] for r in combo], axis=0)
                true = combo[0]['true']
                run_dict = {'pred': avg_pred, 'true': true}
                
                raw_r = row_pearson(avg_pred, true).mean()
                c_r = centred_r(run_dict, cells).mean()
                
                raw_scores[K].append(raw_r)
                centred_scores[K].append(c_r)
        return raw_scores, centred_scores

    p2_raw, p2_centred = evaluate_ensemble(p2_files)
    v2_raw, v2_centred = evaluate_ensemble(v2_files)
    c8b_raw, c8b_centred = evaluate_ensemble(c8b_files)
    
    assert abs(p2_raw[3][0] - 0.4662) <= 5e-5, f"P2 K=3 raw mismatch: {p2_raw[3][0]}"
    assert abs(p2_centred[3][0] - 0.5018) <= 5e-5, f"P2 K=3 centred mismatch: {p2_centred[3][0]}"
    
    with open("model/results/v9_dev_score_V2_snap3.json", "r") as f:
        v2_snap3_data = json.load(f)
        
    v2_k1_raw_mean = np.mean(v2_raw[1])
    v2_k1_centred_mean = np.mean(v2_centred[1])
    
    assert abs(v2_k1_raw_mean - v2_snap3_data["mean"]) <= 1e-6, f"V2 K=1 raw mean mismatch: {v2_k1_raw_mean} vs {v2_snap3_data['mean']}"
    assert abs(v2_k1_centred_mean - v2_snap3_data["mean_centred"]) <= 1e-6, f"V2 K=1 centred mean mismatch: {v2_k1_centred_mean} vs {v2_snap3_data['mean_centred']}"
    
    with open("model/results/v9_dev_score_V1_full.json", "r") as f:
        v1_data = json.load(f)
        
    gain_mc = v1_data["vs_baseline"]["delta_per_row_mean"]
    gain_snapshot = v2_snap3_data["vs_baseline"]["delta_per_row_mean"]
    gain_seed3 = p2_raw[3][0] - v2_snap3_data["vs_baseline"]["baseline_mean"]
    
    assert abs(gain_seed3 - (0.4662 - 0.43693)) <= 5e-5, f"gain_seed3 mismatch: {gain_seed3}"

    for k in [1, 2, 3]:
        for i, val in enumerate(p2_raw[k]): p_val("F6", f"P2_raw_K{k}_{i}", val)
        for i, val in enumerate(p2_centred[k]): p_val("F6", f"P2_centred_K{k}_{i}", val)
        for i, val in enumerate(v2_raw[k]): p_val("F6", f"V2_raw_K{k}_{i}", val)
        for i, val in enumerate(v2_centred[k]): p_val("F6", f"V2_centred_K{k}_{i}", val)
        for i, val in enumerate(c8b_raw[k]): p_val("F6", f"C8b_raw_K{k}_{i}", val)
        for i, val in enumerate(c8b_centred[k]): p_val("F6", f"C8b_centred_K{k}_{i}", val)

    p_val("F6", "gain_mc", gain_mc)
    p_val("F6", "gain_snapshot", gain_snapshot)
    p_val("F6", "gain_seed3", gain_seed3)
        
    fig = plt.figure(figsize=(7.2, 3.4))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], hspace=0.8, wspace=0.42)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    setup_axes(ax1)
    setup_axes(ax2)
    setup_axes(ax3)
    
    def plot_ensemble(ax, raw_dict, color, label):
        ks = [1, 2, 3]
        means = []
        for k in ks:
            vals = raw_dict[k]
            ax.scatter([k]*len(vals), vals, color=color, zorder=3)
            means.append(np.mean(vals))
        ax.plot(ks, means, color=color, label=label, zorder=2)
        
    plot_ensemble(ax1, p2_raw, COLOR_V9, "P2 (one 12-epoch model per seed)")
    plot_ensemble(ax1, v2_raw, COLOR_COMP, "V2 (each seed = mean of 3 snapshots)")
    
    plot_ensemble(ax2, p2_centred, COLOR_V9, "P2 (one 12-epoch model per seed)")
    plot_ensemble(ax2, v2_centred, COLOR_COMP, "V2 (each seed = mean of 3 snapshots)")
    
    ax1.set_xticks([1, 2, 3])
    ax1.set_xlabel("seeds averaged (K)", fontsize=8)
    ax1.set_ylabel("per-row delta Pearson", fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    ax2.set_xticks([1, 2, 3])
    ax2.set_xlabel("seeds averaged (K)", fontsize=8)
    ax2.set_ylabel("cell-centred per-row delta Pearson", fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=7)

    # Panel C
    c_labels = [
        "MC dropout + stoch. depth × 8 (1 run)",
        "snapshots × 3 (1 run)",
        "seeds × 3 (3 runs)"
    ]
    c_vals = [gain_mc, gain_snapshot, gain_seed3]
    y_pos = np.arange(len(c_labels))[::-1]
    
    ax3.plot(c_vals, y_pos, 'o', color=COLOR_V9, zorder=3)
    ax3.axvline(0, color=COLOR_TEXT, linestyle='-', linewidth=1.5, zorder=2)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(c_labels, fontsize=8)
    for i, val in enumerate(c_vals):
        ax3.annotate(f"{val:+.4f}", (val, y_pos[i]), xytext=(6, 0), textcoords='offset points', va='center', ha='left',
                     color=COLOR_SEC_TEXT, fontsize=8)
    ax3.set_xlim(0, max(c_vals) * 1.22)
    ax3.set_ylim(-0.6, len(c_labels) - 0.4)
        
    ax3.set_xlabel("Δ per-row delta Pearson over a single model (dev cells)", fontsize=8)
    ax3.tick_params(axis='x', which='major', labelsize=8)
    
    # Note under panel C
    ax3.text(0, -0.62, "MC row: paired vs the same checkpoint's deterministic pass; other rows: vs P2's single-model mean", transform=ax3.transAxes, fontsize=7, color=COLOR_SEC_TEXT, va='top')
    
    plt.savefig(os.path.join(OUT_DIR, "f6_seed_ensemble.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "f6_seed_ensemble.svg"), bbox_inches='tight')
    plt.close()

# F7
def make_f7():
    with open("model/results/v9_dev_align_P2_baseline_aux.json", "r") as f:
        p2_aux = json.load(f)
        
    with open("model/results/v9_dev_align_P2_baseline_incell_aux.json", "r") as f:
        p2_incell = json.load(f)
        
    with open("model/results/v9_dev_align_C6_signhead_aux.json", "r") as f:
        c6_incell = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    setup_axes(ax1)
    setup_axes(ax2)
    
    train_prior = p2_aux["references"]["training_prior"]
    loco_prior = p2_aux["references"]["loco_prior"]
    mean_null_mean = p2_aux["mean_null_mean"]
    
    p_val("F7", "training_prior", train_prior)
    p_val("F7", "loco_prior", loco_prior)
    p_val("F7", "mean_null_mean", mean_null_mean)
    
    ax1.axhline(train_prior, color=COLOR_REF, linestyle='-', label="cell-agnostic training-row prior")
    ax1.axhline(loco_prior, color=COLOR_REF, linestyle='--', label="leave-one-cell-out prior")
    ax1.axhline(mean_null_mean, color=COLOR_GRAY, linestyle=':', label="column-permutation null mean")
    
    seeds = [chk["seed"] for chk in p2_aux["per_checkpoint"]]
    alignments = [chk["alignment"] for chk in p2_aux["per_checkpoint"]]
    shuf_means = [chk["cell_shuffle_mean"] for chk in p2_aux["per_checkpoint"]]
    shuf_sds = [chk["cell_shuffle_sd"] for chk in p2_aux["per_checkpoint"]]
    
    for i, s in enumerate(seeds):
        p_val("F7", f"seed_{s}_alignment", alignments[i])
        p_val("F7", f"seed_{s}_cell_shuffle_mean", shuf_means[i])
        p_val("F7", f"seed_{s}_cell_shuffle_sd", shuf_sds[i])
    
    ax1.scatter(seeds, alignments, color=COLOR_V9, zorder=3, label="P2 per seed")
    ax1.errorbar(seeds, shuf_means, yerr=shuf_sds, fmt='o', color=COLOR_THIRD, capsize=2, zorder=3, label="another cell's readout")
    
    ax1.set_xticks(seeds)
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Alignment")
    ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)  # off the null line (PI)
    
    p2_means = p2_incell["in_cell"]["per_cell_3seed_mean"]
    c6_means = c6_incell["in_cell"]["per_cell_3seed_mean"]
    
    cells = sorted(p2_means.keys())
    x = np.arange(len(cells))
    width = 0.35
    
    p2_vals = [p2_means[c] for c in cells]
    c6_vals = [c6_means[c] for c in cells]
    
    for c in cells:
        p_val("F7", f"P2_incell_{c}", p2_means[c])
        p_val("F7", f"C6_incell_{c}", c6_means[c])
        
    ax2.bar(x - width/2, p2_vals, width, label='P2', color=COLOR_V9, zorder=3)
    ax2.bar(x + width/2, c6_vals, width, label='C6', color=COLOR_COMP, zorder=3)
    
    ax2.axhline(0, color=COLOR_TEXT, linestyle='-', zorder=2)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(cells, rotation=45, ha='right')
    ax2.set_ylabel("own readout − other cells' mean readout (ρ)")
    
    lic_p2 = p2_incell["in_cell"]["licensed"]
    cp_p2 = p2_incell["in_cell"]["cells_positive"]
    
    p_val("F7", "P2_licensed", lic_p2)
    p_val("F7", "P2_cells_positive", cp_p2)
    
    lic_c6 = c6_incell["in_cell"]["licensed"]
    cp_c6 = c6_incell["in_cell"]["cells_positive"]
    p_val("F7", "C6_licensed", lic_c6)
    p_val("F7", "C6_cells_positive", cp_c6)
    ax2.text(0.02, 0.98, f"in this cell: P2 {'licensed' if lic_p2 else 'not licensed'} ({cp_p2} / 6 cells), "
             f"C6 {'licensed' if lic_c6 else 'not licensed'} ({cp_c6} / 6)",
             transform=ax2.transAxes, va='top', ha='left', fontsize=9)
    
    ax2.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f7_pathway_alignment.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f7_pathway_alignment.svg"))
    plt.close()

# F8
def make_f8(src=P7_JSON, ref=O2_JSON, out_dir=OUT_DIR, stem="f8_p7_h2h"):
    if not os.path.exists(src):
        p_val("F8", "skipped", src)
        return

    with open(src, "r") as f:
        src_data = json.load(f)
    with open(ref, "r") as f:
        ref_data = json.load(f)

    src_cells = {c["cell"]: c for c in src_data["per_cell"]}
    ref_cells = {c["cell"]: c for c in ref_data["per_cell"]}
    assert all(c["cell"] in ref_cells for c in src_data["per_cell"])
    assert src_data["split"] == ref_data["split"]

    cells = sorted(src_data["per_cell"], key=lambda x: x["n_scored"])

    # PI polish: stacked (side by side was 11 in wide, set by panel B's long row labels), column width 7.2 in
    import textwrap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), gridspec_kw={'height_ratios': [3, 1.25]})
    setup_axes(ax1)
    setup_axes(ax2)

    # PANEL A
    y_pos = np.arange(len(cells))
    labels = [f"{c['cell']} (n={c['n_scored']})" for c in cells]

    d_c_medians = [c["d_c_median"] for c in cells]
    d_c_err_low = [c["d_c_median"] - c["d_c_median_ci95"][0] for c in cells]
    d_c_err_high = [c["d_c_median_ci95"][1] - c["d_c_median"] for c in cells]

    ax1.errorbar(d_c_medians, y_pos, xerr=[d_c_err_low, d_c_err_high], fmt='o', color=COLOR_V9, capsize=2, zorder=3, label="v9 (3 seeds, snapshot average)")

    ref_d_c_medians = [ref_cells[c["cell"]]["d_c_median"] for c in cells]
    ax1.plot(ref_d_c_medians, y_pos, 'o', color='none', markeredgecolor=COLOR_GRAY, markeredgewidth=1.5, zorder=4, label="§87: one v9 run")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.axvline(0, color=COLOR_REF, linestyle='-', zorder=1)

    cluster_mean = src_data["cluster"]["mean_of_d_c"]
    cluster_ci = src_data["cluster"]["cluster_ci95"]

    ax1.axvspan(cluster_ci[0], cluster_ci[1], alpha=0.2, color=COLOR_THIRD, zorder=1, label="cluster mean 95 % CI")
    ax1.axvline(cluster_mean, color=COLOR_V9, linestyle=':', zorder=2, label="cluster mean")
    ax1.set_xlabel("v9 − XPert: per-row Δ Pearson, median per cell", fontsize=9)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax1.tick_params(axis='x', labelsize=8)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)

    for i, c in enumerate(cells):
        p_val("F8", f"cell_{c['cell']}_d_c_median", d_c_medians[i])
        p_val("F8", f"cell_{c['cell']}_ci95_0", c["d_c_median_ci95"][0])
        p_val("F8", f"cell_{c['cell']}_ci95_1", c["d_c_median_ci95"][1])
        p_val("F8", f"cell_{c['cell']}_ref_d_c_median", ref_d_c_medians[i])

    p_val("F8", "cluster_mean", cluster_mean)
    p_val("F8", "cluster_ci95_0", cluster_ci[0])
    p_val("F8", "cluster_ci95_1", cluster_ci[1])
    p_val("F8", "cells_favouring_ours", src_data["cluster"]["cells_favouring_ours"])
    p_val("F8", "n_cells", src_data["cluster"]["n_cells"])

    # PANEL B
    has_alt = "alt" in src_data
    y_b = [2, 1, 0] if has_alt else [2, 0]
    
    b_means = []
    b_ci_low = []
    b_ci_high = []
    b_labels = []
    b_colors = []
    b_fmts = []
    b_mfc = []
    b_mec = []
    b_texts = []

    # Row 1 (y=2)
    b_means.append(cluster_mean)
    b_ci_low.append(cluster_mean - cluster_ci[0])
    b_ci_high.append(cluster_ci[1] - cluster_mean)
    b_labels.append("v9, snapshot average (3 seeds)")
    b_colors.append(COLOR_V9)
    b_fmts.append('o')
    b_mfc.append(COLOR_V9)
    b_mec.append(COLOR_V9)
    b_texts.append(f"{src_data['cluster']['cells_favouring_ours']} / {src_data['cluster']['n_cells']} cells")

    if has_alt:
        alt_mean = src_data["alt"]["cluster"]["mean_of_d_c"]
        alt_ci = src_data["alt"]["cluster"]["cluster_ci95"]
        b_means.append(alt_mean)
        b_ci_low.append(alt_mean - alt_ci[0])
        b_ci_high.append(alt_ci[1] - alt_mean)
        b_labels.append(src_data["alt"]["label"])
        b_colors.append(COLOR_GRAY)
        b_fmts.append('o')
        b_mfc.append(COLOR_GRAY)
        b_mec.append(COLOR_GRAY)
        b_texts.append(f"{src_data['alt']['cluster']['cells_favouring_ours']} / {src_data['alt']['cluster']['n_cells']} cells")

        p_val("F8", "alt_cluster_mean", alt_mean)
        p_val("F8", "alt_cluster_ci95_0", alt_ci[0])
        p_val("F8", "alt_cluster_ci95_1", alt_ci[1])
        p_val("F8", "alt_cells_favouring_ours", src_data["alt"]["cluster"]["cells_favouring_ours"])
        p_val("F8", "alt_n_cells", src_data["alt"]["cluster"]["n_cells"])

    ref_mean = ref_data["cluster"]["mean_of_d_c"]
    ref_ci = ref_data["cluster"]["cluster_ci95"]
    b_means.append(ref_mean)
    b_ci_low.append(ref_mean - ref_ci[0])
    b_ci_high.append(ref_ci[1] - ref_mean)
    b_labels.append("§87: one v9 run")
    b_colors.append(COLOR_GRAY)
    b_fmts.append('o')
    b_mfc.append('none')
    b_mec.append(COLOR_GRAY)
    b_texts.append(f"{ref_data['cluster']['cells_favouring_ours']} / {ref_data['cluster']['n_cells']} cells")

    p_val("F8", "ref_cluster_mean", ref_mean)
    p_val("F8", "ref_cluster_ci95_0", ref_ci[0])
    p_val("F8", "ref_cluster_ci95_1", ref_ci[1])
    p_val("F8", "ref_cells_favouring_ours", ref_data["cluster"]["cells_favouring_ours"])
    p_val("F8", "ref_n_cells", ref_data["cluster"]["n_cells"])

    for i in range(len(b_means)):
        ax2.errorbar(b_means[i], y_b[i], xerr=[[b_ci_low[i]], [b_ci_high[i]]], fmt=b_fmts[i], color=b_colors[i], mfc=b_mfc[i], mec=b_mec[i], capsize=2, zorder=3)
        ax2.annotate(b_texts[i], (b_means[i] + b_ci_high[i], y_b[i]), xytext=(6, 0), textcoords='offset points', va='center', ha='left', color=COLOR_SEC_TEXT, fontsize=8)

    ax2.set_yticks(y_b)
    ax2.set_yticklabels([textwrap.fill(l, 30) for l in b_labels], fontsize=8)
    ax2.axvline(0, color=COLOR_REF, linestyle='-', zorder=1)
    ax2.set_xlabel("v9 − XPert: cluster mean of per-cell medians (95 % CI)", fontsize=9)
    ax2.tick_params(axis='x', labelsize=8)
    
    # ensure space for text
    ax2.set_xlim(right=ax2.get_xlim()[1] + 0.05)
    _lo = min([0.0] + [m - l for m, l in zip(b_means, b_ci_low)])     # PI: keep the zero line off the spine
    ax2.set_xlim(left=_lo - 0.08 * (ax2.get_xlim()[1] - _lo))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"{stem}.svg"), bbox_inches='tight')
    plt.close()

CHROM_POWER_JSON = "model/results/chromatin_power_91.json"
CHROM_FUNNEL_JSON = "model/results/chromatin_funnel_91.json"
T4_PER_CELL_JSON = "model/results/v9_dev_T4_T4b_per_cell.json"

def make_f9(power=CHROM_POWER_JSON, funnel=CHROM_FUNNEL_JSON, t4=T4_PER_CELL_JSON, out_dir=OUT_DIR, stem="f9_chromatin"):
    if not os.path.exists(power):
        p_val("F9", "skipped", power)
        return
        
    with open(power, "r") as f:
        power_data = json.load(f)
    with open(funnel, "r") as f:
        funnel_data = json.load(f)
    with open(t4, "r") as f:
        t4_data = json.load(f)
        
    assert funnel_data["row_set_of_record"] == "known"
    assert power_data["summary"]["row_set_of_record"] == "known"
    
    fig = plt.figure(figsize=(7.2, 7.6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.95)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    
    setup_axes(ax_a)
    setup_axes(ax_b)
    setup_axes(ax_c)
    
    for ax, letter in zip([ax_a, ax_b, ax_c], ['a', 'b', 'c']):
        ax.text(0.0, 1.05, letter, transform=ax.transAxes, fontweight='bold', va='bottom', ha='left')
        
    # --- PANEL A ---
    pis = power_data["pis"]
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("planted effect (fraction of the cell-specific residual)", fontsize=8)
    ax_a.set_ylabel("T1 Δ (drug-known dev rows)", fontsize=8)
    
    ax_a.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1])
    ax_a.set_xticklabels(["0.5 %", "1 %", "2 %", "5 %", "10 %"], fontsize=8)
    ax_a.tick_params(axis='y', labelsize=8)
    
    p1_meds, p2_meds = [], []
    for pi in pis:
        pi_str = str(pi)
        p1_deltas = []
        p2_deltas = []
        p1_passes = 0
        p2_passes = 0
        
        for draw in range(power_data["draws"]):
            rec_p1 = power_data["records"][draw]["T1"][f"P1_{pi_str}"]["known"]
            rec_p2 = power_data["records"][draw]["T1"][f"P2_{pi_str}"]["known"]
            
            p1_deltas.append(rec_p1["delta_all"])
            p2_deltas.append(rec_p2["delta_all"])
            
            p1_passes += int(rec_p1["pass"])
            p2_passes += int(rec_p2["pass"])
            
            jitter_p1 = 0.94 + 0.03 * draw
            jitter_p2 = 1.06 - 0.03 * draw
            
            if rec_p1["pass"]:
                ax_a.plot(pi * jitter_p1, rec_p1["delta_all"], 'o', color=COLOR_THIRD, zorder=3)
            else:
                ax_a.plot(pi * jitter_p1, rec_p1["delta_all"], 'o', mfc='white', mec=COLOR_THIRD, zorder=3)
                
            if rec_p2["pass"]:
                ax_a.plot(pi * jitter_p2, rec_p2["delta_all"], 's', color=COLOR_GRAY, zorder=3)
            else:
                ax_a.plot(pi * jitter_p2, rec_p2["delta_all"], 's', mfc='white', mec=COLOR_GRAY, zorder=3)
                
        p1_med = np.median(p1_deltas)
        p2_med = np.median(p2_deltas)
        
        p1_meds.append(p1_med)
        p2_meds.append(p2_med)
        
        p_val("F9", f"P1_{pi_str}_median", p1_med)
        p_val("F9", f"P2_{pi_str}_median", p2_med)
        p_val("F9", f"P1_{pi_str}_passes", p1_passes)
        p_val("F9", f"P2_{pi_str}_passes", p2_passes)
        
    ax_a.plot(pis, p1_meds, '-', color=COLOR_THIRD, linewidth=1, zorder=2)
    ax_a.plot(pis, p2_meds, '-', color=COLOR_GRAY, linewidth=1, zorder=2)
    bar_val = funnel_data["row_sets"]["known"]["T1"]["bars"]["t1"]
    assert abs(bar_val - 0.004) < 1e-9
    p_val("F9", "bar", bar_val)
    ax_a.axhline(bar_val, color=COLOR_SEC_TEXT, linestyle='--', zorder=1)
    ax_a.text(0.12, bar_val * 0.92, "bar +0.004", va='top', ha='right', color=COLOR_SEC_TEXT, fontsize=7)
    
    real_delta = funnel_data["row_sets"]["known"]["M4"]["S_FBC_minus_S_FB"]
    p_val("F9", "real_delta", real_delta)
    ax_a.axhline(real_delta, color=COLOR_TEXT, linestyle=':', zorder=1)
    ax_a.text(0.12, real_delta * 0.92, f"real data +{real_delta:.4f}", va='top', ha='right', color=COLOR_TEXT, fontsize=7)
    
    null_deltas = [power_data["records"][i]["null_T1"]["known"]["delta_all"] for i in range(power_data["draws"])]
    null_max_abs = max(abs(d) for d in null_deltas)
    p_val("F9", "null_max_abs", null_max_abs)
    
    ax_a.set_ylim(bottom=1e-4)
    ax_a.set_xlim(0.004, 0.125)
    ax_a.axhspan(1e-4, null_max_abs, color=COLOR_GRID, zorder=0)
    
    leg_a_elements = [
        plt.Line2D([0], [0], marker='o', color='none', mfc=COLOR_THIRD, mec=COLOR_THIRD, label="chromatin gain (P1)"),
        plt.Line2D([0], [0], marker='s', color='none', mfc=COLOR_GRAY, mec=COLOR_GRAY, label="drug-specific shift (P2)"),
        plt.Line2D([0], [0], marker='o', color='none', mfc=COLOR_GRAY, mec=COLOR_GRAY, label="passed T1's rule"),
        plt.Line2D([0], [0], marker='o', color='none', mfc='white', mec=COLOR_GRAY, label="did not pass"),
        Patch(facecolor=COLOR_GRID, edgecolor='none', label=f"nothing planted: |Δ| ≤ {null_max_abs:.5f} (5 draws)")
    ]
    ax_a.legend(handles=leg_a_elements, loc='lower left', bbox_to_anchor=(-0.25, 1.06), ncol=2, frameon=False, fontsize=6.5,
                columnspacing=1.0, handletextpad=0.4)
    
    # --- PANEL B ---
    known_scores = funnel_data["row_sets"]["known"]["scores"]
    b0_all = known_scores["B0"]["all"]
    b_keys = ["FB", "FC", "FBC", "N1", "N2"]
    b_labels = ["basal expression\n(FB)", "chromatin only\n(FC)", "basal + chromatin\n(FBC)",
                "basal + every cell's\nmean chromatin (N1)", "basal + another dev\ncell's chromatin (N2)"]
    
    b_vals = []
    for k in b_keys:
        val = known_scores[k]["all"] - b0_all
        b_vals.append(val)
        p_val("F9", f"B_{k}", val)
        
    assert abs((known_scores["FBC"]["all"] - b0_all) - (known_scores["FB"]["all"] - b0_all) - funnel_data["row_sets"]["known"]["M4"]["S_FBC_minus_S_FB"]) < 1e-9
    assert abs((known_scores["FC"]["all"] - b0_all) - funnel_data["row_sets"]["known"]["M4"]["S_FC_minus_S_B0"]) < 1e-9
    
    y_pos_b = np.arange(len(b_vals))[::-1]
    ax_b.barh(y_pos_b, b_vals, color=COLOR_GRAY, zorder=3)
    ax_b.axvline(0, color=COLOR_TEXT, linestyle='-', zorder=2)
    
    for i, val in enumerate(b_vals):
        ax_b.text(val, y_pos_b[i], f" {val:+.4f}", va='center', ha='left' if val > 0 else 'right', color=COLOR_SEC_TEXT, fontsize=7)
        
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(b_labels, fontsize=7)
    ax_b.set_xlim(0, 0.0052)
    ax_b.set_xlabel("Δ over B0 (drug-known dev rows)", fontsize=8)
    ax_b.tick_params(axis='x', labelsize=8)
    
    # --- PANEL C ---
    v9_marks = funnel_data["T0"]["v9"]["marks_dev"]
    rn_marks = funnel_data["T0"]["rank_normal"]["marks_dev"]
    mark_abbrev = {"ATAC": "ATAC", "H3K27ac": "K27ac", "H3K27me3": "K27me3"}
    
    dev_cells = sorted(t4_data["arms"]["T4"]["per_cell_mean"].keys(), key=lambda c: t4_data["arms"]["T4"]["per_cell_mean"][c], reverse=True)
    
    c_labels = []
    c_failed_marks_list = []
    
    for cell in dev_cells:
        marks = v9_marks[cell]
        failed = [m for m in marks if m not in rn_marks[cell]]
        p_val("F9", f"failed_marks_{cell}", failed)
        c_failed_marks_list.append(failed)
        
        parts = []
        for m in marks:
            part = mark_abbrev.get(m, m)
            if m in failed:
                part += " (failed)"
            parts.append(part)
        tracks_str = " / ".join(parts)
        
        n_c = t4_data["cells"][cell]
        c_labels.append(f"{cell} (n={n_c}) · {tracks_str}")
        
    c_labels.append(f"all dev rows (n={t4_data['n_rows']})")
    y_pos_c = np.arange(len(c_labels))[::-1]
    
    bar_width = 0.35
    
    for i, cell in enumerate(dev_cells):
        y = y_pos_c[i]  # label i sits at y_pos_c[i] (PI fix: W29 drew cell i at y_pos_c[i + 1], one row low)
        
        t4_val = t4_data["arms"]["T4"]["per_cell_mean"][cell]
        ax_c.barh(y + bar_width/2, t4_val, height=bar_width, color=COLOR_V9, zorder=3)
        p_val("F9", f"T4_{cell}", t4_val)
        
        for k, seed_val in enumerate(t4_data["arms"]["T4"]["per_cell_per_seed"][cell]):
            ax_c.plot(seed_val, y + bar_width/2, 'o', color=COLOR_TEXT, markersize=2, zorder=4)
            p_val("F9", f"T4_{cell}_seed{k}", seed_val)
            
        failed = c_failed_marks_list[i]
        if failed:
            t4b_val = t4_data["arms"]["T4b"]["per_cell_mean"][cell]
            ax_c.barh(y - bar_width/2, t4b_val, height=bar_width, color=COLOR_V9, alpha=0.35, edgecolor=COLOR_V9, zorder=3)
            p_val("F9", f"T4b_{cell}", t4b_val)
            
            for k, seed_val in enumerate(t4_data["arms"]["T4b"]["per_cell_per_seed"][cell]):
                ax_c.plot(seed_val, y - bar_width/2, 'o', color=COLOR_TEXT, markersize=2, zorder=4)
        else:
            ax_c.text(0.0005, y - bar_width/2, "no failed track", va='center', ha='left', color=COLOR_SEC_TEXT, fontsize=6.5)
            
    # "all dev rows"
    y_all = y_pos_c[-1]  # 'all dev rows' is the last label
    t4_all = t4_data["arms"]["T4"]["all"]
    ax_c.barh(y_all + bar_width/2, t4_all, height=bar_width, color=COLOR_V9, zorder=3)
    p_val("F9", "T4_all", t4_all)
    for k, seed_val in enumerate(t4_data["arms"]["T4"]["per_seed"]):
        ax_c.plot(seed_val, y_all + bar_width/2, 'o', color=COLOR_TEXT, markersize=2, zorder=4)
        
    t4b_all = t4_data["arms"]["T4b"]["all"]
    ax_c.barh(y_all - bar_width/2, t4b_all, height=bar_width, color=COLOR_V9, alpha=0.35, edgecolor=COLOR_V9, zorder=3)
    p_val("F9", "T4b_all", t4b_all)
    for k, seed_val in enumerate(t4_data["arms"]["T4b"]["per_seed"]):
        ax_c.plot(seed_val, y_all - bar_width/2, 'o', color=COLOR_TEXT, markersize=2, zorder=4)
        
    ax_c.axvline(0, color=COLOR_TEXT, linestyle='-', zorder=2)
    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(c_labels, fontsize=8)
    # PI guard (W29 drew every cell one row below its label): each bar's value must be its own row's cell's value.
    row_label = {round(float(y), 6): lab for y, lab in zip(y_pos_c, c_labels)}
    n_checked = 0
    for bar in ax_c.patches:
        is_t4b = bar.get_alpha() is not None
        y_row = round(bar.get_y() + bar.get_height() / 2 + (bar_width / 2 if is_t4b else -bar_width / 2), 6)
        lab = row_label[y_row]
        arm = t4_data["arms"]["T4b" if is_t4b else "T4"]
        want = arm["all"] if lab.startswith("all dev rows") else arm["per_cell_mean"][lab.split(" (")[0]]
        assert abs(bar.get_width() - want) < 1e-12, (lab, bar.get_width(), want)
        n_checked += 1
    assert n_checked == len(c_labels) + 1 + sum(bool(f) for f in c_failed_marks_list), n_checked
    p_val("F9", "C_bars_checked_against_row_labels", n_checked)
    ax_c.set_xlabel("ablated − intact: per-row Δ Pearson (mean over rows; 3 seeds averaged)", fontsize=8)
    ax_c.tick_params(axis='x', labelsize=8)
    
    leg_c_elements = [
        Patch(facecolor=COLOR_V9, edgecolor='none', label="all chromatin ablated (T4)"),
        Patch(facecolor=COLOR_V9, alpha=0.35, edgecolor=COLOR_V9, label="failed H3K27me3 only (T4b)")
    ]
    ax_c.legend(handles=leg_c_elements, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=8)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"{stem}.svg"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    make_f1()
    make_f2()
    make_f3()
    make_f4()
    make_f5()
    make_f6()
    make_f7()
    make_f8()
    make_f9()

