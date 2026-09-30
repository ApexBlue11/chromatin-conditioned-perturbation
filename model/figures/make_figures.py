import json
import os
import glob
import numpy as np
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
        ("C8b_postpath", "C8b post-drug pathway layer (3 seeds)", "not accepted", "v9_dev_score_C8b_postpath.json")
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
    for i, (key, label, decision, fname) in enumerate(rows):
        labels.append(label)
        with open(os.path.join("model/results", fname), "r") as f:
            data = json.load(f)
        
        vb = data["vs_baseline"]
        delta = vb["delta_per_row_mean"]
        sd = data.get("sd", 0.0)
        
        color = COLOR_V9 if decision == "accepted" else COLOR_GRAY
        y = y_pos[i]
        
        if vb.get("n_seeds_variant", 1) == 3:
            err = 2 * np.sqrt((s0**2)/3 + (sd**2)/3)
            ax1.errorbar(delta, y, xerr=err, fmt='o', color=color, capsize=2, zorder=3)
            p_val("F3", f"{key}_err", err)
        else:
            ax1.plot(delta, y, 'o', color=color, zorder=3)
            
        p_val("F3", f"{key}_delta_per_row_mean", delta)
        
        ax1.text(ax1.get_xlim()[1], y, f"  {decision}", va='center', ha='left', color=color, fontsize=9)
        
        if "delta_centred" in vb:
            dc = vb["delta_centred"]
            ax2.plot(dc, y, 'o', color=color, zorder=3)
            p_val("F3", f"{key}_delta_centred", dc)
        else:
            ax2.text(0, y, "n/a", va='center', ha='center', color=COLOR_GRAY, fontsize=9)
            
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Δ vs baseline, per-row delta Pearson (dev cells)")
    ax2.set_xlabel("Δ vs baseline, per-row delta Pearson (dev cells)\ncell-centred")
    
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
    c8b_raw, c8b_centred = evaluate_ensemble(c8b_files)
    
    assert abs(p2_raw[3][0] - 0.4662) <= 5e-5, f"P2 K=3 raw mismatch: {p2_raw[3][0]}"
    assert abs(p2_centred[3][0] - 0.5018) <= 5e-5, f"P2 K=3 centred mismatch: {p2_centred[3][0]}"
    
    for k in [1, 2, 3]:
        for i, val in enumerate(p2_raw[k]): p_val("F6", f"P2_raw_K{k}_{i}", val)
        for i, val in enumerate(p2_centred[k]): p_val("F6", f"P2_centred_K{k}_{i}", val)
        for i, val in enumerate(c8b_raw[k]): p_val("F6", f"C8b_raw_K{k}_{i}", val)
        for i, val in enumerate(c8b_centred[k]): p_val("F6", f"C8b_centred_K{k}_{i}", val)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    setup_axes(ax1)
    setup_axes(ax2)
    
    def plot_ensemble(ax, raw_dict, color, label):
        ks = [1, 2, 3]
        means = []
        for k in ks:
            vals = raw_dict[k]
            ax.scatter([k]*len(vals), vals, color=color, zorder=3)
            means.append(np.mean(vals))
        ax.plot(ks, means, color=color, label=label, zorder=2)
        
    plot_ensemble(ax1, p2_raw, COLOR_V9, "P2 baseline")
    plot_ensemble(ax1, c8b_raw, COLOR_COMP, "C8b")
    
    plot_ensemble(ax2, p2_centred, COLOR_V9, "P2 baseline")
    plot_ensemble(ax2, c8b_centred, COLOR_COMP, "C8b")
    
    ax1.set_xticks([1, 2, 3])
    ax1.set_xlabel("Ensemble size (K)")
    ax1.set_ylabel("Mean per-row delta Pearson")
    ax1.legend(frameon=False)
    
    ax2.set_xticks([1, 2, 3])
    ax2.set_xlabel("Ensemble size (K)")
    ax2.set_ylabel("Cell-centred mean per-row delta Pearson")
    ax2.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f6_seed_ensemble.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "f6_seed_ensemble.svg"))
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

if __name__ == "__main__":
    make_f1()
    make_f2()
    make_f3()
    make_f4()
    make_f5()
    make_f6()
    make_f7()
