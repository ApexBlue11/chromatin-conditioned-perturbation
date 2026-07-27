# -*- coding: utf-8 -*-
"""
Manuscript figures, generated from the SAVED result artifacts (no re-computation, no re-training).

Palette: Okabe-Ito 3-colour core (#0072B2, #D55E00, #009E73) -- validated colourblind-safe
(worst adjacent CVD deltaE 11.0 deutan / 25.8 normal; all contrasts >= 3:1). Every bar is also
directly labelled, so identity never rests on colour alone. No dual axes anywhere.

Sources: finalized_analyses.json (metric sweep, corrected pathway ablation), pathway_maps.json
(conductance variance split); the remaining constants are the documented values in RESULTS.md /
CLAIMS.md, each annotated with its section.

Run: python model/results/make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
})


def _clean(ax, ygrid_only=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid_only:
        ax.xaxis.grid(False)


def fig1_dti():
    """DTI enrichment rises monotonically with reference confidence (RESULTS.md 3)."""
    tiers = ["All edges\n(19,174; 98% STITCH)", "ChEMBL curated\n(447)", "Both-source gold\n(156)"]
    atom = [0.9, 2.1, 2.6]      # ca_gene_norm recall@5, fold vs random
    yhat = [1.5, 1.5, 2.1]      # |Yhat| recall@10, fold vs random
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    b1 = ax.bar(x - w/2, atom, w, color=BLUE, label="atom→gene attention (recall@5)")
    b2 = ax.bar(x + w/2, yhat, w, color=VERM, label="predicted |Ŷ| (recall@10)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                    f"{b.get_height():.1f}×", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.text(2.48, 1.03, "random", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylabel("enrichment for known targets\n(fold vs random ranking)")
    ax.set_title("Attribution localises to drug targets — and strengthens with reference quality")
    ax.set_ylim(0, 3.1); ax.legend(loc="upper left"); _clean(ax)
    fig.savefig(os.path.join(FIG, "fig1_dti_enrichment.png")); plt.close(fig)


def fig2_epi_familiarity():
    """Epigenetics contribution tracks CELL familiarity (CLAIMS 2.3)."""
    labels = ["In-distribution\n(cell seen)", "Unseen compound\n(cell seen)", "Unseen cell\n(cell NEW)"]
    vals = [0.089, 0.035, -0.004]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    cols = [GREEN, GREEN, VERM]
    bars = ax.bar(labels, vals, 0.55, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (0.003 if v >= 0 else -0.008),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_ylabel("epigenetics contribution (ΔR²)")
    ax.set_title("Chromatin helps only where the cell was seen in training")
    ax.set_ylim(-0.02, 0.105); _clean(ax)
    fig.savefig(os.path.join(FIG, "fig2_epi_cell_familiarity.png")); plt.close(fig)


def fig3_metric_convention(fin):
    """The same predictions score 0.44 -> 1.0 purely by metric convention (RESULTS.md 17)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for key, col, lab in [("unseen CELL", BLUE, "unseen cell"),
                          ("unseen COMPOUND", VERM, "unseen compound")]:
        pts = fin["metric_sweep"][key]
        xs = [max(p["basal_delta_var_ratio"], 0.02) for p in pts]
        ys = [p["pcc"] for p in pts]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=5, label=lab)
    ax.set_xscale("log")
    for y, lab in [(0.743, "published SOTA, unseen cell (0.743)"),
                   (0.870, "published SOTA, unseen compound (0.870)")]:
        ax.axhline(y, color=MUTED, lw=1.0, ls=":")
        ax.text(0.022, y + 0.012, lab, fontsize=7.5, color=MUTED)
    ax.annotate("our reported\ndifferential metric", xy=(0.02, 0.455), xytext=(0.045, 0.30),
                fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.set_xlabel("basal : drug-effect variance ratio of the evaluation target  (log scale)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Identical predictions, scored under different metric conventions")
    ax.set_ylim(0.3, 1.03); ax.legend(loc="lower right"); _clean(ax, ygrid_only=False)
    fig.savefig(os.path.join(FIG, "fig3_metric_convention.png")); plt.close(fig)


def fig4_variance_partition():
    """Model under-expresses the drug x cell interaction (RESULTS.md 2)."""
    comps = ["drug × gene", "cell × gene", "drug × cell\ninteraction"]
    truth = [42.3, 9.8, 47.9]; model = [59.1, 14.4, 26.5]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    b1 = ax.bar(x - w/2, truth, w, color=BLUE, label="measured truth")
    b2 = ax.bar(x + w/2, model, w, color=VERM, label="model predictions")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(comps)
    ax.set_ylabel("share of reproducible response variance (%)")
    ax.set_title("Cell-specificity is under-expressed — but that is near-optimal under noise")
    ax.set_ylim(0, 66); ax.legend(); _clean(ax)
    fig.savefig(os.path.join(FIG, "fig4_variance_partition.png")); plt.close(fig)


def fig5_pathway_ablation(fin):
    """Ablating a multiplicative gate to 1 breaks its learned SCALE (CLAIMS 3.1/3.1a/3.6)."""
    labels = ["unseen cell", "unseen compound"]
    naive = [fin["unseen CELL"]["c_to_1"]["delta_r2"], fin["unseen COMPOUND"]["c_to_1"]["delta_r2"]]
    ctrl = [fin["unseen CELL"]["c_to_mean"]["delta_r2"], fin["unseen COMPOUND"]["c_to_mean"]["delta_r2"]]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    b1 = ax.bar(x - w/2, naive, w, color=VERM, label="ablate to 1  (also destroys learned scale)")
    b2 = ax.bar(x + w/2, ctrl, w, color=BLUE, label="ablate to mean  (scale preserved)")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + (0.003 if h >= 0 else -0.004),
                    f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("apparent contribution (ΔR²)")
    ax.set_title("A 30× artefact: the same component, two ablation designs")
    ax.legend(loc="upper right"); ax.set_ylim(-0.02, 0.14); _clean(ax)
    fig.savefig(os.path.join(FIG, "fig5_pathway_ablation_artefact.png")); plt.close(fig)


def fig6_interaction_shrinkage():
    """Interaction expression rises with signature reliability => MSE shrinkage (RESULTS.md 3b)."""
    centres = [0.4, 0.65, 0.95, 1.35, 2.0]
    std_ratio = [0.158, 0.196, 0.312, 0.476, 0.492]
    corr = [0.006, 0.028, 0.177, 0.325, 0.415]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(centres, std_ratio, "-o", color=BLUE, lw=2, ms=6, label="expressed magnitude  std(pred)/std(true)")
    ax.plot(centres, corr, "-s", color=GREEN, lw=2, ms=6, label="pattern accuracy  corr(pred, true)")
    ax.set_xlabel("signature strength  (mean |Y|)  →  more reproducible")
    ax.set_ylabel("drug × cell interaction")
    ax.set_title("The model expresses cell-specificity only where the signal is real")
    ax.set_ylim(0, 0.58); ax.legend(loc="upper left"); _clean(ax, ygrid_only=False)
    fig.savefig(os.path.join(FIG, "fig6_interaction_vs_reliability.png")); plt.close(fig)


def main():
    fin = json.load(open(os.path.join(HERE, "finalized_analyses.json")))
    fig1_dti(); fig2_epi_familiarity(); fig3_metric_convention(fin)
    fig4_variance_partition(); fig5_pathway_ablation(fin); fig6_interaction_shrinkage()
    for f in sorted(os.listdir(FIG)):
        print("wrote figures/" + f)


if __name__ == "__main__":
    main()
