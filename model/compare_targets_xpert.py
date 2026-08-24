"""
Do XPert's Level-3 delta and our Level-5 MODZ z-score measure the SAME thing?

129,236 (cell, pert, dose, time) conditions exist in both datasets. For matched conditions this compares
the two TARGETS directly -- no model involved. If they agree, cross-paper score comparison is meaningful
once the convention is matched. If they do not, the two literatures are predicting different quantities
and no amount of protocol-matching makes the numbers comparable.
"""
import h5py, numpy as np, csv, sys
from scipy.stats import pearsonr

XP = 'l1000_mdmt_full_336852.h5ad'
f = h5py.File(XP, 'r')

def cat(col):
    g = f['obs'][col]
    if isinstance(g, h5py.Group):
        c = g['categories'][:]; k = g['codes'][:]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in c])[k]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in g[:]])

def dec(a): return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

# ---- gene alignment by SYMBOL (959 shared; the rest are symbol aliases) ----
xp_genes = dec(f['var']['gene_symbol'][:])
our_genes = [l.strip() for l in open(r'C:\Projects\LINCS\Data Info\pathway_landmark_genes.txt') if l.strip()]
our_pos = {g: i for i, g in enumerate(our_genes)}
pairs = [(i, our_pos[g]) for i, g in enumerate(xp_genes) if g in our_pos]
xp_gi = np.array([p[0] for p in pairs]); our_gi = np.array([p[1] for p in pairs])
print(f"aligned on {len(pairs)} shared gene symbols")

# ---- condition matching ----
def norm_dose(s):
    s = str(s).lower().replace('\u00b5', 'u').replace('\u03bc', 'u')
    nm = 'nm' in s
    s = s.replace('um', '').replace('nm', '').strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return round(v / 1000.0, 4) if nm else round(v, 4)

def norm_time(s):
    try: return float(str(s).lower().replace('h', '').strip())
    except ValueError: return None

xp_key = list(zip(cat('cell_iname'), cat('pert_id'),
                  [norm_dose(d) for d in cat('pert_idose')],
                  [norm_time(t) for t in cat('pert_itime')]))
xp_index = {}
for i, k in enumerate(xp_key):
    xp_index.setdefault(k, i)

matched = []   # (xpert_row, our_Y_row)
with open(r'C:\Projects\LINCS\phase2_assembly\outputs\signatures_usable.tsv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh, delimiter='\t'):
        k = (r['cell_id'], r['pert_id'], norm_dose(r['dose']), norm_time(r['time']))
        j = xp_index.get(k)
        if j is not None:
            matched.append((j, int(r['row'])))
print(f"matched conditions: {len(matched):,}")

rng = np.random.default_rng(0)
N = min(4000, len(matched))
sel = rng.choice(len(matched), N, replace=False)
sub = sorted(matched[i] for i in sel)
# our signature file has several rows per condition, so the same XPert row can match more than once;
# h5py needs strictly increasing indices, so read each XPert row ONCE and map back
xr_all = np.array([s[0] for s in sub]); yr = np.array([s[1] for s in sub])
xr, inv = np.unique(xr_all, return_inverse=True)

# ---- load the two targets for the same conditions ----
Xt = f['X'][xr, :][:, xp_gi]
Xc = f['obsm']['X_ctl'][xr, :][:, xp_gi]
xp_delta = (Xt - Xc)[inv]                                   # Level-3 log-expression delta
Y = np.load(r'C:\Projects\LINCS\phase2_assembly\outputs\Y_target_level5_978.npy', mmap_mode='r')
our_Y = np.asarray(Y[yr, :], np.float64)[:, our_gi]  # Level-5 MODZ z-score
print(f"loaded {xp_delta.shape} matched pairs\n")

r = np.array([pearsonr(a, b)[0] for a, b in zip(xp_delta, our_Y)])
strength = np.abs(our_Y).mean(1)                     # our reproducibility proxy
xp_strength = np.abs(xp_delta).mean(1)

print("=" * 74)
print("AGREEMENT BETWEEN THE TWO TARGETS, same biological conditions")
print("=" * 74)
print(f"  per-condition Pearson(XPert Level-3 delta, our Level-5 z-score)")
print(f"    mean {r.mean():.4f}   median {np.median(r):.4f}   sd {r.std():.4f}")
print(f"    fraction with r > 0.5 : {(r > 0.5).mean():.3f}")
print(f"    fraction with r < 0.2 : {(r < 0.2).mean():.3f}")
print()
q = np.quantile(strength, [0, .25, .5, .75, 1.0])
print("  stratified by OUR signature strength (mean|Y|):")
for i in range(4):
    m = (strength >= q[i]) & (strength <= q[i + 1])
    print(f"    mean|Y| {q[i]:.2f}-{q[i+1]:.2f}  n={m.sum():<5} agreement r = {r[m].mean():.4f}")
rep = strength >= 1.0
print(f"\n  REPRODUCIBLE stratum (our mean|Y|>=1, n={rep.sum()}): agreement r = {r[rep].mean():.4f}")
print()
print(f"  magnitude: their mean|delta| median {np.median(xp_strength):.3f} | our mean|Y| median {np.median(strength):.3f}")
print(f"  corr(their strength, our strength) = {pearsonr(xp_strength, strength)[0]:+.4f}")

# ---- is the disagreement explained by OUR target's measured noise? ----
# Our replicate reliability rho was measured per strength bin [6.1]:
#   mean|Y| 0.48/0.65/0.80/1.11/1.63/2.50 -> rho .074/.088/.136/.365/.646/.751
# If their target were relatively clean and ours = signal + noise, then
#   corr(theirs, ours) <= sqrt(rho_ours)
RHO_X = [0.48, 0.65, 0.80, 1.11, 1.63, 2.50]
RHO_Y = [0.074, 0.088, 0.136, 0.365, 0.646, 0.751]
print()
print("=" * 74)
print("IS THE DISAGREEMENT JUST OUR NOISE?  observed vs our reliability ceiling")
print("=" * 74)
print(f"  {'our mean|Y| bin':>18} {'n':>6} {'observed r':>11} {'rho':>7} {'ceiling':>9} {'% of ceiling':>13}")
for lo, hi in [(0.5,0.8),(0.8,1.0),(1.0,1.5),(1.5,2.5),(2.5,99)]:
    m = (strength >= lo) & (strength < hi)
    if m.sum() < 30: continue
    mid = np.clip(np.median(strength[m]), RHO_X[0], RHO_X[-1])
    rho = np.interp(mid, RHO_X, RHO_Y)
    ceil = np.sqrt(rho)
    obs = r[m].mean()
    print(f"  {lo:>8.1f}-{hi:<7.1f} {m.sum():>6} {obs:>11.4f} {rho:>7.3f} {ceil:>9.4f} {100*obs/ceil:>12.0f}%")
print()
print("  If 'observed' sits near the ceiling, the two targets measure the SAME biology and the")
print("  gap is OUR measurement noise -- i.e. the target, not the model, is the limiting factor.")
