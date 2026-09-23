## Diff
```diff
diff --git a/model/v9/alpha_sweep.py b/model/v9/alpha_sweep.py
index bdb8a32..f482d34 100644
--- a/model/v9/alpha_sweep.py
+++ b/model/v9/alpha_sweep.py
@@ -45,7 +45,7 @@ def check_diagonal_available(model):
         )
     return True, ""
 
-def evaluate_chunks_alpha(model, chunks, alpha, key='atoms', operator='full'):
+def evaluate_chunks_alpha(model, chunks, alpha, key='atoms', operator='full', cut='none'):
     r_full_list, r_ablated_list = [], []
     dymax_atom = 0.0
 
@@ -65,6 +65,13 @@ def evaluate_chunks_alpha(model, chunks, alpha, key='atoms', operator='full'):
             else:
                 c_alpha['drug_alpha'] = alpha
                 c_ablated_alpha['drug_alpha'] = alpha
+                
+            if cut in ('xattn', 'both'):
+                c_alpha['drug_xattn_global_only'] = True
+                c_ablated_alpha['drug_xattn_global_only'] = True
+            if cut in ('global', 'both'):
+                c_alpha['drug_global_self_only'] = True
+                c_ablated_alpha['drug_global_self_only'] = True
 
             out_full = model(c_alpha)
             y_full = out_full['delta'] if isinstance(out_full, dict) else out_full
@@ -95,8 +102,12 @@ def main():
     ap.add_argument('--seed', type=int, default=0)
     ap.add_argument('--key', default='atoms')
     ap.add_argument('--operator', choices=['full', 'atom_only'], default='full')
+    ap.add_argument('--cut', choices=['none', 'xattn', 'global', 'both'], default='none')
     a = ap.parse_args()
 
+    if a.cut != 'none' and a.operator != 'atom_only':
+        ap.error("--cut requires --operator atom_only")
+
     alphas = [float(x.strip()) for x in a.alphas.split(',')]
 
     dev = 'cuda' if torch.cuda.is_available() else 'cpu'
@@ -116,6 +127,8 @@ def main():
     key_tag = '' if a.key == 'atoms' else f'_key-{a.key}'
     if a.operator != 'full':
         key_tag += f'_op-{a.operator}'
+    if a.cut != 'none':
+        key_tag += f'_cut-{a.cut}'
     if a.n_eval != 1500:
         key_tag += f'_n{a.n_eval}'
     dst_json = os.path.join(ROOT, 'model', 'results', f'v9_alpha_sweep_{ckpt_stem}{key_tag}.json')
@@ -175,6 +188,8 @@ def main():
         'operator': a.operator,
         'splits': {}
     }
+    if a.cut != 'none':
+        out['cut'] = a.cut
 
     # Two generators, deliberately. The delegated version drew BOTH the row sample and every alpha's
     # bootstrap from one generator, with the bootstrap inside the alpha loop. Two consequences, and the
@@ -231,7 +246,7 @@ def main():
         boot_idx_split = None
 
         for alpha in alphas:
-            r_full, r_ablated, dymax_atom = evaluate_chunks_alpha(model, chunks, alpha, key=a.key, operator=a.operator)
+            r_full, r_ablated, dymax_atom = evaluate_chunks_alpha(model, chunks, alpha, key=a.key, operator=a.operator, cut=a.cut)
             row_dumps[f'{name}__a{alpha}'] = (r_full, r_ablated)
             ok = np.isfinite(r_full) & np.isfinite(r_ablated)
             r_f, r_a = r_full[ok], r_ablated[ok]
diff --git a/model/v9/model_v9.py b/model/v9/model_v9.py
index 53e9379..6beee61 100644
--- a/model/v9/model_v9.py
+++ b/model/v9/model_v9.py
@@ -54,14 +54,23 @@ class PerturbBlock(nn.Module):
         # crossEncoder does. Default off; `--drug_self_attn` turns it on for the A/B.
         self.drug_sa = _DrugBlock(cfg, p_drop) if getattr(cfg, 'drug_self_attn', False) else None
 
-    def forward(self, h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0, drug_atom_alpha=1.0):
+    def forward(self, h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0, drug_atom_alpha=1.0, drug_global_self_only=False, xattn_global_only=False):
+        if drug_global_self_only and self.drug_sa is None:
+            raise ValueError("drug_global_self_only=True requested but self.drug_sa is None")
+            
         # The updated D is RETURNED, so contextualisation compounds across blocks exactly as it does in
         # XPert, where crossEncoder.forward emits drug_SA_embed alongside the cell output.
         # TASK W4: diagonal=True masks drug_sa to the identity for the zero-cross-atom ablation arm.
         diag = diagonal or drug_diagonal
         if self.drug_sa is not None:
-            D = self.drug_sa(D, key_mask=key_mask, diagonal=diag, alpha=drug_alpha, atom_alpha=drug_atom_alpha)
-        a = self.cross(self.nc(h), D, key_mask)
+            D = self.drug_sa(D, key_mask=key_mask, diagonal=diag, alpha=drug_alpha, atom_alpha=drug_atom_alpha, global_self_only=drug_global_self_only)
+            
+        xmask = key_mask
+        if xattn_global_only:
+            xmask = key_mask.clone()
+            xmask[:, 1:] = True
+        a = self.cross(self.nc(h), D, xmask)
+        
         h = h + self.sdc(a)
         h = self.gene(h)
         return (h, D, a) if return_attn else (h, D)
@@ -113,12 +122,14 @@ class LincsV9(nn.Module):
         self.expr_cell.fit(X_cell_train if X_cell_train is not None else X_ctl_train)
         return self
 
-    def forward(self, batch, return_aux=False, return_interp=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0, drug_atom_alpha=1.0):
+    def forward(self, batch, return_aux=False, return_interp=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0, drug_atom_alpha=1.0, drug_global_self_only=False, drug_xattn_global_only=False):
         diag = (diagonal or drug_diagonal or
                 (batch.get('drug_diagonal', False) if isinstance(batch, dict) else False) or
                 (batch.get('diagonal', False) if isinstance(batch, dict) else False))
         alpha = batch.get('drug_alpha', drug_alpha) if isinstance(batch, dict) else drug_alpha
         atom_alpha = batch.get('drug_atom_alpha', drug_atom_alpha) if isinstance(batch, dict) else drug_atom_alpha
+        batch_drug_global_self_only = batch.get('drug_global_self_only', drug_global_self_only) if isinstance(batch, dict) else drug_global_self_only
+        batch_drug_xattn_global_only = batch.get('drug_xattn_global_only', drug_xattn_global_only) if isinstance(batch, dict) else drug_xattn_global_only
         E, r = batch['E'], batch['r']
         x_ctl = batch['x_ctl']
         x_cell = batch.get('x_cell', x_ctl)
@@ -152,9 +163,9 @@ class LincsV9(nn.Module):
         attn = None
         for i, blk in enumerate(self.perturb):
             if return_interp and i == len(self.perturb) - 1:
-                h, D, attn = blk(h, D, key_mask, return_attn=True, diagonal=diag, drug_alpha=alpha, drug_atom_alpha=atom_alpha)
+                h, D, attn = blk(h, D, key_mask, return_attn=True, diagonal=diag, drug_alpha=alpha, drug_atom_alpha=atom_alpha, drug_global_self_only=batch_drug_global_self_only, xattn_global_only=batch_drug_xattn_global_only)
             else:
-                h, D = blk(h, D, key_mask, diagonal=diag, drug_alpha=alpha, drug_atom_alpha=atom_alpha)
+                h, D = blk(h, D, key_mask, diagonal=diag, drug_alpha=alpha, drug_atom_alpha=atom_alpha, drug_global_self_only=batch_drug_global_self_only, xattn_global_only=batch_drug_xattn_global_only)
 
         out, epi_contrib = self.heads(h, E, r, x_ctl)
         if return_aux or return_interp:
diff --git a/model/v9/modules_v9.py b/model/v9/modules_v9.py
index b9c4bdd..d16a168 100644
--- a/model/v9/modules_v9.py
+++ b/model/v9/modules_v9.py
@@ -215,13 +215,15 @@ class _DrugAttention(QKNormAttention):
     flow is eliminated.
     """
 
-    def forward(self, x, key_mask=None, diagonal=False, alpha=1.0, atom_alpha=1.0):
+    def forward(self, x, key_mask=None, diagonal=False, alpha=1.0, atom_alpha=1.0, global_self_only=False):
+        if global_self_only and (alpha < 1.0 or diagonal):
+            raise ValueError("Cannot combine global_self_only=True with alpha < 1.0 or diagonal=True")
         if atom_alpha < 1.0 and (alpha < 1.0 or diagonal):
             raise ValueError("Cannot combine atom_alpha < 1.0 with alpha < 1.0 or diagonal=True")
 
         if diagonal:
             alpha = 0.0
-        if alpha == 1.0 and atom_alpha == 1.0:
+        if alpha == 1.0 and atom_alpha == 1.0 and not global_self_only:
             return super().forward(x, key_mask=key_mask)
             
         B, L, _ = x.shape
@@ -243,7 +245,7 @@ class _DrugAttention(QKNormAttention):
             # Explicitly zero padded columns to ensure they are strictly 0.0 before blending
             A = A.masked_fill(key_mask[:, None, None, :], 0.0)
             
-        if atom_alpha < 1.0:
+        if atom_alpha < 1.0 or global_self_only:
             allow = torch.zeros(L, L, dtype=torch.bool, device=S.device)
             allow[:, 0] = True                      # every row may attend to the global token
             allow[torch.arange(L), torch.arange(L)] = True   # and to itself
@@ -251,7 +253,10 @@ class _DrugAttention(QKNormAttention):
             B_mat = torch.nan_to_num(torch.softmax(S_b, dim=-1), nan=0.0)
             
             A_new = A.clone()
-            A_new[..., 1:, :] = atom_alpha * A[..., 1:, :] + (1.0 - atom_alpha) * B_mat[..., 1:, :]
+            if atom_alpha < 1.0:
+                A_new[..., 1:, :] = atom_alpha * A[..., 1:, :] + (1.0 - atom_alpha) * B_mat[..., 1:, :]
+            if global_self_only:
+                A_new[..., 0, :] = B_mat[..., 0, :]
             
             # max over non-padded query rows of |A_new.sum(-1) - 1|
             rowsum = A_new.sum(-1)
@@ -298,8 +303,16 @@ class _DrugBlock(nn.Module):
         self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
         self.sd1, self.sd2 = StochasticDepth(p_drop), StochasticDepth(p_drop)
 
-    def forward(self, D, key_mask=None, diagonal=False, alpha=1.0, atom_alpha=1.0):
-        if hasattr(self.attn, 'forward') and 'atom_alpha' in self.attn.forward.__code__.co_varnames:
+    def forward(self, D, key_mask=None, diagonal=False, alpha=1.0, atom_alpha=1.0, global_self_only=False):
+        has_global_self_only = hasattr(self.attn, 'forward') and 'global_self_only' in self.attn.forward.__code__.co_varnames
+        has_atom_alpha = hasattr(self.attn, 'forward') and 'atom_alpha' in self.attn.forward.__code__.co_varnames
+
+        if global_self_only and not has_global_self_only:
+            raise ValueError('global_self_only=True requested but %s does not implement it; refusing to fall back.' % type(self.attn).__name__)
+
+        if has_global_self_only and has_atom_alpha:
+            attn_out = self.attn(self.n1(D), key_mask=key_mask, diagonal=diagonal, alpha=alpha, atom_alpha=atom_alpha, global_self_only=global_self_only)
+        elif has_atom_alpha:
             attn_out = self.attn(self.n1(D), key_mask=key_mask, diagonal=diagonal, alpha=alpha, atom_alpha=atom_alpha)
         elif hasattr(self.attn, 'forward') and 'alpha' in self.attn.forward.__code__.co_varnames:
             attn_out = self.attn(self.n1(D), key_mask=key_mask, diagonal=diagonal, alpha=alpha)
```

## Verification Output

### `test_residual_cuts.py`
```
T7 diff_rand: 0.0
T7 diff_mean: 0.0
T8 diff_xattn: 0.13643133640289307
T8 diff_g: 0.43807607889175415
All tests passed.
```

### `test_atom_only.py`
```
T10 influence of atom 3 on atom 1:
  atom_alpha=0.0: 0.0
  atom_alpha=0.25: 0.09328436851501465
  atom_alpha=0.5: 0.18780428171157837
  atom_alpha=0.75: 0.2835272550582886
  atom_alpha=1.0: 0.38038721680641174
T11 delta output moved: 0.514825701713562 (Because atom 3 reaches the global token in perturb block 1, and the global token informs genes in perturb block 2)
All tests passed.
```

### `test_alpha_sweep.py`
```
Test 1 passed: alpha=1.0 difference is 0.00e+00
Test 2 passed: alpha=0.0 difference is 0.00e+00
Test 3: Monotone information flow
  alpha=0.00: diff=0.0000e+00
  alpha=0.25: diff=9.5408e-01
  alpha=0.50: diff=1.9089e+00
  alpha=0.75: diff=2.8653e+00
  alpha=1.00: diff=3.8238e+00
Test 4 passed: capacity is 502903 at every alpha, asserted at 5 values
Test 5 passed: padding holds at every alpha
Test 6 passed: no NaN at any alpha
```

### `test_drugsa_v9.py`
```
[PASS] drug self-attention CONTEXTUALISES: perturbing atom 3 moves atom 1  -- |d|max 0.2754
[PASS] baseline arm is a BAG: with no drug_sa, atom 1 is bit-identical when atom 3 changes
[PASS] padded atoms do not leak into real ones  -- |d|max 0.00e+00
[PASS] shape preserved and output finite  -- (2, 6, 64)
[PASS] flag defaults to False and is switchable  -- off=False on=True
[INFO] _DrugBlock params at d_model=64: 32,868 (per perturb block)
[PASS] diagonal removes cross-atom flow: perturbing atom 3 moves atom 1 by exactly 0.00e+00  -- |d|max 0.00e+00
[PASS] capacity is preserved: parameter count under diagonal=True is identical  -- 32,868 params
[PASS] diagonal mode is not a no-op: SwiGLU and residual are live  -- |d|max 1.1266
[PASS] padding still holds under diagonal: changing padded token leaves real tokens untouched  -- |d|max 0.00e+00
[PASS] default path unchanged: omitted argument is bit-identical to diagonal=False  -- diff 0.00e+00
[PASS] PerturbBlock threads diagonal: atom 1 untouched under atom 3 perturbation  -- |d|max 0.00e+00
[PASS] LincsV9 threads diagonal: default bit-identical to diagonal=False, diagonal=True runs finite

11/11 drug-self-attention checks passed
```

### `test_interaction_2x2.py`
```
=== Testing 2x2 Interaction Harness [TASK W5] ===

[PASS] Check 1: all four cells return finite scores (S11=+0.03362, S01=+0.01084, S10=+0.02723, S00=+0.00698, interaction=+0.00253)
[PASS] Check 2: diagonal flag changes scores and activations (S10 != S11: +0.02723 != +0.03362, S00 != S01: +0.00698 != +0.01084, |dY|max_context=0.1366)
[PASS] Check 3: interaction equals (S11 - S01) - (S10 - S00) to 1e-12 (harness=0.002530000000, recomputed=0.002530000000, diff=6.07e-18)
[PASS] Check 4: with drug_self_attn=False, harness reports diagonal-unavailable rather than producing numbers:
         'diagonal mode unavailable for this checkpoint: cfg.drug_self_attn=False, perturb blocks with drug_sa=0/1'

ALL 4 CHECKS PASSED: harness verified on small untrained model.
```

### `test_v9.py`
```
C:\Projects\LINCS\model\v9\modules_v9.py:98: UserWarning: torch.searchsorted(): input value tensor is non-contiguous, this will lower the performance due to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous input value tensor if possible. This message will only appear once per program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen/native/BucketizationUtils.h:34.)
  return torch.bucketize(x, self.edges[0].to(x.dtype), right=False).clamp_(0, self.n_bins - 1)
[PASS] M_pathway is [P, 978] and P matches the config  -- (800, 978)
[PASS] every pathway node has at least one member gene (no dead node)  -- 0 empty
[PASS] row p of M is row p of pathway_info_v9.tsv, verified by rebuilding the gene set  -- checked every 37th row, via the HGNC bridge
[PASS] the symbol bridge is complete: every current symbol maps to exactly one canonical row
[PASS] every pathway node carries a curated NAME and a source  -- 367 Reactome / 433 GO:BP
[PASS] landmark symbol map is in canonical order and resolves all 978  -- 34 renamed
[PASS] STRING v9 adjacency is [978, 978] and denser than the truncated one  -- 13001 undirected edges vs 12,665 before
[PASS] pretrained gene vectors are [978, d_gene_vec]  -- (978, 128)
[PASS] a binned model REFUSES to run before its quantiser is fitted (an unfitted quantiser would silently ignore the expression input entirely)
[PASS] after fit_bins the model reports its quantisers fitted
[PASS] head "abs" is [B, 978] and finite
[PASS] head "delta" is [B, 978] and finite
[PASS] head "l5" is [B, 978] and finite
[PASS] return_aux does not change the prediction
[PASS] THE ABSOLUTE HEAD IS ANCHORED: abs == x_ctl + delta, exactly  -- max dev 9.54e-07
[PASS] quantiser starts UNFITTED, so evaluation data can never define the bins
[PASS] after fitting on TRAIN rows the quantiser is marked fitted
[PASS] bins are quantile-spaced: usage is spread, not collapsed onto one level  -- 128/128 bins used
[PASS] binning is monotone in the value (larger expression never gets a smaller bin)
[PASS] n_bins is 128, the field's setting
[PASS] fitting on a sample containing ONE NaN row is REFUSED, not silently NaN-poisoned
[PASS] after filtering the NaN rows the same sample fits and DISCRIMINATES
[PASS] a NaN-edged quantiser reports itself unusable even though fitted == 1
[PASS] and it REFUSES to run rather than returning a constant embedding
[PASS] a fitted quantiser separates real values into many bins (not all into one)  -- 128 distinct bins on 256 rows
[PASS] the raw encoder remains available as the A/B control arm
[PASS] matched control is live: ablating it to the batch mean moves the prediction  -- dScore +0.0005  |dY|max 0.0330
[PASS] chromatin is live as a gene embedding (|dY|max > 0 distinguishes a null from a dead branch)  -- dScore +0.2295  |dY|max 3.3578
[PASS] STRING message passing is zero-init, so it is an exact no-op before training  -- |dY|max 0.0000 (expected exactly 0)
[PASS] STRING message passing is genuinely WIRED IN: give it non-zero weights and it moves the output  -- dScore +0.0179  |dY|max 0.6029
[PASS] genes with no STRING edge receive exactly zero message  -- 932/978 genes have an edge
[PASS] named pathway readout is live  -- dScore +0.2654  |dY|max 12.3192
[PASS] ablation helper reports |dY|max alongside the score change (method rule 2)
[PASS] chromatin enters the gene representation as an embedding
[PASS] the signed additive chromatin HEAD is kept as well
[PASS] chromatin contribution is signed (both directions occur)
[PASS] reliability r=0 zeroes the chromatin head contribution for those genes  -- max |contrib| at r=0: 0.00e+00
[PASS] the control encoder shares no parameters with the perturbation stream  -- 22 vs 64 tensors
[PASS] the two control views are both used: dropping the per-cell view changes the prediction  -- |dY|max 8.7927
[PASS] no learned task-weight parameter exists (v7 measured Kendall weighting sending ~90% of the gradient to the auxiliaries)
[PASS] auxiliary weights are small and fixed  -- pathway 0.05, epi 0.05
[PASS] loss is finite and multi-task  -- abs, aux_epi, aux_pathway, delta, l5, pcc
[PASS] a fully-masked target contributes no NaN (per-target row masks work)
[PASS] the anchored absolute head makes the abs loss IDENTICAL to the delta loss (so w_abs is not a second task: the effective delta weight is w_abs + w_delta)  -- abs 1.166905 vs delta 1.166905
[PASS] aux targets come from the MEASURED response and match the head shapes
[PASS] pathway aux target is a masked mean of |delta| (non-negative, bounded by max|delta|)
[PASS] pathway activations are one per NAMED node  -- (4, 800, 32)
[PASS] a measured chance level is available for any readout (method rule 6)  -- null mean -0.0101 sd 0.0735, p=0.900
[PASS] pathway alignment returns a rank correlation in [-1, 1]  -- +0.0201
[PASS] CCLE is off by default (redundant and dominated, RESULTS 27.4), but still switchable
[PASS] all three targets are predicted
[PASS] the Level-5 head is retained so v3-v7 numbers stay comparable
[PASS] parameter count is in a trainable range for a T4 x2 budget  -- 10.7M
[PASS] construction is deterministic under a fixed seed (weights AND quantiser)
[PASS] B=1 works

55/55 checks passed
```

### `alpha_sweep.py --help`
```
usage: alpha_sweep.py [-h] --ckpt CKPT [--alphas ALPHAS] [--batch BATCH]
                      [--n_eval N_EVAL] [--n_boot N_BOOT] [--seed SEED]
                      [--key KEY] [--operator {full,atom_only}]
                      [--cut {none,xattn,global,both}]

options:
  -h, --help            show this help message and exit
  --ckpt CKPT
  --alphas ALPHAS
  --batch BATCH
  --n_eval N_EVAL
  --n_boot N_BOOT
  --seed SEED
  --key KEY
  --operator {full,atom_only}
  --cut {none,xattn,global,both}
```

### `alpha_sweep.py` failure check (invalid arguments)
```
usage: alpha_sweep.py [-h] --ckpt CKPT [--alphas ALPHAS] [--batch BATCH]
                      [--n_eval N_EVAL] [--n_boot N_BOOT] [--seed SEED]
                      [--key KEY] [--operator {full,atom_only}]
                      [--cut {none,xattn,global,both}]
alpha_sweep.py: error: --cut requires --operator atom_only
```

### `alpha_sweep.py` refusal check (invalid checkpoint)
```
=== v9 Alpha Sweep [TASK W6] ===
Checkpoint: r0_ckpt_v9_fold0_seed0.pt | Device: cuda
Alphas: [0.0]
[REFUSED] diagonal mode unavailable for this checkpoint: cfg.drug_self_attn=False, perturb blocks with drug_sa=0/4
```

## What I was unsure about
The instructions were very precise and clear, so there were no major points of uncertainty.
* One small fix: In `test_residual_cuts.py` T10, I set `cfg_no_sa.n_genes = 10` to match the `M_pathway` matrix shape provided to the model constructor, otherwise it hit a generic assertion error rather than the tested `ValueError`. This resolved the issue.
* I used `.mean(dim=1)` instead of `.mean(dim=0)` in T7 (acceptance test with batch mean) because the test was meant to replace all atoms in a molecule with their mean in the batch, or alternatively I used `.mean(dim=0)` as the sweep does. Actually, I followed the sweep's `make_atom_ablated_batch` which does `.mean(dim=0, keepdim=True).expand_as(ref)`. So I matched exactly what the sweep does to ensure consistency.

All constraints and acceptance tests were met successfully.
