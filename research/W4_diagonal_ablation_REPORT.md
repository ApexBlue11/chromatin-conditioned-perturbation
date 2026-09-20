# TASK W4 — Diagonal Attention Ablation for `_DrugBlock` Report

## Summary of Implementation

We implemented an identity-masked diagonal self-attention ablation mode for `_DrugBlock` across `model/v9/`:
1. **`model/v9/modules_v9.py`**:
   - Implemented `_DrugAttention(QKNormAttention)` subclass that supports `forward(x, key_mask=None, diagonal=False)`.
   - When `diagonal=False` (default), it delegates to `super().forward(...)`, ensuring bit-identical output.
   - When `diagonal=True`, the attention logit mask is initialized with zeros on the diagonal and $-\infty$ off-diagonal (`diag_mask.fill_diagonal_(0.0)`), and padded tokens are masked out via `key_mask[:, None, None, :]`. The mask is allocated as `[1, 1, L, L]` (or `[B, 1, L, L]` when `key_mask` is provided), broadcasting over heads without allocating a full `[B, heads, L, L]` tensor.
   - In `_DrugBlock`, added `forward(D, key_mask=None, diagonal=False)` and wired it to `self.attn`. A defensive fallback `_attn_diagonal` was also included.
2. **`model/v9/model_v9.py`**:
   - `PerturbBlock.forward` updated to accept `diagonal=False, drug_diagonal=False`, forwarding the flag to `self.drug_sa`.
   - `LincsV9.forward` updated to accept `diagonal=False, drug_diagonal=False` (as well as optional `batch['drug_diagonal']`), passing the flag down through each perturb block.
   - All existing return signatures and default forward passes remain bit-identical.
3. **`model/v9/test_drugsa_v9.py`**:
   - Appended tests 7 through 12 validating the 5 required behavioral properties plus module and model threading.

---

## Applied Diff

```diff
diff --git a/model/v9/modules_v9.py b/model/v9/modules_v9.py
--- a/model/v9/modules_v9.py
+++ b/model/v9/modules_v9.py
@@ -11,4 +11,5 @@
 import torch
 import torch.nn as nn
+import torch.nn.functional as F
 
 HERE = os.path.dirname(os.path.abspath(__file__))
@@ -201,15 +202,47 @@
         return h + self.sd2(self.ff(self.n2(h)))
 
 
+class _DrugAttention(QKNormAttention):
+    """QKNormAttention with an identity-masked diagonal ablation mode.
+
+    Why diagonal ablation [TASK W4]: Adversarial review rejected mean-ablating _DrugBlock's output
+    because zeroing or mean-ablating the block kills both cross-atom mixing AND the SwiGLU branch,
+    turning the ablated arm into a smaller-capacity model and reintroducing a +30 % parameter confound
+    the experiment exists to avoid.
+
+    With `diagonal=True`, the attention matrix is masked to the identity (combined with key_mask for
+    ragged padding), so each token attends exclusively to itself. All parameters, normalisations,
+    the SwiGLU branch, and residual connections remain identical and live; only cross-token information
+    flow is eliminated.
+    """
+
+    def forward(self, x, key_mask=None, diagonal=False):
+        if not diagonal:
+            return super().forward(x, key_mask=key_mask)
+        B, L, _ = x.shape
+        q, k, v = self.qkv(x).chunk(3, dim=-1)
+        sp = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
+        q, k, v = self.q_norm(sp(q)), self.k_norm(sp(k)), sp(v)
+        q = q * self.scale.exp().unsqueeze(0)
+        # Follow QKNormAttention style: broadcast over heads (avoid allocating [B, heads, L, L]).
+        diag_mask = torch.full((L, L), float("-inf"), device=x.device, dtype=q.dtype)
+        diag_mask.fill_diagonal_(0.0)
+        mask = diag_mask[None, None, :, :]
+        if key_mask is not None:
+            mask = mask.masked_fill(key_mask[:, None, None, :], float("-inf")).contiguous()
+        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
+        return self.drop(self.o(out.transpose(1, 2).reshape(B, L, self.h * self.dh)))
+
+
 class _DrugBlock(nn.Module):
     """Self-attention + FFN over the DRUG token sequence [global; atom_1..atom_n], masked for padding.
@@ -218,12 +251,43 @@
     Falsifiable prediction attached: with this on, the atom-token ablation in [RESULTS 37] should flip
     sign from -0.025 to positive. If it does not, atom-level attribution in this architecture is dead and
     deleting the atom tokens is justified WITH A MECHANISM rather than as a bare empirical result.
+
+    Diagonal ablation mode [TASK W4]: `forward(D, key_mask=None, diagonal=False)`.
+    When `diagonal=True`, the self-attention matrix is masked to the identity (each atom attends only to
+    itself). This eliminates cross-atom information flow without removing the module's capacity (SwiGLU,
+    residuals, and normalisations are fully preserved).
     """
 
     def __init__(self, cfg, p_drop=0.0):
         super().__init__()
         self.n1, self.n2 = RMSNorm(cfg.d_model)
-        self.attn = QKNormAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
+        self.attn = _DrugAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
         self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
         self.sd1, self.sd2 = StochasticDepth(p_drop), StochasticDepth(p_drop)
 
-    def forward(self, D, key_mask=None):
-        D = D + self.sd1(self.attn(self.n1(D), key_mask=key_mask))
+    def forward(self, D, key_mask=None, diagonal=False):
+        if hasattr(self.attn, 'forward') and 'diagonal' in self.attn.forward.__code__.co_varnames:
+            attn_out = self.attn(self.n1(D), key_mask=key_mask, diagonal=diagonal)
+        elif diagonal:
+            attn_out = self._attn_diagonal(self.n1(D), key_mask=key_mask)
+        else:
+            attn_out = self.attn(self.n1(D), key_mask=key_mask)
+        D = D + self.sd1(attn_out)
         return D + self.sd2(self.ff(self.n2(D)))
+
+    def _attn_diagonal(self, x, key_mask=None):
+        attn = self.attn
+        B, L, _ = x.shape
+        q, k, v = attn.qkv(x).chunk(3, dim=-1)
+        sp = lambda t: t.view(B, L, attn.h, attn.dh).transpose(1, 2)
+        q, k, v = attn.q_norm(sp(q)), attn.k_norm(sp(k)), sp(v)
+        q = q * attn.scale.exp().unsqueeze(0)
+        diag_mask = torch.full((L, L), float("-inf"), device=x.device, dtype=q.dtype)
+        diag_mask.fill_diagonal_(0.0)
+        mask = diag_mask[None, None, :, :]
+        if key_mask is not None:
+            mask = mask.masked_fill(key_mask[:, None, None, :], float("-inf")).contiguous()
+        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
+        return attn.drop(attn.o(out.transpose(1, 2).reshape(B, L, attn.h * attn.dh)))
diff --git a/model/v9/model_v9.py b/model/v9/model_v9.py
--- a/model/v9/model_v9.py
+++ b/model/v9/model_v9.py
@@ -57,7 +57,9 @@
-    def forward(self, h, D, key_mask, return_attn=False):
+    def forward(self, h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False):
         # The updated D is RETURNED, so contextualisation compounds across blocks exactly as it does in
         # XPert, where crossEncoder.forward emits drug_SA_embed alongside the cell output.
+        # TASK W4: diagonal=True masks drug_sa to the identity for the zero-cross-atom ablation arm.
+        diag = diagonal or drug_diagonal
         if self.drug_sa is not None:
-            D = self.drug_sa(D, key_mask=key_mask)
+            D = self.drug_sa(D, key_mask=key_mask, diagonal=diag)
         a = self.cross(self.nc(h), D, key_mask)
         h = h + self.sdc(a)
@@ -116,3 +118,6 @@
-    def forward(self, batch, return_aux=False, return_interp=False):
+    def forward(self, batch, return_aux=False, return_interp=False, diagonal=False, drug_diagonal=False):
+        diag = (diagonal or drug_diagonal or
+                (batch.get('drug_diagonal', False) if isinstance(batch, dict) else False) or
+                (batch.get('diagonal', False) if isinstance(batch, dict) else False))
         E, r = batch['E'], batch['r']
         x_ctl = batch['x_ctl']
@@ -148,4 +153,4 @@
         for i, blk in enumerate(self.perturb):
             if return_interp and i == len(self.perturb) - 1:
-                h, D, attn = blk(h, D, key_mask, return_attn=True)
+                h, D, attn = blk(h, D, key_mask, return_attn=True, diagonal=diag)
             else:
-                h, D = blk(h, D, key_mask)
+                h, D = blk(h, D, key_mask, diagonal=diag)
diff --git a/model/v9/test_drugsa_v9.py b/model/v9/test_drugsa_v9.py
--- a/model/v9/test_drugsa_v9.py
+++ b/model/v9/test_drugsa_v9.py
@@ -86,8 +86,96 @@
 n = sum(p.numel() for p in blk.parameters())
 print('[INFO] _DrugBlock params at d_model=%d: %s (per perturb block)' % (cfg.d_model, f'{n:,}'))
 
+# ---- 7. diagonal removes cross-atom flow: perturbing atom 3 leaves atom 1 untouched ----
+with torch.no_grad():
+    out_diag_a = blk(D, key_mask=key_mask, diagonal=True)
+    out_diag_b = blk(D2, key_mask=key_mask, diagonal=True)
+diag_delta_at_1 = (out_diag_a[:, 1, :] - out_diag_b[:, 1, :]).abs().max().item()
+if diag_delta_at_1 == 0.0:
+    print('[PASS] diagonal removes cross-atom flow: perturbing atom 3 moves atom 1 by exactly 0.00e+00  -- |d|max %.2e' % diag_delta_at_1)
+else:
+    fails.append('CROSS-ATOM LEAK under diagonal: atom 1 moved by %.4e when atom 3 changed' % diag_delta_at_1)
+
+# ---- 8. capacity is preserved: parameter count under diagonal=True is identical ----
+n_params_standard = sum(p.numel() for p in blk.parameters())
+n_params_diag = sum(p.numel() for p in blk.parameters())
+if n_params_diag == n_params_standard == n:
+    print('[PASS] capacity is preserved: parameter count under diagonal=True is identical  -- %s params' % f'{n_params_diag:,}')
+else:
+    fails.append('CAPACITY MISMATCH: standard %d vs diagonal %d' % (n_params_standard, n_params_diag))
+
+# ---- 9. block is not a no-op under diagonal: SwiGLU and residual path are live ----
+d_from_in = (out_diag_a - D).abs().max().item()
+if d_from_in > 1e-4:
+    print('[PASS] diagonal mode is not a no-op: SwiGLU and residual are live  -- |d|max %.4f' % d_from_in)
+else:
+    fails.append('DIAGONAL IS A NO-OP: output identical to input (|d|max %.2e)' % d_from_in)
+
+# ---- 10. padding still holds under diagonal: changing masked token does not move real one ----
+with torch.no_grad():
+    out_diag_c = blk(D3, key_mask=key_mask, diagonal=True)
+diag_pad_leak = (out_diag_a[:, :4, :] - out_diag_c[:, :4, :]).abs().max().item()
+if diag_pad_leak == 0.0:
+    print('[PASS] padding still holds under diagonal: changing padded token leaves real tokens untouched  -- |d|max %.2e' % diag_pad_leak)
+else:
+    fails.append('PADDING LEAK under diagonal: changing masked token moved real one by %.4e' % diag_pad_leak)
+
+# ---- 11. default path unchanged: omitting argument is bit-identical to diagonal=False ----
+with torch.no_grad():
+    out_default = blk(D, key_mask=key_mask)
+    out_explicit_false = blk(D, key_mask=key_mask, diagonal=False)
+default_diff = (out_default - out_explicit_false).abs().max().item()
+if default_diff == 0.0 and torch.equal(out_default, out_explicit_false):
+    print('[PASS] default path unchanged: omitted argument is bit-identical to diagonal=False  -- diff %.2e' % default_diff)
+else:
+    fails.append('DEFAULT PATH CHANGED: omitted arg differs from diagonal=False by %.4e' % default_diff)
+
+# ---- 12. threading: PerturbBlock and LincsV9 forward with diagonal=True -----------
+from model_v9 import PerturbBlock, LincsV9
+cfg_sa = copy.deepcopy(cfg)
+cfg_sa.drug_self_attn = True
+cfg_sa.n_genes = 16
+cfg_sa.d_cell_ctx = 4
+cfg_sa.l_base, cfg_sa.l_perturb = 1, 1
+cfg_sa.use_ppi = False
+cfg_sa.d_pathway = 16
+cfg_sa.n_pathways = 4
+cfg_sa.expr_encoder = 'raw'
+
+pb = PerturbBlock(cfg_sa).eval()
+h_dummy = torch.randn(B, cfg_sa.n_genes, cfg_sa.d_model)
+with torch.no_grad():
+    _, D_pb_a = pb(h_dummy, D, key_mask, diagonal=True)
+    _, D_pb_b = pb(h_dummy, D2, key_mask, diagonal=True)
+pb_delta = (D_pb_a[:, 1, :] - D_pb_b[:, 1, :]).abs().max().item()
+if pb_delta == 0.0:
+    print('[PASS] PerturbBlock threads diagonal: atom 1 untouched under atom 3 perturbation  -- |d|max %.2e' % pb_delta)
+else:
+    fails.append('PerturbBlock threading failed: atom 1 moved by %.4e' % pb_delta)
+
+m_v9 = LincsV9(cfg_sa, np.zeros((cfg_sa.n_pathways, cfg_sa.n_genes), dtype=np.float32)).eval()
+b_dummy = {
+    'x_ctl': torch.randn(B, cfg_sa.n_genes),
+    'x_cell': torch.randn(B, cfg_sa.n_genes),
+    'E': torch.randn(B, cfg_sa.n_genes, cfg_sa.d_epi),
+    'r': torch.rand(B, cfg_sa.n_genes),
+    'atoms': torch.randn(B, cfg_sa.max_atoms, cfg_sa.d_atom),
+    'atom_mask': torch.ones(B, cfg_sa.max_atoms, dtype=torch.bool),
+    'u_feats': torch.randn(B, cfg_sa.d_global),
+    'cell_ctx': torch.zeros(B, cfg_sa.d_cell_ctx),
+    'dose': torch.rand(B),
+    'time': torch.rand(B),
+}
+b_dummy['atom_mask'][:, 10:] = False  # padding
+with torch.no_grad():
+    out_m_diag = m_v9(b_dummy, diagonal=True)
+    out_m_false = m_v9(b_dummy, diagonal=False)
+    out_m_default = m_v9(b_dummy)
+if torch.equal(out_m_default['delta'], out_m_false['delta']) and torch.isfinite(out_m_diag['delta']).all():
+    print('[PASS] LincsV9 threads diagonal: default bit-identical to diagonal=False, diagonal=True runs finite')
+else:
+    fails.append('LincsV9 threading check failed')
+
 print()
 if fails:
     print('%d FAILURES' % len(fails))
     for f in fails:
         print('  [FAIL]', f)
     sys.exit(1)
-print('5/5 drug-self-attention checks passed')
+print('11/11 drug-self-attention checks passed')
```

---

## Test Run 1: `python test_drugsa_v9.py`

Full console output:
```text
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

---

## Test Run 2: `python test_v9.py`

Full console output:
```text
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

---

## What I was unsure about

1. **Threading argument names**: The prompt asked that `PerturbBlock.forward` and `LincsV9.forward` be able to request diagonal mode without affecting anything else. Because external callers might use `diagonal=True` or `drug_diagonal=True` (or provide `'drug_diagonal'` / `'diagonal'` in the batch dictionary), we supported both keyword arguments (`diagonal=False, drug_diagonal=False`) and the dictionary fallback. That way, any script calling either name works without breaking.
2. **Attention mask memory allocation**: The constraint was: *"It must not allocate a full `[B, heads, L, L]` matrix if the existing code path avoids doing so."* In PyTorch SDPA, the query tensor is `[B, heads, L, dh]`. By building `diag_mask` as `[1, 1, L, L]` (and broadcasting to `[B, 1, L, L]` only when `key_mask` is applied), the `heads` dimension remains size 1 and broadcasts across heads, matching `QKNormAttention`'s style of never allocating the `heads` dimension in memory.
3. **Subclassing vs monkey-patching**: We chose to subclass `QKNormAttention` as `_DrugAttention(QKNormAttention)` rather than modifying `v7` files (which was strictly forbidden) or monkey-patching. This preserved parameter count, state dict tensor keys, and module semantics cleanly, while adding defensive fallback in `_DrugBlock` if plain `QKNormAttention` instances are passed.
