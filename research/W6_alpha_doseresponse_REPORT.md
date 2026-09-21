# TASK W6 — alpha dose-response REPORT

## Diff Applied
```diff
diff --git a/model/v9/model_v9.py b/model/v9/model_v9.py
index 1dc87c8..248573e 100644
--- a/model/v9/model_v9.py
+++ b/model/v9/model_v9.py
@@ -54,13 +54,13 @@ class PerturbBlock(nn.Module):
         # crossEncoder does. Default off; `--drug_self_attn` turns it on for the A/B.
         self.drug_sa = _DrugBlock(cfg, p_drop) if getattr(cfg, 'drug_self_attn', False) else None
 
-    def forward(self, h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False):
+    def forward(self, h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0):
         # The updated D is RETURNED, so contextualisation compounds across blocks exactly as it does in
         # XPert, where crossEncoder.forward emits drug_SA_embed alongside the cell output.
         # TASK W4: diagonal=True masks drug_sa to the identity for the zero-cross-atom ablation arm.
         diag = diagonal or drug_diagonal
         if self.drug_sa is not None:
-            D = self.drug_sa(D, key_mask=key_mask, diagonal=diag)
+            D = self.drug_sa(D, key_mask=key_mask, diagonal=diag, alpha=drug_alpha)
         a = self.cross(self.nc(h), D, key_mask)
         h = h + self.sdc(a)
         h = self.gene(h)
@@ -113,10 +113,11 @@ class LincsV9(nn.Module):
         self.expr_cell.fit(X_cell_train if X_cell_train is not None else X_ctl_train)
         return self
 
-    def forward(self, batch, return_aux=False, return_interp=False, diagonal=False, drug_diagonal=False):
+    def forward(self, batch, return_aux=False, return_interp=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0):
         diag = (diagonal or drug_diagonal or
                 (batch.get('drug_diagonal', False) if isinstance(batch, dict) else False) or
                 (batch.get('diagonal', False) if isinstance(batch, dict) else False))
+        alpha = batch.get('drug_alpha', drug_alpha) if isinstance(batch, dict) else drug_alpha
         E, r = batch['E'], batch['r']
         x_ctl = batch['x_ctl']
         x_cell = batch.get('x_cell', x_ctl)
@@ -150,9 +151,9 @@ class LincsV9(nn.Module):
         attn = None
         for i, blk in enumerate(self.perturb):
             if return_interp and i == len(self.perturb) - 1:
-                h, D, attn = blk(h, D, key_mask, return_attn=True, diagonal=diag)
+                h, D, attn = blk(h, D, key_mask, return_attn=True, diagonal=diag, drug_alpha=alpha)
             else:
-                h, D = blk(h, D, key_mask, diagonal=diag)
+                h, D = blk(h, D, key_mask, diagonal=diag, drug_alpha=alpha)
 
         out, epi_contrib = self.heads(h, E, r, x_ctl)
         if return_aux or return_interp:
diff --git a/model/v9/modules_v9.py b/model/v9/modules_v9.py
index e405109..373962a 100644
--- a/model/v9/modules_v9.py
+++ b/model/v9/modules_v9.py
@@ -215,22 +215,37 @@ class _DrugAttention(QKNormAttention):
     flow is eliminated.
     """
 
-    def forward(self, x, key_mask=None, diagonal=False):
-        if not diagonal:
+    def forward(self, x, key_mask=None, diagonal=False, alpha=1.0):
+        if diagonal:
+            alpha = 0.0
+        if alpha == 1.0:
             return super().forward(x, key_mask=key_mask)
+            
         B, L, _ = x.shape
         q, k, v = self.qkv(x).chunk(3, dim=-1)
         sp = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
         q, k, v = self.q_norm(sp(q)), self.k_norm(sp(k)), sp(v)
         q = q * self.scale.exp().unsqueeze(0)
-        # Follow QKNormAttention style: broadcast over heads (avoid allocating [B, heads, L, L]).
-        diag_mask = torch.full((L, L), float("-inf"), device=x.device, dtype=q.dtype)
-        diag_mask.fill_diagonal_(0.0)
-        mask = diag_mask[None, None, :, :]
+        
+        # Post-softmax attention matrix blending
+        attn = q @ k.transpose(-2, -1)
         if key_mask is not None:
-            mask = mask.masked_fill(key_mask[:, None, None, :], float("-inf")).contiguous()
-        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
-        return self.drop(self.o(out.transpose(1, 2).reshape(B, L, self.h * self.dh)))
+            attn = attn.masked_fill(key_mask[:, None, None, :], float("-inf"))
+            
+        A = F.softmax(attn, dim=-1)
+        # Avoid NaN on fully masked rows (e.g. completely padded sequences)
+        A = torch.nan_to_num(A, nan=0.0)
+        
+        if key_mask is not None:
+            # Explicitly zero padded columns to ensure they are strictly 0.0 before blending
+            A = A.masked_fill(key_mask[:, None, None, :], 0.0)
+            
+        I = torch.eye(L, device=x.device, dtype=A.dtype)[None, None, :, :]
+        
+        A_blend = alpha * A + (1.0 - alpha) * I
+        
+        out = self.drop(A_blend) @ v
+        return self.o(out.transpose(1, 2).reshape(B, L, self.h * self.dh))
 
 
 class _DrugBlock(nn.Module):
@@ -262,10 +277,12 @@ class _DrugBlock(nn.Module):
         self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
         self.sd1, self.sd2 = StochasticDepth(p_drop), StochasticDepth(p_drop)
 
-    def forward(self, D, key_mask=None, diagonal=False):
-        if hasattr(self.attn, 'forward') and 'diagonal' in self.attn.forward.__code__.co_varnames:
+    def forward(self, D, key_mask=None, diagonal=False, alpha=1.0):
+        if hasattr(self.attn, 'forward') and 'alpha' in self.attn.forward.__code__.co_varnames:
+            attn_out = self.attn(self.n1(D), key_mask=key_mask, diagonal=diagonal, alpha=alpha)
+        elif hasattr(self.attn, 'forward') and 'diagonal' in self.attn.forward.__code__.co_varnames:
             attn_out = self.attn(self.n1(D), key_mask=key_mask, diagonal=diagonal)
-        elif diagonal:
+        elif diagonal or alpha == 0.0:
             attn_out = self._attn_diagonal(self.n1(D), key_mask=key_mask)
         else:
             attn_out = self.attn(self.n1(D), key_mask=key_mask)
```

## `test_alpha_sweep.py` Output
```text
Test 1 passed: alpha=1.0 difference is 0.00e+00
Test 2 passed: alpha=0.0 difference is 0.00e+00
Test 3: Monotone information flow
  alpha=0.00: diff=0.0000e+00
  alpha=0.25: diff=9.3374e-01
  alpha=0.50: diff=1.8734e+00
  alpha=0.75: diff=2.8175e+00
  alpha=1.00: diff=3.7634e+00
Test 4 passed: capacity is 502903 at all alphas (structurally guaranteed)
Test 5 passed: padding holds at every alpha
Test 6 passed: no NaN at any alpha
```

## `test_drugsa_v9.py` Output
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

## `test_interaction_2x2.py` Output
```text
=== Testing 2x2 Interaction Harness [TASK W5] ===

[PASS] Check 1: all four cells return finite scores (S11=+0.03362, S01=+0.01084, S10=+0.02723, S00=+0.00698, interaction=+0.00253)
[PASS] Check 2: diagonal flag changes scores and activations (S10 != S11: +0.02723 != +0.03362, S00 != S01: +0.00698 != +0.01084, |dY|max_context=0.1366)
[PASS] Check 3: interaction equals (S11 - S01) - (S10 - S00) to 1e-12 (harness=0.002530000000, recomputed=0.002530000000, diff=6.07e-18)
[PASS] Check 4: with drug_self_attn=False, harness reports diagonal-unavailable rather than producing numbers:
         'diagonal mode unavailable for this checkpoint: cfg.drug_self_attn=False, perturb blocks with drug_sa=0/1'

ALL 4 CHECKS PASSED: harness verified on small untrained model.
```

## `test_v9.py` Output
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

## Refusal Output for `alpha_sweep.py`
```text
=== v9 Alpha Sweep [TASK W6] ===
Checkpoint: r0_ckpt_v9_fold0_seed0.pt | Device: cpu
Alphas: [0.5]
[REFUSED] diagonal mode unavailable for this checkpoint: cfg.drug_self_attn=False, perturb blocks with drug_sa=0/4
```

## What I was unsure about
- I replaced `F.scaled_dot_product_attention` with explicit standard attention (softmax over `q @ k.T`). The original code used the PyTorch `F.scaled_dot_product_attention`. Explicit attention allows blending the attention matrix (A) directly. This forces allocations of an $L \times L$ attention matrix in the `0.0 < alpha < 1.0` pathway but since $L$ is roughly 34 tokens, the memory overhead is negligible as mandated by the instructions.
- Handling NaN in the post-softmax matrix: `F.softmax` gives NaN if the row values are entirely -inf. This scenario occurs for fully masked query rows (e.g. completely padded queries). I addressed it by using `torch.nan_to_num(A, nan=0.0)` which aligns perfectly with how `F.scaled_dot_product_attention` yields zeroes natively.
- The `alpha=1.0` requirement strictly instructed `super().forward()` if `not diagonal`. Therefore, memory allocations scaling to $L \times L$ are completely bypassed for `alpha=1.0`.
- The identity matrix `I` behaves cleanly. The padded tokens attend effectively only to themselves via the trace, which evaluates to exactly zero influence upon real tokens in the final `out`. Additionally, padded keys are explicitly zeroed prior to blending.
