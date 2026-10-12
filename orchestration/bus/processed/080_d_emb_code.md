# PACKET 080 — CODE: D's embedding precompute (W39 + PI), before any caption kernel runs (RESULTS §98.15–98.17)
packet_id: 080
created: 2026-10-12
repo_commit: (this commit)
type: **CODE REVIEW** (no D output exists; nothing has run except the PI's two probes, which used public SMILES and random
tensors)

## What to review
- **`model/v9/p10_pertdit_emb.py`** (W39 + PI):
  - `compound_map` and `dose_values` read only `meta_pert_idx` / `meta_dose`;
  - `build_inputs` produces the restricted tables, the 5 cost-balanced slices, the NaN placeholder, the dose list, the
    compound map and the manifest;
  - `make_emb_script` makes 98.16 C1's two substitutions plus a header;
  - `merge_caches` and `check_outputs`.
- **`orchestration/make_p10_d_emb_kernels.py`** (W39 + PI): 5 caption kernels and 1 merge kernel. All are free CPU, internet-on, on
  the pinned image.
- **`model/v9/test_p10_pertdit_emb.py`** (W39 + PI): 25 tests. They include **executing** their generated script with a
  stub `transformers`, and executing a caption kernel and the merge kernel with `/kaggle` redirected.
- **The real input files,** built by the PI from the bundle's `meta_pert_idx` / `meta_dose` and XPert's SMILES table:
  `model/results/p10/p10d_inputs/` (`inputs_manifest.json` lists the sha1s).
- **The generated script** `model/results/p10/p10d_inputs/prepare_pertdit_embeddings_p10.py`, with its diff against their file
  below.
- **The generated kernels** under `external/kaggle_kernels/kern_p10d_{cap_s0..s4,emb_merge}/`.

## The PI's changes to W39's code (disclosed)
1. **`build_inputs` refuses in-memory inputs.** W39 had a dict path that skipped guard (g1); W36 had the same bypass.
2. **`make_emb_script` checks its source's sha1** against `SRC_SHA1`. W39's header asserted the sha1 without checking it.
3. **The builder had no module-level `fatal()`,** so any builder failure was a `NameError`. Fixed. The builder also no longer
   falls back to `model/v9/` or regenerates the script. Every pinned file must sit in the folder that becomes the dataset.
4. **The merge record** keeps each slice cache's sha1.
5. **Tests added:**
   - each substitution target missing alone;
   - the source pin;
   - the in-memory refusal;
   - the builder's refusals.

## The mutation check
- **The first run, against W39's suite:** **11 of 31 mutants survived.**
  - Every kernel guard had been exercised only on its happy path: the mounted-file sha1s, both weight sha256s, the slice-cache
    keys, the cache-against-record check, the merge's `check_outputs`, and the skip line.
  - Two `check_outputs` branches were also untested: the dtype, and dose finiteness.
- **The fix:** the PI added 4 tests that build one world per guard and run the generated kernel as a subprocess. Each world fails
  only that guard, and the test asserts its FATAL message.
- **The re-run:** **1 of 31 survives**, and it is equivalent. It is `check_outputs`' extra-dose-key loop, which can't be reached:
  the count check and the missing-key check together exclude an extra key.
- **The test count:** 25 now pass (21 W39 + PI; 4 PI guard worlds).

## The generated script's diff against their file
```diff
diff --git a/external/bai_benchmark_1d750cd/experiments/main_benchmark/data_prep/prepare_pertdit_embeddings.py b/model/results/p10/p10d_inputs/prepare_pertdit_embeddings_p10.py
index 7330d8a..a697d5c 100644
--- a/external/bai_benchmark_1d750cd/experiments/main_benchmark/data_prep/prepare_pertdit_embeddings.py
+++ b/model/results/p10/p10d_inputs/prepare_pertdit_embeddings_p10.py
@@ -1,3 +1,4 @@
+# Source sha1: 2bea30573a174d039b1d6de2569ad562f1aee4e5; RESULTS 98.15 / 98.16, P10 arm D
 """Generate per-drug LLM embeddings for the 8981 drugs in the L1000 SDST set.
 
 Mirrors PertDiT upstream preprocessing.ipynb:
@@ -25,6 +26,7 @@ IDX2SMI_PATH = os.path.join(XPERT_ROOT, "data/XPert/processed_data/all_drugs_idx
 ORIGINAL_EMB_PATH = os.path.join(
     XPERT_ROOT, "data", "PertDiT", "extracted", "pert_smiles_emb.pkl")
 OUT_DIR = os.path.join(XPERT_ROOT, "data", "PertDiT", "bench_drug_emb")
+DOSE_LIST_PATH = os.path.join(XPERT_ROOT, "data", "PertDiT", "p10_dose_list.json")
 
 
 def main():
@@ -71,14 +73,7 @@ def main():
     t0 = time.time()
 
     if remaining:
-        # Generating captions requires CUDA — fail fast on CPU-only nodes.
-        if not torch.cuda.is_available():
-            sys.exit(
-                f"ERROR: {len(remaining)} SMILES still need MolT5 captions, "
-                "which require a CUDA GPU. Either run this script on a GPU "
-                "node, or copy a complete _caption_cache.pkl into "
-                f"{OUT_DIR}/ from a previous GPU run."
-            )
+        print(f"P10 (RESULTS 98.15 ruling 4): MolT5 captions on {device}")
         from transformers import T5Tokenizer, T5ForConditionalGeneration
 
         print("Loading MolT5 (laituan245/molt5-large-smiles2caption)...")
@@ -156,12 +151,18 @@ def main():
     # 5. Dose embedding
     #    Exact same as upstream preprocessing.ipynb cell-11, cell-12
     # ----------------------------------------------------------------
-    dose_prompt = "The dosage is 10.0 micromoles."
+    import json
+    with open(DOSE_LIST_PATH, "r", encoding="utf-8") as f:
+        p10_doses = json.load(f)
+    dose_emb = {}
     with torch.no_grad():
-        inputs = tokenizer_bert(dose_prompt, return_tensors="pt").to(bert_device)
-        outputs = model_bert(**inputs)
-        dose_emb = outputs.last_hidden_state.detach().squeeze(0).cpu()
-    print(f"Dose embedding shape: {dose_emb.shape} for prompt: \"{dose_prompt}\"")
+        for d in p10_doses:
+            d = float(d)
+            dose_prompt = f"The dosage is {d} micromoles."
+            inputs = tokenizer_bert(dose_prompt, return_tensors="pt").to(bert_device)
+            outputs = model_bert(**inputs)
+            dose_emb[d] = outputs.last_hidden_state.detach().squeeze(0).cpu()
+            print(f"Dose embedding shape: {dose_emb[d].shape} for prompt: \"{dose_prompt}\"")
 
     # ----------------------------------------------------------------
     # 6. Save
```

## ASKS
1. **Is the generated script exactly 98.16 C1** (two code changes and a header)? In particular:
   - the dose loop keeps their three tokenizer / model / `last_hidden_state` lines;
   - each prompt is `f"The dosage is {d} micromoles."` on the float from `dose_list.json`.
2. **The compact index** (0..1969, in sorted xpert order) is forced by their sanity check (`range(len(idx2smi))`). Is that
   handled on the data side without loss? The GPU kernel maps `pert_idx` through the same `compound_map.json`, sha1-pinned in both.
3. **The slices** are round-robin by descending SMILES length, about 394 each. At the probe's 35–72 s per SMILES that is about
   6–8 h, under the 41,400 s timeout (Kaggle's cap is 12 h).
   - Is the timeout handling right (fatal; nothing persisted)?
   - Should a caption kernel also checkpoint its cache to `/kaggle/working` partway, so a timeout isn't a total loss? Their own
     checkpoint fires every 500, which is never reached inside a slice.
4. **The guards' order:** pip, then the HF revision, then the mounted-file sha1s, then the run, then the weight sha256s.
   - Is a weight guard after loading acceptable? A mismatch is still fatal before anything persists.
   - Or should the kernel pre-download with `snapshot_download(revision=pin)` and verify before their script loads?
5. **Anything that could make the merged embeddings differ from what their pipeline would produce** on the same SMILES on a GPU,
   beyond the disclosed CPU beam ties.
