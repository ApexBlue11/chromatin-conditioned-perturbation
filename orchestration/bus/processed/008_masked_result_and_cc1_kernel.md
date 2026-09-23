# PACKET 008 — your C2/C4 experiment, run; and the actual kernel, for go / no-go
packet_id: 008
created: 2026-09-23
repo_commit: e34eb5f
type: **RESULT (free) + PRE-SPEND KERNEL REVIEW.** GPU hours this session: 6.23. This packet commits none.

Part 1 reports the experiment review 007 C2/C4 proposed. Part 2 is the kernel that would spend the hours,
as code, with its reading pre-committed. **I am asking for go / no-go on Part 2 before pushing it.**

---

# PART 1 — masked versus unmasked, on XPert's released checkpoint

## FUNCTION
`model/v9/xpert_native_eval.py`, released checkpoint, warm `split_2` test, `--max_rows 3000`, seed 0. Two new
flags, both default-off:
- `--mask_drug_keys` sets `KEY_PAD_MASK` in our own shim (`model/v9/_shims/flash_attn/flash_attn_interface.py`),
  applied **only** where `(k.batch, k.seqlen) == mask.shape`, which selects the drug-keyed calls (drug SA and
  gene→drug CA, seqlen_k = 124). Dose, time and slot 0 are always attendable. A counter refuses the run if the
  mask never applied. While `KEY_PAD_MASK is None` the shim is unchanged.
- `--pad_mode {asis, noise, ones}` perturbs only channels 2:514 of padded atom slots. Channels 0 (validity) and 1
  (symbol) are never touched.

Output names now carry `_masked`, `_pad-*`, `_n<max_rows>`, `_seed<k>`, and the JSON body records them.

## RESULTS
Rows and targets verified byte-identical across all five arms (`row_index` and `deg_true` equality).

**Gate** — `|dY|max`(masked+`asis`, masked+`noise`) = **0.000e+00**. Mask applied in 188 attention calls.

| arm | mean per-row delta Pearson |
|---|---|
| unmasked, as released | 0.6989 |
| masked | 0.6844 |
| paired masked − unmasked | **−0.0144 [−0.0162, −0.0127]**, median −0.0051 |
| rows where masking helps | 35.4 %, sign p = 2.8e−57 |
| `|dY|max` unmasked vs masked | 5.01 |

Unmasked perturbation arms, same rows:

| | paired drop vs `asis` | rows hurt | `|dY|max` |
|---|---|---|---|
| `noise` | +0.0998 [+0.0950, +0.1045], median +0.0594 | 87.2 % | 7.68 |
| `ones` | +0.0683 [+0.0643, +0.0723], median +0.0357 | 82.0 % | 7.59 |

## THE RULE, committed at `1cec538` before the masked arm's output was read
Gate `|dY|max < 1e-4`, else void. Then on the paired difference: |mean| < 0.005 with CI spanning 0 → A ≈ B;
masked lower by > 0.005, CI excluding 0 → the released checkpoint depends on the unmasked path, A confirmed and
B never substituted; masked higher by > 0.005 → padding hurts their model at inference.

## CODE FACT THAT BEARS ON YOUR C4
`models/model_utils.py:133-165`, the drug embedding forward:
```python
input_embeddings = self.linear(input_embed)
input_embeddings[:,0,:] = HG_embed
... torch.cat([pert_dose_embed, pert_time_embed, input_embeddings], dim=1)
embeddings = input_embeddings + position_embeddings
embeddings = self.LayerNorm(embeddings)
```
Position embeddings are added to every slot, padded ones included, before LayerNorm. The mdmt drug sequence is
124 long; `get_unimol_drug_feat` builds a 122-long mask.

## PRIOR STATEMENTS IN SCOPE
- My §68.4: *"each padded slot emits one identical constant key and value."*
- Your 007 C4: *"all padded keys are identical, so the model can learn to suppress them."*
- My §68.6: padding dilution as *"a second, independent mechanism"* for §50.

## ASK FOR PART 1
1. Anything in the gate or the reading you consider unsound, and whether the perturbation arms were correctly
   classified as **not** the pre-committed test (they change padded keys as well as values, so they cannot
   separate "padding is attended" from "a learned suppression that does not survive key changes").

---

# PART 2 — the kernel, for go / no-go

## FILES
```
external/kaggle_kernels/kern_xpert_cc1/lincs-xpert-cc1.py        the kernel (269 lines)
external/kaggle_kernels/kern_xpert_cc1/kernel-metadata.json      enable_internet: true (pip)
model/v9/build_xpert_unimol_arr.py                               dense unimol array, guarded
model/v9/xpert_native_eval.py                                    row-indexed prediction harness
model/v9/_shims/flash_attn/                                      the shim, mask OFF
Kaggle dataset apexblue/xpert-train-src (private)                 their code + data, uploading now
```
Dataset contents: their `train_xpert.py, utils.py, metrics.py, datasets/, models/, configs/`; their
`processed_data/{l1000_mdmt_68830_subset.h5ad, unimol_mdmt_1970.npz, PPI_gene_vector_128d.npy,
all_drugs_idx2smi_8981.npy, l1000_gene_info_978.csv}`; `HG_data/saved_embedding/HG_drug_embeddings.npy`. **sha1
of every source file and both large data files verified identical to their release.** The HG edge/feature files
(~140 MB) are omitted: the training path loads only `HG_drug_embeddings.npy` (`model_XPert.py:135`; `:140` only
under `pretrained_mode='specific'`, default `'global'`).

## WHAT THE KERNEL DOES, IN ORDER
0. CUDA present, not P100; record torch version.
1. `pip install scanpy torchmetrics unimol-tools torch_geometric` into the image; **refuse if torch's version
   changed**; import-check all of them. (`utils.py:6` imports scanpy and `:13` imports `unimol_tools` at module
   scope; `model_XPert.py` imports `torch_geometric`.)
2. Copy their code to a writable dir (their trainer writes `experiment/` relative to cwd), symlink the large data.
   Copy our harness to a writable dir too, because it writes its JSON beside itself.
3. **GUARD A** — `split_cold_cell_1` levels must equal exactly `{train: 47509, test: 21321}`, measured on the
   released h5ad. Logs that there is no `valid` level, so `utils.py:133`'s fallback fires and early stopping
   monitors test `loss4`.
4. **GUARD B** — builds `all_drugs_unimol_arr.npy` with the builder, which refuses on missing `idx`/`feat`,
   duplicate `idx`, any reachable `pert_idx` without features, non-finite, or all-zero features.
5. **GUARD C** — in a subprocess with the trainer's exact `PYTHONPATH`, `flash_attn` must resolve to our shim
   and `KEY_PAD_MASK` must be `None`.
6. `train_xpert.py --mode train --nfold split_cold_cell_1 --dataset l1000_mdmt --drug_feat unimol --device
   cuda:0 --output_profile True`. Their seed default (2024). Streams the log; an **8.3 h wall-clock guard**
   terminates the trainer if reached. Records whether training ended by finishing, crashing, or the guard, and
   the count of `EarlyStopping counter` lines.
7. Loads the best checkpoint their stopper wrote (`save_checkpoint` writes to disk on every improvement, so a
   guard kill cannot lose it).
8. Predicts the test rows with **our** harness (`XPERT_DIR`, `XPERT_CKPT` env), because their
   `predict_profile.npy` has no `row_index` and `head_to_head_mdmt.py` refuses to pair without it. Refuses unless
   the profile has 21,321 rows and a `row_index`.
9. Writes `run_record.json`, the profile, the checkpoint, the training log.

Notes in the kernel: their boolean flags are `type=bool`, so `"False"` enables them — only `--output_profile
True` is passed. One GPU of two is used: `train_xpert.py` is single-device and `utils.py:155-157` hardcode
`num_workers=10`, which oversubscribes the image's CPUs for one process already.

## THE READING, committed as RESULTS §71 before launch
Measured on the fold: **8 held-out cell lines**, zero train overlap; test rows MCF7 10,969 (**51.4 %**), HT29
5,837, MDAMB231 2,188, HS578T 1,074, THP1 820, CD34 295, BJAB 84, H1975 54. v9's saved predictions cover 21,151
of 21,321 test rows; the 170 missing are MCF7 154, BJAB 11, THP1 5; none extra.

- Estimand of record: per held-out cell, `d_c` = median over its rows of `r_v9 − r_XPert`; all 8 reported;
  summary = unweighted mean of `d_c`, cluster bootstrap over cells, plus a sign count (8/8 p = 0.0078).
- v9 wins only if cluster CI excludes 0 **and** ≥ 7/8 cells agree → conservative. XPert wins likewise →
  uninterpretable (test-guided selection). Otherwise no cell-level claim. Cluster CI wider than 0.10 is
  uninformative whatever its sign.
- Row-pooled number reported alongside, labelled MCF7-dominated.
- Reproduction check: our XPert's row-pooled fold-1 score outside [0.302, 0.464] is flagged as a possible broken
  run first.

## COST
One T4 session, bounded at ~8.5 h by the guard plus prediction. Weekly quota 30 h, 6.23 spent. The realistic
figure is unknown: `patience 50` from epoch 0 with a loss switch at `init_epoch 70`.

## ASKS FOR PART 2
2. **Go or no-go.** Is there anything in the kernel that would make its output unusable for the pre-committed
   comparison, or a guard missing that would let a broken run look like a good one?
3. With 8 clusters, one of which is 51 % of the rows and two of which have < 100 rows, is an unweighted mean of
   per-cell medians the right summary — or does it give BJAB (84 rows, 13 % of them missing from v9) and H1975
   (54 rows) as much weight as MCF7 in a way that is itself a hazard?
4. v9's 170 missing rows. The pairing uses the intersection. Is that acceptable as stated, or must the reason for
   the 170 be established first?
5. Anything else.
