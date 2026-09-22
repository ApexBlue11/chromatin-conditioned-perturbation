# PACKET 007 — §46.5 feasibility, and a choice about what "XPert as published" means
packet_id: 007
created: 2026-09-22
repo_commit: 726a5a2
type: **PRE-SPEND.** GPU hours this session: 6.23. This packet commits none. Everything in it cost zero.

Review 006 C5 ranked §46.5 first and I accepted (§66.8). Before buying an hour I audited whether their
code runs here. It does, but not at the cost I quoted, and not without a declared deviation. **The
deviation is a choice between two different models, and I do not want to make it alone.**

## OBJECTIVE
Train XPert ourselves on `split_cold_cell_1..5` and compare against v9 on identical held-out rows. The
project objective in `state.json` is whether v9 generalises to unseen cell lines better than published
SOTA; review 001 C8 left that inadmissible because v9's 0.4734 is fold 1 / one seed, XPert's published
0.383 ± 0.027 is a five-fold mean, and **no XPert run exists on any cold-cell fold.**

## FUNCTION — what their code does, with line numbers
All paths relative to `external/xpert/code/XPert/`.

**Folds.** `train_xpert.py:27` takes `--nfold` (help: `'split, split_cold_drug, split_cold_cell'`);
`:423` splits it on commas. `processed_data/l1000_mdmt_68830_subset.h5ad` carries `split_cold_cell_1..5`,
`split_cold_drug_1..5` and `split_1..5` as native `obs` columns — the same splits v9 was evaluated on.
`head_to_head_mdmt.py` refuses to pair unless `row_index` matches exactly.

**Training length.** `configs/config_l1000.yaml`: `num_epochs: 2500`, `init_epoch: 70`, `patience: 50`,
`batch_size: 128`, `train_lr: 0.004`, `hidden_size: 256`, `trt_structure: CA+SA+SA+CA`.

**Attention kernel.** `models/model_utils.py:8` imports `flash_attn_func` unconditionally; it is called at
`:226` (SelfAttention) and `:281` (CrossAttention). The branch that selects it is at `:200`:
```python
if output_attention:
    ... explicit matmul + softmax, applies attention_mask ...
else:
    context = flash_attn_func(query, key, value, dropout_p=self.dropout_p if self.training else 0.0)
```
`train_xpert.py:52` defaults `--output_attention` to `False`.

**The mask.** `SelfAttention.forward(hidden_states, attention_mask=None, sparse_flag=False,
output_attention=False)` references `attention_mask` **only inside the `if output_attention:` branch**.
`CrossAttention` at `:281` likewise. The mask is constructed and passed: `model_XPert.py:204` calls
`get_unimol_drug_feat`, which returns `atom_musk = (1.0 - atom_musk_raw) * -10000.0`, and `:225` passes it
as `drug_attention_mask`. `cell_attention_mask` is passed as `None` at `:225`.

`sparse_flag` is threaded through `:186`, `:251`, `:342`, `:361` and **is never branched on anywhere** in
the file.

**Projections.** `self.query/key/value = nn.Linear(hidden_size, hidden_size)` at `:179-181` and
`:244-246` — default `bias=True`.

**Missing asset.** `config_l1000.yaml` sets `drug_unimol_path: processed_data/all_drugs_unimol_arr.npy`,
which is not in the released assets. What is released is `processed_data/unimol_mdmt_1970.npz` with `idx`
(1970,) and `feat` (1970, 122, 514). The loader does `self.drug_feat[pert_idx]`
(`datasets/MyDataset.py:146`).

## RESULTS — measurements, all on their own released data

`processed_data/unimol_mdmt_1970.npz`, shape (1970, 122, 514), channel 0 = validity, channel 1 = symbol,
channels 2:514 = the 512-d atom features:

| | |
|---|---|
| valid atoms per drug | min **5**, max 122, mean **53.9** of 122 slots |
| padded slots | 134,172 = **55.8 %** of all slots |
| padded atom features | **exactly zero**, `|max| = 0` |
| padded symbols | exactly zero |
| valid atom features | `|max| = 7.437`, `mean|·| = 0.890` |

Share of softmax mass taken by padded slots **if all scores were equal**:

| valid atoms | 5 | 27 | 53.9 (mean) | 80 | 122 |
|---|---|---|---|---|---|
| padded mass | 95.9 % | 77.9 % | 55.8 % | 34.4 % | 0 % |

`model/v9/build_xpert_unimol_arr.py --verify_only`, run against their h5ad:
```
npz: idx (1970,) in [1, 8949], feat (1970, 122, 514) float32
h5ad: 1970 unique pert_idx in [1, 8949]; 0 not covered by the npz
valid atoms/drug: min 5 max 122 mean 53.9 of 122 slots | padded slots 55.8% | padded features all zero: True
dense array would be (8950, 122, 514) float32 = 2.24 GB
VERIFY_ONLY: all guards passed, nothing written.
```
So the dense array their loader wants is derivable, every reachable `pert_idx` is covered, and the
unreachable rows are zero. It is 2.24 GB against the 494 MB npz it comes from, so the plan is to rebuild it
inside the kernel rather than upload the expansion.

## WHAT WAS CONTROLLED
- Every statement above is from the released source at `726a5a2` with line numbers, or measured from the
  released `.npz` / `.h5ad`.
- The reachability guard is checked against the actual h5ad at build time, not asserted from this packet.
- The builder refuses on: a missing `idx`/`feat`, duplicate `idx`, any reachable `pert_idx` without
  features, non-finite features, or an all-zero feature array.

## PRIOR RETRACTIONS IN SCOPE
- §36.3: I claimed XPert's numbers were "not extractable". They were in Supplementary Table R8.
- §46: three claims of mine about XPert retracted after investigation; in each case *we* had erred.
- §47.6: the sparse-vs-dense reading was wrong; the real difference was drug self-attention.
- §68.3, this packet: my first reading was that `flash_attn` is dead code under `sparse_flag: False`, by
  analogy with §47.6. **That was wrong.** The branch is on `output_attention`, not `sparse_flag`. Caught
  before acting on it.
- §32: a quantiser reported `fitted == 1.0` while its bins were NaN and every value bucketed to 0.

## WHAT I EXPLICITLY DO NOT CLAIM
- I **cannot** verify what the **released checkpoint** was trained with. The above is what *this code* does
  with *these defaults*.
- I have **not run `flash_attn`** — it is not installed — so I have not observed its behaviour directly.
  Its signature takes no padding mask (variable-length masking is `flash_attn_varlen_func`), but that is
  documentation, not a measurement I made.
- A padded slot's **value** is also the learned bias, so a constant additive term might be absorbed
  harmlessly. I can construct that argument; I cannot test it without training.

## ASKS
1. **The choice, and the one I most want independent.** Training XPert here requires replacing
   `flash_attn_func`. Three options, and they are not the same experiment:

   | | what it is | deviation |
   |---|---|---|
   | **A** | `F.scaled_dot_product_attention`, **no mask** | faithful to their training path as coded; kernel swap only |
   | **B** | SDPA **with** their additive mask applied | faithful to their evident intent; kernel swap **and** a behaviour change |
   | **C** | install `flash_attn` on Kaggle | no deviation; build risk, long install, may fail on the image |

   v9 masks padding properly (its tests assert a padded token moves a real token by exactly 0.00e+00), so
   **B compares two masked models and A compares a masked model against an unmasked one.** Which is the
   benchmark? Or must both be run, making this ~2x the cost?

2. **Is my reading of the mask omission correct at all**, or is there a path I have missed by which
   padding is suppressed in the flash branch? This is the fourth time I have read this codebase and
   concluded something unflattering about it, and the previous three were my errors.

3. **The epoch cap.** `num_epochs: 2500`, `patience: 50` makes runtime unbounded, so my §66.8 estimate of
   5.8 h does not transfer — that was a v9-shaped figure. Any cap we impose is a deviation from their
   recipe. What cap, and what would make a capped run still admissible for the paired comparison? Is
   "train to their own early-stopping criterion or 9 h, whichever first" defensible, given a run cut off
   mid-schedule is not comparable across folds?

4. **Scope.** One fold or five? Their published 0.383 ± 0.027 is a five-fold mean. A single-fold XPert
   against a single-fold v9 is a paired comparison on identical rows but reproduces neither published
   number. Does the admissibility argument in your C5 require all five folds — and if so, what does that
   do to the ranking against §50.5?

5. Anything in the builder's guards you consider insufficient, and anything else.
