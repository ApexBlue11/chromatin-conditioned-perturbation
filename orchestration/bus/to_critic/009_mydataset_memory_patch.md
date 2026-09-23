# PACKET 009 — a change to one line of XPert's executed code, for go / no-go
packet_id: 009
created: 2026-09-23
repo_commit: (see 009_gitstat.txt)
type: **PRE-SPEND, DEVIATION REVIEW.** GPU hours this session: 6.61. This packet commits none.

Launches v2 and v3 of the kernel you approved in 008 failed. v2 was an invocation error; v3 is a memory limit
that the published recipe cannot meet on this hardware. Fixing v3 requires changing **one line of their executed
code**. It is the first deviation of that kind, so I am asking before launching.

## WHAT HAPPENED

**v2** — `FileNotFoundError: configs/config.yaml`. Their `--config` defaults to `config`; the release has no such
file. Their `scripts/train.sh:15` for `l1000_mdmt` is:
```
python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol
       --nfold split_cold_drug_1,split_cold_cell_1,split_1 --dataset l1000_mdmt
       --use_gradscaler True --include_cell_idx True
```
The kernel had used argparse defaults and differed in three flags. Our own harness, `xpert_native_eval.py:21`,
documents `--include_cell_idx True` as non-default and required by the released checkpoint's architecture.
v3 uses train.sh:15 with the fold list reduced to `split_cold_cell_1` (`train_xpert.py:425-455` builds a fresh
model, `init_weights()` and optimizer per fold; the seed is set once at `:402`, before that loop).

Two guards were added before v3:
- **GUARD E** parses the trainer's printed `---------args-----------` block and terminates it on any mismatch
  with the published flags. Replayed on v2's real log it flags exactly `config`, `use_gradscaler`,
  `include_cell_idx`.
- **GUARD F** calls their `arg_parse()`, their `load_dataloader()`, and `XPertNet` on one real batch with one
  forward pass, in the trainer's environment, and asserts `len(val.dataset) == len(test.dataset)`.

**v3** — GUARDs A–E passed. GUARD F failed. The probe's stderr ends in their tqdm over the 21,321-row test set:
~2,000 rows/s to 86 %, then 29 and 19 rows/s, a stall of over a minute at row 18,528, then process death with no
Python traceback. GUARD F did not record the return code.

## THE MECHANISM, MEASURED
`datasets/MyDataset.py`, `load_data`, executed once per row:
```python
drug_feat = tensor(drug_feat, dtype=torch.float32) if self.args.drug_feat != 'smi' else drug_feat
```
Measured by building their `MyDataset` on 400 real test rows of `split_cold_cell_1` with our harness's loaders:

| | |
|---|---|
| bytes per row | 268 KB |
| of which the drug block, (122, 514) float32 | 245 KB |
| distinct drugs in the sample | 58 |
| drugs whose rows hold separate copies | 58 |

Rows built by their loader on this fold: train 47,509 + val 21,321 + test 21,321 (val is the test set under
`utils.py:133`, built as a second `MyDataset`) = 90,151. At 268 KB/row: **24.7 GB**. The image reports ~29 GB; the
dense unimol array their loader `np.load`s is 2.24 GB.

## THE PROPOSED CHANGE
Replace that one line with:
```python
if self.args.drug_feat != 'smi':
    _k = pert_id if self.args.dataset == 'transigen_sdst' else pert_idx
    _c = self.__dict__.setdefault('_drug_tensor_cache', {})
    if _k not in _c:
        _c[_k] = tensor(drug_feat, dtype=torch.float32)
    drug_feat = _c[_k]
```
i.e. the same `tensor(...)` call, made once per drug and reused for every row of that drug.

Applied at staging to the **copy** in `/kaggle/working`; the uploaded dataset keeps their file verbatim. The kernel
refuses unless the target line occurs exactly once, and records sha1 before and after.

## THE PROOF
`model/v9/prove_mydataset_patch.py`: loads their `MyDataset.py` verbatim and the patched copy as two modules,
builds both on the same 600 test rows, and asserts for every item and every field: `torch.is_tensor` on both,
equal dtype, equal shape, `torch.equal`; non-tensor fields `==`.
```
rows 600 | tensors compared 6000 | ALL torch.equal, dtypes and shapes identical
unique storage: original 164.6 MB, patched 33.4 MB (4.9x less)
projected for train + val + test (90151 rows): original 24.7 GB, patched 5.52 GB
PROOF PASSED
```
The patched projection double-counts shared drug storage, so 5.52 GB is an upper bound.

The kernel's `_ORIG` / `_PATCH` strings and the proof's `ORIG` / `PATCH` were compared as AST literals: identical.

Code checked for anything that could observe the sharing: `MyDataset.__getitem__` returns `self.data[index]`
unmodified; there is no custom `collate_fn`, so the default collate stacks per-row tensors into a new batch
tensor; a search of their source for in-place operations on `drug_feat` (`+=`, `-=`, `*=`, `/=`, `add_`, `mul_`,
`copy_`, `fill_`, `zero_`, `clamp_`, item assignment) returns nothing. In the model, `get_unimol_drug_feat`
slices and casts the **batch** tensor.

## ALSO CHANGED IN v4
GUARD F now writes its return code, `killed_by_signal`, and the stderr tail with progress bars removed into
`run_record.json`. The v3 kernel log came back empty from the Kaggle CLI on Windows (`'charmap' codec can't
encode`, tqdm block characters) and was recovered with `PYTHONUTF8=1`.

## PRIOR RETRACTIONS AND ERRORS IN SCOPE
- RESULTS 71.9 / 71.10: v1 and v2 of this kernel; v2's command was built from argparse defaults.
- RESULTS 70.3(a): "padded keys are identical" — false for both of us.
- RESULTS 69.2: packet 007 asked for a decision the repo had already made.

## ASKS
1. **Go or no-go on the memory change.** Is replacing a per-row `torch.tensor` copy with a per-drug shared
   tensor, with values proven identical on 6,000 tensors, a deviation that bears on "XPert as published" — or
   is it in the same class as the empty `__init__.py` files, a change to resource use that does not alter the
   computation?
2. Is 600 rows an adequate sample for the equivalence proof, given the patch's only data-dependent branch is the
   cache key (`pert_idx`)? The sample contains 58 distinct drugs.
3. Is there a remaining memory hazard I have not addressed? In particular their `DataLoader`s use
   `num_workers=10` with fork on a list of ~90k tuples of tensors; object-header pages are copied on write as
   workers touch refcounts, but tensor storage is not.
4. Anything else.
