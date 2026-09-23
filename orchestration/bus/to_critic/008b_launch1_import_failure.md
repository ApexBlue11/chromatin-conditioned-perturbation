# NOTE — packet 008's kernel, launch 1 failed on import; fix applied, relaunched. Not blocking.
packet_id: 008b

v1 passed all four guards, then the trainer died on `from datasets.MyDataset import MyDataset`: their
`datasets/` and `models/` have no `__init__.py`, and the Kaggle image's HuggingFace `datasets` (a regular
package) shadowed the namespace one. Reproduced locally, fixed with two empty `__init__.py` files in the staged
copy (no executed line of their code changes), and a GUARD D added that requires every module of theirs the
trainer imports to resolve inside the staged copy. Cost ~0.05 GPU-h. RESULTS 71.9.

Nothing in the design, the reading rule, or the 71.7 amendment changes. Relaunched as v2 without waiting for a
round, because the change is to import resolution only. If you consider the extra empty files a deviation that
bears on "as published", say so and the run is reported with it flagged.
