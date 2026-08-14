# v6 — self-contained rebuild

Everything about this model lives here. Start with **[ARCHITECTURE.md](ARCHITECTURE.md)**: the task, the
provenance of every input tensor, the data flow, and the evidence behind each design decision.

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | **read first** — data sources, flow diagram, why each component exists, **what the pathway readout is and is NOT**, what v6 is expected NOT to fix, how it must be judged |
| `config_v6.py` | all hyperparameters + data paths, each non-obvious value annotated with the measurement justifying it |
| `modules_v6.py` | BaselineEncoder / ChromatinEncoder / ModalityFusion (late integration) + GeneHead (signed chromatin term) |
| `model_v6.py` | `LincsV6` — the full forward |
| `train_v6_tpu.py` | TPU v3-8 trainer — start it via `launch_v6_tpu.py`, never directly (`TPU_NOTES.md` #4) |
| `eval_v6.py` | accuracy on the reproducible stratum, protocol-matched to v5 + **ablate-to-mean** for all 7 components |
| `probe_pathways_v6.py` | pathway readout vs the **measured** response, with gene-permutation and cell-shuffle nulls |
| `test_v6.py` | 29 code-vs-design checks — **must pass before any accelerator time** |

`PathwayBottleneck` is imported from `../modules.py` (shared with v5 so the two can be compared directly).

## Quickstart
```bash
python model/v6/test_v6.py     # 29 checks, ~2 min, no data or accelerator needed
```
Evaluation is written and tested; it needs only a checkpoint. See `../HANDOFF.md` §2 for the two commands.

⚠️ **The pathway readout is drug-invariant** — it is computed before the drug enters the forward pass, so it
is a *cell × dose/time* readout and must never be scored against drug→target annotation. ARCHITECTURE.md §7.

## Why v6 exists
v5's interpretability claims were falsified by our own tests: atom→gene attention put known drug targets at
median rank percentile **0.560 — worse than chance**, and the pathway prior steered attention only **1.1×
random**. Two structural causes were identified and fixed:

1. **Hard connectivity mask, not a soft bias** (P-NET / DCell / DrugCell). 360 *named* Reactome pathway
   nodes: gene *g* reaches only pathways containing *g*, so `pathway_activations[:, p]` **is** a masked
   aggregation of pathway *p*'s members (verified, not assumed — test_v6).
   ⚠️ **But unlike P-NET this is a masked SIDE BRANCH, not a bottleneck** — `model_v6` adds it residually
   (`h = h + delta`), so the main stream *can* route around it. The readout is faithful; "the prediction
   had to pass through it" is **not** established. See ARCHITECTURE.md §7 and claim 4.13.
2. **Late modality integration, not early fusion** (MOLI / DeepCDR). Separate encoders for baseline
   expression and chromatin, each with its own normalisation, so both stay independently ablatable —
   v5 summed them into one token and chromatin entangled with baseline (corr 0.74).

Kept because it survived every test: the **signed additive chromatin head** (ΔR² +0.089, correct sign,
robust across all four baseline-expression quartiles).
Kept for accuracy with **no interpretability claim**: atom→gene cross-attention.
