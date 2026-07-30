# v6 — self-contained rebuild

Everything about this model lives here. Start with **[ARCHITECTURE.md](ARCHITECTURE.md)**: the task, the
provenance of every input tensor, the data flow, and the evidence behind each design decision.

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | **read first** — data sources, flow diagram, why each component exists, what v6 is expected NOT to fix, how it must be judged |
| `config_v6.py` | all hyperparameters + data paths, each non-obvious value annotated with the measurement justifying it |
| `modules_v6.py` | BaselineEncoder / ChromatinEncoder / ModalityFusion (late integration) + GeneHead (signed chromatin term) |
| `model_v6.py` | `LincsV6` — the full forward |
| `test_v6.py` | 16 code-vs-design checks — **must pass before any accelerator time** |

`PathwayBottleneck` is imported from `../modules.py` (shared with v5 so the two can be compared directly).

## Quickstart
```bash
python model/v6/test_v6.py     # 16 checks, ~30 s, no data or accelerator needed
```

## Why v6 exists
v5's interpretability claims were falsified by our own tests: atom→gene attention put known drug targets at
median rank percentile **0.560 — worse than chance**, and the pathway prior steered attention only **1.1×
random**. Two structural causes were identified and fixed:

1. **Hard connectivity mask, not a soft bias** (P-NET / DCell / DrugCell). 360 *named* Reactome pathway
   nodes; information cannot route around membership. `pathway_activations[:, p]` **is** pathway *p*.
2. **Late modality integration, not early fusion** (MOLI / DeepCDR). Separate encoders for baseline
   expression and chromatin, each with its own normalisation, so both stay independently ablatable —
   v5 summed them into one token and chromatin entangled with baseline (corr 0.74).

Kept because it survived every test: the **signed additive chromatin head** (ΔR² +0.089, correct sign,
robust across all four baseline-expression quartiles).
Kept for accuracy with **no interpretability claim**: atom→gene cross-attention.
