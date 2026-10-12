# PACKET 078 — GATE DECISION: D (PertDiT) is "not run (adapter)" (RESULTS §98.14); plus M+dt reported (§98.13)
packet_id: 078
created: 2026-10-12
repo_commit: (this commit)
type: **GATE READING** (from code only; no GPU spent) + **RECORD**

## D (§98.14)
- **The gate** (98.2 / 98.6 C6): a data adapter alone, the embedding precompute included, and timing ≤ 6 GPU-h.
- **Two failures, both read from Bai et al.'s code at 1d750cd** (vendored under `external/bai_benchmark_1d750cd/`):
  1. **The target space.** `train_pertdit.py:146–159` transforms X and X_ctl (`clip` → `normalize_total` → `log1p`). It trains
     and samples there and saves Δ in that space (`:437–444`), with no inverse. Our estimand is raw Δ, and an inverse needs
     each row's true raw total (test truth).
  2. **The embeddings.** `prepare_pertdit_embeddings.py` needs MolT5-large + BioLinkBERT-large (≈ 4 GB of HuggingFace
     weights) and embeds one dose (10 µM). Our bundle is multi-dose, and per-dose prompts aren't in their pipeline.
- **Reported, not tested:** in Bai et al.'s unified outputs, PertDiT's PCC_DEG is on the transformed target, the others' on raw
  Δ.

## M+dt (§98.13), reported only
- **Clean:** −0.0050 [−0.0096, +0.0001], 92 / 346, NO CLAIM. **Distinct-control:** +0.042. **Row-pooled:** 0.638.
- It shows M's pattern stratum for stratum.

## ASKS
1. Is "not run (adapter)" the right mechanical reading of the gate? Is either failure mis-read from their code?
2. Is the note on their benchmark (PertDiT scored on a transformed target) a fair reading of the code, worded as reported?
3. **Next week's quota after O9 session 3 and B1** (≈ 15 of 30 h): would a registered **fold-2 replication** be worth ≈ 6.7
   GPU-h? It would be v9 against M on `split_cold_drug_2`, with the distinct-control stratum as the primary unit, plus IDEAS B3's
   control-substitution M (CPU). Or should the remaining quota stay for O9's later sessions? (A question for planning; no
   registration yet.)
