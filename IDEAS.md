# IDEAS — the tracked backlog

Every idea this project has taken from the literature or from its own measurements, with where it came
from, whether the source claim was **verified by the PI** or only **reported by a worker**, and what it
would cost. Nothing is deleted from this file; items move to `status: DONE` with a pointer to the RESULTS
section that settled them, or `status: KILLED` with the measurement that killed them.

**Why this file exists.** Ideas were accumulating inside numbered RESULTS sections written for other
purposes and becoming invisible. Asked directly whether the combined STRING+Reactome graph had ever been
tried, the answer was buried in §48.3 item 1 and the answer was *no*.

Direction set by the principal, 2026-09-21: **pursue the mechanism/MoA line (A-group) while keeping the
load-bearing dissection (B-group) and the cold-cell head-to-head (C-group) alive.** Do not drop B and C.

---

## A. Make chromatin and the drug atoms earn their place — the mechanism line

### A1. 🔵 Chromatin-modulated edges on a unioned biological graph — NOT TRIED, and the strongest novelty claim available
**Status: NOT TRIED. Nothing like it has been built.** This is the principal's idea, raised 2026-09-21,
and it is a step beyond what §48.3 queued.

What v9 does today, confirmed by reading the code rather than from memory:
- **STRING** enters as `PPIMessagePassing(ppi, d)` over one [978,978] adjacency, added residually
  (`model_v9.py:99,145`).
- **Reactome** enters somewhere else entirely: as `M_norm` in the **auxiliary loss**
  (`model_v9.py:180-186`), and in the v5/v6 lineage as a pathway bottleneck.
- **They are never combined into one graph.**
- **Chromatin gates GENE tokens** (EpiGate), never edges.

Why the combination is the interesting version: §48.1 verified against arXiv:2505.14919 that TxPert's graph
is **static and cell-independent** — *"each perturbation p is associated with a randomly initialized input
node embedding … learned via backpropagation"*. So **no published model makes the biological graph a
function of the cell's chromatin state.** Making edge weights depend on accessibility/activation/repression
at both endpoints gives chromatin a **different job** from the one that measured null: not "how much does
this gene move" (§55: +0.000360, CI [−0.005433, +0.006153]) but **"which edges conduct in this cell"** —
i.e. which pathways are available to propagate a perturbation at all.

That is a mechanistically distinct hypothesis, not a retry of the dead one. It also directly serves the
MoA objective: a cell-specific conductive subgraph is an *interpretable object* — you can name the
pathways it opens and closes.

- **Cost:** one training run (~5.8 GPU-h) once the graph is built; graph construction is CPU-only and free.
- **Blocked on:** A2 (the union graph must exist first).
- **Null it must beat:** the same model with chromatin mean-ablated in the edge gate only, plus the
  existing gene-level EpiGate left untouched — so the arm isolates edge gating from gene gating.
- **Pre-register before spending:** method rules 13, 16, 17 all apply. Null key and estimand fixed first.

### A2. Union the four graphs with multi-hot provenance edge features — QUEUED since §48.3, never run
**Source:** TxPert, §48.1, **verified by PI** against arXiv full text: *"Exphormer-MG … adapted for
multi-graph learning via a union graph methodology"*, edges carry a **multi-hot feature** encoding which
source graph each came from. Their ablation: combining helps monotonically, all four beats the best three
(**p < 0.027**). ⚠️ Per-graph numbers are **figure-panel only** — not extractable from text or tables, worker
checked the LaTeX source.

We hold **STRING, Reactome, GO:BP and DTI** and have only ever used them separately or not at all.

- ⚠️ **Explicit trade to decide, not to make silently:** `dti_reference.tsv` (19,174 edges, 1,718 drugs) is
  currently a **held-out validation set**. Putting DTI in the graph spends it. Three-graph union avoids this.
- **Cost:** CPU-only to build. Free.

### A3. Chromatin direction/sign head — "the one form of the question this project has never put"
**Source:** §49.4, from Agrawal et al. (F1000Research), the closest flank found by the adversarial novelty
sweep. They reach **AUC 0.71–0.89** predicting **direction of change** on **extreme responders only** from
H3K27ac. We predicted **continuous magnitude for every gene** and got +0.004.

§49.4's reading, which still stands: chromatin may carry **which genes move and which way**, not **how
much**; and the signal may live in the **tails**, which the reproducible-stratum filter keeps but a per-gene
regression loss dilutes across all 978.

- **Build:** auxiliary direction/sign head conditioned on chromatin, scored as **AUC on top-k up vs top-k
  down genes**. Null: the same head with chromatin mean-ablated.
- **Cost:** one run, and it can ride along as an auxiliary head on any other arm — near-free if bundled.
- **Independent of** the token-vs-additive question, so it does not collide with A1/A2.

### A4. Dose and time as tokens INSIDE the drug stream — the one way atoms might yet matter
**Source:** §47.4, **verified by PI** in XPert's code. Their drug sequence is
`[dose, time, HG_embed, atom_1..atom_n]` (`unimol_Embeddings`, `model_utils.py:133`). Ours applies dose/time
as a **FiLM scale/shift on the gene tokens**, far from the atoms — same information, **structurally unable
to express the interaction** "this atom matters more at this exposure".

Relevant now: §61.7 established that atom tokens *hurt* even with self-attention. A4 is a different reason
they might be useful, and it is the only untested one that changes what the atoms can express.

- §47.4's own instruction: **test separately, do not bundle with 47.3.**
- **Cost:** one training run (~5.8 GPU-h).

### A5. Tail-weighted / top-k loss
**Source:** derived from §49.4's diagnosis, **not** from a specific paper — no citation yet, so it is an
idea, not a finding. If chromatin's signal lives in the responders, a loss averaging over all 978 genes
dilutes it. A magnitude-invariant correlation loss was tried once in a gentle finetune and failed to lift
correlation (RESULTS line 96) — that is evidence about *that* loss in *that* setting, not about tail
weighting.
- **Cost:** free to implement, one run to test. Cheapest paired with A3.

### A6. Token-vs-additive ablation for auxiliary biology
**Source:** §48.4, where a worker's novelty claim was **REJECTED** and the defensible residue recorded:
TxPert and State supply auxiliary biology as an **additive embedding**, PertAdapt as a **static attention
mask**, and **none of them ablates tokens-versus-additive.** That is a gap in the evidence, not an
established novelty — §48.4 is explicit that the worker's stronger claim is not accepted.
- ⚠️ State and PertAdapt claims are **REPORTED-NOT-VERIFIED** and must be checked before being cited.
- **Cost:** two runs, or one run with a flag.

### A7. 🔵 "New startup mechanisms" — raised by the principal, no recorded source
The principal listed this alongside new loss functions on 2026-09-21. **There is no item in RESULTS it maps
onto**, so it is recorded here unattributed rather than invented. v9 currently uses a **WSD schedule** with
`warmup_frac 0.03`, AdamW, `decay_frac 0.2`.
- **Next step:** say which mechanism was meant, or run a literature pass. Not actionable as written.

---

## B. What is load-bearing — the dissection line (keep alive)

### B1. Bai et al.'s retrain-from-scratch drug ablation, on v9
**Source:** §50, Bai/Prince/Nitschke bioRxiv 2026-05-15, **verified by PI against the preprint full text**,
which also caught a worker error (§50.5). Their finding: a **drug-free MLP scores 0.637 against XPert's
0.633**, and zeroing drug features at inference moves ΔPCC_DEG by at most 0.012.
The distinction that saved CLAIMS 1.9 and 5.1: they **retrain from scratch** drug-blind; we ablated at
**inference**. Those are different experiments.
- **Now the strongest form of "the drug branch does not earn its place"**, given §61.7.
- **Open question they never asked:** whether drug features matter for unseen-**cell** generalisation.
- **Cost:** ~5.8 GPU-h for the v9 drug-blind retrain; their drug-free MLP baseline on our splits is cheap.

### B2. Remove the control ENCODER on cold-cell
**Source:** §48.3 item 2, TxPert, **verified by PI**: *"no basal state encoder is by far the most effective
option"* for cross-cell-line transfer, with ŷ = x + gφ(Σ z_p). v9 runs **two** control encoders
(`ctl_enc`, `cell_enc`) mixed into the gene tokens. We already anchor absolute as control + delta (§43.1),
so we are half-way; the untested half is whether encoding the basal state **at all** hurts on unseen cells.
- **Cost:** cheap ablation, directly on the split that matters.

---

## C. Cold-cell head-to-head — the competitor line (keep alive)

### C1. Train XPert, TranSiGen and others ourselves on `split_cold_cell_1..5`
**Source:** §46.5. Published reference points (§46, **verified by PI** in XPert's Supplementary Table R8):
XPert warm **0.688 ± 0.011**, cold-drug **0.645 ± 0.008**, cold-cell **0.383 ± 0.027**; TranSiGen cold-cell
**0.293 ± 0.017**. Cold-cell is where the whole field is weakest and where §45.0 puts us **+0.178 over
ridge**. The same machinery admits TranSiGen, PRnet, DeepCE and CIGER.
- **Cost:** one training run per competitor per fold. The expensive item on this list; scope before buying.

---

## D. Open verification debts
- **MultiFlow's perturbation-type detail** — still needs the full text (§49.5).
- **State and PertAdapt** claims from W2 — REPORTED-NOT-VERIFIED (§48).
- **TxPert per-graph ablation numbers** — figure-panel only, not extractable (§48.3).

---

## Settled, kept so they are not re-proposed
| idea | outcome | where |
|---|---|---|
| Chromatin conditioning as the core contribution | **KILLED.** +0.000360, CI [−0.005433, +0.006153], 2/5 cells positive, sign test p=1.000 | §54, §55 |
| Sparse-vs-dense attention as XPert's advantage | **KILLED** — dead code in their repo; the real difference is drug self-attention | §47.6 |
| In-loop drug self-attention rescues the atom tokens | **KILLED.** Atoms still hurt at −0.01814 [−0.02635, −0.00916], 66 % of rows, with SA trained and present | §60.7, §61.7 |
| The 2x2 interaction as evidence of mechanism | **RETRACTED** — the pre-committed estimand failed its own null key | §60.9, §61.1 |
| Atom→gene attention recovers drug targets | **FALSIFIED** — median target rank percentile 0.560, worse than chance | v5 era, README |
| Auxiliary biology as attended tokens is "completely novel" | **REJECTED** as a worker overclaim; the defensible residue is A6 | §48.4 |
