# REVIEW OF PACKET 111 (stage-2 code, before any run; §103.7)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: e48c5d7

**The code implements §103.6 / §103.7 faithfully. 14 / 14 new tests pass on my run** (6 s).
- **No L1000 input from the held-out cell:** the inputs are gene vectors, CCLE landmarks and PCs, global chromatin, landmark
  and node chromatin with masks, and node expression. Targets alone come from L1000.
- **LOCO** uses `stage1.fold_targets` with training rows in the 9 CCLE cells and references from every knockdown cell.
- **Early stopping:** on 10 % of rows in LOCO, 10 % of training genes in cold-gene.
- **M3–M5 start from one shared vector at the knocked-down node,** so M5 gets no real-graph information.
- **The injection** hits its variance share over training rows, with residualised φ.
- **Scoring** excludes the own coordinate, with the reachable-landmark variant.

**But two things would waste the quota or break blindness, and both should be fixed before the sessions run.**

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | rung 3 for the graph family is unlearnable by construction (Ask 4) | **Every injection multiplies by ψ(g) ~ N(0, 1), a random number per gene** (`inject`: `base = phi[c] * psi[g]`).<br>• **M1 / M2** can learn ψ(g) in LOCO: each gene is seen in the training cells, and its 128-d gene vector is a unique key the MLP can memorise.<br>• **M3 / M4 / M5 have no gene-specific input** (review 110 C2's shared start vector). A gene's output is a function of its graph location and the cell's node features only. Its random sign and scale can't be represented, except by chance correlation with location.<br>• **So the graph family's recovery is ≈ 0 by design,** for the graph shape and the dense shape alike. The 72 M3 / M4 injection jobs (cost 10 each, the bulk of the injection budget) would read "undetectable" whatever the model can do. That leaves R-chrom-graph's null uninformative.<br>• **The tests can't catch this:** they check variance shares and K-hop locality, not whether each family can learn its own injected term. | **For the graph family, use an injection it can express:** u = φ(c, j) · [Â³e_g]_j with **ψ ≡ 1** (or ψ a fixed function of graph position), at the same variance share.<br>**Keep ψ(g) for the MLP family.**<br>**Add a learnability test:** on a toy graph, GraphProp trained on φ · Â³e_g recovers it (r above a set level), and fails on the ψ-random version. M1 recovers its dense injection.<br>**Optionally,** drop the graph family's dense injection (36 heavy jobs). It can only fail. |
| 2 | MAJOR | blindness of the real-target jobs (Ask 3) | **The runner prints every job's score to the kernel log:** `print(f'[gpu{gpu}] {tag} mean_r {r:.4f} …')` (`lincs-cellrep-s2.py` l. 37). `stage2.main` prints `run_job`'s `(tag, mean_r)` too.<br>• Real-target jobs share the sessions with the injection jobs.<br>• So downloading or tailing the logs (as the check-in does for errors) exposes the **real** LOCO and cold-gene scores before the injection reading is committed. | **Print only the tag, status and time for every job.** The scores stay in the per-job JSON.<br>**Make the real-target reader refuse to run** unless the committed injection reading exists (path and sha1 pinned), as review 101 did for H-DD's counts. |
| 3 | MINOR | details (Ask 4) | **(a)** The docstring says the graph shape is "/ SD". The code doesn't normalise, which is harmless because s rescales.<br>**(b)** Â³ counts walks of exactly 3 steps, so distance-2 landmarks get weight only through odd cycles (triangles). STRING has many, so this is fine. But M3's (I + GÂW)³ also spans Â, Â², Â³, so the injection is narrower than the model.<br>**(c)** ψ is drawn over each setting's sorted gene list, so a gene's ψ can differ between folds. Within a fold, every model sees the same values. | **(a)** Fix the docstring.<br>**(b)** State it, or use Σ_{k≤3} Âᵏe_g.<br>**(c)** Optionally key ψ by gene symbol (a hash) so it is fixed across folds. |

## Answers to the asks

**Ask 1 — faithful, except the graph family's injection** (C1).
- **The inputs:** no held-out-cell L1000. The shared start vector is as amended.
- **The injection:** scale and shapes are as registered.
- **The stopping sets:** rows in LOCO, genes in cold-gene.
- **Rung 4's usage variants:** they swap only the chromatin cell, at inference, and gate mean / SD are recorded for M3.

**Ask 2 — yes.** The models need CCLE features, so the 9 CCLE knockdown cells train, and the 6 non-CCLE cells (HA1E, NPC,
ASC, HEKTE, SKL, HEK293T) enter only as reference cells.
- **State the training cells' sizes:** HCC515 2,408, SW480 232, VCAP 28 at 96 h, next to the six core cells.

**Ask 3 — acceptable once C2 is fixed.**
- **The outputs** are per-job JSON and predictions, with a sha1-pinned COMPLETE marker. They can stay unopened.
- **The log lines** must not carry scores.

**Ask 4 — C1 and C2.**
- **Otherwise the runs look usable:**
  - the data loads once per worker;
  - `.FAILED` files hold the tracebacks;
  - batch sizes are 256 / 32;
  - patience is 3;
  - M2's chromatin term starts at 0 (s = 0; b gets gradient after the first step).

## What I checked and found sound

- **`stage2.py` in full:**
  - `Data` (features, PCs, chromatin masks, `loco` / `cold` / `d_reference`), `inner_split`;
  - `phi_matrix`, `graph_shape`, `inject`;
  - `norm_adj`, MLP1, MLP2, GraphProp (`cell_gates`, the forward pass, the pooled readout);
  - `build_model`, `batch`, `row_scores`, `predict`, `train`, `run_job`.
- **`stage2_jobs.py`'s job list:** 450 jobs. Injection jobs are all LOCO: M1 / M2 at 3 shares × 2 shapes plus raw ACT, and
  M3 / M4 at 5 % × 2 shapes. The real jobs include the K = 4 sensitivity runs.
- **The runner's logging and COMPLETE marker.**
- **`test_stage2.py`:** 14 passed (`CUDA_VISIBLE_DEVICES=-1`).

## What I could not assess, and why

- **Training behaviour on real data.** Nothing has trained there.
- **`stage2_graph_prep.py`'s rewiring beyond the reported figures** (degrees preserved, 3.5 % of edges kept). I read the
  summary, not the output graph.
