# PACKET 023 — RESULT: the gradient × activation MoA probe (RESULTS 86) reads NULL
packet_id: 023
created: 2026-09-25
repo_commit: (the commit carrying this packet)
type: **RESULT.** 0 GPU-hours (Kaggle CPU).

You pre-registered this with me in 020 (amended by your review). The reading script was committed at `dfc8abc` before any
output was read; it fixes the one numeric form the rules left open: a stratum "shows it" iff diff ≤ −0.02 and p < 0.05
on all three seeds.

`model/v9/probe_moa_v9.py` on Kaggle CPU (kernels `lincs-moa-{r0,r1,r2,u0,u1,u2}`, checkpoints sha1-checked in-kernel);
read by `model/v9/read_moa_86.py`, committed at `dfc8abc` **before any output was read** (it fixes the one numeric form 86
left open: a stratum "shows it" iff diff ≤ −0.02 and p < 0.05 on all three seeds). 496 compounds scored (fold-0 test
rows with ≥ 1 ChEMBL mechanism target in a named node's full gene set); all 800 nodes matched a GMT term. 0 GPU-hours.

| seed | rho_raw / rho_del | S | diff vs Null 1 | p | Null 2 | diff − untrained | SIGNAL conditions | untrained diff |
|---|---|---|---|---|---|---|---|---|
| r0 | 0.040 / 0.170 | 0.3997 | -0.0186 | 0.008 | 0.4253 | -0.0162 | no | -0.0024 (p 0.393) |
| r1 | 0.060 / 0.167 | 0.4100 | -0.0203 | 0.003 | 0.4250 | -0.0439 | yes | +0.0235 (p 0.996) |
| r2 | 0.031 / 0.155 | 0.4269 | -0.0053 | 0.258 | 0.4286 | -0.0164 | no | +0.0111 (p 0.860) |

⇒ **Validity (86.2 step 4): the untrained controls read NULL — the probe is valid.** Gate G passes on every seed
(rho_del 0.155–0.170, far below 0.95: the drug reorders the gradient readout rather than scaling it). **Pre-committed
reading, 86.3: NULL** — the SIGNAL conditions hold on seed 1 only (seed 0 misses the −0.02 floor at −0.0186; seed 2
is −0.0053, p 0.26), and PARTIAL's second clause fails because seed 2's p is not < 0.05. **C8b — a trained
post-perturbation named pathway readout — enters §85's candidate list**, with its own pre-registered MoA test.

*Reported, not read:* the direction is favourable on every seed; the gradient readout's S sits well below the output
projection's and the data projection's (gradient vs output vs data, per seed: 0.400 vs 0.563 vs 0.552 / 0.410 vs 0.555 vs 0.552 / 0.427 vs 0.554 vs 0.552 — the projections rank target
pathways worse than their nulls' typical ~0.43); strata:

| seed | seen | unseen | targets not responsive | landmark-only tier |
|---|---|---|---|---|
| r0 | -0.0227 (p 0.006) | -0.0111 (p 0.225) | -0.0178 (p 0.018) | -0.0101 (p 0.171) |
| r1 | -0.0172 (p 0.037) | -0.0240 (p 0.052) | -0.0159 (p 0.022) | -0.0200 (p 0.055) |
| r2 | -0.0056 (p 0.275) | -0.0088 (p 0.249) | -0.0043 (p 0.335) | +0.0046 (p 0.641) |


## ASKS
1. Is NULL the correct mechanical reading (untrained NULL → valid; SIGNAL on 1 of 3; PARTIAL fails on seed 2's p)?
2. Was fixing the stratum criterion in the reading script, before reading, legitimate?
3. For C8b (a trained post-perturbation named readout), what must its MoA test pre-register beyond 86's design?
4. Anything else.
