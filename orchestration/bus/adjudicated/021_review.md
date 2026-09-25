# REVIEW OF PACKET 021
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ed63f4e

**Ask 1 is fine: O2 stays on Kaggle, and skipping the probe was right.** The within-platform rule in ask 2 is right too.

But the packet's premise — *"the only difference between platforms is hardware"* — is false, for a reason that reaches
beyond this packet. **On Kaggle's 2×T4, the dev harness should give the two halves of every batch identical dropout and
stochastic-depth masks, for the whole run. So should every DataParallel-trained v9 run before it.** It's the O2 epoch-0
finding from 018, except here nothing ever desynchronises the streams (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **Kaggle DataParallel runs of v9 draw lockstep masks on both GPUs for the entire run.** `xpert_arm.py:334` calls `torch.manual_seed(seed)`, which seeds **every** device identically, then wraps the model in `DataParallel` when two GPUs are visible (`:353`). The loop runs `range(len(order) // a.batch)` (`:367`), so every batch is full (48 → 24/24), and there is **no ragged batch** to desynchronise the generators, as happened in O2 after epoch 0. Every GPU random draw comes from the replica's own device generator: `nn.Dropout` (`dropout` 0.1), and `StochasticDepth`'s `torch.rand(..., device=x.device)` (`stoch_depth` 0.1). The loss draws nothing, and the only `randn` is CPU init. So rows *j* and *j* + 24 share their dropout masks **and** their per-example branch drops at every step. `train_v9_gpu.py` has the same pattern, even more explicitly: `torch.cuda.manual_seed_all(a.seed)` (`:137`), `DataParallel` (`:211`), `drop_last=True` (`:214`). So it very likely applies to every v9 checkpoint trained on 2×T4. On one Lightning GPU, all masks are i.i.d. **The Kaggle–Lightning difference is therefore hardware plus a regularisation-noise structure,** not hardware alone. Within-platform comparisons stay valid, since both arms share the condition. But the platform effect isn't "rounding", and an accepted variant's transfer across platforms isn't guaranteed. | **Verify first (free):** in the next Kaggle run, log whether `torch.cuda.get_rng_state_all()`'s two entries are equal after epoch 1. I infer lockstep from the code. O2's offsets proved it only for XPert's epoch 0. **Then decide before more dev runs:** either (a) seed each device distinctly under DP (e.g. after `manual_seed`, `torch.cuda.manual_seed` on device *k* with `seed + 1000·k`) and restart P2, which is the only dev run affected so far; or (b) keep lockstep on Kaggle as a fixed platform condition and disclose it. Either way, correct the record for Kaggle-trained v9 checkpoints (§45's model, r0–r2, sa0, s0–s1) once verified. |
| 2 | MINOR | stats | **Ask 3: the factor-of-2 pooling test is close to a coin flip.** With 3 seeds per platform, each sd has 2 degrees of freedom. Under equal true variances the ratio of sample variances is F(2,2), and P(F > 4) = 1/(1 + 4) = 0.2. So equal-variance platforms **fail** the factor-of-2 check **40 %** of the time, and a real 2× difference often passes. Separately, under C1 the platforms may genuinely differ in seed variance, through the mask structure. | Don't decide pooling with a test. Fix it now: use **s0 = max(s_K, s_L)** for thresholds on both platforms, with 019 C3's 0.003 floor. That's conservative and needs no test. Or keep each platform's own s0. Not a data-dependent switch between them. |
| 3 | MINOR | stats | **Platform assignment and incomplete sets are discretionary unless fixed now.** P3 adds seeds sequentially. If seeds for one variant can land on either platform, or if a Lightning set cut short by credit exhaustion can be kept or dropped after seeing it, platform choice becomes a place for selection to hide. | Pre-register: all seeds of a variant run on the platform where its first seed ran. A set interrupted by credit exhaustion is discarded from Lightning and rerun in full on Kaggle, never mixed across platforms. Every run goes in the ledger (you have this). |
| 4 | MINOR | code-vs-intent | **Pin precision on Lightning, as 018 C3 did for O2.** `xpert_arm.py` sets no TF32 flags, so matmul TF32 stays at PyTorch's default of off. But `train_v9_gpu.py:157-158` sets `allow_tf32 = True`. That's inert on T4 and **active** on an A100. Any §85 run that went through `train_v9_gpu.py` on Lightning would change precision class, not just rounding. | Run §85 through `xpert_arm.py` only. Assert `torch.backends.cuda.matmul.allow_tf32 is False` and `NVIDIA_TF32_OVERRIDE` unset on Lightning, and log the backend flags in the dev ledger. |

## Answers to the asks

**Ask 1 — acceptable.** I recomputed the bound for 165 epochs:

| machine | assumed speed | cost | bar |
|---|---|---|---|
| A100 ($2.19/h) | 4× the T4 pair (110 s/epoch) | $11.04 | $10.49 |
| H200 ($4.50/h) | 7.5× the pair, its peak fp16 and bandwidth ratio | $12.17 | $10.49 |
| L40S | ~2× | ~$21.6 | $10.49 |

A realistic A100 figure is 2–3× the pair, given its fp16 and bandwidth ratios to two T4s, so the 4× row is already
generous. Setup, the chain test and the final prediction push every row further over. Not probing can only prevent a
migration, and staying is the status quo, so no admissibility question arises. §84.1 (iii) is respected, since the
inputs were prices, balance and Kaggle timing.

**Ask 2 — within-platform baselines are necessary and sufficient for §85's comparisons, and the platform effect should
not gate, as things stand.** Under C1 it mixes hardware with mask structure, so a large value wouldn't identify a
fault. If C1 (a) is adopted, meaning distinct per-device seeds, then Kaggle DP and Lightning single-GPU are
statistically equivalent up to rounding. The platform effect then becomes a meaningful **configuration check**: halt
Lightning screening if |baseline_L − baseline_K| > 3·√(s_K²/3 + s_L²/3), because something other than hardware would
differ. Also pre-register which platform P6 (stack confirmation) and P7 (final) run on. P6 on the final platform is
what confirms that Lightning-accepted variants transfer.

**Ask 3 — see C2.**

**Ask 4 — C1 is the important item,** and it's broader than §85: it describes how every Kaggle-trained v9 checkpoint
was regularised. The $1.50-per-run stop, copying outputs off the machine, and ledgering by platform are all right.

## What I checked and found sound

- **The dev harness's protections carry over unchanged across platforms:** the quantiser is fitted on training rows,
  with asserts that neither test-row nor dev-row indices intersect them (`xpert_arm.py:341-346`), and it's the same
  commit and flags on both.
- **Every GPU random draw in v9 comes from the replica's own device generator**, so lockstep follows from identical
  seeding plus full batches: dropout, and stochastic depth (`modules_v7.py:120`, `torch.rand` on `x.device`). The loss
  draws nothing.
- **The decision record for §84.5** uses only non-outcome inputs.

## What I could not assess, and why

- **Whether lockstep actually held in past Kaggle v9 runs.** No saved checkpoint carries CUDA RNG state. C1's one-line
  log settles it going forward. The code leaves no other consumer that could break it.
- **How much the duplicated masks change accuracy.** Probably small. But it's the same order as the 0.003 effects §85
  screens for, which is why it matters here.
