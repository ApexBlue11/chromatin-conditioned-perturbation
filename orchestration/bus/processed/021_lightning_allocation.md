# PACKET 021 — DESIGN: O2 stays on Kaggle (84.4's rule fails by bound); Lightning credits fund v9 development instead
packet_id: 021
created: 2026-09-25
repo_commit: ed63f4e
type: **DESIGN + a decision record.** Nothing has run on Lightning.

## 1. The 84.4 migration decision, taken without the probe
Credit balance read from the billing API: **B = $14.99** (project balance; free credits, no card). 84.4's rule: migrate
iff all guards pass and `C ≤ 0.7 × B = $10.49`, with `C` the full cost to the 297 horizon. After session 2's boundary
(~epoch 131) at most **165 epochs** remain. On 2×T4 DataParallel an epoch takes 440 s. One A100 ($2.19/h) is ~2–2.5× that
pair on fp16 compute and bandwidth, so ~180–200 s/epoch → **C ≈ $19** for training alone. Even at an implausible 4× the
pair (110 s/epoch), training is $11.0 before setup, the chain test and the final prediction. H200 ($4.50/h) and L40S
($2.14/h) give similar or worse totals. **No plausible epoch time satisfies the rule, so O2 stays on Kaggle.** I did not
spend the probe's ≤ $2 to confirm this. The decision uses only prices, balance and Kaggle-measured timing (84.1 (iii)),
and not probing can only prevent a migration, never cause one. The principal then chose to allocate the credits to v9.

## 2. Proposal: a Lightning block for §85's development runs
§85's dev comparisons are variant − baseline on the 6 dev cells. P2's baseline (seeds 0–2) is running on **Kaggle 2×T4**
now. A variant trained on an **A100** differs from it in hardware, kernels and rounding. That is a systematic
platform effect inside Δ, and it would not show up in the seed sd.

**Rule proposed:** every variant screened on Lightning is compared with a **Lightning baseline**, never with P2's Kaggle
baseline. The Lightning block opens with the baseline at seeds 0–2 on the same machine type (`L-base`). The platform
effect `baseline_L − baseline_K` is reported, not read. §85's advance/accept rules then apply unchanged within each
platform. `s0` may be pooled across platforms only if the two baselines' seed sds agree within a factor of 2, which is a
variance check fixed now. Machine: one A100 (Lambda `gpu_1x_a100_sxm4`, $2.19/h) if the free account can start it,
else the cheapest approved ≥ 40 GB GPU. Same code commit and the same `xpert_arm.py` flags as on Kaggle. No bf16 or
compile changes, so the only difference between platforms is hardware.

**Budget:** the first L-base seed doubles as the timing probe. Stop if one run costs more than $1.50. At ~$1 per run,
$14.99 buys L-base (3 runs) plus ~8–10 variant runs. The state and outputs of every run are copied off the machine when
it finishes (84.4 condition 4's logic). Every run goes in the dev ledger with its platform.

## ASKS
1. Is taking 84.4's NO without the probe acceptable, given the bound?
2. Is "Lightning variants against a Lightning baseline" sufficient, or should the platform effect gate anything?
3. Is the factor-of-2 rule for pooling `s0` across platforms right?
4. Anything else.
