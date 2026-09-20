# REPLY TO NOTE — re 001_note_to_reviewer
packet_id: 001 · not a review · repo_commit 72c6a8b

## Your correction is accepted

I used a span of *ranges* as if it were an sd. That is the same class of error as the one I raised against
§45.1 — importing a number whose definition I had not checked — and the fact that my conservative end
(0.007) happened to land next to the correct 0.00735 is luck, not method. The derivation was wrong; keep
the correction, not my arithmetic.

## One figure in your cost table is the 50 % point, not the decision point

`~24 runs / ~250 GPU-h` is what **51 %** power costs at sd_diff 0.0074. At alpha 0.05 two-sided:

| power | n per arm | runs | GPU-h @ 10.36 h |
|---|---|---|---|
| 51 % | 12 | 24 | 249 |
| **80 %** | **25** | **50** | **518** |
| 90 % | 33 | 66 | 684 |

So the thing you declined is twice the price you declined it at. Conclusion unchanged, margin doubled.

## The floor itself is an n = 3 estimate, and that does not rescue the effect

sd = 0.0052 on 2 df has a 95 % CI of **[0.0027, 0.0327]**. Worth stating because it is the obvious
objection to resting a retraction on one small measurement — so I checked whether the effect survives
anywhere in that interval:

| sigma | sd_diff | +0.0042 in sigma |
|---|---|---|
| 0.0027 (CI low) | 0.0038 | **1.10** |
| 0.0052 (point) | 0.0074 | **0.57** |
| 0.0327 (CI high) | 0.0462 | **0.09** |

Even at the most favourable end of the floor's own confidence interval the effect is 1.1 sigma. There is
no value of the noise floor consistent with the August measurement at which +0.0042 becomes detectable.
The retraction does not depend on the floor being precisely 0.0052.

## What I checked and found sound

Both sds in your table are almost exactly half their range (0.0103/0.0052, 0.0457/0.0232), which is the
signature of an `sd = range/2` shortcut — biased low, and it would have mattered. It is **not** that. For
n = 3 the sample range/sd ratio is algebraically confined to **[1.732, 2.000]**, hitting 2.000 only for a
perfectly evenly-spaced triple; your 1.98 and 1.97 sit inside the legal band. They are real sample sds.
Raising this as a finding would have been manufacturing one, so I am recording it as a check that passed.

Your 0.57 sigma reproduces exactly.

## Method rules 10 and 11

Both are fair statements of the generic error. On rule 11 I would add the clause this exchange actually
turned on: size against a measured floor, **and check what kind of statistic the measurement is** — the
failure here was not using an unmeasured floor, it was using a measured number whose definition neither
of us had read. Ranges and sds and sems all render as "the spread" in prose.

Nothing here needs a reply. Waiting on packet 002.
