# NOTE TO REVIEWER — re 001_adjudication_ack
not a packet · no review requested · 2026-09-20 · repo_commit 72c6a8b

Your arithmetic point is accepted and acted on. One correction to it, which moves the conclusion your way.

## Your figure was the wrong kind of number

You read §30's "0.005–0.045 on our cold splits" as a **standard deviation** and derived
`sd_diff ≈ 0.007–0.064`. Those are **ranges**. The three-seed table they come from is explicit:

| split | range | sd |
|---|---|---|
| unseen cell | 0.0103 | **0.0052** |
| unseen both | 0.0457 | 0.0232 |

So "0.005–0.045" is the span of *ranges* across splits, and using it as an sd inflates the floor ~2×.

## Done with the directly relevant number

Our regime is unseen **cell**, and its seed sd is measured: **0.0052** (v7, 3 seeds, protocol-matched).

```
sd_diff = 0.0052 × √2 = 0.0074
observed effect        = 0.0042
                       = 0.57 σ
```

**Your conclusion holds and is stronger than you stated**: the effect is not "unproven against an
unmeasured floor" — it is inside a floor this project measured in August, by a factor of two.

## Cost, since you raised it

One cold-cell arm is **37,290 s = 10.36 h** (in the arm JSON).

| | runs | GPU-h |
|---|---|---|
| item 3 as written (3 seeds × 2 arms) | 6 | **62** |
| powering +0.0042 at sd_diff 0.0074 | ~24 | **~250** |
| items 1+2 combined | **2** | **~21** |

**Item 3 is dropped** as a decision procedure. You asked me to be explicit about which of the two things
I would be paying for: neither, at that price. If the this-split spread is wanted for the record it is a
separate and much later purchase, labelled as such.

## Your second point is simply right

Items 1 and 2 are one experiment. Revised: **two runs, ~21 GPU-h, `--save_pred`, distinct filenames, args
written into the JSON, comparison routed through `head_to_head_mdmt.py` so its guards execute.** That
fixes C2 and delivers C5 in the same spend.

## Your generalisation is now method rule 10

> Compute the direction of a selection bias. Never assume it.

§44 asserted per-arm-best λ "favours the chromatin arm". It deflated it. You had the same prior and
computed it anyway, which is why you found the inversion. Also added as rule 11: size an experiment
against a *measured* noise floor before buying it, and don't use §30's ranges as sds.

## What happens next

The re-run is queued. When it lands you get **packet 002** carrying: both arm JSONs under distinct names
with args recorded, the saved predictions, the per-cell-line paired delta, the three-way placebo split you
specified (≥2 tracks / 0 tracks / pooled), and matched-λ ridge values with their own interval.

It will also ship artefact paths and `git show --stat` output so you have no reason to touch `git log`.
