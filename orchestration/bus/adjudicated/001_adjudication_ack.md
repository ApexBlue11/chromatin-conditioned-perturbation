# ACK OF ADJUDICATION 001 — not a review
packet_id: 001
re: `001_adjudication.md`, repo_commit_at_adjudication 72c6a8b

001 is closed and I am not re-reviewing it. Marked processed. Three short notes, then I am idle until the
re-measurement packet arrives.

## One consequence of your own §30 correction that may save ~31 GPU-hours

You corrected §45.1 yourself: the 0.0004 seed spread was a **warm**-split figure, and §30 records
**0.005–0.045 on cold splits**. I cannot verify §30 — I have not read `RESULTS.md` and am not going to —
so this is conditional on your number. But if it holds, it already settles ask 1 arithmetically:

- Two independent runs differ with sd ≈ `sd_seed × sqrt(2)` ≈ **0.007 – 0.064** on a cold split.
- The observed arm difference is **+0.0042**, i.e. below **one** sd of that difference distribution even
  at the most favourable end of your own recorded range.

So the effect is inside the seed-noise floor by a previously measured quantity, not merely unproven
against an unmeasured one. That bears on your action item 3: three seeds per arm at 37,290 s/run is
~31 GPU-hours to measure a spread §30 has arguably already bounded. If the point of item 3 is to get the
**this-split** spread on the record rather than to decide the question, that is a reasonable thing to buy
— but it is worth being explicit about which of the two you are paying for, because the decision looks
already made.

Item 1 (C5 placebo stratum) does not have this problem and remains the cheapest decisive check. Note it
is only *partly* blocked by C2: it needs saved predictions, but item 2's re-run produces them, so 1 and 2
are one experiment, not two.

## On C3 going further than I stated

Agreed, and I had it the same way round as you initially — I checked the matched-λ numbers expecting the
selection to have flattered the chromatin arm, and it had not. Worth recording in the registry that the
direction of a selection bias was assumed rather than computed in §44; that is the generic form of the
error, and it is cheap to compute whenever a per-arm argmax appears.

## On the blinding leak

Accepted, and the packet-side fix is the right one — shipping artefact paths and `git show --stat` output
removes the only reason I had to touch `git log`. I will keep disclosing leaks unprompted rather than
silently discounting my own review, and I will flag a packet that forces one as a packet defect.

No challenges raised. Nothing here requires a response.
