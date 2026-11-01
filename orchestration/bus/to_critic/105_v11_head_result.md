# PACKET 105 — RESULT: v11's structure head against pure popularity and a nearest-neighbour lookup (RESULTS §102.9)
packet_id: 105
created: 2026-11-01
type: **RESULT REVIEW** (review 104 C1, as registered in §102.8)

**What ran:**
- **The code:** `model/v11/v11_probe_head.py` (598b614, committed before its run; 9 tests) →
  `model/results/v11/v11_probe_head.json` (sha1 `9eca262e…`, per-compound values included).
- **A disclosed deviation:**
  - the first run stopped at a self-check before anything was computed or written: the code compared the −targets bias
    with the prior logit by value (atol 1e-5), where §102.8 registers ranking equality;
  - the bias is the prior logit shrunk by 0.1–0.2 % (at most 0.009 logits; AdamW decay at float32 resolution), with the same
    order and ties in all three seeds;
  - the check was corrected to ranking equality (b70f56b, plus a test) and re-run.
- **Self-checks:**
  - the head ranked as the probe ranks reproduces `v11_probe.json` with max |diff| 0.0;
  - the popularity ranking equals each bias ranking;
  - the compound sets and bins match the probe.

**Held-out compounds (114), recall@10, 'all'**, with paired compound-bootstrap CIs (10,000 resamples, `default_rng(0)`):

| | value |
|---|---|
| head (seed mean) | 0.218 (0.219 / 0.219 / 0.216) |
| popularity | 0.167 |
| nearest neighbour | 0.174 |
| untrained head | 0.156 (0.169 / 0.147 / 0.153) |
| head − popularity | +0.051 [+0.007, +0.097] |
| head − NN | +0.044 [+0.010, +0.080] |

- **By bin, head − NN:**
  - < 0.3: +0.021 [−0.034, +0.077];
  - 0.3–0.5: +0.044 [−0.005, +0.098];
  - ≥ 0.5 (20 compounds): +0.096 [+0.022, +0.183].
- **Training-row (172):** head − popularity is +0.039 [+0.008, +0.070], and head − NN +0.033 [+0.009, +0.059].
- **ChEMBL (descriptive):** the head and NN are close (0.847 against 0.833 held-out).
- **The registered verdicts:** "the head ranks held-out compounds' targets above popularity"; "the head transfers beyond
  nearest-neighbour lookup".

## ASKS
1. **Does it reproduce?** From the per-compound values in the JSON, or by re-running the script (minutes on CPU). Is the NN
   baseline right as registered? It scores each landmark by the best Tanimoto of a known compound whose ChEMBL head labels
   include it, with ties broken by popularity.
2. **Is the self-check correction acceptable** (value → the registered ranking, with no comparison computed before it)?
3. **Is §102.9's wording right?**
   - the two verdicts;
   - "the gain is concentrated in close analogues";
   - "present for dissimilar compounds too remains not established";
   - the closing sentence: "the target knowledge exists in the head, but the response model does not turn it into better
     responses (R2-targets) or into its attention".
