# PACKET 104 — RESULT: v11 R1 / R2 read once, and the DTI probe (RESULTS §102.7): no claim on any R
packet_id: 104
created: 2026-11-01
type: **RESULT REVIEW**

**What ran:**
- **The runs:** every v11 slot resolved (4 arms × 3 seeds), each on its first attempt with no guard stop, ledgered at
  download (`model/results/v11/v11_ledger.jsonl`, last 5e34280). s0 / s1 ran on `wwhhhhhhhh`, every s2 on `suryababu179`.
- **The reading:** `model/v11/v11_read.py` (102.3, 8 tests; with the v11 probe tests, 12 / 12 pass locally) run once →
  `model/results/v11/v11_reading.json` (sha1 `3ff930bd…`).
  - The reference is T-B's off arm, checked against `tb_ledger.jsonl`;
  - every v11 record and prediction file was checked against the v11 ledger.
- **The probe:** `model/v11/v11_probe.py` run once on CPU → `model/results/v11/v11_probe.json` (sha1 `1118bef5…`).
  - **Its roots:** `external/kaggle_alt2_stage/{lincs-v11-src, lincs-v10c-full, lincs-v9-bundle}` plus `drug/outputs/dti`.
    Before it ran, every one of the 29 inputs that `find()` resolves was checked against the v11 kernels' sha1 pins, and the
    local model / data code against the pinned code: 0 mismatches.
  - **A first attempt was killed when the Claude session ended,** with no traceback, no output and nothing read. The probe
    refuses to run if its output exists, so the re-run is the only output.

**The readings** (dev contrast r, 12,202 rows; §100.4's rule):

| reading | Δ | threshold | cells | verdict |
|---|---|---|---|---|
| R1 full vs T-B off | +0.0036 | 0.0042 | 6 / 6 | no claim |
| R2 chromatin | +0.0024 | 0.0039 | 6 / 6 | no claim |
| R2 targets | +0.0002 | 0.0030 | 3 / 6 | no claim |
| R2 CCLE | +0.0014 | 0.0030 | 4 / 4 CCLE cells | no claim |

**Reported:**
- **Permutations:** each run's own score minus the permuted one is ≤ 0.0019 in every run, and ≤ 0.0011 in all but one.
- **Per compound:** 0 BH discoveries in any pair, and 0 under all 9 relabellings.
- **The known-target setting** moves contrast r by −0.0002 to +0.0012.
- **The probe:**
  - atom → gene attention recall@10 on held-out compounds is 0.013 / 0.023 / 0.015 (full / −targets / reference), against
    chance 0.010;
  - v11-full's structure head scores 0.218 against the popularity baseline's 0.156 (114 compounds). By bin it is 0.131 /
    0.185 / 0.499 against 0.093 / 0.157 / 0.303.

## ASKS
1. **Does R1 / R2 reproduce from `external/kaggle_out/v11/` and T-B's off arm,** with the ledger checks? Every input is a
   ledgered prediction file, so per-row values can be recomputed (review 102 C3).
2. **Is §102.7's reading worded right?**
   - "no claim on any of the four";
   - "the reference's seed SD (0.0035) is as large as R1's Δ";
   - "**at inference v11 barely reads the cell-specific inputs it was given**", from permutation costs ≤ 0.0019. Is that a
     fair reading of a permutation test on a trained model?
3. **The probe:**
   - Is "the target knowledge lives in the head, not in the attention" supported by these numbers?
   - Is −targets' untrained head the right comparator for v11-full's head (§102.5 C3 follow-up)?
   - The probe JSON holds aggregates by subset and bin, not per-compound ranks. It was already running when review 102 C3
     arrived. Say if the per-compound ranks are needed; they can be recomputed from the checkpoints in about 1.3 h of CPU.
4. **The PI proposes not to run v11's test-set evaluation (§102 item 5), since no R passed.** Agree?
5. **The section's last bullet points to the next programme** (IDEAS §E, the principal's direction of 2026-11-01: cell
   representations screened on cell-level proxy targets before any drug-response run). Is that appropriate in a result
   section, or should it move out?
