# PACKET 050 — PRE-REGISTRATION: why chromatin does not transfer (RESULTS §93, draft), before any code or new data
packet_id: 050
created: 2026-10-06
repo_commit: 178213b
type: **PRE-REGISTRATION** (no GPU; Kaggle CPU only; the principal approved the public-data downloads on 2026-10-06)

## A. Origin
- **The principal's request** (2026-10-03):
  - make chromatin genuinely useful, with richer data;
  - keep mechanistic interpretability;
  - hypothesise why it fails, and test the failures;
  - consider fusing chromatin with the pathway graph;
  - avoid the training locking into a shortcut early.
- **This packet is the cheap, CPU-only stage** that decides what earns GPU next week.

## B. Also in this commit (review 049)
- **92.10:** the C1 wording, and a descriptive U937 note beside the verdict.
- **88.7:** the item-2 statement (u0–u2 = trained seeds 0–2's starting weights, from the code path) and the code-identity record.

## C. The design (§93)
- **93.1:**
  - the hypotheses H1–H5;
  - new facts from building the manifest: **VCAP's ATAC in v9 is an EpiMap imputed track**, and AGS, NPC and PHH use H3K4me3
    as an accessibility proxy.
- **93.2, Stage 1a (no new data):**
  - **H2:** a learning curve, T1 refitted on k ∈ {4, 6, 8, all} covered fitting cells, 5 subsets per k. Reported only, with the
    planted-P1 curve beside it for small-k bias.
  - **H3:** a gradient-boosting form (LightGBM) with T1's bars, its N1 null, and a planted-effect calibration first.
- **93.3, Stage 1b (new public data):**
  - **H1:** promoter-window (F_prom) against enhancer-window (F_enh) and regulon-weighted TF-motif accessibility (F_reg) from the
    same samples v9 used, via ChIP-Atlas by GSM → SRX and ENCODE.
  - **Rows:** five measured dev cells (VCAP missing), so the cell conjuncts are ≥ 4 of 5.
  - **Calibration:** M2.
- **93.4:**
  - Stage 1a runs first;
  - Stage 1b's test runs only after its data's sha1s are committed;
  - written after §91 and §92, which is disclosed.

## D. Feasibility (desk, 2026-10-06)
- **v9's accessibility sources:** Cistrome 18 cells, ENCODE 6, EpiMap 1 (imputed). The map is saved as
  `epigenetics/outputs/atac_sources_v9_93.json`: per cell, the GSMs and ENCODE experiments.
- **ChIP-Atlas `bed05`:** of 7 sampled GSMs, 4 gave peak files of 0.8–13 MB, 2 were empty (the ETag is the empty-file MD5), and 1
  was 293 bytes.
- **Downloads:** all large downloads (≈ 1 GB of peaks, hg38) run inside a Kaggle CPU kernel with internet; the laptop gets only
  the small outputs.

## ASKS
1. **Are H2 and H3 decisive enough to be worth running?** In particular, H2 is "reported only". Should it gate anything (e.g.
   whether to invest in more chromatin-covered cells)?
2. **H3's settings:** are they fixed tightly enough to prevent tuning on dev? (Trees 200, leaves 31, lr 0.05, min-child by LOCO
   over 3 values.)
3. **H1:**
   - Is F_prom the right same-data reference?
   - Are ≥ 4 of 5 cells right with VCAP missing?
   - Should dev-train LOCO be a co-primary rather than a secondary, given only 5 dev cells?
4. **The VCAP finding** (imputed ATAC plus a failed K27me3): does it change how §91.11's VCAP reading or §92's VCAP numbers should
   be worded now?
5. **Anything missing for H5 or H4,** or for the principal's two other directions: chromatin gating TF-regulon or pathway edges,
   and pretraining chromatin→expression across many cells. The PI proposes those only after Stage 1, each with its own packet.
