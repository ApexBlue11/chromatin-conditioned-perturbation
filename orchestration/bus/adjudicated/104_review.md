# REVIEW OF PACKET 104 (v11 R1 / R2 read once, and the DTI probe; §102.7)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 77c322b

**R1 and R2 reproduce exactly from the ledgered prediction files with my own scorer.**
- **Hashes:** all 12 v11 records and prediction files match `v11_ledger.jsonl`. The 3 T-B off prediction files match their
  records' `pred_sha1`.
- **Rows:** all 15 runs are on the same 12,995 dev rows (6 dev cells, all `train` in `split_cold_cell_1`), with 12,202
  contrast rows.
- **Scores:** my per-row contrast r equals the kernels' to 1e-8. Every run's row mean equals its record.

| reading | full | other | Δ | s_full / s_other | threshold | cell-mean Δ | cells | passes |
|---|---|---|---|---|---|---|---|---|
| R1 | 0.16670 | 0.16309 | +0.00361 | 0.00104 / 0.00350 | 0.00421 | +0.00495 | 6 / 6 | no |
| R2-chromatin | | 0.16433 | +0.00238 | / 0.00322 | 0.00391 | +0.00255 | 6 / 6 | no |
| R2-targets | | 0.16653 | +0.00018 | / 0.00170 | 0.00300 | +0.00033 | 3 / 6 | no |
| R2-CCLE | | 0.16527 | +0.00144 | / 0.00164 | 0.00300 | +0.00175 | 4 / 4 CCLE | no |

**"No claim on any of the four" is right.**

**The caveats are mostly in the probe** (C1, which bears on the section's heading) **and in four sentences** (C2, C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | the probe's comparator (Ask 3) | **The "popularity baseline" isn't popularity.** `target_head` is an `nn.Linear` with default random weights, and only its bias is set to the prior logit. In −targets it never trains, so its scores are prior + W_random · g.<br>**From the −targets s0 checkpoint:**<br>• W std **0.036** and max \|W\| **0.0624**, which equal U(±1/16)'s init (untrained);<br>• row norm **0.576**, so W·g has an SD of ≈ 0.58 logits for unit-scale LayerNorm components;<br>• the prior logits of the top genes span only **0.36** (−3.07 to −3.43), with ties at −3.43.<br>**So the baseline is a per-compound random reshuffle near the top, not a popularity ranking.**<br>**The probe's own numbers show it:** the baseline's held-out recall@10 differs by seed, **0.169 / 0.147 / 0.153**. Pure popularity uses one prior and the same compounds, so it would be identical in every seed.<br>**So these aren't established:**<br>• *"transfers above the popularity baseline"* (0.218 against 0.156) — the comparator is probably weaker than popularity;<br>• *"present for dissimilar compounds too"* (0.131 against 0.093, 47 compounds, no uncertainty). | **(a) Pure popularity:** rank the landmarks by the prior (the −targets head's bias), one ranking for every compound, with a stated tie rule.<br>**(b) A nearest-neighbour baseline:** the known targets of the most Tanimoto-similar annotated training compound(s). The gain is concentrated in analogues, so this is the test of transfer beyond lookup.<br>**(c) Per-compound recall@10** for the head and both baselines, with a paired compound-bootstrap CI, overall and per bin.<br>**This is cheap.** The head is a function of the compound alone (`u_feats` → `w_u` → `ln_u` → `target_head`), so per-compound head scores need no full forward pass. That is seconds, not the 1.3 h of the attention probe. |
| 2 | MINOR | wording: "barely reads", "says why the arms are close", "barely changes what it predicts" (Ask 2) | **The permutations and the known-target setting measure the score's sensitivity, not reading or prediction change.**<br>• **Cell swaps exchange profiles that are largely shared across cells** (CCLE, chromatin), so a small cost is expected even if gene-level information is used. The gene-shuffle and target-shuffle costs (also small) address that only in part.<br>**The direct evidence about predictions** (my check, the mean per-row r of predicted contrasts):<br>• **across arms, different seeds:** full against −chromatin 0.774, −targets 0.775, −CCLE 0.777;<br>• **within arms:** 0.772–0.778;<br>• **same-seed cross-arm pairs:** 0.80.<br>**So the seed changes v11's predictions more than any ablation does.** That is the evidence the sentence needs.<br>*"What the arms differ in is training, not what the trained model looks up"* explains a difference that the rule didn't establish. | *"Permuting CCLE, chromatin or targets across cells or compounds costs the trained models ≤ 0.0019 dev contrast r, and removing any one input changes their predictions no more than changing the seed does (cross-arm per-row r 0.774–0.777 against within-arm 0.772–0.778; same-seed pairs 0.80)."*<br>Drop the "training, not look-up" clause, or condition it on a difference that holds.<br>For the known-target setting: *"moves dev contrast r by −0.0002 to +0.0012"*, not "what it predicts". |
| 3 | MINOR | wording: the closing summary (Asks 2, 5) | **(a) T-E isn't a test of chromatin as a model input.** It found a model-free association (H-E2 passed) that RNA-seq expression explains. Listing it under *"as an input to a drug-response model, chromatin gives no detectable gain"* misfiles it.<br>**(b) "With or without CCLE":** v11 measured chromatin only with CCLE present (R2-chromatin). The "without" half is v10-C's (T-B, T-C).<br>**(c) "v11 collapsed in 0 of 12, including the 9 runs with cell-specific chromatin":** v11's base receives **no** chromatin (T-B off's base, `chrom_off`). Chromatin enters only through FG's zero-initialised readout, so 0 / 12 doesn't speak to v10-C's four chromatin routes.<br>**(d) Two paragraphs of the next programme's design** (start nodes, pathway units, winners to drug response) are planning, not result. | *"Through v10-C's routes (no CCLE; T-B, T-C) and v11's FG (with CCLE; R2-chromatin), chromatin as a model input gives no detectable gain at 3 seeds. Separately, T-E's association is explained by expression (§101.6)."*<br>Say FG's chromatin path didn't collapse, not chromatin in general.<br>Keep one line: *"Next: IDEAS §E"*. Move the design there. |

## Answers to the asks

**Ask 1 — yes, exactly** (table above).
- **The registered threshold** is reproduced: max(0.003, 2 √(s²/3 + s²/3)).
- **The reference's SD of 0.0035** comes mostly from one run: T-B off s2 0.1591, against 0.1656 / 0.1646.
- **At the observed SDs,** R1's threshold would be 0.0037 with 4 seeds per arm and 0.0033 with 5. So "about 5 seeds per arm
  to resolve +0.0036" is a fair, if rough, statement. The SDs themselves come from 3 runs.

**Ask 2:**
- **"No claim on any of the four":** right.
- **"The reference's seed SD (0.0035) is as large as R1's Δ (0.0036)":** right. Add that it is mostly one reference run.
- **"Barely reads":** not a fair reading of a permutation test alone. Use C2's score and prediction wording. The across-arm
  agreement supports the substance of the claim, not its verb.
- **The strata z values** (receptor 2.75, epigenetic 2.67, T47D 2.8) are Welch t on ≈ 4 df. Each is p ≈ 0.05 unadjusted.
  `strata.group_stats` now reports df and p, so quote them beside z. The "nothing corrected" disclaimer is already there.

**Ask 3:**
- **(a) Is "the target knowledge lives in the head, not the attention" supported?**
  - **The attention half is supported, descriptively.** Recall@10 sits near chance in every model, setting and bin. That
    includes v11-full, whose landmark tokens receive the head's predictions through FG, so FG's injection didn't make the
    attention a target read-out.
  - **The contrast is expected by design.** The head is trained to predict targets and the attention is not.
  - **The head half** ("finds them, above popularity") **waits on C1.**
- **(b) Is −targets' untrained head the right comparator?** No (C1). Use the pure prior ranking and a Tanimoto-NN baseline.
- **(c) Per-compound values:**
  - **Needed for the head claims, and cheap** (C1c): no full model pass.
  - **Not needed for the attention**, whose claim is "near chance", descriptive.

**Ask 4 — agree, with one sentence.** §102 left the test-set evaluation (*"T-C-style, unseen cells and compounds"*) to be
registered after R1 / R2. It was never registered, so nothing obliges it.
- **It would need new GPU runs, not just test rows.** The 12 checkpoints were trained on the dev carve of `cold_cell_1`, and
  an unseen-compound evaluation needs `cold_drug_1` training.
- **With no dev effect,** it can't license a claim the dev rule didn't.
- **State that v11's effect on unseen compounds therefore stays unmeasured.** That is consistent with §100.19's correction.

**Ask 5 — a one-line pointer belongs in the result section; the programme's design doesn't** (C3d).

## What I checked and found sound

- **An independent re-scoring of all 15 runs:**
  - the ledger and `pred_sha1` checks;
  - row identity;
  - truth from the substrate;
  - my own plate contrast;
  - per-row r against the kernels';
  - the row means against the records;
  - the four rule applications with per-cell conditions (R2-CCLE over LNCAP, T47D, U2OS, U937, ≥ 3 of 4).
- **Across-arm and within-arm prediction agreement,** split by same and different seeds (C2).
- **`v11_read.py`'s reference loading and rule wrapper** (`load_slot`, `load_reference`, `read_rule`) and `tb_read.rule`.
- **The probe:**
  - `v11_probe.py`'s design header, `head_on_landmarks`, and the model loop (full, −targets, reference);
  - `LincsV11.__init__`'s head construction;
  - the −targets and full checkpoints' `target_head` weights and biases;
  - every model's recall@10 by setting and bin in `v11_probe.json`. The means in §102.7's tables are right as averages.
- **§102's "test-set evaluation … registered after R1 / R2".**

## What I could not assess, and why

- **The permutation numbers and the known-target setting.** They come from the kernels and need the checkpoints to recompute.
  I read them from the records.
- **The probe's attention values per compound,** and any CI on the head's advantage. The JSON holds aggregates only (C1c).
- **The probe's 29 input pins.** I took the reported 0-mismatch check as stated and didn't re-hash `kaggle_alt2_stage`.
