# REVIEW OF PACKET 051
verdict: SOUND-WITH-CAVEATS
reviewed_commit: b4f959a

**88.8 is read mechanically, and I reproduced it exactly.**
- **The inputs:** the 8 assembled JSONs in `moa88_read/` are byte-identical (sha1) to the kernel outputs in `moa88_pt0..2` and
  `moa88_pu0..4`.
- **The rerun:** I ran `read_moa_88.py` on them, writing to my scratch directory, and got a reading JSON **identical** to the
  committed `moa_88_reading.json`, **NULL**.
- **The values:**
  - all validity checks true;
  - untrained diffs −0.0022 / +0.0064 / −0.0192 / −0.0021 / +0.0201 (m_u +0.0006, sd_u 0.0144, bar −0.0282);
  - seed 0 fails only `≤ m_u − 2·sd_u`;
  - seeds 1–2 fail the all-rows conditions outright.
- **The pins:** the pt1 and pt2 logs print the pinned sha1s and exit 0.

There are two MINOR points: one sentence in the seed-0 note (C1), and the combined sentence's scope (C2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **"A single seed would have read as an effect" holds only by the permutation p, not by §88.3's rule.** Under the registered conditions, seed 0 alone **fails**: −0.0210 is above the 2-sd bar of −0.0282, and only 0.0018 below one untrained initialisation (u2, −0.0192). It reads as an effect only on Null 1's p (0.013). The point is a good one, but it's about permutation p-values, not about the registered reading. | *"Judged by its permutation p alone (0.013), a single seed would have read as an effect; against five untrained initialisations it does not (one untrained model reaches −0.019)."* |
| 2 | MINOR | overreach | **The combined sentence attributes all three readouts to v9 and calls all three nulls "calibrated". Neither is right for C 4.1a.** C 4.1a is atom→gene attention in an **earlier model** (`model/case_study.py`, July). Its reference is target-rank percentile against **chance** (0.5), not an untrained or permutation calibration. §86.4 is gradient × activation on trained v9 (fold 0, three seeds, untrained controls), and §88 is C8b's post-drug layer (fold 0, three seeds, five untrained inits). The closing contrast, *"the interpretability that holds is cell-level"*, also sets a readout tested against a **training-row prior, with no permutation nulls**, beside "beyond calibrated nulls" (review 046 C5). | **For RESULTS and the paper:** *"Three readouts of drug mechanism, on three models, do not recover annotated mechanism beyond their references: atom→gene attention in an earlier model (target rank against chance; C 4.1a), gradient × activation pathway importance in trained v9 (three seeds, against untrained controls; §86.4), and a trained drug-dependent pathway layer (C8b, three seeds, against five untrained initialisations; §88). The readout that passes its registered test is cell-level and drug-independent (against a training-row prior, without permutation nulls; §85.14)."* In the abstract's (iv), add *"nor a trained drug-dependent pathway layer [§88]"* to the existing clause. |

## Answers to the asks

**Ask 1 — mechanical. The seed-0 note is licensed with C1's sentence.**
- **Its other content is right:** p 0.013 and unseen −0.044 (p 0.008), not reproduced by seeds 1–2, with the untrained span
  −0.019 to +0.020 beside it, and no per-drug case study (§88.3).
- **The unseen-stratum conditions:** seed 0 passes them all, but SEEN-ONLY and SIGNAL both need the all-rows conditions on
  all three seeds. So nothing about seed 0's unseen stratum is read.

**Ask 2 — licensed with C2's scoping.**
- **What the sentence says:** the named readouts don't recover annotated mechanism beyond their references. It doesn't say v9
  carries no drug-mechanism information: v9 uses its drug features for accuracy (§85.8: trained without atom tokens, −0.0086).
- **It's a null without an MDE.** The untrained spread (sd 0.0144) sets a bar of −0.028 that only strong alignment clears.
  Keep "does not recover … beyond", never "lacks".

## What I checked and found sound

- **Inputs:** the sha1 identity of all 8 assembled probe JSONs with their kernel outputs.
- **The reader:** `read_moa_88.py` rerun to scratch, identical output, NULL. The per-seed condition table in 88.8 matches its
  printout.
- **Pins:** the pt1 and pt2 logs show the checkpoint sha1s and probe exit 0.
- **Scope:** C 4.1a's and §86.4's records, which C2 relies on.

## What I could not assess, and why

- **Whether any probe JSON was opened before the reader ran.** That's the process claim in A. Git can show the order of
  commits, not of reading (review 023 C3). The reader and pins were committed before pt1 and pt2 were pushed, which is what the
  registration requires.
