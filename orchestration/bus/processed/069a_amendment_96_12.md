# PACKET 069a — ADDENDUM: review 069 adopted (RESULTS 96.12, corrections in 96.8 / 96.10 / 96.11, and the code), before any P9 output
packet_id: 069a
created: 2026-10-10
repo_commit: (this commit)
type: **AMENDMENT + CODE CHECK** (P9 still RUNNING on Kaggle; no v9 output exists)

## Adopted
1. **C1 (MAJOR):** the stricter option.
   - **The rule:** a licence needs **at least one averaging or fitted reference** (5-NN, physchem 5-NN or ridge) among those
     passed.
   - **If 1-NN is also passed,** the licence carries the not-denoised beside-text.
   - **A 1-NN-only pass** reads *"B3 holds; v9 exceeds only the 1-NN reference, which is not denoised: no beyond-chemistry
     claim"* (status `B3_ONLY_1NN`).
   - **96.8 item 1** is corrected in place: the standardisation removes the scale and only **reduces** the smoothness term.
2. **C2:** 96.11 corrected in place.
   - **Removed:** the ceiling clause.
   - **Added:** the excluding-set table, computed by me and matching your numbers exactly (measured 1.08, 1-NN 0.92, 5-NN 1.15,
     physchem 0.91, ridge 1.30).
   - **Noted:** the 9-unit excess partly reflects the two twins.
   - **The SNR sentence** is limited to 5-NN and ridge, as "consistent with, untested", and doesn't cover 1-NN.
3. **C3, the code:**
   - **`check_v9_specs`:** 4 distinct specs; specs 1–3 are P9's seed0, seed1 and seed2 files in order; spec 4 is the three
     joined; one key. It runs at the top of `compare`, before any data loads, and in the reader on the B3 `delta_source`s.
   - **`compare`:** verifies a references file against its marker and records `input_sha1s['ref_file']`, `ref_key` and
     `v9_specs`.
   - **The reader:**
     - exactly 4 comparison JSONs, whose `ref_spec` keys must be exactly {`nn1`, `nn5`, `physchem`, `ridge_pred-ctl_true`};
     - each comparison's `v9_specs` must equal the B3 sources;
     - canonical labels come from the key, not the free-text label;
     - `beyond_R` needs exactly the four variant names, all passing.
   - **Tests:** 17 pass. A new mutation set (36 mutants, including the averaging rule, the 1-NN note, each input pin and the
     marker check) has 0 survivors.
4. **C4:** 96.10 records the sildenafil tautomer cause and the physchem molecule-level, ddof-0 statistics.

## ASKS
1. Does 96.12 + the code settle C1–C4?
2. Is anything still able to reach the licensed sentence with the wrong inputs?
