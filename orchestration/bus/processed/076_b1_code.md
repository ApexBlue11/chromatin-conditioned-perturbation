# PACKET 076 — CODE: B1, v9 retrained drug-blind (RESULTS §98.9, §98.2 B1, §98.6 C4), plus the M re-run record (§98.8); before any B1 GPU
packet_id: 076
created: 2026-10-12
repo_commit: (this commit)
type: **CODE REVIEW** (B1) + **RECORD** (M v1 crash, the fix, and the re-run)

## B1 (§98.9)
- **`xpert_arm.py --drug_blind`** (W38 + PI; `c55ce257`):
  - `u` is set to its training-row mean;
  - one constant atom token: the atom-weighted mean over the unique training compounds' full atom blocks (no truncation);
  - `assert_drug_constant` on every batch;
  - `drug_permutation_check` after each seed (deranged identities, max |Δ| ≤ 1e-5, no metric against truth).
  - With the flag off, behaviour is unchanged apart from a `drug_blind: False` key in the JSON and in the checkpoint.
    W38 factored `build_parser` and `get_candidate_flags_and_suffix` out of `main()`; please confirm the default path's
    suffix and flags are unchanged.
- **The tests:** 9 + 8 pass. Mutation: 4 of 22 → **0 of 22**.
- **The kernel:** `make_b1_kernel.py` takes P9's template from `make_p9_kernel.py` by asserted substitutions. The 30-line diff
  from `kern_v9p9` is listed in §98.9, including guards B1a and B1b. The source dataset is E3's upload with only `xpert_arm.py`
  replaced.
- **The local smoke** (the kernel's smoke mode) passes end to end: the trained model's permutation check gives 0, B1a and B1b
  pass, and 1,426 compounds / 40,804 atoms are pooled.

## M (§98.8)
- **The crash:** v1 trained to its early stop and then crashed on staging, because `np.savez` appends `.npz` to the `.part` path.
- **The fix** is in the kernel template only; `p10_mlp.py`, the dataset and the pins are unchanged.
- **A new test executes each generated kernel** end to end on synthetic inputs, and fails on the v1 template.
- **v2 is re-running** (five kernels pushed 02:17; the sixth when a slot frees). No v1 output was used.
- **The disclosure:** the echoed validation curve showed the drug-free MLP's best validation PCC_DEG, 0.640 (M s0) and 0.642
  (M+dt s1).

## ASKS
1. **B1's flag-off path:** is it unchanged? In particular, check the refactored suffix and flag construction against the
   previous code.
2. **B1's flag-on path:**
   - Is anything compound-indexed still reaching the model? (Your 073 C4 list was `u_feats`, `atoms` and `atom_mask`.)
   - Is the permutation check meaningful, given that blinding makes it hold by construction? Or is the tensor guard the real
     guard, and the check a confirmation?
3. **The kernel:** is the 30-line diff complete for B1? Is anything P9-specific left (names, row guard, manifest)?
4. **The M disclosure (0.640 / 0.642 validation):** does it need anything beyond §98.8? R1 is mechanical and its registration
   is fixed.
5. **Order at the reset:** O9 session 3 first (8.3 h), then B1 (≈ 6 h); D's timing after. Agree?
