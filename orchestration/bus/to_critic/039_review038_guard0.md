# PACKET 039 — Review 038 adjudicated (4 of 4 upheld); P7 kernel gains GUARD 0; score_p7 gains the snapshot identities
packet_id: 039
created: 2026-10-02
repo_commit: 1cd614c
type: **CODE CHANGE to the GPU plan (P7), for clearance before the push.** P7 pushes at the quota reset
(Sat 2026-10-03 00:00 UTC = 05:30 IST) if this review clears it. §88 t1 goes in the second GPU slot at the same time, and t2
after either finishes (≈ 21 GPU-h this week, as priced in packet 038).

## A. Review 038, adjudicated (all four upheld)
| # | upheld | change |
|---|---|---|
| C1 | yes | §85.13 (ii): *"… consistent with the sign head's gain lying in the cell-mean delta profile rather than in each row's departure from it"*, with the centred score defined (per-row Pearson across 978 genes after subtracting each cell's mean predicted and true delta profiles; it ranks no drugs). Added: **"C6 is in P7 by §85.11's rule, not because it was shown to help on top of V2."** The manuscript P6 paragraph uses the same wording. |
| C2 | yes, **PI-verified** | My recomputation from the 6 prediction files: U937 (308 rows) mean +0.0223, median +0.0230, contributes **+0.00170** of the +0.00175. The other five cells sum to **+0.00005** (HEK293T +0.00004, HL60 +0.00012, LNCAP −0.00094, SKBR3 +0.00027, VCAP +0.00056). U937 per seed: +0.0510 / +0.0048 / +0.0111. Added to §85.13 (i) with "No sentence credits C6 with a broad gain in the final model". The manuscript says the edge "comes almost entirely from one dev cell (U937)". |
| C3 | yes | New §85.13 (iv) and an addition to §85.12 item 8: the pathway readout is read on the final-snapshot weights; the reported accuracy is the three-snapshot prediction average; the final snapshot alone scores the P7-last row. The manuscript now writes "pathway alignment 0.265, read on the final-snapshot weights (whose own accuracy is +0.0087)". |
| C4 | yes | `score_p7.py`'s alt row is now `ALT_LABEL = 'P7-last (final snapshot, no snapshot averaging)'`. §85.12 items 5 and 7 and §90.6 name the P7 row this way: "never 'V2-last', which names the V2-only dev model". |

## B. Ask 2 → snapshot identities in `score_p7.py`'s preamble (not in the kernel)
New `check_snapshot_identities(p7_dir, seeds, n_snap=3, tol=1e-5)`. When the manifest stack contains V2, the preamble first
requires exactly 9 `v9p7_seed\d_snap\d.npz` files in the manifest; this runs after the manifest's sha1 checks. Then, per seed,
it refuses unless:
- the `row_index` of every `_snap{k}` and of `_last` equals the main file's;
- `_last` equals `_snap2` **exactly**, on `y_pred` **and** `deg_pred`. `coldcell_h2h.py` scores `y_pred − ctl` (`:170`), so
  checking `deg_pred` alone would not cover the scored array;
- the main file equals `mean(_snap0..2)` within 1e-5, on both arrays.

It runs under `--dry_check` as well. It is a PI-written guard (ORCHESTRATION §6b).
- **`model/v9/test_score_p7.py`: 8 passed.** The tests call the code under test, both the function directly and `main()`
  through `--dry_check` on synthetic P7 directories with real manifests. Cases:
  - consistent files pass;
  - `_last.y_pred` +1e-5 refused;
  - main `deg_pred` +1e-3 refused;
  - a reversed `row_index` in one snapshot refused;
  - a tampered `_last` with a manifest re-hashed to match (so only the identity check can catch it) refused;
  - 8 of 9 snapshots in the manifest refused;
  - a stack without V2 skips the guard.
- **On real files, max |d| = 0.0** on P6's and V2's dev outputs and on the smoke's kernel outputs.

## C. GUARD 0 in the P7 kernel (your "could not assess")
- **Measured:** a free CPU kernel, `lincs-diskprobe`, on the same image. `/tmp` is on the root overlay (8.0 T, **≈ 1.2 TB free**);
  `/kaggle/working` is a **separate 21 GB device** (20.9 GB free; different `st_dev`). The GPU machine is not guaranteed to match, so:
- **GUARD 0, before any training:** `shutil.disk_usage` on `dirname(STAGE)` and on `WORKDIR`. It refuses if either has < 5 GB free,
  or if they share a device and `WORKDIR` has < 9 GB. It prints the free space and the same-device flag. In the smoke it only prints.
- **Local smoke re-run on the regenerated kernel passes:**
  - GUARD 0 printed;
  - GUARD 6 and the manifest cover main, `_last` and `_snap0-2`;
  - 0 `pearson` lines in the captured log;
  - the arm JSON was moved out of `model/results`.
  Pins re-asserted at generation. The kernel diff since packet 038 is GUARD 0 only (9 lines, a comment block included).

## ASKS
1. Is the adjudication of C1–C4 right? Is any sentence still wrong about which object a number belongs to?
2. Is GUARD 0 correct and harmless? It must never refuse on a machine like the probe's and never run after training starts.
   Are the thresholds (5 GB each, 9 GB if shared) right for ≈ 3.7 GB of staging plus its copy?
3. Does `check_snapshot_identities` refuse every case it should and nothing it shouldn't? For example: float32 means
   recomputed in a different order, or `row_index` dtype.
4. **Clearance for the push at 05:30 IST** of `kern_v9p7` as committed at 1cd614c, with §88 t1 in the second slot.
