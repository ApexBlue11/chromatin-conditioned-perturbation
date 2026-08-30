# -*- coding: utf-8 -*-
"""
Fetch ONLY the rows of XPert's `all_drugs_unimol_arr.npy` that a given evaluation actually touches.

The full array is (8981, 122, 514) float64 = 4.5 GB, and this disk has 28 GB free. But a .npy is a short
ASCII header followed by one contiguous C-order block, so row i lives at a computable byte offset and
Zenodo honours range requests. Fetching the ~2 k drugs one evaluation needs costs a few hundred MB.

Stored as float32 (their own MyDataset casts to float32 before it reaches the model, so this is the same
number the model would have seen) in a dict keyed by pert_idx, which is exactly the shape their
`MyDataset` expects for the non-unimol feature types -- `self.drug_feat[pert_idx]` works unchanged.

CORRECTNESS OF THE OFFSETS is checked, not assumed, three ways:
  1. a row re-fetched through a differently-aligned range must be byte-identical;
  2. column 0 is a padding mask, so every row must be a run of 1s followed by 0s;
  3. Uni-Mol tokenises every atom INCLUDING hydrogens plus two special tokens, and the array holds 122
     atom slots, so mask_len == min(n_atoms + 2, 122) EXACTLY for every molecule. Two earlier versions of
     this check were wrong on correct data -- one compared against the HEAVY-atom count, the other demanded
     a constant offset and fired on the 30 molecules of 1,970 large enough to be truncated at the cap.

    python model/v9/fetch_xpert_unimol.py --idx_file idx.json --out unimol_subset.npz
"""
import os, io, json, time, argparse, subprocess

import numpy as np

RECORD = '17182939'
URL = 'https://zenodo.org/api/records/%s/files/all_drugs_unimol_arr.npy/content' % RECORD
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'
SMI = r'C:\Projects\LINCS\external\xpert\code\XPert\processed_data\all_drugs_idx2smi_8981.npy'


def curl(start, end, tries=5):
    cmd = ['curl', '-sL', '--fail', '-m', '1800', '-A', UA, '-H', 'Range: bytes=%d-%d' % (start, end), URL]
    want = end - start + 1
    for i in range(tries):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and len(r.stdout) == want:
            return r.stdout
        print('    retry %d (rc=%d, got %d/%d)' % (i + 1, r.returncode, len(r.stdout), want), flush=True)
        time.sleep(3 + 5 * i)
    raise SystemExit('FATAL: range %d-%d failed after %d tries' % (start, end, tries))


def header():
    """Parse the .npy header remotely: (shape, dtype, first data byte)."""
    h = curl(0, 511)
    if h[:6] != b'\x93NUMPY':
        raise SystemExit('FATAL: not a .npy (got %r)' % h[:16])
    major = h[6]
    if major == 1:
        hlen = int.from_bytes(h[8:10], 'little'); off = 10
    else:
        hlen = int.from_bytes(h[8:12], 'little'); off = 12
    meta = eval(h[off:off + hlen].decode('latin1'))          # numpy writes a literal dict
    if meta['fortran_order']:
        raise SystemExit('FATAL: fortran_order array; row offsets would not be contiguous')
    return meta['shape'], np.dtype(meta['descr']), off + hlen


def runs(idx, gap=16):
    """Merge sorted indices into contiguous [lo, hi] runs, tolerating small gaps to cut request count."""
    idx = np.unique(np.asarray(idx, np.int64))
    out, lo, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= gap:
            prev = i
        else:
            out.append((lo, prev)); lo = prev = i
    out.append((lo, prev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx_file', required=True, help='JSON list of pert_idx to fetch')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gap', type=int, default=16)
    a = ap.parse_args()

    want = sorted({int(v) for v in json.load(open(a.idx_file))})
    shape, dt, data0 = header()
    n, A, F = shape
    stride = A * F * dt.itemsize
    print('remote array %s %s, %d bytes/row, data starts at %d' % (shape, dt, stride, data0), flush=True)
    if max(want) >= n:
        raise SystemExit('FATAL: pert_idx %d out of range for %d drugs' % (max(want), n))

    rr = runs(want, a.gap)
    total = sum(hi - lo + 1 for lo, hi in rr)
    print('%d drugs wanted -> %d runs, %d rows fetched, %.0f MB over the wire, %.0f MB stored as float32'
          % (len(want), len(rr), total, total * stride / 1e6, len(want) * A * F * 4 / 1e6), flush=True)

    # Written straight to a memmap rather than accumulated in a dict: holding 1,970 x 122 x 514 float32
    # (494 MB) in RAM alongside the raw float64 blobs got this process KILLED at run 850/1063 after 40
    # minutes of downloading, with no traceback. On disk it also makes the fetch RESUMABLE, which matters
    # when a run costs 45 minutes of someone else's bandwidth.
    pos = {int(v): i for i, v in enumerate(want)}
    keep = set(want)
    mm_path = a.out + '.partial.f32'
    done_path = a.out + '.partial.done.npy'
    mm = np.memmap(mm_path, dtype=np.float32, mode=('r+' if os.path.exists(mm_path) else 'w+'),
                   shape=(len(want), A, F))
    done = (np.load(done_path) if os.path.exists(done_path)
            else np.zeros(len(want), bool))
    if done.any():
        print('resuming: %d/%d drugs already fetched' % (int(done.sum()), len(want)), flush=True)
    t0 = time.time()
    for k, (lo, hi) in enumerate(rr):
        if all(done[pos[i]] for i in range(lo, hi + 1) if i in keep):
            continue
        blob = curl(data0 + lo * stride, data0 + (hi + 1) * stride - 1)
        arr = np.frombuffer(blob, dtype=dt).reshape(hi - lo + 1, A, F)
        for j in range(hi - lo + 1):
            if lo + j in keep:
                mm[pos[lo + j]] = arr[j].astype(np.float32)
                done[pos[lo + j]] = True
        del arr, blob
        if (k + 1) % 25 == 0 or k + 1 == len(rr):
            mm.flush(); np.save(done_path, done)
            print('  run %d/%d  %d/%d drugs  %.0fs' % (k + 1, len(rr), int(done.sum()), len(want),
                                                       time.time() - t0), flush=True)
    mm.flush(); np.save(done_path, done)
    if not done.all():
        raise SystemExit('FATAL: %d drugs still missing after the sweep' % int((~done).sum()))
    feats = {int(v): np.asarray(mm[pos[int(v)]]) for v in want}

    # ---- offset check 1: a re-fetch of one row through a different alignment must be byte-identical ----
    probe = want[len(want) // 2]
    blob = curl(data0 + probe * stride, data0 + (probe + 1) * stride - 1)
    solo = np.frombuffer(blob, dtype=dt).reshape(A, F).astype(np.float32)
    if not np.array_equal(solo, feats[probe]):
        raise SystemExit('FATAL: row %d differs between a batched and a solo fetch -- offsets are wrong'
                         % probe)
    print('offset check 1 OK: row %d identical via two different ranges' % probe, flush=True)

    # ---- offset check 2: column 0 is a padding mask, so every row is 1s then 0s ----
    bad = []
    for i, v in feats.items():
        m = v[:, 0]
        if not set(np.unique(m).tolist()) <= {0.0, 1.0}:
            bad.append((i, 'mask not binary'))
        elif m.sum() and not np.array_equal(m, np.concatenate([np.ones(int(m.sum()), np.float32),
                                                               np.zeros(A - int(m.sum()), np.float32)])):
            bad.append((i, 'mask not a prefix run'))
    if bad:
        raise SystemExit('FATAL: %d rows have a malformed atom mask, e.g. %s' % (len(bad), bad[:3]))
    print('offset check 2 OK: all %d atom masks are prefix runs of 1s' % len(feats), flush=True)

    # ---- offset check 3: mask_len must equal min(all_atom_count + 2, atom capacity) ----
    # Uni-Mol tokenises every atom INCLUDING hydrogens and adds two special tokens, and the array holds at
    # most A=122 atom slots, so a correctly-strided row satisfies mask_len == min(n_atoms + 2, A) EXACTLY.
    # Two earlier versions of this check were wrong on correct data: the first compared against the
    # HEAVY-atom count, the second demanded a constant offset and so fired on the 30 molecules of 1,970
    # that are large enough to be TRUNCATED at the cap. The offsets were right both times. This form is
    # exact and covers both regimes, so a real stride error still cannot pass it.
    note = 'rdkit or SMILES unavailable -- check skipped'
    if os.path.exists(SMI):
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
            smi = np.load(SMI, allow_pickle=True).item()
            n_ok = n_trunc = 0
            bad = []
            for i in list(feats)[:400]:
                s_i = smi.get(i)
                m = Chem.MolFromSmiles(s_i) if s_i else None
                if m is None:
                    continue
                want_len = min(Chem.AddHs(m).GetNumAtoms() + 2, A)
                got_len = int(feats[i][:, 0].sum())
                if got_len != want_len:
                    bad.append((i, got_len, want_len))
                elif got_len == A:
                    n_trunc += 1
                else:
                    n_ok += 1
            if bad:
                raise SystemExit('FATAL: %d molecules have mask_len != min(n_atoms + 2, %d), e.g. %s '
                                 '-- the row stride is wrong' % (len(bad), A, bad[:3]))
            note = '%d molecules match n_atoms + 2 exactly, %d are truncated at the %d-atom cap' % (
                n_ok, n_trunc, A)
        except ImportError:
            pass
    print('offset check 3 OK: %s' % note, flush=True)

    np.savez(a.out, idx=np.array(sorted(feats), np.int64),
             feat=np.stack([feats[i] for i in sorted(feats)]))
    del feats, mm
    for p_tmp in (mm_path, done_path):
        try:
            os.remove(p_tmp)
        except OSError:
            pass
    print('wrote %s  (%.0f MB)' % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
