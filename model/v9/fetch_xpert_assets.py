# -*- coding: utf-8 -*-
"""
Fetch the XPert assets their code needs but their GitHub release does not ship, WITHOUT downloading the
whole 1.6 GB archive.

Why this exists. `external/xpert/code/XPert/processed_data/` in the release contains a single gitkeep.txt.
Their `train_xpert.py` cannot run without it, so their released checkpoint cannot be evaluated on rows of
our choosing -- and evaluating their checkpoint on the SAME rows as ours is the only comparison with no
confound left in it. RESULTS 40 shows what happens without that: two numbers measured on different rows
were read as a model difference, in both directions, twice.

The assets live in a Zenodo record as members of one zip. Zenodo honours HTTP range requests, so we read
the zip's central directory (~1 KB at the end of the file), then range-fetch ONLY the members we need:

    l1000_mdmt_68830_subset.h5ad   714 MB compressed   their main mdmt benchmark + its split columns
    PPI_gene_vector_128d.npy         1 MB              cell_emb.linear expects this exact 128-d input
    l1000_gene_info_978.csv          - MB              gene order, to prove our gene axis is theirs

and skip l1000_sdst_78453.h5ad (759 MB), all_drugs_idx2KPGT.npy (90 MB) and all_drugs_idx2morgan.npy,
which their unimol-trained checkpoint never reads. ~800 MB instead of 1.6 GB, on a disk with 28 GB free.

Every member is CRC32-verified against the value in the central directory before it is put in place, so a
truncated or proxy-mangled download fails here rather than silently becoming a number in a table.

    python model/v9/fetch_xpert_assets.py --list
    python model/v9/fetch_xpert_assets.py --fetch l1000_mdmt_68830_subset.h5ad PPI_gene_vector_128d.npy
"""
import os, sys, json, zlib, struct, time, hashlib, argparse, subprocess

RECORD = '17182939'                    # 10.5281/zenodo.17182939 -- XPert v1.1 (15357711 redirects here)
ZIP_URL = 'https://zenodo.org/api/records/%s/files/processed_data.zip/content' % RECORD
UNIMOL_URL = 'https://zenodo.org/api/records/%s/files/all_drugs_unimol_arr.npy/content' % RECORD
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'
DEST = r'C:\Projects\LINCS\external\xpert\code\XPert\processed_data'


def curl(url, start=None, end=None, out=None, tries=4):
    """Range GET. Zenodo occasionally answers 403/202 under load; retry rather than write a short file."""
    cmd = ['curl', '-sL', '--fail', '-m', '3600', '-A', UA]
    if start is not None:
        cmd += ['-H', 'Range: bytes=%d-%d' % (start, end)]
    if out:
        cmd += ['-o', out]
    cmd.append(url)
    for i in range(tries):
        r = subprocess.run(cmd, capture_output=not out)
        if r.returncode == 0:
            got = os.path.getsize(out) if out else len(r.stdout)
            want = None if start is None else end - start + 1
            if want is None or got == want:
                return True if out else r.stdout
            print('  short read %d/%d, retry %d' % (got, want, i + 1), flush=True)
        else:
            print('  curl rc=%d, retry %d' % (r.returncode, i + 1), flush=True)
        time.sleep(3 + 5 * i)
    raise SystemExit('FATAL: could not fetch %s [%s-%s] after %d tries' % (url, start, end, tries))


def content_length(url):
    out = subprocess.run(['curl', '-sIL', '-m', '120', '-A', UA, url], capture_output=True, text=True).stdout
    n = None
    for line in out.splitlines():
        if line.lower().startswith('content-length:'):
            n = int(line.split(':')[1].strip())
    if not n:
        raise SystemExit('FATAL: no content-length; cannot locate the zip central directory')
    return n


def central_directory(url):
    """Parse the zip's central directory from its last 128 KB. Returns one dict per member."""
    n = content_length(url)
    tail = curl(url, max(0, n - 131072), n - 1)
    p = tail.rfind(b'PK\x05\x06')
    if p < 0:
        raise SystemExit('FATAL: no End-Of-Central-Directory record; not a zip, or a proxy rewrote it')
    cd_size, cd_off = struct.unpack('<II', tail[p + 12:p + 20])
    if 0xFFFFFFFF in (cd_size, cd_off):
        q = tail.rfind(b'PK\x06\x06')
        cd_size, cd_off = struct.unpack('<QQ', tail[q + 40:q + 56])
    cd = curl(url, cd_off, cd_off + cd_size - 1)
    ents, i = [], 0
    while i + 46 <= len(cd) and cd[i:i + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[i + 10:i + 12])
        crc, csize, usize = struct.unpack('<III', cd[i + 16:i + 28])
        nlen, elen, clen = struct.unpack('<HHH', cd[i + 28:i + 34])
        lho, = struct.unpack('<I', cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode('utf-8', 'replace')
        ents.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, lho=lho))
        i += 46 + nlen + elen + clen
    if not ents:
        raise SystemExit('FATAL: central directory parsed to zero entries')
    return ents, n


def fetch_member(url, ent, dest_dir, work):
    """Range-fetch one zip member, inflate it, and verify CRC32 before it is allowed into place."""
    base = os.path.basename(ent['name'])
    final = os.path.join(dest_dir, base)
    # the local header is 30 bytes + name + extra, and its extra length may DIFFER from the central one
    lh = curl(url, ent['lho'], ent['lho'] + 29)
    if lh[:4] != b'PK\x03\x04':
        raise SystemExit('FATAL: %s: local header signature missing at offset %d' % (base, ent['lho']))
    nlen, elen = struct.unpack('<HH', lh[26:30])
    data_off = ent['lho'] + 30 + nlen + elen
    raw = os.path.join(work, base + '.deflate')
    print('  %s: %.1f MB compressed -> %.1f MB' % (base, ent['csize'] / 1e6, ent['usize'] / 1e6), flush=True)
    t0 = time.time()
    curl(url, data_off, data_off + ent['csize'] - 1, out=raw)
    print('    downloaded in %.0fs, inflating' % (time.time() - t0), flush=True)

    tmp = final + '.part'
    d = zlib.decompressobj(-zlib.MAX_WBITS) if ent['method'] == 8 else None
    crc = 0
    h = hashlib.sha256()
    with open(raw, 'rb') as fi, open(tmp, 'wb') as fo:
        while True:
            chunk = fi.read(1 << 22)
            if not chunk:
                break
            out = d.decompress(chunk) if d else chunk
            crc = zlib.crc32(out, crc)
            h.update(out)
            fo.write(out)
        if d:
            out = d.flush()
            crc = zlib.crc32(out, crc)
            h.update(out)
            fo.write(out)
    os.remove(raw)
    got = os.path.getsize(tmp)
    if got != ent['usize']:
        os.remove(tmp)
        raise SystemExit('FATAL: %s: inflated to %d bytes, central directory says %d'
                         % (base, got, ent['usize']))
    if (crc & 0xFFFFFFFF) != ent['crc']:
        os.remove(tmp)
        raise SystemExit('FATAL: %s: CRC32 %08x != %08x -- corrupt' % (base, crc & 0xFFFFFFFF, ent['crc']))
    os.replace(tmp, final)
    print('    OK, CRC32 %08x verified' % ent['crc'], flush=True)
    return dict(name=ent['name'], bytes=got, crc32='%08x' % ent['crc'], sha256=h.hexdigest(),
                source=url, fetched=time.strftime('%Y-%m-%dT%H:%M:%S'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true', help='print the archive contents and exit')
    ap.add_argument('--fetch', nargs='*', default=None, help='member basenames to fetch')
    ap.add_argument('--dest', default=DEST)
    ap.add_argument('--work', default=None, help='scratch dir for the compressed bytes')
    a = ap.parse_args()

    ents, total = central_directory(ZIP_URL)
    if a.list or a.fetch is None:
        print('zenodo record %s  processed_data.zip  %.1f MB, %d members\n' % (RECORD, total / 1e6, len(ents)))
        for e in ents:
            print('  %-46s %8.1f MB compressed  %8.1f MB on disk'
                  % (os.path.basename(e['name']), e['csize'] / 1e6, e['usize'] / 1e6))
        return

    os.makedirs(a.dest, exist_ok=True)
    work = a.work or a.dest
    os.makedirs(work, exist_ok=True)
    by_base = {os.path.basename(e['name']): e for e in ents if e['usize'] > 0}
    prov_path = os.path.join(a.dest, 'PROVENANCE.json')
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {'record': RECORD, 'files': {}}

    for name in a.fetch:
        if name not in by_base:
            raise SystemExit('FATAL: %s is not in the archive; --list shows what is' % name)
        final = os.path.join(a.dest, name)
        if os.path.exists(final) and os.path.getsize(final) == by_base[name]['usize'] \
                and name in prov['files']:
            print('  %s: already present and verified, skipping' % name, flush=True)
            continue
        prov['files'][name] = fetch_member(ZIP_URL, by_base[name], a.dest, work)
        json.dump(prov, open(prov_path, 'w'), indent=1)
    print('\nprovenance -> %s' % prov_path)


if __name__ == '__main__':
    main()
