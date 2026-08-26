# -*- coding: utf-8 -*-
"""
Stage the COMPLETE v9 Kaggle bundle: every file train_v9_gpu.py needs, flat, in one directory.

Deliberately self-contained rather than layered on `apexblue/lincs-train-bundle`. That bundle is missing
`sig_strength.npy` and `scaffold_split.json`, which is precisely the incident the handoff records -- a
rebuilt Kaggle dataset silently dropped the compound holdout AND reliability weighting, and nothing in the
logs said so. A run that depends on which of several datasets happens to be attached is a run that can
degrade quietly, so v9 ships one dataset with everything and `check_inputs_v9` refuses to train if any of
it is missing.

fp16 for the three big arrays. The Level-5 target is a z-score and the atom reprs are features under AMP,
both already fp16-safe. For Level-3 log expression in [0, 15] the fp16 step at value 8 is 0.0039, so a
delta reconstructed from two fp16 values carries ~0.006 of quantisation error against a signal of 0.377
and a MEASURED control noise of 0.185 -- about 3 % of the noise already in the data. `check_inputs_v9`
verifies mean|delta| after the round trip, so a gross corruption cannot pass as precision loss.

Usage:
    python model/v9/assemble_bundle_v9.py <out_dir>
    kaggle datasets create -p <out_dir> -t        (PRIVATE by default; -u would publish it)
    kaggle datasets version -p <out_dir> -t -m "..."                          (to update in place)

-t (--keep-tabular) is REQUIRED. Without it Kaggle converts .tsv files to CSV on upload, and
signatures_usable.tsv is parsed as tab-delimited -- the run would read one column and fail, or worse,
mis-parse. -u is --public, not --unzip.
"""
import os, sys, json, shutil, hashlib

import numpy as np

ROOT = r'C:\Projects\LINCS'
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get('TEMP', '.'), 'lincs_v9_bundle')

#   (path relative to ROOT, destination basename, mode)
FILES = [
    # --- the v9 Level-3 substrate [RESULTS 27.1] ---
    ('phase2_assembly/outputs/level3_sig/X_trt_l3.npy', 'X_trt_l3.npy', 'fp16'),
    ('phase2_assembly/outputs/level3_sig/X_ctl_l3.npy', 'X_ctl_l3.npy', 'fp16'),
    ('phase2_assembly/outputs/level3_sig/l3_covered.npy', 'l3_covered.npy', 'copy'),
    ('phase2_assembly/outputs/level3_sig/l3_yrow.npy', 'l3_yrow.npy', 'copy'),
    ('phase2_assembly/outputs/level3_sig/l3_coverage.json', 'l3_coverage.json', 'copy'),
    # --- the Level-5 target, kept so v3-v7 numbers stay comparable ---
    ('phase2_assembly/outputs/Y_target_level5_978.npy', 'Y_target_level5_978.npy', 'fp16'),
    ('phase2_assembly/outputs/signatures_usable.tsv', 'signatures_usable.tsv', 'copy'),
    # --- WITHOUT THESE TWO the run silently loses reliability weighting and the compound holdout ---
    ('phase2_assembly/outputs/sig_strength.npy', 'sig_strength.npy', 'copy'),
    ('drug/outputs/splits/scaffold_split.json', 'scaffold_split.json', 'copy'),
    # --- chromatin ---
    ('phase2_assembly/outputs/E_final.npy', 'E_final.npy', 'copy'),
    ('phase2_assembly/outputs/E_final_mask.npy', 'E_final_mask.npy', 'copy'),
    ('phase2_assembly/outputs/E_reliability.tsv', 'E_reliability.tsv', 'copy'),
    # --- v9 priors, full proteome + HGNC-current symbols [RESULTS 27.3] ---
    ('network/outputs/v9/M_pathway_v9.npy', 'M_pathway_v9.npy', 'copy'),
    ('network/outputs/v9/pathway_info_v9.tsv', 'pathway_info_v9.tsv', 'copy'),
    ('network/outputs/v9/STRING_adj_978_v9.npy', 'STRING_adj_978_v9.npy', 'copy'),
    ('network/outputs/v9/gene_vectors_978.npy', 'gene_vectors_978.npy', 'copy'),
    ('network/outputs/v9/landmark_symbols_v9.tsv', 'landmark_symbols_v9.tsv', 'copy'),
    ('Data Info/pathway_landmark_genes.txt', 'pathway_landmark_genes.txt', 'copy'),
    # --- cell context ---
    ('baseline/outputs/ccle_baseline_lincs_v5/lincs_cell_index.json', 'lincs_cell_index.json', 'copy'),
    ('baseline/outputs/cellfeat/cell_lineage.npy', 'cell_lineage.npy', 'copy'),
    # CCLE: default OFF (redundant and dominated, RESULTS 27.4) but shipped so --use_ccle stays runnable
    ('baseline/outputs/ccle_baseline_lincs_v5/X_base_lincs.npy', 'X_base_lincs.npy', 'copy'),
    # --- drug features ---
    ('drug/outputs/drug_feature_index.json', 'drug_feature_index.json', 'copy'),
    ('drug/outputs/drug_descriptors.npy', 'drug_descriptors.npy', 'copy'),
    ('drug/outputs/drug_fingerprints.npy', 'drug_fingerprints.npy', 'copy'),
    ('drug/outputs/drug_unimol.npy', 'drug_unimol.npy', 'copy'),
    ('drug/outputs/drug_atom_reprs.npy', 'drug_atom_reprs.npy', 'fp16'),
    ('drug/outputs/drug_atom_offsets.npy', 'drug_atom_offsets.npy', 'copy'),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest, total, missing = {}, 0, []
    for src, dst, mode in FILES:
        s = os.path.join(ROOT, src)
        if not os.path.exists(s):
            missing.append(src)
            continue
        d = os.path.join(OUT, dst)
        if mode == 'fp16':
            a = np.load(s, mmap_mode='r')
            np.save(d, np.asarray(a, np.float16))
            del a
        else:
            shutil.copy2(s, d)
        sz = os.path.getsize(d)
        total += sz
        h = hashlib.sha256()
        with open(d, 'rb') as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        manifest[dst] = dict(source=src, mode=mode, bytes=sz, sha256=h.hexdigest())
        print(f'{dst:34s} {sz / 1e6:9.1f} MB  ({mode})', flush=True)
    if missing:
        raise SystemExit('FATAL: missing sources, refusing to build a partial bundle:\n  - '
                         + '\n  - '.join(missing))

    # round-trip check on the fp16 arrays: quantisation must not have changed what the data MEANS
    trt = np.load(os.path.join(OUT, 'X_trt_l3.npy'), mmap_mode='r')
    ctl = np.load(os.path.join(OUT, 'X_ctl_l3.npy'), mmap_mode='r')
    cov = np.load(os.path.join(OUT, 'l3_covered.npy'))
    idx = np.flatnonzero(cov)[:20000]
    d16 = np.asarray(trt[idx], np.float32) - np.asarray(ctl[idx], np.float32)
    t32 = np.load(os.path.join(ROOT, 'phase2_assembly/outputs/level3_sig/X_trt_l3.npy'), mmap_mode='r')
    c32 = np.load(os.path.join(ROOT, 'phase2_assembly/outputs/level3_sig/X_ctl_l3.npy'), mmap_mode='r')
    d32 = np.asarray(t32[idx], np.float32) - np.asarray(c32[idx], np.float32)
    err = float(np.abs(d16 - d32).max())
    print(f'\nfp16 round trip: mean|delta| {np.abs(d32).mean():.4f} -> {np.abs(d16).mean():.4f}, '
          f'max element error {err:.5f} (control noise measured at 0.185)')
    if err > 0.05:
        raise SystemExit(f'FATAL: fp16 delta error {err:.4f} is too large; ship fp32 instead.')

    manifest['_meta'] = dict(total_bytes=total, n_files=len(manifest),
                             fp16_delta_max_error=round(err, 6),
                             mean_abs_delta_fp32=round(float(np.abs(d32).mean()), 4),
                             mean_abs_delta_fp16=round(float(np.abs(d16).mean()), 4))
    json.dump(manifest, open(os.path.join(OUT, 'bundle_manifest.json'), 'w'), indent=2)
    json.dump({'title': 'LINCS v9 bundle', 'id': 'apexblue/lincs-v9-bundle',
               'licenses': [{'name': 'CC0-1.0'}]},
              open(os.path.join(OUT, 'dataset-metadata.json'), 'w'), indent=2)
    print(f'\n{len(FILES)} files, {total / 1e9:.2f} GB -> {OUT}')
    print('next:  kaggle datasets create -t -p "%s"' % OUT)
    print('       (-t keeps .tsv as .tsv; NO -u, which would make it public)')


if __name__ == '__main__':
    main()
