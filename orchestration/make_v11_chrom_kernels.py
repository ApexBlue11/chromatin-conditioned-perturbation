"""v11 genome-wide chromatin (RESULTS 101.1, the principal's approval of 2026-10-21): the v10-C L2/L3 kernels re-run on the
STRING genes with a MANE Select TSS (model/results/v11/tss_hg38_string.tsv). Same ChIP-Atlas tracks (the v10-C selection),
same five windows, same CCLE RRBS file; only the gene table differs. All CPU with internet, all outcome-free.
The v10-C builder is reused unchanged; this wrapper only renames outputs and the gene-count check.

  python orchestration/make_v11_chrom_kernels.py --selection model/results/v10c/chipatlas_selection.json \
      --cells model/results/v10c/full_cells.json
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_v10c_data_kernels as base  # noqa: E402

REPO = base.REPO
SRC_DIR = os.path.join(REPO, 'external', 'kaggle_v11_chromsrc')
KERN_DIR = base.KERN_DIR
DATASET = 'apexblue/lincs-v11-chromsrc'
TSS = os.path.join(REPO, 'model', 'results', 'v11', 'tss_hg38_string.tsv')
V10C_ASSERT = "assert len(GENES) == 977, len(GENES)  # 978 landmarks; MUC1 has no MANE Select TSS"


def n_genes(tss):
    with open(tss, encoding='utf-8') as f:
        return sum(1 for _ in f) - 1


def templates(n):
    assert base.COMMON.count(V10C_ASSERT) == 1
    common = base.COMMON.replace(V10C_ASSERT, f"assert len(GENES) == {n}, len(GENES)  # v11: STRING genes with a MANE Select TSS")
    signal = base.SIGNAL.replace('v10c_signal_s', 'v11_signal_s').replace('V10C_SIGNAL_COMPLETE', 'V11_SIGNAL_COMPLETE')
    methyl = base.METHYL.replace('v10c_methyl', 'v11_methyl').replace('V10C_METHYL_COMPLETE', 'V11_METHYL_COMPLETE')
    for t, old in ((signal, 'v10c_signal'), (methyl, 'v10c_methyl')):
        assert old not in t
    return common, signal, methyl


def metadata(name):
    m = base.metadata(name)
    m['dataset_sources'] = [DATASET]
    return m


def build(selection, cells_json, tss=TSS, signal_py=None, src_dir=SRC_DIR, kern_dir=KERN_DIR):
    signal_py = signal_py or os.path.join(REPO, 'model', 'v10c', 'chromatin_signal.py')
    os.makedirs(src_dir, exist_ok=True)
    files = {'selection.json': selection, 'full_cells.json': cells_json, 'tss_hg38.tsv': tss, 'chromatin_signal.py': signal_py}
    for nm, p in files.items():
        shutil.copyfile(p, os.path.join(src_dir, nm))
    with open(os.path.join(src_dir, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'title': DATASET.split('/')[1], 'id': DATASET, 'licenses': [{'name': 'other'}]}, f, indent=1)
    pins = {nm: base.sha1(os.path.join(src_dir, nm)) for nm in files}
    common, signal, methyl = templates(n_genes(tss))
    common = common.replace('__PINS__', repr(pins))
    built = {}
    jobs = [(f'lincs-v11-signal-s{k}', signal.replace('__SHARD__', str(k))) for k in range(base.N_SHARD)]
    jobs.append(('lincs-v11-methyl', methyl))
    for name, body in jobs:
        body = (body.replace('__N_SHARD__', str(base.N_SHARD)).replace('__BUDGET_S__', repr(base.BUDGET_S))
                .replace('__BW_URL__', repr(base.BW_URL)).replace('__PYBIGWIG__', repr(base.PYBIGWIG))
                .replace('__CCLE_URL__', repr(base.CCLE_URL)))
        assert '__' not in body.replace('__name__', '').replace('__version__', ''), name
        d = os.path.join(kern_dir, 'kern_' + name.replace('lincs-', '').replace('-', '_'))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name + '.py'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(common + body)
        with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata(name), f, indent=1)
        built[name] = d
    return pins, built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', required=True)
    ap.add_argument('--cells', required=True)
    a = ap.parse_args()
    pins, built = build(a.selection, a.cells)
    print(json.dumps(pins, indent=1))
    for nm, d in built.items():
        print(nm, d)


if __name__ == '__main__':
    main()
