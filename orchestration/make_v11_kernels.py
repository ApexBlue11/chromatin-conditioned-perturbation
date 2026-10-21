"""Build the v11 GPU kernels (RESULTS 102 / 102.1): 2 x T4, internet off, every input sha1-pinned and mounted once.
make_v10c_kernels' template and retry rule (100.8), with train_v11 as the entry point.

  python orchestration/make_v11_kernels.py timing                  -> kern_v11_timing (arm full, 300 steps)
  python orchestration/make_v11_kernels.py run --tag v11_full_s0 --args "--dev --seed 0 --arm full --guard"
  python orchestration/make_v11_kernels.py run --tag v11_full_s0 --args "..." --start_attempt 1 --kernel_name lincs-v11-full-s0-r1

The source dataset apexblue/lincs-v11-src carries all code (v7, v9, v10-C base, v11) and FG's data files; the substrate is
apexblue/lincs-v10c-full; drug features and the base's priors come from apexblue/lincs-v9-bundle with the audit99 pins.
apexblue/lincs-v10c-src is NOT mounted (its copies of the base code would be second copies).
"""
import argparse, json, os, shlex, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'orchestration'))
import make_v10c_kernels as mk  # noqa: E402

SRC_DIR = os.path.join(REPO, 'external', 'kaggle_v11_src')
CODE = {'model_v11.py': 'model/v11', 'data_v11.py': 'model/v11', 'train_v11.py': 'model/v11',
        'model_v10c.py': 'model/v10c', 'data_v10c.py': 'model/v10c', 'train_v10c.py': 'model/v10c',
        'modules_v9.py': 'model/v9', 'modules_v7.py': 'model/v7'}
SMALL = {'v10c_chromatin.npz': 'external/v10c_chromatin.npz', 'cell_tissue.json': 'model/results/v10c/cell_tissue.json',
         'string_graph_v11_expdb.npz': 'network/outputs/v11/string_graph_v11_expdb.npz',
         'gene_vectors_v11_full.npy': 'network/outputs/v11/gene_vectors_full.npy',
         'v11_chromatin.npz': 'external/v11_data/derived/v11_chromatin.npz',
         'ccle_expr_v11.npz': 'external/v11_data/derived/ccle_expr_v11.npz',
         'chembl_dti_edges.tsv': 'drug/outputs/dti/chembl_dti_edges.tsv',
         'drug_inchikeys.tsv': 'drug/outputs/dti/drug_inchikeys.tsv',
         'action_sign_map.json': 'model/results/v11/action_sign_map.json',
         'dti_probe_split.json': 'model/results/v11/dti_probe_split.json'}
DATASETS = ['apexblue/lincs-v11-src', 'apexblue/lincs-v10c-full', 'apexblue/lincs-v9-bundle']
ENTRY_OLD, ENTRY_NEW = "'model/v10c/train_v10c.py'", "'model/v11/train_v11.py'"
DONE_OLD, DONE_NEW = "'V10C_KERNEL_COMPLETE.json'", "'V11_KERNEL_COMPLETE.json'"
HEAD_OLD, HEAD_NEW = 'RESULTS 100.4, v10-C.', 'RESULTS 102 / 102.1, v11.'


def template():
    t = mk.TEMPLATE
    for old, new in ((ENTRY_OLD, ENTRY_NEW), (DONE_OLD, DONE_NEW), (HEAD_OLD, HEAD_NEW)):
        if t.count(old) != 1:
            raise SystemExit(f'FATAL: the v10-C template changed ({old})')
        t = t.replace(old, new)
    return t


def build_src(src_dir=SRC_DIR):
    os.makedirs(src_dir, exist_ok=True)
    for n, sub in CODE.items():
        shutil.copyfile(os.path.join(REPO, sub, n), os.path.join(src_dir, n))
    for n, rel in SMALL.items():
        shutil.copyfile(os.path.join(REPO, rel), os.path.join(src_dir, n))
    with open(os.path.join(src_dir, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'title': 'lincs-v11-src', 'id': 'apexblue/lincs-v11-src', 'licenses': [{'name': 'other'}]}, f, indent=1)
    pins = {n: mk.sha1(os.path.join(src_dir, n)) for n in list(CODE) + list(SMALL)}
    pins[mk.SUBSTRATE[0]] = mk.sha1(os.path.join(REPO, mk.SUBSTRATE[1]))
    probe = json.load(open(mk.PROBE, encoding='utf-8'))
    for n in mk.FROM_V9:
        if probe[n]['copies'] != 1:
            raise SystemExit(f'FATAL: {n} has {probe[n]["copies"]} mounted copies in the probe')
        pins[n] = probe[n]['sha1'][0]
    return pins


def render(name, runs, pins, kern_dir=mk.KERN_DIR, run_h=4.5, start_attempt=0):
    saved = (mk.TEMPLATE, mk.CODE, mk.DATASETS)
    try:
        mk.TEMPLATE, mk.CODE, mk.DATASETS = template(), CODE, DATASETS
        return mk.render(name, runs, pins, kern_dir=kern_dir, run_h=run_h, start_attempt=start_attempt)
    finally:
        mk.TEMPLATE, mk.CODE, mk.DATASETS = saved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['timing', 'run'])
    ap.add_argument('--tag')
    ap.add_argument('--args', default='')
    ap.add_argument('--run_h', type=float, default=4.5, help='projected hours of one run (100.8 retry budget)')
    ap.add_argument('--start_attempt', type=int, default=0, help='100.8 retry in a new kernel (review 090 C4)')
    ap.add_argument('--kernel_name', help='a new-kernel retry: e.g. lincs-v11-full-s0-r1, keeping --tag as the slot')
    a = ap.parse_args(argv)
    pins = build_src()
    if a.mode == 'timing':
        runs = [['timing_full', ['--dev', '--timing_steps', '300', '--d_model', '256', '--arm', 'full']]]
        d = render('lincs-v11-timing', runs, pins)
    else:
        if not a.tag:
            raise SystemExit('--tag is required for a run kernel')
        name = a.kernel_name or 'lincs-' + a.tag.replace('_', '-')
        if a.start_attempt and not a.kernel_name:
            raise SystemExit('FATAL: a new-kernel retry (--start_attempt) needs --kernel_name ...-r<k>')
        d = render(name, [[a.tag, shlex.split(a.args)]], pins, run_h=a.run_h, start_attempt=a.start_attempt)
    print(json.dumps(pins, indent=1))
    print(d)


if __name__ == '__main__':
    main()
