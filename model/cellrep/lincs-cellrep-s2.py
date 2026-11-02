# Cell-representation programme, stage 2 (RESULTS 103.6 / 103.7): runs this session's jobs from stage2_jobs.json on the two
# T4s (one worker per GPU, data loaded once per worker), writing per-job JSON + predictions and a COMPLETE marker with the
# sha1s of the code and data it read. SESSION is set when the kernel is rendered.
import glob, hashlib, json, os, sys, time, traceback
import multiprocessing as mp

SESSION = '__SESSION__'
ROOTS = ['/kaggle/input']
OUT = '/kaggle/working/out'


def find(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    if not hits:
        raise SystemExit(f'FATAL: {name} missing')
    return hits[0]


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def worker(gpu, jobs, src):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sys.path.insert(0, src)
    import torch
    import stage2
    D = stage2.Data(ROOTS)
    for j in jobs:
        t0 = time.time()
        try:
            tag, _ = stage2.run_job(j, ROOTS, OUT, 'cuda', D=D)
            print(f'[gpu{gpu}] {tag} done {time.time() - t0:.0f}s', flush=True)       # never a score (review 111 C2)
        except Exception:
            err = traceback.format_exc()
            print(f'[gpu{gpu}] FAILED {j}\n{err}', flush=True)
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, stage2.job_tag(j) + '.FAILED'), 'w').write(err)


if __name__ == '__main__':
    t0 = time.time()
    src = os.path.dirname(find('stage2.py'))
    jobs = json.load(open(find('stage2_jobs.json')))['sessions'][SESSION]
    import torch
    n_gpu = torch.cuda.device_count()
    print(f'session {SESSION}: {len(jobs)} jobs on {n_gpu} GPUs', flush=True)
    if n_gpu < 2:
        raise SystemExit('FATAL: fewer than 2 GPUs')
    halves = [jobs[0::2], jobs[1::2]]
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(g, halves[g], src)) for g in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    files = ['stage1.py', 'stage2.py', 'stage2_jobs.json', 'stage2_graph.npz', 'l1000_genetic_lm978.npz', 'ccle_extra.npz',
             'ccle_expr_v11.npz', 'v10c_chromatin.npz', 'v11_chromatin.npz', 'gene_vectors_v11_full.npy']
    done = sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{OUT}/*.json'))
    failed = sorted(os.path.basename(p)[:-7] for p in glob.glob(f'{OUT}/*.FAILED'))
    json.dump({'session': SESSION, 'n_jobs': len(jobs), 'done': len(done), 'failed': failed,
               'pins': {f: sha1(find(f)) for f in files}, 'elapsed_s': round(time.time() - t0)},
              open('/kaggle/working/STAGE2_COMPLETE.json', 'w'), indent=1)
    print(f'done {len(done)} failed {len(failed)} in {time.time() - t0:.0f}s', flush=True)
