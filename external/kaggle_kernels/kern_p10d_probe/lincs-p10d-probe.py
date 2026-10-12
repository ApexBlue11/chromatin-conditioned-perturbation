# P10 D probe (RESULTS 98.15 rulings 4-5): on the pinned image, which of PertDiT's dependencies exist, does PertDiT's
# environment's transformers stack install, what are the pinned HuggingFace revisions' file sha256s, and how long does one
# MolT5-large caption take on Kaggle CPU (their exact call: beam 5, max_length 512)? Public SMILES only; no LINCS data is read.
import importlib, json, os, subprocess, sys, time
print('python', sys.version.split()[0], 'cpus', os.cpu_count(), flush=True)
try:
    print('mem GB', round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1), flush=True)
except Exception as e:
    print('mem ?', e)
for m in ['torch', 'diffusers', 'einops', 'rdkit', 'transformers', 'tokenizers', 'sentencepiece', 'huggingface_hub',
          'accelerate', 'safetensors', 'scanpy', 'anndata', 'numpy', 'scipy', 'pandas']:
    try:
        print('HAVE', m, getattr(importlib.import_module(m), '__version__', '?'), flush=True)
    except Exception as e:
        print('MISSING', m, repr(e)[:120], flush=True)
pins = ['transformers==4.39.1', 'tokenizers==0.15.2', 'sentencepiece==0.2.0', 'huggingface-hub==0.24.6']
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pins, capture_output=True, text=True)
print('pip rc', r.returncode, (r.stdout + r.stderr)[-1500:], flush=True)
for m in ['transformers', 'tokenizers', 'sentencepiece', 'huggingface_hub', 'torch']:
    importlib.invalidate_caches()
print('AFTER', subprocess.run([sys.executable, '-c', 'import transformers, tokenizers, huggingface_hub, torch;'
                               'print(transformers.__version__, tokenizers.__version__, huggingface_hub.__version__, torch.__version__)'],
                              capture_output=True, text=True).stdout.strip(), flush=True)
REPOS = {'laituan245/molt5-large-smiles2caption': 'e307e7a1f959d4c9ba6553ce656440cd5d3b4660',
         'michiyasunaga/BioLinkBERT-large': '1eb6d81c5fc1c42d3a43c71956b0e526558ae053'}
code = r'''
import json, sys, time, torch
from huggingface_hub import HfApi
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
REPOS = json.loads(sys.argv[1])
api = HfApi()
for repo, rev in REPOS.items():
    info = api.model_info(repo, revision=rev, files_metadata=True)
    print('REV', repo, info.sha, 'pinned', rev, 'match', info.sha == rev, flush=True)
    for s in info.siblings:
        print('FILE', repo, s.rfilename, s.size, (s.lfs or {}).get('sha256') if isinstance(s.lfs, dict) else getattr(s.lfs, 'sha256', None), flush=True)
print('torch threads', torch.get_num_threads(), flush=True)
smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
          'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2c(O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)c(O)cc(O)c12']
t0 = time.time()
rev = REPOS['laituan245/molt5-large-smiles2caption']
tok = T5Tokenizer.from_pretrained('laituan245/molt5-large-smiles2caption', model_max_length=512, revision=rev)
m = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-smiles2caption', revision=rev).eval()
print('molt5 load s %.0f' % (time.time() - t0), flush=True)
caps = []
with torch.no_grad():
    for s in smiles:
        t = time.time()
        ids = tok(s, return_tensors='pt').input_ids
        out = m.generate(ids, num_beams=5, max_length=512)
        caps.append(tok.decode(out[0], skip_special_tokens=True))
        print('CAPTION smiles_len %d out_tokens %d s %.1f' % (len(s), out.shape[1], time.time() - t), flush=True)
del m
rev = REPOS['michiyasunaga/BioLinkBERT-large']
tb = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large', revision=rev)
mb = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large', revision=rev).eval()
with torch.no_grad():
    for c in caps + ['The dosage is 10.0 micromoles.']:
        t = time.time()
        o = mb(**tb(c, return_tensors='pt')).last_hidden_state.squeeze(0)
        print('BERT shape', tuple(o.shape), 's %.2f' % (time.time() - t), flush=True)
'''
r = subprocess.run([sys.executable, '-c', code, json.dumps(REPOS)], capture_output=True, text=True)
print(r.stdout[-6000:], flush=True)
print('rc', r.returncode, r.stderr[-3000:], flush=True)
