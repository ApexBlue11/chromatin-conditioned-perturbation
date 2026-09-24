# -*- coding: utf-8 -*-
"""Local test of xpert_resume_patch's mechanics, CPU, deterministic -- so resume equivalence can be demanded BIT FOR BIT
here, before the GPU test of RESULTS 81.3 (where atomics make it only approximate).

Their real `utils.EarlyStopping` is used (scanpy / unimol_tools stubbed as empty modules: unused by it). The model and
the loop are stand-ins that keep what the patch depends on: a class named XPertNet in `models.model_XPert`, a module
`train_xpert` with a global `train()` called with `epoch=`, and their loop's end-of-epoch order -- train, validate,
stopper.step, (break on early stop), lr_scheduler.step (train_xpert.py:528-545). Dropout is ON and the loader shuffles,
so the RNG restore is exercised.

    straight: 4 epochs in one process
    resumed : 2 epochs (stop) -> a FRESH process restores -> 2 more epochs
    must be : identical parameters, identical per-epoch losses, identical stopper state, bit for bit
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
XPERT = r'C:\Projects\LINCS\external\xpert\code\XPert'

CHILD = r'''
import os, sys, types, json
for name in ('scanpy', 'unimol_tools'):
    m = types.ModuleType(name); m.UniMolRepr = None; sys.modules[name] = m
sys.path.insert(0, %(xpert)r); sys.path.insert(0, %(here)r)
import torch, numpy as np
import utils as U                                   # THEIR EarlyStopping

models = types.ModuleType('models'); MX = types.ModuleType('models.model_XPert')
class XPertNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(6, 16); self.d = torch.nn.Dropout(0.3); self.b = torch.nn.Linear(16, 1)
        self.unused = torch.nn.Linear(3, 3)
    def forward(self, x):
        return self.b(self.d(torch.relu(self.a(x)))).squeeze(-1)
MX.XPertNet = XPertNet; models.model_XPert = MX
sys.modules['models'] = models; sys.modules['models.model_XPert'] = MX

T = types.ModuleType('train_xpert'); sys.modules['train_xpert'] = T
g = torch.Generator().manual_seed(1)
X = torch.randn(96, 6, generator=g); Y = (X[:, 0] - 0.5 * X[:, 1]) + 0.1 * torch.randn(96, generator=g)
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=16, shuffle=True)
def train(model, opt, loader, scaler=None, epoch=0):
    model.train(); tot = 0.0
    for x, y in loader:
        opt.zero_grad(); l = ((model(x) - y) ** 2).mean(); l.backward(); opt.step(); tot += float(l)
    return tot
def validate(model):
    model.eval()
    with torch.no_grad():
        return float(((model(X) - Y) ** 2).mean())
T.train = train
def main(n_epochs, folder, resume_from=None):
    torch.manual_seed(2024)
    model = XPertNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0 if e < 2 else 0.5)
    start = 0
    if resume_from:
        start = torch.load(resume_from)['epoch'] + 1              # what their --resume_from does, :489
    os.makedirs(folder, exist_ok=True)                           # their main creates expt_folder first
    stopper = U.EarlyStopping(mode='lower', metric='mse', patience=50, n_fold=0, folder=folder)
    log = []
    for epoch in range(start, n_epochs):
        tl = T.train(model, opt, dl, epoch=epoch)                  # looked up on the module, as their main does
        vl = validate(model)
        early = stopper.step(vl, model, epoch, opt)
        log.append([epoch, tl, vl])
        if early:
            break
        sch.step()
    return model, stopper, log
T.main = main

import xpert_resume_patch as RP
os.environ['LINCS_FROZEN_PARAMS'] = json.dumps(['unused.weight', 'unused.bias'])
RP.apply()
out = os.environ['OUT']
try:
    model, stopper, log = main(4, os.environ['FOLDER'], os.environ.get('RESUME_FROM'))
    status = 'finished'
except SystemExit:
    model, stopper, log = RP.REG['model'], RP.REG['stopper'], None
    status = 'stopped'
torch.save({'params': {k: v.clone() for k, v in model.state_dict().items()}, 'log': log, 'status': status,
            'stopper': [stopper.counter, stopper.best_score, stopper.early_stop],
            'frozen_grad_none': all(p.grad is None for n, p in model.named_parameters() if n.startswith('unused'))}, out)
'''


def run(tmp, tag, state_dir, resume_dir=None, stop_after=None, resume_from=None):
    code = CHILD % {'xpert': XPERT, 'here': HERE}
    env = dict(os.environ, OUT=os.path.join(tmp, tag + '.pt'), FOLDER=os.path.join(tmp, tag + '_folder'),
               LINCS_STATE_DIR=state_dir)
    for k, v in (('LINCS_RESUME_DIR', resume_dir), ('LINCS_STOP_AFTER_EPOCH', stop_after), ('RESUME_FROM', resume_from)):
        if v is not None:
            env[k] = str(v)
        else:
            env.pop(k, None)
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    print(r.stdout.strip())
    for marker in ('LINCS SESSION BOUNDARY', 'LINCS HORIZON REACHED'):
        if marker in r.stdout:
            print('   [%s seen]' % marker)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit('child %s failed' % tag)
    import torch
    return torch.load(env['OUT'], weights_only=False)


def main():
    import torch
    tmp = tempfile.mkdtemp()
    straight = run(tmp, 'straight', os.path.join(tmp, 'st_straight'))
    first = run(tmp, 'first', os.path.join(tmp, 'st_resumed'), stop_after=2)
    second = run(tmp, 'second', os.path.join(tmp, 'st_resumed2'), resume_dir=os.path.join(tmp, 'st_resumed'),
                 resume_from=os.path.join(tmp, 'st_resumed', 'resume_from.pt'))
    assert first['status'] == 'stopped' and second['status'] == 'finished'
    same = all(torch.equal(straight['params'][k], second['params'][k]) for k in straight['params'])
    print('parameters identical bit for bit:', same)
    print('straight log  :', [[e, round(t, 6), round(v, 6)] for e, t, v in straight['log']])
    print('resumed log   :', [[e, round(t, 6), round(v, 6)] for e, t, v in second['log']])
    print('stopper straight %r | resumed %r' % (straight['stopper'], second['stopper']))
    assert same, 'resumed parameters differ from the straight run'
    assert straight['log'][2:] == second['log'], 'per-epoch losses after the resume differ'
    assert straight['stopper'] == second['stopper'], 'stopper state differs'
    assert straight['frozen_grad_none'] and second['frozen_grad_none'], 'frozen parameters received gradients'
    # a key-set mismatch must be refused, never silently tolerated [review 012 C1]
    st = torch.load(os.path.join(tmp, 'st_resumed', 'full_state.pt'), weights_only=False)
    st['model']['module.' + next(iter(st['model']))] = st['model'].pop(next(iter(st['model'])))
    bad = os.path.join(tmp, 'st_bad'); os.makedirs(bad)
    torch.save(st, os.path.join(bad, 'full_state.pt'))
    try:
        run(tmp, 'bad', os.path.join(tmp, 'st_bad_out'), resume_dir=bad,
            resume_from=os.path.join(tmp, 'st_resumed', 'resume_from.pt'))
        raise AssertionError('a prefixed key set was NOT refused')
    except SystemExit as e:
        print('prefixed key set refused, as required:', e)
    # RESULTS 81.3a: the live state right after restore equals the saved state, bit for bit, every field
    dump = os.path.join(tmp, 'dump_after_restore.pt')
    os.environ['LINCS_DUMP_AFTER_RESTORE'] = dump
    try:
        run(tmp, 'rt', os.path.join(tmp, 'st_rt'), resume_dir=os.path.join(tmp, 'st_resumed'),
            resume_from=os.path.join(tmp, 'st_resumed', 'resume_from.pt'))
    finally:
        del os.environ['LINCS_DUMP_AFTER_RESTORE']
    sys.path.insert(0, HERE)
    import xpert_resume_patch as RP
    saved = torch.load(os.path.join(tmp, 'st_resumed', 'full_state.pt'), weights_only=False)
    live = torch.load(dump, weights_only=False)
    assert live.pop('start_epoch_expected') == saved['epoch'] + 1
    diff = RP.compare_states(saved, live)
    print('81.3a round-trip, fields differing:', diff)
    assert diff == [], diff
    # and the comparator is not vacuous: one flipped Adam moment must be caught
    k0 = next(iter(live['opt']['state']))
    live['opt']['state'][k0]['exp_avg'] = live['opt']['state'][k0]['exp_avg'].clone()
    live['opt']['state'][k0]['exp_avg'].view(-1)[0] += 1e-7
    assert RP.compare_states(saved, live) == ['state.opt.state.%s.exp_avg' % k0], RP.compare_states(saved, live)
    print('comparator catches a 1e-7 change in one Adam moment: True')

    # review 014 C4: --resume_from on the command line without the strict restore must fail hard
    code = ("import sys, types\nfor n in ('scanpy','unimol_tools'):\n m=types.ModuleType(n); m.UniMolRepr=None; "
            "sys.modules[n]=m\nsys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import utils\nMX=types.ModuleType('models.model_XPert'); MX.XPertNet=type('XPertNet',(object,),{})\n"
            "sys.modules['models']=types.ModuleType('models'); sys.modules['models.model_XPert']=MX\n"
            "T=types.ModuleType('train_xpert'); T.train=lambda *a, **k: None; sys.modules['train_xpert']=T\n"
            "sys.argv=['train_xpert.py','--resume_from','x.pt']\nimport xpert_resume_patch as RP\nRP.apply()\n"
            % (XPERT, HERE))
    env = dict(os.environ); env.pop('LINCS_RESUME_DIR', None)
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'must never be the only restore' in r.stderr, r.stderr[-600:]
    print('C4: --resume_from without the strict restore fails hard: True')
    code2 = code + "T.train(None, None, [], epoch=1)\n"      # the restore hook never fired before the first train()
    env2 = dict(os.environ, LINCS_RESUME_DIR=tmp)
    r = subprocess.run([sys.executable, '-c', code2], env=env2, capture_output=True, text=True)
    assert r.returncode != 0 and 'did not run before the first train' in r.stderr, r.stderr[-600:]
    print('C4: restore hook never fired before the first train() -> fails hard: True')
    # packet 015: THEIR stopper reporting early stopping must print the marker and write early_stop.json
    code3 = ("import sys, types, os, json\nfor n in ('scanpy','unimol_tools'):\n m=types.ModuleType(n); m.UniMolRepr=None; "
             "sys.modules[n]=m\nsys.path.insert(0, %r); sys.path.insert(0, %r)\nimport torch, utils\n"
             "MX=types.ModuleType('models.model_XPert'); MX.XPertNet=type('XPertNet',(object,),{})\n"
             "sys.modules['models']=types.ModuleType('models'); sys.modules['models.model_XPert']=MX\n"
             "T=types.ModuleType('train_xpert'); T.train=lambda *a, **k: None; sys.modules['train_xpert']=T\n"
             "import xpert_resume_patch as RP\nRP.apply()\nd=os.environ['FOLDER']; os.makedirs(d, exist_ok=True)\n"
             "st=utils.EarlyStopping(mode='lower', metric='mse', patience=1, n_fold=0, folder=d, logger=__import__('logging').getLogger('t'))\n"
             "m=torch.nn.Linear(2,2); o=torch.optim.Adam(m.parameters())\n"
             "r=[]\nfor e, v in enumerate([1.0, 2.0, 3.0]):\n    r.append(st.step(v, m, e, o))\n    if r[-1]: break\nprint('STEPS', r)\n"
             % (XPERT, HERE))
    env3 = dict(os.environ, FOLDER=os.path.join(tmp, 'es_folder'), LINCS_STATE_DIR=os.path.join(tmp, 'es_state'))
    r = subprocess.run([sys.executable, '-c', code3], env=env3, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert 'LINCS EARLY STOP' in r.stdout, r.stdout[-800:]
    mk = json.load(open(os.path.join(tmp, 'es_state', 'early_stop.json')))
    assert mk['epoch'] == 1 and mk['counter'] == 1, mk
    print('early-stop marker printed and written:', mk)
    # RESULTS 84.2 (015 C2): a deadline already passed ends the session at the FIRST epoch boundary, cleanly
    import time as _time
    os.environ['LINCS_DEADLINE'] = str(_time.time())
    try:
        bd = run(tmp, 'boundary', os.path.join(tmp, 'st_boundary'))
    finally:
        del os.environ['LINCS_DEADLINE']
    assert bd['status'] == 'stopped', bd['status']
    stb = torch.load(os.path.join(tmp, 'st_boundary', 'full_state.pt'), weights_only=False)
    assert stb['epoch'] == 0, stb['epoch']
    # and a session resumed from that boundary reaches the straight run's end state, bit for bit
    cont = run(tmp, 'cont', os.path.join(tmp, 'st_cont'), resume_dir=os.path.join(tmp, 'st_boundary'),
               resume_from=os.path.join(tmp, 'st_boundary', 'resume_from.pt'))
    assert all(torch.equal(straight['params'][k], cont['params'][k]) for k in straight['params'])
    print('deadline stops at an epoch boundary, and the resumed session equals the straight run: True')
    # RESULTS 84.1: the horizon ends the run, FINAL, after that many completed epochs
    os.environ['LINCS_HORIZON_EPOCHS'] = '3'
    try:
        hz = run(tmp, 'horizon', os.path.join(tmp, 'st_horizon'))
    finally:
        del os.environ['LINCS_HORIZON_EPOCHS']
    sth = torch.load(os.path.join(tmp, 'st_horizon', 'full_state.pt'), weights_only=False)
    assert hz['status'] == 'stopped' and sth['epoch'] == 2, (hz['status'], sth['epoch'])
    print('horizon of 3 epochs stops after epoch 2: True')
    print('RESUME MECHANICS TEST PASSED')


if __name__ == '__main__':
    main()
