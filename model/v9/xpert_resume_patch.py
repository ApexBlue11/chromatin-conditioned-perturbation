# -*- coding: utf-8 -*-
"""Full-state checkpoint and resume for XPert's train_xpert.main(), applied at RUNTIME so their files stay verbatim.
[RESULTS 81.2; review 012 C1, C5; 78.5]  DRAFT pending review 014.

Their own --resume_from reloads model weights only: the Adam load is commented out (train_xpert.py:487), the LambdaLR is
rebuilt so its epoch count restarts at 0 (:460), EarlyStopping is constructed fresh (:524), and its key filter
(:484-486) silently matches nothing on a prefixed checkpoint while logging success (review 012 C1). A multi-session run
chained through it would not be their continuous training. This module makes it so:

  * Registration. XPertNet, torch.optim.Adam, LambdaLR, GradScaler and EarlyStopping record their instance at
    construction. Their main() builds exactly one of each per fold; a second one is refused.
  * Freezing [RESULTS 80.5, Amendment A]. The parameters named in LINCS_FROZEN_PARAMS -- those that receive grad None in
    the single-GPU recipe -- are set requires_grad=False at XPertNet construction, before the optimizer exists.
  * Save, at every epoch boundary: after lr_scheduler.step() (train_xpert.py:545), which runs after stopper.step()
    (:538), so every piece of state belongs to the same finished epoch. Written atomically (tmp + os.replace):
      full_state.pt   model / Adam / GradScaler / LambdaLR state dicts; stopper counter, best_score, early_stop;
                      torch CPU, CUDA (every device), numpy and python RNG states; the finished epoch index
      best.pth        the bytes of the stopper's on-disk best checkpoint, if it exists
      resume_from.pt  {'epoch', 'model_state_dict'} -- what their --resume_from reads, and only to set start_epoch
  * Restore, in the EarlyStopping.__init__ hook -- the last construction before their epoch loop, when every object
    exists: a STRICT model load (the loaded key set must equal the model's), then Adam, GradScaler, LambdaLR, the
    stopper's fields, the best checkpoint under THIS session's time-stamped folder, and the RNG states last, so the
    next draw is the next epoch's shuffle. Their filtered load ran first and is overwritten. The first train() call
    is asserted to be the epoch after the saved one.

Environment: LINCS_STATE_DIR (write), LINCS_RESUME_DIR (read; absent on session 1), LINCS_FROZEN_PARAMS (JSON list).
Test mode only [RESULTS 81.5, 81.3]: LINCS_STOP_AFTER_EPOCH (exit cleanly after saving that many finished epochs),
LINCS_TRUNCATE_BATCHES (train and validate on the first N batches of each epoch), LINCS_DUMP_AFTER_RESTORE (write the
live state right after restore, for the exact round-trip test), LINCS_DETERMINISTIC=1
(torch.use_deterministic_algorithms; the caller sets CUBLAS_WORKSPACE_CONFIG before CUDA initialises).

Production, RESULTS 84 [review 015]: LINCS_DEADLINE (unix time; after each save, stop cleanly if the slowest epoch
so far would pass it -- so a session only ever ends at an epoch boundary, 015 C2), LINCS_HORIZON_EPOCHS (stop, FINAL,
once that many epochs have completed, 84.1), LINCS_EXPECT_TORCH / LINCS_EXPECT_CUDA (the stack must not move between
sessions, 015 C3). Test only: LINCS_TEST_PATIENCE (the C4 chain test).

Review 014 C4: if --resume_from is on the command line and the strict restore has not run by the first train() call,
the trainer fails hard -- a hook that silently failed to fire would reproduce exactly review 012 C1's failure.
"""
import hashlib
import io
import itertools
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

REG = {}
_STATE = {'first_train_checked': False, 'expect_first_epoch': None, 'restored': False, 'resume_flag': False,
          'epoch_t0': None, 'epoch_durations': []}


def _register(cls, key, after=None):
    orig = cls.__init__
    if getattr(orig, '_lincs_resume', False):
        return

    def init(self, *a, **k):
        orig(self, *a, **k)
        if key in REG and REG[key] is not self:
            raise RuntimeError('xpert_resume_patch: a second %s was constructed; one per fold is expected' % key)
        REG[key] = self
        if after is not None:
            after(self)

    init._lincs_resume = True
    cls.__init__ = init


def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _freeze(model):
    names = json.loads(os.environ.get('LINCS_FROZEN_PARAMS', '[]'))
    params = dict(model.named_parameters())
    missing = [n for n in names if n not in params]
    if missing:
        raise RuntimeError('xpert_resume_patch: frozen names not in the model: %r' % missing)
    for n in names:
        params[n].requires_grad_(False)
    print('LINCS froze %d unused parameters' % len(names), flush=True)


def _sha1(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_state():
    """Everything restorable, in one structure -- used by the save AND by the post-restore dump, so the exact
    round-trip test of RESULTS 81.3a compares like with like, field for field."""
    m, opt, sch, sc, st = REG['model'], REG['opt'], REG['sched'], REG.get('scaler'), REG['stopper']
    return {'epoch': int(sch.last_epoch) - 1,       # LambdaLR counts the step just taken; the finished epoch is one less
            'model': {k: v.detach().cpu() for k, v in m.state_dict().items()},
            'opt': opt.state_dict(), 'sched': sch.state_dict(),
            'scaler': sc.state_dict() if sc is not None else None,
            'stopper': {'counter': st.counter, 'best_score': st.best_score, 'early_stop': st.early_stop},
            'best_sha1': _sha1(st.filepath),
            'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                    'numpy': np.random.get_state(), 'python': random.getstate()}}


def compare_states(saved, live, path='state'):
    """RESULTS 81.3a: every restorable field BITWISE equal. Returns the list of paths that differ (empty = pass).
    Tensors must match in dtype, shape and every bit; numpy arrays likewise; containers element for element."""
    bad = []
    if torch.is_tensor(saved) or torch.is_tensor(live):
        if not (torch.is_tensor(saved) and torch.is_tensor(live) and saved.dtype == live.dtype
                and saved.shape == live.shape and torch.equal(saved.cpu(), live.cpu())):
            bad.append(path)
    elif isinstance(saved, np.ndarray) or isinstance(live, np.ndarray):
        if not (isinstance(saved, np.ndarray) and isinstance(live, np.ndarray) and saved.dtype == live.dtype
                and np.array_equal(saved, live)):
            bad.append(path)
    elif isinstance(saved, dict):
        if not isinstance(live, dict) or set(saved) != set(live):
            bad.append(path + ' (keys)')
        else:
            for k in saved:
                bad += compare_states(saved[k], live[k], '%s.%s' % (path, k))
    elif isinstance(saved, (list, tuple)):
        if not isinstance(live, (list, tuple)) or len(saved) != len(live):
            bad.append(path + ' (length)')
        else:
            for i, (x, y) in enumerate(zip(saved, live)):
                bad += compare_states(x, y, '%s[%d]' % (path, i))
    elif saved != live:
        bad.append(path)
    return bad


def save_state():
    d = os.environ['LINCS_STATE_DIR']
    os.makedirs(d, exist_ok=True)
    st = REG['stopper']
    state = collect_state()
    epoch = state['epoch']
    _atomic_save(state, os.path.join(d, 'full_state.pt'))
    _atomic_save({'epoch': epoch, 'model_state_dict': state['model']}, os.path.join(d, 'resume_from.pt'))
    if os.path.exists(st.filepath):
        tmp = os.path.join(d, 'best.pth.tmp')
        shutil.copyfile(st.filepath, tmp)
        os.replace(tmp, os.path.join(d, 'best.pth'))
    now = time.time()
    if _STATE['epoch_t0'] is not None:
        _STATE['epoch_durations'].append(now - _STATE['epoch_t0'])
    _STATE['epoch_t0'] = now
    last_dur = _STATE['epoch_durations'][-1] if _STATE['epoch_durations'] else float('nan')
    print('LINCS STATE SAVED epoch %d | best_score %r | counter %d | epoch_s %.1f'
          % (epoch, st.best_score, st.counter, last_dur), flush=True)
    horizon = os.environ.get('LINCS_HORIZON_EPOCHS')
    if horizon is not None and epoch + 1 >= int(horizon):
        print('LINCS HORIZON REACHED %s' % json.dumps({'epoch': epoch, 'best_score': st.best_score,
                                                       'counter': int(st.counter)}), flush=True)
        raise SystemExit(0)
    deadline = os.environ.get('LINCS_DEADLINE')
    if deadline is not None and _STATE['epoch_durations'] and now + max(_STATE['epoch_durations']) > float(deadline):
        print('LINCS SESSION BOUNDARY %s' % json.dumps({'epoch': epoch, 'best_score': st.best_score,
                                                         'counter': int(st.counter),
                                                         'max_epoch_s': max(_STATE['epoch_durations'])}), flush=True)
        raise SystemExit(0)
    stop_after = os.environ.get('LINCS_STOP_AFTER_EPOCH')
    if stop_after is not None and epoch + 1 >= int(stop_after):
        print('LINCS_STOP_AFTER_EPOCH reached after epoch %d; exiting cleanly (test mode)' % epoch, flush=True)
        raise SystemExit(0)


def restore_state():
    d = os.environ['LINCS_RESUME_DIR']
    # map_location='cpu', independently of their load at train_xpert.py:483 [review 014 C4]; torch.set_rng_state needs a
    # CPU ByteTensor.
    st = torch.load(os.path.join(d, 'full_state.pt'), map_location='cpu', weights_only=False)
    m = REG['model']
    want, have = set(st['model']), set(m.state_dict())
    if want != have:
        raise RuntimeError('xpert_resume_patch: key sets differ (missing %d, unexpected %d) -- refusing [review 012 C1]'
                           % (len(have - want), len(want - have)))
    m.load_state_dict(st['model'], strict=True)
    REG['opt'].load_state_dict(st['opt'])
    REG['sched'].load_state_dict(st['sched'])
    if st['scaler'] is not None:
        REG['scaler'].load_state_dict(st['scaler'])
    stopper = REG['stopper']
    stopper.counter, stopper.best_score, stopper.early_stop = (st['stopper']['counter'], st['stopper']['best_score'],
                                                               st['stopper']['early_stop'])
    best = os.path.join(d, 'best.pth')
    # 015 C2: the three per-epoch files are each atomic but not atomic together; refuse an inconsistent set.
    if _sha1(best) != st['best_sha1']:
        raise RuntimeError('xpert_resume_patch: best.pth sha1 %s != saved best_sha1 %s -- inconsistent state set'
                           % (_sha1(best), st['best_sha1']))
    rf = torch.load(os.path.join(d, 'resume_from.pt'), map_location='cpu', weights_only=False)
    if int(rf['epoch']) != int(st['epoch']):
        raise RuntimeError('xpert_resume_patch: resume_from epoch %s != full_state epoch %s' % (rf['epoch'], st['epoch']))
    if os.path.exists(best):
        os.makedirs(os.path.dirname(stopper.filepath), exist_ok=True)
        shutil.copyfile(best, stopper.filepath)
    torch.set_rng_state(st['rng']['torch'])
    torch.cuda.set_rng_state_all(st['rng']['cuda'])
    np.random.set_state(st['rng']['numpy'])
    random.setstate(st['rng']['python'])
    _STATE['expect_first_epoch'] = st['epoch'] + 1
    # 015 C3: the exact round-trip of RESULTS 81.3a, IN PROCESS, every session: any difference is fatal.
    live = collect_state()
    diff = compare_states(st, live)
    if diff:
        raise RuntimeError('xpert_resume_patch: restored state differs from the saved state in %r' % diff[:10])
    print('LINCS RESTORE ROUND-TRIP exact: every field bitwise equal', flush=True)
    _STATE['restored'] = True
    dump = os.environ.get('LINCS_DUMP_AFTER_RESTORE')
    if dump:
        live = collect_state()
        live['start_epoch_expected'] = _STATE['expect_first_epoch']
        _atomic_save(live, dump)
    print('LINCS RESUME restored epoch %d | best_score %r | counter %d | best checkpoint %s'
          % (st['epoch'], stopper.best_score, stopper.counter, 'restored' if os.path.exists(best) else 'none yet'),
          flush=True)


def apply():
    """Install the hooks. Must run after their modules are importable and before train_xpert.main()."""
    import models.model_XPert as MX
    import utils as U
    import train_xpert as T
    from torch.optim.lr_scheduler import LambdaLR
    from torch.cuda.amp import GradScaler

    _STATE['resume_flag'] = any(a == '--resume_from' or a.startswith('--resume_from=') for a in sys.argv)
    if _STATE['resume_flag'] and not os.environ.get('LINCS_RESUME_DIR'):
        raise RuntimeError('xpert_resume_patch: --resume_from given without LINCS_RESUME_DIR; their filtered load '
                           'must never be the only restore [review 012 C1, 014 C4]')
    for var, have in (('LINCS_EXPECT_TORCH', torch.__version__), ('LINCS_EXPECT_CUDA', str(torch.version.cuda))):
        want = os.environ.get(var)
        if want is not None and want != have:
            raise RuntimeError('xpert_resume_patch: %s is %s, this session has %s -- the stack moved [015 C3]'
                               % (var, want, have))
    if os.environ.get('LINCS_DETERMINISTIC') == '1':
        torch.use_deterministic_algorithms(True)
    _register(MX.XPertNet, 'model', after=_freeze)
    _register(torch.optim.Adam, 'opt')
    _register(GradScaler, 'scaler')

    # The save hook wraps LambdaLR.step on the CLASS and acts only for the registered instance. Wrapping it on the
    # instance would put a local function into __dict__, which LambdaLR.state_dict() copies -- and torch.save cannot
    # pickle it. The constructor's own initial step runs before registration, so it never saves.
    _register(LambdaLR, 'sched')
    if not getattr(LambdaLR.step, '_lincs_resume', False):
        orig_step = LambdaLR.step

        def step(self, *a, **k):
            r = orig_step(self, *a, **k)
            if self is REG.get('sched') and os.environ.get('LINCS_STATE_DIR'):
                save_state()
            return r
        step._lincs_resume = True
        LambdaLR.step = step

    def after_stopper(self):
        tp = os.environ.get('LINCS_TEST_PATIENCE')
        if tp is not None:                       # TEST ONLY -- the kernel asserts it absent in production
            self.patience = int(tp)
        if os.environ.get('LINCS_RESUME_DIR'):
            restore_state()
    _register(U.EarlyStopping, 'stopper', after=after_stopper)   # T.EarlyStopping is this same class object

    # EARLY-STOP MARKER [packet 015]. When THEIR stopper reports early stopping, the session is the FINAL one: print a
    # marker line the kernel's watchdog acts on, and write it to the state dir, so a final session can never be
    # mistaken for a resume point. Wrapped on the class, acting only for the registered instance.
    if not getattr(U.EarlyStopping.step, '_lincs_resume', False):
        orig_es_step = U.EarlyStopping.step

        def es_step(self, score, model, current_epoch, optimizer, *a, **k):
            stop = orig_es_step(self, score, model, current_epoch, optimizer, *a, **k)
            if stop and self is REG.get('stopper'):
                info = {'epoch': int(current_epoch), 'best_score': self.best_score, 'counter': int(self.counter)}
                d = os.environ.get('LINCS_STATE_DIR')
                if d:
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, 'early_stop.json'), 'w') as f:
                        json.dump(info, f)
                print('LINCS EARLY STOP %s' % json.dumps(info), flush=True)
            return stop
        es_step._lincs_resume = True
        U.EarlyStopping.step = es_step

    orig_train, orig_validate = T.train, getattr(T, 'validate', None)
    trunc = os.environ.get('LINCS_TRUNCATE_BATCHES')

    def train(*a, **k):
        if not _STATE['first_train_checked']:
            _STATE['first_train_checked'] = True
            _STATE['epoch_t0'] = time.time()
            if _STATE['resume_flag'] and not _STATE['restored']:
                raise RuntimeError('xpert_resume_patch: --resume_from is set but the strict restore did not run before '
                                   'the first train() call -- refusing [review 014 C4]')
            exp = _STATE['expect_first_epoch']
            if exp is not None and k.get('epoch') != exp:
                raise RuntimeError('xpert_resume_patch: first epoch %r, expected %r' % (k.get('epoch'), exp))
        if trunc:
            a = list(a)
            a[2] = itertools.islice(a[2], int(trunc))       # train(model, opt, dataloader, ...)
        return orig_train(*a, **k)

    def validate(*a, **k):
        a = list(a)
        a[1] = itertools.islice(a[1], int(trunc))           # validate(model, dataloader, ...)
        return orig_validate(*a, **k)
    T.train = train
    if trunc and orig_validate is not None:
        T.validate = validate
    return sorted(REG.keys())
