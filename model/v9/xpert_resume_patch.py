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

Environment: LINCS_STATE_DIR (write), LINCS_RESUME_DIR (read; absent on session 1), LINCS_FROZEN_PARAMS (JSON list),
LINCS_STOP_AFTER_EPOCH (tests only: exit cleanly after saving that many finished epochs).
"""
import io
import json
import os
import random
import shutil
import sys

import numpy as np
import torch

REG = {}
_STATE = {'first_train_checked': False, 'expect_first_epoch': None}


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


def save_state():
    d = os.environ['LINCS_STATE_DIR']
    os.makedirs(d, exist_ok=True)
    m, opt, sch, sc, st = REG['model'], REG['opt'], REG['sched'], REG.get('scaler'), REG['stopper']
    epoch = int(sch.last_epoch) - 1                 # LambdaLR counts the step just taken; the finished epoch is one less
    state = {'epoch': epoch,
             'model': {k: v.detach().cpu() for k, v in m.state_dict().items()},
             'opt': opt.state_dict(), 'sched': sch.state_dict(),
             'scaler': sc.state_dict() if sc is not None else None,
             'stopper': {'counter': st.counter, 'best_score': st.best_score, 'early_stop': st.early_stop},
             'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                     'numpy': np.random.get_state(), 'python': random.getstate()}}
    _atomic_save(state, os.path.join(d, 'full_state.pt'))
    _atomic_save({'epoch': epoch, 'model_state_dict': state['model']}, os.path.join(d, 'resume_from.pt'))
    if os.path.exists(st.filepath):
        tmp = os.path.join(d, 'best.pth.tmp')
        shutil.copyfile(st.filepath, tmp)
        os.replace(tmp, os.path.join(d, 'best.pth'))
    print('LINCS STATE SAVED epoch %d | best_score %r | counter %d' % (epoch, st.best_score, st.counter), flush=True)
    stop_after = os.environ.get('LINCS_STOP_AFTER_EPOCH')
    if stop_after is not None and epoch + 1 >= int(stop_after):
        print('LINCS_STOP_AFTER_EPOCH reached after epoch %d; exiting cleanly (test mode)' % epoch, flush=True)
        raise SystemExit(0)


def restore_state():
    d = os.environ['LINCS_RESUME_DIR']
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
    if os.path.exists(best):
        os.makedirs(os.path.dirname(stopper.filepath), exist_ok=True)
        shutil.copyfile(best, stopper.filepath)
    torch.set_rng_state(st['rng']['torch'])
    torch.cuda.set_rng_state_all(st['rng']['cuda'])
    np.random.set_state(st['rng']['numpy'])
    random.setstate(st['rng']['python'])
    _STATE['expect_first_epoch'] = st['epoch'] + 1
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
        if os.environ.get('LINCS_RESUME_DIR'):
            restore_state()
    _register(U.EarlyStopping, 'stopper', after=after_stopper)   # T.EarlyStopping is this same class object

    orig_train = T.train

    def train(*a, **k):
        if not _STATE['first_train_checked']:
            _STATE['first_train_checked'] = True
            exp = _STATE['expect_first_epoch']
            if exp is not None and k.get('epoch') != exp:
                raise RuntimeError('xpert_resume_patch: first epoch %r, expected %r' % (k.get('epoch'), exp))
        return orig_train(*a, **k)
    T.train = train
    return sorted(REG.keys())
