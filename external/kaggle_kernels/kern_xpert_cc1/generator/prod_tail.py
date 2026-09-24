RECORD['train_cmd'] = ' '.join(cmd)
RECORD['published_command_source'] = ('scripts/train.sh:15 (mdmt) and README; fold list reduced to %s; '
                                      'folds independent per train_xpert.py:425-455' % FOLD)
RECORD['seed_note'] = ('seed 2024 set once at train_xpert.py:402 before the fold loop, so this fold starts '
                       'from a fresh seed-2024 state, not the state their 3-fold sequential run would reach')
# ========================================================================================================
# O2 PRODUCTION SESSION [RESULTS 84; review 015]
#   FINAL terminations are exactly two (84.1): their early stopping (the resume patch prints LINCS EARLY STOP), or
#   the 297-epoch horizon (LINCS HORIZON REACHED). A session otherwise ends ONLY at an epoch boundary, cleanly
#   (LINCS SESSION BOUNDARY), before the deadline. Anything else -- crash, watchdog backstop, quota kill -- is
#   INCOMPLETE and is never read through 71.7. Between sessions only timing, state integrity and quota may inform a
#   decision; never the logged loss.
# ========================================================================================================
import glob
import hashlib
import threading
assert not MEASURE_ONLY
STATE = os.path.join(W, 'state')
os.makedirs(STATE, exist_ok=True)
TEST_VARS = ('LINCS_STOP_AFTER_EPOCH', 'LINCS_TRUNCATE_BATCHES', 'LINCS_DUMP_AFTER_RESTORE', 'LINCS_DETERMINISTIC',
             'LINCS_TEST_PATIENCE')
STACK = {'torch': torch.__version__, 'cuda': str(torch.version.cuda)}
RECORD['session'] = {'index': SESSION, 'horizon_epochs': HORIZON_EPOCHS, 'stack': STACK, 'prev': PREV}
cmd_prod = [sys.executable, '-u', 'run_train_dp.py'] + cmd[3:]
DEADLINE = T0 + BUDGET_H * 3600
ENV_PROD = {k: v for k, v in ENV.items() if k not in TEST_VARS}
ENV_PROD.update(LINCS_STATE_DIR=STATE, LINCS_FROZEN_PARAMS=json.dumps(FROZEN),
                LINCS_HORIZON_EPOCHS=str(HORIZON_EPOCHS), LINCS_DEADLINE=str(DEADLINE),
                LINCS_EXPECT_TORCH=(PREV or STACK)['torch'], LINCS_EXPECT_CUDA=(PREV or STACK)['cuda'])


def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


STATE_FILES = ('full_state.pt', 'resume_from.pt', 'best.pth')
if SESSION > 1:
    # 015 C3: the attached state must be byte-for-byte the state session k-1 handed off, whose sha1s are literals here.
    if STACK != {'torch': PREV['torch'], 'cuda': PREV['cuda']}:
        fatal('the stack moved between sessions: %s -> %s. Stop and return [015 C3].' % (PREV, STACK))
    fs = glob.glob('/kaggle/input/**/full_state.pt', recursive=True)
    if len(fs) != 1:
        fatal('expected exactly one attached full_state.pt, found %d' % len(fs))
    RESUME_DIR = os.path.dirname(fs[0])
    got = {f: _sha1(os.path.join(RESUME_DIR, f)) for f in STATE_FILES}
    if got != PREV['sha1']:
        fatal('attached state sha1s %s != session %d handoff literals %s' % (got, SESSION - 1, PREV['sha1']))
    ENV_PROD['LINCS_RESUME_DIR'] = RESUME_DIR
    cmd_prod = cmd_prod + ['--resume_from', os.path.join(RESUME_DIR, 'resume_from.pt')]
    RECORD['session']['resume_dir'] = RESUME_DIR
    RECORD['session']['attached_sha1_verified'] = True
    log('SESSION %d resumes from epoch %d; attached state verified against the git-committed handoff'
        % (SESSION, PREV['final_epoch']))


def run_trainer(argv, env, tag, backstop):
    """Run their main() through run_train_dp.py, parse the marker lines, and return what ended it. GUARD E (executed
    args) applies to every run."""
    rec = {'tag': tag, 'outcome': None, 'marker': None, 'saved': [], 'epoch_s': [], 'last_epoch_index': None,
           'last_counter_logged': None}
    logf = os.path.join(W, 'train_%s.log' % tag)
    with open(logf, 'w') as lf:
        p = subprocess.Popen(argv, cwd=X, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        # The line loop can only check the clock when the trainer prints; a SILENT hang would run into Kaggle's
        # hard limit, whose kill discards /kaggle/working -- the state included. A timer kills it regardless.
        killer = threading.Timer(max(1.0, backstop - time.time()), p.kill)
        killer.daemon = True
        killer.start()
        in_args, seen_args, args_checked = False, {}, False
        for line in p.stdout:
            lf.write(line)
            if not args_checked:
                if '---------args-----------' in line:
                    in_args = True
                    continue
                if in_args:
                    if ':' in line:
                        k, v = line.split(':', 1)
                        seen_args[k.strip()] = v.strip()
                    elif not line.strip() and seen_args:
                        in_args, args_checked = False, True
                        bad = {k: (seen_args.get(k), v) for k, v in EXPECT_ARGS.items() if seen_args.get(k) != v}
                        rec['executed_args'] = seen_args
                        if bad:
                            p.terminate()
                            fatal('GUARD E (%s): executed args differ from the published recipe: %s' % (tag, bad))
            if line.startswith('LINCS STATE SAVED'):
                rec['saved'].append(line.strip())
                try:
                    rec['epoch_s'].append(float(line.rsplit('epoch_s', 1)[1]))
                except (IndexError, ValueError):
                    pass
            if 'Valid Total Loss' in line:
                try:
                    rec['last_epoch_index'] = int(line.split('Epoch ')[1].split(',')[0])
                except (IndexError, ValueError):
                    pass
                n = len(rec['saved']) + 1
                if n <= 3 or n % 10 == 0:
                    log(tag, 'epoch', rec['last_epoch_index'], '|', line.strip()[-120:])
            if 'EarlyStopping counter:' in line:
                try:
                    rec['last_counter_logged'] = int(line.split('EarlyStopping counter:')[1].split('out of')[0])
                except (IndexError, ValueError):
                    pass
            for key, outcome in (('LINCS EARLY STOP ', 'early_stop'), ('LINCS HORIZON REACHED ', 'horizon'),
                                 ('LINCS SESSION BOUNDARY ', 'boundary')):
                if line.startswith(key):
                    rec['outcome'], rec['marker'] = outcome, json.loads(line[len(key):])
                    log(tag, line.strip())
            if 'Traceback' in line or 'Error' in line:
                log(tag, line.strip()[-300:])
            if rec['outcome'] == 'early_stop':
                # 015 ask 2: terminate at the marker; their post-loop test pass is never read (decisions_locked).
                p.terminate()
                break
            if time.time() > backstop:
                rec['outcome'] = 'watchdog_backstop'
                log(tag, 'WATCHDOG BACKSTOP fired: the epoch-boundary stop did not act in time.')
                p.terminate()
                break
        try:
            rc = p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
    killer.cancel()
    if rc is not None and rc < 0 and time.time() >= backstop and rec['outcome'] is None:
        rec['outcome'] = 'watchdog_backstop'
    rec['returncode'] = rc
    if rec['outcome'] is None:
        # C4: rc 0 without a marker means their main() ran to its end unseen -- never a silent "finished".
        rec['outcome'] = 'exited_without_marker' if rc == 0 else ('killed_by_signal' if rc < 0 else 'crashed')
    return rec


def best_checkpoint():
    cks = glob.glob(os.path.join(X, 'experiment', '**', '%s_fold_early_stop.pth' % FOLD), recursive=True)
    if not cks:
        fatal('no best checkpoint written by their stopper')
    return max(cks, key=os.path.getmtime)


def finalize(rec, patience, tag):
    """The two FINAL terminations only (84.1). Convergence record, 71.7, and our row-indexed prediction."""
    assert rec['outcome'] in ('early_stop', 'horizon'), rec['outcome']
    ck_path = best_checkpoint()
    best_epoch = int(torch.load(ck_path, map_location='cpu', weights_only=False).get('epoch', -1))
    last_epoch_index = int(rec['marker']['epoch'])          # from the MARKER (015 ask 2), not from the state dir
    counter_at_end = last_epoch_index - best_epoch
    out = {'stopped_by': 'finished' if rec['outcome'] == 'early_stop' else 'horizon',
           'best_checkpoint': ck_path, 'best_epoch': best_epoch, 'last_epoch_index': last_epoch_index,
           'counter_at_end': counter_at_end, 'marker': rec['marker'], 'patience': patience,
           'best_selected_before_init_epoch_70': best_epoch < 70}
    if rec['outcome'] == 'early_stop':
        # 84.1(4): by construction, and verified against utils.py's counter >= patience
        if not (counter_at_end == patience == int(rec['marker']['counter'])):
            fatal('%s: counter_at_end %d, patience %d, marker counter %s disagree' % (tag, counter_at_end, patience,
                                                                                    rec['marker']['counter']))
        out['admissible_for_v9_win'] = True
    else:
        out['admissible_for_v9_win'] = counter_at_end >= 45           # 71.7 applied to the final horizon stop
    prof_path = os.path.join(W, 'xpert_trained_%s_%s_test_profile.npy' % (FOLD, tag))
    env2 = dict(ENV, XPERT_DIR=X, XPERT_CKPT=ck_path)
    pr = subprocess.run([sys.executable, '-u', os.path.join(V9, 'xpert_native_eval.py'), '--nfold', FOLD,
                         '--rows', 'test', '--device', 'cuda', '--batch', '128', '--out', prof_path],
                        capture_output=True, text=True, env=env2)
    if pr.returncode != 0 or not os.path.exists(prof_path):
        print(pr.stdout[-3000:], pr.stderr[-3000:])
        fatal('%s: prediction with the best checkpoint failed' % tag)
    prof = np.load(prof_path, allow_pickle=True).item()
    if 'row_index' not in prof or len(prof['row_index']) != 21321:
        fatal('%s: profile has %s rows / row_index present=%s' % (tag, len(prof.get('y_pred', [])), 'row_index' in prof))
    out['profile'] = {'path': prof_path, 'n': int(len(prof['row_index']))}
    return out


# ---- C4 (015): the one new path, end to end, in test mode, BEFORE real training -- session 1 only ------------------
if SESSION == 1:
    chain_state = os.path.join(W, 'chain_state')
    env_c = dict(ENV_PROD, LINCS_STATE_DIR=chain_state, LINCS_TRUNCATE_BATCHES='5', LINCS_TEST_PATIENCE='1',
                 LINCS_STOP_AFTER_EPOCH='25')
    for k in ('LINCS_DEADLINE', 'LINCS_HORIZON_EPOCHS'):
        env_c.pop(k)
    before = set(glob.glob(os.path.join(X, 'experiment', '*', '*')))
    rc_chain = run_trainer(cmd_prod, env_c, 'chaintest', backstop=time.time() + 1800)
    if rc_chain['outcome'] != 'early_stop':
        RECORD['chain_test'] = rc_chain
        fatal('C4 chain test: expected an early stop with patience 1 in <= 25 truncated epochs, got %s'
              % rc_chain['outcome'])
    ft = finalize(rc_chain, patience=1, tag='chaintest')
    RECORD['chain_test'] = {'trainer': {k: rc_chain[k] for k in ('outcome', 'marker', 'returncode', 'last_epoch_index')},
                            'finalize': ft, 'PASS': True}
    log('C4 chain test PASSED: marker -> termination -> counter_at_end == patience -> 21321-row prediction')
    # remove every chain-test artefact so the real run's checkpoint can never be confused with it
    for d in set(glob.glob(os.path.join(X, 'experiment', '*', '*'))) - before:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(chain_state, ignore_errors=True)
    os.remove(ft['profile']['path'])

# ---- the real training session ----------------------------------------------------------------------------------
for k in TEST_VARS:
    assert k not in ENV_PROD, 'test-mode variable %s in the production environment' % k
log('TRAIN (session %d):' % SESSION, ' '.join(cmd_prod))
res = run_trainer(cmd_prod, ENV_PROD, 'session%d' % SESSION, backstop=DEADLINE + 1800)
RECORD['host_memory'] = mem_summary()
sess = {'index': SESSION, 'outcome': res['outcome'], 'returncode': res['returncode'], 'marker': res['marker'],
        'epochs_this_session': len(res['saved']), 'first_saved': res['saved'][:1], 'last_saved': res['saved'][-1:],
        'epoch_s': res['epoch_s'], 'wall_h': round((time.time() - T0) / 3600, 3),
        'executed_args': res.get('executed_args')}
if SESSION == 1 and len(res['epoch_s']) >= 2:
    first2 = sum(res['epoch_s'][:2]) / 2
    sess['amendment_E'] = {'first_two_mean_s': round(first2, 1), 'projection_s': 482.2,
                           'reprice_before_session_2': first2 > 1.25 * 482.2}
RECORD['session_log'] = sess
log('session %d ended:' % SESSION, res['outcome'], '| epochs this session', len(res['saved']))

if res['outcome'] in ('early_stop', 'horizon'):
    fin = finalize(res, patience=50, tag='final')
    RECORD.update(fin)
    shutil.copy(fin['best_checkpoint'], os.path.join(W, 'xpert_trained_%s.pth' % FOLD))
    RECORD['framing'] = ('XPert trained to its published recipe on %s. NOT a reproduction of their cold-cell '
                         'run: an independent draw of the recipe.' % FOLD)
    log('FINAL (%s): best epoch %d, last %d, counter %d, admissible %s' % (fin['stopped_by'], fin['best_epoch'],
        fin['last_epoch_index'], fin['counter_at_end'], fin['admissible_for_v9_win']))
elif res['outcome'] == 'boundary':
    RECORD['stopped_by'] = 'session_boundary'
    RECORD['handoff'] = {'sha1': {f: _sha1(os.path.join(STATE, f)) for f in STATE_FILES},
                         'final_epoch': int(res['marker']['epoch']), 'torch': STACK['torch'], 'cuda': STACK['cuda']}
    log('HANDOFF for session %d:' % (SESSION + 1), RECORD['handoff'])
else:
    # crash, backstop, exit without a marker: INCOMPLETE (84.1). The state dir holds the last completed boundary.
    RECORD['stopped_by'] = 'incomplete_' + res['outcome']
    if os.path.exists(os.path.join(STATE, 'full_state.pt')):
        RECORD['handoff_candidate'] = {'sha1': {f: _sha1(os.path.join(STATE, f)) for f in STATE_FILES
                                                if os.path.exists(os.path.join(STATE, f))}}
    os.system('tail -60 %s' % os.path.join(W, 'train_session%d.log' % SESSION))

# The staged copy is several GB and rebuildable; everything needed is in the state dir, the logs and (final
# session) the copied best checkpoint and the profile.
shutil.rmtree(X, ignore_errors=True)
for junk in (ARR,):                       # 2.24 GB derived file; rebuildable, not an output
    try:
        os.remove(junk)
    except OSError:
        pass
RECORD['total_hours'] = round((time.time() - T0) / 3600, 3)
json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2, default=str)
log('DONE', json.dumps({k: RECORD.get(k) for k in ('stopped_by', 'total_hours')}))
if RECORD['stopped_by'].startswith('incomplete'):
    raise SystemExit(1)
