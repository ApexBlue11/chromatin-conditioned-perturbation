"""The RESULTS 88.3 reader on synthetic probe records: every reading reachable, every guard firing."""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from read_moa_88 import read, ROWS


def rec(diff=-0.05, p=0.001, ps=0.001, S=0.30, null2=0.40, un_diff=-0.05, un_p=0.001, rho=0.5, untrained=False,
        epoch=11, q='q1', void=False, out_diff=0.0, out_p=0.5, data_diff=0.0, data_p=0.5, seed=0, sha='s0',
        data_S=0.5519):
    sc = lambda d, pp, s_=S: {'S': s_, 'null1_mean': s_ - d, 'null1_sd': 0.01, 'diff': d, 'p': pp, 'null2_mean': null2,
                              'n': 496}
    prov = {'init_seed': seed, 'cfg_from_sha1': 's0'} if untrained else {'ckpt_sha1': sha}
    return {**prov, 'void': void, 'row_sha1': ROWS, 'quintile_sha1': q, 'n_compounds': 496, 'max_da': 1.0,
            'epoch': None if untrained else epoch, 'untrained': untrained,
            'cfg_from': 'c8b_ckpt_v9_fold0_seed0.pt' if untrained else None, 'gate_pass': rho < 0.95, 'rho_post': rho,
            'primary': {'score': sc(diff, p), 'null1s': {'S': S, 'null1s_mean': S - diff, 'null1s_sd': 0.01,
                                                          'diff_s': diff, 'p_s': ps}},
            'output_projection': {'score': sc(out_diff, out_p)}, 'data_projection': {'score': sc(data_diff, data_p, data_S)},
            'strata': {'unseen': {'score': sc(un_diff, un_p), 'n': 156}, 'seen': {'score': sc(diff, p), 'n': 340},
                       'no_landmark_target': None, 'target_responsive': {'score': sc(diff, p), 'n': 150},
                       'target_not_responsive': {'score': sc(diff, p), 'n': 150}}}


U = [rec(diff=d, un_diff=d, p=0.4, ps=0.4, untrained=True, seed=k)
     for k, d in enumerate((-0.004, -0.006, -0.002, -0.008, -0.005))]
PINS = ['s0', 's1', 's2']


def tr(**kw):
    return [rec(sha=PINS[s], **kw) for s in range(3)]


T = tr()


def r(T_, U_=U):
    T_ = copy.deepcopy(T_)
    for s, t in enumerate(T_):
        t['ckpt_sha1'] = PINS[s]
    return read(T_, copy.deepcopy(U_), pins=PINS)


def test_signal_with_both_qualifiers():
    o = r(T)
    assert o['reading'] == 'SIGNAL', o
    assert o['wording']['sentence'][1:] == ["beyond the data's own target alignment", "beyond the model's predicted signature"]


def test_signal_qualifiers_three_way():
    o = r([rec(data_diff=-0.03, data_p=0.01), rec(data_diff=-0.03, data_p=0.01), rec(data_diff=-0.03, data_p=0.01)])
    assert o['wording']['sentence'][1:] == ["beyond the model's predicted signature"]       # data aligned -> no data qualifier
    o = r([rec(), rec(out_diff=-0.03, out_p=0.01), rec()])
    assert 'not determined (output projection aligned on 1 of 3 seeds)' in o['wording']['sentence'][-1]
    o = r([rec(out_diff=-0.03, out_p=0.01) for _ in range(3)])
    assert o['wording']['sentence'][-1] == 'a named readout of a predicted signature that is itself target-aligned'


def test_seen_only():
    assert r([rec(), rec(un_diff=-0.01), rec()])['reading'] == 'SEEN-ONLY'      # unseen above the -0.02 floor


def test_null_when_null1s_fails_on_one_seed():
    assert r([rec(), rec(ps=0.2), rec()])['reading'] == 'NULL'


def test_null_when_not_below_untrained_minimum():
    U2 = copy.deepcopy(U)
    U2[3] = rec(diff=-0.06, un_diff=-0.06, p=0.4, ps=0.4, untrained=True, seed=3)  # one init reaches below the trained
    assert r(T, U2)['reading'] == 'NULL'


def test_null_when_not_below_mean_minus_2sd():
    U2 = [rec(diff=d, un_diff=-0.004, p=0.4, ps=0.4, untrained=True, seed=k)
          for k, d in enumerate((-0.049, 0.02, 0.03, -0.01, 0.0))]
    o = r([rec(diff=-0.05) for _ in range(3)], U2)   # -0.05 <= min(-0.049) but m_u - 2 sd_u ~ -0.052
    assert o['untrained']['m_u'] - 2 * o['untrained']['sd_u'] < -0.05
    assert o['reading'] == 'NULL'


def test_null_when_S_not_below_null2():
    assert r([rec(), rec(S=0.45), rec()])['reading'] == 'NULL'


def test_gate_failure_is_null_even_with_signal_numbers():
    assert r([rec(), rec(rho=0.97), rec()])['reading'] == 'NULL'


def test_void_and_invalid():
    assert r([rec(), rec(void=True), rec()])['reading'] == 'VOID'
    assert r([rec(), rec(q='q2'), rec()])['reading'] == 'INVALID'              # quintile assignment differs
    assert r([rec(), rec(epoch=8), rec()])['reading'] == 'INVALID'             # budget-cut trained run
    U2 = copy.deepcopy(U); U2[0]['cfg_from'] = 'v9dev_c8b_dev6s0_seed0.pt'
    assert r(T, U2)['reading'] == 'INVALID'                                  # calibrated on the wrong architecture
    T2 = copy.deepcopy(T); T2[1]['row_sha1'] = 'x'
    assert r(T2)['reading'] == 'INVALID'


def test_provenance_guards():
    T2 = copy.deepcopy(T)
    assert read(copy.deepcopy(T2), copy.deepcopy(U), pins=PINS)['reading'] == 'SIGNAL'
    T2[2]['ckpt_sha1'] = 's1'                                          # a copied trained record
    assert read(T2, copy.deepcopy(U), pins=PINS)['reading'] == 'INVALID'
    U2 = copy.deepcopy(U); U2[4] = copy.deepcopy(U2[3])                 # a duplicated untrained init
    assert r(T, U2)['reading'] == 'INVALID'
    U3 = copy.deepcopy(U); U3[0]['cfg_from_sha1'] = 'other'
    assert r(T, U3)['reading'] == 'INVALID'
    T3 = copy.deepcopy(T); T3[0]['strata']['unseen']['n'] = 150            # not the 86 rows' unseen stratum
    assert r(T3)['reading'] == 'INVALID'
    T4 = copy.deepcopy(T); T4[0]['data_projection']['score']['S'] = 0.6   # the model-free readout must be identical
    assert r(T4)['reading'] == 'INVALID'
    import pytest
    with pytest.raises(SystemExit):
        read(copy.deepcopy(T), copy.deepcopy(U))                        # unpinned


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-q']))
