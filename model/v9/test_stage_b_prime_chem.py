# -*- coding: utf-8 -*-
"""Tests for RESULTS §96.4/96.8/96.9 Chemical Featurisation (stage_b_prime_chem.py).

Runs under drug/.venv-drug.
Tests:
- SMARTS counts on reference molecules
- Largest fragment reduction on salts
- Fingerprint parity and Tanimoto bit calculation parity
- Edge cases: unparsable SMILES, missing InChIKeys
- Writer CLI on synthetic bundle and pert_info TSVs
"""
import json
import os
import sys

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stage_b_prime_chem as sbpc  # noqa: E402


def test_smarts_counts():
    """Verify basic-amine SMARTS matches: imatinib 2, afatinib 1, triethylamine 1; erlotinib, nitrobenzene, acetamide, aniline, guanidine 0."""
    q = Chem.MolFromSmarts(sbpc.BASIC_AMINE_SMARTS)

    cases = {
        'imatinib': ('CN1CCN(Cc2ccc(cc2)C(=O)Nc2ccc(C)c(Nc3nccc(n3)-c3cccnc3)c2)CC1', 2),
        'afatinib': (r'CN(C)C\C=C\C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1', 1),
        'triethylamine': ('CCN(CC)CC', 1),
        'erlotinib': ('COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC', 0),
        'nitrobenzene': ('c1ccccc1[N+](=O)[O-]', 0),
        'acetamide': ('CC(=O)N', 0),
        'aniline': ('c1ccccc1N', 0),
        'guanidine': ('NC(=N)N', 0),
    }

    for name, (smi, expected) in cases.items():
        m = Chem.MolFromSmiles(smi)
        assert m is not None, f"Failed to parse {name}"
        count = len(m.GetSubstructMatches(q))
        assert count == expected, f"{name}: expected {expected}, got {count}"


def test_largest_fragment():
    """CCN(CC)CC.Cl gives triethylamine's descriptors exactly and n_fragments == 2."""
    salt_smi = 'CCN(CC)CC.Cl'
    free_smi = 'CCN(CC)CC'

    _, parsable_salt, n_frag_salt, fp_salt, desc_salt = sbpc.featurize_compound('SALT', salt_smi, 'IK_SALT')
    _, parsable_free, n_frag_free, fp_free, desc_free = sbpc.featurize_compound('FREE', free_smi, 'IK_FREE')

    assert parsable_salt is True
    assert n_frag_salt == 2
    assert parsable_free is True
    assert n_frag_free == 1

    np.testing.assert_array_equal(fp_salt, fp_free)
    np.testing.assert_allclose(desc_salt, desc_free, rtol=1e-12, atol=1e-12)


def test_fingerprint_parity_and_tanimoto():
    """FP bits equal ConvertToNumpyArray(GetMorganFingerprintAsBitVect(...)) and bit Tanimoto equals DataStructs.TanimotoSimilarity for 3 pairs."""
    smis = [
        'CN1CCN(Cc2ccc(cc2)C(=O)Nc2ccc(C)c(Nc3nccc(n3)-c3cccnc3)c2)CC1',  # imatinib
        r'CN(C)C\C=C\C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1',     # afatinib
        'COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC',                         # erlotinib
    ]

    mols = [Chem.MolFromSmiles(s) for s in smis]
    rd_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    arrs = []
    for i, s in enumerate(smis):
        _, _, _, fp, _ = sbpc.featurize_compound(f'P{i}', s, f'IK{i}')
        ref_arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(rd_fps[i], ref_arr)
        np.testing.assert_array_equal(fp, ref_arr)
        arrs.append(fp)

    # 3 pairs: (0, 1), (0, 2), (1, 2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        rd_tanimoto = DataStructs.TanimotoSimilarity(rd_fps[i], rd_fps[j])
        a = arrs[i].astype(np.float64)
        b = arrs[j].astype(np.float64)
        inter = np.dot(a, b)
        denom = np.sum(a) + np.sum(b) - inter
        calc_tanimoto = inter / denom if denom > 0 else 0.0
        np.testing.assert_allclose(calc_tanimoto, rd_tanimoto, rtol=1e-12, atol=1e-12)


def test_edge_cases():
    """Unparsable SMILES gives parsable == False, zero fp and NaN desc. Missing InChIKey gives PERT: key."""
    # Unparsable SMILES
    for bad_smi in ['', '-666', 'nan', 'INVALID_SMILES', None]:
        mkey, parsable, n_frag, fp, desc = sbpc.featurize_compound('BAD_PERT', bad_smi, 'VALIDIK-123')
        assert parsable is False
        assert n_frag == 0
        assert np.all(fp == 0)
        assert np.all(np.isnan(desc))
        assert mkey == 'VALIDIK'

    # Missing InChIKey
    for bad_ik in ['', '-666', 'nan', None]:
        mkey, parsable, n_frag, fp, desc = sbpc.featurize_compound('MY_PERT', 'CCN(CC)CC', bad_ik)
        assert parsable is True
        assert mkey == 'PERT:MY_PERT'

    # Valid InChIKey block 1
    mkey, _, _, _, _ = sbpc.featurize_compound('MY_PERT', 'CCN(CC)CC', 'IKBLOCK1-IKBLOCK2-N')
    assert mkey == 'IKBLOCK1'


def test_writer_main(tmp_path, monkeypatch):
    """Test main() on synthetic bundle plus two pert_info TSVs."""
    # 1. Create two tiny pert_info TSVs
    tsv1 = tmp_path / 'pert_info1.tsv'
    tsv1.write_text(
        "pert_id\tcanonical_smiles\tinchi_key\n"
        "TR1\tCCN(CC)CC\tIK1-A-N\n"
        "TR2\tCCN(CC)CC\tIK1-B-N\n"  # shares inchi_key block 1 with TR1 -> multi-pert molecule in train
        "TR3\tCCN(CC)CC.Cl\tIK3-A-N\n"  # multi-fragment in train
        "TR4\t-666\tIK4-A-N\n",  # unparsable in train
        encoding='utf-8',
    )

    tsv2 = tmp_path / 'pert_info2.tsv'
    tsv2.write_text(
        "pert_id\tcanonical_smiles\tinchi_key\n"
        "TE1\tCOCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC\tIK_TE1-A-N\n"
        "TE2\t\t\n",  # missing smiles and inchi_key in test
        encoding='utf-8',
    )

    monkeypatch.setattr(sbpc, 'PERT_INFO_PATHS', [str(tsv1), str(tsv2)])

    # 2. Create synthetic bundle
    bundle_path = tmp_path / 'bundle.npz'
    labels = np.array(['train', 'train', 'train', 'train', 'test', 'test'])
    perts = np.array(['TR1', 'TR2', 'TR3', 'TR4', 'TE1', 'TE2'])
    np.savez(bundle_path, split_split_cold_drug_1=labels, meta_pert_id=perts)

    out_npz = tmp_path / 'CHEM.npz'

    # 3. Call main()
    monkeypatch.setattr(sys, 'argv', [
        'stage_b_prime_chem.py',
        '--bundle', str(bundle_path),
        '--out_npz', str(out_npz),
    ])
    sbpc.main()

    # 4. Verify outputs exist
    assert out_npz.exists()
    sidecar_path = tmp_path / 'CHEM.npz.json'
    marker_path = tmp_path / 'CHEM.npz.marker'
    assert sidecar_path.exists()
    assert marker_path.exists()

    # 5. Verify marker
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    assert marker['complete'] is True
    assert marker['sha1'] == sbpc.sha1_file(str(out_npz))

    # 6. Verify sidecar counts
    sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
    assert sidecar['compounds_per_role'] == {'train': 4, 'test': 2}
    assert sidecar['parsable_per_role'] == {'train': 3, 'test': 1}
    assert sidecar['multi_fragment_per_role'] == {'train': 1, 'test': 0}
    assert sidecar['multi_pert_id_molecules_train'] == 1  # IK1 covers TR1 and TR2

    # 7. Verify npz data arrays
    data = np.load(out_npz, allow_pickle=True)
    assert set(data.files) == {'pert_id', 'role', 'molecule_key', 'parsable', 'n_fragments', 'fp', 'desc'}
    assert len(data['pert_id']) == 6
    assert list(data['role']) == ['train', 'train', 'train', 'train', 'test', 'test']
    assert data['fp'].shape == (6, 2048)
    assert data['desc'].shape == (6, 6)
    assert data['desc'].dtype == np.float64
    assert data['fp'].dtype == np.uint8
