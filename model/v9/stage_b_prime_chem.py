# -*- coding: utf-8 -*-
"""RESULTS §96.4, §96.8, §96.9: Chemical featurisation for Stage B' cold-drug benchmark.

Featurises training and test compounds of split_cold_drug_1 with RDKit:
- LargestFragmentChooser parent standardization
- ECFP4 (Morgan radius 2, 2048 bits)
- 6 physicochemical descriptors: MolLogP, CalcTPSA, MolWt, NumHDonors, NumHAcceptors, basic amines
- InChIKey first-block molecule keys (fallback PERT:<pert_id>)

Runs ONLY under drug/.venv-drug. Does NOT import decoupler or scipy.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERT_INFO_PATHS = [
    os.path.join(REPO_ROOT, 'Data Info', 'GSE92742_Broad_LINCS_pert_info.txt', 'GSE92742_Broad_LINCS_pert_info.txt'),
    os.path.join(REPO_ROOT, 'Data Info', 'GSE70138_Broad_LINCS_pert_info_2017-03-06.txt', 'GSE70138_Broad_LINCS_pert_info.txt'),
]

BASIC_AMINE_SMARTS = '[NX3;!a;!$(N=*);!$(N#*);!$(N-[C,S,P]=[O,S,N]);!$(N-a);!$(N-[N,O])]'


def sha1_file(path):
    """Compute sha1 hash of file contents."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_pert_info(paths=None):
    """Load and merge LINCS pert_info tables, deduplicating on pert_id."""
    paths = paths or PERT_INFO_PATHS
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p, sep='\t', low_memory=False))
    if not dfs:
        return pd.DataFrame().set_index(pd.Index([], name='pert_id'))
    combined = pd.concat(dfs, ignore_index=True)
    return combined.drop_duplicates('pert_id').set_index('pert_id')


def featurize_compound(pert_id, smiles, inchi_key, basic_amine_query=None):
    """Featurize a single compound given its SMILES and InChIKey.

    Returns: (molecule_key, parsable, n_fragments, fp_uint8, desc_float64)
    """
    if basic_amine_query is None:
        basic_amine_query = Chem.MolFromSmarts(BASIC_AMINE_SMARTS)

    # Molecule key from InChIKey first block, or PERT:<pert_id>
    ik_str = '' if (pd.isna(inchi_key) or inchi_key is None) else str(inchi_key).strip()
    if ik_str and ik_str not in ('', '-666', 'nan', 'NaN'):
        block1 = ik_str.split('-')[0].strip()
        molecule_key = block1 if block1 and block1 not in ('', '-666', 'nan', 'NaN') else f'PERT:{pert_id}'
    else:
        molecule_key = f'PERT:{pert_id}'

    # Parse SMILES
    s_str = '' if (pd.isna(smiles) or smiles is None) else str(smiles).strip()
    if not s_str or s_str in ('', '-666', 'nan', 'NaN'):
        return molecule_key, False, 0, np.zeros(2048, dtype=np.uint8), np.full(6, np.nan, dtype=np.float64)

    try:
        mol = Chem.MolFromSmiles(s_str)
    except Exception:
        mol = None

    if mol is None:
        return molecule_key, False, 0, np.zeros(2048, dtype=np.uint8), np.full(6, np.nan, dtype=np.float64)

    n_fragments = len(Chem.GetMolFrags(mol))
    chooser = rdMolStandardize.LargestFragmentChooser()
    mol = chooser.choose(mol)

    if mol is None:
        return molecule_key, False, n_fragments, np.zeros(2048, dtype=np.uint8), np.full(6, np.nan, dtype=np.float64)

    # ECFP4 (Morgan radius 2, 2048 bits)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, fp)

    # Descriptors: MolLogP, CalcTPSA, MolWt, NumHDonors, NumHAcceptors, basic amines
    n_basic = len(mol.GetSubstructMatches(basic_amine_query))
    desc = np.array([
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Descriptors.MolWt(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        float(n_basic),
    ], dtype=np.float64)

    return molecule_key, True, n_fragments, fp, desc


def build_chem_npz(bundle_path, out_npz, pert_info_paths=None, split='split_cold_drug_1'):
    """Extract and featurize all cold-drug train/test compounds from bundle, write npz, sidecar and marker."""
    bundle = np.load(bundle_path, allow_pickle=True)
    split_key = f'split_{split}' if f'split_{split}' in bundle else split
    labels = bundle[split_key]
    pert_ids = np.asarray(bundle['meta_pert_id']).astype(str)

    train_perts = sorted(set(pert_ids[labels == 'train']))
    test_perts = sorted(set(pert_ids[labels == 'test']))

    info = load_pert_info(pert_info_paths)
    q = Chem.MolFromSmarts(BASIC_AMINE_SMARTS)

    rows = []
    # Train compounds first, then test compounds
    for role, plist in [('train', train_perts), ('test', test_perts)]:
        for pid in plist:
            s = info['canonical_smiles'].get(pid) if pid in info.index else None
            ik = info['inchi_key'].get(pid) if pid in info.index else None
            mkey, parsable, n_frag, fp, desc = featurize_compound(pid, s, ik, basic_amine_query=q)
            rows.append({
                'pert_id': pid,
                'role': role,
                'molecule_key': mkey,
                'parsable': parsable,
                'n_fragments': n_frag,
                'fp': fp,
                'desc': desc,
            })

    n = len(rows)
    arr_pert = np.array([r['pert_id'] for r in rows], dtype=object)
    arr_role = np.array([r['role'] for r in rows], dtype=object)
    arr_mkey = np.array([r['molecule_key'] for r in rows], dtype=object)
    arr_pars = np.array([r['parsable'] for r in rows], dtype=bool)
    arr_nfrag = np.array([r['n_fragments'] for r in rows], dtype=np.int64)
    arr_fp = np.stack([r['fp'] for r in rows], axis=0) if n > 0 else np.zeros((0, 2048), dtype=np.uint8)
    arr_desc = np.stack([r['desc'] for r in rows], axis=0) if n > 0 else np.zeros((0, 6), dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(
        out_npz,
        pert_id=arr_pert,
        role=arr_role,
        molecule_key=arr_mkey,
        parsable=arr_pars,
        n_fragments=arr_nfrag,
        fp=arr_fp,
        desc=arr_desc,
    )

    # Sidecar counts
    compounds_per_role = {
        'train': int(np.sum(arr_role == 'train')),
        'test': int(np.sum(arr_role == 'test')),
    }
    parsable_per_role = {
        'train': int(np.sum((arr_role == 'train') & arr_pars)),
        'test': int(np.sum((arr_role == 'test') & arr_pars)),
    }
    multi_frag_per_role = {
        'train': int(np.sum((arr_role == 'train') & (arr_nfrag > 1))),
        'test': int(np.sum((arr_role == 'test') & (arr_nfrag > 1))),
    }

    # multi-pert_id molecules among training compounds
    train_mask = (arr_role == 'train')
    train_mkeys = arr_mkey[train_mask]
    unique_keys, counts = np.unique(train_mkeys, return_counts=True)
    multi_pert_train = int(np.sum(counts > 1))

    sidecar = {
        'compounds_per_role': compounds_per_role,
        'parsable_per_role': parsable_per_role,
        'multi_fragment_per_role': multi_frag_per_role,
        'multi_pert_id_molecules_train': multi_pert_train,
    }

    sidecar_path = out_npz + '.json'
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2)

    sha = sha1_file(out_npz)
    marker = {
        'complete': True,
        'file': os.path.basename(out_npz),
        'sha1': sha,
    }
    marker_path = out_npz + '.marker'
    with open(marker_path, 'w', encoding='utf-8') as f:
        json.dump(marker, f, indent=2)

    return sidecar


def main():
    parser = argparse.ArgumentParser(description="RESULTS §96.4/96.8/96.9: Chemical featurisation for Stage B'")
    parser.add_argument('--bundle', required=True, help="Path to XPert split bundle (xpert_mdmt_splits.npz)")
    parser.add_argument('--out_npz', '--out', dest='out_npz', required=True, help="Path to output CHEM.npz")
    parser.add_argument('--split', default='split_cold_drug_1', help="Split name (default: split_cold_drug_1)")
    args = parser.parse_args()

    sidecar = build_chem_npz(args.bundle, args.out_npz, split=args.split)
    print(f"Wrote {args.out_npz} (sidecar: {json.dumps(sidecar)})")


if __name__ == '__main__':
    main()
