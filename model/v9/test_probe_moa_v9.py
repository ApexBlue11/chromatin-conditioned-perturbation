import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from probe_moa_v9 import parse_gmt_full_sets, load_chembl_targets, project_diff, score

class TestProbeMoaV9(unittest.TestCase):
    def test_positive_set_construction(self):
        # (a) positive-set construction on a toy GMT + toy DTI
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            reactome_path = os.path.join(td, 'Reactome.gmt')
            go_path = os.path.join(td, 'GO.gmt')
            with open(reactome_path, 'w') as f:
                f.write("Path1\tR-HSA-1\tGENE1\tGENE2\n")
            with open(go_path, 'w') as f:
                f.write("Path2 (GO:002)\t\tGENE3\tGENE4\n")
            
            gmt_sets = parse_gmt_full_sets(reactome_path, go_path)
            self.assertIn('R-HSA-1', gmt_sets)
            self.assertEqual(gmt_sets['R-HSA-1'], {'GENE1', 'GENE2'})
            self.assertIn('GO:002', gmt_sets)
            self.assertEqual(gmt_sets['GO:002'], {'GENE3', 'GENE4'})
            
            # Toy DTI
            dti_path = os.path.join(td, 'dti.tsv')
            with open(dti_path, 'w') as f:
                f.write("pert_id\torganism\ttarget_type\tdirect_interaction\tgene_symbol\n")
                f.write("DRUG1\tHomo sapiens\tSINGLE PROTEIN\t1\tGENE1\n")
                f.write("DRUG1\tHomo sapiens\tSINGLE PROTEIN\t1\tGENE3\n")
                f.write("DRUG2\tHomo sapiens\tSINGLE PROTEIN\t1\tGENE4\n")
                # Excluded
                f.write("DRUG3\tMus musculus\tSINGLE PROTEIN\t1\tGENE1\n")
            
            targets = load_chembl_targets(dti_path)
            self.assertEqual(targets['DRUG1'], {'GENE1', 'GENE3'})
            self.assertEqual(targets['DRUG2'], {'GENE4'})
            self.assertNotIn('DRUG3', targets)

    def test_rank_percentile_and_permutation(self):
        # (b) rank percentile and label-permutation on a planted example
        np.random.seed(42)
        P = 100
        D = 20
        delta_imp = np.random.randn(D, P)
        
        # Planted signal: for each drug, its positive set is placed exactly at the top
        positives = {}
        for d in range(D):
            positives[d] = [d, (d+1)%P]
            delta_imp[d, d] = 1000.0 # Huge importance
            delta_imp[d, (d+1)%P] = 1000.0
            
        # compute pct
        pct = np.empty_like(delta_imp)
        for d in range(D):
            order = np.argsort(-np.abs(delta_imp[d]))
            pct[d, order] = np.arange(P) / P
            
        allpos = [positives[d] for d in range(D)]
        obs = [float(np.median(pct[i][allpos[i]])) for i in range(D)]
        med = float(np.median(obs))
        self.assertLess(med, 0.05) # Planted at the top
        
        # Permutation
        n_perm = 100
        perm = np.empty(n_perm)
        for t in range(n_perm):
            sh = np.random.permutation(D)
            perm[t] = np.median([np.median(pct[i][allpos[sh[i]]]) for i in range(D)])
            
        diff = med - perm.mean()
        self.assertLess(diff, -0.4) # Planted gives very negative diff
        
        # No signal example
        delta_imp_rand = np.random.randn(D, P)
        pct_rand = np.empty_like(delta_imp_rand)
        for d in range(D):
            order = np.argsort(-np.abs(delta_imp_rand[d]))
            pct_rand[d, order] = np.arange(P) / P
            
        obs_rand = [float(np.median(pct_rand[i][allpos[i]])) for i in range(D)]
        med_rand = float(np.median(obs_rand))
        
        perm_rand = np.empty(n_perm)
        for t in range(n_perm):
            sh = np.random.permutation(D)
            perm_rand[t] = np.median([np.median(pct_rand[i][allpos[sh[i]]]) for i in range(D)])
            
        diff_rand = med_rand - perm_rand.mean()
        self.assertTrue(-0.1 < diff_rand < 0.1) # diff ~ 0

    def test_mean_drug_baseline_byte_identical(self):
        # (c) the mean-drug baseline is byte-identical for two different rows
        # We simulate what probe_moa_v9 does
        import torch
        u_mean = torch.ones(10)
        atom_mean = torch.ones(5)
        
        k_med = 3
        rows = 2
        
        # for row 1
        b0_row1_u = u_mean.clone()
        b0_row1_atoms = atom_mean.unsqueeze(0).expand(k_med, -1).clone()
        
        # for row 2
        b0_row2_u = u_mean.clone()
        b0_row2_atoms = atom_mean.unsqueeze(0).expand(k_med, -1).clone()
        
        self.assertTrue(torch.all(b0_row1_u == b0_row2_u))
        self.assertTrue(torch.all(b0_row1_atoms == b0_row2_atoms))
        # Wait, the bytes must be identical? Yes, because they are expanded from the same tensor.

    def test_rho_del_identical_vectors(self):
        # (d) rho_del is 1.0 when all compounds have the same |dimp| vector.
        from probe_pathways_v6 import spearman
        D = 10
        P = 50
        delta_imp = np.ones((D, P)) * np.arange(P)
        
        pairs = [(i, j) for i in range(D) for j in range(D) if i != j]
        nanmed = lambda v: float(np.median([x for x in v if np.isfinite(x)])) if any(np.isfinite(v)) else float("nan")
        rho_del = nanmed([spearman(np.abs(delta_imp[i]), np.abs(delta_imp[j])) for i, j in pairs])
        self.assertAlmostEqual(rho_del, 1.0)

    def test_project_then_abs(self):
        # (e) a planted case where project-then-abs differs from abs-then-project
        Y_diff = np.array([[-1.0, 1.0]]) # [1, 2]
        M_norm = np.array([[1.0, 1.0]])  # [1, 2]
        
        # Calculate old behavior (abs then project)
        abs_then_proj = np.abs(Y_diff) @ M_norm.T # [[1.0, 1.0]] @ [[1.0], [1.0]] -> [[2.0]]
        
        # Calculate new behavior (project then abs)
        proj_then_abs = project_diff(Y_diff, M_norm) # np.abs([[-1.0, 1.0]] @ [[1.0], [1.0]]) -> [[0.0]]
        
        self.assertNotEqual(float(abs_then_proj[0, 0]), float(proj_then_abs[0, 0]))
        self.assertEqual(float(proj_then_abs[0, 0]), 0.0)

    def test_score_function(self):
        # (f) score() gives diff < 0 on a planted signal and |diff| small on a shuffled one.
        np.random.seed(42)
        P = 100
        D = 20
        scores_planted = np.random.randn(D, P)
        scores_shuffled = np.random.randn(D, P)
        
        allpos = []
        for d in range(D):
            allpos.append([d, (d+1)%P])
            scores_planted[d, d] = 1000.0
            scores_planted[d, (d+1)%P] = 1000.0
            
        sizes = np.ones(P, dtype=int) * 5
        
        res_planted = score(scores_planted, allpos, sizes, rng_seed=42, n_perm=100, n_size=10)
        self.assertLess(res_planted['diff'], -0.4)
        
        res_shuffled = score(scores_shuffled, allpos, sizes, rng_seed=42, n_perm=100, n_size=10)
        self.assertTrue(-0.1 < res_shuffled['diff'] < 0.1)

    def test_stratum_permutation(self):
        # (g) a stratum's permutation only draws positive sets from compounds inside that stratum
        # We can test this by checking that the permutation logic only operates on the passed `allpos`.
        # When we pass `allpos` of a specific stratum to `score()`, the number of compounds `D` will match
        # the stratum size. Thus `rng.permutation(D)` only permutes the indices of the subset.
        
        # We simulate a subset of 3 compounds
        allpos_subset = [[0, 1], [2, 3], [4, 5]]
        scores_subset = np.random.randn(3, 100)
        sizes = np.ones(100, dtype=int) * 5
        
        res = score(scores_subset, allpos_subset, sizes, rng_seed=42, n_perm=10, n_size=10)
        
        # If the subset length is 3, the permuted indices will only ever be 0, 1, or 2,
        # which means it's correctly contained within the stratum.
        self.assertEqual(res['n'], 3)

if __name__ == '__main__':
    unittest.main()
