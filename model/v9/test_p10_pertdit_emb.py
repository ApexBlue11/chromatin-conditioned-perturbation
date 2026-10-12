# -*- coding: utf-8 -*-
"""Tests for P10 arm D PertDiT drug and dose embeddings precompute and kernels (RESULTS §98.15, §98.16)."""
import ast
import difflib
import glob
import hashlib
import json
import os
import pickle
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
sys.path.insert(0, os.path.join(REPO_ROOT, "orchestration"))

import make_p10_d_emb_kernels  # noqa: E402
import p10_pertdit_emb  # noqa: E402

VENDOR_SRC = os.path.join(
    REPO_ROOT,
    "external",
    "bai_benchmark_1d750cd",
    "experiments",
    "main_benchmark",
    "data_prep",
    "prepare_pertdit_embeddings.py",
)


@pytest.fixture
def synthetic_data(tmp_path):
    """Synthetic bundle (meta_pert_idx and meta_dose only) and synthetic idx2smi table."""
    sparse_indices = [7, 15, 42, 108, 300, 512, 1024, 2048, 4096, 7000, 8000, 8949]  # 12 compounds
    n_rows = 40
    row_pert_idx = np.array([sparse_indices[i % len(sparse_indices)] for i in range(n_rows)], dtype=np.int64)

    # 3 doses including 10.0, float32 rounding check
    doses_float32 = [np.float32(0.37), np.float32(2.31), np.float32(10.0)]
    row_dose = np.array([doses_float32[i % len(doses_float32)] for i in range(n_rows)], dtype=np.float32)

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, meta_pert_idx=row_pert_idx, meta_dose=row_dose)

    # Synthetic idx2smi: 12 compounds, unique SMILES
    idx2smi_table = {idx: f"C{i+1}C(=O)O{idx}" for i, idx in enumerate(sparse_indices)}
    idx2smi_path = str(tmp_path / "all_drugs_idx2smi_8981.npy")
    np.save(idx2smi_path, idx2smi_table, allow_pickle=True)

    expected_doses = [0.37, 2.31, 10.0]

    return {
        "bundle_path": bundle_path,
        "idx2smi_path": idx2smi_path,
        "sparse_indices": sparse_indices,
        "idx2smi_table": idx2smi_table,
        "expected_doses": expected_doses,
    }


# ==============================================================================
# 1. Synthetic bundle npz & compound_map / dose_values
# ==============================================================================

def test_compound_map_and_dose_values(synthetic_data, monkeypatch):
    """Check compound_map gives ranks 0..n-1 and dose_values rounds float32 doses exactly."""
    b_path = synthetic_data["bundle_path"]
    cmap = p10_pertdit_emb.compound_map(b_path)

    assert len(cmap) == 12
    sorted_indices = sorted(synthetic_data["sparse_indices"])
    for rank, idx in enumerate(sorted_indices):
        assert cmap[idx] == rank

    # Monkeypatch EXPECT_DOSES to test dose_values on 3 doses
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])
    doses = p10_pertdit_emb.dose_values(b_path)
    assert doses == [0.37, 2.31, 10.0]
    # Check that 0.37 is float 0.37, not float32 0.3700000047683716
    assert doses[0] == 0.37


# ==============================================================================
# 2. Isolated guards: (g1)-(g3), dose_values, (m1)-(m3), (o1)-(o4)
# ==============================================================================

def test_isolated_guard_dose_values(synthetic_data):
    """dose_values list guard fails when EXPECT_DOSES is not matched, naming the guard."""
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.dose_values(synthetic_data["bundle_path"])
    assert "FATAL: dose_values guard" in str(exc.value)


def test_isolated_guard_g1(synthetic_data, tmp_path):
    """(g1) idx2smi sha1 mismatch fails naming guard (g1)."""
    out_dir = str(tmp_path / "out_g1")
    # idx2smi_path sha1 does not equal IDX2SMI_FULL_SHA1
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], synthetic_data["idx2smi_path"], out_dir)
    assert "FATAL: guard (g1)" in str(exc.value)


def test_isolated_guard_g2(synthetic_data, monkeypatch, tmp_path):
    """(g2) bundle compound missing from idx2smi table fails naming guard (g2)."""
    out_dir = str(tmp_path / "out_g2")
    # Table missing key 8949
    broken_table = dict(synthetic_data["idx2smi_table"])
    del broken_table[8949]
    broken_path = str(tmp_path / "broken_idx2smi.npy")
    np.save(broken_path, broken_table, allow_pickle=True)

    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(broken_path))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 12)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], broken_path, out_dir)
    assert "FATAL: guard (g2)" in str(exc.value)


def test_isolated_guard_g3_duplicate_smiles(synthetic_data, monkeypatch, tmp_path):
    """(g3) two compounds sharing a SMILES fails naming guard (g3)."""
    out_dir = str(tmp_path / "out_g3_dup")
    dup_table = dict(synthetic_data["idx2smi_table"])
    # Two compounds share SMILES
    dup_table[8949] = dup_table[7]
    dup_path = str(tmp_path / "dup_idx2smi.npy")
    np.save(dup_path, dup_table, allow_pickle=True)

    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(dup_path))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 12)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], dup_path, out_dir)
    assert "FATAL: guard (g3)" in str(exc.value)


def test_isolated_guard_g3_compound_count(synthetic_data, monkeypatch, tmp_path):
    """(g3) compound count mismatch with EXPECT_N_COMPOUNDS fails naming guard (g3)."""
    out_dir = str(tmp_path / "out_g3_count")
    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(synthetic_data["idx2smi_path"]))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 1970)  # synthetic has 12
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], synthetic_data["idx2smi_path"], out_dir)
    assert "FATAL: guard (g3)" in str(exc.value)


def test_isolated_guard_m1(tmp_path):
    """(m1) two caches giving the same SMILES different captions fails naming guard (m1)."""
    c1 = {"CCO": "caption A", "CCN": "caption B"}
    c2 = {"CCO": "caption A_diff", "CCC": "caption C"}
    p1 = str(tmp_path / "c1.pkl")
    p2 = str(tmp_path / "c2.pkl")
    with open(p1, "wb") as f:
        pickle.dump(c1, f)
    with open(p2, "wb") as f:
        pickle.dump(c2, f)
    idx2smi = {0: "CCO", 1: "CCN", 2: "CCC"}

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.merge_caches([p1, p2], idx2smi)
    assert "FATAL: guard (m1)" in str(exc.value)


def test_isolated_guard_m2(tmp_path):
    """(m2) union key set not matching idx2smi_all fails naming guard (m2)."""
    c1 = {"CCO": "caption A"}
    c2 = {"CCN": "caption B"}
    p1 = str(tmp_path / "c1.pkl")
    p2 = str(tmp_path / "c2.pkl")
    with open(p1, "wb") as f:
        pickle.dump(c1, f)
    with open(p2, "wb") as f:
        pickle.dump(c2, f)
    idx2smi = {0: "CCO", 1: "CCN", 2: "CCC"}  # CCC missing

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.merge_caches([p1, p2], idx2smi)
    assert "FATAL: guard (m2)" in str(exc.value)


def test_isolated_guard_m3(tmp_path):
    """(m3) empty or non-str caption fails naming guard (m3)."""
    c1 = {"CCO": ""}
    p1 = str(tmp_path / "c1.pkl")
    with open(p1, "wb") as f:
        pickle.dump(c1, f)
    idx2smi = {0: "CCO"}

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.merge_caches([p1], idx2smi)
    assert "FATAL: guard (m3)" in str(exc.value)


def test_isolated_guard_o1(tmp_path):
    """(o1) bench_drug_emb.pkl keys mismatch fails naming guard (o1)."""
    out_dir = tmp_path / "out_o1"
    out_dir.mkdir()
    # Missing negative_ctrl
    drug_emb = {0: torch.randn(5, 1024)}
    torch.save(drug_emb, str(out_dir / "bench_drug_emb.pkl"))
    torch.save({10.0: torch.randn(1, 1024)}, str(out_dir / "bench_dose_emb.pt"))
    idx2smi = {0: "CCO"}
    dose_list = [10.0]

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.check_outputs(str(out_dir), idx2smi, dose_list)
    assert "FATAL: guard (o1)" in str(exc.value)


def test_isolated_guard_o2(tmp_path):
    """(o2) non-finite or wrong-shape drug tensor fails naming guard (o2)."""
    out_dir = tmp_path / "out_o2"
    out_dir.mkdir()
    # Drug tensor not 2-D or contains non-finite
    drug_emb = {
        0: torch.tensor([float("inf")] * 1024, dtype=torch.float32).unsqueeze(0),
        "negative_ctrl": torch.full((2, 1024), float("nan")),
    }
    torch.save(drug_emb, str(out_dir / "bench_drug_emb.pkl"))
    torch.save({10.0: torch.randn(1, 1024)}, str(out_dir / "bench_dose_emb.pt"))
    idx2smi = {0: "CCO"}
    dose_list = [10.0]

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.check_outputs(str(out_dir), idx2smi, dose_list)
    assert "FATAL: guard (o2)" in str(exc.value)


def test_isolated_guard_o3(tmp_path):
    """(o3) zero not NaN negative_ctrl fails naming guard (o3)."""
    out_dir = tmp_path / "out_o3"
    out_dir.mkdir()
    drug_emb = {
        0: torch.randn(5, 1024),
        "negative_ctrl": torch.zeros((2, 1024)),  # zero instead of NaN
    }
    torch.save(drug_emb, str(out_dir / "bench_drug_emb.pkl"))
    torch.save({10.0: torch.randn(1, 1024)}, str(out_dir / "bench_dose_emb.pt"))
    idx2smi = {0: "CCO"}
    dose_list = [10.0]

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.check_outputs(str(out_dir), idx2smi, dose_list)
    assert "FATAL: guard (o3)" in str(exc.value)


def test_isolated_guard_o4_float32_key(tmp_path):
    """(o4) float32-cast 0.37 key fails naming guard (o4)."""
    out_dir = tmp_path / "out_o4"
    out_dir.mkdir()
    drug_emb = {
        0: torch.randn(5, 1024),
        "negative_ctrl": torch.full((2, 1024), float("nan")),
    }
    torch.save(drug_emb, str(out_dir / "bench_drug_emb.pkl"))
    # float(np.float32(0.37)) is 0.3700000047683716 != 0.37
    dose_emb = {float(np.float32(0.37)): torch.randn(1, 1024)}
    torch.save(dose_emb, str(out_dir / "bench_dose_emb.pt"))
    idx2smi = {0: "CCO"}
    dose_list = [0.37]

    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.check_outputs(str(out_dir), idx2smi, dose_list)
    assert "FATAL: guard (o4)" in str(exc.value)


# ==============================================================================
# 3. Slices
# ==============================================================================

def test_slices_round_robin_and_determinism(synthetic_data, monkeypatch, tmp_path):
    """Slices are disjoint, union equals all, keys are 0..n_k-1, round-robin by descending length."""
    # Test 7-SMILES exact assignment
    smiles_7 = [
        "CCCCCCCCCC",  # len 10
        "CCCCCCCCC",   # len 9
        "CCCCCCCC",    # len 8
        "CCCCCCC",     # len 7
        "CCCCCC",      # len 6
        "CCCCC",       # len 5
        "CCCC",        # len 4
    ]
    # Slices deal round-robin with N_SLICES = 5
    # S0: "CCCCCCCCCC" (10), "CCCCC" (5)
    # S1: "CCCCCCCCC" (9), "CCCC" (4)
    # S2: "CCCCCCCC" (8)
    # S3: "CCCCCCC" (7)
    # S4: "CCCCCC" (6)
    sparse_7 = list(range(7))
    b_path_7 = str(tmp_path / "bundle7.npz")
    np.savez(b_path_7, meta_pert_idx=sparse_7, meta_dose=[10.0] * 7)
    table_7 = {i: s for i, s in enumerate(smiles_7)}
    t_path_7 = str(tmp_path / "table7.npy")
    np.save(t_path_7, table_7, allow_pickle=True)

    out_1 = str(tmp_path / "out_slices_1")
    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(t_path_7))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 7)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", [10.0])

    p10_pertdit_emb.build_inputs(b_path_7, t_path_7, out_1)

    s0 = np.load(os.path.join(out_1, "idx2smi_slice0.npy"), allow_pickle=True).item()
    s1 = np.load(os.path.join(out_1, "idx2smi_slice1.npy"), allow_pickle=True).item()
    s2 = np.load(os.path.join(out_1, "idx2smi_slice2.npy"), allow_pickle=True).item()
    s3 = np.load(os.path.join(out_1, "idx2smi_slice3.npy"), allow_pickle=True).item()
    s4 = np.load(os.path.join(out_1, "idx2smi_slice4.npy"), allow_pickle=True).item()

    assert set(s0.values()) == {"CCCCCCCCCC", "CCCCC"}
    assert set(s1.values()) == {"CCCCCCCCC", "CCCC"}
    assert set(s2.values()) == {"CCCCCCCC"}
    assert set(s3.values()) == {"CCCCCCC"}
    assert set(s4.values()) == {"CCCCCC"}

    # Keys are 0..n_k-1 in sorted order
    for s in [s0, s1, s2, s3, s4]:
        assert list(s.keys()) == list(range(len(s)))
        assert list(s.values()) == sorted(s.values())

    # Determinism: second run produces identical hashes
    out_2 = str(tmp_path / "out_slices_2")
    p10_pertdit_emb.build_inputs(b_path_7, t_path_7, out_2)
    for k in range(5):
        h1 = p10_pertdit_emb.sha1_file(os.path.join(out_1, f"idx2smi_slice{k}.npy"))
        h2 = p10_pertdit_emb.sha1_file(os.path.join(out_2, f"idx2smi_slice{k}.npy"))
        assert h1 == h2


# ==============================================================================
# 4. make_emb_script
# ==============================================================================

def test_make_emb_script(tmp_path):
    """make_emb_script exact substitution diff and ast parse."""
    dst = str(tmp_path / "prepare_pertdit_embeddings_p10.py")
    out_code = p10_pertdit_emb.make_emb_script(VENDOR_SRC, dst)

    with open(VENDOR_SRC, "r", encoding="utf-8") as f:
        src = f.read()

    diff = list(difflib.unified_diff(src.splitlines(keepends=True), out_code.splitlines(keepends=True)))
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    added = [l for l in diff if l.startswith("+") and not l.startswith("+++")]

    # Exact line counts from briefed changes
    assert len(removed) == 13
    assert len(added) == 14

    ast.parse(out_code)


def test_pi_make_emb_script_each_target_and_source_pin(tmp_path, monkeypatch):
    """PI: each of the three substitution targets, missing alone, fails; a source whose sha1 is not SRC_SHA1 fails; a string
    (not a file) fails."""
    with open(VENDOR_SRC, "r", encoding="utf-8") as f:
        src = f.read()
    targets = {
        "OUT_DIR": ('OUT_DIR = os.path.join(XPERT_ROOT, "data", "PertDiT", "bench_drug_emb")',
                    'OUT_DIR = os.path.join(XPERT_ROOT, "data", "PertDiT", "bench_drug_emb2")'),
        "CUDA gate": ("        if not torch.cuda.is_available():\n            sys.exit(",
                      "        if torch.cuda.is_available() is None:\n            sys.exit("),
        "dose block": ('    dose_prompt = "The dosage is 10.0 micromoles."', '    dose_prompt = "Something else"'),
    }
    for name, (a, b) in targets.items():
        assert src.count(a) == 1, name
        bad = tmp_path / f"bad_{name.replace(' ', '_')}.py"
        bad.write_text(src.replace(a, b, 1), encoding="utf-8")
        monkeypatch.setattr(p10_pertdit_emb, "SRC_SHA1", p10_pertdit_emb.sha1_file(str(bad)))
        with pytest.raises(SystemExit) as exc:
            p10_pertdit_emb.make_emb_script(str(bad), str(tmp_path / "out.py"))
        assert f"make_emb_script: {name}" in str(exc.value), (name, str(exc.value))
    monkeypatch.setattr(p10_pertdit_emb, "SRC_SHA1", "0" * 40)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.make_emb_script(VENDOR_SRC, str(tmp_path / "out.py"))
    assert "source sha1" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.make_emb_script(src, str(tmp_path / "out.py"))
    assert "must be a file path" in str(exc.value)


def test_pi_build_inputs_refuses_in_memory_inputs(synthetic_data, tmp_path):
    """PI: guard (g1) can't be bypassed by handing build_inputs a dict."""
    table = np.load(synthetic_data["idx2smi_path"], allow_pickle=True).item()
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], table, str(tmp_path / "o"))
    assert "must be file paths" in str(exc.value)


# ==============================================================================
# 5. Generated script execution for real with stub transformers
# ==============================================================================

class StubTokenizerOutput:
    def __init__(self, text):
        self.input_ids = torch.tensor([[101, 102]], dtype=torch.long)
        self.text = text

    def to(self, device):
        return self


class StubBERTTokenizerOutput(dict):
    def __init__(self, text):
        super().__init__()
        self["input_ids"] = torch.tensor([[1, 2, 3]], dtype=torch.long)
        self["text"] = text
        self.text = text

    def to(self, device):
        return self


class StubBERTModelOutput:
    def __init__(self, text):
        L = len(text) % 7 + 2
        self.last_hidden_state = torch.full((1, L, 1024), float(len(text)), dtype=torch.float32)


def test_generated_script_runs_with_stub_transformers(tmp_path, monkeypatch, capsys):
    """Execute generated script end-to-end with stub transformers, verify behavior and cache skip."""
    # Run on CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    recorded_generate_kwargs = []
    recorded_bert_prompts = []

    class StubT5Tokenizer:
        def __init__(self, *args, **kwargs):
            self.last_text = "smiles"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, text, *args, **kwargs):
            self.last_text = text
            return StubTokenizerOutput(text)

        def decode(self, token_ids, **kwargs):
            return "caption of " + getattr(self, "last_text", "smiles")

    class StubT5Model:
        def __init__(self):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, dev):
            return self

        def eval(self):
            return self

        def generate(self, input_ids, **kwargs):
            recorded_generate_kwargs.append(kwargs)
            return torch.tensor([[10, 20]], dtype=torch.long)

    class StubAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, text, *args, **kwargs):
            recorded_bert_prompts.append(text)
            return StubBERTTokenizerOutput(text)

    class StubAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, dev):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            text = inputs.get("text", "dummy")
            return StubBERTModelOutput(text)

    # Install stub transformers in sys.modules
    stub_transformers = types.ModuleType("transformers")
    stub_transformers.T5Tokenizer = StubT5Tokenizer
    stub_transformers.T5ForConditionalGeneration = StubT5Model
    stub_transformers.AutoTokenizer = StubAutoTokenizer
    stub_transformers.AutoModel = StubAutoModel
    monkeypatch.setitem(sys.modules, "transformers", stub_transformers)

    # Layout tmp ROOT
    root_dir = tmp_path / "root"
    exp_dir = root_dir / "experiments" / "main_benchmark" / "data_prep"
    exp_dir.mkdir(parents=True)
    script_path = str(exp_dir / "prepare_pertdit_embeddings_p10.py")
    p10_pertdit_emb.make_emb_script(VENDOR_SRC, script_path)

    # 5-compound world
    compounds = {0: "CCO", 1: "CCN", 2: "CCC", 3: "CCF", 4: "CCl"}
    xpert_dir = root_dir / "data" / "XPert" / "processed_data"
    xpert_dir.mkdir(parents=True)
    idx2smi_path = str(xpert_dir / "all_drugs_idx2smi_8981.npy")
    np.save(idx2smi_path, compounds, allow_pickle=True)

    pertdit_dir = root_dir / "data" / "PertDiT"
    extracted_dir = pertdit_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    torch.save({"negative_ctrl": torch.full((2, 1024), float("nan"))}, str(extracted_dir / "pert_smiles_emb.pkl"))

    dose_list_path = str(pertdit_dir / "p10_dose_list.json")
    with open(dose_list_path, "w", encoding="utf-8") as f:
        json.dump([0.37, 10.0], f)

    # First run: generates captions and embeddings
    monkeypatch.setattr(sys, "argv", [script_path])
    res_globals = runpy.run_path(script_path, run_name="__main__")

    # Assert generation kwargs
    assert len(recorded_generate_kwargs) == 5
    assert recorded_generate_kwargs[0]["num_beams"] == 5
    assert recorded_generate_kwargs[0]["max_length"] == 512

    # Assert dose prompt strings
    assert "The dosage is 10.0 micromoles." in recorded_bert_prompts
    assert "The dosage is 0.37 micromoles." in recorded_bert_prompts

    out1 = capsys.readouterr()
    assert "P10 (RESULTS 98.15 ruling 4): MolT5 captions on cpu" in out1.out

    out_dir = str(root_dir / "data" / "PertDiT" / "bench_drug_emb")
    p10_pertdit_emb.check_outputs(out_dir, idx2smi_path, dose_list_path)

    # Second run: full cache -> MolT5 stub must NEVER be called
    class ExplodingT5:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MolT5 was instantiated when cache was full!")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("MolT5 was instantiated when cache was full!")

    stub_transformers.T5Tokenizer = ExplodingT5
    stub_transformers.T5ForConditionalGeneration = ExplodingT5

    recorded_bert_prompts.clear()
    runpy.run_path(script_path, run_name="__main__")
    assert len(recorded_bert_prompts) >= 5

    out2 = capsys.readouterr()
    assert "All 5 captions already cached; skipping MolT5" in out2.out


# ==============================================================================
# 6. Kernel builder static checks & end-to-end kernel execution
# ==============================================================================

def test_kernel_builder_static(tmp_path, synthetic_data, monkeypatch):
    """Static checks for generated kernels: 5+1 folders, metadata, ast parsing, env var defaults."""
    inputs_dir = tmp_path / "inputs"
    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(synthetic_data["idx2smi_path"]))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 12)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])
    p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], synthetic_data["idx2smi_path"], str(inputs_dir))

    # Also make prepare_pertdit_embeddings_p10.py in inputs_dir
    p10_pertdit_emb.make_emb_script(VENDOR_SRC, str(inputs_dir / "prepare_pertdit_embeddings_p10.py"))

    # PI: no fallbacks -- without p10_pertdit_emb.py in the folder the builder refuses, by name
    out_root = tmp_path / "kernels"
    with pytest.raises(SystemExit) as exc:
        make_p10_d_emb_kernels.build_kernels(str(out_root), str(inputs_dir))
    assert "p10_pertdit_emb.py" in str(exc.value) and "FATAL" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        make_p10_d_emb_kernels.build_kernels(str(out_root), str(tmp_path / "no_such_dir"))
    assert "inputs_dir does not exist" in str(exc.value)
    shutil.copyfile(p10_pertdit_emb.__file__, str(inputs_dir / "p10_pertdit_emb.py"))
    make_p10_d_emb_kernels.build_kernels(str(out_root), str(inputs_dir))

    # 5 caption + 1 merge = 6 folders
    folders = sorted(os.listdir(str(out_root)))
    expected_folders = [f"kern_p10d_cap_s{k}" for k in range(5)] + ["kern_p10d_emb_merge"]
    assert folders == expected_folders

    for k in range(5):
        k_dir = out_root / f"kern_p10d_cap_s{k}"
        meta = json.load(open(str(k_dir / "kernel-metadata.json"), encoding="utf-8"))
        assert meta["enable_gpu"] is False
        assert meta["enable_internet"] is True
        assert meta["docker_image"] == make_p10_d_emb_kernels.DEFAULT_DOCKER_IMAGE
        assert meta["dataset_sources"] == ["apexblue/p10d-emb-inputs"]

        code = open(str(k_dir / f"lincs-p10d-cap-s{k}.py"), encoding="utf-8").read()
        ast.parse(code)
        # Check env var defaults are real paths
        assert 'os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")' in code
        assert 'os.environ.get("KAGGLE_TEMP_DIR", "/kaggle/temp")' in code
        assert 'os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")' in code

    # Merge metadata
    m_dir = out_root / "kern_p10d_emb_merge"
    m_meta = json.load(open(str(m_dir / "kernel-metadata.json"), encoding="utf-8"))
    assert m_meta["enable_gpu"] is False
    assert m_meta["enable_internet"] is True
    assert m_meta["docker_image"] == make_p10_d_emb_kernels.DEFAULT_DOCKER_IMAGE
    assert m_meta["kernel_sources"] == [f"apexblue/lincs-p10d-cap-s{k}" for k in range(5)]
    m_code = open(str(m_dir / "lincs-p10d-emb-merge.py"), encoding="utf-8").read()
    ast.parse(m_code)
    assert 'os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")' in m_code
    assert 'os.environ.get("KAGGLE_TEMP_DIR", "/kaggle/temp")' in m_code
    assert 'os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")' in m_code


def test_pi_generated_kernel_runs_end_to_end(tmp_path, synthetic_data, monkeypatch):
    """Execute generated caption kernel and merge kernel as subprocesses with redirected paths and stubs."""
    inputs_dir = tmp_path / "inputs"
    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(synthetic_data["idx2smi_path"]))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 12)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])
    p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], synthetic_data["idx2smi_path"], str(inputs_dir))

    # Generated script in inputs_dir
    p10_pertdit_emb.make_emb_script(VENDOR_SRC, str(inputs_dir / "prepare_pertdit_embeddings_p10.py"))

    # Copy p10_pertdit_emb.py into inputs_dir
    shutil.copyfile(p10_pertdit_emb.__file__, str(inputs_dir / "p10_pertdit_emb.py"))

    kernels_dir = tmp_path / "kernels"
    make_p10_d_emb_kernels.build_kernels(str(kernels_dir), str(inputs_dir))

    # Setup shim directory with stubs
    shim = tmp_path / "shim"
    shim.mkdir()

    # Stub transformers package
    trans_dir = shim / "transformers"
    trans_dir.mkdir()
    (trans_dir / "__init__.py").write_text(
        '''
import torch
__version__ = "4.39.1"

class _TokOut(dict):
    def __init__(self, text):
        super().__init__()
        self["input_ids"] = torch.tensor([[101, 102]], dtype=torch.long)
        self["text"] = text
        self.input_ids = self["input_ids"]
        self.text = text
    def to(self, dev): return self

class _BertOut:
    def __init__(self, text):
        self.last_hidden_state = torch.full((1, len(text) % 7 + 2, 1024), float(len(text)), dtype=torch.float32)

class T5Tokenizer:
    def __init__(self, *args, **kwargs):
        self.last_text = "smiles"
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def __call__(self, text, *args, **kwargs):
        self.last_text = text
        return _TokOut(text)
    def decode(self, token_ids, **kwargs): return "caption of " + getattr(self, "last_text", "smiles")

class T5ForConditionalGeneration:
    def __init__(self): pass
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def to(self, dev): return self
    def eval(self): return self
    def generate(self, input_ids, **kwargs): return torch.tensor([[10, 20]], dtype=torch.long)

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def __call__(self, text, *args, **kwargs): return _TokOut(text)

class AutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def to(self, dev): return self
    def eval(self): return self
    def __call__(self, **inputs):
        text = inputs.get("text", "dummy")
        return _BertOut(text)
''',
        encoding="utf-8",
    )

    # Stub tokenizers and sentencepiece
    (shim / "tokenizers").mkdir()
    (shim / "tokenizers" / "__init__.py").write_text('__version__ = "0.15.2"\n', encoding="utf-8")
    (shim / "sentencepiece").mkdir()
    (shim / "sentencepiece" / "__init__.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8")

    # Stub huggingface_hub
    (shim / "huggingface_hub").mkdir()
    (shim / "huggingface_hub" / "__init__.py").write_text(
        f'''
__version__ = "0.24.6"
class _Info:
    def __init__(self, sha): self.sha = sha
class HfApi:
    def model_info(self, repo):
        if repo == "laituan245/molt5-large-smiles2caption":
            return _Info("{make_p10_d_emb_kernels.PIN_MOLT5_REV}")
        return _Info("{make_p10_d_emb_kernels.PIN_BERT_REV}")
''',
        encoding="utf-8",
    )

    # Fake HF cache files
    fake_hf_temp = tmp_path / "temp_cap" / "hf"
    molt5_snap = fake_hf_temp / "hub" / "models--laituan245--molt5-large-smiles2caption" / "snapshots" / make_p10_d_emb_kernels.PIN_MOLT5_REV
    bert_snap = fake_hf_temp / "hub" / "models--michiyasunaga--BioLinkBERT-large" / "snapshots" / make_p10_d_emb_kernels.PIN_BERT_REV
    molt5_snap.mkdir(parents=True)
    bert_snap.mkdir(parents=True)
    (molt5_snap / "pytorch_model.bin").write_bytes(b"fake molt5 weights")
    (bert_snap / "pytorch_model.bin").write_bytes(b"fake bert weights")

    fake_m_sha256 = p10_pertdit_emb.sha256_file(str(molt5_snap / "pytorch_model.bin"))
    fake_b_sha256 = p10_pertdit_emb.sha256_file(str(bert_snap / "pytorch_model.bin"))

    # 1. Run caption kernel 0
    cap_work = tmp_path / "work_cap"
    cap_work.mkdir()
    cap_temp = tmp_path / "temp_cap"

    cap_script = kernels_dir / "kern_p10d_cap_s0" / "lincs-p10d-cap-s0.py"
    cap_code = cap_script.read_text(encoding="utf-8")
    # Swap weight pins
    cap_code = cap_code.replace(make_p10_d_emb_kernels.PIN_MOLT5_WEIGHTS_SHA256, fake_m_sha256)
    cap_code = cap_code.replace(make_p10_d_emb_kernels.PIN_BERT_WEIGHTS_SHA256, fake_b_sha256)
    run_cap_script = tmp_path / "run_cap_s0.py"
    run_cap_script.write_text(cap_code, encoding="utf-8")

    env_cap = dict(
        os.environ,
        PYTHONPATH=str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        KAGGLE_INPUT_DIR=str(inputs_dir),
        KAGGLE_TEMP_DIR=str(cap_temp),
        KAGGLE_WORKING_DIR=str(cap_work),
        P10D_SKIP_PIP_INSTALL="1",
        CUDA_VISIBLE_DEVICES="",
    )
    r_cap = subprocess.run([sys.executable, str(run_cap_script)], capture_output=True, text=True, env=env_cap, timeout=300)
    assert r_cap.returncode == 0, r_cap.stdout + "\n" + r_cap.stderr

    # Caption kernel working dir has exactly 2 files
    assert sorted(os.listdir(str(cap_work))) == sorted(["_caption_cache_s0.pkl", "cap_record_s0.json"])

    # 2. Prepare inputs for merge kernel: generate caption caches for slices 0..4
    merge_input = tmp_path / "merge_input"
    shutil.copytree(str(inputs_dir), str(merge_input))

    for k in range(5):
        slice_dict = np.load(str(inputs_dir / f"idx2smi_slice{k}.npy"), allow_pickle=True).item()
        slice_cache = {s: f"caption of {s}" for s in slice_dict.values()}
        c_path = str(merge_input / f"_caption_cache_s{k}.pkl")
        with open(c_path, "wb") as f:
            pickle.dump(slice_cache, f)
        c_sha1 = p10_pertdit_emb.sha1_file(c_path)
        r_path = str(merge_input / f"cap_record_s{k}.json")
        rec = _pi_full_record(inputs_dir, k, c_sha1, fake_b_sha256)  # review 080 C1: the merge checks the record's pins
        with open(r_path, "w", encoding="utf-8") as f:
            json.dump(rec, f)

    merge_work = tmp_path / "work_merge"
    merge_work.mkdir()
    merge_temp = tmp_path / "temp_merge"
    merge_hf = merge_temp / "hf" / "hub" / "models--michiyasunaga--BioLinkBERT-large" / "snapshots" / make_p10_d_emb_kernels.PIN_BERT_REV
    merge_hf.mkdir(parents=True)
    (merge_hf / "pytorch_model.bin").write_bytes(b"fake bert weights")

    merge_script = kernels_dir / "kern_p10d_emb_merge" / "lincs-p10d-emb-merge.py"
    merge_code = merge_script.read_text(encoding="utf-8")
    merge_code = merge_code.replace(make_p10_d_emb_kernels.PIN_BERT_WEIGHTS_SHA256, fake_b_sha256)
    run_merge_script = tmp_path / "run_merge.py"
    run_merge_script.write_text(merge_code, encoding="utf-8")

    env_merge = dict(
        os.environ,
        PYTHONPATH=str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        KAGGLE_INPUT_DIR=str(merge_input),
        KAGGLE_TEMP_DIR=str(merge_temp),
        KAGGLE_WORKING_DIR=str(merge_work),
        P10D_SKIP_PIP_INSTALL="1",
        CUDA_VISIBLE_DEVICES="",
    )
    r_merge = subprocess.run([sys.executable, str(run_merge_script)], capture_output=True, text=True, env=env_merge, timeout=300)
    assert r_merge.returncode == 0, r_merge.stdout + "\n" + r_merge.stderr

    # Merge kernel working dir has exactly 4 files
    assert sorted(os.listdir(str(merge_work))) == sorted([
        "_caption_cache.pkl",
        "bench_dose_emb.pt",
        "bench_drug_emb.pkl",
        "emb_record.json",
    ])


def test_mutation_hf_revision_fails_fast(tmp_path, synthetic_data, monkeypatch):
    """Mutation sanity: with HF stub reporting a different sha, kernel exits FATAL before script runs."""
    inputs_dir = tmp_path / "inputs"
    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(synthetic_data["idx2smi_path"]))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 12)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])
    p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], synthetic_data["idx2smi_path"], str(inputs_dir))
    p10_pertdit_emb.make_emb_script(VENDOR_SRC, str(inputs_dir / "prepare_pertdit_embeddings_p10.py"))
    shutil.copyfile(p10_pertdit_emb.__file__, str(inputs_dir / "p10_pertdit_emb.py"))

    kernels_dir = tmp_path / "kernels"
    make_p10_d_emb_kernels.build_kernels(str(kernels_dir), str(inputs_dir))

    shim = tmp_path / "shim_mut"
    shim.mkdir()
    (shim / "transformers").mkdir()
    (shim / "transformers" / "__init__.py").write_text('__version__ = "4.39.1"\n', encoding="utf-8")
    (shim / "tokenizers").mkdir()
    (shim / "tokenizers" / "__init__.py").write_text('__version__ = "0.15.2"\n', encoding="utf-8")
    (shim / "sentencepiece").mkdir()
    (shim / "sentencepiece" / "__init__.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8")

    # Mutated HF stub returns wrong sha
    (shim / "huggingface_hub").mkdir()
    (shim / "huggingface_hub" / "__init__.py").write_text(
        '''
__version__ = "0.24.6"
class _Info:
    def __init__(self, sha): self.sha = sha
class HfApi:
    def model_info(self, repo):
        return _Info("WRONG_SHA_MUTANT")
''',
        encoding="utf-8",
    )

    cap_script = kernels_dir / "kern_p10d_cap_s0" / "lincs-p10d-cap-s0.py"
    temp_dir = tmp_path / "temp"
    work_dir = tmp_path / "work"
    temp_dir.mkdir()
    work_dir.mkdir()

    env = dict(
        os.environ,
        PYTHONPATH=str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        KAGGLE_INPUT_DIR=str(inputs_dir),
        KAGGLE_TEMP_DIR=str(temp_dir),
        KAGGLE_WORKING_DIR=str(work_dir),
        P10D_SKIP_PIP_INSTALL="1",
    )
    r = subprocess.run([sys.executable, str(cap_script)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0
    # Must exit FATAL on HF revision check, before any emb.log is written
    assert "FATAL: HF revision check failed" in r.stderr or "FATAL: HF revision check failed" in r.stdout
    assert not (temp_dir / "emb.log").exists()



# ==============================================================================
# PI (mutation check, W39): 11 of 31 mutants survived W39's suite -- every kernel guard was exercised only on its happy path.
# Each world below fails exactly one guard and asserts that guard's FATAL message.
# ==============================================================================

def test_pi_check_outputs_dtype_and_dose_finite(tmp_path):
    """(o2) a float64 drug tensor fails on dtype; (o4) a NaN dose tensor fails on finiteness."""
    out = tmp_path / "o"
    out.mkdir()
    torch.save({0: torch.randn(5, 1024, dtype=torch.float64), "negative_ctrl": torch.full((2, 1024), float("nan"))},
               str(out / "bench_drug_emb.pkl"))
    torch.save({10.0: torch.randn(1, 1024)}, str(out / "bench_dose_emb.pt"))
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.check_outputs(str(out), {0: "CCO"}, [10.0])
    assert "guard (o2)" in str(exc.value) and "dtype" in str(exc.value)
    torch.save({0: torch.randn(5, 1024), "negative_ctrl": torch.full((2, 1024), float("nan"))},
               str(out / "bench_drug_emb.pkl"))
    torch.save({10.0: torch.full((3, 1024), float("nan"))}, str(out / "bench_dose_emb.pt"))
    with pytest.raises(SystemExit) as exc:
        p10_pertdit_emb.check_outputs(str(out), {0: "CCO"}, [10.0])
    assert "guard (o4)" in str(exc.value) and "non-finite" in str(exc.value)


_PI_STUB_TRANSFORMERS = """
import os, torch
__version__ = "4.39.1"
class _Tok(dict):
    def __init__(self, text):
        super().__init__(); self["input_ids"] = torch.tensor([[1, 2]]); self.input_ids = self["input_ids"]; self["text"] = text
    def to(self, dev): return self
class _Out:
    def __init__(self, text):
        v = float("nan") if os.environ.get("PI_STUB_BERT_NAN") == "1" else float(len(text))
        self.last_hidden_state = torch.full((1, len(text) % 7 + 2, 1024), v, dtype=torch.float32)
class T5Tokenizer:
    last = "x"
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def __call__(self, text, *a, **k): T5Tokenizer.last = text; return _Tok(text)
    def decode(self, ids, **k): return "caption of " + T5Tokenizer.last
class T5ForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def to(self, d): return self
    def eval(self): return self
    def generate(self, ids, **k): return torch.tensor([[3, 4]])
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def __call__(self, text, *a, **k): return _Tok(text)
class AutoModel:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def to(self, d): return self
    def eval(self): return self
    def __call__(self, **inputs): return _Out(inputs.get("text", "dummy"))
"""


def _pi_world(tmp_path, synthetic_data, monkeypatch, script_edit=None):
    """Inputs, kernels and stubs for one guard world. script_edit (old, new) alters the generated script BEFORE the build,
    so the builder pins the altered bytes (the mounted-file guard then passes)."""
    inputs = tmp_path / "inputs"
    monkeypatch.setattr(p10_pertdit_emb, "IDX2SMI_FULL_SHA1", p10_pertdit_emb.sha1_file(synthetic_data["idx2smi_path"]))
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_N_COMPOUNDS", 12)
    monkeypatch.setattr(p10_pertdit_emb, "EXPECT_DOSES", synthetic_data["expected_doses"])
    p10_pertdit_emb.build_inputs(synthetic_data["bundle_path"], synthetic_data["idx2smi_path"], str(inputs))
    script = inputs / "prepare_pertdit_embeddings_p10.py"
    p10_pertdit_emb.make_emb_script(VENDOR_SRC, str(script))
    if script_edit:
        txt = script.read_text(encoding="utf-8")
        assert txt.count(script_edit[0]) == 1
        script.write_text(txt.replace(*script_edit), encoding="utf-8")
    shutil.copyfile(p10_pertdit_emb.__file__, str(inputs / "p10_pertdit_emb.py"))
    kernels = tmp_path / "kernels"
    make_p10_d_emb_kernels.build_kernels(str(kernels), str(inputs))
    shim = tmp_path / "shim"
    for pkg, body in (("transformers", _PI_STUB_TRANSFORMERS), ("tokenizers", '__version__ = "0.15.2"\n'),
                      ("sentencepiece", '__version__ = "0.2.0"\n')):
        (shim / pkg).mkdir(parents=True)
        (shim / pkg / "__init__.py").write_text(body, encoding="utf-8")
    (shim / "huggingface_hub").mkdir()
    (shim / "huggingface_hub" / "__init__.py").write_text(
        '__version__ = "0.24.6"\nclass _I:\n    def __init__(self, s): self.sha = s\nclass HfApi:\n'
        '    def model_info(self, repo):\n'
        f'        return _I("{make_p10_d_emb_kernels.PIN_MOLT5_REV}" if "molt5" in repo else "{make_p10_d_emb_kernels.PIN_BERT_REV}")\n',
        encoding="utf-8")
    return {"inputs": inputs, "kernels": kernels, "shim": shim}


def _pi_fake_hf(temp_dir, molt5=True):
    """Fake HF snapshot files under temp_dir/hf; returns their sha256s."""
    out = {}
    for name, rev, on in (("models--laituan245--molt5-large-smiles2caption", make_p10_d_emb_kernels.PIN_MOLT5_REV, molt5),
                          ("models--michiyasunaga--BioLinkBERT-large", make_p10_d_emb_kernels.PIN_BERT_REV, True)):
        if not on:
            continue
        snap = temp_dir / "hf" / "hub" / name / "snapshots" / rev
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "pytorch_model.bin").write_bytes(("fake " + name).encode())
        out[name] = p10_pertdit_emb.sha256_file(str(snap / "pytorch_model.bin"))
    return out


def _pi_run(code_path, input_dir, temp_dir, work_dir, shim, swaps, extra_env=None):
    code = code_path.read_text(encoding="utf-8")
    for a, b in swaps:
        code = code.replace(a, b)
    run = temp_dir.parent / ("run_" + code_path.name)
    run.write_text(code, encoding="utf-8")
    work_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               KAGGLE_INPUT_DIR=str(input_dir), KAGGLE_TEMP_DIR=str(temp_dir), KAGGLE_WORKING_DIR=str(work_dir),
               P10D_SKIP_PIP_INSTALL="1", CUDA_VISIBLE_DEVICES="", **(extra_env or {}))
    return subprocess.run([sys.executable, str(run)], capture_output=True, text=True, env=env, timeout=300)


def _pi_cap(w, tmp_path, tag, swap_molt5=True, input_dir=None, pre_cache=None):
    temp = tmp_path / f"temp_{tag}"
    sha = _pi_fake_hf(temp)
    swaps = [(make_p10_d_emb_kernels.PIN_BERT_WEIGHTS_SHA256, sha["models--michiyasunaga--BioLinkBERT-large"])]
    if swap_molt5:
        swaps.append((make_p10_d_emb_kernels.PIN_MOLT5_WEIGHTS_SHA256, sha["models--laituan245--molt5-large-smiles2caption"]))
    if pre_cache is not None:
        cdir = temp / "p10d" / "data" / "PertDiT" / "bench_drug_emb"
        cdir.mkdir(parents=True)
        with open(cdir / "_caption_cache.pkl", "wb") as f:
            pickle.dump(pre_cache, f)
    return _pi_run(w["kernels"] / "kern_p10d_cap_s0" / "lincs-p10d-cap-s0.py", input_dir or w["inputs"], temp,
                   tmp_path / f"work_{tag}", w["shim"], swaps)


def _pi_merge_input(w, tmp_path, tag, bert_sha=None):
    mi = tmp_path / f"merge_in_{tag}"
    shutil.copytree(str(w["inputs"]), str(mi))
    for k in range(5):
        sl = np.load(str(w["inputs"] / f"idx2smi_slice{k}.npy"), allow_pickle=True).item()
        cp = mi / f"_caption_cache_s{k}.pkl"
        with open(cp, "wb") as f:
            pickle.dump({smi: "caption of " + smi for smi in sl.values()}, f)
        (mi / f"cap_record_s{k}.json").write_text(
            json.dumps(_pi_full_record(w["inputs"], k, p10_pertdit_emb.sha1_file(str(cp)), bert_sha or _PI_FAKE_BERT_SHA256)),
            encoding="utf-8")
    return mi


def _pi_merge(w, tmp_path, tag, mi, swap_bert=True, extra_env=None):
    temp = tmp_path / f"temp_{tag}"
    sha = _pi_fake_hf(temp, molt5=False)
    swaps = [(make_p10_d_emb_kernels.PIN_BERT_WEIGHTS_SHA256, sha["models--michiyasunaga--BioLinkBERT-large"])] if swap_bert else []
    return _pi_run(w["kernels"] / "kern_p10d_emb_merge" / "lincs-p10d-emb-merge.py", mi, temp, tmp_path / f"work_{tag}",
                   w["shim"], swaps, extra_env)


def _pi_fatal(r, needle):
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "FATAL" in out and needle in out, out[-3000:]


def test_pi_caption_kernel_guards_each_fail_loud(tmp_path, synthetic_data, monkeypatch):
    w = _pi_world(tmp_path, synthetic_data, monkeypatch)
    ok = _pi_cap(w, tmp_path, "ok")
    assert ok.returncode == 0, (ok.stdout + ok.stderr)[-3000:]
    # a mounted input whose bytes differ from the build's
    bad_in = tmp_path / "bad_in"
    shutil.copytree(str(w["inputs"]), str(bad_in))
    with open(bad_in / "dose_list.json", "a", encoding="utf-8") as f:
        f.write(" ")
    _pi_fatal(_pi_cap(w, tmp_path, "mounted", input_dir=bad_in), "mounted file dose_list.json sha1 mismatch")
    # MolT5 weights that are not the pinned bytes
    _pi_fatal(_pi_cap(w, tmp_path, "molt5", swap_molt5=False), "MolT5 weights sha256 mismatch")
    # a cache holding a SMILES outside the slice (pre-seeded where their script resumes from)
    sl = np.load(str(w["inputs"] / "idx2smi_slice0.npy"), allow_pickle=True).item()
    _pi_fatal(_pi_cap(w, tmp_path, "extra", pre_cache={"C#N_not_in_slice": "caption"}), "caption cache keys mismatch")
    assert "C#N_not_in_slice" not in sl.values()


def test_pi_merge_kernel_guards_each_fail_loud(tmp_path, synthetic_data, monkeypatch):
    w = _pi_world(tmp_path, synthetic_data, monkeypatch)
    ok = _pi_merge(w, tmp_path, "ok", _pi_merge_input(w, tmp_path, "ok"))
    assert ok.returncode == 0, (ok.stdout + ok.stderr)[-3000:]
    rec = json.loads((tmp_path / "work_ok" / "emb_record.json").read_text(encoding="utf-8"))
    assert sorted(rec["slice_cache_sha1s"]) == [f"s{k}" for k in range(5)]
    # mounted input altered
    mi = _pi_merge_input(w, tmp_path, "mounted")
    with open(mi / "compound_map.json", "a", encoding="utf-8") as f:
        f.write(" ")
    _pi_fatal(_pi_merge(w, tmp_path, "mounted", mi), "mounted file compound_map.json sha1 mismatch")
    # a caption cache that is not the one its record names
    mi = _pi_merge_input(w, tmp_path, "rec")
    r2 = json.loads((mi / "cap_record_s2.json").read_text(encoding="utf-8"))
    r2["cache_sha1"] = "0" * 40
    (mi / "cap_record_s2.json").write_text(json.dumps(r2), encoding="utf-8")
    _pi_fatal(_pi_merge(w, tmp_path, "rec", mi), "caption cache s2 sha1 mismatch")
    # BioLinkBERT weights that are not the pinned bytes
    _pi_fatal(_pi_merge(w, tmp_path, "bert", _pi_merge_input(w, tmp_path, "bert", bert_sha=make_p10_d_emb_kernels.PIN_BERT_WEIGHTS_SHA256), swap_bert=False),
              "BioLinkBERT weights sha256 mismatch")
    # non-finite embeddings reach check_outputs inside the kernel
    _pi_fatal(_pi_merge(w, tmp_path, "nan", _pi_merge_input(w, tmp_path, "nan"), extra_env={"PI_STUB_BERT_NAN": "1"}),
              "guard (o2)")
    assert not (tmp_path / "work_nan" / "bench_drug_emb.pkl").exists()


def test_pi_merge_kernel_requires_the_skip_line(tmp_path, synthetic_data, monkeypatch):
    """A script that never prints their skip line (pinned as built, so the mounted guard passes) is refused."""
    w = _pi_world(tmp_path, synthetic_data, monkeypatch,
                  script_edit=("captions already cached; skipping MolT5", "captions already cached; skipping T5"))
    _pi_fatal(_pi_merge(w, tmp_path, "skip", _pi_merge_input(w, tmp_path, "skip")), "merge kernel log missing required line")



# ==============================================================================
# Review 080: the merge checks each slice record's pins (C1); records carry torch and the CPU (C2); versions asserted (C3)
# ==============================================================================
_PI_PINNED_VERSIONS = "transformers==4.39.1 tokenizers==0.15.2 sentencepiece==0.2.0 huggingface-hub==0.24.6"
_PI_FAKE_BERT_SHA256 = hashlib.sha256(b"fake models--michiyasunaga--BioLinkBERT-large").hexdigest()


def _pi_full_record(inputs_dir, k, cache_sha1, bert_sha256):
    """A caption record as a real caption kernel writes it, for the merge's review 080 C1 checks."""
    inputs_dir = str(inputs_dir)
    names = ["prepare_pertdit_embeddings_p10.py", "p10_pertdit_emb.py", f"idx2smi_slice{k}.npy",
             "pert_smiles_emb_placeholder.pkl", "dose_list.json", "compound_map.json", "inputs_manifest.json"]
    return {"slice": k, "cache_sha1": cache_sha1, "versions": _PI_PINNED_VERSIONS,
            "verified_sha1s": {n: p10_pertdit_emb.sha1_file(os.path.join(inputs_dir, n)) for n in names},
            "verified_sha256s": {"molt5_weights": make_p10_d_emb_kernels.PIN_MOLT5_WEIGHTS_SHA256, "bert_weights": bert_sha256}}


def test_pi_review080_merge_refuses_foreign_slice_records(tmp_path, synthetic_data, monkeypatch):
    """C1: a record for another slice, one that verified other inputs, one with other weights, one with other versions."""
    w = _pi_world(tmp_path, synthetic_data, monkeypatch)
    ok = _pi_merge(w, tmp_path, "ok", _pi_merge_input(w, tmp_path, "ok"))
    assert ok.returncode == 0, (ok.stdout + ok.stderr)[-3000:]
    rec = json.loads((tmp_path / "work_ok" / "emb_record.json").read_text(encoding="utf-8"))
    assert sorted(rec["slice_record_sha1s"]) == [f"s{k}" for k in range(5)]
    assert {"torch_version", "cpu_model"} <= set(rec) and "compound_map.json" in rec["verified_sha1s"]
    worlds = {
        "slice": (lambda r: r.update(slice=3), "is for slice 3, not 2"),
        "inputs": (lambda r: r["verified_sha1s"].update({"idx2smi_slice2.npy": "0" * 40}), "verified other inputs"),
        "weights": (lambda r: r["verified_sha256s"].update(molt5_weights="0" * 64), "weight sha256s"),
        "versions": (lambda r: r.update(versions="transformers==4.40.0"), "versions"),
    }
    for tag, (mutate, needle) in worlds.items():
        mi = _pi_merge_input(w, tmp_path, tag)
        r2 = json.loads((mi / "cap_record_s2.json").read_text(encoding="utf-8"))
        mutate(r2)
        (mi / "cap_record_s2.json").write_text(json.dumps(r2), encoding="utf-8")
        _pi_fatal(_pi_merge(w, tmp_path, tag, mi), needle)


def test_pi_review080_caption_record_and_version_assert(tmp_path, synthetic_data, monkeypatch):
    """C2: the caption record carries torch and the CPU; C3: a version line that isn't the pin is fatal."""
    w = _pi_world(tmp_path, synthetic_data, monkeypatch)
    ok = _pi_cap(w, tmp_path, "ok")
    assert ok.returncode == 0, (ok.stdout + ok.stderr)[-3000:]
    rec = json.loads((tmp_path / "work_ok" / "cap_record_s0.json").read_text(encoding="utf-8"))
    assert rec["versions"] == _PI_PINNED_VERSIONS and rec["torch_version"] == torch.__version__ and rec["cpu_model"]
    (w["shim"] / "tokenizers" / "__init__.py").write_text('__version__ = "0.15.1"\n', encoding="utf-8")
    _pi_fatal(_pi_cap(w, tmp_path, "ver"), "installed versions")
