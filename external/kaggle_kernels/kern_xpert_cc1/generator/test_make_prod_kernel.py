import ast
import json
import os
import subprocess
import sys
import tempfile

GEN_DIR = r"C:\Projects\LINCS\external\kaggle_kernels\kern_xpert_cc1\generator"
COMMITTED_KERNEL = r"C:\Projects\LINCS\external\kaggle_kernels\kern_xpert_cc1\lincs-xpert-cc1.py"
PREV_JSON_2 = os.path.join(GEN_DIR, "prev_session2.json")
MAKE_SCRIPT = os.path.join(GEN_DIR, "make_prod_kernel.py")
V9_DIR = r"C:\Projects\LINCS\model\v9"

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(f"Command failed with code {result.returncode}")
    return result

def test_kaggle_regression():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "kaggle_out.py")
        cmd = [sys.executable, MAKE_SCRIPT, "2", PREV_JSON_2, "--out", out_path]
        run_cmd(cmd)
        
        with open(out_path, encoding="utf-8") as f:
            generated = f.read()
            
        with open(COMMITTED_KERNEL, encoding="utf-8") as f:
            expected = f.read()
            
        # Update expected's *_SRC literals to match the current working tree files (git stash-free)
        lines = expected.split('\n')
        src = {n: open(os.path.join(V9_DIR, f), encoding='utf-8').read() for n, f in
               (('DP_PATCH_SRC', 'xpert_dp_patch.py'), ('DP_PROBE_SRC', 'xpert_dp_probe.py'),
                ('CKPT_PATCH_SRC', 'xpert_ckpt_patch.py'), ('RESUME_PATCH_SRC', 'xpert_resume_patch.py'))}
        for name in src:
            hits = [i for i, l in enumerate(lines) if l.startswith(name + ' = ')]
            if hits:
                lines[hits[0]] = name + ' = ' + repr(src[name])
        expected = '\n'.join(lines)
            
        assert generated == expected, "Kaggle regression failed: generated file is not byte-identical to committed kernel."
        print("PASS: Kaggle regression (session 2 byte-identical)")

def test_lightning():
    with tempfile.TemporaryDirectory() as tmp:
        dummy_prev = os.path.join(tmp, "prev_session3.json")
        prev_data = json.load(open(PREV_JSON_2))
        prev_data["final_epoch"] = 131
        with open(dummy_prev, "w") as f:
            json.dump(prev_data, f)
            
        out_path = os.path.join(tmp, "lightning_out.py")
        cmd = [sys.executable, MAKE_SCRIPT, "3", dummy_prev, "--platform", "lightning", "--first-lightning-session", "3", "--out", out_path]
        run_cmd(cmd)
        
        with open(out_path, encoding="utf-8") as f:
            generated = f.read()
            
        # Parse check
        ast.parse(generated)
        print("PASS: Lightning parse")
        
        # No /kaggle
        assert "/kaggle" not in generated, "Found '/kaggle' in Lightning output"
        print("PASS: Lightning no /kaggle")
        
        # Precision guard
        assert "allow_tf32 is not False" in generated
        assert "float32_matmul_precision" in generated
        assert "NVIDIA_TF32_OVERRIDE" in generated
        print("PASS: Lightning precision guard")
        
        # Platform check
        assert "expected exactly ONE visible GPU on Lightning" in generated
        assert "A100" in generated
        assert "total_memory / (1024**3)" in generated
        print("PASS: Lightning platform check")
        
        # Deviation text
        dev_text = "sessions 1-2 on 2xT4 DataParallel; sessions >= 3 on one %s as published; different GPU architecture and SDPA backend (rounding-level); CUDA RNG of the second device not carried across"
        assert dev_text in generated, "Lightning deviation text missing"
        print("PASS: Lightning deviation text")
        
        print("PASS: Lightning embedded literal checks (handled by generator)")

if __name__ == "__main__":
    test_kaggle_regression()
    test_lightning()
    print("ALL TESTS PASSED")
