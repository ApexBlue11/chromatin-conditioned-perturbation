# TASK W13 — CUDA RNG restore across a change in GPU count, in `model/v9/xpert_resume_patch.py`

## Context
`model/v9/xpert_resume_patch.py` saves and restores the FULL training state of a third-party trainer (XPert) across
cloud sessions. It is production code for a pre-registered experiment: behaviour must not change in any case except the
one below. Read the whole file first, and `model/v9/test_xpert_resume_local.py` (its existing test, which must keep
passing unchanged).

Today `restore_state()` calls `torch.cuda.set_rng_state_all(st['rng']['cuda'])`. The saved state has **two** CUDA
generator states (it was saved on 2 GPUs). The next session runs on **one** GPU, where that call cannot restore
device 1. A later session might run on two GPUs again after a one-GPU save.

## Contract (binding — RESULTS 84.4 condition 2, from review 018 C2)
1. Add a function `map_cuda_rng(saved, n_devices, seed_fn, base_seed=2024, stride=1000)`:
   - `saved`: list of CPU ByteTensors (CUDA generator states as returned by `torch.cuda.get_rng_state_all()`).
   - For every device index `i < min(len(saved), n_devices)`: the target state is `saved[i]` (restored).
   - For every `i >= len(saved)` (more devices than saved): seed device `i` with `base_seed + stride * i` by calling
     `seed_fn(i, seed)`, which seeds that device and returns its resulting state as a CPU ByteTensor.
   - For every saved index `i >= n_devices` (fewer devices than saved): the state is dropped; record its sha1.
   - Returns `(targets, record)` where `targets` is the list of length `n_devices` and `record` is a JSON-serialisable
     dict: `{"n_saved", "n_devices", "restored": [indices], "seeded": {i: seed}, "dropped_sha1": {i: sha1hex}}`.
   - **After mapping, if `n_devices > 1`, no two target states may be byte-equal; raise `RuntimeError` otherwise.**
2. In `restore_state()`, replace the `set_rng_state_all` call with: compute the mapping (the real `seed_fn` uses
   `torch.cuda.default_generators[i].manual_seed(seed)` then `torch.cuda.get_rng_state(i)`), set each device's state
   to its target with `torch.cuda.set_rng_state(targets[i], i)`, print one line
   `LINCS CUDA RNG MAP <json of record>` (flush), and keep the record in `_STATE['cuda_rng_map']`.
3. **The in-process round-trip (`compare_states(st, live)` in `restore_state`) must compare the live CUDA states with
   the mapped `targets`, not with `st['rng']['cuda']`.** Every other field is compared exactly as today. When
   `len(saved) == n_devices` the targets are the saved states, so behaviour is byte-for-byte what it is today.
4. Do not change anything else: not the save path, not the other RNGs, not the hooks, not the prints that exist.

## Tests — new file `model/v9/test_cuda_rng_map.py`, runnable as `python model/v9/test_cuda_rng_map.py`
Use a FAKE `seed_fn` (e.g. returns `torch.tensor(list(seed.to_bytes(8,'little')) + [0]*8, dtype=torch.uint8)`) so the
mapping logic is tested without GPUs. Cases, each an assert:
- (a) 2 saved, 1 device: targets == [saved[0]]; dropped_sha1 has key 1 with the right sha1; seeded empty.
- (b) 1 saved, 2 devices: targets[0] == saved[0]; seeded == {1: 3024}; targets[1] == fake_seed_fn(1, 3024).
- (c) 2 saved, 2 devices, distinct: targets == saved; restored == [0, 1]; nothing seeded or dropped.
- (d) 2 saved, 2 devices, byte-identical states: raises RuntimeError.
- (e) 0 saved, 1 device: seeded == {0: 2024}.
- (f) the record is `json.dumps`-able.
- (g) **if** `torch.cuda.is_available()`: a real-device check on device 0 — build `saved = [s0, s1]` where `s0` is the
  current state of device 0 after `torch.cuda.manual_seed(7)` and `s1` differs; run the real restore mapping helper
  for `n_devices = torch.cuda.device_count()`; assert device 0's state now equals `s0`.
Then run `python model/v9/test_xpert_resume_local.py` and confirm it still prints `RESUME MECHANICS TEST PASSED`.

## Rules
- Only edit `model/v9/xpert_resume_patch.py` and create `model/v9/test_cuda_rng_map.py`. Nothing else.
- Use the interpreter `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` for every run. Do not install packages.
- No training, no long GPU jobs; the tests above take seconds.
- In your final message, paste the full output of both test runs. Do not summarise them.
