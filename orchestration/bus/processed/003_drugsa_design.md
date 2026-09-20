# PACKET 003 — PRE-SPEND DESIGN REVIEW: drug self-attention
packet_id: 003
created: 2026-09-20
repo_commit: f8e185c
type: **DESIGN + COST REVIEW BEFORE ANY GPU IS SPENT** — no results yet

The principal's standing instruction: no GPU hours are committed until the design and its cost have been
reviewed. Nothing here has been run. **Attack the design and the cost estimate, not a result.**

## ARTEFACTS
```
model/v9/modules_v9.py::_DrugBlock              the new module
model/v9/model_v9.py::PerturbBlock              where it is wired (behind cfg.drug_self_attn, default False)
model/v9/config_v9.py                           the flag
model/v9/test_drugsa_v9.py                      5 discrimination checks, all passing
model/v9/probe_v9.py                            the ablate-to-mean prober that produced the -0.025 below
model/results/v9_probe_ckpt_v9_fold0_seed0.json the SA-OFF ablation table
external/v9_checkpoints/                         existing fold0 checkpoints (SA-off)
```
`git show --stat f8e185c` is in `003_gitstat.txt`.

## OBJECTIVE
`probe_v9.py` measured, by ablate-to-mean on a trained model, that **removing the per-atom drug tokens
IMPROVES accuracy**: dPearson **−0.007 / −0.025 / −0.022** on unseen-cell / unseen-compound / unseen-both.
Atom tokens are a load-bearing design commitment of this project (they are the substrate for atom→gene
attribution), so a negative contribution needs either a mechanism or a deletion.

## FUNCTION — what the code change does
Read from XPert's executed path: `crossEncoder.forward` calls `drug_SA(drug, ...)` **inside every
cross-encoder block**, so its 978 gene queries cross-attend over a molecule whose atoms have been mutually
contextualised. v9 built `D = [global_token; linear(atoms)]` **once, outside the block loop**, and handed
the same `D` to every block — so v9's gene queries attend over a **bag of independently projected per-atom
Uni-Mol vectors with no intramolecular structure**.

`_DrugBlock` is self-attention + SwiGLU over the drug sequence with the ragged-atom `key_mask`, applied
before cross-attention in each `PerturbBlock`; the updated `D` is returned so contextualisation compounds
across blocks.

Verified by `test_drugsa_v9.py` (checks that it DISCRIMINATES, not that it exists):
perturbing atom 3 moves atom 1 by |d|max **0.2754**; with the arm off atom 1 is bit-identical under the
same perturbation; changing a **padded** atom moves a real one by **0.00e+00**. `test_v9.py` still 55/55
with the arm off.

## MEASURED COSTS (not estimates)

**Parameters** — at FULL width (d_model 256, l_perturb 4), measured by construction:

| arm | params |
|---|---|
| drug_self_attn=False | 10,466,725 |
| drug_self_attn=True | 13,612,741 |
| delta | **+3,146,016 = +30.06 %** |

⚠️ **This corrects our own §47.6**, which quoted +11.38 % from a reduced-width test (d_model 64). The
capacity confound is ~3× larger than we recorded.

**Wall clock** — from §29.1, Kaggle T4×2, d_model 256, full depth, 12 epochs: **5.62 h** over 179,772
training rows = 1,636 s/epoch = 0.0091 s per row-epoch. Scaling by training rows:

| split | train rows | projected 12-epoch run |
|---|---|---|
| our fold0 (matches the −0.025 basis) | 179,772 | **~5.6 h** |
| `split_cold_drug_1` | 55,385 | ~1.7 h |
| `split_cold_cell_1` | 47,509 | ~1.5 h |

Weekly GPU quota is **30 h**.

## THE DESIGN I PROPOSE, AND WHY IT IS ONE RUN

The obvious design is three arms — SA-off, SA-on, and a matched-capacity control — compared on headline
accuracy. At fold0 that is ~16.8 h, 56 % of the weekly quota, and the accuracy comparison is between-run,
so it also needs ≥3 seeds to clear the ±0.046 band. That is not affordable and would not be decisive.

**Proposed instead: one SA-on run at fold0 (~5.6 h), then TWO ablations WITHIN it:**

| ablation (ablate-to-mean, within-run, same weights) | prediction if contextualisation is the mechanism |
|---|---|
| ablate **atom tokens** | dPearson becomes **positive** (atoms now help) |
| ablate **the `drug_sa` module** | atom contribution **reverts toward harmful**, recovering ≈ −0.025 |

Both are within-run on identical signatures, so neither carries seed variance [method rule 7], and the
**interaction between them is measured inside a single set of weights** — which is what makes it one run
rather than three.

**And if this is the decisive test, the matched-capacity control is not needed at all**, because the
capacity is identical across both ablation arms. Capacity only confounds the *headline accuracy*
comparison, which this design does not make.

## ASKS — all about the design, before money is spent

1. **Is the within-run interaction test actually decisive?** My concern with the simpler version — compare
   the SA-on run's atom-ablation delta against §37's SA-off −0.025 — is that it compares two ablation
   magnitudes across two training runs, so run variance re-enters through the back door. Does the
   `drug_sa`-ablation arm genuinely remove that problem, or have I just moved it?
2. **Is ablating `drug_sa` to its mean a valid no-op?** Ablate-to-mean on a self-attention block is not
   obviously the same kind of operation as ablating an input feature. What would the right neutralisation
   be — mean over the sequence axis, identity passthrough, or something else — and does the choice change
   what the test measures?
3. **Does dropping the matched-capacity control hold up?** Or does +30 % capacity change what the *atom
   ablation itself* measures, even within one run?
4. **Is fold0 the right split at 3.3× the cost?** The −0.025 basis lives there, but the cheaper mdmt
   splits are ~1.5–1.7 h. Is a same-split comparison worth 4 extra hours here?
5. **Is there a free or CPU-only version of any part of this** that we have missed, given this project has
   now twice paid for something already on disk [§46.1, §54.1]?

Free to ignore these and raise anything else. If the design is unsound, saying so before the spend is the
single most valuable output available.
