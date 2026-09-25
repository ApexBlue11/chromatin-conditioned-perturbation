# REVIEW OF PACKET 020
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 6a5192b

**Porting the gradient × activation probe to trained v9 is the right next question, and running it once, pre-registered,
is the right way to ask it.** Three things need changing before any code runs. Each could decide the outcome:
- **The gate tests the wrong quantity.** It departs from the probe it ports, and in the direction of false nulls (C1).
- **There is no untrained control.** The probe's own history shows the pipeline produces signal-like differences with
  no training (C2).
- **SIGNAL's wording claims model-internal mechanism,** and the readout, as the architecture computes it, sits close to
  the model's output (C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **Gate G gates on raw importance at 0.98. The v6 probe it ports gates on the drug-specific \|Δimp\| at 0.95.** `probe_moa_v6.py:200-205` computes both `rho_raw` and `rho_del`, and sets `gate_ok = rho_del < 0.95`. The "+1.0000" quoted in the packet is its *raw*, delta-free ranking on an untrained model. In v9 the activations `a[p,c]` are **identical across drugs by construction**, since the readout runs before the drug enters, so raw `imp` is dominated by an `a`-weighted component every drug shares. A trained checkpoint can therefore show raw agreement ≥ 0.98 while \|Δimp\|, the quantity the statistic ranks by, is drug-specific. G would then return NULL without the test ever running. The packet also demotes "the cross-drug Spearman of Δimp" to reported-not-read, which inverts the probe's own design. | Gate on the median cross-drug Spearman of \|Δimp\| (`rho_del`), as the v6 probe does, with its 0.95 threshold (or justify a change now), and report `rho_raw` beside it. Keep the degeneracy check, max \|imp\| > 1e-12, ahead of it, as the v6 probe does. |
| 2 | MAJOR | stats | **No untrained v9 control. The probe's history shows the pipeline produces signal-like differences with nothing learned.** On untrained v6, S = 0.218 against a label-permutation null of 0.229: diff **−0.011**, p 0.13 (CLAIMS 4.15). That's half the proposed −0.02 floor, from structure alone. v9 scores many more annotated compounds than v6 did, so the same structural diff gains power and can reach p < 0.05. Label permutation also doesn't break every structural association: L1000 doesn't assign compounds to cells at random, and the readout is cell-dependent through `a`. | Run the identical pipeline on seed-matched **untrained** v9 checkpoints (random init, the same rows, compounds and nulls). Pre-register: **the probe is valid only if the untrained control reads NULL**. Report `diff_trained − diff_untrained` for every seed. If the untrained control reads PARTIAL or SIGNAL, nothing about the trained models is read, and the confound is the finding. |
| 3 | MAJOR | overreach | **SIGNAL reads "v9 carries drug-specific pathway mechanism", but this readout sits close to the model's output.** In v9, node *p* = GELU(mean over its landmark members of `proj(h)`), computed before the drug, and it writes back **only to its own members** (`modules_v9.py:372-374`). With the `sq` objective, ∂O/∂Ŷ = 2Ŷ, and `a` is shared between the drug and mean-drug passes. So Δimp[p] is, to first order, a membership-weighted projection of *how the prediction changes with the drug* onto *p*'s landmark genes. A SIGNAL could then mean either of two things, neither of which is the pathway layer carrying mechanism: **(a)** the predicted drug-specific signature is enriched in target-pathway member genes, which is reproducible with no pathway layer at all; or **(b)** the measured data have that enrichment and the model reproduces it, including memorised signatures of **seen** compounds. The seen/unseen split is currently reported-not-read. | Pre-register two reference readouts, on the same rows, compounds and nulls. **Output projection:** rank nodes by \|`M_norm`·(Ŷ_d − Ŷ_mean)\|. **Data projection:** \|`M_norm`·(Δ_measured,d − Δ̄_cell)\|. Scope SIGNAL's wording by them: *"model-internal, readable by gradient × activation"* only if the gradient readout beats the output projection. Otherwise, *"v9's predicted signatures are enriched in target pathways"*, stating whether the measured ones are too. Pre-register that any structure-to-mechanism claim needs the **unseen-compound** stratum to show it. Seen compounds can't separate mechanism from memory (017). |
| 4 | MINOR | code-vs-intent | **The mean-drug baseline has to be one fixed input.** "Drug inputs replaced by their mean over the scored compounds" leaves the atom **mask** unspecified. If the baseline inherits each row's mask, it carries that drug's atom count, which changes attention normalisation, so the baseline itself varies with the drug. | Specify one baseline tensor set for every row: the same `u`, the same atom tokens, the same mask. Say how the mean over variable-length molecules is formed. Dose and time stay at the row's values (they act before the readout, identically in both passes). That's correct, and worth stating. |
| 5 | MINOR | provenance | **Scope of any claim.** r0–r2 are fold-0 models without drug self-attention. The model in the XPert comparison (§45, `v9_cc1_epi`) is a different model. A mechanism claim attaches to the r-series architecture and training, not automatically to the compared model. | State that scope in the readings. If the paper pairs the interpretability claim with the XPert comparison, run the probe on the §45 model too (reported, not read). |

## Answers to the asks

**Ask 1 — Δimp against a mean drug is a sound first-order estimand, once C1, C3 and C4 are fixed.** Nothing makes it
structurally zero: the drug moves the Jacobian of Ŷ with respect to the member genes' hidden states, through
cross-attention and the gene blocks. `a` being exactly shared is intended, since it isolates the drug's effect in the
gradient. The `sq` objective prevents sign cancellation, but it ties Δimp to response *magnitude*. Ranking within each
compound removes overall scale, but not the pattern below (ask 4).

**Ask 2 — the full-gene-set positive set is legitimate.** It's L1000's own reporter logic: perturbing a target moves the
pathway, and the pathway's landmark members report it. Two properties make Null 2 essential rather than optional:
- **Membership in the full set scales with pathway size.**
- **GO:BP terms nest,** so one target pulls in a chain of large generic parent terms.

Report how many of the 800 nodes matched a GMT term id (none should be dropped silently), and the distribution of
positive-set sizes. Keeping landmark-only as a secondary tier is right.

**Ask 3 — thresholds:**
- **Gate:** see C1.
- **Floor:** −0.02 is only twice the untrained structural diff, so tie it to C2's control rather than to 0.
- **All-three-seeds rule:** right. A single-seed signal in an attribution method is weak evidence.
- **PARTIAL:** its "all three with −0.02 < diff < 0" should also require p < 0.05. Otherwise it can fire on three noise
  draws below the null mean.

**Ask 4 — `use_aux` leaks nothing drug-specific, but it adds a weighting that one check needs to catch.** The aux target
is the row's measured per-pathway `mean|Δ|`, but the activations it trains are drug-blind. So they can only learn
per-cell pathway responsiveness, and ChEMBL targets never enter any loss. The weighting itself: `a` multiplies the
gradient, so pathways responsive in that cell dominate Δimp for every drug. Within-compound ranks and Null 1 absorb the
*generic* part. They don't absorb one interaction:
- **Strong perturbers** (proteasome, HDAC, CDK, HSP90 inhibitors) have targets in the most responsive pathways.
- **For those compounds** the readout concentrates on exactly those pathways.
- **Weak perturbers'** rankings are noisier.

That pairing beats label permutation with no mechanism in the model. The v6 probe's **target-responsiveness
stratification** (4.1b) is the check for this, and C3's data projection is the other. Promote the stratification from
reported-not-read to a scoping condition: SIGNAL should hold in the stratum where the targets are *not* among the
responsive genes.

**Ask 5:** C4 and C5. Keeping per-drug case studies out unless separately pre-registered is right.

## What I checked and found sound

- **The readout's structure** (`modules_v9.py:372-374`: mean over members of `proj(h)`, then GELU, then write-back
  through `M`) and its position before the perturb blocks (`model_v9.py`). This is what makes `a` drug-invariant, and
  C1 and C3 follow from it.
- **The ported probe's design** (`probe_moa_v6.py`): the degeneracy check, the gate on `rho_del`, label permutation as
  the primary reference, the size-matched null, and the target-responsiveness stratification. The v9 port should keep
  all five.
- **Checkpoint configs:** r0 has `drug_self_attn` unset (off), `use_aux` true, `d_pathway` 32, 4 perturb blocks, PPI
  on, epoch 11 (the 12th). sa0 is the same with self-attention on.

## What I could not assess, and why

- **How close gradient × activation actually comes to the output projection** in trained v9. The cross-gene mixing in
  the gene blocks could separate them, and C3's reference readout measures it.
- **The GMT match rate for the 800 nodes,** until the port runs.
