# Orchestration — PI / worker / adversary architecture

**Portable.** Nothing here is LINCS-specific except the examples. Copy `orchestration/` into any project.

---

## 0. The one rule

> **The PI is the only component permitted to believe anything.**

Workers produce *claims*. The adversary produces *challenges*. Neither produces *findings*. A claim
becomes a finding only when the PI has verified it against a primary source, and the verification is
recorded next to the claim. Everything else is an unverified assertion wearing a confident tone.

This is not paranoia. In this project's first delegated batch, two of three workers were accurate on
every checkable claim — and the third's single error was a **conclusion that flattered the requester's
hypothesis**. That is the failure mode: not random hallucination, but *agreeable* hallucination, which is
invisible precisely when you most want it to be true.

---

## 1. Roles

| role | who | sees | must never |
|---|---|---|---|
| **PI** | Opus 5 (this session) | everything | delegate the judgement of whether a number is real |
| **Worker** | `agy`, `copilot`, `gh agent-task` | its brief + the repo | be trusted without verification |
| **Adversary** | a separate Claude session | **objective, function, results, code — NOT the PI's reasoning** | be told what the PI concluded |
| **Human** | you | everything | be bypassed on a consequential action |

### Why the adversary is blinded
An adversary shown the PI's reasoning critiques the *reasoning*. An adversary shown only the objective,
the code, and the raw numbers re-derives the conclusion independently — and *disagreement then carries
information*. Told the answer first, it anchors, and you get agreement that means nothing.

This is the same reason a permutation null is computed rather than assumed.

---

## 2. The loop

```
        SCIENTIFIC OBJECTIVE
                 │
                 ▼
         ┌───► PI (plans, verifies, decides)
         │       │
         │       ├── dispatch ──► WORKERS (agy / copilot / gh)  ──┐
         │       │                                               │
         │       ├── compute ───► Kaggle GPU / Kaggle CPU / local CPU
         │       │                                               │
         │       ▼                                               │
         │   EVIDENCE PACKET  ◄──────────────────────────────────┘
         │   (objective, function, results, code — NO reasoning)
         │       │
         │       ▼
         │   bus/to_critic/ ──► ADVERSARY (blinded, independent)
         │                            │
         │                            ▼
         │                     bus/to_pi/  (challenges, severity-ranked)
         │       ┌────────────────────┘
         │       ▼
         │   PI adjudicates: each challenge is UPHELD / REJECTED / NEEDS-MEASUREMENT
         │       │
         │       ├── UPHELD ──────► correct the record, re-run, re-issue packet
         │       ├── REJECTED ────► record why, with evidence
         │       └── NEEDS-MEAS ──► new experiment → back to the top
         │                                   │
         └───────────────────────────────────┘
                     │
                     ▼
            KNOWLEDGE BASE (RESULTS.md / CLAIMS.md / AGENT_REGISTRY.json)
```

**Stopping condition.** A loop iteration ends when every challenge is adjudicated *and* no challenge is
`NEEDS-MEASUREMENT`. Otherwise it spawns the next experiment. Unbounded iteration is prevented by §5.

---

## 3. Evidence packet — the handoff format

Written by the PI to `bus/to_critic/NNN_<slug>.md`. **Deliberately excludes the PI's interpretation.**

```markdown
# PACKET NNN — <slug>
packet_id: NNN
created: <ISO date>
repo_commit: <sha>

## OBJECTIVE
What scientific question this bears on. One paragraph. No claims about the answer.

## FUNCTION
What the code does, mechanically. Files + line ranges. What it consumes, what it emits.
State the intended semantics so the adversary can check code against intent.

## RESULTS
Raw numbers, as produced. Include n, error bars, the null/baseline, and the exact metric definition
(per-row then mean? median? which stratum?). Point at the artefact files.

## CODE
Paths and commit sha. Diff if small. Never a summary of what the code "basically does".

## WHAT WAS CONTROLLED
Splits, seeds, ablation targets, what was held fixed between arms.

## PRIOR RETRACTIONS IN SCOPE
Errors this project has already made in this area, so the adversary does not spend effort re-finding them.

## ASKS
Specific questions, if any. The adversary is free to ignore these and raise anything else.
```

**Forbidden in a packet:** "we conclude", "this shows", "as expected", "therefore", confidence language,
and any mention of what the PI thinks the result means.

---

## 4. Adversary output format

Written to `bus/to_pi/NNN_review.md`:

```markdown
# REVIEW OF PACKET NNN
verdict: SOUND | SOUND-WITH-CAVEATS | NOT-SUPPORTED | CANNOT-ASSESS

## Challenges
| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING / MAJOR / MINOR | confound / wrong-quantity / missing-null / stats / leakage / code-vs-intent / overreach | ... | the specific measurement |

## What I checked and found sound
(so the PI knows the coverage, not just the complaints)

## What I could not assess, and why
```

**Severity contract.** `BLOCKING` = the stated result does not follow from the evidence. `MAJOR` = a
plausible alternative explanation is untested. `MINOR` = presentation or robustness. Inflating severity
destroys the signal; the PI tracks calibration in the registry.

---

## 5. Safety gates — as set by the principal, 2026-09-20

The loop runs **autonomously** on this project. The principal has explicitly delegated GPU spend,
deletion and pushing. Recorded here so the delegation is auditable and so a future session does not
quietly re-tighten or re-loosen it.

### Delegated — proceed without asking
| action | why it is safe here |
|---|---|
| **Kaggle GPU spend** | the principal's own quota, resets weekly, no external effect |
| **`git push`** to the project's own private repo | it is the backup mechanism; withholding it is the larger risk |
| **Deleting files** | see the engineering practice below |

### Still gated — NOT delegated, ask every time
| action | why |
|---|---|
| **Credentials** — entering, storing, or moving any secret | never delegable |
| **Publishing or sending outward** — posting, emailing, submitting, or putting project content on any external service | irreversible in a way a git push is not; content can be cached or indexed even if retracted. Particularly relevant here, since this project holds findings about a published paper |
| **Anything targeting a repo or account that is not the principal's** | out of scope |

### Engineering practice on deletion (not a permission gate)
Deletion is delegated. Carelessness is not. Before removing anything that is not scratch:
1. confirm the thing is **derived or re-obtainable**, and say from where;
2. confirm the **downstream consumers** are satisfied by what remains;
3. leave a **manifest** recording what was removed and how to get it back.

This is the pattern used when 27 GB of raw Level-5 GCTX was removed: consumers traced, derived arrays
verified intact, `DELETED_MANIFEST.txt` written, GEO accessions recorded. It costs a minute and it is why
that deletion is reversible.

### Periodic human checkpoint
`bus/state.json` carries `responses_since_checkpoint` and `checkpoint_every` (**30**). The PI increments
the counter each turn and, on reaching the threshold, **stops and reports** rather than opening new work:
what ran, what it cost, what changed in the claims ledger, what it proposes next. The principal resumes
or redirects. This replaces per-action approval with periodic review.

### Kill switch
`orchestration/STOP` halts both loops at the next wake. Both pollers check it before anything else.
Delete the file to resume.

### Iteration cap
`bus/state.json` caps `iteration` at `max_iterations` (12). At the cap the PI stops and reports rather
than opening iteration 13.

### Standing compute constraints (unchanged — these are physical, not policy)
- **Never the local GPU for training.** 4 GB; inference sweeps only.
- **No multi-hour local CPU.** This is a laptop; thermal limits are real. Free Kaggle CPU first.
- **Never P100** on Kaggle (no sm_60 kernels). T4 x2 only. Max 2 concurrent GPU, 5 concurrent CPU.
- **Never hard-cancel a long Kaggle run** — `/kaggle/working` is discarded.
- **Gate GPU spend on a free-CPU result where one exists.** Not a permission gate; a cost discipline.
  A ridge that costs nothing has repeatedly predicted what the GPU run would show.

## 6. Worker routing

Full detail and live reliability data: `AGENT_REGISTRY.json`. Briefing tactics: `PROMPTING.md`.

Routing heuristic, in order:
1. **Can it be settled by reading a file in this repo?** → PI does it. Never delegate a question whose
   answer is a `grep` away; the delegation costs more than the answer.
2. **Is the output verifiable against a primary source?** → delegate freely.
3. **Is the output a judgement about whether our own result is real?** → PI only. Never delegate.
4. **Is it many-source, wide, shallow?** → `agy` + Gemini 3.8 Flash (high).
5. **Is it deep, conceptual, single-source?** → `agy` + Gemini 3.1 Pro (high), or Claude Opus 4.6.
6. **Is it a PR-shaped code change?** → `gh agent-task`.
7. **Does it need a second opinion from a different vendor's model?** → route the *same* brief to two
   surfaces and compare. Disagreement is signal.

---

## 6b. Code delegation — the PI writes the CONTRACT, a worker writes the BODY

**Standing correction, 2026-09-20.** The PI wrote `atom_ablation_ci.py` (~200 lines) by hand while holding
a precise spec and a cheap mechanical check. That is the exact shape of a task that should have been
delegated, and writing it in-house was a waste of the expensive agent. The principal was right to call it.

### The division that works
| the PI writes | a worker writes |
|---|---|
| the **spec**: what is computed, on what rows, with which estimator | the implementation |
| the **guards**: every assertion whose failure would let a wrong number through | the plumbing around them |
| the **verification contract**: the exact command to run and the exact output that counts as pass | the code that satisfies it |
| the **statistical choice**: estimand, estimator, which n is the denominator | — |

The rule: **delegate the body, never the contract.** Being wrong about the estimator is expensive and
silent; being wrong about a loop is cheap and loud.

### What makes a code task safe to delegate
All four must hold:
1. The spec is precise enough that two competent implementers would produce the same behaviour.
2. **Verification is mechanical** — a test that prints a number the PI can check, not a judgement call.
3. Failure is **loud**: a broken implementation fails a test rather than shifting a result by 0.004.
4. The blast radius is bounded — named files, explicit "do not touch" list, no git, no GPU.

If verification would require the PI to re-derive the answer anyway, delegating costs more than it saves.
**Never delegate**: whether a number is real, the estimand, adjudicating a review, or a spend decision.

### The brief template that has worked
Background (only what is needed) → numbered requirements → **the tests to add, specified as behaviours
not existence** → the exact verification commands and what their output must say → explicit constraints
(files touched, no git, no GPU, style) → a report file with a mandatory `## What I was unsure about`.

Include the project's relevant scar tissue. W4's brief carried the quantiser story (a guard that asserted
`fit()` had been *called* while the bins were NaN) so the worker would understand *why* behaviour-testing
is demanded rather than treating it as a style preference.

---

## 6c. Compute routing — cheapest sufficient tier, always

Checked in this order. Do not skip a tier without saying why.

| tier | cost | use for | constraint |
|---|---|---|---|
| **0. already on disk** | free | **CHECK FIRST, EVERY TIME** | This project has paid three times for something already in hand: §46.1 supplementary tables, §54.1 saved predictions, §56.2 existing seed checkpoints. Enumerate artefacts — **including gitignored dirs** — before anything else. |
| **1. closed form** | free | ridge, bootstrap, re-scoring saved predictions | A closed-form ridge has repeatedly predicted what the GPU run would show, with no seed noise. |
| **2. free Kaggle CPU** | free | any CPU-bound job over a few minutes | **5 concurrent sessions.** This is the default for real CPU work, not the local machine. |
| **3. local CPU** | free, thermally limited | short interactive checks only | It is a laptop. A job that exceeds ~2 minutes locally belongs on tier 2. |
| **4. local GPU** | free, 4 GB | **inference sweeps only** | Never training. Batch 8 ceiling, so its absolute numbers are not comparable to batch-48 Kaggle runs. |
| **5. Kaggle T4 x2** | **30 h/week** | training only | Never P100. Max 2 concurrent. Never hard-cancel (working dir is discarded). **Cost estimate goes through the adversary BEFORE the spend** — review 003 caught a design that was not decisive and a premise that was false, before any hours were committed. |

**Cost estimates are measured, not scaled.** §47.6's 5.6 h projection scaled a rate measured with the new
module *off*, so it was a floor of unknown tightness. And a "runs to power it" figure is the ~51 % point
unless it used `(z_alpha + z_beta)`; the 80 % number is roughly 2x larger [§53.1].

## 6d. Toolchain hazards on THIS machine — each of these has cost a round trip

These are not bugs to fix, they are properties of the environment. Every one was discovered by a silent
corruption, not by an error message pointing at the cause.

### BOM on every file PowerShell writes (hit 4x)
`Set-Content` and `Out-File -Encoding utf8` on Windows PowerShell 5.1 prepend `EF BB BF`. That breaks
`json.load` and the Kaggle CLI, which fails with the useless `Expecting value: line 1 column 1 (char 0)`.
`-Encoding utf8NoBOM` **does not exist on 5.1** — it is PowerShell 7+ and errors out here.

> **Rule.** Any file a program will parse is written with Python `io.open(path, "w", encoding="utf-8")`.
> Never PowerShell redirection, never `Set-Content`. Verify with `head -c 4 file | xxd` if in doubt:
> `7b0a` is good, `efbb bf7b` is the bug. `state.json` carries this warning in a field of its own.

### Backslashes collapse inside Bash-tool heredocs (hit 1x, 2026-09-21)
A heredoc with a single-quoted delimiter is supposed to pass its body through literally. It does not pass
doubled backslashes through: a patch script written with a doubled backslash arrives with a single one,
so Python then reads it as an escape and the anchor silently fails to match the target file. The failure
mode is an `AssertionError` on a string that looks identical to the file when printed — only `repr()`
shows the difference.

> **Rule.** A patch script's anchor must contain **no backslash at all**. If the target line has one
> (an escaped quote, a regex, a Windows path), match the line by a backslash-free prefix and replace it
> whole, or pipe the script to `python -` instead of writing it with a heredoc. When an anchor assertion
> fails on a string that looks right, print `repr()` of both sides before editing anything.

### `git add -A` from the repo root stages the worktrees (hit 1x, 2026-09-21)
`git worktree` put both sibling checkouts under `LINCS/`, so a blanket add staged them as embedded git
repositories. Git warns, but it warns *after* staging and the commit still succeeds.

> **Rule.** `LINCS/` is in `.gitignore`. Read the output of `git add`, not just its exit code.

### A smoke test overwrote a finished result (hit 1x, 2026-09-21)
`alpha_sweep.py` keyed its output path on `--key` but not `--n_eval`, so a 96-row smoke test run to check
a code change **overwrote a completed 1500-row result** that had taken forty minutes. It was recoverable
only because the full run had already been committed. `n_eval` was recorded inside the JSON body — which
documents the file but does nothing to stop the overwrite.

> **Rule.** The output path carries **every** argument that changes the numbers, not just the one that
> bit last time. And commit a finished result **before** touching the script that produced it — the commit
> is the only backup, and it cost nothing here because it was already policy.

### `git add -A` while a delegated worker is editing the tree (hit 1x, 2026-09-21)
I committed and **pushed** a worker's half-finished edits to `modules_v9.py` and `model_v9.py` — two files
every v9 result depends on — in a commit whose message said the task had been *delegated*, before a single
test had run. The code happened to be structurally sound. That was luck, not process.

> **Rule.** While a worker is running, stage **explicit paths** you have verified, never `-A`. If a commit
> must go out mid-delegation, `git add` the specific files you wrote and check `git status` for the
> worker's. And the commit that first carries a worker's code must be the commit that reports its test
> output — those two facts belong in the same message or the history lies about what was verified.

### An `agy` exit code of 0 means the process ended, not that the task succeeded
A worker that backgrounds its own tests, goes idle, and gets its tasks killed on session exit returns 0
having changed nothing. See §6b: **verification is running the tests yourself**, never reading the report.

## 6e. Before asking, or claiming a finding: grep the repo (method rule 20)

Three packets have asked the adversary for something this project already had:

| packet | asked for | already existed |
|---|---|---|
| 002 | a chromatin re-score needing GPU | the predictions were on disk in `external/v9_mdmt_preds/` |
| 005 | a paired-mean interval | `interaction_2x2.py` already emitted it |
| 007 | a decision on how to replace `flash_attn` | `model/v9/_shims/flash_attn/` had made it, tested it, and validated it against a published number |

The 007 case was worse than a wasted round trip: the mask omission was written up in `RESULTS` as a
**new finding** when a docstring in our own tree had stated it verbatim two days earlier. That is a
provenance error in our own record, not just a duplicated effort.

> **Rule.** Before writing "we should build X" or "I have found Y":
> `grep -ril "<concept>" model/ research/ orchestration/ network/` and read `IDEAS.md`.
> The adversary must not be the mechanism by which we discover what we have already done.

## 7. Provenance contract

Every number that enters `RESULTS.md` or `CLAIMS.md` carries:
- **who produced it** (PI / worker id / which model)
- **verified?** — `VERIFIED` (PI checked primary source) / `REPORTED-NOT-VERIFIED` / `UNVERIFIABLE`
- **the artefact path** it can be recomputed from

`REPORTED-NOT-VERIFIED` is a legitimate state. Citing one as though it were verified is the error.

---

## 8. Meta-learning (Level 10)

After each delegation the PI appends an observation to `AGENT_REGISTRY.json → observations[]`:
model, task class, verifiable claims made, how many survived verification, failure mode if any, wall time.

Routing policy updates from two sources, and **observed reliability outranks published benchmarks**:
1. **Observed** — this registry. Small n, but it is *our* task distribution.
2. **Published** — public benchmarks, refreshed periodically, with the date recorded.

A published score is a prior. An observation is evidence.
