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

## 5. Safety gates — non-negotiable

The loop is **closed for analysis, open for consequence.** Autonomous iteration is permitted only for
reading, measuring, reviewing and writing to the repo. The following always require explicit human
approval, every time, and approval never generalises to the next occurrence:

1. Spending GPU quota (Kaggle or otherwise)
2. `git push`, PR creation, or anything leaving the machine
3. Deleting or overwriting data
4. Anything touching credentials
5. Publishing, sending, or posting

**Kill switch.** A file named `orchestration/STOP` halts both loops at the next wake. Both pollers check
it first. Delete it to resume.

**Iteration cap.** `orchestration/bus/state.json` holds `iteration` and `max_iterations` (default 12).
The PI refuses to open a new iteration past the cap without a human saying so.

**Local-CPU thermal rule.** This is a laptop. No multi-hour local compute. CPU-bound work goes to free
Kaggle CPU kernels first; local CPU only for short interactive jobs. Never the local GPU for training.

---

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
