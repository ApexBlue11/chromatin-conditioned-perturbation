# Briefing workers — tactics, with the evidence that earned them

Not generic prompt engineering. These are the specific moves that changed outcomes in this project's
first delegated batch, plus the ones that failed.

---

## The finding that should shape every brief you write

Two workers, **same model family**, same session, same repo:

| | brief framing | verifiable claims | survived PI verification |
|---|---|---|---|
| **W2** | "extract these mechanisms; does the literature address my hypothesis?" | 5 | **4** — the failure was its *conclusion*, which said the PI's hypothesis was novel |
| **W3** | "**your task is to destroy this claim**" | 7 | **7** — including the decisive one, and it correctly refused to inflate a near-miss into a kill |

> **An agent asked to check a hypothesis tends to confirm it. An agent asked to destroy one tends to be
> accurate.** Invert the incentive wherever the answer matters.

The danger is not random hallucination. It is *agreeable* hallucination — which arrives exactly where you
most want the answer to be yes, and reads as confident and well-sourced.

---

## The eight moves

### 1. State the stakes and the cost of being wrong
> *"This project has been burned by comparing numbers that were not comparable — once by citing a
> case-study figure as a benchmark result for six weeks. A wrong number here propagates into a manuscript."*

Grounds the model in a real consequence rather than an abstract instruction to be careful.

### 2. Make `UNKNOWN` the *preferred* answer, explicitly
> *"'UNKNOWN' is a correct and valued answer. Accuracy beats completeness."*
> *"If a number exists only inside a figure panel, write `UNKNOWN — figure panel only`. Never estimate."*

Without this, models fill gaps with plausible text. With it, W2 correctly returned `UNKNOWN` for TxPert's
per-graph ablation numbers rather than inventing them — and it was right; they are figure-only.

### 3. Demand a verbatim quote per claim
> *"A 'killer' REQUIRES a verbatim quote showing the input/output structure. No quote, no killer — put it
> in Partial instead."*

This is the single highest-leverage rule. It converts "the model asserts X" into something you can check
in one fetch, and it suppresses confident paraphrase of things the source never said.

### 4. Invert the incentive on anything that matters
> *"Your task is to destroy that claim if it can be destroyed. You are not helping me defend it. Finding
> one paper that kills it is the single most valuable outcome, and far more useful than a reassuring
> 'no prior work found'."*

See the table above. This is why W3 outperformed W2.

### 5. Supply the known near-misses up front
> *"Known near-misses — I have already checked these, do not re-report them as new: [4 papers, each with
> why it fails]. Your job is to find what I have NOT found."*

Stops the worker spending its budget rediscovering your existing knowledge, and it gives a worked example
of the standard of reasoning you expect.

### 6. Name both error directions
> *"A paper that fails any of 1–3 does NOT kill the claim — say so rather than reporting a near-miss as a
> hit. **Equally: do not dismiss a genuine hit because it is in an obscure venue or uses different words.**
> Both error directions are costly."*

One-sided warnings produce one-sided bias.

### 7. Require a search log and a "could not determine" section
> *"`## Search log` — every query you ran and roughly how many results you inspected. This lets me judge
> how hard you actually looked; a short log means a weak search."*

Makes effort observable. You cannot audit a confident summary; you can audit 48 logged queries.

### 8. Decompose the criteria and make them checkable
Don't ask "is this novel?". Give numbered criteria and require the worker to state **which one** each
candidate fails. W3's output was auditable precisely because every entry named its failing criterion.

---

## Structural requirements for every brief

- **Scope INCLUDE and scope EXCLUDE**, explicitly. W1's brief excluded genetic/single-cell/IC50 work by
  name, which is what kept the competitor list clean.
- **A `BORDERLINE` bucket**, so ambiguity gets reported rather than silently resolved.
- **A fixed output schema** — a table with named fields beats prose, because missing fields become visible.
- **One deliverable file path**, plus *"Do not write any other files. Do not modify any existing file."*
  Then verify with `git status` after the run. (Done for the first batch: no tracked file was touched.)
- **Mandatory closing sections**: `## What I could NOT determine` and a confidence rating per item.
- **Say which version of a source was read** — preprint vs published differ, and a claim sourced from a
  preprint must be labelled `REPORTED-NOT-VERIFIED` until checked against the published version.

---

## Operational notes for `agy`

- Buffers stdout until the turn ends → **always background it and poll the log file**.
- Needs `--dangerously-skip-permissions` in headless mode or it blocks on tool approval.
- `--mode plan` is read-only; good for investigation, but it cannot then write its report.
- `--json-schema` enforces structured output — use it when you want fields, not prose.
- It can spawn its own subagents (`define_subagent` / `invoke_subagent`), so a wide task can be given to
  one worker rather than fanned out manually.
- **Wall time scales with sources, not difficulty.** A 3-paper deep read took ~6 min; an 8-source
  landscape with PDFs and supplementary was still running at 40+.

---

## The anti-pattern to watch for in returned work

Read the worker's **conclusion section last and hardest**. Per-item findings with quotes are usually
sound. The synthesis is where the model reaches beyond its evidence — and it reaches in the direction the
brief implied you wanted.

W2's per-paper mechanism extraction: verified accurate. W2's closing paragraph: overreached from three
papers to "the current literature" and told the PI his hypothesis was novel. Same worker, same run.
