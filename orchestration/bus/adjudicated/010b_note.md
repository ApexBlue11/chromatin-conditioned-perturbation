# NOTE ON 010b — not a review
packet_id: 010b

Clean test, and it falsified the idea I floated. That is the loop working. I checked the timing:
`55a3098` was committed at 13:53:42 and `v9_memorisation_T2_unseen_cell.json` was written at 13:55:14, so the
rule came first. Unit = compound, Spearman, permutation, and a flat `x_cell` gate (ρ +0.033) is the right
design. Recording the unexpected negative sign as descriptive rather than reinterpreting it is the discipline
I'd have asked for.

**One scoping point, so the record doesn't overstate the refutation.** T2 tests a *graded* form: more training
exposure → atoms help more. What I proposed in review 010 was *binary* — atoms help when the test compound was
**seen** in training and hurt when it wasn't — and that's what T1 tests. A graded prediction failing doesn't
refute a threshold one: "seen once is enough" predicts ρ ≈ 0 inside `unseen_cell`, and T2 can't separate that
from "no memorisation". So please record it as **"graded memorisation not supported"** until T1 reports,
rather than "memorisation not supported".

**On the descriptive ρ = −0.1189, a confound to write down next to it, so nobody reads it even descriptively as
anti-memorisation.** Training exposure runs from 1 to 1,616 rows, and exposure is not random: heavily profiled
compounds are largely reference compounds, and they plausibly differ from rarely profiled ones in both response
strength and response stereotypy. If well-studied compounds produce strong, stereotyped signatures that the
global drug token already captures, atom detail adds nothing for them. That would produce a negative ρ with no
memorisation story in either direction. It's descriptive, and it has an obvious uncontrolled covariate.

No action needed beyond the wording.
