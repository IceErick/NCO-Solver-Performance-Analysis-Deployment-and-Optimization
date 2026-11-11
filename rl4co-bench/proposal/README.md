# `proposal/` — Research Proposal (LaTeX)

## Files

| File | What it is |
|---|---|
| `proposal.tex` | **The FYP submission.** 5 pages = cover + 4 content pages. Compiles clean. |
| `proposal-full.tex` | The long-form research plan (9 pp, two-column). Reference material for the project itself — **not** a submission document. It exceeds the 2,000-word FYP limit roughly threefold. |
| `references.bib` | Bibliography, shared by both. Some entries need verification — see below. |

`proposal-full.tex` holds the detail that the FYP format has no room for: the
Layer A–D methodology, the `E` normalised-efficiency metric and its calibration
anchor, the experimental factor table, and the risk table. Keep it — it is what
you will actually work from, and it expands back into the dissertation.

## Compiling

TeX Live 2025 is installed at `/Library/TeX/texbin` but is **not on the default
PATH**. Either add it to your shell profile, or use the full path:

```bash
export PATH="/Library/TeX/texbin:$PATH"
cd rl4co-bench/proposal
pdflatex proposal && bibtex proposal && pdflatex proposal && pdflatex proposal
```

`latexmk -pdf proposal.tex` also works. `latexmk -c` cleans the auxiliary files.

## FYP compliance

| Requirement | Status |
|---|---|
| Cover page: title, name, ID, email, supervisor, programme | Done — page 1 |
| Background and motivation (~1 page) | 1 page — four paragraphs; names the sub-field in the first paragraph and closes with the author's own reading |
| Aim(s) and objectives (~0.5 page) | 1 sentence aim + 5 objectives |
| Project plan (~1 page), with Gantt chart | 8 tasks + Gantt, with exam/winter-break shading |
| Bibliography (~0.5 page) | 16 entries, `abbrvnat` |
| ≤ 2,000 words excluding cover | **1,357** words before the objectives were cut from 7 to 5 and the milestones paragraph removed; comfortably lower now |

The Gantt chart (`proposal.tex`, search for `tikzpicture`) shades the examination
period (mid December to early January, darker) and the winter break that follows
it (to late February, lighter). **Check these against your actual calendar** — the
bands are drawn by coordinate, so `(5,0) rectangle (6.5,8)` is exams and
`(6.5,0) rectangle (9.4,8)` is the winter break. The axis runs 0–12 in six
two-unit months from October, so x=5 is mid December and x=9.4 is late February.
Move a band by changing its two x values. The task bars are the `\foreach
\k/\xa/\xb` list; each row is an `x0/x1` pair.

## Before you submit

**1. Complete the `aet2026` bibliography entry.** It is currently stubbed as
`{Anonymous}` and renders literally as "Anonymous" in the PDF, with the note
"Author list to be completed". Objective O7 rests on this paper, and References
is 10% of the mark with "complete" as the criterion. It cannot ship like this.

**2. Verify these bibliography entries.** Details were reconstructed from partial
information and, for the 2025–2026 entries, could not be independently confirmed.
A wrong arXiv ID is worse than a missing citation, because it looks verified.

| Entry | What to check |
|---|---|
| `aet2026` | **Authors completely unknown.** Also confirm the arXiv ID `2605.14624`. |
| `ye2024glop` | arXiv ID `2312.08224` and the author list recalled from memory, not confirmed. |
| `wang2025strudico` | Venue and year supplied by you; not independently confirmed. |
| `luo2024icam` | Author list from memory — confirm, especially the ordering. |
| `bai2023analytics` | Author list recalled from memory; confirm against the published article. |
| `duan2023htsp`, `ma2021dact`, `hottung2022eas`, `joshi2019gnn`, `wouda2024pyvrp` | Venue and author list recalled from memory. |

**One correction already applied.** `li2024fastt2t` is cited as **NeurIPS 2024**,
not NeurIPS 2025. The published proceedings place it at NeurIPS 2024.

## Editing notes

- **Paragraph style:** `\parindent` is 0 and `\parskip` is one `\baselineskip` —
  paragraphs are set flush left with a blank line between them. Do not
  reintroduce `\indent` or per-paragraph `\vspace`.
- **Single platform, deliberately.** The cross-backend design (NVIDIA vs Apple
  M2) was dropped on 2026-11-03: the study is now one NVIDIA GPU workstation,
  with hardware held fixed so the measured quality–latency trade-off is a
  property of the solver, not the machine. Generalisation is tested across
  **solver families** (constructive, heatmap-based, improvement) plus TSP→CVRP.
  Do not reintroduce a second machine or the "second execution backend" phrasing
  in O5/T7 — it would contradict the scope statement in §1.
- **Hardware is deliberately generic.** The document says "an NVIDIA GPU
  workstation", not a specific card, because the model is not yet settled. Do
  not put a part number back in; update the generic wording only if the setup
  changes in kind.
- The section is titled **Project Plan**, not "Work Plan" (the user's wording),
  and it carries no milestones — the Gantt marks tasks and the two
  low-availability bands only. Objectives are O1–O5; tasks map onto them, with
  T3/T4 → O3 and T6/T7 → O5.
- Single column, `margin=2.5cm`. Do not switch back to two columns: the spec's
  "2,000 words ≈ 4 pages" is a single-column estimate, and two columns would make
  the document look short against the stated page budget.
- Section headings are `\section{}` (numbered 1–3); References is generated
  unnumbered by `\bibliography`.
- Captions go **below** figures: `\end{tikzpicture}` then `\caption{}`.
- In TikZ `\foreach` lists, a control sequence name cannot contain a digit —
  `\x0` parses as `\x` followed by a literal `0` and breaks the loop. Use `\xa`,
  `\xb`. Likewise `\ ` (control space) inside a `\foreach` item breaks the
  parser; use `~` instead.
