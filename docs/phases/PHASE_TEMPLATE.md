# <TITLE> — Template (agents/phases)

- **Motivation/Background**: Establish technical specification, requirements, and deliverables for pipeline phase PHASE_TEMPLATE.
- **Purpose**: Guide the execution and quality gates of phase PHASE_TEMPLATE.
- **Overview Pipeline**: Phase scope definition -> implementation guidelines -> verification gates -> status sign-off.
- **Detailed Plan**: §1 Phase Overview & Scope; §2 Technical Specification; §3 Deliverables & Artifacts; §4 Verification Protocol.
- **References**: `docs/OVERVIEW.md`, `docs/PURPOSE.md`, `agents/rules/`.
- **Created**: 2026-08-18T08:56:25+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

Copy this file into `agents/phases/` as `<PHASE>.md` for each phase/stage
(DATA_PREP.md, TRAINING_INFO.md, MODEL.md, EVAL.md, ...). Everything for a
phase stays in this single file.

---

## Header

- **Title:** <Short phase name, e.g. "Data Preparation">
- **Date created:** YYYY-MM-DD
- **Last updated:** YYYY-MM-DD
- **Description:** <One sentence: what this phase is.>
- **Status:** [To Do | In Progress | Done | On Hold | Canceled]

## Background

Why this phase exists and what problem it addresses.

## Goals / Purpose

- What "done" looks like, concretely
- What this phase explicitly does NOT try to solve

## Input / Output

- **Input:** <files, data sources, upstream artifacts, formats>
- **Output:** <files, artifacts, expected shape/schema>

## How to do it (general plan)

1. Step one
2. Step two
3. ...

## Pipeline

<Concrete sequence of scripts/commands.>

```
src/... → src/... → ... → data/processed/...
```

## Detailed plan / gotchas

- Specific parameters, thresholds, edge cases
- Known gotchas / things that broke before
- Code links: `src/.../file.py#Lxx` (relative link + anchor)
- External refs: <https://...>

## Links

- Progress tracking: 
- Related phases: [OVERVIEW.md](../OVERVIEW.md), [EVAL.md](../phases/EVAL.md)
