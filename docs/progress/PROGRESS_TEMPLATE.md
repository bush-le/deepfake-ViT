# <TITLE> — Template (agents/progress)

- **Motivation/Background**: Provide real-time progress tracking, checklist status, and milestone completion for PROGRESS_TEMPLATE.
- **Purpose**: Maintain an accurate audit trail of completed tasks and active blockers for PROGRESS_TEMPLATE.
- **Overview Pipeline**: Milestone tracking -> task checklist review -> verification status update.
- **Detailed Plan**: §1 Current Milestone Status; §2 Completed Deliverables; §3 Active Blockers; §4 Next Priorities.
- **References**: `docs/OVERVIEW.md`, `docs/phases/`.
- **Created**: 2026-08-18T08:56:25+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

Copy this file into `agents/progress/` as `<PHASE>_STATUS.md` for each phase.
Update after every meaningful change. This is the fast thing to read without
re-reading the whole phase doc.

---

## Header

- **Title:** <Short phase name, e.g. "Data Preparation">
- **Date created:** YYYY-MM-DD
- **Last updated:** YYYY-MM-DD
- **Description:** <One sentence: what this status doc tracks.>
- **Status:** [To Do | In Progress | Done | On Hold | Canceled]
- **Phase doc:** <link to ../phases/<PHASE>.md>

## Log

- YYYY-MM-DD: <what changed>
- YYYY-MM-DD: <blocker encountered / decision made>

## Blockers (if any)

- <what's blocking, and what's needed to unblock>

## Decisions

- <decision> — <rationale>

## Next step

- <single concrete next action, not a wishlist>

## Links

- Phase doc: 
- Experiment record: 
