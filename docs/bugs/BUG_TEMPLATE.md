# <TITLE> — Template (agents/bugs)

- **Motivation/Background**: Document bug diagnosis, root cause analysis, and regression prevention for BUG_TEMPLATE.
- **Purpose**: Track lifecycle and remediation evidence for issue BUG_TEMPLATE.
- **Overview Pipeline**: Bug discovery -> root cause analysis -> patch verification -> regression testing.
- **Detailed Plan**: §1 Bug Description & Symptoms; §2 Root Cause Diagnosis; §3 Remediation & Code Changes; §4 Verification & Prevention.
- **References**: `src/`, `tests/`, `agents/rules/CODEBASE_AUDIT.md`.
- **Created**: 2026-08-18T08:56:25+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

Copy this file into `agents/bugs/` as `BUG_<NN>_<SHORT_SNAKE_NAME>.md` for each
new bug. Register it in `agents/bugs/README.md`.

---

## Header

- **Title:** <Short bug title, e.g. "DataLoader BrokenPipeError in Python 3.14">
- **Bug ID:** `BUG-<NN>`
- **Date identified:** YYYY-MM-DD
- **Description:** <One sentence: what fails and where.>
- **Status:** [Open | In Progress | Resolved | Won't Fix]
- **Severity:** [Low | Medium | High | Critical]
- **Category:** <e.g. Runtime / Data / Training / Evaluation / Environment>
- **Target component:** <file(s)/module(s) affected>

## 1. Symptom & Error Traceback

<What happened. Paste the full traceback / error message.>

## 2. Root Cause Analysis

<Why it happened. Numbered list of contributing causes.>

## 3. Solution & Remediation

<What was changed and why. Include code snippets / file links.>

## 4. Verification Evidence

<How it was confirmed fixed — command, expected output.>

## 5. Related Links

- Master bug index: [README.md](README.md)
- Related code: <relative links with anchors>
