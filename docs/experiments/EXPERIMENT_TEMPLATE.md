# <TITLE> — Template (agents/experiments)

- **Motivation/Background**: Document experiment hypothesis, parameters, evaluation methodology, and results for EXPERIMENT_TEMPLATE.
- **Purpose**: Provide rigorous experimental documentation and tracking for EXPERIMENT_TEMPLATE.
- **Overview Pipeline**: Hypothesis formulation -> dataset split preparation -> model training/eval -> metrics analysis.
- **Detailed Plan**: §1 Experiment Objective & Hypotheses; §2 Configuration & Hyperparameters; §3 Execution Protocol; §4 Results & Findings; §5 Next Actions.
- **References**: `configs/`, `src/training/`, `src/eval/`, `experiments/results/`.
- **Created**: 2026-08-18T08:56:25+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

Copy this file into `agents/experiments/` as `<EXP_OR_NAME>.md` for each
experiment. Register it in `agents/experiments/README.md`.

---

## Header

- **Title:** <Short experiment name>
- **Date created:** YYYY-MM-DD
- **Last updated:** YYYY-MM-DD
- **Description:** <One sentence: what this experiment tests.>
- **Status:** [To Do | In Progress | Done | On Hold | Canceled]
- **Experiment ID:** <e.g. EXP-07, or a short slug>

## Objective

<What hypothesis is being tested; what success looks like (with a number).>

## Single variable changed / held constant

- **Changed:** <the one thing under test>
- **Held constant:** <everything else — data, loss, scheduler, seed...>

## Setup

- **Data / split:** <source, train/val/test sizes>
- **Models / base:** <architectures, checkpoints>
- **Hyperparameters:** <lr, epochs, batch, loss, scheduler, seed>

## Results

| Metric | Before | After |
|---|---|---|
| Test accuracy | <x>% | <y>% |
| <other metric> | ... | ... |

## Analysis / interpretation

<What the numbers mean; whether the hypothesis holds; caveats.>

## Reproduce

<Commands or notebook path to rerun. Note if runtime > 5 min.>

## Links

- Notebook: <relative path>
- Status: 
