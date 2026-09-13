# Deviations from the frozen plan

Anything that departed from `predictions/h1_predictions.json` or the protocol in
`docs/pmkvq-h1-todo.md` is recorded here, dated, with the reason. Predictions are
frozen before the first run; this file is the audit trail for everything after.

## Scaffold

- **2026-09-13** — Repo scaffolded per the todo skeleton. Directory misalignment
  fixed: the initial tree carried empty `src/` and `figures/` directories; the
  todo places canonical modules at the repo root and uses `assets/` for rendered
  png/gif. `src/` and `figures/` removed, `assets/` created. No module code was
  lost (both directories were empty).
- **2026-09-13** — `scripts/_prompts.py` added (not in the skeleton) to hold a
  small in-repo default prompt set so the arms run without a dataset dependency.
  The default prompts are placeholders; swap in the paper's long-CoT eval set
  before reporting numbers and note the swap here.
- **2026-09-13** — Canonical modules moved from the repo root into a `pmkvq/`
  package, departing from the todo skeleton's "modules at repo root" convention.
  Imports became `from pmkvq import <module>`; entrypoints already put the repo
  root on `sys.path`, so `show_source` and the mise tasks are unaffected. The
  six modules and their behaviour are unchanged.

## Runs

_None yet. No checkpoint has been executed; `results/` and `assets/` are empty._
