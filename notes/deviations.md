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

- **2026-09-13** — Constrained run configuration (`mise run h1-lite`) adopted for
  the reference run on CPU-only hardware (Intel i5-12400, no GPU; BF16 + eager
  attention). This is a personal-repo reproduction, not the full-power protocol.
  Two multipliers were reduced from the frozen defaults:
    - **`n_sequences` 16 -> 4.** Lossless, not a sample-size cut. `_prompts.py`
      holds only 4 unique prompts; decoding is greedy (`temperature = 0`, argmax)
      with `torch.use_deterministic_algorithms(True)`, so sequences 4-15 are
      bit-identical repeats of 0-3. The default 16 is pseudo-replication that
      would have produced a falsely narrow bootstrap CI. Bootstrap CIs in
      `results/fit.json` therefore reflect **4 independent prompts**, and should
      be read as such. To increase genuine N, add unique prompts (a file passed
      via `--prompts`), not repeats.
    - **arm-C `n_t0` 12 -> 6.** Coarser sampling of the Green's function
      G(T, t0) for P5. This lowers resolution, not accuracy: the kernel is
      mapped at 6 impulse positions instead of 12. Does not affect the arm-A/B
      exponent alpha.
  What was deliberately **not** changed: `max_new_tokens` / the recorded
  T-range stay at 4096. Alpha is the log-log slope of eps(T) over T, so
  truncating the T-range would bias it toward saturation; the full range is
  preserved and alpha is unaffected by this deviation. `bit_widths` for the
  sweep (P6) is kept at the full six values {16, 8, 6, 4, 3, 2}.
  The reduced N (4 prompts) must be stated in the README before the numbers are
  reported. Per-run outcomes (alpha, gate verdicts, natural decode lengths) are
  appended below once `h1-lite` completes.

_No checkpoint run has completed yet; `results/` and `assets/` are still empty._
