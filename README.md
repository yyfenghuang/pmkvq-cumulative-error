# pmkvq-cumulative-error

Hypothesis: does per-step KV-cache quantization error *accumulate* with
decode position, or does it saturate? PM-KVQ (arXiv 2505.18610, ICLR 2026)
asserts accumulation qualitatively. This repo converts the assertion into a
growth law

$$
\varepsilon(T) = \frac{c}{1-\gamma^2} + A\gamma^2 T^{\alpha}, \qquad \alpha = \gamma^2 - 1,
$$

and measures the exponent $\alpha$ with a single log-log regression. The
paper's claim is exactly the statement $\alpha > 0$, and that statement is now
falsifiable.

See [`docs/pmkvq-h1-cumulative-error-thinkbook.md`](docs/pmkvq-h1-cumulative-error-thinkbook.md)
for the derivation and [`docs/pmkvq-h1-todo.md`](docs/pmkvq-h1-todo.md) for the
build plan this scaffold follows.

## Layout

The canonical modules live in the `pmkvq/` package, imported as
`from pmkvq import quantizer`. Entrypoints (`scripts/`, `tests/`, the notebook)
put the repo root on `sys.path` so `pmkvq` resolves without an editable install,
and `show_source` still renders the real definition.

| File | Role |
|---|---|
| `pmkvq/quantizer.py` | Definition 1, asymmetric uniform group quantizer, reimplemented so the noise floor is known analytically |
| `pmkvq/cache_hook.py` | KV-cache interception, per-position injection control, the injection mask |
| `pmkvq/observables.py` | `eps(T)`, `T_eff(T)`, clipping rate, realised `sigma^2` |
| `pmkvq/run_experiment.py` | `decode_loop` (the feedback switch), `arm_a`, `arm_b`, `arm_c`, `bitsweep` |
| `pmkvq/analysis.py` | `loglog_fit`, `bootstrap_ci`, `recover_gamma`, `classify_outcome` |
| `pmkvq/style.py` | brand palette, rcParams, colormaps — the single source for every asset |
| `pmkvq_cumulative_error.ipynb` | the article, written after the runs finish |

`predictions/` holds the frozen thresholds, `scripts/` holds the heavy
executables run under `mise`, `tests/` holds the invariants (silent on pass),
`results/` and `assets/` hold artifacts produced by real execution, and
`notes/deviations.md` records anything that departed from the frozen file.

## Discipline

- Predictions are recorded **before** execution in `predictions/h1_predictions.json`.
- The notebook never loads a checkpoint and never produces a number that lands
  in `results/`. Heavy execution belongs to `scripts/` under a `mise` task.
- A passing test prints nothing and returns 0. Any output under a test is a
  failure report.
- `eager` attention is mandatory. Fused/flash kernels hide the attention
  probabilities and make `T_eff` unobservable.

## Run ladder

```bash
mise run setup             # uv sync
mise run assets-synthetic  # closed-form figures, no checkpoint needed
mise run test              # invariants, silent on pass
mise run freeze-predictions
mise run h1                # arm-a, arm-b, arm-c, bitsweep, fit, gate, assets
mise run nb                # open the article
```

| Task | What it predicts | Artifact |
|---|---|---|
| `arm-a` | With feedback off, error does not grow — the log-log slope `α ≤ 0` (**P1**) — and it tracks the readout's averaging, `ε ∝ 1/T_eff` (**P4**) | `results/arm_a.csv` |
| `arm-b` | With feedback on, error grows as a power law, `α > 0` with `R² ≥ 0.9` (**P2**) | `results/arm_b.csv` |
| `arm-c` | Noise injected at early positions dominates terminal error over the same noise injected late (**P5**) | `results/arm_c.csv` |
| `bitsweep` | Terminal error falls `6.02 dB` per added bit for `b ≥ 4` (**P6**) | `results/bitsweep.csv` |
| `fit` | The feedback gain `γ = √(α+1) > 1`, recovered from the arm-A/arm-B slope gap (**P3**) | `results/fit.json` |
| `gate` | Every prediction above, scored against the frozen pass/falsifier thresholds | `results/gate.json` |

## Status

Scaffold. The pure-numpy layers (`quantizer.py`, `observables.py`,
`analysis.py`, `style.py`, and the injection mask in `cache_hook.py`) are
implemented and covered by `test_quantizer_floor.py` and `test_fit_recovery.py`,
which run with only `numpy`/`scipy`. The checkpoint-driven arms
(`cache_hook.py` cache classes, `run_experiment.py`) are implemented against
`transformers==4.44.2` with `attn_implementation="eager"` and require a
Qwen3-0.6B checkpoint plus the synced environment to execute. No `results/` or
`assets/` are committed yet — every gate artifact is produced by a real run.

Run parameters: Qwen3-0.6B BF16 reference, group size 128,
per-channel Key / per-token Value, first 128 and last 128 positions preserved
in BF16, `b ∈ {16, 8, 6, 4, 3, 2}`, log-spaced `T` over ≥ 2 decades, `N ≥ 16`
sequences per configuration.
