# pmkvq-h1-todo

Repo. `pmkvq-cumulative-error` (personal account, topics `kv-cache`, `quantization`, `first-principles`).
Source document. `pmkvq-h1-cumulative-error-thinkbook.md`, Sections 9 through 11.
Convention. Follows `oracle-cache-enrichment`. Canonical modules at repo root, scripts produce assets, tests are silent on pass, one notebook that reads as an article.
Discipline. Predictions recorded before execution. Every gate artifact produced by real execution.

---

## 1. What the notebook is

`pmkvq_cumulative_error.ipynb` is the article. It is written after the runs finish and it reports them. It holds the derivation in prose and LaTeX, small illustrative code cells on toy arrays, `show_source` of the canonical functions so the page cannot drift from the module, `run_test` for the invariants, and `show_image` and `show_gif` for assets rendered from real runs.

The notebook never loads a checkpoint and never produces a number that lands in `results/`. Heavy execution belongs to `scripts/` under a `mise` task. This keeps the article cheap to re-run and keeps the results reproducible from a clean checkout.

The reading contract of the reference repo carries over. A passing test prints nothing and returns 0, so any output under a `run_test` cell is a failure report.

---

## 2. Directory skeleton

```
pmkvq-cumulative-error/
├── .mise.toml
├── README.md
├── pyproject.toml
├── pmkvq_cumulative_error.ipynb     # the article
├── quantizer.py                     # Definition 1, reimplemented, not imported from pm_kvq
├── cache_hook.py                    # KV cache interception, per-position injection control
├── observables.py                   # eps(T), T_eff(T), clipping rate, realised sigma^2
├── run_experiment.py                # decode_loop, arm_a, arm_b, arm_c, bitsweep
├── analysis.py                      # loglog_fit, bootstrap_ci, recover_gamma, classify_outcome
├── style.py                         # brand palette, rcParams, colormaps. Single source for every asset
├── predictions/
│   └── h1_predictions.json          # written once, frozen, committed before the first run
├── scripts/
│   ├── freeze_predictions.py
│   ├── run_arm_a.py
│   ├── run_arm_b.py
│   ├── run_arm_c.py
│   ├── run_bitsweep.py
│   ├── fit_all.py
│   ├── make_assets.py               # every png and gif under assets/
│   └── gate.py
├── tests/
│   ├── test_quantizer_floor.py      # realised sigma^2 against R^2 / 12(2^b-1)^2
│   ├── test_injection_mask.py       # first 128 and last 128 positions untouched
│   ├── test_eager_attention.py      # attention probabilities exposed, T_eff computable
│   └── test_fit_recovery.py         # synthetic power law in, alpha and gamma out
├── results/
│   ├── h1_arm_a.csv
│   ├── h1_arm_b.csv
│   ├── h1_arm_c.csv
│   ├── h1_bitsweep.csv
│   ├── h1_fit.json
│   └── h1_gate.json
├── assets/                          # png and gif consumed by the notebook
└── notes/
    └── deviations.md                # anything that departed from the frozen predictions
```

Modules sit at repo root so the notebook's `sys.path.insert(0, REPO_ROOT)` reaches them and `show_source` renders the real definition.

`quantizer.py` reimplements Definition 1 so the noise floor

$$
\sigma^2(b) = \frac{R^2}{12\,(2^b-1)^2}
$$

is known analytically. Importing `pm_kvq.quantization.quantizer` would inherit an unverified floor and leave P6 without a reference line.

---

## 3. Notebook outline

Section numbering follows the thinkbook. Each section states the mechanism, shows the code that makes it concrete, then shows the asset that measures it.

| Cell group | Content | Assets and tests |
|---|---|---|
| Opening | Why a cache slice has to survive quantization before it is worth transferring. The three possible growth outcomes stated up front. | |
| Helpers | `show_source`, `run_test`, `show_image`, `show_gif`, `REPO_ROOT`, `ASSETS`. Same shape as the reference repo. | |
| 1. A quantizer is a noise injection with a known floor | Definition 1 and Lemma 1. Toy numpy cell quantizing a synthetic group, printing realised against predicted $\sigma^2$. The 86.8 dB figure for 16-bit down to 2-bit. | `show_source(quantize_group)`, `run_test("tests/test_quantizer_floor.py")`, `A1` |
| 2. Where the error enters the readout | Proposition 1. Toy softmax perturbation cell showing the value path and the mean-centred key path on an 8-token attention vector. | `A2` |
| 3. Averaging suppresses independent noise | Lemma 3 and Corollary 2. Synthetic sweep of $\frac{1-\rho}{T_{\mathrm{eff}}} + \rho$. The statement that no readout regime grows. | `A3` |
| 4. Feedback is the only growth mechanism | Model 1 and Theorem 1. The ODE, the integrating factor, the closed form $\varepsilon(T) = \frac{c}{1-\gamma^2} + A\gamma^2 T^{\alpha}$. | `A4`, `run_test("tests/test_fit_recovery.py")` |
| 5. The measurement | Arm A against Arm B. `show_source(decode_loop)` to make the feedback switch visible in the control flow. Fitted $\alpha$ per arm, recovered $\gamma = \sqrt{\alpha+1}$. | `A5`, `A6`, `A7` |
| 6. Early tokens dominate | Theorem 2 and Arm C. Green's function measured against predicted, no free parameter beyond $c_0$. | `A8`, `A9`, `A12` |
| 7. The optimal schedule | Theorem 3. Derived $b^{*}(t) = b_0 - \gamma^2\log_4 t$ against the reference staircase. Recorded as an open check. | `A10` |
| 8. What this does not say | Scope boundary from thinkbook Section 13, then the ESCP bridge of Section 12 with placeholder $E_{\text{byte}}$. | `A11` |

Only Sections 5 and 6 may state measured numbers. Sections 1 through 4 are derivation and synthetic illustration, which keeps the mechanism readable without a run behind it.

---

## 4. Assets

Produced by `scripts/make_assets.py`, read from `results/`, embedded base64 into the notebook so the finished article is self-contained on GitHub. Static panels are png, sweeps are gif.

### Measured

| Id | File | Content |
|---|---|---|
| `A5` | `h1_loglog.png` | $\varepsilon$ against $T$ on log-log axes, Arm A and Arm B, fitted lines with bootstrap band, slope annotated per arm. Carries P1, P2 and P3. |
| `A6` | `h1_residuals.png` | Residuals of the log-log fit against $\log T$, one panel per arm. Structure here indicates the power law is the wrong family. |
| `A7` | `h1_teff.png` | Arm A only. $\varepsilon(T)$ on the left axis, $1/T_{\mathrm{eff}}(T)$ on the right, inset scatter with Pearson $r$ annotated. P4. |
| `A8` | `h1_green.png` | Terminal $\varepsilon$ against injection position $t_0$, log-log, overlaid with $G(T,t_0) = \frac{c_0\gamma^2}{t_0}(T/t_0)^{\alpha}$ at the $\gamma$ recovered from `A5`. The overlay is a prediction, not a fit. P5. |
| `A11` | `h1_bitsweep.png` | Terminal $\varepsilon$ in dB against $b$, reference line at $-6.02$ dB per bit, clipping rate on a right axis. P6. |

### Sweeps

| Id | File | Content |
|---|---|---|
| `A9` | `h1_green_kernel.gif` | $G(T,t_0)$ over the $(t_0,T)$ plane, swept across the bootstrap range of $\gamma$. `pcolormesh` under `BRAND_SEQ` with `LogNorm`. Renders early-position dominance as a gradient and shows how much of it survives the uncertainty in $\gamma$. |
| `A12` | `h1_eps_by_layer.gif` | Per-layer $\varepsilon$ swept across decode position. Establishes whether growth is uniform up the stack or concentrated in the deeper blocks that Figure 3 of the paper marks as sensitive. |

### Synthetic

Closed form, no data, no run.

| Id | File | Content |
|---|---|---|
| `A1` | `h1_noise_floor.png` | $\sigma^2(b)$ against $b$ on a log axis, with $\frac{R^2}{12}4^{-b}$ overlaid and the departure at small $b$ visible. |
| `A2` | `h1_softmax_paths.png` | The two terms of Proposition 1 on an 8-token toy attention vector, value path and mean-centred key path drawn separately. |
| `A3` | `h1_averaging.png` | $\frac{1-\rho}{T_{\mathrm{eff}}} + \rho$ against $T_{\mathrm{eff}}$ for $\rho \in \{0, 0.25, 0.5, 1\}$. The floor at $\rho$ and the absence of a growing branch. |
| `A4` | `h1_regimes.png` | Three analytic curves for $\gamma < 1$, $\gamma = 1$, $\gamma > 1$, with the measured $\gamma$ marked once `A5` exists. |
| `A10` | `h1_bitschedule.png` | Derived $b^{*}(t) = b_0 - \gamma^2\log_4 t$ against the staircase $16 \to 8 \to 4 \to 2$ at position doublings, shared $\log t$ axis. The gap is the open check. |

### Palette and plot conventions

Every asset imports from `style.py`. No figure declares a color inline.

```python
# === BRAND PALETTE ===
PRIMARY   = "#5C1D74"
SECONDARY = "#E5C300"
TERTIARY  = "#1D7466"
TEXT_COLOR = "#25172A"
BG_COLOR  = "#FCFAFF"
```

Role assignment, fixed across all assets so a color means the same thing on every page.

| Role | Color | Applies to |
|---|---|---|
| Measured, feedback on | `PRIMARY` | Arm B, free running |
| Measured, feedback off | `SECONDARY` | Arm A, teacher forced |
| Measured, third series | `TERTIARY` | Arm C in `A8`, and every right-axis quantity that is not an arm |
| Analytic prediction | `TEXT_COLOR` at `alpha=0.55`, dashed | fitted lines, the Theorem 2 overlay in `A8`, the $-6.02$ dB reference in `A11`, the derived $b^{*}(t)$ in `A10` |
| Canvas | `BG_COLOR` | figure and axes face |
| Ink | `TEXT_COLOR` | labels, ticks, spines, annotations |

Right axes in `A7` and `A11` carry a second quantity rather than a second arm, so $1/T_{\mathrm{eff}}(T)$ and the clipping rate take `TERTIARY` with a matching axis label color. Arm colors stay reserved for arms.

The $\rho$ family in `A3` and the three regimes in `A4` walk `SECONDARY` to `TERTIARY` to `PRIMARY`, ordered so the growing branch lands on `PRIMARY`. The reference staircase in `A10` takes `TERTIARY` as a measured-class object, since it is the published schedule rather than a derivation of this repo.

```python
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.prop_cycle": mpl.cycler(color=[PRIMARY, SECONDARY, TERTIARY]),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# Sequential ramp for the A9 heatmap. Luminance falls monotonically along it.
BRAND_SEQ = LinearSegmentedColormap.from_list(
    "brand_seq", [BG_COLOR, SECONDARY, TERTIARY, PRIMARY]
)

# Diverging map for signed quantities, zero on the canvas.
BRAND_DIV = LinearSegmentedColormap.from_list(
    "brand_div", [SECONDARY, BG_COLOR, PRIMARY]
)
```

`BRAND_DIV` is centred with `TwoSlopeNorm(vcenter=0)` wherever a signed quantity is drawn, which is the residual panel in `A6` if it moves from scatter to image form.

Measured relative luminance of the ramp, which is why `BRAND_SEQ` is ordered this way.

| Stop | Hex | $L$ | Contrast against `BG_COLOR` |
|---|---|---|---|
| `BG_COLOR` | `#FCFAFF` | 0.963 | |
| `SECONDARY` | `#E5C300` | 0.557 | 1.67 |
| `TERTIARY` | `#1D7466` | 0.137 | 5.41 |
| `PRIMARY` | `#5C1D74` | 0.044 | 10.76 |
| `TEXT_COLOR` | `#25172A` | 0.012 | 16.41 |

Luminance falls monotonically along the four ramp stops, so `BRAND_SEQ` stays readable under `LogNorm` and survives conversion to grayscale.

Two consequences to hold. `SECONDARY` against `BG_COLOR` reaches only 1.67, so Arm A is drawn at `linewidth=2.0` with a filled marker, and `SECONDARY` is never used for text, tick labels, or a thin annotation line. `TERTIARY` against `PRIMARY` reaches 1.99, so those two never carry adjacent categories in the same panel. Arm B and Arm C appear together only in `A8`, where Arm C is the sole measured series.

Remaining conventions. Single axes per figure unless stated. Gridlines only as `which="both", alpha=0.3` on log axes. Arm A and Arm B separated by marker shape as well as color, so a grayscale print stays readable. Captions live in the notebook markdown beneath each asset, following the reference repo, where the image carries no text of its own.

Gif frames inherit the same rcParams. `animation.PillowWriter` is given `savefig_kwargs={"facecolor": BG_COLOR}` so frames do not fall back to white on transparent.

---

## 5. Environment

`.mise.toml` toolchain block.

```toml
[tools]
python = "3.11"
uv = "latest"
```

| Package | Purpose | Constraint |
|---|---|---|
| `torch` | model execution | CUDA build if a GPU is present, CPU build workable for a 0.6B checkpoint |
| `transformers` | checkpoint loading, attention output hooks | pinned, `attn_implementation="eager"` |
| `numpy` | observable arithmetic and every toy cell in the notebook | |
| `scipy` | `stats.linregress`, `stats.pearsonr` | |
| `pandas` | CSV assembly and grouping | |
| `matplotlib` | assets, including `animation.PillowWriter` for the gifs | no seaborn, no style packages, palette and rcParams come from `style.py` |
| `jupyterlab` | rendering the article | dev dependency |
| `nbstripout` | strip outputs before commit | dev dependency, installed as a git hook |

Pin note. `eager` attention is required. Fused and flash kernels do not expose the attention probability tensor, and $T_{\mathrm{eff}}(T) = 1/\|\mathbf{a}_T\|_2^2$ is unobtainable without it. `tests/test_eager_attention.py` exists so a silent fallback to a fused kernel fails before a long run starts.

Determinism. Fixed seed per sequence id, `torch.use_deterministic_algorithms(True)`, sampling temperature recorded in every CSV row.

---

## 6. Frozen predictions

`predictions/h1_predictions.json`, committed before the first run.

```json
{
  "P1": {"metric": "alpha_tf",            "pass": "<= 0.05",               "falsifier": ">= 0.20 with R2 >= 0.9"},
  "P2": {"metric": "alpha_fr",            "pass": ">= 0.20 and R2 >= 0.9", "falsifier": "<= 0.05"},
  "P3": {"metric": "alpha_fr - alpha_tf", "pass": ">= 0.15",               "falsifier": "CI contains 0"},
  "P4": {"metric": "pearson_r_eps_invTeff_armA", "pass": ">= 0.8",         "falsifier": "<= 0.3"},
  "P5": {"metric": "G_ratio_early_late",  "pass": "within 2x of Theorem 2","falsifier": "monotonicity reversed"},
  "P6": {"metric": "db_per_bit",          "pass": "[5.5, 6.5] for b >= 4", "falsifier": "outside [4, 8]"}
}
```

---

## 7. mise tasks

```toml
[tasks.setup]
description = "Create venv and install pinned dependencies"
run = "uv sync"

[tasks.test]
description = "All invariants. Silent on pass, exit 0"
run = """
python tests/test_quantizer_floor.py
python tests/test_injection_mask.py
python tests/test_eager_attention.py
python tests/test_fit_recovery.py
"""

[tasks.freeze-predictions]
description = "Hash and lock predictions/h1_predictions.json"
run = "python scripts/freeze_predictions.py"

[tasks.arm-a]
description = "P1 and P4. Teacher forced, feedback off"
depends = ["test", "freeze-predictions"]
run = "python scripts/run_arm_a.py --out results/h1_arm_a.csv"

[tasks.arm-b]
description = "P2. Free running, feedback on"
depends = ["test", "freeze-predictions"]
run = "python scripts/run_arm_b.py --out results/h1_arm_b.csv"

[tasks.arm-c]
description = "P5. Impulse injection, sweep t0"
depends = ["test", "freeze-predictions"]
run = "python scripts/run_arm_c.py --out results/h1_arm_c.csv"

[tasks.bitsweep]
description = "P6. Terminal deviation against bit-width"
depends = ["test", "freeze-predictions"]
run = "python scripts/run_bitsweep.py --out results/h1_bitsweep.csv"

[tasks.fit]
description = "P3. Log-log regression, bootstrap CI, gamma = sqrt(alpha+1)"
depends = ["arm-a", "arm-b", "arm-c", "bitsweep"]
run = "python scripts/fit_all.py --out results/h1_fit.json"

[tasks.gate]
description = "Evaluate h1_fit.json against the frozen predictions"
depends = ["fit"]
run = "python scripts/gate.py --out results/h1_gate.json"

[tasks.assets]
description = "Render every png and gif under assets/"
depends = ["fit"]
run = "python scripts/make_assets.py --outdir assets/"

[tasks.assets-synthetic]
description = "Render the closed-form assets only. No run required"
run = "python scripts/make_assets.py --outdir assets/ --synthetic-only"

[tasks.nb]
description = "Register the kernel, install the nbstripout hook, open the article"
depends = ["setup"]
run = """
python -m ipykernel install --user --name pmkvq-h1
nbstripout --install
jupyter lab pmkvq_cumulative_error.ipynb
"""

[tasks.h1]
description = "Full ladder, artifacts and assets"
depends = ["arm-a", "arm-b", "arm-c", "bitsweep", "fit", "gate", "assets"]
```

`nb` sits outside the ladder because the article consumes artifacts and produces none.

Task to prediction mapping.

| Task | Predictions | Artifact |
|---|---|---|
| `test` | precondition for P4 and P6 | exit status |
| `arm-a` | P1, P4 | `h1_arm_a.csv` |
| `arm-b` | P2 | `h1_arm_b.csv` |
| `arm-c` | P5 | `h1_arm_c.csv` |
| `bitsweep` | P6 | `h1_bitsweep.csv` |
| `fit` | P3 | `h1_fit.json` |
| `gate` | all | `h1_gate.json` |

---

## 8. Run parameters

| Parameter | Value | Rationale |
|---|---|---|
| Checkpoint | Qwen3-0.6B, BF16 reference | 28 layers, 8 key-value heads, head dimension 128, matching the reference repo's target |
| Group size | 128 | matches the PM-KVQ implementation |
| Granularity | per-channel Key, per-token Value | matches KIVI and PM-KVQ |
| Preserved positions | first 128, last 128 in BF16 | matches the reference, defines the injection set $\mathcal{Q}$, pinned by `test_injection_mask.py` |
| Bit-widths | $b \in \{16, 8, 6, 4, 3, 2\}$ | $b \ge 4$ tests the 6.02 dB law, $b \le 3$ probes the clipping departure |
| $T$ sampling | log-spaced, at least two decades | the prediction is a power law |
| Sequences | $N \ge 16$ per configuration | mean and interquartile range reported |
| Arm C | window $w = 256$, $t_0$ log-spaced | isolates $G(T,t_0)$ without the uniform-mixing approximation |

---

## 9. Order of work

0. `mise run assets-synthetic`, then notebook Sections 1 through 4. Closed form only, no checkpoint. Writing the derivation first fixes the frozen thresholds against the model rather than against intuition.
1. `quantizer.py` and `tests/test_quantizer_floor.py`. The analytic floor is verified before any checkpoint loads.
2. `predictions/h1_predictions.json` frozen and committed.
3. `cache_hook.py` and `tests/test_injection_mask.py`. Per-position injection control is the shared dependency of all three arms.
4. `observables.py` and `tests/test_eager_attention.py`. Confirm $T_{\mathrm{eff}}$ is recoverable before any long run.
5. Arm A. Cheapest arm, and P1 failing invalidates Lemma 3 before Arm B is worth running.
6. Arm B, then Arm C, then bitsweep.
7. `fit`, `gate`, `assets`.
8. Notebook Sections 5 through 8, written against the assets that now exist.

Stop condition. `results/h1_gate.json` holds a verdict per prediction, `assets/` holds every referenced file, the notebook runs top to bottom with all `run_test` cells silent, and `notes/deviations.md` records anything that departed from the frozen file.

---

## 10. Open items carried from the thinkbook

| Item | Status |
|---|---|
| Uniform mixing approximation $a_{T,t}\approx 1/T$ in Theorem 1 | unvalidated, Arm C is the check |
| Depth dependence of $\gamma$ | partially addressed by `A12`, out of scope across model sizes |
| Shape mismatch between $b^{*}(t)$ and the reference staircase | open, `A10` records it |
| Whether $D(b)$ or $\varepsilon(T)$ is the terminal quantity for ESCP | design decision, unresolved |
| $E_{\text{byte}}$ and $E_{\text{frame}}$ in notebook Section 8 | placeholder until the harness and `espnow-frame-overhead` supply values, captioned as illustrative |
