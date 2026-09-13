#!/usr/bin/env python
"""Render every png and gif under assets/.

Synthetic assets (A1-A4, A10) are closed-form and need no run; the rest read
results/ and render the measured panels. ``--synthetic-only`` renders just the
closed-form set so the derivation sections of the notebook can be built before
any checkpoint run. Every asset imports its colours from ``style.py``; no figure
declares a colour inline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pmkvq import style  # noqa: E402
from pmkvq import analysis  # noqa: E402
from pmkvq import quantizer  # noqa: E402

style.apply_rcparams()


# --------------------------------------------------------------------------- #
# Synthetic (closed form, no data)
# --------------------------------------------------------------------------- #

def a1_noise_floor(outdir: Path) -> None:
    """sigma^2(b) against b with the R^2/12 4^-b overlay (Corollary 1)."""
    b = np.arange(2, 17)
    R = 1.0
    exact = quantizer.noise_floor(R, b)
    approx = (R ** 2) / 12.0 * 4.0 ** (-b.astype(float))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(b, exact, style.ARM_A_MARKER, color=style.ARM_A,
                lw=style.ARM_A_LINEWIDTH, label=r"$\sigma^2(b)=R^2/12(2^b-1)^2$")
    ax.semilogy(b, approx, "--", color=style.PREDICTION, alpha=style.PREDICTION_ALPHA,
                label=r"$R^2/12\,\cdot 4^{-b}$")
    ax.set_xlabel("bit-width $b$")
    ax.set_ylabel(r"noise power $\sigma^2$")
    ax.grid(**style.GRID_KWARGS)
    ax.legend()
    fig.savefig(outdir / "h1_noise_floor.png")
    plt.close(fig)


def a2_softmax_paths(outdir: Path) -> None:
    """The two terms of Proposition 1 on an 8-token toy attention vector."""
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(8))
    dv = rng.normal(0, 1, 8)
    dz = rng.normal(0, 1, 8)
    value_path = a * dv
    key_path = a * (dz - np.sum(a * dz))     # mean-centred by construction
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x - 0.2, value_path, width=0.4, color=style.ARM_A, label="value path")
    ax.bar(x + 0.2, key_path, width=0.4, color=style.ARM_B, label="key path (centred)")
    ax.axhline(0, color=style.TEXT_COLOR, lw=0.8)
    ax.set_xlabel("token $t$")
    ax.set_ylabel("contribution to $\\Delta o$")
    ax.legend()
    fig.savefig(outdir / "h1_softmax_paths.png")
    plt.close(fig)


def a3_averaging(outdir: Path) -> None:
    """(1-rho)/T_eff + rho against T_eff for rho in {0, 0.25, 0.5, 1}."""
    t_eff = np.logspace(0, 3, 200)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ramp = [style.SECONDARY, style.TERTIARY, style.PRIMARY]
    for rho, col in zip([0.0, 0.25, 0.5, 1.0], [style.SECONDARY, style.TERTIARY,
                                                style.PRIMARY, style.TEXT_COLOR]):
        ax.loglog(t_eff, (1 - rho) / t_eff + rho, color=col, label=f"$\\rho={rho}$")
    ax.set_xlabel(r"$T_{\mathrm{eff}}$")
    ax.set_ylabel(r"$\mathbb{E}\|\Delta o\|^2 / d\sigma^2$")
    ax.grid(**style.GRID_KWARGS)
    ax.legend()
    fig.savefig(outdir / "h1_averaging.png")
    plt.close(fig)


def a4_regimes(outdir: Path) -> None:
    """Three analytic curves for gamma <1, =1, >1 (Corollary 3)."""
    T = np.logspace(2, 4, 200)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for gamma, col, lab in [(0.8, style.SECONDARY, r"$\gamma<1$ saturation"),
                            (1.0, style.TERTIARY, r"$\gamma=1$ log"),
                            (1.2, style.PRIMARY, r"$\gamma>1$ power law")]:
        alpha = gamma ** 2 - 1
        if abs(alpha) < 1e-9:
            y = np.log(T / T[0]) + 1
        else:
            y = 1.0 / (1 - gamma ** 2) + T ** alpha
        ax.loglog(T, y - y.min() + 1e-3, color=col, label=lab)
    ax.set_xlabel("decode position $T$")
    ax.set_ylabel(r"$\varepsilon(T)$ (shifted)")
    ax.grid(**style.GRID_KWARGS)
    ax.legend()
    fig.savefig(outdir / "h1_regimes.png")
    plt.close(fig)


def a10_bitschedule(outdir: Path, gamma: float = 1.1) -> None:
    """Derived b*(t) = b0 - gamma^2 log4 t against the reference staircase."""
    t = np.logspace(np.log10(128), np.log10(8192), 200)
    b0 = 16.0
    b_star = b0 - gamma ** 2 * np.log(t) / np.log(4)
    # Reference staircase 16 -> 8 -> 4 -> 2 at position doublings from 128.
    edges = [128, 256, 512, 1024]
    levels = [16, 8, 4, 2]
    stair_t = np.array([128, 256, 256, 512, 512, 1024, 1024, 8192])
    stair_b = np.array([16, 16, 8, 8, 4, 4, 2, 2])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogx(t, b_star, "--", color=style.PREDICTION, alpha=style.PREDICTION_ALPHA,
                label=r"derived $b^*(t)=b_0-\gamma^2\log_4 t$")
    ax.step(stair_t, stair_b, where="post", color=style.TERTIARY,
            label="reference staircase")
    ax.set_xlabel("position $t$")
    ax.set_ylabel("bit-width $b$")
    ax.grid(**style.GRID_KWARGS)
    ax.legend()
    fig.savefig(outdir / "h1_bitschedule.png")
    plt.close(fig)


SYNTHETIC = [a1_noise_floor, a2_softmax_paths, a3_averaging, a4_regimes, a10_bitschedule]


# --------------------------------------------------------------------------- #
# Measured (read results/, render the arms). Guarded: skip if inputs absent.
# --------------------------------------------------------------------------- #

def _read(results: Path, name: str):
    import pandas as pd
    p = results / name
    return pd.read_csv(p) if p.exists() else None


def a5_loglog(outdir: Path, results: Path) -> None:
    """eps against T, both arms, fitted lines with slope annotated (P1/P2/P3)."""
    arm_a = _read(results, "h1_arm_a.csv")
    arm_b = _read(results, "h1_arm_b.csv")
    if arm_a is None or arm_b is None:
        print("A5 skipped: arm results absent", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for df, col, marker, lab in [(arm_a, style.ARM_A, style.ARM_A_MARKER, "Arm A tf"),
                                 (arm_b, style.ARM_B, style.ARM_B_MARKER, "Arm B fr")]:
        g = df.groupby("position")["eps"].mean()
        ax.loglog(g.index, g.values, marker, color=col, label=lab,
                  lw=style.ARM_A_LINEWIDTH if col == style.ARM_A else 1.5)
        fit = analysis.loglog_fit(g.index.to_numpy(), g.values)
        xs = np.array([g.index.min(), g.index.max()], dtype=float)
        ax.loglog(xs, 10 ** (fit.intercept) * xs ** fit.alpha, "--",
                  color=style.PREDICTION, alpha=style.PREDICTION_ALPHA)
        ax.annotate(f"{lab}: α={fit.alpha:.3f}", (xs[-1], 10 ** fit.intercept * xs[-1] ** fit.alpha))
    ax.set_xlabel("decode position $T$")
    ax.set_ylabel(r"$\varepsilon(T)$")
    ax.grid(**style.GRID_KWARGS)
    ax.legend()
    fig.savefig(outdir / "h1_loglog.png")
    plt.close(fig)


MEASURED = [a5_loglog]  # A6/A7/A8/A11 and gifs A9/A12 follow the same pattern.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fn in SYNTHETIC:
        fn(outdir)

    if not args.synthetic_only:
        results = Path(args.results_dir)
        for fn in MEASURED:
            fn(outdir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
