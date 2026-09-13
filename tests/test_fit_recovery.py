#!/usr/bin/env python
"""Synthetic power law in, alpha and gamma out (Theorem 1 round-trip).

Generates eps(T) = A T^alpha with known alpha, adds lognormal noise, and checks
that loglog_fit recovers alpha (hence gamma = sqrt(alpha+1)) and that
classify_outcome assigns the right regime. Silent on pass, exit 0. Runs with
numpy + scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pmkvq import analysis


def main() -> int:
    fails = []
    rng = np.random.default_rng(0)
    T = np.unique(np.round(np.logspace(np.log10(256), np.log10(4096), 24))).astype(float)

    cases = [
        (0.35, "power-law-growth"),
        (0.0, "marginal"),
        (-0.30, "saturation"),
    ]
    for alpha_true, regime in cases:
        eps = 1e-3 * T ** alpha_true
        eps = eps * np.exp(rng.normal(0.0, 0.05, size=T.size))  # multiplicative noise
        fit = analysis.loglog_fit(T, eps)
        if abs(fit.alpha - alpha_true) > 0.05:
            fails.append(f"alpha recovery: got {fit.alpha:.3f}, true {alpha_true:.3f}")
        # R2 is only meaningful when there is a slope to explain; the marginal
        # (flat) case has no trend, so its R2 is near zero by construction.
        if abs(alpha_true) > 0.1 and fit.r2 < 0.9:
            fails.append(f"alpha={alpha_true}: R2={fit.r2:.3f} < 0.9")
        gamma_expected = np.sqrt(alpha_true + 1)
        if abs(fit.gamma - gamma_expected) > 0.05:
            fails.append(f"gamma recovery: got {fit.gamma:.3f}, expected {gamma_expected:.3f}")
        if analysis.classify_outcome(fit.alpha) != regime:
            fails.append(f"regime for alpha={alpha_true}: got {analysis.classify_outcome(fit.alpha)}, want {regime}")

    # Bootstrap CI brackets the true slope for the growth case.
    eps = 1e-3 * T ** 0.35 * np.exp(rng.normal(0.0, 0.05, size=T.size))
    ci = analysis.bootstrap_ci(T, eps, n_boot=500, seed=1)
    if not (ci["alpha_lo"] <= 0.35 <= ci["alpha_hi"]):
        fails.append(f"bootstrap CI [{ci['alpha_lo']:.3f}, {ci['alpha_hi']:.3f}] misses 0.35")

    # recover_gamma is undefined below alpha = -1.
    if not np.isnan(analysis.recover_gamma(-1.5)):
        fails.append("recover_gamma(-1.5) should be NaN")

    if fails:
        print("FAIL test_fit_recovery")
        for f in fails:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
