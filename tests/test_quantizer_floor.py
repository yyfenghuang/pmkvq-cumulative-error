#!/usr/bin/env python
"""Realised sigma^2 against the analytic floor R^2 / 12 (2^b-1)^2 (Lemma 1).

Silent on pass, exit 0. Any stdout under this test is a failure report.
Runs with only numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pmkvq import quantizer
from pmkvq import observables


def main() -> int:
    rng = np.random.default_rng(0)
    fails = []

    # High-resolution regime: realised variance tracks the analytic floor, and
    # the ratio approaches 1 as b grows. Use a large sample of a smooth density
    # so the uniform-error model holds.
    x = rng.normal(0.0, 1.0, size=(64, 128))     # 64 groups of size 128
    R = float(np.mean(x.max(axis=1) - x.min(axis=1)))
    for b in (8, 6, 4):
        realised = observables.realised_sigma2(x, b, group_size=128)
        predicted = observables.predicted_sigma2(x, b, group_size=128)
        ratio = realised / predicted
        # High-resolution model: within 25% at these widths.
        if not (0.75 <= ratio <= 1.25):
            fails.append(f"b={b}: realised/predicted={ratio:.3f} outside [0.75, 1.25]")

    # 6.02 dB per bit between adjacent high-res widths.
    s8 = observables.predicted_sigma2(x, 8, 128)
    s7 = observables.predicted_sigma2(x, 7, 128)
    db = 10.0 * np.log10(s7 / s8)
    if not (5.5 <= db <= 6.5):
        fails.append(f"dB/bit between b=8,7 is {db:.3f}, outside [5.5, 6.5]")

    # Error bound: away from clipping, |delta| <= S/2 elementwise.
    b = 6
    for i in range(x.shape[0]):
        g = x[i]
        S, _ = quantizer.quant_params(g, b)
        delta = quantizer.fake_quant_group(g, b) - g
        if np.max(np.abs(delta)) > S / 2 + 1e-9:
            fails.append(f"group {i}: |delta|max={np.max(np.abs(delta)):.4g} exceeds S/2={S/2:.4g}")
            break

    # Degenerate (constant) group round-trips exactly.
    const = np.full(128, 3.14)
    if np.max(np.abs(quantizer.fake_quant_group(const, 4) - const)) > 1e-9:
        fails.append("constant group did not round-trip exactly")

    if fails:
        print("FAIL test_quantizer_floor")
        for f in fails:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
