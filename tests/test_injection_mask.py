#!/usr/bin/env python
"""First 128 and last 128 positions are never quantized (the injection set Q).

Silent on pass, exit 0. Runs with only numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pmkvq import cache_hook


def main() -> int:
    fails = []

    T = 2048
    m = cache_hook.injection_mask(T, preserve_front=128, preserve_back=128)

    if m[:128].any():
        fails.append("first 128 positions are not all preserved")
    if m[-128:].any():
        fails.append("last 128 positions are not all preserved")
    if not m[128:T - 128].all():
        fails.append("interior positions are not all injected")
    expected = (T - 256)
    if int(m.sum()) != expected:
        fails.append(f"injected count {int(m.sum())} != expected {expected}")

    # Degenerate: T <= 256 preserves everything (no interior to inject).
    m_small = cache_hook.injection_mask(200, 128, 128)
    if m_small.any():
        fails.append("T=200 should inject nothing (bands overlap)")

    # Arm C window intersects Q: outside the window nothing is injected, and the
    # preserved bands still win inside it.
    win = (300, 556)
    mw = cache_hook.injection_mask(T, 128, 128, window=win)
    if mw[:300].any() or mw[556:].any():
        fails.append("window mask leaked outside [300, 556)")
    if not mw[300:556].all():
        fails.append("window interior (all within Q) not fully injected")

    # A window that dips into the front band must not override the band.
    mw2 = cache_hook.injection_mask(T, 128, 128, window=(64, 320))
    if mw2[:128].any():
        fails.append("window overrode the front preserved band")
    if not mw2[128:320].all():
        fails.append("window portion inside Q not injected")

    # quantize_positions leaves preserved positions bit-exact.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(T, 8, 128))            # [pos, heads, head_dim]
    q = cache_hook.quantize_positions(x, b=4, mask=m, axis=0)
    if not np.array_equal(q[:128], x[:128]) or not np.array_equal(q[-128:], x[-128:]):
        fails.append("quantize_positions altered a preserved position")
    if np.array_equal(q[128:T - 128], x[128:T - 128]):
        fails.append("quantize_positions left the interior untouched")

    if fails:
        print("FAIL test_injection_mask")
        for f in fails:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
