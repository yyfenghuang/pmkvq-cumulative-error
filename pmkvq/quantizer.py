"""Definition 1 of the thinkbook: the asymmetric uniform group quantizer.

Reimplemented here rather than imported from ``pm_kvq`` so the noise floor

    sigma^2(b) = R^2 / (12 (2^b - 1)^2)

is known analytically instead of inherited. Without the analytic floor, P6
(the 6.02 dB-per-bit law) has no reference line.

See ``docs/pmkvq-h1-cumulative-error-thinkbook.md`` Section 2. This matches
``UntrainableQuantizer.fake_quant`` in the reference implementation with
``round_zeros=True``, ``symmetric=False``.
"""
from __future__ import annotations

import numpy as np

# Positions preserved at full precision are stored at this width in the
# reference implementation. Kept here as a named constant so the injection
# machinery and the tests agree on one value.
INT16_BITS = 16

# The reference group size for the KV cache.
GROUP_SIZE = 128


def quant_params(x: np.ndarray, b: int) -> tuple[float, float]:
    """Scale ``S`` and zero point ``Z`` for one group (asymmetric, round_zeros).

    ``S = R / (2^b - 1)`` with ``R = x_max - x_min`` and
    ``Z = round(-x_min / S)``. A degenerate (constant) group has ``R = 0``; we
    fall back to ``S = 1`` so the group round-trips exactly with zero error.
    """
    x = np.asarray(x, dtype=np.float64)
    x_min = float(x.min())
    x_max = float(x.max())
    R = x_max - x_min
    q_max = (1 << b) - 1
    S = R / q_max if R > 0 else 1.0
    Z = float(np.rint(-x_min / S))
    return S, Z


def fake_quant_group(x: np.ndarray, b: int) -> np.ndarray:
    """Quantize-dequantize a single 1-D group. Returns a float64 array.

    ``Q_b(x) = S (clamp(round(x / S) + Z, 0, 2^b - 1) - Z)``.

    A constant group (range 0) has a single distinct value, representable at any
    bit-width by one code, so it round-trips exactly with zero error (its
    analytic floor is 0). Returned unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.max() == x.min():
        return x.copy()
    S, Z = quant_params(x, b)
    q_max = (1 << b) - 1
    q = np.clip(np.rint(x / S) + Z, 0, q_max)
    return S * (q - Z)


def fake_quant(x: np.ndarray, b: int, group_size: int = GROUP_SIZE, axis: int = -1) -> np.ndarray:
    """Group-wise fake quantization along ``axis``.

    Groups of ``group_size`` consecutive elements along ``axis`` are quantized
    independently. A trailing partial group (when the axis length is not a
    multiple of ``group_size``) is quantized on its own. Vectorized across all
    leading dimensions; the only Python loop is over group blocks.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    n = shape[-1]
    flat = x.reshape(-1, n)
    out = np.empty_like(flat)
    q_max = (1 << b) - 1
    for start in range(0, n, group_size):
        g = flat[:, start:start + group_size]
        g_min = g.min(axis=1, keepdims=True)
        g_max = g.max(axis=1, keepdims=True)
        R = g_max - g_min
        S = np.where(R > 0, R / q_max, 1.0)
        Z = np.rint(-g_min / S)
        q = np.clip(np.rint(g / S) + Z, 0.0, float(q_max))
        deq = S * (q - Z)
        # Constant groups (R == 0) round-trip exactly; pass them through.
        out[:, start:start + group_size] = np.where(R > 0, deq, g)
    out = out.reshape(shape)
    return np.moveaxis(out, -1, axis)


def quant_error(x: np.ndarray, b: int, group_size: int = GROUP_SIZE, axis: int = -1) -> np.ndarray:
    """Per-element error delta = Q_b(x) - x (Definition 2)."""
    return fake_quant(x, b, group_size=group_size, axis=axis) - np.asarray(x, dtype=np.float64)


def noise_floor(R: float | np.ndarray, b: int) -> float | np.ndarray:
    """Analytic high-resolution error variance sigma^2(b) = R^2 / 12 (2^b - 1)^2.

    Lemma 1 of the thinkbook. ``R`` is the group range ``x_max - x_min``.
    """
    return (np.asarray(R, dtype=np.float64) ** 2) / (12.0 * ((2 ** b - 1) ** 2))


def db_per_bit() -> float:
    """The 6.02 dB-per-bit slope, 10 log10(4) (Corollary 1)."""
    return 10.0 * np.log10(4.0)


def clip_rate_group(x: np.ndarray, b: int) -> float:
    """Fraction of elements in a single group that land on q_min or q_max."""
    x = np.asarray(x, dtype=np.float64)
    S, Z = quant_params(x, b)
    q_max = (1 << b) - 1
    q_unclamped = np.rint(x / S) + Z
    hit = (q_unclamped <= 0) | (q_unclamped >= q_max)
    return float(np.mean(hit))
