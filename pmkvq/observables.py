"""The instrumented quantities: eps(T), T_eff(T), clipping rate, realised sigma^2.

Pure numpy. Every function accepts arrays already pulled off the model, so this
module has no torch dependency and is testable in isolation.
"""
from __future__ import annotations

import numpy as np

from pmkvq import quantizer


def eps(o_quant: np.ndarray, o_fp: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalised deviation of the attention readout at one position.

        eps(T) = || o_quant_T - o_fp_T ||^2 / || o_fp_T ||^2

    Reduces over ``axis`` (the head-dimension by default). Normalisation removes
    residual-stream scale drift, which would otherwise be confounded with
    growth. Returns an array with ``axis`` removed.
    """
    o_quant = np.asarray(o_quant, dtype=np.float64)
    o_fp = np.asarray(o_fp, dtype=np.float64)
    num = np.sum((o_quant - o_fp) ** 2, axis=axis)
    den = np.sum(o_fp ** 2, axis=axis)
    return num / np.where(den > 0, den, np.nan)


def t_eff(attn: np.ndarray, axis: int = -1) -> np.ndarray:
    """Effective attention support T_eff(T) = 1 / sum_t a_{T,t}^2 (Definition 3).

    ``attn`` is a row-stochastic attention vector (or a stack of them, reduced
    over ``axis``, the key/position axis). Bounded in ``[1, T]``.
    """
    attn = np.asarray(attn, dtype=np.float64)
    return 1.0 / np.sum(attn ** 2, axis=axis)


def clipping_rate(x: np.ndarray, b: int, group_size: int = quantizer.GROUP_SIZE,
                  axis: int = -1) -> float:
    """Fraction of elements hitting q_min or q_max, group-wise along ``axis``.

    Departures of the bit-sweep from 6.02 dB/bit are attributed to clipping
    (P6), so this is recorded per run.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    q_max = (1 << b) - 1
    total_hit = 0
    total = 0
    for start in range(0, n, group_size):
        g = flat[:, start:start + group_size]
        g_min = g.min(axis=1, keepdims=True)
        g_max = g.max(axis=1, keepdims=True)
        R = g_max - g_min
        S = np.where(R > 0, R / q_max, 1.0)
        Z = np.rint(-g_min / S)
        q_unclamped = np.rint(g / S) + Z
        hit = (q_unclamped <= 0) | (q_unclamped >= q_max)
        total_hit += int(np.count_nonzero(hit))
        total += g.size
    return total_hit / total if total else 0.0


def realised_sigma2(x: np.ndarray, b: int, group_size: int = quantizer.GROUP_SIZE,
                    axis: int = -1) -> float:
    """Empirical variance of the quantization error, Var(Q_b(x) - x).

    Compared against the analytic floor from ``quantizer.noise_floor`` in
    ``test_quantizer_floor.py`` and in the P6 asset.
    """
    delta = quantizer.quant_error(x, b, group_size=group_size, axis=axis)
    return float(np.var(delta))


def predicted_sigma2(x: np.ndarray, b: int, group_size: int = quantizer.GROUP_SIZE,
                     axis: int = -1) -> float:
    """Range-weighted analytic floor averaged over groups, R^2 / 12 (2^b-1)^2.

    The high-resolution prediction that ``realised_sigma2`` is checked against.
    Averages the per-group floor weighted by group element count.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    weighted_sum = 0.0
    total = 0
    for start in range(0, n, group_size):
        g = flat[:, start:start + group_size]
        R = g.max(axis=1) - g.min(axis=1)
        weighted_sum += float(np.sum(quantizer.noise_floor(R, b)) * g.shape[1])
        total += g.size
    return weighted_sum / total if total else 0.0
