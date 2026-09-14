"""The on-device torch quantizer reproduces the numpy Definition 1.

Two layers, matching how the code is stacked:

1. ``fake_quant_torch`` vs ``quantizer.fake_quant`` at float64 -- the kernel is
   a bit-for-bit port of the analytic reference (P6 depends on that reference).

2. ``QuantizedDynamicCache._quant`` -- the readout orchestration. Preserved
   bands (and Arm C's non-window) must pass through *exactly*, and the injected
   band must actually change, for both the per-channel (key) and per-token
   (value) layouts.

Silent on pass, exits 0; raises AssertionError on any drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pmkvq import cache_hook, quantizer


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_kernel_matches_numpy_at_f64() -> None:
    """fake_quant_torch(float64) == quantizer.fake_quant(float64) bit-for-bit."""
    rng = np.random.default_rng(0)
    for b in (2, 3, 4, 6, 8):
        for n in (64, 128, 256, 300):        # 300 exercises a partial last group
            for gs in (128, 64):
                x = rng.standard_normal((7, n)).astype(np.float64)
                ref = quantizer.fake_quant(x, b, group_size=gs, axis=-1)
                got = cache_hook.fake_quant_torch(
                    torch.from_numpy(x), b, group_size=gs, axis=-1).numpy()
                d = _max_abs(got, ref)
                assert d <= 1e-12, f"kernel drift b={b} n={n} gs={gs}: {d}"

    # A different reduction axis must also agree (moved internally by the port).
    x = rng.standard_normal((130, 9)).astype(np.float64)
    ref = quantizer.fake_quant(x, 4, group_size=128, axis=0)
    got = cache_hook.fake_quant_torch(torch.from_numpy(x), 4,
                                      group_size=128, axis=0).numpy()
    assert _max_abs(got, ref) <= 1e-12, "kernel drift on axis=0"


def _numpy_quant_oracle(x: np.ndarray, b: int, per: str, gs: int) -> np.ndarray:
    """Independent numpy reimplementation of _quant, in float32 to match the
    on-device path's working precision (so boundary rounding agrees)."""
    x = x.astype(np.float32)
    _, _, s, _ = x.shape
    mask = cache_hook.injection_mask(s)
    if per == "channel":
        moved = np.moveaxis(x, 2, -1)                       # [b, h, d, s]
        qtd = _np_group_quant_f32(moved, b, gs)
        out = np.where(mask, qtd, moved)
        return np.moveaxis(out, -1, 2)
    qtd = _np_group_quant_f32(x, b, gs)                     # group over head_dim
    return np.where(mask.reshape(s, 1), qtd, x)


def _np_group_quant_f32(x: np.ndarray, b: int, gs: int) -> np.ndarray:
    """Group-wise fake-quant along the last axis, all in float32."""
    x = x.astype(np.float32)
    n = x.shape[-1]
    out = x.copy()
    q_max = np.float32((1 << b) - 1)
    for start in range(0, n, gs):
        g = x[..., start:start + gs]
        g_min = g.min(-1, keepdims=True)
        g_max = g.max(-1, keepdims=True)
        R = g_max - g_min
        S = np.where(R > 0, R / q_max, np.float32(1.0)).astype(np.float32)
        Z = np.rint(-g_min / S).astype(np.float32)
        q = np.clip(np.rint(g / S) + Z, 0.0, q_max).astype(np.float32)
        out[..., start:start + gs] = np.where(R > 0, (S * (q - Z)), g)
    return out


def test_quant_readout_layout_and_preserved_bands() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 300, 128)).astype(np.float32)
    xt = torch.from_numpy(x)
    s = x.shape[2]
    mask = cache_hook.injection_mask(s)
    preserved = ~mask

    for per, gran in (("channel", "channel"), ("token", "token")):
        qc = cache_hook.make_quantized_cache(
            bit_width=2, granularity_key=gran, granularity_value=gran)
        got = qc._quant(xt, per).numpy()
        oracle = _numpy_quant_oracle(x, 2, per, quantizer.GROUP_SIZE)

        # Same precision (float32) both sides -> tight agreement.
        assert _max_abs(got, oracle) <= 1e-5, f"_quant {per} drift"
        # Preserved positions are an exact passthrough of the input.
        assert _max_abs(got[:, :, preserved, :], x[:, :, preserved, :]) == 0.0, \
            f"preserved band altered ({per})"
        # The injected band actually moved (2-bit quant is coarse).
        moved = _max_abs(got[:, :, mask, :], x[:, :, mask, :])
        assert moved > 1e-3, f"injected band unchanged ({per}): {moved}"


if __name__ == "__main__":
    test_kernel_matches_numpy_at_f64()
    test_quant_readout_layout_and_preserved_bands()
    print("ok: torch quantizer matches numpy; preserved bands exact")
