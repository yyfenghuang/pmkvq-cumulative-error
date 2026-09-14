"""KV-cache interception and per-position injection control.

Two layers live here:

1. ``injection_mask``: a pure-numpy boolean mask over positions, encoding the
   injection set Q = {128 < t <= T - 128} of the thinkbook (Section 3), plus an
   optional impulse window for Arm C. Importable and testable without torch;
   ``test_injection_mask.py`` pins it.

2. ``QuantizedDynamicCache``: a ``transformers`` cache that stores keys/values
   at full precision and, at readout, returns a group-quantized view for the
   positions the mask selects. Torch/transformers are imported lazily so layer
   (1) has no heavy dependency.

The quantizer is the reimplemented Definition 1 in ``quantizer.py``, applied
on-device by ``fake_quant_torch`` (verified bit-for-bit against the numpy
reference); keeping one floor definition is deliberate (P6 needs the analytic
reference).
"""
from __future__ import annotations

import numpy as np

from pmkvq import quantizer

PRESERVE_FRONT = 128
PRESERVE_BACK = 128


def injection_mask(T: int, preserve_front: int = PRESERVE_FRONT,
                   preserve_back: int = PRESERVE_BACK,
                   window: tuple[int, int] | None = None) -> np.ndarray:
    """Boolean mask of length ``T``; True where quantization is injected.

    The first ``preserve_front`` and last ``preserve_back`` positions are held
    at full precision (the reference stores them in INT16), so the mask is True
    only on ``preserve_front <= i < T - preserve_back`` (0-indexed). This is the
    index set Q of the thinkbook, expressed as a boundary condition.

    ``window = (t0, t0 + w)`` further restricts injection to ``[t0, t0 + w)``
    for the Arm C impulse, intersected with Q so the preserved bands still win.
    """
    mask = np.zeros(T, dtype=bool)
    lo = preserve_front
    hi = T - preserve_back
    if hi > lo:
        mask[lo:hi] = True
    if window is not None:
        w0, w1 = window
        wmask = np.zeros(T, dtype=bool)
        wmask[max(0, w0):min(T, w1)] = True
        mask &= wmask
    return mask


def quantize_positions(x: np.ndarray, b: int, mask: np.ndarray,
                       group_size: int = quantizer.GROUP_SIZE,
                       axis: int = 0) -> np.ndarray:
    """Return a copy of ``x`` with the masked positions group-quantized.

    ``axis`` is the position/token axis of ``x``. Positions where ``mask`` is
    False are copied through untouched (preserved bands and, for Arm C, every
    position outside the impulse window).
    """
    x = np.asarray(x, dtype=np.float64)
    x_pos = np.moveaxis(x, axis, 0)
    out = x_pos.copy()
    sel = np.asarray(mask, dtype=bool)
    if sel.any():
        # Quantize each selected token's feature vector group-wise on the last
        # axis (per-channel granularity is handled by the caller's layout).
        out[sel] = quantizer.fake_quant(x_pos[sel], b, group_size=group_size, axis=-1)
    return np.moveaxis(out, 0, axis)


# --------------------------------------------------------------------------- #
# transformers-facing layer. Torch is imported lazily so importing this module
# for `injection_mask` alone never requires a torch install.
# --------------------------------------------------------------------------- #

def _require_torch():
    try:
        import torch  # noqa: F401
        from transformers.cache_utils import DynamicCache
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "QuantizedDynamicCache requires torch and transformers==4.44.2. "
            "Run `mise run setup` first."
        ) from exc
    return torch, DynamicCache


def fake_quant_torch(x, b: int, group_size: int = quantizer.GROUP_SIZE,
                     axis: int = -1):
    """On-device torch port of :func:`quantizer.fake_quant` (Definition 1).

    Bit-for-bit identical to the numpy reference at float64 (pinned by
    ``tests/test_quant_torch_matches_numpy.py``); run at float32 on GPU it stays
    on the device, so the KV readout no longer round-trips through host numpy.
    Group-wise asymmetric uniform quantize-dequantize along ``axis``; a constant
    group (range 0) round-trips exactly and is passed through unchanged.
    """
    import torch

    x = x.movedim(axis, -1)
    shape = x.shape
    n = shape[-1]
    flat = x.reshape(-1, n)
    out = flat.clone()
    q_max = float((1 << b) - 1)
    for start in range(0, n, group_size):
        g = flat[:, start:start + group_size]
        g_min = g.amin(dim=1, keepdim=True)
        g_max = g.amax(dim=1, keepdim=True)
        R = g_max - g_min
        pos = R > 0
        S = torch.where(pos, R / q_max, torch.ones_like(R))
        Z = torch.round(-g_min / S)
        q = torch.clamp(torch.round(g / S) + Z, 0.0, q_max)
        # Constant groups (R == 0) pass through; matches np.where(R > 0, deq, g).
        out[:, start:start + group_size] = torch.where(pos, S * (q - Z), g)
    return out.reshape(shape).movedim(-1, axis)


def make_quantized_cache(bit_width: int, granularity_key: str = "channel",
                         granularity_value: str = "token",
                         group_size: int = quantizer.GROUP_SIZE,
                         preserve_front: int = PRESERVE_FRONT,
                         preserve_back: int = PRESERVE_BACK,
                         window: tuple[int, int] | None = None):
    """Construct a ``QuantizedDynamicCache`` bound to a bit-width and layout.

    Factory kept separate from the class definition so the torch import stays
    lazy. ``granularity_key='channel'`` and ``granularity_value='token'`` match
    KIVI and PM-KVQ: Key is quantized per channel (group over the head-dim after
    a transpose), Value per token (group over the head-dim directly).
    """
    torch, DynamicCache = _require_torch()

    class QuantizedDynamicCache(DynamicCache):
        """DynamicCache that returns a quantized readout view of K and V.

        Full-precision tensors are kept so the reference (BF16) path and the
        quantized path share identical history; the quantization is applied on
        the *readout*, honouring the preserved bands and, for Arm C, the impulse
        window. Applied every step because the last-128 preserved band slides.
        """

        def __init__(self):
            super().__init__()
            self.bit_width = bit_width
            self.group_size = group_size
            self.granularity_key = granularity_key
            self.granularity_value = granularity_value
            self.preserve_front = preserve_front
            self.preserve_back = preserve_back
            self.window = window

        def _quant(self, tensor, per: str):
            # tensor: [batch, heads, seq, head_dim]. Everything stays on the
            # tensor's device; the group-quant runs in torch (fake_quant_torch)
            # so there is no host round-trip. Positions outside the injection
            # mask (preserved bands, and Arm C's non-window) are quantized too,
            # then restored via torch.where -- token-wise quant is independent
            # per token, so the restored result is identical to skipping them.
            b, h, s, d = tensor.shape
            mask = injection_mask(s, self.preserve_front, self.preserve_back,
                                  self.window)
            inject = torch.from_numpy(mask).to(tensor.device)   # [s], True=quant
            work = tensor.detach().to(torch.float32)
            if per == "channel":
                # group over the sequence axis per (head, channel): move seq last
                moved = work.movedim(2, -1)               # [b, h, d, s]
                qtd = fake_quant_torch(moved, self.bit_width,
                                       group_size=self.group_size, axis=-1)
                out = torch.where(inject, qtd, moved).movedim(-1, 2)
            else:  # per == "token": group over head_dim, mask over seq
                qtd = fake_quant_torch(work, self.bit_width,
                                       group_size=self.group_size, axis=-1)
                out = torch.where(inject.view(s, 1), qtd, work)
            return out.to(dtype=tensor.dtype)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            """Store K/V at full precision, return a quantized readout view.

            ``super().update`` appends the new states to the (full-precision)
            layer cache and returns the whole accumulated ``[b, h, s, d]`` K/V,
            which is exactly the tensor attention consumes. We quantize a *copy*
            of that return value; the internal cache stays BF16 so the reference
            path and the quantized path share identical history. This is the
            injection point transformers actually calls -- the old ``readout``
            method was never invoked, so the quantization never fired.
            """
            k, v = super().update(key_states, value_states, layer_idx,
                                  cache_kwargs)
            if self.bit_width >= quantizer.INT16_BITS:
                return k, v
            return (self._quant(k, self.granularity_key),
                    self._quant(v, self.granularity_value))

    return QuantizedDynamicCache()


class AttentionOutputRecorder:
    """Forward hooks that capture per-layer attention outputs o_T and probs.

    ``eps(T)`` is defined on the attention readout o_T, so a run registers this
    on both the FP and the quantized model, then diffs the captured outputs.
    ``T_eff`` comes from the attention probabilities, which is why eager
    attention is mandatory (fused kernels drop the probs).
    """

    def __init__(self):
        self.outputs: dict[int, "np.ndarray"] = {}
        self.attn: dict[int, "np.ndarray"] = {}
        self._handles = []

    def attach(self, model) -> "AttentionOutputRecorder":
        torch, _ = _require_torch()
        layer_idx = 0
        for name, module in model.named_modules():
            if name.endswith("self_attn"):
                idx = layer_idx
                layer_idx += 1

                def hook(mod, inp, out, _idx=idx):
                    # transformers self-attn returns (attn_output, attn_weights, ...)
                    if isinstance(out, tuple):
                        o = out[0]
                        w = out[1] if len(out) > 1 else None
                    else:
                        o, w = out, None
                    self.outputs[_idx] = o.detach().to(torch.float32).cpu().numpy()
                    if w is not None:
                        self.attn[_idx] = w.detach().to(torch.float32).cpu().numpy()

                self._handles.append(module.register_forward_hook(hook))
        return self

    def clear(self) -> None:
        self.outputs.clear()
        self.attn.clear()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
