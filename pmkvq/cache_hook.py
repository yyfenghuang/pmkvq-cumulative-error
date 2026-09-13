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

The quantizer is the reimplemented Definition 1 in ``quantizer.py`` applied via
a numpy round-trip; keeping one floor definition is deliberate (P6 needs the
analytic reference).
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
            # tensor: [batch, heads, seq, head_dim]
            b, h, s, d = tensor.shape
            mask = injection_mask(s, self.preserve_front, self.preserve_back,
                                  self.window)
            arr = tensor.detach().to(torch.float32).cpu().numpy()
            if per == "channel":
                # group over the sequence axis per (head, channel): move seq last
                moved = np.moveaxis(arr, 2, -1)          # [b, h, d, s]
                # position mask applies along the last axis; expand quantizer
                # over each channel-group of the transposed layout
                flat = moved.reshape(-1, s)
                keep = ~mask
                qtd = quantizer.fake_quant(flat, self.bit_width,
                                           group_size=self.group_size, axis=-1)
                qtd[:, keep] = flat[:, keep]
                moved = qtd.reshape(moved.shape)
                out = np.moveaxis(moved, -1, 2)          # back to [b, h, s, d]
            else:  # per == "token": group over head_dim, mask over seq
                out = arr.copy()
                sel = mask
                if sel.any():
                    sub = arr[:, :, sel, :]              # [b, h, n_sel, d]
                    out[:, :, sel, :] = quantizer.fake_quant(
                        sub, self.bit_width, group_size=self.group_size, axis=-1)
            return torch.from_numpy(out.astype(np.float32)).to(
                dtype=tensor.dtype, device=tensor.device)

        def readout(self, layer_idx: int):
            """Return the (quantized) key/value tensors used for attention."""
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
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
