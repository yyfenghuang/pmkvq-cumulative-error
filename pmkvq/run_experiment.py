"""The experiment: decode_loop and the four arms.

The contrast between Arm A (teacher forced, feedback off) and Arm B (free
running, feedback on) *is* the experiment; everything else is held fixed. The
feedback switch lives in ``decode_loop`` and is deliberately a single visible
branch so ``show_source(decode_loop)`` in the notebook makes it legible.

This module drives a checkpoint and therefore requires torch, transformers, and
a Qwen3-0.6B model. Heavy imports are lazy so the module imports cleanly for
introspection (``show_source``) without a GPU present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from pmkvq import cache_hook
from pmkvq import observables
from pmkvq import quantizer

# Deterministic cuBLAS requires this to be set before the first CUDA/cuBLAS
# call, otherwise torch.use_deterministic_algorithms(True) raises on GPU
# ("CUBLAS_WORKSPACE_CONFIG"). Harmless on CPU. Set at import so it lands
# before any torch/CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class RunConfig:
    checkpoint: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    group_size: int = quantizer.GROUP_SIZE
    granularity_key: str = "channel"
    granularity_value: str = "token"
    preserve_front: int = cache_hook.PRESERVE_FRONT
    preserve_back: int = cache_hook.PRESERVE_BACK
    max_new_tokens: int = 4096
    temperature: float = 0.0            # recorded in every CSV row
    n_sequences: int = 16               # N >= 16
    bit_widths: tuple[int, ...] = (16, 8, 6, 4, 3, 2)
    t_positions: tuple[int, ...] = field(default_factory=tuple)  # log-spaced, filled by scripts
    seed: int = 0
    device: str | None = None           # None -> auto: cuda if available, else cpu


def log_positions(t_min: int, t_max: int, n: int = 24) -> np.ndarray:
    """Log-spaced integer decode positions over at least two decades.

    A power law is best sampled logarithmically; linear sampling wastes
    resolution at small T (thinkbook Section 10, "Positions").
    """
    pts = np.unique(np.round(np.logspace(np.log10(t_min), np.log10(t_max), n)))
    return pts.astype(int)


def _resolve_device(cfg: RunConfig) -> str:
    """The run device: ``cfg.device`` if set, else cuda when available."""
    import torch

    if cfg.device:
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(cfg: RunConfig):
    """Load the checkpoint with eager attention on the resolved device. Lazy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    device = _resolve_device(cfg)
    tok = AutoTokenizer.from_pretrained(cfg.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.checkpoint,
        dtype=getattr(torch, cfg.dtype),
        attn_implementation="eager",   # mandatory: exposes attention probs
    )
    model.to(device)
    model.eval()
    return model, tok


def decode_loop(model, tok, input_ids, cfg: RunConfig, *, feedback: bool,
                bit_width: int, window=None, seed: int = 0):
    """Decode ``cfg.max_new_tokens`` steps, recording eps(T) and T_eff(T).

    The reference (BF16) path and the quantized path are advanced in lockstep.

    THE FEEDBACK SWITCH
    -------------------
    ``feedback=False`` (Arm A, teacher forced): the next input token is taken
    from the *reference* sequence, so the generated token never depends on the
    perturbed output. The readout term of Proposition 1 is isolated.

    ``feedback=True`` (Arm B, free running): the next input token is the
    quantized model's own argmax/sample, so error re-enters as signal. This is
    the autoregressive loop of Section 6.

    Returns a list of per-position row dicts.
    """
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)     # prompt onto the model's device
    # One recorder, two passes per step. It overwrites its buffers on each
    # forward, so we snapshot after the FP pass, then again after the quantized
    # pass, and diff the two snapshots. Keeping a single recorder avoids both
    # hooks firing on the same forward.
    rec = cache_hook.AttentionOutputRecorder().attach(model)

    fp_cache = _new_cache(bit_width=quantizer.INT16_BITS, cfg=cfg)  # BF16 reference
    q_cache = cache_hook.make_quantized_cache(
        bit_width=bit_width, granularity_key=cfg.granularity_key,
        granularity_value=cfg.granularity_value, group_size=cfg.group_size,
        preserve_front=cfg.preserve_front, preserve_back=cfg.preserve_back,
        window=window,
    )

    rows = []
    cur_fp = input_ids
    cur_q = input_ids
    positions = set(int(t) for t in cfg.t_positions) if cfg.t_positions else None

    with torch.no_grad():
        for step in range(cfg.max_new_tokens):
            T = step + input_ids.shape[1]

            rec.clear()
            out_fp = model(cur_fp, past_key_values=fp_cache, use_cache=True,
                           output_attentions=True)
            o_fp = dict(rec.outputs)          # snapshot FP attention outputs
            a_fp = dict(rec.attn)             # snapshot FP attention probs

            rec.clear()
            out_q = model(cur_q, past_key_values=q_cache, use_cache=True,
                          output_attentions=True)
            o_q = dict(rec.outputs)           # snapshot quantized outputs

            if positions is None or T in positions:
                rows.append(_row_from_capture(o_fp, o_q, a_fp, T, bit_width,
                                              feedback, cfg, seed))

            fp_tok = out_fp.logits[:, -1].argmax(-1, keepdim=True)
            q_tok = out_q.logits[:, -1].argmax(-1, keepdim=True)

            # ---- the switch ----
            if feedback:
                cur_q = q_tok           # Arm B: perturbed output feeds back
            else:
                cur_q = fp_tok          # Arm A: teacher forced by the reference
            cur_fp = fp_tok

            if fp_tok.item() == tok.eos_token_id:
                break

    rec.detach()
    return rows


def _new_cache(bit_width: int, cfg: RunConfig):
    return cache_hook.make_quantized_cache(
        bit_width=bit_width, granularity_key=cfg.granularity_key,
        granularity_value=cfg.granularity_value, group_size=cfg.group_size,
        preserve_front=cfg.preserve_front, preserve_back=cfg.preserve_back,
    )


def _row_from_capture(o_fp, o_q, a_fp, T, bit_width, feedback, cfg, seed):
    """Assemble a per-position record from the captured attention outputs.

    ``eps`` is averaged over layers at the final query position; ``t_eff`` is
    averaged over layers and heads from the FP attention probabilities of the
    final query row (1 / ||a_T||^2).
    """
    eps_layers = []
    teff_layers = []
    for idx in o_fp:
        if idx not in o_q:
            continue
        eps_layers.append(observables.eps(_last_pos(o_q[idx]),
                                          _last_pos(o_fp[idx]), axis=-1))
        if idx in a_fp:
            # a_fp[idx]: [batch, heads, q_len, kv_len]; take the last query row.
            last_row = a_fp[idx][..., -1, :]
            teff_layers.append(observables.t_eff(last_row, axis=-1))
    eps_val = float(np.mean([np.mean(e) for e in eps_layers])) if eps_layers else float("nan")
    teff_val = float(np.mean([np.mean(t) for t in teff_layers])) if teff_layers else float("nan")
    return {
        "position": int(T),
        "eps": eps_val,
        "t_eff": teff_val,
        "bit_width": int(bit_width),
        "feedback": bool(feedback),
        "temperature": cfg.temperature,
        "seed": int(seed),
    }


def _last_pos(a: np.ndarray) -> np.ndarray:
    """Final query position's feature vector, shape [..., d]."""
    return a[:, -1, :] if a.ndim == 3 else a[..., -1, :]


# --------------------------------------------------------------------------- #
# Arms. Each returns a list of row dicts; the scripts assemble them into a
# DataFrame and write the CSV.
# --------------------------------------------------------------------------- #

def arm_a(cfg: RunConfig, prompts):
    """Teacher forced, feedback off. Carries P1 and P4."""
    model, tok = _load_model(cfg)
    rows = []
    for sid, prompt in enumerate(prompts[: cfg.n_sequences]):
        ids = tok(prompt, return_tensors="pt").input_ids
        for r in decode_loop(model, tok, ids, cfg, feedback=False,
                             bit_width=_probe_bit(cfg), seed=cfg.seed + sid):
            r["sequence_id"] = sid
            r["arm"] = "A"
            rows.append(r)
    return rows


def arm_b(cfg: RunConfig, prompts):
    """Free running, feedback on. Carries P2."""
    model, tok = _load_model(cfg)
    rows = []
    for sid, prompt in enumerate(prompts[: cfg.n_sequences]):
        ids = tok(prompt, return_tensors="pt").input_ids
        for r in decode_loop(model, tok, ids, cfg, feedback=True,
                             bit_width=_probe_bit(cfg), seed=cfg.seed + sid):
            r["sequence_id"] = sid
            r["arm"] = "B"
            rows.append(r)
    return rows


def arm_c(cfg: RunConfig, prompts, t0_grid, window_width: int = 256):
    """Impulse injection, sweep t0. Carries P5.

    Quantize only ``[t0, t0 + window_width)`` and hold everything else at BF16,
    measuring the Green's function G(T, t0) directly without the uniform-mixing
    approximation of Theorem 1.
    """
    model, tok = _load_model(cfg)
    rows = []
    for sid, prompt in enumerate(prompts[: cfg.n_sequences]):
        ids = tok(prompt, return_tensors="pt").input_ids
        for t0 in t0_grid:
            window = (int(t0), int(t0) + window_width)
            for r in decode_loop(model, tok, ids, cfg, feedback=False,
                                 bit_width=_probe_bit(cfg), window=window,
                                 seed=cfg.seed + sid):
                r["sequence_id"] = sid
                r["arm"] = "C"
                r["t0"] = int(t0)
                r["window_width"] = window_width
                rows.append(r)
    return rows


def bitsweep(cfg: RunConfig, prompts):
    """Terminal deviation against bit-width. Carries P6."""
    model, tok = _load_model(cfg)
    rows = []
    for sid, prompt in enumerate(prompts[: cfg.n_sequences]):
        ids = tok(prompt, return_tensors="pt").input_ids
        for b in cfg.bit_widths:
            for r in decode_loop(model, tok, ids, cfg, feedback=False,
                                 bit_width=b, seed=cfg.seed + sid):
                r["sequence_id"] = sid
                r["arm"] = "bitsweep"
                rows.append(r)
    return rows


def _probe_bit(cfg: RunConfig) -> int:
    """The working bit-width for the growth arms (not the sweep)."""
    return 2 if 2 in cfg.bit_widths else min(cfg.bit_widths)
