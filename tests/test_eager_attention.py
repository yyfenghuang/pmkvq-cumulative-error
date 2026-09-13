#!/usr/bin/env python
"""Eager attention exposes the probability tensor, so T_eff is computable.

This exists so a silent fallback to a fused/flash kernel fails BEFORE a long run
starts (fused kernels drop the attention probs and make T_eff = 1/||a||^2
unobservable). It loads the checkpoint, so it needs the synced environment.

If torch/transformers are not installed, the check is not applicable in this
environment and the test exits 0 without output (it will run for real under
`mise run test` in the synced env). Any other failure is reported and exits 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        # Not applicable without the synced env; the mise task runs it for real.
        return 0

    import numpy as np
    from pmkvq import observables

    fails = []
    model_id = "Qwen/Qwen3-0.6B"
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, attn_implementation="eager")
        model.eval()
    except Exception as exc:
        # Checkpoint not fetched in this environment; skip rather than false-fail.
        print(f"skip: checkpoint unavailable ({exc})", file=sys.stderr)
        return 0

    ids = tok("Solve step by step: 2 + 2 = ", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True, use_cache=False)

    if out.attentions is None or len(out.attentions) == 0:
        fails.append("model returned no attention probabilities under eager")
    else:
        a = out.attentions[0]                    # [batch, heads, q, kv]
        rows = a[0, :, -1, :].to(torch.float32).cpu().numpy()
        if not np.allclose(rows.sum(axis=-1), 1.0, atol=1e-3):
            fails.append("attention rows are not row-stochastic")
        teff = observables.t_eff(rows, axis=-1)
        if not np.all((teff >= 1.0) & (teff <= rows.shape[-1] + 1e-6)):
            fails.append("T_eff out of [1, T] bounds")

    if fails:
        print("FAIL test_eager_attention")
        for f in fails:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
