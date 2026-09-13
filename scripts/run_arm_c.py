#!/usr/bin/env python
"""Arm C: impulse injection, sweep t0. Carries P5.

Quantize only a window [t0, t0 + w] and hold every other position in BF16,
measuring the Green's function G(T, t0) directly without the uniform-mixing
approximation of Theorem 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd  # noqa: E402

from pmkvq import run_experiment  # noqa: E402
from _prompts import load_prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--t0-min", type=int, default=256)
    ap.add_argument("--t0-max", type=int, default=3072)
    ap.add_argument("--n-t0", type=int, default=12)
    ap.add_argument("--window", type=int, default=256, help="impulse window width w")
    args = ap.parse_args()

    cfg = run_experiment.RunConfig()
    t0_grid = run_experiment.log_positions(args.t0_min, args.t0_max, args.n_t0)

    rows = run_experiment.arm_c(cfg, load_prompts(args.prompts, cfg.n_sequences),
                                t0_grid=t0_grid, window_width=args.window)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
