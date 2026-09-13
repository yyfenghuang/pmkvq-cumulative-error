#!/usr/bin/env python
"""Arm A: teacher forced, feedback off. Carries P1 and P4.

Writes per-position eps(T), T_eff(T), clipping rate, realised sigma^2, sequence
id, temperature to the CSV. Requires the synced environment and a checkpoint.
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
    ap.add_argument("--prompts", default=None, help="one prompt per line; else built-in default")
    ap.add_argument("--t-min", type=int, default=256)
    ap.add_argument("--t-max", type=int, default=4096)
    ap.add_argument("--n-positions", type=int, default=24)
    args = ap.parse_args()

    cfg = run_experiment.RunConfig()
    cfg.t_positions = tuple(run_experiment.log_positions(args.t_min, args.t_max, args.n_positions))
    prompts = load_prompts(args.prompts, cfg.n_sequences)

    rows = run_experiment.arm_a(cfg, prompts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
