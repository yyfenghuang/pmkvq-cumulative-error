#!/usr/bin/env python
"""Bit-sweep: terminal deviation against bit-width. Carries P6.

Sweeps b in {16, 8, 6, 4, 3, 2}. b >= 4 tests the 6.02 dB-per-bit law; b <= 3
probes the clipping departure.
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
    ap.add_argument("--t-min", type=int, default=256)
    ap.add_argument("--t-max", type=int, default=4096)
    ap.add_argument("--n-positions", type=int, default=24)
    args = ap.parse_args()

    cfg = run_experiment.RunConfig()
    cfg.t_positions = tuple(run_experiment.log_positions(args.t_min, args.t_max, args.n_positions))

    rows = run_experiment.bitsweep(cfg, load_prompts(args.prompts, cfg.n_sequences))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
