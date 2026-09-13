#!/usr/bin/env python
"""Evaluate results/fit.json against the frozen predictions.

Emits a verdict per prediction: PASS, FALSIFIED, or INCONCLUSIVE. The pass and
falsifier thresholds are read from predictions/h1_predictions.json but the
comparison for each is coded explicitly here rather than parsed from the string,
so an ambiguous threshold cannot silently pass. Writes results/gate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "predictions" / "h1_predictions.json"

PASS, FALSIFIED, INCONCLUSIVE = "PASS", "FALSIFIED", "INCONCLUSIVE"


def _verdict(passed: bool, falsified: bool) -> str:
    if passed and not falsified:
        return PASS
    if falsified and not passed:
        return FALSIFIED
    return INCONCLUSIVE


def evaluate(fit: dict) -> dict:
    v: dict = {}

    # P1: alpha_tf <= 0.05 ; falsifier >= 0.20 with R2 >= 0.9
    a_tf = fit["alpha_tf"]["alpha"]
    r2_tf = fit["alpha_tf"]["r2"]
    v["P1"] = {"metric": a_tf, "r2": r2_tf,
              "verdict": _verdict(a_tf <= 0.05, a_tf >= 0.20 and r2_tf >= 0.9)}

    # P2: alpha_fr >= 0.20 and R2 >= 0.9 ; falsifier <= 0.05
    a_fr = fit["alpha_fr"]["alpha"]
    r2_fr = fit["alpha_fr"]["r2"]
    v["P2"] = {"metric": a_fr, "r2": r2_fr,
              "verdict": _verdict(a_fr >= 0.20 and r2_fr >= 0.9, a_fr <= 0.05)}

    # P3: gap >= 0.15 ; falsifier CI contains 0
    gap = fit["alpha_gap"]["value"]
    fr_ci = fit["alpha_gap"]["alpha_fr_ci"]
    tf_ci = fit["alpha_gap"]["alpha_tf_ci"]
    # Conservative CI on the gap: fr_lo - tf_hi .. fr_hi - tf_lo
    gap_lo = fr_ci[0] - tf_ci[1]
    gap_hi = fr_ci[1] - tf_ci[0]
    contains_zero = gap_lo <= 0.0 <= gap_hi
    v["P3"] = {"metric": gap, "gap_ci": [gap_lo, gap_hi],
              "verdict": _verdict(gap >= 0.15, contains_zero)}

    # P4: pearson r >= 0.8 ; falsifier <= 0.3
    r = fit["pearson_r_eps_invTeff_armA"]
    v["P4"] = {"metric": r, "verdict": _verdict(r >= 0.8, r <= 0.3)}

    # P5: measured within 2x of predicted ; falsifier monotonicity reversed
    if "G_ratio_early_late" in fit:
        g = fit["G_ratio_early_late"]
        ratio = g["measured"] / g["predicted"] if g["predicted"] else float("inf")
        within = 0.5 <= ratio <= 2.0
        v["P5"] = {"measured": g["measured"], "predicted": g["predicted"],
                  "monotonic": g["monotonic"],
                  "verdict": _verdict(within and g["monotonic"], not g["monotonic"])}
    else:
        v["P5"] = {"verdict": INCONCLUSIVE, "note": "arm C results absent"}

    # P6: db_per_bit in [-6.5, -5.5] for b>=4 ; falsifier outside [-8, -4]
    if "db_per_bit" in fit:
        s = fit["db_per_bit"]["db_per_bit"]
        v["P6"] = {"metric": s,
                  "verdict": _verdict(-6.5 <= s <= -5.5, not (-8.0 <= s <= -4.0))}
    else:
        v["P6"] = {"verdict": INCONCLUSIVE, "note": "bit-sweep results absent"}

    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", default=str(REPO / "results" / "fit.json"))
    args = ap.parse_args()

    fit_path = Path(args.fit)
    if not fit_path.exists():
        print(f"missing {fit_path}; run `mise run fit` first", file=sys.stderr)
        return 1

    fit = json.loads(fit_path.read_text())
    predictions = json.loads(PRED.read_text())
    verdicts = evaluate(fit)
    summary = {"predictions": predictions, "verdicts": verdicts,
               "n_pass": sum(1 for x in verdicts.values() if x["verdict"] == PASS),
               "n_falsified": sum(1 for x in verdicts.values() if x["verdict"] == FALSIFIED)}
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")

    for pid, x in verdicts.items():
        print(f"{pid}: {x['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
