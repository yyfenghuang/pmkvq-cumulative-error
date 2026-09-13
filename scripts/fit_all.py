#!/usr/bin/env python
"""P3 and the fit summary: log-log regression, bootstrap CI, gamma = sqrt(alpha+1).

Reads the four result CSVs, produces the metrics the gate consumes, and writes
results/fit.json. Every metric here maps to a prediction:

    alpha_tf                    P1   (Arm A slope)
    alpha_fr                    P2   (Arm B slope)
    alpha_fr - alpha_tf         P3   (with bootstrap CI on the gap)
    pearson_r_eps_invTeff_armA  P4
    G_ratio_early_late          P5   (Arm C, measured vs Theorem 2)
    db_per_bit                  P6   (bit-sweep)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from pmkvq import analysis  # noqa: E402


def _terminal_by_group(df: pd.DataFrame, group_col: str, value_col: str = "eps") -> pd.DataFrame:
    """Mean terminal (max-position) value per group level."""
    idx = df.groupby([group_col, "sequence_id"])["position"].idxmax()
    term = df.loc[idx]
    return term.groupby(group_col)[value_col].mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    args = ap.parse_args()
    rd = Path(args.results_dir)

    out: dict = {}

    # P1 / P2: per-arm log-log slope of eps against position, pooled over seqs.
    arm_a = pd.read_csv(rd / "arm_a.csv")
    arm_b = pd.read_csv(rd / "arm_b.csv")
    fit_a = analysis.loglog_fit(arm_a["position"], arm_a["eps"])
    fit_b = analysis.loglog_fit(arm_b["position"], arm_b["eps"])
    out["alpha_tf"] = fit_a.to_dict()
    out["alpha_fr"] = fit_b.to_dict()

    # P3: the gap and its bootstrap CI (resample the per-arm slopes' pairs).
    ci_a = analysis.bootstrap_ci(arm_a["position"], arm_a["eps"])
    ci_b = analysis.bootstrap_ci(arm_b["position"], arm_b["eps"])
    gap = fit_b.alpha - fit_a.alpha
    out["alpha_gap"] = {
        "value": float(gap),
        "alpha_fr_ci": [ci_b["alpha_lo"], ci_b["alpha_hi"]],
        "alpha_tf_ci": [ci_a["alpha_lo"], ci_a["alpha_hi"]],
        "gamma_fr": ci_b["gamma"],
    }

    # P4: Pearson r between eps and 1/T_eff in Arm A.
    a = arm_a.dropna(subset=["eps", "t_eff"])
    inv_teff = 1.0 / a["t_eff"].to_numpy()
    if len(a) >= 2:
        r, _ = stats.pearsonr(a["eps"].to_numpy(), inv_teff)
    else:
        r = float("nan")
    out["pearson_r_eps_invTeff_armA"] = float(r)

    # P5: Arm C Green's-function ratio, earliest vs latest t0, measured vs pred.
    arm_c_path = rd / "arm_c.csv"
    if arm_c_path.exists():
        arm_c = pd.read_csv(arm_c_path)
        term_c = _terminal_by_group(arm_c, "t0")
        t0s = np.sort(term_c.index.to_numpy())
        if len(t0s) >= 2:
            early, late = int(t0s[0]), int(t0s[-1])
            measured = float(term_c.loc[early] / term_c.loc[late])
            T = float(arm_c["position"].max())
            predicted = analysis.green_ratio(early, late, T, fit_b.alpha)
            out["G_ratio_early_late"] = {
                "measured": measured, "predicted": predicted,
                "t0_early": early, "t0_late": late,
                "monotonic": bool(np.all(np.diff(term_c.loc[t0s].to_numpy()) <= 0)),
            }

    # P6: dB per bit from the bit-sweep terminal deviation.
    bs_path = rd / "bitsweep.csv"
    if bs_path.exists():
        bs = pd.read_csv(bs_path)
        term_b = _terminal_by_group(bs, "bit_width")
        mask = term_b.index >= 4
        out["db_per_bit"] = analysis.db_per_bit_fit(
            term_b.index[mask].to_numpy(), term_b[mask].to_numpy())

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
