"""Fitting layer: loglog_fit, bootstrap_ci, recover_gamma, classify_outcome.

The paper's qualitative claim reduces to the sign of a single exponent alpha,
recovered as the slope of eps against T on log-log axes (thinkbook Section 6).
The gain is gamma = sqrt(alpha + 1).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    alpha: float          # slope of log10(eps) against log10(T)
    intercept: float      # log10 intercept
    r2: float             # coefficient of determination
    gamma: float          # sqrt(alpha + 1), the recovered feedback gain
    n: int                # number of points in the fit

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(T: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite and non-positive points before taking logs."""
    T = np.asarray(T, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(T) & np.isfinite(y) & (T > 0) & (y > 0)
    return T[m], y[m]


def loglog_fit(T: np.ndarray, eps: np.ndarray) -> FitResult:
    """Least-squares fit of log10(eps) = alpha log10(T) + intercept.

    Returns the slope alpha, the R^2, and the recovered gain gamma. A log-log
    plot of eps against T is linear with slope alpha; that is the whole
    measurement (Section 6, "Consequence for measurement").
    """
    Tc, yc = _clean(T, eps)
    if Tc.size < 2:
        return FitResult(np.nan, np.nan, np.nan, np.nan, int(Tc.size))
    logT = np.log10(Tc)
    logy = np.log10(yc)
    res = stats.linregress(logT, logy)
    alpha = float(res.slope)
    return FitResult(
        alpha=alpha,
        intercept=float(res.intercept),
        r2=float(res.rvalue ** 2),
        gamma=recover_gamma(alpha),
        n=int(Tc.size),
    )


def bootstrap_ci(T: np.ndarray, eps: np.ndarray, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap CI on alpha (and the implied gamma).

    Resamples the (log T, log eps) pairs with replacement. Returns the point
    estimate plus the lower/upper bounds at the requested confidence level.
    """
    Tc, yc = _clean(T, eps)
    point = loglog_fit(Tc, yc)
    if Tc.size < 2:
        return {"alpha": point.alpha, "alpha_lo": np.nan, "alpha_hi": np.nan,
                "gamma": point.gamma, "gamma_lo": np.nan, "gamma_hi": np.nan,
                "n_boot": 0}
    rng = np.random.default_rng(seed)
    logT = np.log10(Tc)
    logy = np.log10(yc)
    n = Tc.size
    alphas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        res = stats.linregress(logT[idx], logy[idx])
        alphas[i] = res.slope
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    a_lo, a_hi = np.quantile(alphas, [lo_q, hi_q])
    return {
        "alpha": point.alpha,
        "alpha_lo": float(a_lo),
        "alpha_hi": float(a_hi),
        "gamma": point.gamma,
        "gamma_lo": recover_gamma(float(a_lo)),
        "gamma_hi": recover_gamma(float(a_hi)),
        "r2": point.r2,
        "n_boot": n_boot,
    }


def recover_gamma(alpha: float) -> float:
    """gamma = sqrt(alpha + 1). Returns NaN for alpha <= -1 (outside the model)."""
    val = alpha + 1.0
    return float(np.sqrt(val)) if val >= 0 else float("nan")


def classify_outcome(alpha: float, tol: float = 0.05) -> str:
    """Map alpha to one of the three regimes of Corollary 3.

        alpha < -tol  -> "saturation"       (gamma < 1)
        |alpha| <= tol -> "marginal"        (gamma = 1, logarithmic)
        alpha > +tol  -> "power-law-growth" (gamma > 1)
    """
    if not np.isfinite(alpha):
        return "undetermined"
    if alpha > tol:
        return "power-law-growth"
    if alpha < -tol:
        return "saturation"
    return "marginal"


def db_per_bit_fit(b: np.ndarray, terminal_eps: np.ndarray) -> dict:
    """Slope of terminal deviation (in dB) against bit-width, for P6.

    ``terminal_eps`` is a linear power ratio; it is converted to dB internally.
    The high-resolution prediction is -6.02 dB per bit (Corollary 1).
    """
    b = np.asarray(b, dtype=np.float64)
    eps_lin = np.asarray(terminal_eps, dtype=np.float64)
    m = np.isfinite(b) & np.isfinite(eps_lin) & (eps_lin > 0)
    b, eps_lin = b[m], eps_lin[m]
    if b.size < 2:
        return {"db_per_bit": np.nan, "r2": np.nan, "n": int(b.size)}
    db = 10.0 * np.log10(eps_lin)
    res = stats.linregress(b, db)
    return {"db_per_bit": float(res.slope), "r2": float(res.rvalue ** 2),
            "n": int(b.size)}


def green_ratio(t0_early: float, t0_late: float, T: float, alpha: float) -> float:
    """Predicted G(T, t0_early) / G(T, t0_late) from Theorem 2.

        G(T, t0) = c0 gamma^2 / t0 * (T / t0)^alpha

    The c0 gamma^2 prefactor cancels in the ratio, leaving a parameter-free
    prediction (P5).
    """
    early = (1.0 / t0_early) * (T / t0_early) ** alpha
    late = (1.0 / t0_late) * (T / t0_late) ** alpha
    return float(early / late)
