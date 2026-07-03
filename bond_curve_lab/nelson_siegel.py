"""
Nelson-Siegel / Nelson-Siegel-Svensson Yield Curve Models
==========================================================

Implements:
- Nelson-Siegel (NS) 4-parameter model: level, slope, curvature, decay
- Nelson-Siegel-Svensson (NSS) 6-parameter model: adds hump components
- Forward rate curve derivation
- Discount factor computation
- Parameter interpretation

References
----------
- Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves."
  Journal of Business, 60(4), 473-489.
- Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates:
  Sweden 1992-1994." NBER Working Paper No. 4871.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Nelson-Siegel core functions
# ---------------------------------------------------------------------------

def _ns_yield(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """
    Nelson-Siegel yield function.

    y(tau) = beta0 + beta1 * ((1 - exp(-tau/lambda)) / (tau/lambda))
                   + beta2 * ((1 - exp(-tau/lambda)) / (tau/lambda) - exp(-tau/lambda))

    Parameters
    ----------
    tau : ndarray
        Maturities in years.
    beta0 : float
        Level factor (long-term rate).
    beta1 : float
        Slope factor (negative = upward sloping).
    beta2 : float
        Curvature factor.
    lam : float
        Decay parameter (shape of hump).

    Returns
    -------
    y : ndarray
        Fitted yields.
    """
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_div_lam = tau / lam
        # Handle tau -> 0 limit
        exp_neg = np.exp(-t_div_lam)
        # (1 - exp(-x)) / x  with limit 1 as x->0
        factor1 = np.where(t_div_lam < 1e-8, 1.0, (1.0 - exp_neg) / t_div_lam)
        factor2 = factor1 - exp_neg
    return beta0 + beta1 * factor1 + beta2 * factor2


def _nss_yield(tau: np.ndarray, beta0: float, beta1: float, beta2: float,
               beta3: float, lam1: float, lam2: float) -> np.ndarray:
    """
    Nelson-Siegel-Svensson yield function.

    Adds a second hump term: beta3 * ((1-exp(-tau/lam2))/(tau/lam2) - exp(-tau/lam2))
    """
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = tau / lam1
        t2 = tau / lam2
        e1 = np.exp(-t1)
        e2 = np.exp(-t2)
        f1_1 = np.where(t1 < 1e-8, 1.0, (1.0 - e1) / t1)
        f2_1 = f1_1 - e1
        f1_2 = np.where(t2 < 1e-8, 1.0, (1.0 - e2) / t2)
        f2_2 = f1_2 - e2
    return beta0 + beta1 * f1_1 + beta2 * f2_1 + beta3 * f2_2


# ---------------------------------------------------------------------------
# Public fitting API
# ---------------------------------------------------------------------------

def fit_nelson_siegel(
    maturities: np.ndarray,
    yields: np.ndarray,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
) -> Dict:
    """
    Fit Nelson-Siegel model to yield curve data.

    Parameters
    ----------
    maturities : ndarray
        Maturities in years.
    yields : ndarray
        Observed yields in percent.
    initial_guess : tuple, optional
        (beta0, beta1, beta2, lambda) starting values.

    Returns
    -------
    dict with keys:
        model : str ('Nelson-Siegel')
        params : list [beta0, beta1, beta2, lambda]
        pcov : covariance matrix or None
        fitted_yields : ndarray
        residuals : ndarray
        r_squared : float
        rmse : float
        formula : str
    """
    maturities = np.asarray(maturities, dtype=float)
    yields_arr = np.asarray(yields, dtype=float)

    # Remove NaNs
    mask = ~np.isnan(yields_arr)
    tau = maturities[mask]
    y = yields_arr[mask]

    if len(tau) < 4:
        raise ValueError("Need at least 4 data points for Nelson-Siegel fit (4 parameters).")

    if initial_guess is None:
        # Sensible defaults for CGB
        lr = y[-1]  # long rate as initial level
        sr = y[0]   # short rate
        initial_guess = (lr, sr - lr, 0.5, 1.5)

    bounds = (
        [-5.0, -10.0, -10.0, 0.05],    # lower bounds
        [15.0, 10.0, 10.0, 10.0],      # upper bounds
    )

    try:
        popt, pcov = curve_fit(
            _ns_yield, tau, y,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000,
        )
    except RuntimeError as e:
        raise RuntimeError(f"Nelson-Siegel fitting failed: {e}")

    fitted = _ns_yield(maturities, *popt)
    residuals = yields_arr - fitted
    ss_res = np.nansum(residuals ** 2)
    ss_tot = np.nansum((yields_arr - np.nanmean(yields_arr)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.nanmean(residuals ** 2))

    beta0, beta1, beta2, lam = popt

    formula = (
        f"y(τ) = {beta0:.4f} + {beta1:.4f}·[(1-e^(-τ/{lam:.4f}))/(τ/{lam:.4f})] "
        f"+ {beta2:.4f}·[(1-e^(-τ/{lam:.4f}))/(τ/{lam:.4f}) - e^(-τ/{lam:.4f})]"
    )

    return {
        "model": "Nelson-Siegel",
        "params": list(popt),
        "param_names": ["beta0", "beta1", "beta2", "lambda"],
        "pcov": pcov.tolist(),
        "fitted_yields": fitted.tolist(),
        "residuals": residuals.tolist(),
        "r_squared": r_squared,
        "rmse": rmse,
        "formula": formula,
    }


def fit_nelson_siegel_svensson(
    maturities: np.ndarray,
    yields: np.ndarray,
    initial_guess: Optional[Tuple[float, ...]] = None,
) -> Dict:
    """
    Fit Nelson-Siegel-Svensson model to yield curve data.

    Parameters
    ----------
    maturities : ndarray
        Maturities in years.
    yields : ndarray
        Observed yields in percent.
    initial_guess : tuple, optional
        (beta0, beta1, beta2, beta3, lambda1, lambda2) starting values.

    Returns
    -------
    dict with same structure as fit_nelson_siegel plus 'model': 'Nelson-Siegel-Svensson'.
    """
    maturities = np.asarray(maturities, dtype=float)
    yields_arr = np.asarray(yields, dtype=float)

    mask = ~np.isnan(yields_arr)
    tau = maturities[mask]
    y = yields_arr[mask]

    if len(tau) < 6:
        raise ValueError("Need at least 6 data points for NSS fit (6 parameters).")

    if initial_guess is None:
        lr = y[-1]
        sr = y[0]
        initial_guess = (lr, sr - lr, 0.5, -0.3, 1.5, 5.0)

    bounds = (
        [-5.0, -10.0, -10.0, -10.0, 0.05, 0.05],
        [15.0, 10.0, 10.0, 10.0, 10.0, 20.0],
    )

    try:
        popt, pcov = curve_fit(
            _nss_yield, tau, y,
            p0=initial_guess,
            bounds=bounds,
            maxfev=20000,
        )
    except RuntimeError as e:
        raise RuntimeError(f"NSS fitting failed: {e}")

    fitted = _nss_yield(maturities, *popt)
    residuals = yields_arr - fitted
    ss_res = np.nansum(residuals ** 2)
    ss_tot = np.nansum((yields_arr - np.nanmean(yields_arr)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.nanmean(residuals ** 2))

    beta0, beta1, beta2, beta3, lam1, lam2 = popt

    formula = (
        f"y(τ) = {beta0:.4f} + {beta1:.4f}·SL1(τ,{lam1:.4f}) "
        f"+ {beta2:.4f}·CU1(τ,{lam1:.4f}) + {beta3:.4f}·CU2(τ,{lam2:.4f})"
    )

    return {
        "model": "Nelson-Siegel-Svensson",
        "params": list(popt),
        "param_names": ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"],
        "pcov": pcov.tolist(),
        "fitted_yields": fitted.tolist(),
        "residuals": residuals.tolist(),
        "r_squared": r_squared,
        "rmse": rmse,
        "formula": formula,
    }


# ---------------------------------------------------------------------------
# Derived curves
# ---------------------------------------------------------------------------

def forward_rate_curve(params: list, tenors: np.ndarray, model: str = "ns") -> np.ndarray:
    """
    Derive instantaneous forward rate curve from model parameters.

    NS forward: f(τ) = β₀ + β₁·e^(-τ/λ) + β₂·(τ/λ)·e^(-τ/λ)

    NSS forward adds: β₃·(τ/λ₂)·e^(-τ/λ₂)
    """
    tenors = np.asarray(tenors, dtype=float)

    if model.lower() in ("ns", "nelson-siegel"):
        beta0, beta1, beta2, lam = params
        e = np.exp(-tenors / lam)
        fwd = beta0 + beta1 * e + beta2 * (tenors / lam) * e

    elif model.lower() in ("nss", "nelson-siegel-svensson", "svensson"):
        beta0, beta1, beta2, beta3, lam1, lam2 = params
        e1 = np.exp(-tenors / lam1)
        e2 = np.exp(-tenors / lam2)
        fwd = (
            beta0
            + beta1 * e1
            + beta2 * (tenors / lam1) * e1
            + beta3 * (tenors / lam2) * e2
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return fwd


def discount_factors(params: list, tenors: np.ndarray, model: str = "ns") -> np.ndarray:
    """
    Compute discount factors from model parameters.

    DF(τ) = exp(-y(τ) * τ / 100)
    """
    tenors = np.asarray(tenors, dtype=float)

    if model.lower() in ("ns", "nelson-siegel"):
        spot = _ns_yield(tenors, *params)
    elif model.lower() in ("nss", "nelson-siegel-svensson", "svensson"):
        spot = _nss_yield(tenors, *params)
    else:
        raise ValueError(f"Unknown model: {model}")

    return np.exp(-spot * tenors / 100.0)


# ---------------------------------------------------------------------------
# Parameter interpretation
# ---------------------------------------------------------------------------

def interpret_params(params: list, model: str = "ns") -> Dict[str, str]:
    """
    Provide human-readable interpretation of Nelson-Siegel parameters.

    Parameters
    ----------
    params : list
        Model parameters.
    model : str
        'ns' or 'nss'.

    Returns
    -------
    dict
        Keyed interpretation strings.
    """
    if model.lower() in ("ns", "nelson-siegel"):
        beta0, beta1, beta2, lam = params
        interpretations = {
            "Long-term level (β₀)": (
                f"{beta0:.4f}% — Long-term equilibrium rate. "
                + ("Above historical average." if beta0 > 3.5 else "Moderate long-term level.")
            ),
            "Short-term slope (β₁)": (
                f"{beta1:.4f} — "
                + ("Curve is upward-sloping (normal)." if beta1 < 0
                   else "Curve is downward-sloping (inverted)." if beta1 > 0.2
                   else "Curve is relatively flat.")
            ),
            "Curvature (β₂)": (
                f"{beta2:.4f} — "
                + ("Significant hump in the medium-term segment." if abs(beta2) > 1.0
                   else "Moderate curvature in the medium-term." if abs(beta2) > 0.3
                   else "Little curvature; curve is fairly smooth.")
            ),
            "Decay parameter (λ)": (
                f"{lam:.4f} years — "
                + ("Hump/curvature peaks around 2-3 year maturity." if lam < 3.0
                   else "Hump/curvature peaks around 5+ year maturity." if lam > 4.0
                   else "Hump/curvature peaks at medium-term maturity.")
            ),
            "Instantaneous short rate": (
                f"{beta0 + beta1:.4f}% — Implied overnight rate from model."
            ),
        }
    elif model.lower() in ("nss", "nelson-siegel-svensson", "svensson"):
        beta0, beta1, beta2, beta3, lam1, lam2 = params
        interpretations = {
            "Long-term level (β₀)": f"{beta0:.4f}% — Long-term equilibrium rate.",
            "Short-term slope (β₁)": f"{beta1:.4f} — {'Upward-sloping' if beta1 < 0 else 'Downward-sloping'} curve.",
            "First curvature (β₂)": f"{beta2:.4f} — Primary curvature (medium-term hump).",
            "Second curvature (β₃)": f"{beta3:.4f} — Secondary curvature (long-end hump).",
            "First decay (λ₁)": f"{lam1:.4f} — Decay for first hump.",
            "Second decay (λ₂)": f"{lam2:.4f} — Decay for second hump (typically larger).",
            "Instantaneous short rate": f"{beta0 + beta1:.4f}% — Implied overnight rate.",
        }
    else:
        raise ValueError(f"Unknown model: {model}")

    return interpretations


# ---------------------------------------------------------------------------
# Batch fitting over time
# ---------------------------------------------------------------------------

def fit_ns_over_time(
    dates: list,
    yields_matrix: np.ndarray,
    maturities: np.ndarray,
) -> Dict[str, list]:
    """
    Fit Nelson-Siegel model to each date in a historical panel.

    Returns
    -------
    dict with keys: dates, betas (list of [beta0, beta1, beta2, lambda]),
    r_squared, rmse, fitted_matrix
    """
    betas = []
    r2_list = []
    rmse_list = []
    fitted_matrix = []

    for i, date_str in enumerate(dates):
        y = yields_matrix[i]
        try:
            result = fit_nelson_siegel(maturities, y)
            betas.append(result["params"])
            r2_list.append(result["r_squared"])
            rmse_list.append(result["rmse"])
            fitted_matrix.append(result["fitted_yields"])
        except Exception:
            betas.append([np.nan] * 4)
            r2_list.append(np.nan)
            rmse_list.append(np.nan)
            fitted_matrix.append([np.nan] * len(maturities))

    return {
        "dates": dates,
        "betas": betas,
        "r_squared": r2_list,
        "rmse": rmse_list,
        "fitted_matrix": fitted_matrix,
    }
