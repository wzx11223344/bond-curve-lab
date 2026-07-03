"""
Bond Market Indicators
======================

Computes key fixed-income indicators from yield curve data:
- Term premium (10Y-2Y spread) — classic recession signal
- Slope steepness — overall curve slope metric
- Curvature (butterfly spread)
- Real yield estimate
- Credit spread proxy
"""

import numpy as np
from typing import List, Optional


def term_premium(yields_10y: np.ndarray, yields_2y: np.ndarray) -> np.ndarray:
    """
    Compute 10Y-2Y term spread.

    Parameters
    ----------
    yields_10y : ndarray
        10-year yields (percent), can be scalar or vector.
    yields_2y : ndarray
        2-year yields (percent), same shape as yields_10y.

    Returns
    -------
    ndarray
        10Y - 2Y spread in basis points (multiplied by 100 for bp).
    """
    return np.asarray(yields_10y) - np.asarray(yields_2y)


def slope_steepness(yields_long: np.ndarray, yields_short: np.ndarray) -> np.ndarray:
    """
    Overall curve slope: long-end yield minus short-end yield.

    Parameters
    ----------
    yields_long : ndarray
        Long-end yields (e.g., 30Y).
    yields_short : ndarray
        Short-end yields (e.g., 3M or 6M).

    Returns
    -------
    ndarray
        Slope in percent.
    """
    return np.asarray(yields_long) - np.asarray(yields_short)


def curvature(
    yields_5y: Optional[np.ndarray] = None,
    yields_10y: Optional[np.ndarray] = None,
    yields_2y: Optional[np.ndarray] = None,
    yields: Optional[np.ndarray] = None,
    idx_short: int = 3,   # 2Y
    idx_mid: int = 5,     # 5Y
    idx_long: int = 7,    # 10Y
) -> np.ndarray:
    """
    Butterfly spread: 2 * mid_yield - short_yield - long_yield.

    A classic curvature metric. Positive = more curvature (hump),
    negative = less curvature (saddle).

    Can be called either with explicit yields or with a full yield vector + indices.
    """
    if yields_5y is not None and yields_10y is not None and yields_2y is not None:
        return 2.0 * np.asarray(yields_5y) - np.asarray(yields_2y) - np.asarray(yields_10y)
    elif yields is not None:
        y = np.asarray(yields)
        if y.ndim == 1:
            return 2.0 * y[idx_mid] - y[idx_short] - y[idx_long]
        else:
            return 2.0 * y[:, idx_mid] - y[:, idx_short] - y[:, idx_long]
    else:
        raise ValueError("Provide either explicit yields or a yields array with indices.")


def real_yield(
    nominal_yield: float,
    cpi_yoy: float,
) -> float:
    """
    Approximate real yield using Fisher equation.

    real_yield ≈ (1 + nominal) / (1 + inflation) - 1

    Parameters
    ----------
    nominal_yield : float
        Nominal yield in percent (e.g., 2.5 for 2.5%).
    cpi_yoy : float
        CPI year-over-year in percent.

    Returns
    -------
    float
        Approximate real yield in percent.
    """
    nom = nominal_yield / 100.0
    cpi = cpi_yoy / 100.0
    real = (1.0 + nom) / (1.0 + cpi) - 1.0
    return real * 100.0


def credit_spread(
    aaa_yield: float,
    treasury_yield: float,
) -> float:
    """
    Approximate credit spread: AAA corporate yield minus CGB yield.

    Parameters
    ----------
    aaa_yield : float
        AAA-rated corporate bond yield (percent).
    treasury_yield : float
        CGB yield of comparable maturity (percent).

    Returns
    -------
    float
        Credit spread in percent.
    """
    return aaa_yield - treasury_yield


def compute_all_indicators(
    maturities: np.ndarray,
    yields: np.ndarray,
    cpi: Optional[float] = None,
) -> dict:
    """
    Compute all standard indicators from a single yield curve snapshot.

    Returns a dict with descriptive keys.
    """
    # Map maturities to indices
    def _idx(target: float) -> int:
        return int(np.argmin(np.abs(np.asarray(maturities) - target)))

    i_3m = _idx(0.25)
    i_6m = _idx(0.5)
    i_2y = _idx(2.0)
    i_5y = _idx(5.0)
    i_10y = _idx(10.0)
    i_30y = _idx(30.0)

    y = np.asarray(yields)

    indicators = {
        "date": None,  # filled by caller
        "term_premium_10y2y_bp": float(term_premium(y[i_10y], y[i_2y]) * 100),
        "slope_30y3m_bp": float(slope_steepness(y[i_30y], y[i_3m]) * 100),
        "curvature_2y5y10y_bp": float(curvature(
            yields_5y=y[i_5y], yields_10y=y[i_10y], yields_2y=y[i_2y]
        ) * 100),
        "yield_3m": float(y[i_3m]),
        "yield_2y": float(y[i_2y]),
        "yield_5y": float(y[i_5y]),
        "yield_10y": float(y[i_10y]),
        "yield_30y": float(y[i_30y]),
    }

    if cpi is not None:
        indicators["real_yield_10y"] = float(real_yield(y[i_10y], cpi))

    # Inversion warning
    tp = indicators["term_premium_10y2y_bp"]
    if tp < -10:
        indicators["signal"] = "STRONG INVERSION — historically precedes recession"
    elif tp < 0:
        indicators["signal"] = "Mild inversion — monitor closely"
    elif tp < 50:
        indicators["signal"] = "Flattening — reduced term premium"
    else:
        indicators["signal"] = "Normal steep curve"

    return indicators


def compute_historical_indicators(
    dates: List[str],
    yields_matrix: np.ndarray,
    maturities: np.ndarray,
) -> dict:
    """
    Compute time-series of standard indicators.

    Returns dict with lists for each indicator.
    """
    def _idx(target: float) -> int:
        return int(np.argmin(np.abs(np.asarray(maturities) - target)))

    i_3m = _idx(0.25)
    i_2y = _idx(2.0)
    i_5y = _idx(5.0)
    i_10y = _idx(10.0)
    i_30y = _idx(30.0)

    tp_bp = (yields_matrix[:, i_10y] - yields_matrix[:, i_2y]) * 100
    slope_bp = (yields_matrix[:, i_30y] - yields_matrix[:, i_3m]) * 100
    curv_bp = (2 * yields_matrix[:, i_5y] - yields_matrix[:, i_2y] - yields_matrix[:, i_10y]) * 100

    return {
        "dates": dates,
        "term_premium_10y2y_bp": tp_bp.tolist(),
        "slope_30y3m_bp": slope_bp.tolist(),
        "curvature_2y5y10y_bp": curv_bp.tolist(),
        "yield_10y": yields_matrix[:, i_10y].tolist(),
        "yield_2y": yields_matrix[:, i_2y].tolist(),
    }
