"""
China Government Bond Yield Curve Data Fetcher
==============================================

Uses akshare to fetch CGB (China Government Bond) yield curve data.
Supports real-time spot yields and historical time-series for curve analysis.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard CGB key maturities (in years)
TENORS = [3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]

TENOR_LABELS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

CACHE_DIR = Path(__file__).resolve().parent.parent / "output"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"cache_{key}.json"


def _read_cache(key: str, ttl_seconds: int = 3600) -> Optional[dict]:
    """Read cached data if it exists and is within TTL."""
    p = _cache_path(key)
    if not p.exists():
        return None
    stale = time.time() - os.path.getmtime(p) > ttl_seconds
    if stale:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def date_str(d: Optional[datetime] = None) -> str:
    """Return YYYY-MM-DD string."""
    return (d or datetime.now()).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# ChinaBondFetcher
# ---------------------------------------------------------------------------

class ChinaBondFetcher:
    """
    Fetches CGB yield data from akshare.

    Primary data sources:
    - ``akshare.bond_china_close_return`` — daily CGB closing yields
    - Spot yields at standard maturities (3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y)
    """

    CACHE_TTL_SECONDS = 3600 * 6  # 6 hours

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    # ------------------------------------------------------------------
    # Spot yields — single date
    # ------------------------------------------------------------------

    def fetch_spot_yields(
        self, date: Optional[str] = None, refresh: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch CGB spot yields for standard maturities.

        Returns
        -------
        maturities : np.ndarray (shape (10,))
            Maturities in years: [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        yields : np.ndarray (shape (10,))
            Corresponding yields in percent.
        """
        date = date or date_str()
        cache_key = f"spot_{date}"

        if self.use_cache and not refresh:
            cached = _read_cache(cache_key, ttl=self.CACHE_TTL_SECONDS)
            if cached:
                return np.array(cached["maturities"]), np.array(cached["yields"])

        yields = self._fetch_from_akshare(date)

        data = {"date": date, "maturities": list(TENORS), "yields": list(yields)}
        _write_cache(cache_key, data)
        return np.array(TENORS), np.array(yields)

    # ------------------------------------------------------------------
    # Historical curve data
    # ------------------------------------------------------------------

    def fetch_historical_curves(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        refresh: bool = False,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Fetch historical CGB yield curves for time-series analysis.

        Returns
        -------
        dates : List[str]
            List of date strings (YYYY-MM-DD).
        yields_matrix : np.ndarray (n_dates, 10)
            Yield matrix, each row is one day's curve.
        """
        if end_date is None:
            end_date = date_str()
        if start_date is None:
            # Default to 2 years back
            sd = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=730)
            start_date = sd.strftime("%Y-%m-%d")

        cache_key = f"hist_{start_date}_{end_date}"

        if self.use_cache and not refresh:
            cached = _read_cache(cache_key, ttl=self.CACHE_TTL_SECONDS)
            if cached:
                return cached["dates"], np.array(cached["yields_matrix"])

        dates, matrix = self._fetch_history_akshare(start_date, end_date)

        data = {"start": start_date, "end": end_date, "dates": dates, "yields_matrix": matrix.tolist()}
        _write_cache(cache_key, data)
        return dates, matrix

    # ------------------------------------------------------------------
    # Internal — akshare extraction
    # ------------------------------------------------------------------

    def _fetch_from_akshare(self, date_str: str) -> List[float]:
        """
        Extract CGB spot yields from akshare bond_china_close_return.

        bond_china_close_return returns a DataFrame with columns:
        '曲线名称', '日期', '0.08Y', '0.17Y', '0.25Y', '0.5Y', '0.75Y',
        '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y', etc.
        """
        import akshare as ak

        df = ak.bond_china_close_return(
            symbol="国债收益率曲线",
            period="每日",
            start_date=date_str,
            end_date=date_str,
            plot=False,
        )

        if df.empty:
            raise ValueError(f"No CGB data available for {date_str}")

        # The latest trading day on or before date_str
        row = df.iloc[-1]

        # Map standard tenors to column names
        tenor_to_col = {
            3 / 12: "0.25Y",
            6 / 12: "0.5Y",
            1: "1Y",
            2: "2Y",
            3: "3Y",
            5: "5Y",
            7: "7Y",
            10: "10Y",
            20: "20Y",
            30: "30Y",
        }

        yields = []
        for t in TENORS:
            col = tenor_to_col.get(t)
            if col and col in row.index:
                val = float(row[col])
                yields.append(val if val > 0 else np.nan)
            else:
                yields.append(np.nan)

        # Forward-fill any missing
        yields = pd.Series(yields).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill").tolist()

        if len(yields) != len(TENORS):
            raise ValueError("Incomplete yield data.")
        return yields

    def _fetch_history_akshare(self, start: str, end: str) -> Tuple[List[str], np.ndarray]:
        """Fetch historical daily CGB curves."""
        import akshare as ak

        df = ak.bond_china_close_return(
            symbol="国债收益率曲线",
            period="每日",
            start_date=start,
            end_date=end,
            plot=False,
        )

        if df.empty:
            raise ValueError(f"No historical CGB data for {start} to {end}")

        tenor_to_col = {
            3 / 12: "0.25Y",
            6 / 12: "0.5Y",
            1: "1Y",
            2: "2Y",
            3: "3Y",
            5: "5Y",
            7: "7Y",
            10: "10Y",
            20: "20Y",
            30: "30Y",
        }

        cols = [tenor_to_col[t] for t in TENORS]
        available = [c for c in cols if c in df.columns]

        dates = df["日期"].tolist()
        matrix = df[available].values.astype(float)

        # Interpolate NaNs
        matrix = pd.DataFrame(matrix).interpolate(method="linear", axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1).values

        return dates, matrix

    # ------------------------------------------------------------------
    # CPI data (for real yield)
    # ------------------------------------------------------------------

    def fetch_cpi(self, refresh: bool = False) -> float:
        """Fetch latest China CPI year-over-year percentage."""
        cache_key = "cpi_latest"
        if self.use_cache and not refresh:
            cached = _read_cache(cache_key, ttl=self.CACHE_TTL_SECONDS)
            if cached:
                return cached["cpi"]

        import akshare as ak

        df = ak.macro_china_cpi_yearly()
        cpi = float(df.iloc[-1]["cpi"])

        _write_cache(cache_key, {"cpi": cpi})
        return cpi


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

_shared_fetcher = None


def _get_fetcher() -> ChinaBondFetcher:
    global _shared_fetcher
    if _shared_fetcher is None:
        _shared_fetcher = ChinaBondFetcher(use_cache=True)
    return _shared_fetcher


def fetch_spot_yields(date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: fetch CGB spot yields."""
    return _get_fetcher().fetch_spot_yields(date=date)


def fetch_historical_curves(
    start: Optional[str] = None, end: Optional[str] = None
) -> Tuple[List[str], np.ndarray]:
    """Convenience wrapper: fetch historical CGB curves."""
    return _get_fetcher().fetch_historical_curves(start_date=start, end_date=end)
