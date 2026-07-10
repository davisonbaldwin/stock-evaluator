"""Comparison of a stock against a total-market index investor (default VTI)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .history import TRADING_DAYS, cagr, max_drawdown

MONTHS_PER_YEAR = 12


def align(stock: pd.Series, index: pd.Series) -> pd.DataFrame:
    """Daily closes aligned on common dates (limits both to the shared listing window)."""
    df = pd.concat({"stock": stock, "index": index}, axis=1).dropna()
    return df


def capm_regression(stock: pd.Series, index: pd.Series, risk_free: float,
                    years: float = 5) -> dict:
    """Monthly CAPM regression: beta, annualized alpha, correlation, R²."""
    df = align(stock, index)
    df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    monthly = df.resample("ME").last().pct_change().dropna()
    if len(monthly) < 24:
        return {}
    rf_m = risk_free / MONTHS_PER_YEAR
    ex_s = monthly["stock"] - rf_m
    ex_i = monthly["index"] - rf_m
    beta, alpha_m = np.polyfit(ex_i, ex_s, 1)
    corr = float(ex_s.corr(ex_i))
    return {
        "beta": float(beta),
        "alpha_annual": float((1 + alpha_m) ** MONTHS_PER_YEAR - 1),
        "correlation": corr,
        "r_squared": corr ** 2,
        "n_months": len(monthly),
    }


def capture_ratios(stock: pd.Series, index: pd.Series, years: float = 5) -> dict:
    """Up/down capture: stock's average monthly move per 1% index move, split by direction."""
    df = align(stock, index)
    df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    monthly = df.resample("ME").last().pct_change().dropna()
    up = monthly[monthly["index"] > 0]
    down = monthly[monthly["index"] < 0]
    out = {}
    if len(up) >= 6 and up["index"].mean() != 0:
        out["up_capture"] = float(up["stock"].mean() / up["index"].mean())
    if len(down) >= 6 and down["index"].mean() != 0:
        out["down_capture"] = float(down["stock"].mean() / down["index"].mean())
    return out


def information_ratio(stock: pd.Series, index: pd.Series, years: float = 5) -> dict:
    """Active return over tracking error, on daily data."""
    df = align(stock, index)
    df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    rets = df.pct_change().dropna()
    if len(rets) < 120:
        return {}
    active = rets["stock"] - rets["index"]
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    if te == 0:
        return {}
    return {
        "active_return_annual": float(active.mean() * TRADING_DAYS),
        "tracking_error": te,
        "information_ratio": float(active.mean() * TRADING_DAYS / te),
    }


def rolling_win_rate(stock: pd.Series, index: pd.Series, window_years: int) -> Optional[dict]:
    """Share of all rolling N-year holding periods in which the stock beat the index."""
    df = align(stock, index)
    n = window_years * TRADING_DAYS
    if len(df) < n + 10:
        return None
    stock_roll = df["stock"].pct_change(n).dropna()
    index_roll = df["index"].pct_change(n).dropna()
    both = pd.concat({"s": stock_roll, "i": index_roll}, axis=1).dropna()
    if both.empty:
        return None
    wins = (both["s"] > both["i"]).mean()
    return {
        "win_rate": float(wins),
        "median_outperformance": float((both["s"] - both["i"]).median()),
        "n_windows": len(both),
    }


def growth_of_10k(stock: pd.Series, index: pd.Series, years: Optional[float] = None) -> pd.DataFrame:
    """$10,000 invested in each at the start of the common (or trailing N-year) window."""
    df = align(stock, index)
    if years is not None:
        df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    return df / df.iloc[0] * 10_000


def compare(stock: pd.Series, index: pd.Series, risk_free: float) -> dict:
    """Full stock-vs-index comparison bundle."""
    df = align(stock, index)
    common_years = (df.index[-1] - df.index[0]).days / 365.25
    g10k = growth_of_10k(stock, index)
    result = {
        "common_start": df.index[0],
        "common_years": common_years,
        "stock_cagr_common": cagr(df["stock"]),
        "index_cagr_common": cagr(df["index"]),
        "index_cagr": {label: cagr(index, yrs) for label, yrs in
                       [("1y", 1), ("3y", 3), ("5y", 5), ("10y", 10)]},
        "stock_mdd": max_drawdown(df["stock"]).get("depth"),
        "index_mdd": max_drawdown(df["index"]).get("depth"),
        "final_10k_stock": float(g10k["stock"].iloc[-1]),
        "final_10k_index": float(g10k["index"].iloc[-1]),
        "capm": capm_regression(stock, index, risk_free),
        "capture": capture_ratios(stock, index),
        "active": information_ratio(stock, index),
        "rolling_wins": {f"{y}y": rolling_win_rate(stock, index, y) for y in (1, 3, 5)},
    }
    return result
