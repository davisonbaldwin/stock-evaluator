"""Historical performance analytics computed from an adjusted close series."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(series: pd.Series, years: Optional[float] = None) -> Optional[float]:
    """Compound annual growth rate over the trailing `years` (or full series)."""
    s = series.dropna()
    if len(s) < 2:
        return None
    if years is not None:
        cutoff = s.index[-1] - pd.DateOffset(years=years)
        s = s[s.index >= cutoff]
        if len(s) < 2:
            return None
        actual_years = (s.index[-1] - s.index[0]).days / 365.25
        # Don't report a "10y CAGR" for a stock with only 4 years of data
        if actual_years < years * 0.9:
            return None
    actual_years = (s.index[-1] - s.index[0]).days / 365.25
    if actual_years <= 0 or s.iloc[0] <= 0:
        return None
    return (s.iloc[-1] / s.iloc[0]) ** (1 / actual_years) - 1


def trailing_return(series: pd.Series, months: int) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    cutoff = s.index[-1] - pd.DateOffset(months=months)
    window = s[s.index >= cutoff]
    if len(window) < 2 or (s.index[-1] - window.index[0]).days < months * 20:
        return None
    return window.iloc[-1] / window.iloc[0] - 1


def annualized_volatility(series: pd.Series, years: Optional[float] = None) -> Optional[float]:
    s = series.dropna()
    if years is not None:
        s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    rets = s.pct_change().dropna()
    if len(rets) < 30:
        return None
    return float(rets.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(series: pd.Series, risk_free: float, years: float = 5) -> Optional[float]:
    s = series.dropna()
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    rets = s.pct_change().dropna()
    if len(rets) < 60:
        return None
    excess = rets - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return None
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(series: pd.Series, risk_free: float, years: float = 5) -> Optional[float]:
    s = series.dropna()
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    rets = s.pct_change().dropna()
    if len(rets) < 60:
        return None
    excess = rets - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 5 or downside.std() == 0:
        return None
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(series: pd.Series) -> dict:
    """Worst peak-to-trough loss with peak/trough dates and recovery time."""
    s = series.dropna()
    if len(s) < 2:
        return {}
    running_max = s.cummax()
    dd = s / running_max - 1
    trough_date = dd.idxmin()
    depth = float(dd.min())
    peak_date = s[:trough_date].idxmax()
    after = s[trough_date:]
    recovered = after[after >= s[peak_date]]
    recovery_date = recovered.index[0] if not recovered.empty else None
    return {
        "depth": depth,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_days": (recovery_date - trough_date).days if recovery_date is not None else None,
    }


def current_drawdown(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1] / s.max() - 1)


def calmar_ratio(series: pd.Series, years: float = 5) -> Optional[float]:
    s = series.dropna()
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    growth = cagr(s)
    mdd = max_drawdown(s)
    if growth is None or not mdd or mdd["depth"] == 0:
        return None
    return growth / abs(mdd["depth"])


def calendar_year_returns(series: pd.Series) -> pd.Series:
    s = series.dropna()
    yearly = s.resample("YE").last()
    rets = yearly.pct_change().dropna()
    rets.index = rets.index.year
    return rets


def rolling_return_stats(series: pd.Series, window_years: int = 1) -> dict:
    """Distribution of rolling N-year returns: how often was an N-year hold positive?"""
    s = series.dropna()
    n = window_years * TRADING_DAYS
    if len(s) < n + 10:
        return {}
    roll = s.pct_change(n).dropna()
    return {
        "best": float(roll.max()),
        "worst": float(roll.min()),
        "median": float(roll.median()),
        "pct_positive": float((roll > 0).mean()),
        "n_windows": len(roll),
    }


def drawdown_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return s / s.cummax() - 1


def analyze(close: pd.Series, risk_free: float) -> dict:
    """Full historical metrics bundle for a single adjusted-close series."""
    s = close.dropna()
    years_listed = (s.index[-1] - s.index[0]).days / 365.25
    cal = calendar_year_returns(s)
    return {
        "first_date": s.index[0],
        "last_date": s.index[-1],
        "last_price": float(s.iloc[-1]),
        "years_of_data": years_listed,
        "cagr": {label: cagr(s, yrs) for label, yrs in
                 [("1y", 1), ("3y", 3), ("5y", 5), ("10y", 10), ("15y", 15), ("max", None)]},
        "momentum": {"3m": trailing_return(s, 3), "6m": trailing_return(s, 6), "12m": trailing_return(s, 12)},
        "volatility_1y": annualized_volatility(s, 1),
        "volatility_5y": annualized_volatility(s, 5),
        "sharpe_5y": sharpe_ratio(s, risk_free),
        "sortino_5y": sortino_ratio(s, risk_free),
        "calmar_5y": calmar_ratio(s),
        "max_drawdown": max_drawdown(s),
        "current_drawdown": current_drawdown(s),
        "off_52w_high": float(s.iloc[-1] / s[s.index >= s.index[-1] - pd.DateOffset(years=1)].max() - 1),
        "calendar_returns": cal,
        "best_year": (int(cal.idxmax()), float(cal.max())) if not cal.empty else None,
        "worst_year": (int(cal.idxmin()), float(cal.min())) if not cal.empty else None,
        "rolling_1y": rolling_return_stats(s, 1),
        "rolling_3y": rolling_return_stats(s, 3),
        "rolling_5y": rolling_return_stats(s, 5),
    }
