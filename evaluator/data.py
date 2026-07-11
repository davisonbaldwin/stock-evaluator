"""Data layer: fetches prices, fundamentals, and analyst data with a local disk cache."""
from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_TTL_SECONDS = 12 * 3600  # refresh market data twice a day


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return CACHE_DIR / f"{safe}.pkl"


def _cache_get(key: str) -> Any:
    p = _cache_path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < CACHE_TTL_SECONDS:
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _cache_put(key: str, value: Any) -> None:
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(value, f)
    except Exception:
        pass


def fetch_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Daily OHLCV, dividend-and-split adjusted (total-return basis for Close)."""
    key = f"hist_{ticker}_{period}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price history returned for {ticker!r}. Check the symbol.")
    df.index = df.index.tz_localize(None)
    _cache_put(key, df)
    return df


def fetch_info(ticker: str) -> dict:
    key = f"info_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    _cache_put(key, info)
    return info


def fetch_financials(ticker: str) -> dict:
    """Annual income statement, balance sheet, and cash-flow statements."""
    key = f"fin_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    t = yf.Ticker(ticker)
    out = {}
    for name, attr in [("income", "income_stmt"), ("balance", "balance_sheet"), ("cashflow", "cashflow")]:
        try:
            df = getattr(t, attr)
            out[name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            out[name] = pd.DataFrame()
    _cache_put(key, out)
    return out


def fetch_analyst_targets(ticker: str) -> dict:
    """Analyst price targets: {low, high, mean, median, current} where available."""
    key = f"tgt_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    targets = {}
    try:
        raw = yf.Ticker(ticker).analyst_price_targets
        if isinstance(raw, dict):
            targets = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    _cache_put(key, targets)
    return targets


def risk_free_rate(default: float = 0.04) -> float:
    """Annualized risk-free rate from the 13-week T-bill (^IRX); falls back to `default`."""
    key = "riskfree"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        h = yf.Ticker("^IRX").history(period="5d")
        rate = float(h["Close"].dropna().iloc[-1]) / 100.0
        if not (0.0 <= rate <= 0.15):
            rate = default
    except Exception:
        rate = default
    _cache_put(key, rate)
    return rate


def get_field(info: dict, *keys: str) -> Optional[float]:
    """First present numeric field among `keys` in an info dict."""
    for k in keys:
        v = info.get(k)
        if isinstance(v, (int, float)) and v == v:  # NaN check
            return float(v)
    return None
