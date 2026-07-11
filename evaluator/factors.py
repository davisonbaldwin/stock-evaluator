"""Fama-French three-factor analysis.

CAPM asks one question: how much market exposure explains this stock's return?
The three-factor model adds size (SMB) and value (HML), so "alpha" stops
getting credit for what is really just a factor tilt. Factors come from the
Ken French data library (monthly, in percent) and are cached like everything else.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from .data import _cache_get, _cache_put

FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_CSV.zip")
MONTHS_PER_YEAR = 12


def fetch_ff_factors() -> Optional[pd.DataFrame]:
    """Monthly Mkt-RF, SMB, HML, RF as decimal returns indexed by month period.

    Returns None if the data library is unreachable (the analysis is skipped,
    never penalized).
    """
    cached = _cache_get("ff_factors")
    if cached is not None:
        return cached
    try:
        raw = urllib.request.urlopen(FF_URL, timeout=20).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
    except Exception:
        return None
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        # monthly rows look like: 192607, 2.96, -2.56, -2.43, 0.22
        if len(parts) == 5 and re.fullmatch(r"\d{6}", parts[0]):
            try:
                rows.append((parts[0], *(float(p) for p in parts[1:])))
            except ValueError:
                continue
        elif rows and not re.fullmatch(r"\d{6}", parts[0] or ""):
            break            # annual section follows the monthly block
    if len(rows) < 120:
        return None
    df = pd.DataFrame(rows, columns=["ym", "mkt_rf", "smb", "hml", "rf"])
    df.index = pd.PeriodIndex(df["ym"], freq="M")
    df = df.drop(columns="ym") / 100.0
    _cache_put("ff_factors", df)
    return df


def fama_french_regression(stock: pd.Series, years: float = 5,
                           factors: Optional[pd.DataFrame] = None) -> dict:
    """OLS of monthly excess returns on Mkt-RF, SMB, HML over the trailing window."""
    if factors is None:
        factors = fetch_ff_factors()
    if factors is None or stock is None or len(stock) == 0:
        return {}
    monthly = stock.resample("ME").last().pct_change().dropna()
    monthly = monthly[monthly.index >= monthly.index[-1] - pd.DateOffset(years=years)]
    if len(monthly) < 24:
        return {}
    ret = pd.DataFrame({"r": monthly.values}, index=monthly.index.to_period("M"))
    df = ret.join(factors, how="inner").dropna()
    if len(df) < 24:
        return {}
    y = (df["r"] - df["rf"]).values
    X = np.column_stack([np.ones(len(df)), df["mkt_rf"], df["smb"], df["hml"]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    alpha_m, b_mkt, b_smb, b_hml = (float(c) for c in coef)
    return {
        "alpha_annual": float((1 + alpha_m) ** MONTHS_PER_YEAR - 1),
        "beta_mkt": b_mkt,
        "beta_smb": b_smb,
        "beta_hml": b_hml,
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else None,
        "n_months": int(len(df)),
    }
