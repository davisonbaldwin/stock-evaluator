"""Valuation, growth, and quality metrics from company fundamentals."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .data import get_field


def _statement_row(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            row = df.loc[name].dropna()
            if not row.empty:
                # columns are period-end timestamps, newest first in yfinance
                return row.sort_index()
    return None


def _series_cagr(row: Optional[pd.Series]) -> Optional[float]:
    """CAGR across an annual statement row (oldest → newest)."""
    if row is None or len(row) < 2:
        return None
    first, last = float(row.iloc[0]), float(row.iloc[-1])
    years = (row.index[-1] - row.index[0]).days / 365.25
    if first <= 0 or last <= 0 or years <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def analyze(info: dict, financials: dict) -> dict:
    """Bundle of valuation / growth / quality metrics. Any item may be None."""
    income = financials.get("income", pd.DataFrame())
    cashflow = financials.get("cashflow", pd.DataFrame())
    balance = financials.get("balance", pd.DataFrame())

    revenue_row = _statement_row(income, "Total Revenue", "Operating Revenue")
    netinc_row = _statement_row(income, "Net Income", "Net Income Common Stockholders")
    fcf_row = _statement_row(cashflow, "Free Cash Flow")

    market_cap = get_field(info, "marketCap")
    fcf = get_field(info, "freeCashflow")
    if fcf is None and fcf_row is not None:
        fcf = float(fcf_row.iloc[-1])
    revenue = get_field(info, "totalRevenue")
    if revenue is None and revenue_row is not None:
        revenue = float(revenue_row.iloc[-1])

    # yfinance >=0.2.5x reports dividendYield in percent units (0.5 == 0.5%)
    div_yield = get_field(info, "dividendYield")
    if div_yield is not None:
        div_yield /= 100.0

    fcf_yield = (fcf / market_cap) if (fcf and market_cap) else None
    fcf_margin = (fcf / revenue) if (fcf and revenue) else None
    revenue_growth = get_field(info, "revenueGrowth")

    # Rule of 40 (tech health check): revenue growth % + FCF margin %
    rule_of_40 = None
    if revenue_growth is not None and fcf_margin is not None:
        rule_of_40 = revenue_growth * 100 + fcf_margin * 100

    cash = get_field(info, "totalCash")
    debt = get_field(info, "totalDebt")
    net_cash = (cash - debt) if (cash is not None and debt is not None) else None

    return {
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": market_cap,
        "valuation": {
            "trailing_pe": get_field(info, "trailingPE"),
            "forward_pe": get_field(info, "forwardPE"),
            "peg": get_field(info, "trailingPegRatio", "pegRatio"),
            "price_to_sales": get_field(info, "priceToSalesTrailing12Months"),
            "ev_to_ebitda": get_field(info, "enterpriseToEbitda"),
            "ev_to_revenue": get_field(info, "enterpriseToRevenue"),
            "fcf_yield": fcf_yield,
            "dividend_yield": div_yield,
        },
        "growth": {
            "revenue_growth_yoy": revenue_growth,
            "earnings_growth_yoy": get_field(info, "earningsGrowth"),
            "revenue_cagr_multi": _series_cagr(revenue_row),
            "net_income_cagr_multi": _series_cagr(netinc_row),
            "fcf_cagr_multi": _series_cagr(fcf_row),
            "statement_years": len(revenue_row) if revenue_row is not None else 0,
        },
        "quality": {
            "gross_margin": get_field(info, "grossMargins"),
            "operating_margin": get_field(info, "operatingMargins"),
            "net_margin": get_field(info, "profitMargins"),
            "fcf_margin": fcf_margin,
            "roe": get_field(info, "returnOnEquity"),
            "roa": get_field(info, "returnOnAssets"),
            "debt_to_equity": get_field(info, "debtToEquity"),
            "current_ratio": get_field(info, "currentRatio"),
            "net_cash": net_cash,
            "rule_of_40": rule_of_40,
        },
        "analyst": {
            "recommendation": info.get("recommendationKey"),
            "n_analysts": get_field(info, "numberOfAnalystOpinions"),
        },
        "eps_forward": get_field(info, "forwardEps"),
        "eps_trailing": get_field(info, "trailingEps"),
    }
