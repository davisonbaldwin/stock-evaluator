"""Forward projections: scenario models, Monte Carlo simulation, and reverse DCF.

Three independent lenses on the future:

1. Scenario projections — five growth scenarios (bear → bull) blended from
   historical CAGR, CAPM-implied return, and analyst targets, projected over
   1/3/5/10-year horizons.
2. Monte Carlo — joint bootstrap of historical (stock, index) daily-return pairs,
   preserving fat tails and their correlation, plus a GBM cross-check. Yields
   outcome percentiles, probability of loss, and probability of beating the index.
3. Reverse DCF — the FCF growth rate the market is pricing in at today's value.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .history import TRADING_DAYS, cagr

EQUITY_RISK_PREMIUM = 0.045
HORIZONS = (1, 3, 5, 10)


# ---------------------------------------------------------------- scenarios

def blended_drift(hist: dict, capm: dict, risk_free: float) -> dict:
    """Expected-return building blocks. Historical CAGR is capped: past hypergrowth
    is the worst predictor of the future, so anything above 25%/yr is haircut."""
    hist_candidates = [hist["cagr"].get(k) for k in ("5y", "10y") if hist["cagr"].get(k) is not None]
    hist_cagr = float(np.mean(hist_candidates)) if hist_candidates else hist["cagr"].get("max")
    hist_capped = min(hist_cagr, 0.25) if hist_cagr is not None else None

    beta = capm.get("beta")
    capm_return = risk_free + (beta if beta is not None else 1.2) * EQUITY_RISK_PREMIUM

    if hist_capped is not None:
        base = 0.5 * hist_capped + 0.5 * capm_return
    else:
        base = capm_return
    return {"historical_capped": hist_capped, "capm_return": capm_return, "base": base}


def scenario_projections(last_price: float, drift: dict, vol: float,
                         analyst: dict, index_return: float) -> dict:
    """Five annual-return scenarios projected to each horizon.

    bear          ≈ base minus one sigma-scaled shock (capped at -15%/yr)
    conservative  ≈ index-like return (you paid a premium for nothing)
    base          ≈ blend of capped historical CAGR and CAPM-implied return
    optimistic    ≈ base plus half the gap to uncapped history / analyst upside
    bull          ≈ historical CAGR sustained (uncapped, but ≤ 40%/yr)
    """
    base = drift["base"]
    hist = drift["historical_capped"]
    shock = min(0.5 * (vol if vol else 0.30), 0.25)

    bear = max(base - shock, -0.15)
    conservative = min(base, index_return)
    optimistic = base + shock / 2
    bull_anchor = hist if hist is not None else base
    bull = min(max(bull_anchor * 1.25, optimistic + 0.03), 0.40)

    scenarios = {
        "bear": bear,
        "conservative": conservative,
        "base": base,
        "optimistic": optimistic,
        "bull": bull,
    }
    table = {
        name: {f"{h}y": last_price * (1 + r) ** h for h in HORIZONS}
        for name, r in scenarios.items()
    }

    # Analyst 1-year target overlay, when available
    analyst_row = None
    if analyst and analyst.get("mean"):
        analyst_row = {
            "low": analyst.get("low"), "mean": analyst.get("mean"),
            "high": analyst.get("high"),
            "implied_return": analyst["mean"] / last_price - 1,
        }
    return {"rates": scenarios, "prices": table, "analyst_1y": analyst_row}


# -------------------------------------------------------------- monte carlo

def monte_carlo(stock_close: pd.Series, index_close: pd.Series, drift: dict,
                index_drift: float, n_sims: int = 10_000, years: int = 10,
                lookback_years: int = 10, seed: int = 42) -> dict:
    """Joint block-bootstrap of (stock, index) daily returns.

    21-day blocks preserve short-term autocorrelation and the stock/index
    correlation structure. Each path's empirical drift is re-centered to the
    blended expected return so the simulation isn't just an echo of a
    historically hot (or cold) decade — history supplies the *shape* of risk
    (vol, fat tails, correlation), the blend supplies the central tendency.
    """
    rng = np.random.default_rng(seed)
    df = pd.concat({"s": stock_close, "i": index_close}, axis=1).dropna()
    df = df[df.index >= df.index[-1] - pd.DateOffset(years=lookback_years)]
    rets = np.log(df / df.shift(1)).dropna().to_numpy()
    if len(rets) < 252:
        return {}

    block = 21
    n_days = years * TRADING_DAYS
    n_blocks = int(np.ceil(n_days / block))

    # Re-center daily log drift to target annual returns
    target_s = np.log(1 + drift["base"]) / TRADING_DAYS
    target_i = np.log(1 + index_drift) / TRADING_DAYS
    adj = rets - rets.mean(axis=0) + np.array([target_s, target_i])

    starts = rng.integers(0, len(adj) - block, size=(n_sims, n_blocks))
    # Build paths: terminal log-return per simulation for stock and index
    term = np.zeros((n_sims, 2))
    snapshots = {h: np.zeros((n_sims, 2)) for h in HORIZONS if h <= years}
    for b in range(n_blocks):
        seg_sum = np.zeros((n_sims, 2))
        for k in range(block):
            day = b * block + k
            if day >= n_days:
                break
            seg_sum += adj[starts[:, b] + k]
        term += seg_sum
        for h in snapshots:
            if b == int(np.ceil(h * TRADING_DAYS / block)) - 1:
                snapshots[h][:] = term

    last_s = float(df["s"].iloc[-1])
    out = {"horizons": {}, "n_sims": n_sims}
    for h, vals in sorted(snapshots.items()):
        stock_mult = np.exp(vals[:, 0])
        index_mult = np.exp(vals[:, 1])
        pct = np.percentile(stock_mult, [5, 25, 50, 75, 95])
        out["horizons"][f"{h}y"] = {
            "price_p5": last_s * pct[0], "price_p25": last_s * pct[1],
            "price_p50": last_s * pct[2], "price_p75": last_s * pct[3],
            "price_p95": last_s * pct[4],
            "cagr_p50": pct[2] ** (1 / h) - 1,
            "prob_loss": float((stock_mult < 1).mean()),
            "prob_beat_index": float((stock_mult > index_mult).mean()),
            "prob_double": float((stock_mult >= 2).mean()),
        }
    return out


def gbm_percentiles(last_price: float, drift: float, vol: float) -> dict:
    """Closed-form lognormal (GBM) percentiles as a cross-check on the bootstrap."""
    import math

    def _ppf(q: float) -> float:
        # standard-normal inverse CDF via bisection on erf (avoids a scipy dep)
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (1 + math.erf(mid / math.sqrt(2))) / 2 < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    out = {}
    mu = math.log(1 + drift)
    for h in HORIZONS:
        m = (mu - 0.5 * vol ** 2) * h
        s = vol * math.sqrt(h)
        out[f"{h}y"] = {f"p{q}": last_price * math.exp(m + s * _ppf(q / 100))
                        for q in (5, 50, 95)}
    return out


# -------------------------------------------------------------- reverse DCF

def reverse_dcf(market_cap: Optional[float], fcf: Optional[float],
                discount_rate: float = 0.10, terminal_growth: float = 0.025,
                horizon: int = 10) -> Optional[dict]:
    """Solve for the constant FCF growth rate over `horizon` years that justifies
    the current market cap. High implied growth = expensive; negative = priced
    for decline."""
    if not market_cap or not fcf or fcf <= 0:
        return None

    def dcf_value(g: float) -> float:
        value, cash = 0.0, fcf
        for t in range(1, horizon + 1):
            cash *= (1 + g)
            value += cash / (1 + discount_rate) ** t
        terminal = cash * (1 + terminal_growth) / (discount_rate - terminal_growth)
        return value + terminal / (1 + discount_rate) ** horizon

    lo, hi = -0.50, 1.50
    if dcf_value(hi) < market_cap:
        return {"implied_growth": None, "note": ">150%/yr — price not justifiable by FCF"}
    if dcf_value(lo) > market_cap:
        return {"implied_growth": None, "note": "priced below run-off value"}
    for _ in range(100):
        mid = (lo + hi) / 2
        if dcf_value(mid) < market_cap:
            lo = mid
        else:
            hi = mid
    g = (lo + hi) / 2
    return {
        "implied_growth": g,
        "discount_rate": discount_rate,
        "terminal_growth": terminal_growth,
        "horizon": horizon,
        "fcf_base": fcf,
    }
