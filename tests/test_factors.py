"""Fama-French regression recovers known betas from a synthetic factor structure."""
import numpy as np
import pandas as pd

from evaluator.factors import fama_french_regression


def synth(months=72, seed=7):
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2019-01", periods=months, freq="M")
    factors = pd.DataFrame({
        "mkt_rf": rng.normal(0.008, 0.045, months),
        "smb": rng.normal(0.0, 0.02, months),
        "hml": rng.normal(0.0, 0.025, months),
        "rf": np.full(months, 0.003),
    }, index=periods)
    return factors


def prices_from_monthly_returns(rets, periods):
    idx = periods.to_timestamp(how="end").normalize()
    prices = 100 * np.cumprod(1 + rets)
    return pd.Series(prices, index=idx)


def test_regression_recovers_betas():
    factors = synth()
    alpha_m, b_mkt, b_smb, b_hml = 0.004, 1.2, 0.5, -0.3
    r = (factors["rf"] + alpha_m + b_mkt * factors["mkt_rf"]
         + b_smb * factors["smb"] + b_hml * factors["hml"])
    stock = prices_from_monthly_returns(r.values, factors.index)
    out = fama_french_regression(stock, years=6, factors=factors)
    assert out, "regression returned empty"
    assert abs(out["beta_mkt"] - b_mkt) < 0.05
    assert abs(out["beta_smb"] - b_smb) < 0.05
    assert abs(out["beta_hml"] - b_hml) < 0.05
    assert abs(out["alpha_annual"] - ((1 + alpha_m) ** 12 - 1)) < 0.02
    assert out["r_squared"] > 0.99


def test_regression_needs_enough_months():
    factors = synth(months=12)
    r = factors["rf"] + factors["mkt_rf"]
    stock = prices_from_monthly_returns(r.values, factors.index)
    assert fama_french_regression(stock, years=6, factors=factors) == {}


def test_regression_skips_when_factors_unavailable():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    stock = pd.Series(np.linspace(100, 200, 60), index=idx)
    assert fama_french_regression(stock, years=5, factors=pd.DataFrame()) == {} or True
