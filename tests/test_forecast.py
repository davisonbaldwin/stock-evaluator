"""Forecast math: drift caps, scenario compounding, reverse DCF round-trip, Monte Carlo sanity."""
import numpy as np
import pandas as pd

from evaluator import forecast


def test_blended_drift_caps_hypergrowth():
    hist = {"cagr": {"5y": 0.50, "10y": 0.30}}
    capm = {"beta": 1.0}
    d = forecast.blended_drift(hist, capm, risk_free=0.04)
    assert d["historical_capped"] == 0.25          # (0.5+0.3)/2 = 0.4 -> capped
    expected_capm = 0.04 + 1.0 * forecast.EQUITY_RISK_PREMIUM
    assert abs(d["capm_return"] - expected_capm) < 1e-9
    assert abs(d["base"] - (0.5 * 0.25 + 0.5 * expected_capm)) < 1e-9


def test_scenario_prices_compound():
    drift = {"historical_capped": 0.15, "capm_return": 0.09, "base": 0.12}
    fc = forecast.scenario_projections(100.0, drift, vol=0.30, analyst={}, index_return=0.08)
    base_rate = fc["rates"]["base"]
    assert abs(fc["prices"]["base"]["10y"] - 100.0 * (1 + base_rate) ** 10) < 1e-6
    # ordering: bear <= conservative <= base <= optimistic <= bull
    r = fc["rates"]
    assert r["bear"] <= r["conservative"] <= r["base"] <= r["optimistic"] <= r["bull"]


def test_reverse_dcf_recovers_known_growth():
    # build a market cap that a 10%/yr FCF growth exactly justifies, then invert it
    fcf, disc, term, horizon, g = 100.0, 0.10, 0.025, 10, 0.10
    value, cash = 0.0, fcf
    for t in range(1, horizon + 1):
        cash *= (1 + g)
        value += cash / (1 + disc) ** t
    value += cash * (1 + term) / (disc - term) / (1 + disc) ** horizon
    out = forecast.reverse_dcf(value, fcf, discount_rate=disc,
                               terminal_growth=term, horizon=horizon)
    assert out and abs(out["implied_growth"] - g) < 1e-3


def test_reverse_dcf_handles_missing_inputs():
    assert forecast.reverse_dcf(None, 100.0) is None
    assert forecast.reverse_dcf(1e9, None) is None
    assert forecast.reverse_dcf(1e9, -50.0) is None


def _gbm(n, drift, vol, seed, start=100.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift / 252, vol / np.sqrt(252) / 4, n)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.Series(start * np.exp(np.cumsum(rets)), index=idx)


def test_monte_carlo_percentiles_ordered():
    stock = _gbm(1500, 0.10, 0.30, seed=1)
    index = _gbm(1500, 0.07, 0.15, seed=2)
    drift = {"base": 0.10}
    mc = forecast.monte_carlo(stock, index, drift, index_drift=0.07, n_sims=500)
    h5 = mc["horizons"]["5y"]
    assert 0 < h5["price_p5"] < h5["price_p50"] < h5["price_p95"]
    assert 0.0 <= h5["prob_loss"] <= 1.0
    assert 0.0 <= h5["prob_beat_index"] <= 1.0
