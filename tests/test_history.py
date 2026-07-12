"""History math on constructed price series with known answers."""
import numpy as np
import pandas as pd

from evaluator import history


def make_series(values, start="2020-01-01", freq="D"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_cagr_doubling_in_one_year():
    # price doubles across exactly one year -> 100%/yr
    idx = pd.date_range("2020-01-01", "2021-01-01", freq="D")
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    assert abs(history.cagr(s) - 1.0) < 0.01


def test_cagr_flat_is_zero():
    idx = pd.date_range("2018-01-01", periods=1000, freq="D")
    s = pd.Series(100.0, index=idx)
    assert abs(history.cagr(s)) < 1e-9


def test_max_drawdown_depth_and_recovery():
    # 100 -> 150 (peak) -> 75 (trough, -50%) -> 160 (recovered)
    s = make_series([100, 120, 150, 120, 90, 75, 110, 150, 160])
    dd = history.max_drawdown(s)
    assert abs(dd["depth"] - (-0.5)) < 1e-9
    assert dd["recovery_date"] is not None


def test_drawdown_series_bounds():
    s = make_series([100, 150, 75, 160])
    ds = history.drawdown_series(s)
    assert ds.max() <= 0
    assert abs(ds.min() - (-0.5)) < 1e-9


def test_calendar_year_returns():
    idx = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    # +10% over 2021: last price of 2020 = 100, last of 2021 = 110
    vals = np.where(idx.year == 2020, 100.0, np.linspace(100, 110, (idx.year == 2021).sum()).astype(float)[
        np.maximum(0, np.cumsum(idx.year == 2021) - 1)])
    s = pd.Series(vals, index=idx)
    cal = history.calendar_year_returns(s)
    assert abs(cal.loc[2021] - 0.10) < 0.005
