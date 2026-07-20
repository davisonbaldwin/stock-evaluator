"""Scoring: band interpolation, missing-data neutrality, verdict thresholds."""
from evaluator.score import _avg, _band, _ratio, compute


def test_negative_lower_is_better_ratios_are_not_perfect_scores():
    # regression: a loss-making company has a negative forward P/E, and these
    # bands run from their best anchor downward, so an unguarded negative fell
    # through _band as a flawless 100. It reads as missing instead, which also
    # drops the pillar's data coverage.
    pe_bands = [(12, 100), (20, 75), (30, 50), (45, 25), (70, 0)]
    assert _band(-8.0, pe_bands, higher_is_better=False) == 100   # the trap itself
    assert _ratio(-8.0) is None                                   # so it never reaches it
    assert _ratio(0.0) is None                                    # a zero P/E is not cheap
    assert _ratio(18.4) == 18.4                                   # ordinary values pass


def test_zero_debt_survives_the_ratio_guard():
    # no borrowings is a real, excellent state; only negative equity is nonsense
    assert _ratio(0.0, allow_zero=True) == 0.0
    assert _ratio(-50.0, allow_zero=True) is None


def test_band_endpoints_and_interpolation():
    bands = [(0.0, 0), (1.0, 100)]
    assert _band(-5.0, bands) == 0
    assert _band(5.0, bands) == 100
    assert _band(0.5, bands) == 50


def test_band_direction_is_authored_in_the_anchors():
    # lower-is-better metrics author their anchors descending; the flag must
    # not invert them a second time (regression test for a live scoring bug)
    pe_bands = [(12, 100), (20, 75), (30, 50), (45, 25), (70, 0)]
    assert _band(12, pe_bands, higher_is_better=False) == 100   # cheap: best score
    assert _band(70, pe_bands, higher_is_better=False) == 0     # expensive: worst
    assert _band(16, pe_bands, higher_is_better=False) == 87.5


def test_band_none_passes_through():
    assert _band(None, [(0, 0), (1, 100)]) is None


def test_avg_neutral_when_everything_missing():
    score, cov = _avg([None, None, None])
    assert score == 50.0 and cov == 0.0


def _empty_inputs():
    fund = {
        "valuation": {"forward_pe": None, "peg": None, "fcf_yield": None, "ev_to_ebitda": None},
        "growth": {"revenue_growth_yoy": None, "earnings_growth_yoy": None,
                   "revenue_cagr_multi": None, "fcf_cagr_multi": None},
        "quality": {"gross_margin": None, "operating_margin": None, "roe": None,
                    "rule_of_40": None, "debt_to_equity": None},
    }
    hist = {"momentum": {"12m": None, "6m": None, "3m": None}, "off_52w_high": None,
            "volatility_1y": None, "sharpe_5y": None, "max_drawdown": {}}
    bench = {"capm": {}, "rolling_wins": {}}
    mc = {"horizons": {}}
    return hist, fund, bench, mc, None


def test_missing_data_scores_neutral_50():
    # the documented promise: no data is never a penalty
    hist, fund, bench, mc, rdcf = _empty_inputs()
    sc = compute(hist, fund, bench, mc, rdcf)
    assert sc["total"] == 50.0
    assert sc["coverage"] == 0.0
    assert sc["verdict"].startswith("Mixed")


def test_strong_stock_scores_high():
    hist, fund, bench, mc, _ = _empty_inputs()
    fund["growth"].update(revenue_growth_yoy=0.40, earnings_growth_yoy=0.60,
                          revenue_cagr_multi=0.35, fcf_cagr_multi=0.45)
    fund["quality"].update(gross_margin=0.80, operating_margin=0.40, roe=0.45,
                           rule_of_40=70, debt_to_equity=10)
    fund["valuation"].update(forward_pe=10, peg=0.5, fcf_yield=0.08, ev_to_ebitda=8)
    hist.update(momentum={"12m": 0.90, "6m": 0.40, "3m": 0.20}, off_52w_high=0.0,
                volatility_1y=0.15, sharpe_5y=1.8, max_drawdown={"depth": -0.15})
    bench.update(capm={"beta": 0.8, "alpha_annual": 0.15},
                 rolling_wins={"3y": {"win_rate": 0.9}, "5y": {"win_rate": 0.9}})
    mc["horizons"]["5y"] = {"prob_beat_index": 0.8}
    sc = compute(hist, fund, bench, mc, None)
    assert sc["total"] >= 90
    assert sc["verdict"].startswith("Strong")


def test_weak_stock_scores_low():
    hist, fund, bench, mc, _ = _empty_inputs()
    fund["growth"].update(revenue_growth_yoy=-0.10, earnings_growth_yoy=-0.20,
                          revenue_cagr_multi=-0.05, fcf_cagr_multi=-0.10)
    fund["valuation"].update(forward_pe=90, peg=6.0, fcf_yield=-0.01, ev_to_ebitda=60)
    hist.update(momentum={"12m": -0.40, "6m": -0.25, "3m": -0.10}, off_52w_high=-0.55,
                volatility_1y=0.85, sharpe_5y=-0.4, max_drawdown={"depth": -0.9})
    bench.update(capm={"beta": 2.6, "alpha_annual": -0.10},
                 rolling_wins={"3y": {"win_rate": 0.1}, "5y": {"win_rate": 0.1}})
    mc["horizons"]["5y"] = {"prob_beat_index": 0.2}
    sc = compute(hist, fund, bench, mc, None)
    assert sc["total"] <= 30
    assert sc["verdict"].startswith("Poor") or sc["verdict"].startswith("Weak")
