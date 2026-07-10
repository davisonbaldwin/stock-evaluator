"""Composite 0-100 score across six pillars, with a plain-English verdict.

Scoring philosophy: each metric maps to 0-100 through bands chosen for large-cap
tech. Missing data scores a neutral 50 rather than penalizing the stock — the
report flags low data coverage separately.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def _band(value: Optional[float], bands: List[Tuple[float, float]],
          higher_is_better: bool = True) -> Optional[float]:
    """Linear interpolation through (threshold, score) anchor points."""
    if value is None:
        return None
    pts = sorted(bands)
    if value <= pts[0][0]:
        score = pts[0][1]
    elif value >= pts[-1][0]:
        score = pts[-1][1]
    else:
        score = pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= value <= x1:
                score = y0 + (y1 - y0) * (value - x0) / (x1 - x0)
                break
    return score if higher_is_better else 100 - score


def _avg(scores: List[Optional[float]]) -> Tuple[float, float]:
    """(average of available scores defaulting to 50, coverage fraction)."""
    present = [s for s in scores if s is not None]
    if not present:
        return 50.0, 0.0
    return float(sum(present) / len(present)), len(present) / len(scores)


def compute(hist: dict, fund: dict, bench: dict, mc: dict, rdcf: Optional[dict]) -> dict:
    v, g, q = fund["valuation"], fund["growth"], fund["quality"]

    growth_score, growth_cov = _avg([
        _band(g["revenue_growth_yoy"], [(0.0, 20), (0.10, 50), (0.20, 75), (0.35, 100)]),
        _band(g["earnings_growth_yoy"], [(0.0, 25), (0.10, 50), (0.25, 80), (0.50, 100)]),
        _band(g["revenue_cagr_multi"], [(0.0, 20), (0.08, 50), (0.15, 75), (0.30, 100)]),
        _band(g["fcf_cagr_multi"], [(0.0, 25), (0.10, 55), (0.20, 80), (0.40, 100)]),
    ])

    value_score, value_cov = _avg([
        _band(v["forward_pe"], [(12, 100), (20, 75), (30, 50), (45, 25), (70, 0)], higher_is_better=False),
        _band(v["peg"], [(0.8, 100), (1.5, 70), (2.5, 40), (4.0, 10)], higher_is_better=False),
        _band(v["fcf_yield"], [(0.0, 10), (0.02, 40), (0.04, 70), (0.06, 100)]),
        _band(v["ev_to_ebitda"], [(10, 100), (18, 70), (28, 40), (45, 10)], higher_is_better=False),
        # Reverse-DCF implied growth: the more growth already priced in, the worse
        _band(rdcf["implied_growth"], [(0.05, 100), (0.12, 70), (0.20, 40), (0.30, 10)],
              higher_is_better=False) if rdcf and rdcf.get("implied_growth") is not None else None,
    ])

    quality_score, quality_cov = _avg([
        _band(q["gross_margin"], [(0.30, 25), (0.45, 50), (0.60, 75), (0.75, 100)]),
        _band(q["operating_margin"], [(0.05, 20), (0.15, 50), (0.25, 75), (0.35, 100)]),
        _band(q["roe"], [(0.05, 20), (0.15, 50), (0.25, 75), (0.40, 100)]),
        _band(q["rule_of_40"], [(10, 25), (30, 50), (40, 70), (60, 100)]),
        _band(q["debt_to_equity"], [(30, 100), (80, 70), (150, 40), (300, 10)], higher_is_better=False),
    ])

    mom = hist["momentum"]
    momentum_score, momentum_cov = _avg([
        _band(mom["12m"], [(-0.20, 10), (0.0, 40), (0.15, 65), (0.40, 90), (0.80, 100)]),
        _band(mom["6m"], [(-0.15, 15), (0.0, 45), (0.10, 65), (0.30, 95)]),
        _band(hist["off_52w_high"], [(-0.40, 10), (-0.20, 40), (-0.10, 65), (-0.03, 90), (0.0, 100)]),
    ])

    capm = bench.get("capm", {})
    mdd = hist["max_drawdown"].get("depth")
    risk_score, risk_cov = _avg([
        _band(hist["volatility_1y"], [(0.20, 100), (0.30, 75), (0.45, 45), (0.70, 10)], higher_is_better=False),
        _band(hist["sharpe_5y"], [(0.0, 10), (0.5, 40), (1.0, 70), (1.5, 95)]),
        _band(mdd, [(-0.80, 5), (-0.55, 30), (-0.35, 65), (-0.20, 95)]),
        _band(capm.get("beta"), [(0.9, 100), (1.2, 75), (1.6, 45), (2.2, 10)], higher_is_better=False),
    ])

    wins = bench.get("rolling_wins", {})
    win3 = wins.get("3y") or {}
    win5 = wins.get("5y") or {}
    horizon_5y = (mc.get("horizons") or {}).get("5y", {})
    vs_index_score, vs_index_cov = _avg([
        _band(win3.get("win_rate"), [(0.30, 20), (0.50, 50), (0.65, 75), (0.85, 100)]),
        _band(win5.get("win_rate"), [(0.30, 20), (0.50, 50), (0.65, 75), (0.85, 100)]),
        _band(capm.get("alpha_annual"), [(-0.05, 20), (0.0, 50), (0.05, 75), (0.12, 100)]),
        _band(horizon_5y.get("prob_beat_index"), [(0.35, 20), (0.50, 50), (0.60, 75), (0.75, 100)]),
    ])

    pillars = {
        "Growth": (growth_score, growth_cov, 0.20),
        "Valuation": (value_score, value_cov, 0.20),
        "Quality": (quality_score, quality_cov, 0.20),
        "Momentum": (momentum_score, momentum_cov, 0.10),
        "Risk": (risk_score, risk_cov, 0.15),
        "vs Index": (vs_index_score, vs_index_cov, 0.15),
    }
    total = sum(score * weight for score, _, weight in pillars.values())
    coverage = sum(cov * weight for _, cov, weight in pillars.values())

    if total >= 75:
        verdict = "Strong — broad-based case for owning over the index"
    elif total >= 60:
        verdict = "Favorable — edge over the index, watch the weak pillar(s)"
    elif total >= 45:
        verdict = "Mixed — no clear edge over the index"
    elif total >= 30:
        verdict = "Weak — index likely the better risk-adjusted hold"
    else:
        verdict = "Poor — significant red flags across pillars"

    return {"pillars": pillars, "total": total, "coverage": coverage, "verdict": verdict}
