#!/usr/bin/env python3
"""Local web dashboard for the stock evaluator.

Runs at http://localhost:8742 — open it in any browser, type a ticker, done.
"""
from __future__ import annotations

import traceback
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, send_from_directory
from rich.console import Console

from evaluate import _jsonable, evaluate_ticker
from evaluator.benchmark import growth_of_10k
from evaluator.history import calendar_year_returns, drawdown_series

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.json.sort_keys = False  # keep scenario/horizon ordering as computed
console = Console(quiet=True)

# Local default. In production (Render/Fly/etc.) the host sets PORT and we bind 0.0.0.0.
PORT = int(os.environ.get("PORT", 8742))
HOST = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"


def _weekly(series: pd.Series) -> dict:
    s = series.resample("W").last().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": [round(float(v), 4) for v in s]}


def chart_series(r: dict) -> dict:
    """Downsampled series for the front-end charts."""
    stock, index = r["stock_close"], r["index_close"]
    df = pd.concat({"s": stock, "i": index}, axis=1).dropna()
    g = growth_of_10k(stock, index)
    cal_s = calendar_year_returns(df["s"]).tail(15)
    cal_i = calendar_year_returns(df["i"]).reindex(cal_s.index)
    return {
        "growth10k": {"stock": _weekly(g["stock"]), "index": _weekly(g["index"])},
        "drawdown": {"stock": _weekly(drawdown_series(df["s"])),
                     "index": _weekly(drawdown_series(df["i"]))},
        "calendar": {"years": [int(y) for y in cal_s.index],
                     "stock": [round(float(v), 4) for v in cal_s],
                     "index": [None if pd.isna(v) else round(float(v), 4) for v in cal_i]},
    }


@app.route("/")
def home():
    return send_from_directory(ROOT / "web", "index.html")


@app.route("/api/compare/<tickers>")
@app.route("/api/compare/<tickers>/<benchmark>")
def api_compare(tickers: str, benchmark: str = "VTI"):
    """Evaluate up to 10 tickers and return side-by-side summaries plus a
    common-start growth-of-$10k overlay (stocks + the benchmark)."""
    benchmark = benchmark.upper().strip()
    names = [t.upper().strip() for t in tickers.replace(" ", ",").split(",") if t.strip()]
    names = list(dict.fromkeys(names))[:10]                    # dedupe, cap at 10
    if len(names) < 2:
        return jsonify({"error": "compare needs at least two tickers"}), 400

    rows, series, errors = [], {}, []
    index_close = None
    for t in names:
        try:
            r = evaluate_ticker(t, benchmark, n_sims=4_000, console=console)
        except Exception as e:
            traceback.print_exc()
            errors.append({"ticker": t, "error": str(e)})
            continue
        h, f, b, sc = r["hist"], r["fund"], r["bench"], r["score"]
        capm = b.get("capm") or {}
        wins = (b.get("rolling_wins") or {}).get("5y") or {}
        mdd = h.get("max_drawdown") or {}
        rows.append({
            "ticker": t,
            "name": f.get("name") or t,
            "price": h.get("last_price"),
            "score": sc.get("total"),
            "verdict": sc.get("verdict"),
            "cagr_5y": (h.get("cagr") or {}).get("5y"),
            "cagr_10y": (h.get("cagr") or {}).get("10y"),
            "sharpe_5y": h.get("sharpe_5y"),
            "volatility_1y": h.get("volatility_1y"),
            "max_drawdown": mdd.get("depth"),
            "beta": capm.get("beta"),
            "alpha": capm.get("alpha_annual"),
            "win_rate_5y": wins.get("win_rate"),
            "forward_pe": (f.get("valuation") or {}).get("forward_pe"),
            "revenue_growth": (f.get("growth") or {}).get("revenue_growth_yoy"),
        })
        series[t] = r["stock_close"]
        if index_close is None:
            index_close = r["index_close"]
    if not rows:
        return jsonify({"error": "none of the tickers could be evaluated",
                        "errors": errors}), 400

    # overlay chart: everyone normalized to $10k at the latest common start
    series[benchmark] = index_close
    weekly = {k: v.resample("W").last().dropna() for k, v in series.items()}
    start = max(s.index[0] for s in weekly.values())
    chart = {}
    for k, s in weekly.items():
        s = s[s.index >= start]
        if len(s) < 2:
            continue
        chart[k] = {"dates": [d.strftime("%Y-%m-%d") for d in s.index],
                    "values": [round(float(v / s.iloc[0] * 10_000), 2) for v in s]}
    return jsonify(_jsonable({"benchmark": benchmark, "rows": rows,
                              "errors": errors, "chart": chart,
                              "common_start": start.strftime("%Y-%m-%d")}))


@app.route("/api/evaluate/<ticker>")
@app.route("/api/evaluate/<ticker>/<benchmark>")
def api_evaluate(ticker: str, benchmark: str = "VTI"):
    ticker, benchmark = ticker.upper().strip(), benchmark.upper().strip()
    try:
        r = evaluate_ticker(ticker, benchmark, n_sims=10_000, console=console)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Couldn't evaluate {ticker}: {e}"}), 400
    payload = _jsonable({k: v for k, v in r.items() if k not in ("stock_close", "index_close")})
    payload["charts"] = chart_series(r)
    # calendar_returns Series inside hist isn't JSON-friendly; charts carry it instead
    payload["hist"].pop("calendar_returns", None)
    return jsonify(payload)


if __name__ == "__main__":
    print(f"Stock evaluator running → http://localhost:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
