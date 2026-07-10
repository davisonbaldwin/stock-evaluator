#!/usr/bin/env python3
"""Tech Stock Evaluator — historical analysis, forecasts, and index benchmarking.

Usage:
    python evaluate.py NVDA
    python evaluate.py AAPL MSFT GOOGL --sims 20000
    python evaluate.py CRM --benchmark SPY --json out.json --no-charts
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console

from evaluator import benchmark as bench_mod
from evaluator import data, forecast, fundamentals, history, report, score as score_mod

ROOT = Path(__file__).resolve().parent


def evaluate_ticker(ticker: str, benchmark: str, n_sims: int, console: Console) -> dict:
    console.print(f"[dim]Fetching data for {ticker}…[/dim]")
    stock_close = data.fetch_history(ticker)["Close"]
    index_close = data.fetch_history(benchmark)["Close"]
    info = data.fetch_info(ticker)
    fins = data.fetch_financials(ticker)
    targets = data.fetch_analyst_targets(ticker)
    rf = data.risk_free_rate()

    hist = history.analyze(stock_close, rf)
    fund = fundamentals.analyze(info, fins)
    bench = bench_mod.compare(stock_close, index_close, rf)

    drift = forecast.blended_drift(hist, bench["capm"], rf)
    index_drift_candidates = [bench["index_cagr"].get(k) for k in ("5y", "10y")
                              if bench["index_cagr"].get(k) is not None]
    index_drift = (sum(index_drift_candidates) / len(index_drift_candidates)
                   if index_drift_candidates else 0.08)
    index_drift = min(index_drift, 0.12)  # don't extrapolate a hot index decade either

    fc = forecast.scenario_projections(hist["last_price"], drift, hist["volatility_5y"] or 0.35,
                                       targets, index_drift)
    mc = forecast.monte_carlo(stock_close, index_close, drift, index_drift, n_sims=n_sims)
    rdcf = forecast.reverse_dcf(fund["market_cap"],
                                info.get("freeCashflow") if isinstance(info.get("freeCashflow"), (int, float)) else None)
    sc = score_mod.compute(hist, fund, bench, mc, rdcf)

    return {
        "ticker": ticker, "benchmark": benchmark, "risk_free": rf,
        "stock_close": stock_close, "index_close": index_close,
        "hist": hist, "fund": fund, "bench": bench,
        "drift": drift, "forecast": fc, "monte_carlo": mc,
        "reverse_dcf": rdcf, "score": sc,
    }


def _jsonable(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return None  # raw series excluded from JSON export
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if obj != obj else float(obj)
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+", help="stock symbols to evaluate")
    p.add_argument("--benchmark", default="VTI", help="index fund to compare against (default VTI)")
    p.add_argument("--sims", type=int, default=10_000, help="Monte Carlo path count (default 10000)")
    p.add_argument("--no-charts", action="store_true", help="skip PNG chart export")
    p.add_argument("--json", metavar="FILE", help="also write results as JSON")
    p.add_argument("--refresh", action="store_true", help="clear the data cache first")
    args = p.parse_args()

    if args.refresh:
        shutil.rmtree(ROOT / ".cache", ignore_errors=True)

    console = Console()
    results = []
    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            r = evaluate_ticker(ticker, args.benchmark.upper(), args.sims, console)
        except Exception as e:
            console.print(f"[red]Failed to evaluate {ticker}: {e}[/red]")
            continue
        report.render(r, console)
        if not args.no_charts:
            path = report.save_charts(r, ROOT / "reports")
            console.print(f"  [dim]Charts saved → {path}[/dim]\n")
        results.append(r)

    if len(results) > 1:
        report.comparison_table(results, console)

    if args.json and results:
        payload = {r["ticker"]: _jsonable({k: v for k, v in r.items()
                                           if k not in ("stock_close", "index_close")})
                   for r in results}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        console.print(f"[dim]JSON written → {args.json}[/dim]")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
