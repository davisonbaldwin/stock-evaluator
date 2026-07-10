"""Terminal report (rich) and chart export (matplotlib)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------- formatting

def pct(x: Optional[float], signed: bool = False, digits: int = 1) -> str:
    if x is None:
        return "—"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x * 100:.{digits}f}%"


def money(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(x) >= scale:
            return f"${x / scale:.{digits}f}{unit}"
    return f"${x:,.{digits}f}"


def num(x: Optional[float], digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _score_color(s: float) -> str:
    if s >= 70:
        return "green"
    if s >= 50:
        return "yellow"
    return "red"


# ------------------------------------------------------------------ sections

def render(r: dict, console: Console) -> None:
    """`r` is the full result bundle from evaluate.evaluate_ticker."""
    hist, fund, bench = r["hist"], r["fund"], r["bench"]
    fc, mc, score = r["forecast"], r["monte_carlo"], r["score"]
    ticker, benchmark = r["ticker"], r["benchmark"]

    name = fund.get("name") or ticker
    console.print()
    console.print(Panel(
        f"[bold]{name}[/bold] ({ticker})  ·  {fund.get('sector') or '?'} / {fund.get('industry') or '?'}\n"
        f"Price [bold]{money(hist['last_price'])}[/bold]  ·  Market cap {money(fund['market_cap'], 1)}  ·  "
        f"Data since {hist['first_date']:%Y-%m-%d} ({hist['years_of_data']:.1f}y)\n"
        f"Composite score [bold {_score_color(score['total'])}]{score['total']:.0f}/100[/bold {_score_color(score['total'])}]"
        f"  ·  {score['verdict']}",
        title=f"[bold cyan]{ticker} evaluation[/bold cyan]", border_style="cyan"))

    # --- historical performance vs index
    t = Table(title=f"Historical performance (total return, dividends reinvested)",
              box=box.SIMPLE_HEAVY)
    t.add_column("CAGR", style="bold")
    for col in ("1y", "3y", "5y", "10y", "15y", "max"):
        t.add_column(col, justify="right")
    t.add_row(ticker, *[pct(hist["cagr"].get(k), signed=True) for k in ("1y", "3y", "5y", "10y", "15y", "max")])
    idx_cagr = bench["index_cagr"]
    t.add_row(benchmark, *[pct(idx_cagr.get(k), signed=True) for k in ("1y", "3y", "5y", "10y")],
              "—", pct(bench["index_cagr_common"], signed=True))
    console.print(t)

    mdd = hist["max_drawdown"]
    rec = f"recovered in {mdd['recovery_days']}d" if mdd.get("recovery_days") else "not yet recovered"
    t = Table(title="Risk profile", box=box.SIMPLE_HEAVY)
    t.add_column("Metric"); t.add_column(ticker, justify="right"); t.add_column(benchmark, justify="right")
    t.add_row("Volatility (1y ann.)", pct(hist["volatility_1y"]), "")
    t.add_row("Sharpe (5y)", num(hist["sharpe_5y"]), "")
    t.add_row("Sortino (5y)", num(hist["sortino_5y"]), "")
    t.add_row("Calmar (5y)", num(hist["calmar_5y"]), "")
    t.add_row("Max drawdown (common window)", pct(bench["stock_mdd"]), pct(bench["index_mdd"]))
    t.add_row("Worst drawdown detail",
              f"{pct(mdd.get('depth'))} ({mdd['peak_date']:%Y-%m} → {mdd['trough_date']:%Y-%m}, {rec})", "")
    t.add_row("Current drawdown from ATH", pct(hist["current_drawdown"]), "")
    by, wy = hist["best_year"], hist["worst_year"]
    if by and wy:
        t.add_row("Best / worst calendar year",
                  f"{by[0]}: {pct(by[1], signed=True)}  /  {wy[0]}: {pct(wy[1], signed=True)}", "")
    console.print(t)

    t = Table(title="Rolling holding-period outcomes (all historical entry points)",
              box=box.SIMPLE_HEAVY)
    t.add_column("Hold"); t.add_column("Worst", justify="right"); t.add_column("Median", justify="right")
    t.add_column("Best", justify="right"); t.add_column("% positive", justify="right")
    t.add_column(f"% beat {benchmark}", justify="right")
    for label in ("1y", "3y", "5y"):
        roll = hist.get(f"rolling_{label}")
        win = (bench["rolling_wins"] or {}).get(label)
        if roll:
            t.add_row(label, pct(roll["worst"], signed=True), pct(roll["median"], signed=True),
                      pct(roll["best"], signed=True), pct(roll["pct_positive"]),
                      pct(win["win_rate"]) if win else "—")
    console.print(t)

    # --- fundamentals
    v, g, q = fund["valuation"], fund["growth"], fund["quality"]
    t = Table(title="Fundamentals", box=box.SIMPLE_HEAVY)
    t.add_column("Valuation"); t.add_column("", justify="right")
    t.add_column("Growth"); t.add_column("", justify="right")
    t.add_column("Quality"); t.add_column("", justify="right")
    yrs = g.get("statement_years") or 0
    rows = [
        ("P/E (trailing)", num(v["trailing_pe"], 1), "Revenue YoY", pct(g["revenue_growth_yoy"], True),
         "Gross margin", pct(q["gross_margin"])),
        ("P/E (forward)", num(v["forward_pe"], 1), "Earnings YoY", pct(g["earnings_growth_yoy"], True),
         "Operating margin", pct(q["operating_margin"])),
        ("PEG", num(v["peg"]), f"Revenue CAGR ({yrs} stmts)", pct(g["revenue_cagr_multi"], True),
         "Net margin", pct(q["net_margin"])),
        ("P/S", num(v["price_to_sales"], 1), f"Net income CAGR", pct(g["net_income_cagr_multi"], True),
         "FCF margin", pct(q["fcf_margin"])),
        ("EV/EBITDA", num(v["ev_to_ebitda"], 1), f"FCF CAGR", pct(g["fcf_cagr_multi"], True),
         "ROE", pct(q["roe"])),
        ("FCF yield", pct(v["fcf_yield"]), "", "", "Debt/equity", num(q["debt_to_equity"], 0)),
        ("Dividend yield", pct(v["dividend_yield"]), "", "", "Rule of 40", num(q["rule_of_40"], 0)),
    ]
    for row in rows:
        t.add_row(*row)
    console.print(t)

    # --- vs index
    capm, cap, act = bench["capm"], bench["capture"], bench["active"]
    t = Table(title=f"vs a total-market index investor ({benchmark})", box=box.SIMPLE_HEAVY)
    t.add_column("Metric"); t.add_column("Value", justify="right"); t.add_column("Reading")
    if capm:
        t.add_row("Beta (5y monthly)", num(capm["beta"]),
                  "moves this much per 1.00 of market")
        t.add_row("Alpha (annualized)", pct(capm["alpha_annual"], True),
                  "return unexplained by market exposure")
        t.add_row("R² to index", pct(capm["r_squared"]), "how index-like it behaves")
    if cap:
        t.add_row("Up / down capture", f"{num(cap.get('up_capture'))} / {num(cap.get('down_capture'))}",
                  ">1 up & <1 down is the ideal")
    if act:
        t.add_row("Active return (5y ann.)", pct(act["active_return_annual"], True),
                  f"tracking error {pct(act['tracking_error'])}, IR {num(act['information_ratio'])}")
    t.add_row(f"$10k since {bench['common_start']:%Y-%m-%d}",
              f"{money(bench['final_10k_stock'], 0)} vs {money(bench['final_10k_index'], 0)}",
              f"{ticker} {pct(bench['stock_cagr_common'], True)}/yr vs {pct(bench['index_cagr_common'], True)}/yr")
    console.print(t)

    # --- forecasts
    t = Table(title="Scenario projections (price targets by annual-return scenario)",
              box=box.SIMPLE_HEAVY)
    t.add_column("Scenario", style="bold"); t.add_column("Rate/yr", justify="right")
    for h in ("1y", "3y", "5y", "10y"):
        t.add_column(h, justify="right")
    styles = {"bear": "red", "conservative": "yellow", "base": "white bold",
              "optimistic": "green", "bull": "bright_green"}
    for name_, rate in fc["rates"].items():
        prices = fc["prices"][name_]
        t.add_row(f"[{styles[name_]}]{name_}[/{styles[name_]}]", pct(rate, True),
                  *[money(prices[h], 0) for h in ("1y", "3y", "5y", "10y")])
    if fc.get("analyst_1y"):
        a = fc["analyst_1y"]
        t.add_row("[dim]analyst 1y range[/dim]", pct(a["implied_return"], True),
                  f"{money(a['low'], 0)}–{money(a['high'], 0)} (μ {money(a['mean'], 0)})", "", "", "")
    console.print(t)
    console.print(f"  [dim]base = 50/50 blend of capped historical CAGR ({pct(r['drift']['historical_capped'])}) "
                  f"and CAPM-implied return ({pct(r['drift']['capm_return'])})[/dim]")

    if mc.get("horizons"):
        t = Table(title=f"Monte Carlo — {mc['n_sims']:,} joint bootstrap paths "
                        f"(stock & index sampled together, drift re-centered to base)",
                  box=box.SIMPLE_HEAVY)
        t.add_column("Horizon")
        for col in ("p5", "p25", "median", "p75", "p95"):
            t.add_column(col, justify="right")
        t.add_column("P(loss)", justify="right")
        t.add_column(f"P(beat {benchmark})", justify="right")
        t.add_column("P(2x)", justify="right")
        for h, d in mc["horizons"].items():
            t.add_row(h, money(d["price_p5"], 0), money(d["price_p25"], 0),
                      money(d["price_p50"], 0), money(d["price_p75"], 0), money(d["price_p95"], 0),
                      pct(d["prob_loss"], digits=0), pct(d["prob_beat_index"], digits=0),
                      pct(d["prob_double"], digits=0))
        console.print(t)

    rdcf = r["reverse_dcf"]
    if rdcf:
        if rdcf.get("implied_growth") is not None:
            console.print(f"  [bold]Reverse DCF:[/bold] today's price implies "
                          f"[bold]{pct(rdcf['implied_growth'], True)}/yr[/bold] FCF growth for "
                          f"{rdcf['horizon']} years (discount {pct(rdcf['discount_rate'])}, "
                          f"terminal {pct(rdcf['terminal_growth'])}).")
        else:
            console.print(f"  [bold]Reverse DCF:[/bold] {rdcf.get('note')}")

    # --- score breakdown
    t = Table(title="Score breakdown", box=box.SIMPLE_HEAVY)
    t.add_column("Pillar"); t.add_column("Score", justify="right")
    t.add_column("Weight", justify="right"); t.add_column("Data coverage", justify="right")
    for pillar, (s, cov, w) in score["pillars"].items():
        t.add_row(pillar, f"[{_score_color(s)}]{s:.0f}[/{_score_color(s)}]",
                  f"{w:.0%}", f"{cov:.0%}")
    t.add_row("[bold]Composite[/bold]",
              f"[bold {_score_color(score['total'])}]{score['total']:.0f}[/bold {_score_color(score['total'])}]",
              "100%", f"{score['coverage']:.0%}")
    console.print(t)
    console.print("[dim]  Educational analysis from public data — not investment advice. "
                  "Forecasts are model outputs, not predictions.[/dim]\n")


def comparison_table(results: list, console: Console) -> None:
    """Side-by-side summary when evaluating multiple tickers."""
    t = Table(title="Side-by-side summary", box=box.HEAVY_HEAD)
    t.add_column("Metric", style="bold")
    for r in results:
        t.add_column(r["ticker"], justify="right")
    rows = [
        ("Composite score", lambda r: f"{r['score']['total']:.0f}/100"),
        ("5y CAGR", lambda r: pct(r["hist"]["cagr"].get("5y"), True)),
        ("Forward P/E", lambda r: num(r["fund"]["valuation"]["forward_pe"], 1)),
        ("Revenue growth YoY", lambda r: pct(r["fund"]["growth"]["revenue_growth_yoy"], True)),
        ("Volatility (1y)", lambda r: pct(r["hist"]["volatility_1y"])),
        ("Beta", lambda r: num(r["bench"]["capm"].get("beta"))),
        ("Alpha (ann.)", lambda r: pct(r["bench"]["capm"].get("alpha_annual"), True)),
        ("3y windows beating index", lambda r: pct(((r["bench"]["rolling_wins"] or {}).get("3y") or {}).get("win_rate"))),
        ("MC P(beat index, 5y)", lambda r: pct((r["monte_carlo"].get("horizons", {}).get("5y") or {}).get("prob_beat_index"), digits=0)),
        ("MC median 5y CAGR", lambda r: pct((r["monte_carlo"].get("horizons", {}).get("5y") or {}).get("cagr_p50"), True)),
        ("Implied FCF growth (rDCF)", lambda r: pct((r["reverse_dcf"] or {}).get("implied_growth"), True) if r["reverse_dcf"] else "—"),
    ]
    for label, fn in rows:
        t.add_row(label, *[fn(r) for r in results])
    console.print(t)


# -------------------------------------------------------------------- charts

def save_charts(r: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .benchmark import growth_of_10k
    from .history import calendar_year_returns, drawdown_series

    ticker, benchmark = r["ticker"], r["benchmark"]
    stock, index = r["stock_close"], r["index_close"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{ticker} vs {benchmark} — {pd.Timestamp.today():%Y-%m-%d}", fontsize=14)

    # 1. growth of $10k (log scale)
    ax = axes[0][0]
    g = growth_of_10k(stock, index)
    ax.plot(g.index, g["stock"], label=ticker, lw=1.4)
    ax.plot(g.index, g["index"], label=benchmark, lw=1.4)
    ax.set_yscale("log")
    ax.set_title(f"Growth of $10,000 since {g.index[0]:%Y-%m-%d} (log scale)")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. drawdowns
    ax = axes[0][1]
    df = pd.concat({"s": stock, "i": index}, axis=1).dropna()
    ax.fill_between(df.index, drawdown_series(df["s"]) * 100, 0, alpha=0.5, label=ticker)
    ax.plot(df.index, drawdown_series(df["i"]) * 100, lw=1.0, color="black", label=benchmark)
    ax.set_title("Drawdown from running peak (%)")
    ax.legend(); ax.grid(alpha=0.3)

    # 3. calendar-year returns
    ax = axes[1][0]
    cs = calendar_year_returns(df["s"]).tail(15)
    ci = calendar_year_returns(df["i"]).reindex(cs.index)
    x = np.arange(len(cs))
    ax.bar(x - 0.2, cs.values * 100, width=0.4, label=ticker)
    ax.bar(x + 0.2, ci.values * 100, width=0.4, label=benchmark)
    ax.set_xticks(x); ax.set_xticklabels(cs.index, rotation=45, fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Calendar-year returns (%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # 4. forecast cone: scenarios + Monte Carlo percentiles
    ax = axes[1][1]
    last_price = r["hist"]["last_price"]
    horizons = [0, 1, 3, 5, 10]
    styles = {"bear": "tab:red", "conservative": "tab:orange", "base": "black",
              "optimistic": "tab:green", "bull": "tab:cyan"}
    for name, rate in r["forecast"]["rates"].items():
        ax.plot(horizons, [last_price * (1 + rate) ** h for h in horizons],
                color=styles[name], lw=1.3, label=f"{name} ({rate * 100:+.0f}%/yr)")
    mch = r["monte_carlo"].get("horizons", {})
    if mch:
        hs = [int(k[:-1]) for k in mch]
        for p, marker in [("price_p5", "v"), ("price_p50", "o"), ("price_p95", "^")]:
            ax.scatter(hs, [mch[f"{h}y"][p] for h in hs], color="purple", marker=marker, s=28, zorder=5,
                       label=f"MC {p.split('_')[1]}" )
    ax.set_yscale("log")
    ax.set_title("Forecast cone: scenarios (lines) + Monte Carlo (markers)")
    ax.set_xlabel("years ahead")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{ticker}_{pd.Timestamp.today():%Y%m%d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
