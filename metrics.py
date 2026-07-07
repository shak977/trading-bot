"""Realized performance & risk metrics for the traded book.

Pulled from the JoelLewis/finance_skills wealth-management plugin (performance-metrics + risk
skills) and adapted to this bot's track record. The dashboard already shows win rate and alpha
vs SPY; this adds the institutional read those skills teach — risk-ADJUSTED performance and
tail risk, computed from how each thesis actually resolved.

Each closed thesis is graded win/loss by whether it hit its target or its stop first, so its
realized outcome is an R-multiple: a win returns +rr (its planned reward:risk), a loss returns
−1R. That R-series is the return stream we score:

  * Expectancy      — average R per trade (the edge, in units of risk)
  * Profit factor   — gross win R / gross loss R
  * Payoff ratio    — average win / average loss
  * SQN             — System Quality Number = expectancy / std(R) × √n (Van Tharp) — the standard
                      risk-adjusted quality score for a TRADE-based system (>1.6 tradeable, >2.5 good)
  * Sortino (R)     — expectancy / downside deviation (target-0 semideviation, robust to constant stops)
  * Max drawdown    — deepest peak-to-trough of the cumulative-R equity curve (in R)
  * Recovery        — total R / max drawdown R (return-to-pain)
  * VaR / CVaR (95%) — historical tail risk: the 5th-percentile trade and the mean of the worst 5%

We deliberately DON'T annualize (Sharpe×√trades-per-year): this book runs many concurrent theses
at high daily throughput, so annualizing a per-trade series would be misleading. SQN is the
trade-native equivalent. Pure/offline; never raises (returns None on too little data).
"""
from __future__ import annotations

import numpy as np


def _trade_r(t: dict) -> float | None:
    """Realized R-multiple from how the thesis resolved: target hit = +rr, stop hit = −1R."""
    st = t.get("status")
    if st == "win":
        rr = t.get("rr")
        return float(rr) if rr else 1.0
    if st == "loss":
        return -1.0
    return None                                   # open / unresolved


def _trades_per_year(trades: list[dict]) -> float:
    """Estimate annual trade cadence from the span of advised dates (for annualization)."""
    ds = sorted(str(t.get("advised_date") or t.get("advised_ts") or "")[:10] for t in trades
                if t.get("advised_date") or t.get("advised_ts"))
    ds = [d for d in ds if len(d) == 10]
    if len(ds) < 2:
        return float(len(trades))                 # fallback: treat the whole set as ~1 year
    import datetime as _dt
    try:
        d0 = _dt.date.fromisoformat(ds[0]); d1 = _dt.date.fromisoformat(ds[-1])
        years = max((d1 - d0).days / 365.25, 1 / 365.25)
        return len(trades) / years
    except Exception:  # noqa: BLE001
        return float(len(trades))


def performance(trades: list[dict], min_n: int = 10) -> dict | None:
    """Full performance & risk read from resolved theses. Returns None if too few to be meaningful."""
    if not trades:
        return None
    rs = [r for t in trades if (r := _trade_r(t)) is not None]
    n = len(rs)
    if n < min_n:
        return None
    a = np.array(rs, float)
    wins = a[a > 0]
    losses = a[a < 0]
    win_rate = round(len(wins) / n * 100, 1)
    expectancy = round(float(a.mean()), 3)                       # avg R per trade
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
    payoff = round(float(wins.mean()) / abs(float(losses.mean())), 2) if len(wins) and len(losses) else None

    std = float(a.std(ddof=1)) if n > 1 else 0.0
    # System Quality Number (Van Tharp): the trade-native risk-adjusted quality score.
    sqn = round(float(expectancy / std * np.sqrt(n)), 2) if std > 0 else None
    sharpe_per_trade = round(expectancy / std, 3) if std > 0 else None
    # Downside deviation to a 0R target (semideviation) — robust even when every stop is exactly −1R.
    neg = np.minimum(a, 0.0)
    dd_dev = float(np.sqrt(np.mean(neg ** 2)))
    sortino = round(expectancy / dd_dev, 2) if dd_dev > 0 else None

    eq = np.cumsum(a)                                            # cumulative-R equity curve
    peak = np.maximum.accumulate(eq)
    max_dd = round(float((peak - eq).max()), 2)                 # in R
    total_r = float(eq[-1])
    recovery = round(total_r / max_dd, 2) if max_dd > 0 else None   # return-to-pain

    var5 = float(np.percentile(a, 5))                          # historical 95% VaR (per-trade, in R)
    tail = a[a <= var5]
    cvar5 = float(tail.mean()) if len(tail) else var5           # Expected Shortfall

    # Kelly criterion: optimal fraction of bankroll to risk per trade given win prob + payoff.
    # f* = W − (1−W)/payoff. Negative/zero = no edge → don't size up. Half-Kelly is the practical
    # recommendation (full Kelly is too volatile). Expressed as % of equity to risk per trade.
    kelly = None
    half_kelly = None
    if payoff and payoff > 0:
        W = win_rate / 100.0
        f = W - (1 - W) / payoff
        kelly = round(max(0.0, f) * 100, 1)
        half_kelly = round(kelly / 2, 1)

    return {
        "n": n, "win_rate": win_rate, "expectancy_r": expectancy,
        "profit_factor": profit_factor, "payoff": payoff,
        "sqn": sqn, "sharpe_per_trade": sharpe_per_trade, "sortino": sortino,
        "max_drawdown_r": max_dd, "total_r": round(total_r, 2), "recovery": recovery,
        "trades_per_year": round(_trades_per_year(trades), 1),
        "var95_r": round(var5, 2), "cvar95_r": round(cvar5, 2),
        "kelly_pct": kelly, "half_kelly_pct": half_kelly,
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else None,
    }


def monte_carlo(trades: list[dict], sims: int = 5000, horizon: int = 100,
                seed: int = 7, min_n: int = 20) -> dict | None:
    """Bootstrap the historical trade distribution forward to estimate forward risk. Resamples R
    (with replacement) over `horizon` future trades across `sims` paths, and reports the drawdown +
    terminal-equity distribution — a Monte Carlo VaR / risk-of-ruin read (finance-skills). Pure."""
    rs = [r for t in trades if (r := _trade_r(t)) is not None]
    if len(rs) < min_n:
        return None
    a = np.array(rs, float)
    _eq = np.cumsum(a)
    hist_maxdd = float((np.maximum.accumulate(_eq) - _eq).max())
    rng = np.random.default_rng(seed)
    draws = rng.choice(a, size=(sims, horizon), replace=True)
    eq = np.cumsum(draws, axis=1)
    running_peak = np.maximum.accumulate(eq, axis=1)
    maxdd = (running_peak - eq).max(axis=1)                     # per-path max drawdown (R)
    terminal = eq[:, -1]                                        # per-path terminal equity (R)
    return {
        "sims": sims, "horizon": horizon,
        "median_maxdd_r": round(float(np.percentile(maxdd, 50)), 1),
        "p95_maxdd_r": round(float(np.percentile(maxdd, 95)), 1),
        "median_terminal_r": round(float(np.percentile(terminal, 50)), 1),
        "p05_terminal_r": round(float(np.percentile(terminal, 5)), 1),
        "prob_losing_r_pct": round(float((terminal < 0).mean()) * 100, 1),
        "prob_exceeds_hist_dd_pct": round(float((maxdd > hist_maxdd).mean()) * 100, 1),
        "hist_maxdd_r": round(hist_maxdd, 1),
    }


def from_file(path: str = "track_record.json", min_n: int = 10) -> dict | None:
    """Convenience: load the track record and score it. Never raises."""
    try:
        import json
        with open(path) as f:
            rows = json.load(f)
        rows = rows if isinstance(rows, list) else []
        p = performance(rows, min_n=min_n)
        if p is not None:
            mc = monte_carlo(rows)
            if mc is not None:
                p["montecarlo"] = mc
        return p
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    import json
    print(json.dumps(from_file(), indent=2))
