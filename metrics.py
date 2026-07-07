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

    return {
        "n": n, "win_rate": win_rate, "expectancy_r": expectancy,
        "profit_factor": profit_factor, "payoff": payoff,
        "sqn": sqn, "sharpe_per_trade": sharpe_per_trade, "sortino": sortino,
        "max_drawdown_r": max_dd, "total_r": round(total_r, 2), "recovery": recovery,
        "trades_per_year": round(_trades_per_year(trades), 1),
        "var95_r": round(var5, 2), "cvar95_r": round(cvar5, 2),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else None,
    }


def from_file(path: str = "track_record.json", min_n: int = 10) -> dict | None:
    """Convenience: load the track record and score it. Never raises."""
    try:
        import json
        with open(path) as f:
            rows = json.load(f)
        return performance(rows if isinstance(rows, list) else [], min_n=min_n)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    import json
    print(json.dumps(from_file(), indent=2))
