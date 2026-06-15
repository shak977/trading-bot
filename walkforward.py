"""Walk-forward + out-of-sample validation — the brief's overfitting controls.

A single in-sample backtest flatters itself: you see the whole history, so any parameter that
happened to fit the past looks good. This module answers the harder question — does the edge
survive on data the parameters were NOT chosen on?

Method (true walk-forward):
  1. Split each symbol's history into sequential folds.
  2. On each fold's IN-SAMPLE window, grid-search the moving-average pair and keep the best by
     Sharpe (this is the only thing we "optimize" — kept deliberately small to avoid curve-fitting).
  3. Apply those frozen parameters to the NEXT, unseen OUT-OF-SAMPLE window and record the result.
  4. Stitch every OOS window into one continuous out-of-sample equity curve.
  5. Compare in-sample vs out-of-sample, and run a sensitivity sweep so we can see the edge does
     not hinge on one fragile parameter value.

Everything is wrapped so a data problem yields None, never a broken build. Read-only — it never
places orders; it just grades the strategy honestly.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import run_backtest
from config import Config

# Small, sane MA grid. Small on purpose: a big grid is how you overfit a walk-forward.
DEFAULT_GRID: list[tuple[int, int]] = [(10, 30), (20, 50), (20, 100), (30, 80), (50, 150)]

# A compact, liquid basket to validate on (kept small so the in-build run stays fast).
DEFAULT_BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "JPM", "XLE"]


def _safe_metrics(df: pd.DataFrame, cfg: Config) -> dict | None:
    try:
        return run_backtest(df, cfg).metrics
    except Exception:  # noqa: BLE001
        return None


def _curve(df: pd.DataFrame, cfg: Config) -> pd.Series | None:
    try:
        return run_backtest(df, cfg).equity_curve.dropna()
    except Exception:  # noqa: BLE001
        return None


def walk_forward_symbol(df: pd.DataFrame, cfg: Config, n_folds: int = 4,
                        grid: list | None = None) -> dict | None:
    """Walk-forward one symbol. Returns per-fold + aggregate IS/OOS stats, or None if too short."""
    grid = grid or DEFAULT_GRID
    df = df.dropna()
    L = len(df)
    if L < 220:
        return None
    bounds = [int(L * i / (n_folds + 1)) for i in range(n_folds + 2)]

    folds: list[dict] = []
    oos_fold_returns: list[float] = []
    is_sharpes: list[float] = []
    for i in range(1, n_folds + 1):
        tr_end = bounds[i]
        te_end = bounds[i + 1]
        train = df.iloc[:tr_end]
        if (te_end - tr_end) < 20 or tr_end < 80:
            continue

        # --- optimize MA pair on the in-sample window (best Sharpe, needs real trades) ---
        best = None  # (sharpe, (fast, slow), metrics)
        for f, s in grid:
            if f >= s:
                continue
            m = _safe_metrics(train, replace(cfg, fast_ma=f, slow_ma=s))
            if not m or m["n_trades"] < 2:
                continue
            if best is None or m["sharpe"] > best[0]:
                best = (m["sharpe"], (f, s), m)
        if best is None:
            continue
        f, s = best[1]
        is_sharpes.append(best[0])

        # --- apply frozen params to the unseen window; read OOS return off the stitched curve ---
        curve = _curve(df.iloc[:te_end], replace(cfg, fast_ma=f, slow_ma=s))
        if curve is None or len(curve) < te_end - 2:
            continue
        try:
            start_eq = float(curve.iloc[tr_end])
            end_eq = float(curve.iloc[te_end - 1])
        except Exception:  # noqa: BLE001
            continue
        oos_ret = (end_eq / start_eq - 1) if start_eq > 0 else 0.0
        oos_fold_returns.append(oos_ret)
        folds.append({
            "train_bars": tr_end,
            "test_bars": te_end - tr_end,
            "params": f"{f}/{s}",
            "is_sharpe": round(best[0], 2),
            "oos_return_pct": round(oos_ret * 100, 2),
        })

    if not folds:
        return None
    oos_compound = float(np.prod([1 + r for r in oos_fold_returns]) - 1)
    return {
        "folds": folds,
        "oos_return_pct": round(oos_compound * 100, 2),
        "oos_avg_fold_pct": round(float(np.mean(oos_fold_returns)) * 100, 2),
        "oos_win_folds": int(sum(1 for r in oos_fold_returns if r > 0)),
        "oos_n_folds": len(oos_fold_returns),
        "is_avg_sharpe": round(float(np.mean(is_sharpes)), 2) if is_sharpes else None,
    }


def sensitivity(df: pd.DataFrame, cfg: Config, grid: list | None = None) -> list[dict]:
    """Full-sample sweep of the MA grid — shows whether the edge depends on one fragile setting."""
    grid = grid or DEFAULT_GRID
    out = []
    for f, s in grid:
        if f >= s:
            continue
        m = _safe_metrics(df, replace(cfg, fast_ma=f, slow_ma=s))
        if m:
            out.append({"params": f"{f}/{s}", "total_return_pct": round(m["total_return"] * 100, 1),
                        "sharpe": round(m["sharpe"], 2), "max_dd_pct": round(m["max_drawdown"] * 100, 1)})
    return out


def validate(cfg: Config, live: bool, symbols: list | None = None,
             n_folds: int = 4, bars_fn=None) -> dict | None:
    """Run walk-forward + sensitivity across a small basket and aggregate. Returns a dashboard
    dict, or None on failure / when there isn't enough data. Never raises."""
    if not getattr(cfg, "walkforward_enabled", True):
        return None
    try:
        symbols = symbols or DEFAULT_BASKET
        if bars_fn is None:
            if live:
                from data import get_bars as bars_fn  # noqa: N813
            else:
                from data import synthetic_bars
                bars_fn = lambda s: synthetic_bars(s, n=max(getattr(cfg, "lookback_days", 400), 500))  # noqa: E731

        per_symbol: list[dict] = []
        sens_pool: dict[str, list] = {}
        for sym in symbols:
            try:
                df = bars_fn(sym)
            except Exception:  # noqa: BLE001
                continue
            if df is None or len(df) < 220:
                continue
            wf = walk_forward_symbol(df, cfg, n_folds=n_folds)
            if wf:
                wf["symbol"] = sym
                per_symbol.append(wf)
            for row in sensitivity(df, cfg):
                sens_pool.setdefault(row["params"], []).append(row)

        if not per_symbol:
            return None

        # aggregate OOS across the basket
        oos_rets = [s["oos_avg_fold_pct"] for s in per_symbol]
        oos_compound = [s["oos_return_pct"] for s in per_symbol]
        win_folds = sum(s["oos_win_folds"] for s in per_symbol)
        tot_folds = sum(s["oos_n_folds"] for s in per_symbol)
        is_sharpes = [s["is_avg_sharpe"] for s in per_symbol if s.get("is_avg_sharpe") is not None]

        # aggregate sensitivity (mean across symbols per param)
        sens = []
        for params, rows in sens_pool.items():
            sens.append({
                "params": params,
                "total_return_pct": round(float(np.mean([r["total_return_pct"] for r in rows])), 1),
                "sharpe": round(float(np.mean([r["sharpe"] for r in rows])), 2),
                "max_dd_pct": round(float(np.mean([r["max_dd_pct"] for r in rows])), 1),
            })
        sens.sort(key=lambda r: -r["sharpe"])

        oos_pct = round(float(np.mean(oos_compound)), 2)
        oos_pos = round(win_folds / tot_folds * 100) if tot_folds else 0
        # honest verdict
        if oos_pos >= 55 and oos_pct > 0:
            verdict = ("The edge largely survives out-of-sample — most unseen windows were profitable. "
                       "Treat returns as real but modest; live slippage will shave them further.")
            grade = "holds up"
        elif oos_pos >= 45:
            verdict = ("Mixed out-of-sample — the edge is marginal and regime-dependent. Worth trading small "
                       "and watching, not sizing up on the in-sample numbers.")
            grade = "marginal"
        else:
            verdict = ("The edge mostly does NOT survive out-of-sample — in-sample results look like curve-fit. "
                       "Don't trust the backtest's headline return; this is exactly what OOS testing is for.")
            grade = "fragile"

        return {
            "symbols": [s["symbol"] for s in per_symbol],
            "per_symbol": per_symbol,
            "oos_avg_pct": oos_pct,
            "oos_pos_folds_pct": oos_pos,
            "oos_total_folds": tot_folds,
            "is_avg_sharpe": round(float(np.mean(is_sharpes)), 2) if is_sharpes else None,
            "sensitivity": sens,
            "grade": grade,
            "verdict": verdict,
        }
    except Exception as e:  # noqa: BLE001 - validation must never break the build
        return {"error": str(e)[:120]}
