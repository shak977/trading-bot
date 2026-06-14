"""Trade track record — log every BUY the tool advises, then grade it.

On each run we:
  1. Log any new BUY signal (entry, stop, target, date) it hasn't logged before.
  2. Re-check every still-open logged trade against real price action since it
     was advised: if price hit the target first -> WIN, if it hit the stop
     first -> LOSS, if it's been too long -> EXPIRED (closed at last price),
     otherwise still OPEN.
  3. Save the log to track_record.json and compute reliability stats.

This is a hypothetical paper record (no fees/slippage, stop assumed hit before
target on an ambiguous day) — a rough, honest gauge of how the calls play out,
not a brokerage statement. It accumulates over time, so it's thin at first.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from config import Config
from data import get_bars, synthetic_bars

PATH = os.getenv("TRACK_FILE", "track_record.json")
HOLD_LIMIT_DAYS = 90


def _load() -> list[dict]:
    try:
        with open(PATH) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def _save(rows: list[dict]) -> None:
    try:
        with open(PATH, "w") as f:
            json.dump(rows, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def run(signals: list[dict], cfg: Config, live: bool, today: str) -> dict:
    log = _load()
    by_id = {t["id"]: t for t in log}

    # 1) Log new actionable entries — fresh BUYs (long) and fresh SHORTs (short).
    for s in signals:
        if s.get("action") not in ("BUY", "SHORT"):
            continue
        p = s.get("plan", {})
        if p.get("stop") is None or p.get("target") is None:
            continue
        tid = f"{s['symbol']}:{today}"
        if tid not in by_id:
            # Record whether TradingView agreed at entry, so we can later measure if the
            # cross-check actually predicts winners (win rate when it agreed vs not).
            tv_align = None
            if s.get("tv"):
                try:
                    import tradingview as _tv
                    tv_align = _tv.alignment(s["tv"], s.get("direction", "LONG"))
                except Exception:  # noqa: BLE001
                    tv_align = None
            t = {
                "id": tid, "symbol": s["symbol"], "name": s.get("name", ""),
                "direction": s.get("direction", "LONG"),
                "advised_date": today, "entry": p["entry"], "stop": p["stop"],
                "target": p["target"], "rr": p.get("rr"),
                "conviction": (s.get("conviction") or {}).get("label"),
                "tv_align": tv_align,
                "status": "open",
            }
            by_id[tid] = t
            log.append(t)

    # 2) Evaluate open trades against price action since they were advised.
    for t in log:
        if t["status"] != "open":
            continue
        try:
            df = get_bars(t["symbol"], cfg) if live else synthetic_bars(t["symbol"], n=cfg.lookback_days)
            if df is None or not len(df):
                continue
            # Normalise the index to tz-naive so it compares cleanly with the
            # tz-naive advised_date (live Alpaca bars are tz-aware UTC).
            df = df.copy()
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            cutoff = pd.Timestamp(t["advised_date"])
            run_day = pd.Timestamp(today)
            # Only grade on FULLY-COMPLETED sessions strictly AFTER the advised day:
            #  - exclude the advised day's own bar (you'd enter at its close, so its
            #    intraday high/low already happened — counting them is look-ahead bias)
            #  - exclude today's bar, which may still be forming during an intraday run
            after = df[(df.index.normalize() > cutoff) & (df.index.normalize() < run_day)]
            is_short = t.get("direction") == "SHORT"
            for ts, row in after.iterrows():
                hi, lo = float(row["high"]), float(row["low"])
                # skip implausible single-bar prints (e.g. >35% intraday range) so a bad
                # data tick can't fake a stop/target hit
                if lo > 0 and (hi - lo) / lo > 0.35:
                    continue
                # Stop is checked first on an ambiguous day (conservative). For a SHORT the
                # geometry inverts: stop is ABOVE entry (hi hits it), target is BELOW (lo hits it).
                if is_short:
                    if hi >= t["stop"]:
                        t.update(status="loss", exit=t["stop"], exit_date=str(ts.date()))
                        break
                    if lo <= t["target"]:
                        t.update(status="win", exit=t["target"], exit_date=str(ts.date()))
                        break
                else:
                    if lo <= t["stop"]:
                        t.update(status="loss", exit=t["stop"], exit_date=str(ts.date()))
                        break
                    if hi >= t["target"]:
                        t.update(status="win", exit=t["target"], exit_date=str(ts.date()))
                        break
            if t["status"] == "open" and len(after):
                held = (after.index[-1] - cutoff).days
                if held >= HOLD_LIMIT_DAYS:
                    t.update(status="expired", exit=round(float(after["close"].iloc[-1]), 2),
                             exit_date=str(after.index[-1].date()))
            if t["status"] != "open" and "exit" in t:
                # Long profits when price rises; short profits when it falls.
                gain = (t["entry"] - t["exit"]) if is_short else (t["exit"] - t["entry"])
                t["return_pct"] = round(gain / t["entry"] * 100, 2)
                t["days_held"] = (pd.Timestamp(t["exit_date"]) - cutoff).days
        except Exception:  # noqa: BLE001 - never let one symbol break the whole tracker
            continue

    # Persist only on LIVE runs. Synthetic/dev runs grade against fake prices, so writing would
    # both corrupt the real record and dirty the git tree on every local build — never do it.
    if live:
        _save(log)
    return _stats(log)


def _stats(log: list[dict]) -> dict:
    resolved = [t for t in log if t["status"] in ("win", "loss", "expired")]
    wins = [t for t in resolved if t["status"] == "win"]
    losses = [t for t in resolved if t["status"] == "loss"]
    rets = [t["return_pct"] for t in resolved if "return_pct" in t]
    open_n = sum(1 for t in log if t["status"] == "open")
    recent = sorted(resolved, key=lambda t: t.get("exit_date", ""), reverse=True)[:15]
    win_rets = [t["return_pct"] for t in wins if "return_pct" in t]
    loss_rets = [t["return_pct"] for t in losses if "return_pct" in t]

    def _grp(items):
        rv = [t["return_pct"] for t in items if "return_pct" in t]
        w = sum(1 for t in items if t["status"] == "win")
        n = sum(1 for t in items if t["status"] in ("win", "loss", "expired"))
        return {"n": n, "win_rate": round(w / n * 100, 1) if n else None,
                "avg_return": round(sum(rv) / len(rv), 2) if rv else None}

    by_dir = {d: _grp([t for t in resolved if (t.get("direction") or "LONG") == d]) for d in ("LONG", "SHORT")}
    by_conv = {c: _grp([t for t in resolved if t.get("conviction") == c]) for c in ("High", "Medium", "Low")}
    # Does the TradingView cross-check actually help? Win rate when it agreed vs didn't.
    by_tv = {"agree": _grp([t for t in resolved if t.get("tv_align") == "agree"]),
             "not_agree": _grp([t for t in resolved if t.get("tv_align") in ("mixed", "oppose")])}
    return {
        "advised": len(log),
        "resolved": len(resolved),
        "open": open_n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else None,
        "avg_return": round(sum(rets) / len(rets), 2) if rets else None,
        "avg_win": round(sum(win_rets) / len(win_rets), 2) if win_rets else None,
        "avg_loss": round(sum(loss_rets) / len(loss_rets), 2) if loss_rets else None,
        # expectancy = average return per resolved call (the honest "edge per trade")
        "expectancy": round(sum(rets) / len(rets), 2) if rets else None,
        "by_direction": by_dir,
        "by_conviction": by_conv,
        "by_tv": by_tv,
        "recent": recent,
    }
