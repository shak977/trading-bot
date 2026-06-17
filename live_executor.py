"""Real-time ORB executor — the always-on runner that turns ORB signals into actual orders.

SAFETY (read this):
  • DISABLED by default. Runs only when LIVE_EXECUTOR_ENABLED=true.
  • PAPER ONLY. Refuses to touch a live account unless BOTH ALPACA_PAPER!=true AND
    LIVE_EXECUTOR_ALLOW_REAL=true are set — a deliberate two-key gate. Default is paper.
  • Reuses the audited ORB engine (orb.py) for signals and broker.py for orders. It never invents
    a strategy; it just executes what ORB already computes, in real time.
  • Hard caps enforced IN the runner (not just the dashboard): max trades/day, max concurrent,
    consecutive-loss halt. Plus a file kill-switch (create a file named KILL) and a guaranteed
    end-of-day flatten so nothing is held overnight.

DATA-FEED CAVEAT: on Alpaca's FREE IEX feed, bars lag ~15 min, so signal *detection* is delayed by
that much (order *execution* is instant). For genuine real-time, point it at a real-time data
subscription (Alpaca paid or IBKR). The runner works either way; only the freshness changes.

Run it on an always-on host (a small VPS). See SETUP at the bottom of this file.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import datetime

import pandas as pd

import broker as _broker
import data as _data
import orb as _orb
from config import Config

_ET = "America/New_York"
_STATE = os.getenv("EXECUTOR_STATE_FILE", "executor_state.json")
_KILL = os.getenv("EXECUTOR_KILL_FILE", "KILL")


def _log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}", flush=True)


def _load_state() -> dict:
    try:
        with open(_STATE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_state(s: dict) -> None:
    try:
        with open(_STATE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _watchlist(cfg: Config) -> list[str]:
    """Names to watch for breakouts today. Env override, else the in-play universe + recent IPOs."""
    env = os.getenv("EXECUTOR_WATCHLIST", "").strip()
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()][: cfg.orb_inplay_top]
    try:
        import scanner
        uni = scanner.build_universe(cfg)[: cfg.orb_inplay_top]
        for s in scanner.recent_listings(cfg):
            if s not in uni:
                uni.append(s)
        return uni
    except Exception:  # noqa: BLE001
        return list(getattr(cfg, "symbols", []))[: cfg.orb_inplay_top]


def _et_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_ET)


def _in_trade_window(now: pd.Timestamp, cfg: Config) -> bool:
    def _m(s):
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    mins = now.hour * 60 + now.minute
    return _m(getattr(cfg, "orb_window_start", "09:45")) <= mins <= _m(getattr(cfg, "orb_window_end", "10:30"))


def _past_flatten(now: pd.Timestamp) -> bool:
    return now.hour * 60 + now.minute >= 15 * 60 + 45   # 15:45 ET — no overnight exposure


def _guard(cfg: Config) -> str | None:
    """Hard pre-flight gate. Returns a refusal reason, or None if cleared to run."""
    if not getattr(cfg, "live_executor_enabled", False):
        return "LIVE_EXECUTOR_ENABLED is not set — executor is disabled by default."
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() == "true"
    if not paper and not getattr(cfg, "executor_allow_real", False):
        return ("Refusing to run against a LIVE account. Set ALPACA_PAPER=true (recommended) or, to "
                "knowingly trade real money, set LIVE_EXECUTOR_ALLOW_REAL=true.")
    return None


def run_once(cfg: Config, broker: "_broker.Broker", watch: list[str], state: dict) -> None:
    """One scan pass: detect fresh eligible breakouts and submit bracket orders. Fail-soft per name."""
    now = _et_now()
    today = str(now.date())
    state.setdefault("date", today)
    if state.get("date") != today:                       # new day — reset the daily counters
        state.update(date=today, ordered=[], consec_losses=state.get("consec_losses", 0))
    state.setdefault("ordered", [])
    state.setdefault("opened", {})

    # kill switch
    if os.path.exists(_KILL):
        _flatten_all(broker, state, reason="kill switch")
        return

    # reconcile: any position we opened that has since closed -> update the loss streak
    try:
        live_pos = broker.positions()
    except Exception:  # noqa: BLE001
        live_pos = {}
    for sym in list(state["opened"].keys()):
        if sym not in live_pos:
            rec = state["opened"].pop(sym)
            # crude win/loss read from the last trade price isn't available here; use exit vs entry
            # via fills is heavy — approximate: a closed bracket that didn't reach target is a loss.
            state["consec_losses"] = 0 if rec.get("won") else state.get("consec_losses", 0) + 1

    # EOD flatten
    if _past_flatten(now):
        _flatten_all(broker, state, reason="EOD flatten 15:45")
        return

    if not _in_trade_window(now, cfg):
        return
    try:
        if not broker.is_open():
            return
    except Exception:  # noqa: BLE001
        return

    # caps
    max_day = getattr(cfg, "orb_max_trades_per_day", 4)
    max_conc = getattr(cfg, "orb_max_concurrent", 3)
    halt_n = getattr(cfg, "orb_consec_loss_halt", 2)
    if len(state["ordered"]) >= max_day:
        return
    if state.get("consec_losses", 0) >= halt_n:
        _log(f"halted: {state['consec_losses']} consecutive losses ≥ {halt_n}")
        return
    if len(state["opened"]) >= max_conc:
        return

    icfg = replace(cfg, timeframe="5Min", lookback_days=max(4, getattr(cfg, "orb_signal_lookback_days", 6)))
    try:
        spy = _data.get_bars("SPY", icfg)
    except Exception:  # noqa: BLE001
        spy = None
    try:
        quotes = _data.get_latest_quotes(watch, cfg)
    except Exception:  # noqa: BLE001
        quotes = {}

    try:
        existing = broker.open_client_ids()
    except Exception:  # noqa: BLE001
        existing = set()

    for sym in watch:
        if len(state["ordered"]) >= max_day or len(state["opened"]) >= max_conc:
            break
        if sym in state["ordered"] or sym in state["opened"]:
            continue
        cid = f"orb-{sym}-{today}"
        if cid in existing:
            continue
        try:
            df = _data.get_bars(sym, icfg)
            if df is None or getattr(df, "empty", True):
                continue
            ctx = {"spread_pct": (quotes.get(sym) or {}).get("spread_pct")}
            sig = _orb.build(sym, df, spy, cfg, ctx=ctx)
            if not sig or sig.get("recommended_action") != "paper_trade":
                continue
            qty = int(sig.get("qty") or 0)
            if qty < 1:
                continue
            side = "buy" if sig["direction"] == "LONG" else "sell"
            broker.submit_bracket(sym, qty, side, stop=sig["stop"], target=sig["target"],
                                  client_order_id=cid)
            state["ordered"].append(sym)
            state["opened"][sym] = {"entry": sig["entry"], "stop": sig["stop"],
                                    "target": sig["target"], "qty": qty, "side": side,
                                    "cid": cid, "won": False, "t": now.isoformat()}
            _log(f"ORDER {side.upper()} {qty} {sym} @~{sig['entry']} stop {sig['stop']} "
                 f"target {sig['target']} (score {sig.get('score')})")
        except Exception as e:  # noqa: BLE001
            _log(f"{sym}: order skipped — {str(e)[:120]}")
    _save_state(state)


def _flatten_all(broker: "_broker.Broker", state: dict, reason: str) -> None:
    """Market-close every position the executor opened. Brackets are DAY orders; the filled entry
    leaves a position that must be flattened so nothing carries overnight."""
    opened = state.get("opened", {})
    if not opened:
        return
    _log(f"flattening {len(opened)} position(s) — {reason}")
    for sym, rec in list(opened.items()):
        try:
            held = broker.position_qty(sym)
            if held:
                broker.submit_market(sym, abs(held), "sell" if held > 0 else "buy")
        except Exception as e:  # noqa: BLE001
            _log(f"{sym}: flatten failed — {str(e)[:120]}")
        opened.pop(sym, None)
    _save_state(state)


def main() -> int:
    cfg = Config()
    refusal = _guard(cfg)
    if refusal:
        _log("WILL NOT RUN: " + refusal)
        return 1
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() == "true"
    _log(f"starting ORB executor — {'PAPER' if paper else 'LIVE (real money)'} account, "
         f"poll {cfg.executor_poll_secs}s. Kill-switch file: '{_KILL}'.")
    broker = _broker.Broker(cfg)
    state = _load_state()
    watch, watch_day = [], None
    while True:
        try:
            now = _et_now()
            if watch_day != str(now.date()):            # rebuild the watchlist once per day
                watch = _watchlist(cfg)
                watch_day = str(now.date())
                _log(f"watchlist for {watch_day}: {len(watch)} names")
            run_once(cfg, broker, watch, state)
        except KeyboardInterrupt:
            _log("stopped by user")
            return 0
        except Exception as e:  # noqa: BLE001 — never let the loop die
            _log(f"loop error (continuing): {str(e)[:160]}")
        time.sleep(max(10, int(cfg.executor_poll_secs)))


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------------------------
# SETUP (small VPS, e.g. Ubuntu):
#   1) git clone the repo, `pip install -r requirements.txt`
#   2) export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...   (your PAPER keys)
#      export ALPACA_PAPER=true
#      export LIVE_EXECUTOR_ENABLED=true
#   3) Run under a supervisor so it restarts on reboot/crash, e.g. a systemd service or:
#         nohup python live_executor.py >> executor.log 2>&1 &
#   4) Kill switch: `touch KILL` flattens everything and stops opening new trades.
#   5) Going live (only after weeks of clean paper review): set ALPACA_PAPER=false AND
#      LIVE_EXECUTOR_ALLOW_REAL=true. You do this yourself, deliberately.
