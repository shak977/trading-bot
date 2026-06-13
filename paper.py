"""Optional auto paper-trading — turns the signal screen into a REAL, fills-based track record.

OFF by default. When PAPER_TRADE=true and the account is a PAPER account, each run will:
  1. Read the live paper account (equity, open positions, equity curve) from Alpaca — the
     source of truth for how the calls actually fill and play out (real slippage, real timing).
  2. Submit a bracket order (market entry + stop-loss + take-profit, as one OCO) for each
     fresh HIGH-conviction BUY/SHORT that isn't already open or already submitted today.
     Idempotent via client_order_id; capped per-run and in total; risk-sized; market-hours only.

Safety: refuses outright on a live account. Every order is wrapped so one failure can't break
the run or the dashboard build. A tiny local log (paper_orders.json) remembers what we've
submitted so re-runs don't double-fire and the UI can tie orders back to signals.
"""
from __future__ import annotations

import json
import math
import os

from config import Config

LOG = os.getenv("PAPER_FILE", "paper_orders.json")


def _load() -> list[dict]:
    try:
        with open(LOG) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def _save(rows: list[dict]) -> None:
    try:
        with open(LOG, "w") as f:
            json.dump(rows, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _qty(equity: float, buying_power: float, entry: float, stop: float, risk_pct: float) -> int:
    """Risk-based size: lose ~risk_pct of equity if the stop is hit, capped by buying power."""
    if not entry or entry <= 0:
        return 0
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    q = int((equity * risk_pct) / per_share)
    afford = int((buying_power * 0.9) / entry)
    q = min(q, afford)
    return max(q, 0)


def run(signals: list[dict], cfg: Config, today: str) -> dict | None:
    """Reconcile + (optionally) submit paper orders. Returns a dashboard-ready dict, or None
    when the feature is disabled. Never raises."""
    if not cfg.paper_trade:
        return None
    if not cfg.paper:
        return {"enabled": False, "reason": "Refusing to auto-trade a LIVE account — set ALPACA_PAPER=true."}
    try:
        from broker import Broker
        broker = Broker(cfg)
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "reason": f"broker unavailable: {str(e)[:120]}"}

    log = _load()
    by_cid = {t.get("client_id"): t for t in log}
    submitted_now: list[dict] = []
    notes: list[str] = []

    # --- account + positions (the real, live truth) ---
    try:
        acct = broker.account()
        equity = float(acct.equity)
        last_equity = float(getattr(acct, "last_equity", 0) or equity)
        buying_power = float(getattr(acct, "buying_power", 0) or 0)
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "reason": f"could not read account: {str(e)[:120]}"}

    positions = []
    try:
        positions = broker.positions_detail()
    except Exception:  # noqa: BLE001
        pass
    open_syms = {p["symbol"] for p in positions}
    history = broker.portfolio_history() or None
    try:
        existing_ids = broker.open_client_ids()
    except Exception:  # noqa: BLE001
        existing_ids = set()

    market_open = broker.is_open()

    # --- submit new brackets for fresh High-conviction signals ---
    if market_open and len(open_syms) < cfg.paper_max_open:
        candidates = [s for s in signals
                      if s.get("action") in ("BUY", "SHORT")
                      and (s.get("conviction") or {}).get("label") == "High"]
        # strongest first
        candidates.sort(key=lambda s: -((s.get("conviction") or {}).get("score_pct") or 0))
        new_count = 0
        for s in candidates:
            if new_count >= cfg.paper_max_new_per_run:
                break
            if len(open_syms) + new_count >= cfg.paper_max_open:
                break
            sym = s["symbol"]
            direction = s.get("direction", "LONG")
            if direction == "SHORT" and not cfg.paper_allow_shorts:
                continue
            if sym in open_syms:
                continue
            cid = f"sd-{sym}-{today}"
            if cid in existing_ids or cid in by_cid:
                continue
            plan = s.get("plan") or {}
            entry, stop, target = plan.get("entry"), plan.get("stop"), plan.get("target")
            if not (entry and stop and target):
                continue
            qty = _qty(equity, buying_power, entry, stop, cfg.paper_risk_pct)
            if qty < 1:
                notes.append(f"{sym}: skipped (size < 1 share at risk budget)")
                continue
            side = "buy" if direction != "SHORT" else "sell"
            try:
                broker.submit_bracket(sym, qty, side, stop, target, cid)
                rec = {"client_id": cid, "symbol": sym, "direction": direction,
                       "action": s["action"], "submitted_date": today, "qty": qty,
                       "entry_plan": entry, "stop": stop, "target": target,
                       "conviction": (s.get("conviction") or {}).get("score_pct"), "status": "open"}
                log.append(rec)
                by_cid[cid] = rec
                submitted_now.append(rec)
                new_count += 1
            except Exception as e:  # noqa: BLE001
                notes.append(f"{sym}: order rejected ({str(e)[:80]})")
    elif not market_open:
        notes.append("Market closed — no new orders this run.")

    # --- reconcile local log: mark entries closed once they leave the position book ---
    closed = 0
    for t in log:
        if t.get("status") == "open" and t.get("submitted_date", today) < today \
                and t["symbol"] not in open_syms:
            t["status"] = "closed"
            closed += 1
    _save(log)

    pl = round(equity - last_equity, 2)
    open_logged = sum(1 for t in log if t.get("status") == "open")
    return {
        "enabled": True,
        "paper": True,
        "market_open": market_open,
        "equity": round(equity, 2),
        "day_pl": pl,
        "day_pl_pct": round(pl / last_equity * 100, 2) if last_equity else 0.0,
        "buying_power": round(buying_power, 2),
        "positions": positions,
        "n_open": len(positions),
        "history": history,
        "submitted_now": submitted_now,
        "tracked_open": open_logged,
        "tracked_total": len(log),
        "tracked_closed": sum(1 for t in log if t.get("status") == "closed"),
        "notes": notes[:8],
    }
