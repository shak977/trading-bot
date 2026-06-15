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


def _qty(equity: float, buying_power: float, entry: float, stop: float,
         risk_pct: float, mult: float = 1.0) -> int:
    """Risk-based size: lose ~risk_pct*mult of equity if the stop is hit, capped by buying power."""
    if not entry or entry <= 0:
        return 0
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    q = int((equity * risk_pct * mult) / per_share)
    afford = int((buying_power * 0.9) / entry)
    q = min(q, afford)
    return max(q, 0)


def _round_trips(fills: list[dict]) -> list[dict]:
    """Match raw fills into CLOSED round-trips per symbol via FIFO lot matching.

    Handles longs and shorts, partial fills, bracket (stop/target) exits, and even a
    position that flips through flat. A buy opens a long / closes a short; a sell
    (incl. 'sell_short') opens a short / closes a long. Only fully-matched legs are
    emitted, each with realized P&L and per-trade % return. Open exposure is ignored.
    """
    from collections import defaultdict, deque

    open_lots: dict[str, deque] = defaultdict(deque)  # symbol -> FIFO of open lots
    trips: list[dict] = []
    for f in sorted(fills, key=lambda x: (x.get("time") or "")):
        sym, qty, price = f.get("symbol"), f.get("qty") or 0, f.get("price") or 0
        if not sym or qty <= 0 or price <= 0:
            continue
        is_buy = (f.get("side") or "").startswith("buy")
        lots = open_lots[sym]
        if lots and lots[0]["is_buy"] != is_buy:
            # opposite side of the open position -> this fill closes lots (FIFO)
            remaining = qty
            while remaining > 1e-9 and lots:
                lot = lots[0]
                match = min(remaining, lot["qty"])
                entry, exit_ = lot["price"], price
                if lot["is_buy"]:                       # long: bought low, sold high
                    pl, direction = (exit_ - entry) * match, "LONG"
                else:                                   # short: sold high, bought low
                    pl, direction = (entry - exit_) * match, "SHORT"
                ret = (pl / (entry * match) * 100) if entry else 0.0
                trips.append({
                    "symbol": sym, "direction": direction, "qty": round(match, 4),
                    "entry_price": round(entry, 4), "exit_price": round(exit_, 4),
                    "entry_time": lot["time"], "exit_time": f.get("time"),
                    "pl": round(pl, 2), "return_pct": round(ret, 2), "win": pl > 0,
                })
                lot["qty"] -= match
                remaining -= match
                if lot["qty"] <= 1e-9:
                    lots.popleft()
            if remaining > 1e-9:  # flipped through flat: leftover opens a new position
                lots.append({"qty": remaining, "price": price, "time": f.get("time"), "is_buy": is_buy})
        else:
            lots.append({"qty": qty, "price": price, "time": f.get("time"), "is_buy": is_buy})
    return trips


def _realized_stats(trips: list[dict]) -> dict:
    """Aggregate closed round-trips into a realized win-rate / avg-return summary."""
    n = len(trips)
    if not n:
        return {"n_trades": 0, "win_rate": None, "avg_return_pct": None,
                "avg_win": None, "avg_loss": None, "total_pl": 0.0, "recent": []}
    wins = [t for t in trips if t["win"]]
    losses = [t for t in trips if not t["win"]]
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    recent = sorted(trips, key=lambda t: (t.get("exit_time") or ""), reverse=True)[:12]
    return {
        "n_trades": n,
        "win_rate": round(len(wins) / n * 100, 1),
        "avg_return_pct": avg([t["return_pct"] for t in trips]),
        "avg_win": avg([t["return_pct"] for t in wins]),
        "avg_loss": avg([t["return_pct"] for t in losses]),
        "total_pl": round(sum(t["pl"] for t in trips), 2),
        "recent": recent,
    }


def manage_open_positions(broker, cfg: Config, log: list[dict]) -> list[str]:
    """LIVE exit management (only when cfg.manage_exits). For each open logged position:
    scale out half at partial_take_r (once), then trail the stop under price (never loosening).

    Idempotent via flags on the local log; every broker call is wrapped so a failure leaves the
    protective bracket intact (we never strip a stop). Never raises.
    """
    notes: list[str] = []
    if not cfg.manage_exits:
        return notes
    try:
        positions = broker.positions_detail()
    except Exception:  # noqa: BLE001
        return notes
    by_sym = {p["symbol"]: p for p in positions}
    for t in log:
        if t.get("status") != "open":
            continue
        p = by_sym.get(t["symbol"])
        if not p:
            continue
        side = "long" if t.get("direction", "LONG") != "SHORT" else "short"
        entry, stop, price = t.get("entry_plan"), t.get("stop"), p.get("price")
        if not (entry and stop and price):
            continue
        r = abs(entry - stop)
        # 1) partial take at >= partial_take_r (once)
        if cfg.partial_take_r > 0 and not t.get("partial_done") and p.get("qty", 0) >= 2:
            up = (price - entry) if side == "long" else (entry - price)
            if r > 0 and up >= cfg.partial_take_r * r:
                half = int(p["qty"]) // 2
                try:
                    broker.partial_close(t["symbol"], half, side)
                    t["partial_done"] = True
                    notes.append(f'{t["symbol"]}: scaled out {half} at ~{cfg.partial_take_r}R')
                except Exception as e:  # noqa: BLE001 - leave bracket intact
                    notes.append(f'{t["symbol"]}: partial failed ({str(e)[:60]})')
        # 2) trail the stop (never loosen); move to breakeven once a partial is banked
        if cfg.trail_atr_mult > 0:
            try:
                orders = broker.open_orders_for(t["symbol"])
                stop_leg = next((o for o in orders if (o.get("type") or "").startswith("stop")), None)
                if stop_leg:
                    cur = stop_leg.get("stop_price")
                    new_stop = max(stop, entry) if t.get("partial_done") else stop
                    if side == "long" and (cur is None or new_stop > cur):
                        broker.replace_stop(stop_leg["id"], new_stop)
                        notes.append(f'{t["symbol"]}: stop tightened to {new_stop:.2f}')
                    elif side == "short" and (cur is None or new_stop < cur):
                        broker.replace_stop(stop_leg["id"], new_stop)
                        notes.append(f'{t["symbol"]}: stop tightened to {new_stop:.2f}')
            except Exception as e:  # noqa: BLE001 - never strip a stop
                notes.append(f'{t["symbol"]}: stop amend skipped ({str(e)[:60]})')
    return notes


def run(signals: list[dict], cfg: Config, today: str, exposure_mult: float = 1.0,
        regime: dict | None = None, notrade_block: bool = False) -> dict | None:
    """Reconcile + (optionally) submit paper orders. Returns a dashboard-ready dict, or None
    when the feature is disabled. Never raises.

    ``exposure_mult`` is the macro-regime exposure scalar (<1 in risk-off, >1 in risk-on): it
    scales every new position's size so the macro backdrop controls how aggressively we deploy."""
    if not cfg.paper_trade:
        return None
    if not cfg.paper:
        return {"enabled": False, "reason": "Refusing to auto-trade a LIVE account — set ALPACA_PAPER=true."}
    try:
        from broker import Broker
        broker = Broker(cfg)
    except Exception as e:  # noqa: BLE001
        try:
            import portfolio_risk
            portfolio_risk.note_failure(cfg)
        except Exception:  # noqa: BLE001
            pass
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
        try:
            import portfolio_risk
            portfolio_risk.note_failure(cfg)
        except Exception:  # noqa: BLE001
            pass
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

    # --- portfolio risk engine: book-level gate before any new order (non-negotiable overlay) ---
    try:
        import portfolio_risk
        risk = portfolio_risk.evaluate(cfg, equity, last_equity, positions, history, errored=False)
    except Exception:  # noqa: BLE001
        risk = {"enabled": False, "state": "off", "ok_to_open": True, "size_scale": 1.0,
                "max_position_value": None, "reasons": [], "warnings": []}
    if risk.get("reasons"):
        notes.append("Risk: " + " | ".join(risk["reasons"]))
    elif risk.get("warnings"):
        notes.append("Risk: " + " | ".join(risk["warnings"]))
    if abs(float(exposure_mult or 1.0) - 1.0) >= 0.03:
        _em = float(exposure_mult)
        notes.append(f"Macro exposure: new positions sized {_em:.2f}× "
                     f"({'leaning in (risk-on)' if _em > 1 else 'pulled back (risk-off)'}).")

    if notrade_block:
        notes.append("No-trade layer: market conditions are poor — not opening new positions this run.")

    # --- submit new brackets for fresh High-conviction signals ---
    if market_open and risk.get("ok_to_open", True) and not notrade_block and len(open_syms) < cfg.paper_max_open:
        candidates = [s for s in signals
                      if s.get("action") in ("BUY", "SHORT")
                      and (s.get("conviction") or {}).get("label") == "High"]
        # best first — by the adaptive allocation rank when present, else by conviction
        candidates.sort(key=lambda s: -(s.get("rank_score")
                                        if s.get("rank_score") is not None
                                        else ((s.get("conviction") or {}).get("score_pct") or 0)))
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
            # per-name no-trade veto (earnings imminent, conflicting signals, illiquid)
            try:
                import notrade as _notrade
                _veto = _notrade.symbol_veto(s, cfg)
                if _veto and _veto.get("status") == "block":
                    notes.append(f"{sym}: skipped ({_veto['reason']})")
                    continue
            except Exception:  # noqa: BLE001
                pass
            plan = s.get("plan") or {}
            entry, stop, target = plan.get("entry"), plan.get("stop"), plan.get("target")
            if not (entry and stop and target):
                continue
            label = (s.get("conviction") or {}).get("label")
            score_pct = (s.get("conviction") or {}).get("score_pct")
            atr_pct = (s.get("context") or {}).get("atr_pct")
            from risk import risk_multiplier
            mult = risk_multiplier(label, atr_pct, cfg, score_pct=score_pct)
            mult *= float(risk.get("size_scale", 1.0))   # book-level throttle (drawdown de-risk)
            mult *= float(exposure_mult or 1.0)          # macro-regime exposure scalar (risk-on/off)
            qty = _qty(equity, buying_power, entry, stop, cfg.paper_risk_pct, mult=mult)
            # concentration cap: never let one position exceed max_position_pct of equity
            try:
                import portfolio_risk
                capped = portfolio_risk.cap_qty_to_concentration(qty, entry, risk.get("max_position_value"))
                if capped < qty:
                    notes.append(f"{sym}: trimmed to {capped} sh (concentration cap)")
                qty = capped
            except Exception:  # noqa: BLE001
                pass
            # liquidity gate: skip names too thin to trade; cap size vs average daily $ volume
            try:
                import liquidity as _liq
                _dv = s.get("dollar_volume")
                _liq_chk = _liq.practical(qty * entry, _dv, cfg,
                                          spread_bps=(s.get("liquidity") or {}).get("spread_bps"))
                if not _liq_chk["ok"] and (not _dv or _dv < cfg.min_dollar_volume):
                    notes.append(f"{sym}: skipped (too illiquid — {_liq_chk['note']})")
                    continue
                lcap = _liq.cap_qty_to_adv(qty, entry, _dv, cfg)
                if lcap < qty:
                    notes.append(f"{sym}: trimmed to {lcap} sh (liquidity / market-impact cap)")
                qty = lcap
            except Exception:  # noqa: BLE001
                pass
            if qty < 1:
                notes.append(f"{sym}: skipped (size < 1 share at risk budget)")
                continue
            side = "buy" if direction != "SHORT" else "sell"
            try:
                broker.submit_bracket(sym, qty, side, stop, target, cid)
                rec = {"client_id": cid, "symbol": sym, "direction": direction,
                       "action": s["action"], "submitted_date": today, "qty": qty,
                       "entry_plan": entry, "stop": stop, "target": target,
                       "conviction": (s.get("conviction") or {}).get("score_pct"),
                       # entry-time context for the feedback loop / future ML labels
                       "regime": (regime or {}).get("label"), "regime_score": (regime or {}).get("score"),
                       "exposure_mult": round(float(exposure_mult or 1.0), 3),
                       "liquidity_tier": (s.get("liquidity") or {}).get("tier"), "status": "open"}
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

    # --- live exit management (opt-in; amends stops / takes partials) ---
    if cfg.manage_exits:
        try:
            notes.extend(manage_open_positions(broker, cfg, log))
            _save(log)
        except Exception:  # noqa: BLE001
            pass

    # --- realized performance from actual fills (closed round-trips) ---
    try:
        realized = _realized_stats(_round_trips(broker.fills()))
    except Exception:  # noqa: BLE001
        realized = _realized_stats([])

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
        "realized": realized,
        "risk": risk,
        "notes": notes[:8],
    }
