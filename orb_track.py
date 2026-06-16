"""ORB shadow tracker — grade the day-trade signals against their own session, in their OWN log.

Separate from the swing/intraday/paper records (track_record_orb.json): ORB is a different strategy
and learns from its own outcomes. No orders — a fills-free hypothetical, like the other shadow
trackers. Logs each eligible signal once, grades open ones against intraday bars (stop / target /
EOD flatten), records the 7 score-factors as pass/fail checks so attribution can learn which factors
predict ORB winners, and reports a risk-state (trades today, consecutive losses) for the hard caps.

Never raises — returns a best-effort summary so a bad symbol or feed blip can't break the build.
"""
from __future__ import annotations

import json
import os

import orb

PATH = os.getenv("TRACK_ORB_FILE", "track_record_orb.json")
_RESOLVED = ("win", "loss", "eod")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def _save(path, rows):
    try:
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def run(signals: list[dict], bars_by_sym: dict, today: str, cfg, *,
        path: str | None = None, regime: dict | None = None) -> dict:
    """Log + grade ORB signals. `signals` are today's ORB records; `bars_by_sym` maps symbol→its
    intraday bars. Returns a summary incl. risk_state for the caps. Never raises."""
    path = path or PATH
    log = _load(path)
    by_id = {t.get("id"): t for t in log}

    # 1) Log newly ELIGIBLE signals (score ≥ threshold) once per symbol per session.
    for s in signals or []:
        if s.get("recommended_action") != "paper_trade":
            continue
        tid = f"{s.get('symbol')}:{s.get('session', today)}"
        if tid in by_id:
            continue
        t = {
            "id": tid, "symbol": s.get("symbol"), "session": s.get("session", today),
            "direction": s.get("direction"), "window_min": s.get("window_min"),
            "entry": s.get("entry"), "stop": s.get("stop"), "target": s.get("target"),
            "rr": s.get("rr"), "score": s.get("score"), "entry_time": s.get("entry_time"),
            "checks": orb.score_checks(s),
            "catalyst_score": (s.get("score_components") or {}).get("catalyst"),
            "regime": (regime or {}).get("label"),
            "status": "open", "exit": None, "exit_time": None, "r": None,
        }
        by_id[tid] = t
        log.append(t)

    # 2) Grade open trades against the bars we have so far.
    for t in log:
        if t.get("status") != "open":
            continue
        day = bars_by_sym.get(t.get("symbol"))
        if day is None or getattr(day, "empty", True):
            continue
        exit_px, outcome, exit_ts, r = orb.simulate_exit(t, day)
        if outcome in _RESOLVED:
            t.update(status=outcome, exit=exit_px, exit_time=exit_ts, r=r)

    _save(path, log)

    # 3) Summary + risk-state (for the hard caps).
    todays = [t for t in log if t.get("session") == today]
    resolved = [t for t in log if t.get("status") in _RESOLVED]
    decided = [t for t in resolved if t.get("status") in ("win", "loss")]
    wins = [t for t in decided if t["status"] == "win"]
    win_rate = round(len(wins) / len(decided) * 100, 1) if decided else None
    # consecutive full-stop losses, most-recent first (across resolved trades, by exit time)
    consec = 0
    for t in sorted([t for t in resolved if t.get("exit_time")],
                    key=lambda x: x.get("exit_time"), reverse=True):
        if t["status"] == "loss":
            consec += 1
        else:
            break
    open_today = [t for t in todays if t.get("status") == "open"]
    risk_state = {
        "trades_today": len(todays),
        "open_today": len(open_today),
        "consec_losses": consec,
        "max_trades_per_day": getattr(cfg, "orb_max_trades_per_day", 4),
        "max_concurrent": getattr(cfg, "orb_max_concurrent", 3),
        "consec_loss_halt": getattr(cfg, "orb_consec_loss_halt", 2),
    }
    risk_state["trades_capped"] = risk_state["trades_today"] >= risk_state["max_trades_per_day"]
    risk_state["loss_halted"] = consec >= risk_state["consec_loss_halt"]
    risk_state["blocked"] = risk_state["trades_capped"] or risk_state["loss_halted"]
    return {"advised": len([t for t in log]), "resolved": len(resolved),
            "open": len([t for t in log if t.get("status") == "open"]),
            "win_rate": win_rate, "today": len(todays), "risk_state": risk_state}
