"""Portfolio risk engine + kill switch — the book-level safety overlay.

Per-trade stops protect a *single* position. This module protects the *whole book*. It is the
"non-negotiable overlay" from the strategy brief: before any new order is opened, the engine
checks four things and can throttle or halt trading:

  1. Daily loss limit   — once today's P&L <= -daily_loss_limit_pct, stop opening NEW positions
                          for the rest of the session (existing positions keep their brackets).
  2. Drawdown control   — peak-to-now drawdown >= dd_derisk_pct halves new-position sizing;
                          >= dd_halt_pct stops new entries entirely until equity recovers.
  3. Concentration cap  — no single new position worth more than max_position_pct of equity.
  4. Kill switch        — repeated infrastructure failures (broker/account/data errors) across
                          runs flip a hard kill switch that blocks trading until clean runs pass.

State (peak equity + failure counters) is persisted in risk_state.json so it survives across the
stateless GitHub Actions runs. Everything is gated by cfg.risk_engine_enabled and wrapped so a
failure here can never block reconciliation or break the build — it fails OPEN to "normal" only
for *evaluation errors*, never for an actual tripped limit.
"""
from __future__ import annotations

import json
import os

from config import Config

STATE_FILE = os.getenv("RISK_STATE_FILE", "risk_state.json")


def _load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _peak_from_history(history) -> float:
    """Best-effort high-water mark from the broker equity curve."""
    try:
        eq = (history or {}).get("equity") or []
        vals = [float(v) for v in eq if v]
        return max(vals) if vals else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def note_failure(cfg: Config) -> None:
    """Record an infrastructure failure (broker/account unreachable) toward the kill switch.
    Called from the early-return error paths in paper.run so transient outages are counted."""
    if not getattr(cfg, "risk_engine_enabled", True):
        return
    st = _load_state()
    st["fail_streak"] = int(st.get("fail_streak", 0)) + 1
    st["clean_streak"] = 0
    if st["fail_streak"] >= int(getattr(cfg, "kill_switch_trips", 3)):
        st["killed"] = True
    _save_state(st)


def _recent_loss_streak(path: str = "track_record.json") -> int:
    """Trailing count of consecutive *losing* closed theses in the track record. Read-only (the
    Action owns this file). Ordered by entry date as a best-available proxy — this is an advisory
    throttle, not accounting. Any error -> 0 (no cooldown). Resets to 0 the moment a win lands."""
    try:
        import json
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return 0
        resolved = [t for t in rows if t.get("status") in ("win", "loss")]
        resolved.sort(key=lambda t: str(t.get("advised_ts") or t.get("advised_date") or ""))
        streak = 0
        for t in reversed(resolved):
            if t.get("status") == "loss":
                streak += 1
            else:
                break
        return streak
    except Exception:  # noqa: BLE001 - advisory; never break the risk engine
        return 0


def evaluate(cfg: Config, equity: float, last_equity: float,
             positions: list[dict] | None, history=None, errored: bool = False) -> dict:
    """Assess book-level risk and return a gate dict the caller uses to throttle new entries:

        {
          enabled, state,              # "normal" | "derisk" | "halt" | "killed" | "off"
          ok_to_open,                  # may we open NEW positions this run?
          size_scale,                  # multiply new-position size by this (1.0 / 0.5 / 0.0)
          max_position_value,          # hard $ cap per single new position (concentration)
          drawdown_pct, peak_equity,
          day_pl_pct,
          reasons, warnings,           # human-readable explanations for the dashboard
        }

    Never raises; on its own internal error it returns an "off"/permissive result so it can
    never wedge the runner. A genuinely tripped limit, by contrast, IS enforced.
    """
    if not getattr(cfg, "risk_engine_enabled", True):
        return {"enabled": False, "state": "off", "ok_to_open": True, "size_scale": 1.0,
                "max_position_value": None, "drawdown_pct": None, "peak_equity": None,
                "day_pl_pct": None, "reasons": [], "warnings": []}
    try:
        st = _load_state()

        # --- kill switch bookkeeping (counts infra failures across runs) ---
        if errored:
            st["fail_streak"] = int(st.get("fail_streak", 0)) + 1
            st["clean_streak"] = 0
        else:
            st["fail_streak"] = 0
            st["clean_streak"] = int(st.get("clean_streak", 0)) + 1
        if st.get("fail_streak", 0) >= int(getattr(cfg, "kill_switch_trips", 3)):
            st["killed"] = True
        # auto-reset after enough consecutive clean runs
        if st.get("killed") and st.get("clean_streak", 0) >= int(getattr(cfg, "kill_switch_cooldown_runs", 3)):
            st["killed"] = False

        # --- equity, peak, drawdown ---
        equity = float(equity or 0)
        last_equity = float(last_equity or equity)
        peak = max(float(st.get("peak_equity", 0) or 0), _peak_from_history(history), equity)
        st["peak_equity"] = round(peak, 2)
        dd_pct = round((peak - equity) / peak * 100, 2) if peak > 0 else 0.0
        day_pl_pct = round((equity - last_equity) / last_equity * 100, 2) if last_equity else 0.0

        reasons: list[str] = []
        warnings: list[str] = []
        state = "normal"
        size_scale = 1.0
        ok_to_open = True

        killed = bool(st.get("killed"))
        if killed:
            state = "killed"; ok_to_open = False; size_scale = 0.0
            reasons.append(f"Kill switch ON — {st.get('fail_streak', 0)} consecutive run failures. "
                           f"Trading paused until {int(getattr(cfg, 'kill_switch_cooldown_runs', 3))} clean runs.")

        # --- drawdown control ---
        dd_halt = float(getattr(cfg, "dd_halt_pct", 10.0))
        dd_derisk = float(getattr(cfg, "dd_derisk_pct", 8.0))
        if dd_pct >= dd_halt:
            if state != "killed":
                state = "halt"
            ok_to_open = False; size_scale = 0.0
            reasons.append(f"Drawdown {dd_pct:.1f}% ≥ {dd_halt:.0f}% halt line — no new positions until equity recovers.")
        elif dd_pct >= dd_derisk:
            if state == "normal":
                state = "derisk"
            size_scale = min(size_scale, 0.5)
            warnings.append(f"Drawdown {dd_pct:.1f}% ≥ {dd_derisk:.0f}% — new positions sized at half until recovery.")

        # --- daily loss limit ---
        dll = float(getattr(cfg, "daily_loss_limit_pct", 3.0))
        if day_pl_pct <= -dll:
            ok_to_open = False
            if state == "normal":
                state = "halt"
            reasons.append(f"Daily P&L {day_pl_pct:.1f}% ≤ -{dll:.0f}% loss limit — done opening for today.")
        elif day_pl_pct <= -dll * 0.66:
            warnings.append(f"Daily P&L {day_pl_pct:.1f}% nearing the -{dll:.0f}% loss limit.")

        # --- losing-trade cooldown (consecutive losing closed theses) ---
        loss_streak = _recent_loss_streak()
        halt_n = int(getattr(cfg, "loss_streak_halt", 4))
        derisk_n = int(getattr(cfg, "loss_streak_derisk", 3))
        if loss_streak >= halt_n:
            ok_to_open = False
            if state == "normal":
                state = "halt"
            reasons.append(f"{loss_streak} losing trades in a row — cooldown; no new positions until a win breaks the streak.")
        elif loss_streak >= derisk_n:
            if state == "normal":
                state = "derisk"
            size_scale = min(size_scale, 0.5)
            warnings.append(f"{loss_streak} losing trades in a row — new positions sized at half until the streak breaks.")

        # --- drawdown-recovery ladder: after a drawdown, don't snap back to full size — ease in.
        # Halt drops the size step to 0; once drawdown clears we start at quarter size, then ratchet
        # 0.25 -> 0.5 -> 0.75 -> 1.0, stepping up ONLY after a clean (no losing-streak) and profitable
        # run. Stateful (recovery_step in risk_state.json); the step caps whatever size the gates allow.
        if getattr(cfg, "dd_recovery_ladder_enabled", True):
            step = float(st.get("recovery_step", 1.0))
            if dd_pct >= dd_halt:
                step = 0.0                                   # halted -> floor
            elif dd_pct >= dd_derisk:
                step = min(step, 0.5)
            else:                                            # recovered below the de-risk line
                if step < 0.25:
                    step = 0.25                              # first run out of a halt: quarter size
                elif step < 1.0 and loss_streak == 0 and day_pl_pct > 0:
                    step = min(1.0, round(step + 0.25, 2))   # ratchet up on a clean, green run
            st["recovery_step"] = step
            if 0.0 < step < 1.0:
                size_scale = min(size_scale, step)
                if state == "normal":
                    state = "recovering"
                warnings.append(f"Recovering from drawdown — easing size back in at {int(step*100)}% "
                                "(steps up after a clean, profitable run).")
        else:
            step = 1.0

        # --- concentration cap (per single new position) ---
        max_pos_pct = float(getattr(cfg, "max_position_pct", 15.0))
        max_position_value = round(equity * max_pos_pct / 100, 2) if equity else None

        _save_state(st)
        return {
            "enabled": True, "state": state, "ok_to_open": ok_to_open, "size_scale": size_scale,
            "max_position_value": max_position_value, "max_position_pct": max_pos_pct,
            "drawdown_pct": dd_pct, "peak_equity": round(peak, 2), "loss_streak": loss_streak,
            "recovery_step": step,
            "day_pl_pct": day_pl_pct, "reasons": reasons, "warnings": warnings,
        }
    except Exception as e:  # noqa: BLE001 — evaluation error must fail OPEN, never wedge the runner
        return {"enabled": True, "state": "off", "ok_to_open": True, "size_scale": 1.0,
                "max_position_value": None, "max_position_pct": None, "drawdown_pct": None,
                "peak_equity": None, "day_pl_pct": None, "reasons": [],
                "warnings": [f"risk engine skipped: {str(e)[:80]}"]}


def cap_qty_to_concentration(qty: int, entry: float, max_position_value: float | None) -> int:
    """Trim a position's share count so its notional never exceeds the concentration cap."""
    if not max_position_value or not entry or entry <= 0:
        return qty
    cap = int(max_position_value / entry)
    return max(0, min(qty, cap))
