"""No-trade intelligence layer — sit on your hands when conditions are poor.

A signal firing is necessary but not sufficient. This layer is the explicit "should we even be
trading right now?" gate that the strategy brief asks for. It pulls together the scattered veto
conditions into one transparent read, at two levels:

  • market_gate(...)  — book-wide conditions that pause ALL new entries this run:
       major macro event imminent, abnormally high volatility, a deteriorating model track record,
       or a portfolio drawdown / kill-switch breach (from the risk engine).
  • symbol_veto(row)  — per-name reasons to skip a specific setup: earnings imminent, too illiquid,
       or the lower-timeframe signal flatly conflicting with the daily call.

Each condition is reported as ok / caution / block, so the dashboard can show exactly why the bot
is (or isn't) standing down. It only ever *blocks new entries* — it never force-closes open
positions, and it always defers to the rules-based risk engine. Gated by cfg.notrade_enabled.
"""
from __future__ import annotations

_TOP_TIER = ("consumer price index", "employment situation", "fomc", "federal open market",
             "pce", "personal consumption", "gross domestic product")


def _event_soon(calendar: dict | None, today: str):
    """Return (when, name) for the nearest top-tier macro release that is today/tomorrow, else None."""
    try:
        import datetime as dt
        econ = (calendar or {}).get("econ") or []
        t0 = dt.date.fromisoformat(today)
        best = None
        for e in econ:
            nm = (e.get("name") or "").lower()
            d = e.get("date")
            if not d or not any(k in nm for k in _TOP_TIER):
                continue
            try:
                days = (dt.date.fromisoformat(d) - t0).days
            except Exception:  # noqa: BLE001
                continue
            if 0 <= days <= 1 and (best is None or days < best[0]):
                best = (days, e.get("name"))
        if best:
            return ("today" if best[0] == 0 else "tomorrow", best[1])
    except Exception:  # noqa: BLE001
        pass
    return None


def market_gate(cfg, *, macro_posture=None, macro=None, calendar=None,
                track=None, risk=None, timing=None, book_risk=None, today: str = "") -> dict:
    """Book-wide no-trade read. Returns {block_new, reasons, cautions, checks}."""
    out = {"block_new": False, "reasons": [], "cautions": [], "checks": []}
    if not getattr(cfg, "notrade_enabled", True):
        return out

    def add(name, status, detail):
        out["checks"].append({"name": name, "status": status, "detail": detail})
        if status == "block":
            out["reasons"].append(detail)
            out["block_new"] = True
        elif status == "caution":
            out["cautions"].append(detail)

    # 1) Major macro event imminent
    ev = _event_soon(calendar, today)
    if ev:
        when, nm = ev
        if when == "today":
            add("Major macro event", "block", f"{nm} is due today — standing down on new entries through the release.")
        else:
            add("Major macro event", "caution", f"{nm} is due tomorrow — event risk; keep new size modest.")
    else:
        add("Major macro event", "ok", "No top-tier macro release in the next day.")

    # 2) Abnormal volatility
    vix = (macro or {}).get("vix")
    tags = {t.get("tag") for t in (macro_posture or {}).get("tags", [])}
    if vix is not None and vix >= float(getattr(cfg, "notrade_vix_block", 36)):
        add("Volatility", "block", f"VIX {vix} — panic-level volatility; no new positions until it calms.")
    elif "High-volatility" in tags:
        add("Volatility", "caution", "High-volatility regime — positions are auto-sized down; be selective.")
    else:
        add("Volatility", "ok", f"Volatility normal{f' (VIX {vix})' if vix is not None else ''}.")

    # 3) Deteriorating model performance
    n = (track or {}).get("resolved") or 0
    wr = (track or {}).get("win_rate")
    min_n = int(getattr(cfg, "notrade_perf_min_n", 25))
    floor = float(getattr(cfg, "notrade_perf_winrate", 35))
    if n >= min_n and wr is not None and wr < floor:
        add("Model performance", "block",
            f"Recent win rate {wr}% over {n} resolved trades is below {floor:.0f}% — edge looks broken; pause new trades.")
    elif n >= min_n and wr is not None and wr < floor + 10:
        add("Model performance", "caution", f"Win rate {wr}% over {n} trades is soft — trade smaller until it recovers.")
    elif n < min_n:
        add("Model performance", "ok", f"Still gathering a track record ({n} resolved) — performance veto inactive.")
    else:
        add("Model performance", "ok", f"Win rate {wr}% over {n} trades — healthy.")

    # 4) Risk engine (drawdown / daily loss / kill switch) — reflect its state here too
    if risk and risk.get("enabled"):
        state = risk.get("state")
        if state in ("halt", "killed") or risk.get("ok_to_open") is False:
            why = " | ".join(risk.get("reasons") or []) or "risk limit reached"
            add("Portfolio risk engine", "block", f"Risk engine standing down: {why}")
        elif state == "derisk":
            add("Portfolio risk engine", "caution", "Drawdown elevated — new positions at reduced size.")
        else:
            add("Portfolio risk engine", "ok", "Within all book-level risk limits.")

    # 5) Market timing (O'Neil FTD / distribution) — a confirmed correction BLOCKS new longs
    #    (the tape is in institutional distribution; wait for a Follow-Through Day). A pressure
    #    reading is a caution only, since exposure is already tilted down. Gated + fail-silent.
    if timing and getattr(cfg, "timing_gate_enabled", True):
        tstate = timing.get("state")
        if tstate == "correction":
            add("Market timing", "block",
                f"Indexes in a correction ({timing.get('dd_total', 0)} distribution days) — no new longs "
                "until a Follow-Through Day confirms a new uptrend.")
        elif tstate == "pressure":
            add("Market timing", "caution",
                "Distribution building on the indexes — new longs at reduced size; watch for a Follow-Through Day.")
        elif tstate == "confirmed":
            add("Market timing", "ok",
                f"Follow-Through Day confirmed the uptrend (quality {timing.get('ftd_quality', 0)}/100) — clear to add.")
        else:
            add("Market timing", "ok", "No distribution cluster — timing is not blocking new entries.")

    # 6) Portfolio heat — total open risk-to-stop across the whole book. One trade can look fine while
    #    the book as a whole is over-committed; this catches death-by-a-thousand-bets.
    if isinstance(book_risk, dict) and book_risk.get("heat_pct") is not None:
        heat = book_risk["heat_pct"]
        cap = book_risk.get("heat_cap_pct", 6.0)
        if heat >= cap * 1.5:
            add("Portfolio heat", "block",
                f"Book risk {heat:.1f}% of equity is well over the {cap:.0f}% heat cap — stop adding until open risk comes down.")
        elif heat >= cap:
            add("Portfolio heat", "caution",
                f"Book risk {heat:.1f}% of equity is at the {cap:.0f}% heat cap — new positions add real aggregate risk; be selective.")
        else:
            add("Portfolio heat", "ok", f"Book risk {heat:.1f}% of equity — within the {cap:.0f}% heat cap.")
        # correlation clusters: holdings that move together are one bet, not many
        clusters = book_risk.get("correlated_clusters") or []
        if clusters:
            biggest = max(clusters, key=len)
            add("Correlated holdings", "caution",
                f"{' + '.join(biggest)} move together (>0.75 correlation) — that's really one bet, not "
                f"{len(biggest)}; adding more of the same theme concentrates risk you can't see in per-trade sizing.")

    return out


def symbol_veto(row: dict, cfg) -> dict | None:
    """Per-name reason to skip a specific setup, or None if it's clear to trade.
    Returns {status, reason}. status is 'block' (skip) or 'caution' (flag but allow)."""
    if not getattr(cfg, "notrade_enabled", True):
        return None
    # earnings imminent
    ed = (row.get("fundamentals") or {}).get("earnings_days")
    if ed is not None and ed <= 1:
        return {"status": "block", "reason": f"earnings in {ed}d — binary event can gap through the stop"}
    # lower-timeframe signal flatly conflicts with the daily call
    if row.get("intraday_confirm") == "disagree":
        return {"status": "caution", "reason": "intraday trend disagrees with the daily call — mixed signal"}
    # too illiquid (mirrors the execution gate, surfaced as a no-trade reason)
    tier = (row.get("liquidity") or {}).get("tier")
    if tier == "illiquid":
        return {"status": "block", "reason": "too illiquid to fill cleanly"}
    return None
