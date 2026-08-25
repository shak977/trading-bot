"""Generate a self-contained HTML dashboard of the latest weekly signals.

Pipeline:
  1. Scan the market (live: Alpaca movers + most-active; synthetic: demo list).
  2. Run the MA/RSI strategy on each, compute a relative-volume flow proxy.
  3. Pull recent news for the flagged (BUY/SELL) names.
  4. Write dashboard.html (self-contained) and signals.json.

Data source:
  - Real Alpaca data when ALPACA_API_KEY/SECRET are set.
  - Deterministic synthetic data otherwise, clearly labelled SYNTHETIC.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import market
import scanner
from config import CONFIG


def _mode() -> str:
    if CONFIG.api_key and CONFIG.secret_key:
        return "PAPER" if CONFIG.paper else "LIVE"
    return "SYNTHETIC"


def _md_inline(s: str) -> str:
    """Render the small markdown subset the LLM briefs use (**bold**, *italic*, paragraph breaks)
    into HTML so briefs read as formatted copy, not raw asterisks."""
    import re
    s = (s or "").strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
    parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in parts) if parts else s


def _brief_panels(brief: str, sectors: list | None) -> str:
    """Market brief as a 2x2 quadrant of labelled panels + a colour-coded sector-strength strip."""
    import re
    _ic = {"The Mood": "compass", "What's Driving It": "bolt",
           "What the Screen Flags": "search", "What to Watch": "target"}

    def _inline(t):
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", t)
        t = re.sub(r"(?i)\b(risk-on|bullish|high-conviction|leading|winners?|strength|gains|outperform\w*|benefit\w*)\b",
                   r'<span class="mbhl up">\1</span>', t)
        t = re.sub(r"(?i)\b(risk-off|bearish|downgrade\w*|headwinds?|weakness|weak|fragilit\w*|fragile|sidelined)\b",
                   r'<span class="mbhl dn">\1</span>', t)
        return " ".join(t.split())

    parts = re.split(r"\*\*([^*]+?):\*\*", brief or "")
    panels = ""
    it = iter(parts[1:])
    for label, text in zip(it, it):
        label = label.strip()
        panels += (f'<div class="mbp"><div class="mbp-h">{_svg(_ic.get(label, "sparkle"), 14)} {label}</div>'
                   f'<div class="mbp-b">{_inline(text.strip())}</div></div>')
    if not panels:
        return f'<div class="bt-body">{_md_inline(brief)}</div>'
    chips = ""
    for s in (sectors or [])[:8]:
        nm = s.get("sector", "")
        if not nm or nm.startswith("Other"):
            continue
        pu = s.get("pct_up", 0)
        cls = "up" if pu >= 70 else ("dn" if pu < 50 else "flat")
        arr = "&uarr;" if pu >= 70 else ("&darr;" if pu < 50 else "&rarr;")
        chips += f'<span class="mbsec">{nm} <b class="{cls}">{pu}% {arr}</b></span>'
    strip = f'<div class="mbsecs">{chips}</div>' if chips else ""
    return f'<div class="mb-quad">{panels}</div>{strip}'


# --- Inline SVG icon set (Tabler/Lucide-style, 1.5 stroke, currentColor) ---------------------
# One reusable line-icon library, mirrored in JS as `ICON` for client-side templates. Every
# emoji that used to stand in for meaning is replaced by one of these so colour is inherited and
# the desk reads like a terminal, not a chat window.
_ICON_PATHS = {
    "ai":        '<path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-1 5.8V15a3 3 0 0 0 3 3M15 3a3 3 0 0 1 3 3 3 3 0 0 1 1 5.8V15a3 3 0 0 1-3 3"/><path d="M9 3v18M15 3v18"/>',
    "bot":       '<rect x="4" y="8" width="16" height="11" rx="2.5"/><path d="M12 8V5M9 3.5h6"/><circle cx="9" cy="13" r="1.1"/><circle cx="15" cy="13" r="1.1"/>',
    "bolt":      '<path d="M13 3 4 14h6l-1 7 9-11h-6z"/>',
    "bell":      '<path d="M6 9a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6"/><path d="M10.5 20a2 2 0 0 0 3 0"/>',
    "bank":      '<path d="M3 10 12 4l9 6"/><path d="M5 10v8M9 10v8M15 10v8M19 10v8M3 20h18"/>',
    "clipboard": '<rect x="6" y="4" width="12" height="17" rx="2"/><path d="M9 4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1z"/><path d="M9 11h6M9 15h4"/>',
    "clock":     '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "target":    '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r=".9"/>',
    "chart":     '<path d="M4 4v16h16"/><path d="M7 14l3-3 3 2 4-5"/>',
    "trend-up":  '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "trend-dn":  '<path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/>',
    "stop":      '<circle cx="12" cy="12" r="8.5"/><path d="M9 9h6v6H9z"/>',
    "octagon":   '<path d="M8 3h8l5 5v8l-5 5H8l-5-5V8z"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "star":      '<path d="m12 3 2.6 5.4 5.9.8-4.3 4.1 1 5.9-5.2-2.8-5.2 2.8 1-5.9L3.5 9.2l5.9-.8z"/>',
    "star-fill": '<path d="m12 3 2.6 5.4 5.9.8-4.3 4.1 1 5.9-5.2-2.8-5.2 2.8 1-5.9L3.5 9.2l5.9-.8z" fill="currentColor" stroke="none"/>',
    "search":    '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-3.5-3.5"/>',
    "refresh":   '<path d="M20 11a8 8 0 0 0-14-4.5L3 9"/><path d="M4 13a8 8 0 0 0 14 4.5L21 15"/><path d="M3 4v5h5M21 20v-5h-5"/>',
    "gear":      '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',
    "shield":    '<path d="M12 3 5 6v6c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6z"/>',
    "news":      '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9h6M7 13h10M7 17h10"/><rect x="14" y="8" width="4" height="3" rx=".6"/>',
    "check":     '<path d="M4 12.5 9 17.5 20 6.5"/>',
    "x":         '<path d="M6 6l12 12M18 6 6 18"/>',
    "warn":      '<path d="M12 3 2.5 20h19z"/><path d="M12 10v4M12 17.5h.01"/>',
    "regime":    '<circle cx="12" cy="12" r="8.5"/><path d="M12 12 8 8M12 6v.01M18 12h-.01M12 18v-.01M6 12h.01"/>',
    "traffic":   '<rect x="8" y="2.5" width="8" height="19" rx="3"/><circle cx="12" cy="7" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="17" r="1.6"/>',
    "moon":      '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5"/>',
    "sun":       '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.4 1.4M17.6 17.6 19 19M5 19l1.4-1.4M17.6 6.4 19 5"/>',
    "palette":   '<path d="M12 3a9 9 0 1 0 0 18c1 0 1.5-.8 1.5-1.6 0-1.1-1-1.7-1-2.6 0-.7.6-1.3 1.4-1.3H16a5 5 0 0 0 5-5c0-4.2-4-7.5-9-7.5z"/><circle cx="7.5" cy="11" r="1"/><circle cx="10" cy="7.5" r="1"/><circle cx="15" cy="8" r="1"/>',
    "arrow-up":  '<path d="M12 20V5M6 11l6-6 6 6"/>',
    "arrow-dn":  '<path d="M12 4v15M6 13l6 6 6-6"/>',
    "arrow-rt":  '<path d="M4 12h15M13 6l6 6-6 6"/>',
    "compass":   '<circle cx="12" cy="12" r="8.5"/><path d="m15.5 8.5-2 5-5 2 2-5z"/>',
    "scale":     '<path d="M12 3v18M7 21h10M12 5 5 8l-2 5a3 3 0 0 0 6 0L7 8m5-3 7 3 2 5a3 3 0 0 1-6 0l2-5"/>',
    "calendar":  '<rect x="3" y="4.5" width="18" height="16" rx="2"/><path d="M3 9h18M8 2.5v4M16 2.5v4"/>',
    "briefcase": '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 12h18"/>',
    "microscope":'<path d="M6 20h12M9 20l-1-4M14 4l3 3-6 6-3-3z"/><path d="M11 12a5 5 0 0 1 5 5"/>',
    "globe":     '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.5 2.4 2.5 14.6 0 17M12 3.5c-2.5 2.4-2.5 14.6 0 17"/>',
    "layers":    '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5M3 17l9 5 9-5"/>',
    "ruler":     '<rect x="3" y="8" width="18" height="8" rx="1.5" transform="rotate(0 12 12)"/><path d="M7 8v3M11 8v4M15 8v3M19 8v4"/>',
    "receipt":   '<path d="M5 3h14v18l-2.5-1.5L14 21l-2-1.5L10 21l-2.5-1.5L5 21z"/><path d="M8 8h8M8 12h8M8 16h5"/>',
    "brick":     '<rect x="3" y="4" width="18" height="16" rx="1.5"/><path d="M3 9.3h18M3 14.6h18M9 4v5.3M15 9.3v5.3M9 14.6V20"/>',
    "tv":        '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M8 21h8M12 6 8 2M12 6l4-4"/>',
    "satellite": '<path d="M4 13 11 6M6 15l3 3M9 12l3 3"/><path d="M13 4a5 5 0 0 1 5 5M14 8a2 2 0 0 1 2 2"/><circle cx="6.5" cy="17.5" r="2"/>',
    "book":      '<path d="M4 4h11a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3z"/><path d="M18 7a3 3 0 0 0-3-3H8"/>',
    "dot":       '<circle cx="12" cy="12" r="4" fill="currentColor" stroke="none"/>',
    "sparkle":   '<path d="M12 3v6M12 15v6M3 12h6M15 12h6"/>',
    "chat":      '<path d="M4 5h16v11H8l-4 4z"/><path d="M8 9h8M8 12h5"/>',
    "flame":     '<path d="M12 3c1 3 4 4 4 8a4 4 0 0 1-8 0c0-1.5.5-2.5 1-3 0 1.5 1 2 1.5 2 .3-2-.5-4 1.5-7z"/>',
}


def _svg(name: str, size: int = 16, cls: str = "ico", extra: str = "") -> str:
    """Server-side inline SVG icon. Inherits colour via stroke=currentColor."""
    p = _ICON_PATHS.get(name, _ICON_PATHS["dot"])
    return (f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true"{(" " + extra) if extra else ""}>{p}</svg>')


def _icon_js_object() -> str:
    """Emit the icon library as a JS object literal (for client-side card/modal templates).
    This string is inserted via a single-brace {py} substitution in the render_html f-string,
    so it is written with literal single braces (NOT doubled)."""
    import json as _json
    body = ",".join(f"{_json.dumps(k)}:{_json.dumps(v)}" for k, v in _ICON_PATHS.items())
    return "{" + body + "}"


def _synthetic_news(symbols: list[str]) -> list[dict]:
    templates = [
        "{s} sees unusual options activity into the close",
        "Analysts revise {s} price target after volume spike",
        "{s} momentum builds as moving averages cross",
        "{s} among most-active names this session",
    ]
    out = []
    for i, s in enumerate(symbols[:6]):
        out.append({
            "headline": templates[i % len(templates)].format(s=s),
            "source": "SyntheticWire", "created_at": "(demo)",
            "url": "", "symbols": [s],
        })
    return out


def _load_json_safe(path: str):
    """Read a small committed JSON artifact (study/benchmark files) or None. Never raises."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _setup_check_weights(study: dict | None) -> dict:
    """Map the walk-forward setup study (setups_backtest) into conviction-check weight multipliers,
    keyed by check label. A setup that beats baseline on real data earns a >1 multiplier; one that
    lags earns <1 (toward retirement). Gated on sample size and clamped so it nudges, never dominates.
    These compose multiplicatively with attribution's realized-outcome weights via the same `learned`
    map, so the two learning signals stack rather than fight."""
    out = {}
    if not study:
        return out
    for key, label in (("burst", "Momentum burst?"), ("ep", "Episodic pivot?"),
                       ("vcp", "VCP base setup?"), ("pshort", "Parabolic exhaustion?")):
        s = (study.get("setups") or {}).get(key) or {}
        edge, n = s.get("edge_pct"), s.get("n", 0)
        if edge is None or n < 40:            # not enough fires yet to trust the edge
            continue
        out[label] = round(max(0.5, min(1.5, 1.0 + edge * 0.2)), 3)   # +1% edge ≈ 1.2×, −1% ≈ 0.8×
    return out


def _market_regime(rows: list[dict]) -> dict | None:
    """Read the overall tape: breadth (% above trend), average momentum, # buys."""
    if not rows:
        return None
    above = sum(1 for r in rows if (r.get("context", {}).get("vs_slow_ma_pct") or 0) > 0)
    breadth = round(above / len(rows) * 100)
    rsis = [r["rsi"] for r in rows if r.get("rsi") is not None]
    avg_rsi = round(sum(rsis) / len(rsis), 1) if rsis else None
    buys = sum(1 for r in rows if r["action"] == "BUY")
    if breadth >= 60 and (avg_rsi or 0) >= 50:
        label, note = "Risk-on", "Most stocks are trending up — a friendlier backdrop for buying."
    elif breadth <= 40 or (avg_rsi or 100) < 45:
        label, note = "Risk-off", "Most stocks are below trend — be choosier; long signals are fighting the tape."
    else:
        label, note = "Neutral", "Mixed tape — no strong market-wide direction; pick spots carefully."
    return {"label": label, "breadth": breadth, "avg_rsi": avg_rsi,
            "buys": buys, "total": len(rows), "note": note}


def _concentration(shown: list[dict]) -> dict | None:
    """Flag when fresh entries pile into one sector — they're often the same macro bet in
    disguise, so '5 buys' can really be one position's worth of risk. Returns the dominant
    sector and its share when a single sector holds >=50% of fresh BUY (or SHORT) signals."""
    from collections import Counter
    out = None
    for action, word in (("BUY", "buys"), ("SHORT", "shorts")):
        fresh = [s for s in shown if s.get("action") == action]
        if len(fresh) < 3:
            continue
        by = Counter(s.get("sector") or "Other" for s in fresh)
        sector, n = by.most_common(1)[0]
        frac = n / len(fresh)
        if frac >= 0.5:
            cand = {"action": action, "word": word, "sector": sector, "n": n,
                    "total": len(fresh), "pct": round(frac * 100),
                    "symbols": [s["symbol"] for s in fresh if (s.get("sector") or "Other") == sector]}
            # keep the most concentrated of the two sides
            if out is None or cand["pct"] > out["pct"]:
                out = cand
    return out


def _sector_strength(rows: list[dict]) -> list[dict]:
    """Rank sectors by how many of their stocks are above their trend line."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[scanner.sector_of(r["symbol"])].append(r)
    out = []
    for sec, rs in by.items():
        up = sum(1 for r in rs if (r.get("context", {}).get("vs_slow_ma_pct") or 0) > 0)
        out.append({"sector": sec, "count": len(rs), "pct_up": round(up / len(rs) * 100)})
    out.sort(key=lambda x: -x["pct_up"])
    return out


def _compute_changes(shown: list, sectors: list, news_ideas: list, today: str, live: bool) -> list:
    """Worker: surface the MEANINGFUL diffs vs the previous build — new High-conviction calls,
    direction flips, conviction upgrades, sector flips, fresh catalysts. Persists prev_state.json.
    Live-only; never raises. Returns a short list of human-readable change strings."""
    if not live:
        return []
    _rank = {"Low": 0, "Medium": 1, "High": 2}
    path = "prev_state.json"
    try:
        with open(path) as f:
            prev = json.load(f)
    except Exception:  # noqa: BLE001
        prev = {}
    prev_sig, prev_sec = prev.get("signals", {}), prev.get("sectors", {})
    prev_cat = set(prev.get("catalysts", []))
    actionable = ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")
    cur_sig = {}
    for s in shown:
        c = s.get("conviction") or {}
        cur_sig[s.get("symbol")] = {"action": s.get("action"), "label": c.get("label"),
                                    "score": c.get("score_pct"), "dir": s.get("direction")}
    cur_sec = {x["sector"]: x.get("pct_up") for x in (sectors or [])}
    cur_cat = [(i.get("ticker"), i.get("direction")) for i in (news_ideas or []) if i.get("confidence") == "high"]

    def _persist():
        try:
            with open(path, "w") as f:
                json.dump({"signals": cur_sig, "sectors": cur_sec,
                           "catalysts": [t for t, _ in cur_cat if t], "generated_at": today}, f, indent=2)
        except Exception:  # noqa: BLE001
            pass

    if not prev_sig:                       # first run: seed state, don't flood with "new"
        _persist()
        return []
    changes = []
    for sym, c in cur_sig.items():
        p = prev_sig.get(sym)
        if c["label"] == "High" and c["action"] in ("BUY", "SHORT") and (not p or p.get("label") != "High"):
            changes.append(f"{_svg('sparkle',13)} {sym} &rarr; {c['action']} (High{', ' + str(c['score']) + '%' if c['score'] else ''})")
        elif p and p.get("dir") in ("LONG", "SHORT") and c["dir"] in ("LONG", "SHORT") \
                and p["dir"] != c["dir"] and c["action"] in actionable:
            changes.append(f"{_svg('refresh',13)} {sym} flipped to {c['action']}")
        elif p and p.get("label") and c["label"] and _rank.get(c["label"], 0) > _rank.get(p["label"], 0) \
                and c["action"] in actionable:
            changes.append(f"{_svg('arrow-up',13)} {sym} conviction now {c['label']} (was {p['label']})")
    for sec, pct in cur_sec.items():
        pp = prev_sec.get(sec)
        if pp is not None and pct is not None:
            if pp < 60 <= pct:
                changes.append(f"{_svg('trend-up',13)} {sec} sector turned strong ({pct}% above trend)")
            elif pp > 40 >= pct:
                changes.append(f"{_svg('trend-dn',13)} {sec} sector turned weak ({pct}% above trend)")
    for tk, d in cur_cat:
        if tk and tk not in prev_cat:
            changes.append(f"{_svg('news',13)} New catalyst: {tk} ({d})")
    _persist()
    return changes[:12]


def build_snapshot() -> dict:
    mode = _mode()
    live = mode != "SYNTHETIC"

    # Pin any symbol we already alerted today so the LATEST dashboard always contains every
    # name you were notified about (alerts and dashboard stay in line as the universe rotates).
    _today0 = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pin = set()
    if live:
        try:
            import notify as _nf0
            pin = _nf0.alerted_today(_today0)
        except Exception:  # noqa: BLE001
            pin = set()
        # News-idea candidates: pull names the LLM flagged from recent news INTO the scan so they
        # get a real technical read. They only surface as signals if the engine confirms them —
        # news widens the net, technicals still decide.
        if getattr(CONFIG, "news_idea_candidates", False):
            try:
                import json as _json
                from datetime import timedelta as _td
                with open("news_candidates.json") as _f:
                    _nc = _json.load(_f)
                _cut = (datetime.now(timezone.utc).date() - _td(days=2)).isoformat()
                pin |= {k for k, v in _nc.items() if v >= _cut}
            except Exception:  # noqa: BLE001
                pass

    rows = scanner.scan(CONFIG, live=live, pin=pin)
    for _r in rows:
        _r["alerted"] = _r["symbol"] in pin

    # split chart data out of each row for compactness
    charts = {r["symbol"]: r.pop("chart") for r in rows}
    # lookup of the full analysis row by symbol (used to make momentum rows clickable)
    rows_by_sym = {r["symbol"]: r for r in rows}

    # Dual-momentum leaderboard over the whole scanned universe (best-validated strategy).
    momentum_rows = _momentum_rank(charts)
    # Drop leaders whose scan price disagrees >15% with the consolidated Yahoo quote —
    # the same bad-feed-price guard we apply to signals (keeps MU-at-$981 junk off the list).
    if live and momentum_rows:
        try:
            import research as _r
            mq = _r.yahoo_quotes([m["symbol"] for m in momentum_rows])
            momentum_rows = [m for m in momentum_rows
                             if not (mq.get(m["symbol"]) and m.get("price")
                                     and abs(m["price"] / mq[m["symbol"]]["price"] - 1) > 0.15)]
        except Exception:  # noqa: BLE001
            pass
    # New-vs-holdover: mark which leaders just entered the list vs the previous run.
    try:
        import json as _json
        import datetime as _dt
        _mpath = "momentum_history.json"
        try:
            with open(_mpath) as _f:
                _prev = set((_json.load(_f) or {}).get("symbols", []))
        except Exception:  # noqa: BLE001
            _prev = set()
        for _m in momentum_rows:
            _m["is_new"] = bool(_prev) and _m["symbol"] not in _prev
        if live and momentum_rows:
            try:
                with open(_mpath, "w") as _f:
                    _json.dump({"as_of": _dt.date.today().isoformat(),
                                "symbols": [_m["symbol"] for _m in momentum_rows]}, _f, indent=2)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # Market regime + sector strength from the FULL scanned set (the "tape").
    regime = _market_regime(rows)
    sectors = _sector_strength(rows)

    shown = rows[: CONFIG.show_top]
    # force-include any alerted-today name that ranking truncated, so it's never missing
    if pin:
        _ss = {r["symbol"] for r in shown}
        for _r in rows:
            if _r.get("alerted") and _r["symbol"] not in _ss:
                shown.append(_r)
                _ss.add(_r["symbol"])
    shown_syms = [r["symbol"] for r in shown]

    # Learned directional gate: if the book has PROVEN it can't currently trade a whole side
    # (e.g. shorts getting run over — a wide, well-sampled realised win-rate deficit), stop
    # surfacing NEW entries in that direction as actionable until it earns its way back. This
    # acts on the track record itself, not a hardcoded bias; kill-switch via config.
    dir_gate = {}
    if getattr(CONFIG, "adaptive_direction_gate_enabled", True):
        try:
            import attribution as _attrd
            dir_gate = _attrd.suppressed_directions(
                scope="daily",
                min_n=getattr(CONFIG, "adaptive_direction_min_n", 25),
                winrate_max=getattr(CONFIG, "adaptive_direction_winrate", 35.0))
        except Exception:  # noqa: BLE001
            dir_gate = {}
    _act_dir = {"BUY": "LONG", "SHORT": "SHORT"}

    # Regime filter: don't initiate against a hostile tape. In Risk-off, demote fresh BUYs to
    # the WATCH tier; symmetrically, in Risk-on demote fresh SHORTs — the tool shouldn't fight
    # a strong market-wide direction with a brand-new entry in the opposite direction.
    if CONFIG.regime_block_buys and regime:
        _lbl = regime.get("label")
        for r in shown:
            if _lbl == "Risk-off" and r.get("action") == "BUY":
                r["action"] = "WATCH LONG"
                r["regime_blocked"] = True
                r.setdefault("reasons", []).insert(
                    0, f"{_svg('octagon',13)} Market regime is Risk-off — standing down on new buys; this setup is "
                       "shown as Watch, not a fresh entry.")
            elif _lbl == "Risk-on" and r.get("action") == "SHORT":
                r["action"] = "WATCH SHORT"
                r["regime_blocked"] = True
                r.setdefault("reasons", []).insert(
                    0, f"{_svg('octagon',13)} Market regime is Risk-on — standing down on new shorts; this setup is "
                       "shown as Watch, not a fresh entry against a rising market.")

    # A brand-new, independently-validated setup shouldn't be strangled by the OLD blanket
    # short-loss record before it ever gets to prove itself. An actionable parabolic-exhaustion
    # short is put on PROBATION: exempt from the blanket directional gate (only) so it can build its
    # own track record — provided its own walk-forward edge isn't negative. It still faces the
    # regime filter, no-trade gate, risk engine and the High-conviction bar, and only trades paper.
    _psh_edge = (((_load_json_safe("setups_study.json") or {}).get("setups") or {})
                 .get("pshort") or {}).get("edge_pct")
    _probation_ok = (getattr(CONFIG, "dir_gate_probation_new_setups", True)
                     and (_psh_edge is None or _psh_edge >= 0))
    # Apply the learned directional gate (after regime, so it catches whatever slipped through).
    if dir_gate:
        for r in shown:
            _d = _act_dir.get(r.get("action"))
            if _d and _d in dir_gate:
                if (_d == "SHORT" and _probation_ok
                        and ((r.get("factors") or {}).get("pshort") or {}).get("state") == "actionable"):
                    r["dir_gate_probation"] = True
                    r.setdefault("reasons", []).insert(
                        0, f"{_svg('ai',13)} On probation — a validated parabolic-exhaustion short is exempt "
                           f"from the blanket short gate so it can build its own track record (paper only).")
                    continue
                _st = dir_gate[_d]
                r["action"] = "WATCH LONG" if _d == "LONG" else "WATCH SHORT"
                r["direction_gated"] = True
                r.setdefault("reasons", []).insert(
                    0, f"{_svg('ai',13)} Learned gate — {_d.lower()}s have only won {_st['win_rate']}% of "
                       f"{_st['n']} settled trades, so new {_d.lower()}s are shown as Watch, not "
                       f"fresh entries, until that win rate recovers.")

    # Extension gate (diagnostic Jul 2026): momentum/leading longs win 67% (+1.28%) when NOT
    # extended vs 40% (+0.09%) when chasing. Demote a fresh BUY that's stretched above trend
    # (the "Not chasing?" conviction check failing) to Watch — the edge is in bases/pullbacks.
    if getattr(CONFIG, "extension_gate_enabled", True):
        for r in shown:
            if r.get("action") == "BUY":
                _cks = {c.get("label"): c.get("status") for c in ((r.get("conviction") or {}).get("checks") or [])}
                if _cks.get("Not chasing?") == "fail":
                    r["action"] = "WATCH LONG"
                    r["extension_gated"] = True
                    r.setdefault("reasons", []).insert(
                        0, f"{_svg('octagon',13)} Extended — price is stretched above its trend (chasing). Shown "
                           "as Watch, not a fresh entry: the edge is in non-extended entries (bases/pullbacks), not rips.")

    # Volatility gate (diagnostic Jul 2026): "Calm enough?" longs win 60% (+1.21%) vs 15% (−2.50%)
    # when too jumpy. Demote a fresh BUY on a too-volatile name to Watch — favour calm entries.
    if getattr(CONFIG, "volatility_gate_enabled", True):
        for r in shown:
            if r.get("action") == "BUY":
                _cks = {c.get("label"): c.get("status") for c in ((r.get("conviction") or {}).get("checks") or [])}
                if _cks.get("Calm enough?") == "fail":
                    r["action"] = "WATCH LONG"
                    r["volatility_gated"] = True
                    r.setdefault("reasons", []).insert(
                        0, f"{_svg('octagon',13)} Too jumpy — daily swings are large. Shown as Watch, not a fresh "
                           "entry: calm, low-volatility names have won far more (60% vs 15%).")

    # Dashboard control panel (dashboard_controls.json, written by the Control tab's "Apply to engine"
    # export). Lets you tune settings + accept/reject from the browser; the engine respects them next build.
    _ctrl = _load_json_safe("dashboard_controls.json") or {}
    for _ck, _cv in (_ctrl.get("settings") or {}).items():
        if _ck in ("meta_pwin_floor", "meta_buy_cap", "extension_gate_enabled",
                   "volatility_gate_enabled", "allow_shorts") and _cv is not None:
            try:
                setattr(CONFIG, _ck, _cv)
            except Exception:  # noqa: BLE001
                pass
    _ctrl_reject = {str(s).upper() for s in (_ctrl.get("rejected") or [])}
    _ctrl_accept = {str(s).upper() for s in (_ctrl.get("accepted") or [])}

    # Safety net: shorts are cut at the source in scanner._classify, but neutralise any short that
    # slips in via a forced-include / intraday path so it never trades or pollutes the record.
    if not getattr(CONFIG, "allow_shorts", False):
        for r in shown:
            if r.get("direction") == "SHORT" and r.get("action") in ("SHORT", "HOLD SHORT", "WATCH SHORT"):
                r["action"] = "AVOID"
                r["shorts_disabled"] = True

    # Journal → Engine overrides: your Obsidian judgment (via journal_sync.py → journal_overrides.json).
    # Your **avoid** list suppresses names here; your **watchlist** is seeded into the next scan below.
    _journal = _load_json_safe("journal_overrides.json") if getattr(CONFIG, "journal_overrides_enabled", True) else None
    if _journal:
        _avoid = {str(s).upper() for s in (_journal.get("avoid") or [])}
        if _avoid:
            for r in shown:
                if (r.get("symbol") or "").upper() in _avoid and r.get("action") in ("BUY", "HOLD LONG", "WATCH LONG"):
                    r["action"] = "AVOID"
                    r["journal_avoided"] = True
                    r.setdefault("reasons", []).insert(
                        0, f"{_svg('octagon',13)} On your journal <b>avoid</b> list — suppressed by your own read.")
    # Dashboard rejections — names you hit "Reject" on in the browser.
    if _ctrl_reject:
        for r in shown:
            if (r.get("symbol") or "").upper() in _ctrl_reject and r.get("action") in ("BUY", "HOLD LONG", "WATCH LONG"):
                r["action"] = "AVOID"
                r["user_rejected"] = True
                r.setdefault("reasons", []).insert(
                    0, f"{_svg('octagon',13)} You <b>rejected</b> this on the dashboard — suppressed.")
        _watch = [str(s).upper() for s in (_journal.get("watchlist") or []) if str(s).upper() not in _avoid]
        if _watch:                                    # pin your watchlist into the next scan for a full read
            try:
                import json as _jn
                from datetime import timedelta as _tdj
                _t0j = datetime.now(timezone.utc).date().isoformat()
                try:
                    with open("news_candidates.json") as _f:
                        _ncj = _jn.load(_f)
                except Exception:  # noqa: BLE001
                    _ncj = {}
                for _w in _watch:
                    _ncj[_w] = _t0j
                _oldj = (datetime.now(timezone.utc).date() - _tdj(days=5)).isoformat()
                _ncj = {k: v for k, v in _ncj.items() if v >= _oldj}
                with open("news_candidates.json", "w") as _f:
                    _jn.dump(_ncj, _f, indent=2)
            except Exception:  # noqa: BLE001
                pass

    # Pull news once for everything shown, from MULTIPLE feeds, then bucket per ticker.
    if live:
        import research as _rn
        news = []
        try:                       # Alpaca/Benzinga (symbol-keyed)
            news = market.get_news(shown_syms, CONFIG,
                                   limit=CONFIG.news_per_symbol * max(len(shown_syms), 1))
        except Exception:  # noqa: BLE001
            news = []
        try:                       # + free feeds: Google News (Reuters/Bloomberg/CNBC/…) + Yahoo Finance
            news += _rn.gather_symbol_news(shown_syms, CONFIG,
                                           per_symbol=6, max_symbols=CONFIG.research_top)
        except Exception:  # noqa: BLE001
            pass
        # global dedupe by headline, preserving order (Benzinga → Google → Yahoo)
        _seen, _merged = set(), []
        for _n in news:
            _h = (_n.get("headline") or "").strip().lower()
            if not _h or _h in _seen:
                continue
            _seen.add(_h)
            _merged.append(_n)
        news = _merged or [{"headline": "(no recent news found)", "source": "",
                            "created_at": "", "url": "", "symbols": []}]
        # Interleave by source so the Market news tab leads with a MIX, not 50 Benzinga first.
        from collections import OrderedDict as _OD, deque as _dq
        _g = _OD()
        for _n in news:
            _g.setdefault((_n.get("source") or "").lower(), _dq()).append(_n)
        _inter = []
        while any(_g.values()):
            for _q in _g.values():
                if _q:
                    _inter.append(_q.popleft())
        news = _inter
    else:
        news = _synthetic_news(shown_syms)

    # Company name + exchange for each shown ticker.
    _demo_names = {
        "AAPL": ("Apple Inc.", "NASDAQ"), "MSFT": ("Microsoft Corp.", "NASDAQ"),
        "NVDA": ("NVIDIA Corp.", "NASDAQ"), "AMZN": ("Amazon.com Inc.", "NASDAQ"),
        "TSLA": ("Tesla Inc.", "NASDAQ"), "META": ("Meta Platforms Inc.", "NASDAQ"),
        "GOOGL": ("Alphabet Inc.", "NASDAQ"), "AMD": ("Adv. Micro Devices", "NASDAQ"),
        "SPY": ("SPDR S&P 500 ETF", "NYSE Arca"), "QQQ": ("Invesco QQQ Trust", "NASDAQ"),
    }
    for r in shown:
        if live:
            a = market.get_asset(r["symbol"], CONFIG)
            r["name"], r["exchange"] = a.get("name", ""), a.get("exchange", "")
        else:
            nm, ex = _demo_names.get(r["symbol"], (r["symbol"] + " (demo)", "DEMO"))
            r["name"], r["exchange"] = nm, ex
        r["sector"] = scanner.sector_of(r["symbol"])

    # Attach each ticker's own headlines to its row (for the click-through detail),
    # and fold a plain-English news line into the reasoning so it's news-aware.
    def _interleave_by_source(items):
        """Round-robin across sources so the top headlines show a MIX (not 10 Benzinga first)."""
        from collections import OrderedDict, deque
        groups = OrderedDict()
        for it in items:
            groups.setdefault(it.get("source", ""), deque()).append(it)
        out = []
        while any(groups.values()):
            for q in groups.values():
                if q:
                    out.append(q.popleft())
        return out
    for r in shown:
        # keep more headlines per ticker now that several feeds contribute — richer tone signal,
        # interleaved by source so the modal shows variety, not one outlet stacked on top.
        _matched = [n for n in news if r["symbol"] in (n.get("symbols") or [])]
        r["news"] = _interleave_by_source(_matched)[: max(CONFIG.news_per_symbol, 12)]
        # Catalyst: flag when fresh (<~2 day) news coincides with the signal.
        try:
            import research as _rr
            _fresh = [n for n in r["news"] if _rr._recency_weight(n.get("created_at", "")) >= 0.9]
            if _fresh:
                r["catalyst"] = {"headline": _fresh[0]["headline"],
                                 "source": _fresh[0].get("source", ""), "n": len(_fresh)}
        except Exception:  # noqa: BLE001
            pass
        if r["news"]:
            top = r["news"][0]["headline"]
            n = len(r["news"])
            phrase = "1 recent story mentions" if n == 1 else f"{n} recent stories mention"
            r.setdefault("reasons", []).append(
                f"{_svg('news',13)} In the news: {phrase} {r['symbol']}. "
                f"Latest headline — “{top}”. Worth a read for what's driving it."
            )
        else:
            r.setdefault("reasons", []).append(
                f"{_svg('news',13)} No recent news found for {r['symbol']} — the move looks technical (chart-driven), "
                f"not headline-driven."
            )

    # --- Research layer: news tone (free), analyst/fundamentals (Finnhub) ---
    import research
    # Consolidated (full-market) price + previous close so the cards match Google/Yahoo,
    # since the scan runs on Alpaca's IEX feed whose close can drift a few cents.
    price_drops = []
    if live:
        try:
            yq = research.yahoo_quotes([r["symbol"] for r in shown])
            for r in shown:
                q = yq.get(r["symbol"])
                if q:
                    r["quote_price"] = q["price"]
                    r["prev_close"] = q.get("prev_close")
            # Cross-check: drop any signal whose scan (Alpaca IEX) price disagrees
            # with the consolidated quote by >15% — that means the feed gave a bad
            # price, so the whole signal (indicators, plan, levels) is untrustworthy.
            bad = [r for r in shown
                   if r.get("quote_price") and r.get("price")
                   and abs(r["price"] / r["quote_price"] - 1) > 0.15]
            if bad:
                price_drops = [f"{r['symbol']} (scan ${r['price']:,.2f} vs ${r['quote_price']:,.2f})"
                               for r in bad]
                drop = {r["symbol"] for r in bad}
                shown = [r for r in shown if r["symbol"] not in drop]
                shown_syms = [r["symbol"] for r in shown]
            # rebase kept signals' price + plan levels onto the consolidated quote so the
            # displayed price and the entry/stop/target are internally consistent.
            for r in shown:
                q = r.get("quote_price")
                if not q or not r.get("price"):
                    continue
                ratio = q / r["price"]
                if abs(ratio - 1) > 0.003:
                    for k in ("stop", "target"):
                        if r.get(k) is not None:
                            r[k] = round(r[k] * ratio, 2)
                    p = r.get("plan") or {}
                    for k in ("entry", "stop", "target"):
                        if p.get(k) is not None:
                            p[k] = round(p[k] * ratio, 2)
                r["price"] = q
        except Exception:  # noqa: BLE001
            pass
    for r in shown:
        r["sentiment"] = research.news_sentiment(r.get("news"))
    # TradingView multi-timeframe TA rating — independent cross-check (keyless, unofficial).
    tv_map = {}
    if live:
        try:
            import tradingview as _tv
            tv_map = _tv.ratings([r["symbol"] for r in shown[: CONFIG.research_top]],
                                 proxy=CONFIG.live_quotes_url or None)
        except Exception:  # noqa: BLE001
            tv_map = {}
    for r in shown:
        if tv_map.get(r["symbol"]):
            r["tv"] = tv_map[r["symbol"]]
    fundamentals = {}
    if live and CONFIG.finnhub_api_key:
        try:
            fundamentals = research.finnhub_for_symbols(
                [r["symbol"] for r in shown[: CONFIG.research_top]], CONFIG)
        except Exception:  # noqa: BLE001
            fundamentals = {}
    # Scraped alt-data for the actionable names: SEC EDGAR insider filings (keyless) +
    # StockTwits retail buzz (routed via the Worker, which has a non-datacenter egress).
    insiders, buzz = {}, {}
    if live:
        _act = [r["symbol"] for r in shown
                if r["action"] in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")]
        try:
            import scrape as _scrape
            insiders = _scrape.insider_activity(_act)
        except Exception:  # noqa: BLE001
            insiders = {}
        try:
            import scrape as _scrape
            buzz = _scrape.stocktwits_buzz(_act, proxy=CONFIG.live_quotes_url or None)
        except Exception:  # noqa: BLE001
            buzz = {}

    # News-driven ideas: one LLM pass over recent headlines -> actionable single-stock reads.
    # Feeds a conviction nudge on scanned names + a standalone list (incl. names not in the scan).
    news_ideas = []
    if live:
        try:
            import llm
            news_ideas = llm.news_ideas(news, CONFIG, universe={r["symbol"] for r in shown})
        except Exception:  # noqa: BLE001
            news_ideas = []
    _idea_map = {i["ticker"]: i for i in news_ideas}
    # Persist material (high/medium-confidence) news-idea tickers so the NEXT build pulls them into
    # the scan for a technical read. Pruned after ~5 days. Live-only; never breaks the build.
    if live and getattr(CONFIG, "news_idea_candidates", False):
        try:
            import json as _json
            from datetime import timedelta as _td
            try:
                with open("news_candidates.json") as _f:
                    _ncw = _json.load(_f)
            except Exception:  # noqa: BLE001
                _ncw = {}
            for _i in news_ideas:
                _tk = (_i.get("ticker") or "").upper().strip().lstrip("$")
                if _tk and _i.get("confidence") in ("high", "medium"):
                    _ncw[_tk] = _today0
            # Grok pre-market sweep: names with a FRESH real-time X/news catalyst get pinned into the
            # next scan for a full technical + episodic-pivot read (opt-in; fail-silent; the engine's
            # technicals still decide whether any of them become a signal).
            if getattr(CONFIG, "xai_premarket_scan_enabled", False):
                try:
                    import xai as _xai_pm
                    for _pm in _xai_pm.premarket_catalysts(CONFIG):
                        _pt = (_pm.get("symbol") or "").upper().strip().lstrip("$")
                        if _pt:
                            _ncw[_pt] = _today0
                except Exception:  # noqa: BLE001
                    pass
            _old = (datetime.now(timezone.utc).date() - _td(days=5)).isoformat()
            _ncw = {k: v for k, v in _ncw.items() if v >= _old}
            with open("news_candidates.json", "w") as _f:
                _json.dump(_ncw, _f, indent=2)
        except Exception:  # noqa: BLE001
            pass

    # --- Intraday layer (gated + graceful): run the SAME engine on intraday bars over the same
    # names. Powers the Intraday tab AND a lower-timeframe confirmation that nudges daily
    # conviction. Any failure leaves it empty — the daily build is never affected.
    intraday_shown: list = []
    intraday_by_sym: dict = {}
    # Adaptive learning: per-check weight multipliers from the bot's own resolved wins vs losses.
    # Daily/swing and intraday are DIFFERENT strategies, so each learns from its OWN book — separate
    # weight sets, applied per timeframe. Computed BEFORE the intraday scan so intraday signals are
    # scored with their own learning too. Empty until there's enough data, so early on conviction
    # runs on its transparent default weights.
    daily_learned, intraday_learned = {}, {}
    if getattr(CONFIG, "adaptive_weights_enabled", True):
        try:
            import attribution as _attr
            _mn = CONFIG.adaptive_min_n
            _re = getattr(CONFIG, "adaptive_retire_edge", -15.0)
            daily_learned = _attr.learned_weights(scope="daily", min_n=_mn, retire_edge=_re)
            intraday_learned = _attr.learned_weights(scope="intraday", min_n=_mn, retire_edge=_re)
        except Exception:  # noqa: BLE001
            daily_learned, intraday_learned = {}, {}
    # Fold the walk-forward setup-edge study into the daily learned weights (multiplicatively, so it
    # composes with attribution rather than overriding it): a setup that's proven edge over baseline
    # counts more in conviction, one that lags counts less. No-ops until setups_study.json exists.
    try:
        for _lbl, _mult in _setup_check_weights(_load_json_safe("setups_study.json")).items():
            daily_learned[_lbl] = round(daily_learned.get(_lbl, 1.0) * _mult, 3)
    except Exception:  # noqa: BLE001
        pass

    intraday_track: dict = {}
    if live and getattr(CONFIG, "intraday_enabled", False):
        try:
            from dataclasses import replace as _replace
            _icfg = _replace(CONFIG, timeframe=CONFIG.intraday_timeframe,
                             lookback_days=CONFIG.intraday_lookback_days,
                             fast_ma=CONFIG.intraday_fast_ma, slow_ma=CONFIG.intraday_slow_ma,
                             target_atr_reach=getattr(CONFIG, "intraday_target_atr_reach", 14.0),
                             target_swing_lookback=getattr(CONFIG, "intraday_target_swing_lookback", 78),
                             atr_stop_mult=getattr(CONFIG, "intraday_atr_stop_mult", 2.5))
            _iuni, _iseen = [], set()
            for _s in [r["symbol"] for r in rows] + scanner.recent_listings(CONFIG):
                if _s and _s not in _iseen:
                    _iseen.add(_s)
                    _iuni.append(_s)
            _irows = scanner.scan(_icfg, live=live, universe=_iuni)
            for _ir in _irows:
                _ir.pop("chart", None)
                _ir["intraday"] = True
                # score intraday conviction with the INTRADAY-only learned weights (its own strategy)
                if intraday_learned:
                    try:
                        scanner.rescore(_ir, _icfg, regime=regime, learned=intraday_learned)
                    except Exception:  # noqa: BLE001
                        pass
            intraday_by_sym = {_ir["symbol"]: _ir for _ir in _irows}
            _irows.sort(key=lambda r: -((r.get("conviction") or {}).get("score_pct") or 0))
            intraday_shown = _irows[: CONFIG.intraday_show_top]
            # Shadow track-record for the intraday layer (NO orders): grade intraday calls against
            # intraday bars, in a SEPARATE log so it never mixes with the daily record/paper book.
            try:
                import tracker as _tracker
                import pandas as _pd
                intraday_track = _tracker.run(
                    list(intraday_by_sym.values()), _icfg, live, _today0,
                    path="track_record_intraday.json", intraday=True,
                    now_ts=_pd.Timestamp.utcnow().tz_localize(None), hold_days=3,
                    regime=regime)
            except Exception as _itexc:  # noqa: BLE001
                intraday_track = {}
                print("INTRADAY TRACK: skipped —", _itexc)
            print(f"INTRADAY: {len(intraday_shown)} signals on {CONFIG.intraday_timeframe} bars")
        except Exception as _iexc:  # noqa: BLE001 - never break the daily build
            intraday_shown, intraday_by_sym, intraday_track = [], {}, {}
            print("INTRADAY: skipped —", _iexc)

    # Short interest (squeeze risk) — Yahoo key-stats for the actionable names. Gated + fail-silent.
    short_int = {}
    if live:
        try:
            _si_syms = [r["symbol"] for r in shown
                        if r.get("action") in ("BUY", "SHORT", "WATCH SHORT", "HOLD SHORT", "WATCH LONG", "HOLD LONG")]
            short_int = research.short_interest(_si_syms)
        except Exception:  # noqa: BLE001
            short_int = {}

    # Retail / social attention (ApeWisdom: Reddit + WSB mentions). One keyless call returns the
    # top-attention names globally; we match by symbol. The noisiest input — only a light nudge.
    retail_map = {}
    if live:
        try:
            import scrape as _scrape
            retail_map = _scrape.retail_attention()
        except Exception:  # noqa: BLE001
            retail_map = {}

    _ACTIONABLE = ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")
    _sector_pct = {s["sector"]: s.get("pct_up") for s in (sectors or [])}   # sector momentum map
    for r in shown:
        r["fundamentals"] = fundamentals.get(r["symbol"])
        r["insider"] = insiders.get(r["symbol"])
        r["buzz"] = buzz.get(r["symbol"])
        r["news_idea"] = _idea_map.get(r["symbol"])
        r["short_interest"] = short_int.get(r["symbol"])
        r["retail"] = retail_map.get(r["symbol"])
        # Lower-timeframe confirmation: does the intraday signal agree with this daily trade?
        _isig = intraday_by_sym.get(r["symbol"])
        if _isig and _isig.get("action") in _ACTIONABLE:
            r["intraday_confirm"] = "agree" if _isig.get("direction") == r.get("direction") else "disagree"
        else:
            r["intraday_confirm"] = "none"
        # Episodic Pivot catalyst enrichment: the scanner computes EP technical-only (no news yet);
        # now that headlines are attached, re-classify it with the real catalyst so EARNINGS/FDA/
        # M&A families and the catalyst score flow into the (about-to-run) conviction re-score.
        _ep = (r.get("factors") or {}).get("ep")
        if isinstance(_ep, dict) and _ep.get("base_score") is not None:
            _hl = ((r.get("catalyst") or {}).get("headline")
                   or (r["news"][0]["headline"] if r.get("news") else None))
            _has_news = bool(r.get("catalyst")) or bool(r.get("news"))
            try:
                import screens as _screens
                r["factors"]["ep"] = _screens.reclassify_ep(_ep, _has_news, _hl)
            except Exception:  # noqa: BLE001
                pass
        # Re-score conviction + desk read now that research (news/sector/intraday) is in hand.
        scanner.rescore(r, CONFIG, sentiment=r.get("sentiment"), fundamentals=r.get("fundamentals"),
                        tv=r.get("tv"), regime=regime, insider=r.get("insider"), buzz=r.get("buzz"),
                        news_idea=r.get("news_idea"), intraday=_isig,
                        sector_pct=_sector_pct.get(r.get("sector")), short_interest=r.get("short_interest"),
                        retail=r.get("retail"), learned=daily_learned)

    # LLM structured news scoring: convert recent per-stock headlines into named structured scores
    # (guidance / margin pressure / demand / regulatory risk / …) for the top actionable names only
    # — one cheap batched call. The LLM converts text→numbers; it never decides the trade.
    nlp_scores = {}
    if live and CONFIG.llm_enabled:
        try:
            import llm as _llm_nlp
            nlp_scores = _llm_nlp.structured_scores(shown, CONFIG)
            for r in shown:
                if r["symbol"] in nlp_scores:
                    r["nlp"] = nlp_scores[r["symbol"]]
        except Exception:  # noqa: BLE001
            nlp_scores = {}

    # First-seen dates per signal — powers the "Newest" sort + a date chip on each card.
    # Persisted across runs (like the tracker); a symbol that leaves and returns gets a fresh date.
    try:
        import json as _json2, datetime as _dt2
        _spath = "signals_seen.json"
        try:
            with open(_spath) as _f:
                _seen = (_json2.load(_f) or {}).get("seen", {})
        except Exception:  # noqa: BLE001
            _seen = {}
        _today = _dt2.date.today().isoformat()
        _cur = {}
        for r in shown:
            k = f"{r['symbol']}:{r.get('direction', 'LONG')}"
            first = _seen.get(k) or _today
            _cur[k] = first
            r["first_seen"] = first
            try:
                r["days_old"] = (_dt2.date.fromisoformat(_today) - _dt2.date.fromisoformat(first)).days
            except Exception:  # noqa: BLE001
                r["days_old"] = 0
            r["is_fresh"] = (first == _today)
        if live:  # only the live Action persists (synthetic/dev runs never write)
            try:
                with open(_spath, "w") as _f:
                    _json2.dump({"as_of": _today, "seen": _cur}, _f, indent=2)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # Ray Dalio All Weather allocation + backtest vs SPY (keyless Yahoo history).
    try:
        import allweather as _aw
        all_weather = _aw.build(live)
    except Exception:  # noqa: BLE001
        all_weather = None

    # Survivorship-bias-FREE momentum backtest (fixed ETF universe) — the honest performance
    # read for the momentum strategy, vs the survivorship-biased single-stock ranking.
    try:
        import momentum_lab as _ml
        momentum_bt = _ml.build(live)
    except Exception:  # noqa: BLE001
        momentum_bt = None

    # Walk-forward / out-of-sample validation (gated) — the honest "does the edge survive on
    # data it wasn't tuned on?" read. Small basket, read-only; any failure -> None.
    try:
        import walkforward as _wf
        walk_fwd = _wf.validate(CONFIG, live, n_folds=CONFIG.walkforward_folds)
    except Exception:  # noqa: BLE001
        walk_fwd = None

    # Macro backdrop (FRED) — once per run.
    macro = None
    if live and CONFIG.fred_api_key:
        try:
            macro = research.fred_macro(CONFIG)
        except Exception:  # noqa: BLE001
            macro = None

    # Macro regime → exposure: blend the macro backdrop + equity breadth into a posture and an
    # exposure multiplier that scales new-position sizing. Macro controls EXPOSURE, never a direct
    # buy/sell. Gated + fail-silent; None when disabled or no data.
    try:
        import macro_regime as _macro_regime
        macro_posture = _macro_regime.assess(macro, regime, CONFIG)
    except Exception:  # noqa: BLE001
        macro_posture = None
    _exposure_mult = (macro_posture or {}).get("exposure_mult", 1.0)

    # Market timing (O'Neil): Follow-Through-Day confirmation + distribution-day count on SPY/QQQ.
    # Where macro reads the *backdrop*, this reads the *tape's own* institutional-timing signal —
    # a confirmed FTD is a green light to add, a distribution cluster is a warning to trim. It
    # tilts the exposure multiplier (never buys directly). Live-only (fetches index bars); fail-silent.
    timing_posture = None
    if live:
        try:
            import timing as _timing
            timing_posture = _timing.assess(CONFIG)
        except Exception:  # noqa: BLE001
            timing_posture = None
    if timing_posture:
        # Blend: take the more conservative of macro vs timing so either can throttle, but let a
        # confirmed FTD lift a merely-neutral macro. Clamp to a sane band.
        _tm = timing_posture.get("exposure_mult", 1.0)
        _exposure_mult = round(max(0.4, min(1.25, min(_exposure_mult, _tm) if _tm < 1.0 else (_exposure_mult + _tm) / 2.0)), 3)
        timing_posture["exposure_mult_blended"] = _exposure_mult
        # Attach the latest self-validation study (written by timing_backtest in the post-close CI
        # job) so the panel can show whether the timing signal has actually predicted forward returns.
        try:
            with open("timing_study.json") as _tf:
                timing_posture["study"] = json.load(_tf)
        except Exception:  # noqa: BLE001
            pass

    # Timing gate (O'Neil): in a confirmed correction the tape is in institutional distribution, so
    # demote fresh BUYs to the WATCH tier — same teeth as the Risk-off regime block, but driven by
    # the FTD/distribution engine. Runs before ranking so blocked names can't top the board.
    if timing_posture and getattr(CONFIG, "timing_gate_enabled", True) and timing_posture.get("state") == "correction":
        for r in shown:
            if r.get("action") == "BUY":
                r["action"] = "WATCH LONG"
                r["timing_blocked"] = True
                r.setdefault("reasons", []).insert(
                    0, f"{_svg('octagon',13)} Indexes are in a correction ({timing_posture.get('dd_total', 0)} "
                       "distribution days) — standing down on new buys until a Follow-Through Day confirms a new "
                       "uptrend; shown as Watch, not a fresh entry.")

    # Multi-agent trade committee: 4 LLM analyst roles (technicals / fundamentals / news / macro)
    # debate the top actionable signals and a chair returns accept/reduce/reject + per-role leans.
    # Advisory second opinion — the rules risk engine keeps final authority. Gated + fail-silent.
    committee_verdicts = {}
    if live and CONFIG.llm_enabled and getattr(CONFIG, "committee_enabled", True):
        try:
            import llm as _llm_cm
            committee_verdicts = _llm_cm.committee(shown, CONFIG, regime=regime, macro=macro,
                                                   max_names=getattr(CONFIG, "committee_max_names", 6))
            for r in shown:
                if r["symbol"] in committee_verdicts:
                    r["committee"] = committee_verdicts[r["symbol"]]
                    # Re-score so the AI committee's vote actually COUNTS toward conviction (it's
                    # computed after the main scoring pass). The check is then graded by attribution
                    # like any other, so the AI earns its influence from real outcomes.
                    if getattr(CONFIG, "committee_conviction_enabled", True):
                        try:
                            scanner.rescore(r, CONFIG, sentiment=r.get("sentiment"),
                                            fundamentals=r.get("fundamentals"), tv=r.get("tv"),
                                            regime=regime, intraday=r.get("intraday_sig"),
                                            learned=daily_learned, committee=r["committee"])
                        except Exception:  # noqa: BLE001
                            pass
        except Exception:  # noqa: BLE001
            committee_verdicts = {}

    # Live X (Twitter)/web sentiment from Grok on the top actionable names — the one real-time social
    # read the other models can't give. Opt-in (XAI key + flag), budget-capped, fail-silent. Enters
    # conviction as a self-grading check so it earns its influence from outcomes like everything else.
    if live and getattr(CONFIG, "xai_live_sentiment_enabled", False):
        try:
            import xai as _xai
            _xcap = int(getattr(CONFIG, "xai_max_names", 6))
            # Top names by rank — include the WATCH tier so the pulse still populates after the
            # extension/volatility/short gates demote fresh entries (they were starving this list).
            _xtiers = ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT", "WATCH LONG", "WATCH SHORT")
            _xc = sorted((r for r in shown if r.get("action") in _xtiers),
                         key=lambda r: -((r.get("rank_score") or (r.get("conviction") or {}).get("score_pct") or 0)))[:_xcap]
            for r in _xc:
                _sent = _xai.live_sentiment(r["symbol"], r.get("name", ""), r.get("direction", "LONG"), CONFIG)
                if _sent:
                    r["xai_sentiment"] = _sent
                    try:
                        scanner.rescore(r, CONFIG, sentiment=r.get("sentiment"),
                                        fundamentals=r.get("fundamentals"), tv=r.get("tv"), regime=regime,
                                        intraday=r.get("intraday_sig"), learned=daily_learned,
                                        committee=r.get("committee"), xai_sentiment=_sent)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass

    # Grok pulse status — so the dashboard can say WHY it's empty instead of guessing.
    _xai_n = sum(1 for r in shown if r.get("xai_sentiment"))
    if not getattr(CONFIG, "xai_live_sentiment_enabled", False):
        _xai_status = "off"
    elif not getattr(CONFIG, "xai_api_key", ""):
        _xai_status = "no_key"
    elif not live:
        _xai_status = "not_live"
    else:
        _xai_status = "ok" if _xai_n else "empty"

    # Grok BUZZ scan — "where's the buzz": the most talked-about US stocks on X right now. Powers the
    # pulse as a live discovery feed, AND pins high-buzz bullish/rising names into the next scan so
    # social sentiment can TRIGGER trades — gated by the same technicals + meta-label + risk as
    # everything else. (Blind sentiment-trading is a known trap; sentiment surfaces candidates, the
    # edge decides whether any become an actual signal.)
    xai_buzz = []
    if live and getattr(CONFIG, "xai_live_sentiment_enabled", False) and getattr(CONFIG, "xai_buzz_enabled", True):
        try:
            import xai as _xaib
            xai_buzz = _xaib.buzz_scan(CONFIG, int(getattr(CONFIG, "xai_buzz_max", 10)))
            _bysym = {s.get("symbol"): s for s in shown}
            for _b in xai_buzz:
                _sig = _bysym.get(_b["symbol"])
                if _sig:                                  # this buzzy name is also one we flagged
                    _b["is_signal"] = True
                    _b["signal_action"] = _sig.get("action")
                    _b["p_win"] = _sig.get("p_win")
            # Trade trigger: pin bullish + rising/high-volume buzz names into the candidate pool so the
            # NEXT scan gives them a full technical + conviction + meta-label read (they only become
            # trades if the edge agrees — social buzz opens the door, it doesn't pull the trigger alone).
            try:
                import json as _json2
                from datetime import timedelta as _td2
                _t0 = datetime.now(timezone.utc).date().isoformat()
                try:
                    with open("news_candidates.json") as _f:
                        _nc2 = _json2.load(_f)
                except Exception:  # noqa: BLE001
                    _nc2 = {}
                for _b in xai_buzz:
                    # Never let a suspected coordinated pump seed a trade (avoid getting pumped into).
                    if _b.get("hype_risk") == "high":
                        continue
                    if _b.get("stance") == "bullish" and (_b.get("momentum") == "rising"
                                                          or _b.get("social_volume") == "high"):
                        _nc2[_b["symbol"]] = _t0
                _oldc = (datetime.now(timezone.utc).date() - _td2(days=5)).isoformat()
                _nc2 = {k: v for k, v in _nc2.items() if v >= _oldc}
                with open("news_candidates.json", "w") as _f:
                    _json2.dump(_nc2, _f, indent=2)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    # StockTwits trending discovery (ported from fintwit-bot) — the most-discussed retail names right
    # now, from a free no-key endpoint. An independent crowd source: the top names seed the next scan
    # for a full technical + meta-label read (crowd attention opens the door, the edge still decides).
    retail_trending = []
    if live and getattr(CONFIG, "stocktwits_trending_enabled", True):
        try:
            import scrape as _scrape2
            retail_trending = _scrape2.stocktwits_trending(
                proxy=CONFIG.live_quotes_url or None,
                limit=int(getattr(CONFIG, "stocktwits_trending_max", 12)))
            if retail_trending:
                import json as _json3
                from datetime import timedelta as _td3
                _t0b = datetime.now(timezone.utc).date().isoformat()
                try:
                    with open("news_candidates.json") as _f:
                        _nc3 = _json3.load(_f)
                except Exception:  # noqa: BLE001
                    _nc3 = {}
                for _rt in retail_trending[:8]:
                    _nc3[_rt["symbol"]] = _t0b
                _oldc3 = (datetime.now(timezone.utc).date() - _td3(days=5)).isoformat()
                _nc3 = {k: v for k, v in _nc3.items() if v >= _oldc3}
                with open("news_candidates.json", "w") as _f:
                    _json3.dump(_nc3, _f, indent=2)
        except Exception:  # noqa: BLE001
            pass

    # Premium-selling advisory (Theta Harvest-style) — scores where SELLING options premium is
    # favourable, 0-100, with hard gates. ADVISORY ONLY; needs live Alpaca options data for implied
    # vol and fails soft (empty) without it.
    premium_selling = {}
    if live and getattr(CONFIG, "premium_selling_enabled", True):
        try:
            import premium_selling as _psell
            premium_selling = _psell.run(CONFIG, breadth=(regime or {}).get("breadth"),
                                         max_names=int(getattr(CONFIG, "premium_selling_max", 12)))
        except Exception:  # noqa: BLE001
            premium_selling = {}

    # Meta-label P(win) — the nightly walk-forward model ranks winners far better (OOS AUC 0.77) than
    # the hand-tuned conviction (0.23). Attach P(win) to every long, demote fresh BUYs below the floor
    # to Watch (the dropped cohort wins only ~27% OOS), and expose P(win) for size tilt + display.
    # Fail-safe: no-ops without a model file. Long-only (the model is trained on resolved longs).
    if getattr(CONFIG, "meta_pwin_enabled", True):
        try:
            import meta_score as _ms
            if _ms.load_model():
                _floor = float(getattr(CONFIG, "meta_pwin_floor", 0.45))
                for r in shown:
                    if r.get("direction") != "LONG":
                        continue
                    _pw = _ms.p_win(r)
                    if _pw is None:
                        continue
                    r["p_win"] = round(_pw, 3)
                    if r.get("action") == "BUY" and _pw < _floor and (r.get("symbol") or "").upper() not in _ctrl_accept:
                        r["action"] = "WATCH LONG"
                        r["meta_gated"] = True
                        r.setdefault("reasons", []).insert(
                            0, f"{_svg('octagon',13)} Low model win-probability ({_pw*100:.0f}%) — the learned "
                               "meta-label ranks this below the quality floor. Shown as Watch, not a fresh entry: "
                               "trades below this line have won only ~27% out-of-sample.")
                # Keep only the top-N fresh BUYs by P(win) — a defined, high-quality actionable list; rest → Watch.
                _cap = int(getattr(CONFIG, "meta_buy_cap", 6))
                # accepted names are never capped away — your explicit picks always stay actionable
                _fresh_buys = sorted([r for r in shown if r.get("action") == "BUY"
                                      and (r.get("symbol") or "").upper() not in _ctrl_accept],
                                     key=lambda r: -(r.get("p_win") or 0))
                for r in _fresh_buys[_cap:]:
                    r["action"] = "WATCH LONG"
                    r["meta_capped"] = True
                    r.setdefault("reasons", []).insert(
                        0, f"{_svg('octagon',13)} Beyond today's top {_cap} by model win-probability — shown as "
                           "Watch to keep the actionable list tight and the win rate high.")
        except Exception:  # noqa: BLE001
            pass

    # Regime-specific weighting: in a defensive / high-volatility regime, RAISE the conviction bar
    # a fresh entry must clear — so the bot makes fewer, higher-quality trades when the backdrop is
    # hostile. With-tape setups below the regime threshold are demoted to the Watch tier.
    _regime_threshold = 0
    if getattr(CONFIG, "regime_weighting_enabled", True) and macro_posture:
        _lab = macro_posture.get("label")
        _tagset = {t.get("tag") for t in macro_posture.get("tags", [])}
        _regime_threshold = {"Risk-on": 50, "Neutral": 55, "Risk-off": 62}.get(_lab, 50)
        if "High-volatility" in _tagset:
            _regime_threshold += 6
        macro_posture["entry_threshold"] = _regime_threshold
        for r in shown:
            if r.get("action") not in ("BUY", "SHORT"):
                continue
            sc = (r.get("conviction") or {}).get("score_pct") or 0
            if sc < _regime_threshold:
                r["action"] = "WATCH LONG" if r.get("direction") == "LONG" else "WATCH SHORT"
                r["regime_demoted"] = True
                ctx = r.setdefault("context", {})
                r.setdefault("reasons", []).insert(
                    0, f"{_svg('scale',13)} {_lab} regime raises the bar to {_regime_threshold}% — this {sc}% setup is "
                       "demoted to Watch (fewer, higher-quality entries when the backdrop is tough).")

    # Adaptive asset ranking: score actionable names for CAPITAL ALLOCATION (quality + vol-adj
    # reward + macro fit + liquidity + momentum). Mutates rows (rank_score/rank) + returns a list.
    try:
        import rank as _rank
        ranked = _rank.rank_rows(shown, macro_posture, CONFIG)
    except Exception:  # noqa: BLE001
        ranked = []

    # IPO watch: upcoming-IPO calendar + general news mentioning pre-IPO names
    # (e.g. SpaceX). Private names have no ticker, so this is the only way they surface.
    ipos, ipo_news = [], []
    if live:
        try:
            ipo_news = research.ipo_buzz_news(CONFIG)   # keyless (Google News RSS)
        except Exception:  # noqa: BLE001
            ipo_news = []
        if CONFIG.finnhub_api_key:
            try:
                ipos = research.ipo_calendar(CONFIG)
            except Exception:  # noqa: BLE001
                ipos = []

    # Optional AI analyst note — for every High-conviction actionable setup (BUY/SHORT/HOLD).
    # No hard top-N cap (the High-conviction floor already limits the count); a generous safety
    # ceiling guards against a pathological run firing dozens of calls.
    llm_status = {"enabled": bool(CONFIG.llm_enabled)}
    if CONFIG.llm_enabled:
        import llm
        _ai_picks = [r for r in shown
                     if (r.get("conviction") or {}).get("label") == "High"
                     and r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")]
        _ai_picks.sort(key=lambda r: -((r.get("conviction") or {}).get("score_pct") or 0))
        llm_status["candidates"] = len(_ai_picks)
        _gen = 0
        for r in _ai_picks[:40]:  # effectively all High-conviction actionable; 40 = safety ceiling
            note = llm.analyst_note(r, CONFIG, regime=regime, macro=macro)
            if note:
                r["ai_read"] = note
                _gen += 1
        llm_status["generated"] = _gen
        # If nothing was produced but candidates existed, probe once to surface WHY.
        if _gen == 0 and _ai_picks:
            try:
                llm_status["probe"] = llm.diagnose(CONFIG)
            except Exception as exc:  # noqa: BLE001
                llm_status["probe"] = {"ok": False, "error": str(exc)[:200]}

    # AI market brief (worker): a plain-English "what's happening / what to watch" summary at the
    # top of the dashboard, built only from data we already have. Live + LLM only; never breaks.
    market_brief = None
    if live and CONFIG.llm_enabled:
        try:
            market_brief = llm.market_brief(regime, shown, sectors, momentum_rows, news_ideas, macro, CONFIG)
        except Exception:  # noqa: BLE001
            market_brief = None

    # What-changed worker: meaningful diffs vs the previous build (new High calls, flips, sector shifts).
    try:
        changes = _compute_changes(shown, sectors, news_ideas, _today0, live)
    except Exception:  # noqa: BLE001
        changes = []

    # Event calendar (worker): earnings this week (from fundamentals already fetched) + key macro
    # releases (FRED). Event-risk awareness; never breaks the build.
    calendar = {"earnings": [], "econ": []}
    try:
        _ew = [{"symbol": r["symbol"], "days": (r.get("fundamentals") or {}).get("earnings_days"),
                "date": (r.get("fundamentals") or {}).get("earnings_date")}
               for r in shown
               if (r.get("fundamentals") or {}).get("earnings_days") is not None
               and 0 <= (r.get("fundamentals") or {}).get("earnings_days") <= 7]
        calendar["earnings"] = sorted(_ew, key=lambda x: x["days"])[:12]
        if live:
            calendar["econ"] = research.econ_calendar(CONFIG)
    except Exception:  # noqa: BLE001
        calendar = {"earnings": [], "econ": []}

    # S&P 500 benchmark (SPY) for chart overlay.
    benchmark = None
    try:
        from data import get_bars, synthetic_bars
        bdf = get_bars("SPY", CONFIG) if live else synthetic_bars("SPY", n=CONFIG.lookback_days)
        if bdf is not None and len(bdf):
            bdf = bdf.tail(300)
            benchmark = {
                "symbol": "SPY", "name": "S&P 500",
                "t": [int(pd.Timestamp(d).timestamp() * 1000) for d in bdf.index],
                "close": [round(float(x), 2) for x in bdf["close"]],
            }
    except Exception:  # noqa: BLE001
        benchmark = None

    # Track record: log new BUYs and grade past calls against real prices.
    import tracker
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        track = tracker.run(shown, CONFIG, live, today, regime=macro_posture)
    except Exception:  # noqa: BLE001
        track = None

    # Meta-signal model: a second-opinion verdict (accept/reduce/delay/reject) on every actionable
    # candidate, from regime fit + liquidity + conflicts + how this regime has paid off historically.
    try:
        import meta as _meta
        for r in shown:
            if r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
                r["meta"] = _meta.evaluate(r, macro_posture=macro_posture, track=track, cfg=CONFIG)
    except Exception:  # noqa: BLE001
        pass

    # Structured signal output: one tidy record per actionable trade (confidence, expected return
    # range, hold, risk/liquidity/uncertainty scores, size rec, kill conditions, meta verdict).
    structured = []
    try:
        import structured as _structured
        for r in shown:
            if r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
                _so = _structured.build(r, macro_posture, CONFIG)
                if _so:
                    r["structured"] = _so
                    structured.append(_so)
    except Exception:  # noqa: BLE001
        structured = []

    # No-trade intelligence layer: one unified "should we be trading right now?" read (macro event,
    # abnormal vol, deteriorating performance, drawdown). Computed before paper so it can gate entries.
    try:
        import notrade as _notrade
        notrade_gate = _notrade.market_gate(CONFIG, macro_posture=macro_posture, macro=macro,
                                            calendar=calendar, track=track, risk=None,
                                            timing=timing_posture, today=today)
    except Exception:  # noqa: BLE001
        notrade_gate = {"block_new": False, "reasons": [], "cautions": [], "checks": []}

    # Optional REAL paper-trading record (opt-in via PAPER_TRADE) — submits bracket orders for
    # fresh High-conviction signals and reads the live paper account. Disabled -> None.
    try:
        import paper as _paper
        paper_acct = _paper.run(shown, CONFIG, today, exposure_mult=_exposure_mult,
                                regime=macro_posture, notrade_block=notrade_gate.get("block_new", False))
    except Exception:  # noqa: BLE001
        paper_acct = None

    # Aggregate open-book risk (finance-skills recipe): portfolio heat (total open risk-to-stop) +
    # a parametric 95% VaR/CVaR on the positions held right now. Advisory; None when the book is flat.
    book_risk = None
    try:
        import portfolio_risk as _pr
        _open_theses = [t for t in (_load_json_safe("track_record.json") or []) if t.get("status") == "open"]
        _positions = (paper_acct or {}).get("positions") or []
        book_risk = _pr.book_risk(_positions, _open_theses, (paper_acct or {}).get("equity"), CONFIG)
        # Correlation clustering: which held names actually move together (>0.75)? Build daily returns
        # from the chart closes already in memory for the open positions, then cluster. A cluster of
        # 2+ is really ONE bet — surfaced so the book can't fake diversification.
        if book_risk:
            _held = {p.get("symbol") for p in _positions if p.get("symbol")}
            _ret = {}
            for r in shown:
                if r.get("symbol") in _held:
                    _cl = ((r.get("chart") or {}).get("close")) or []
                    if len(_cl) >= 45:
                        _c = [float(x) for x in _cl]
                        _ret[r["symbol"]] = [_c[i] / _c[i - 1] - 1.0 for i in range(1, len(_c)) if _c[i - 1]]
            _clusters = _pr.correlation_clusters(_ret, threshold=getattr(CONFIG, "corr_cluster_threshold", 0.75))
            book_risk["correlated_clusters"] = _clusters
            book_risk["effective_bets"] = len(_positions) - sum(len(c) - 1 for c in _clusters)
    except Exception:  # noqa: BLE001
        book_risk = None

    # Re-evaluate the no-trade gate WITH the risk-engine state (from paper) so the panel unifies it.
    try:
        import notrade as _notrade
        notrade_gate = _notrade.market_gate(CONFIG, macro_posture=macro_posture, macro=macro,
                                            calendar=calendar, track=track,
                                            risk=(paper_acct or {}).get("risk"),
                                            timing=timing_posture, book_risk=book_risk, today=today)
    except Exception:  # noqa: BLE001
        pass

    # Pairs / mean-reversion diversifier (gated). Market-neutral spread bets on related names —
    # leans in when the tape is trendless. Any failure -> empty list; never breaks the build.
    try:
        import pairs as _pairs
        pairs_data = _pairs.scan(CONFIG, live=live, regime=regime, macro_posture=macro_posture)
    except Exception:  # noqa: BLE001
        pairs_data = {"pairs": [], "regime_fit": False, "note": ""}

    # Alerts: ping configured channels when a NEW high-conviction signal appears (deduped).
    try:
        import notify as _notify
        alerts = _notify.run(shown, today)
    except Exception:  # noqa: BLE001
        alerts = None
    # Pairs alerts: ping when a spread stretches to its ±2σ entry band (deduped, once/day/pair).
    try:
        import notify as _notify
        _notify.run_pairs(pairs_data, today)
    except Exception:  # noqa: BLE001
        pass

    # Stocks-in-play ORB pass (its own strategy bucket): rank in-play names, score breakouts,
    # grade + learn, enforce hard caps. Fail-silent; never breaks the build.
    orb_payload = _run_orb(rows, _idea_map, nlp_scores, regime, CONFIG, live, today,
                           getattr(CONFIG, "starting_cash", 100_000.0))
    try:
        import notify as _notify
        _notify.run_orb(orb_payload.get("signals"), today)
    except Exception:  # noqa: BLE001
        pass

    # System status — a live readout of what's actually wired/running (booleans only, no secrets).
    import os as _os
    _has_worker = bool(CONFIG.live_quotes_url)
    # AI is "on" when the key is set AND there's no known API error. (llm_status has no top-level
    # 'ok'; a failed probe — e.g. exhausted credits / bad key — sets probe.ok = False.)
    _llm = llm_status or {}
    _llm_probe_failed = isinstance(_llm.get("probe"), dict) and _llm["probe"].get("ok") is False
    _llm_on = bool(CONFIG.llm_enabled) and not _llm_probe_failed
    _gen = _llm.get("generated")
    if not CONFIG.llm_enabled:
        _ai_note = "ANTHROPIC_API_KEY not set"
    elif _llm_probe_failed:
        _ai_note = "API error — check credits/key"
    elif _gen:
        _ai_note = f"{CONFIG.llm_model} · {_gen} brief(s) this run"
    else:
        _ai_note = f"{CONFIG.llm_model} · no High-conviction names this run"
    system = {
        "mode": mode,
        "feeds": [
            {"name": "Alpaca (prices/quotes)", "on": bool(CONFIG.api_key and CONFIG.secret_key),
             "note": f"{mode} account · IEX feed"},
            {"name": "Yahoo Finance (charts/history)", "on": _has_worker, "note": "via Cloudflare Worker proxy"},
            {"name": "Finnhub (fundamentals/analysts/earnings)", "on": bool(CONFIG.finnhub_api_key), "note": "API key"},
            {"name": "FRED (macro backdrop)", "on": bool(CONFIG.fred_api_key), "note": "API key"},
            {"name": "SEC EDGAR (insider Form 4)", "on": True, "note": "keyless, official"},
            {"name": "StockTwits (retail buzz)", "on": _has_worker, "note": "via Worker"},
            {"name": "TradingView (TA cross-check)", "on": _has_worker, "note": "via Worker"},
            {"name": "News RSS + Benzinga", "on": True, "note": "Yahoo/Finnhub/Benzinga headlines"},
        ],
        "ai": [
            {"name": "AI analyst briefs", "on": _llm_on, "note": _ai_note},
            {"name": "News-idea engine", "on": _llm_on,
             "note": "headlines → ideas + nudge" if _llm_on else "needs ANTHROPIC_API_KEY + credits"},
        ],
        "engine": [
            {"name": "Multi-strategy confluence (7 long + 7 short)", "on": True, "note": "core engine"},
            {"name": "Relative-strength factor", "on": True, "note": f"conviction weight {CONFIG.rs_conviction_weight}"},
            {"name": "Post-earnings drift (PEAD)", "on": bool(CONFIG.pead_enabled), "note": "confluence input"},
            {"name": "Earnings gate", "on": True, "note": "no fresh entry ≤2d to report"},
            {"name": "Regime-alignment tilt", "on": True, "note": "with/against the tape"},
            {"name": "Regime block on buys", "on": bool(CONFIG.regime_block_buys), "note": "demote longs in Risk-off"},
            {"name": "Conviction×vol sizing", "on": True, "note": "backtest + paper"},
        ],
        "execution": [
            {"name": "Auto paper-trading", "on": bool(CONFIG.paper_trade),
             "note": f"risk {CONFIG.paper_risk_pct:.0%}/trade · max {CONFIG.paper_max_open} open" if CONFIG.paper_trade else "set PAPER_TRADE=true"},
            {"name": "Live exit manager (partials/trail)", "on": bool(CONFIG.manage_exits), "note": "amends real OCO legs"},
            {"name": "Wider universe", "on": bool(CONFIG.wide_universe), "note": "expanded scan pool"},
            {"name": "Time-stop", "on": CONFIG.max_hold_days > 0, "note": f"{CONFIG.max_hold_days}d" if CONFIG.max_hold_days else "off"},
        ],
        "scrapers": [
            {"name": "Insider buys (SEC)", "on": True}, {"name": "Retail buzz (StockTwits)", "on": _has_worker},
            {"name": "Analyst rating changes (Finnhub)", "on": bool(CONFIG.finnhub_api_key)},
        ],
        "delivery": [
            {"name": "Phone push (ntfy)", "on": bool(_os.getenv("ALERT_NTFY_TOPIC"))},
            {"name": "Webhook (Discord/Slack)", "on": bool(_os.getenv("ALERT_WEBHOOK_URL"))},
            {"name": "Email (SMTP)", "on": bool(_os.getenv("ALERT_EMAIL_TO") and _os.getenv("SMTP_HOST"))},
            {"name": "Morning digest + intraday alerts", "on": True, "note": "scheduled tasks"},
        ],
        "infra": [
            {"name": "Cloudflare Worker proxy", "on": _has_worker, "note": "quotes/charts/TV/StockTwits"},
            {"name": "GitHub Actions rebuild", "on": True, "note": "every ~30 min, market hours + on push"},
            {"name": "GitHub Pages hosting", "on": True, "note": "static site"},
        ],
    }

    # What the bot has learned — per strategy (daily/swing vs intraday), for the dashboard panel.
    learned_payload = None
    try:
        import attribution as _attr2
        _mn2 = CONFIG.adaptive_min_n
        learned_payload = {
            "min_n": _mn2,
            "daily": {"weights": daily_learned, "report": _attr2.report(scope="daily")},
            "intraday": {"weights": intraday_learned, "report": _attr2.report(scope="intraday")},
            "orb": (orb_payload.get("learned") or {"weights": {}, "report": _attr2.report(scope="orb")}),
            # retired checks (weight driven to 0) + any direction the learned gate is suppressing
            "retired": sorted([lbl for lbl, m in daily_learned.items() if m == 0.0]),
            "direction_edge": _attr2.direction_edge(scope="daily"),
            "direction_gated": dir_gate,
        }
    except Exception:  # noqa: BLE001 - additive; never break the build
        learned_payload = None

    # Latest autonomous-analyst report (written nightly by analyst.py in its own cloud job).
    analyst_payload = None
    try:
        import os as _os2
        with open(_os2.getenv("ANALYST_FILE", "analyst_report.json")) as _af:
            analyst_payload = json.load(_af)
    except Exception:  # noqa: BLE001
        analyst_payload = None

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT"),
        "generated_ts": int(datetime.now(timezone.utc).timestamp()),
        "mode": mode,
        "scanned": len(rows),
        "diagnostics": list(scanner.LAST_ERRORS),
        "xai_status": _xai_status,
        "xai_buzz": xai_buzz,
        "retail_trending": retail_trending,
        "premium_selling": premium_selling,
        "audit_summary": None,  # filled by main() after the audit — kept early so it survives a truncated fetch
        "news_sources": dict(__import__("collections").Counter(
            (n.get("source") or "?") for n in news).most_common(14)),
        "llm": llm_status,
        "benchmark": benchmark,
        "track": track,
        "paper_acct": paper_acct,
        "paper_held": sorted({p.get("symbol") for p in ((paper_acct or {}).get("positions") or [])
                              if p.get("symbol")}),
        "news_ideas": news_ideas,
        "alerts": alerts,
        "system": system,
        "regime": regime,
        "sectors": sectors,
        "concentration": _concentration(shown),
        "macro": macro,
        "macro_posture": macro_posture,
        "timing": timing_posture,
        "setups_study": _load_json_safe("setups_study.json"),
        "performance": _perf_metrics(),
        "book_risk": book_risk,
        "changelog": _load_json_safe("changelog.json"),
        "notrade": notrade_gate,
        "price_drops": price_drops,
        "momentum": [dict(m, name=scanner.name_of(
                        m["symbol"], {r["symbol"]: r.get("name", "") for r in shown}.get(m["symbol"], "")))
                     for m in momentum_rows],
        "mom_detail": _mom_detail(momentum_rows, rows_by_sym, shown),
        "allweather": all_weather,
        "momentum_bt": momentum_bt,
        "walkforward": walk_fwd,
        "portfolio": _portfolio(shown),
        "ipos": ipos,
        "ipo_news": ipo_news,
        "params": {
            "fast_ma": CONFIG.fast_ma, "slow_ma": CONFIG.slow_ma,
            "rsi_period": CONFIG.rsi_period, "risk_per_trade": CONFIG.risk_per_trade,
            "stop_loss_pct": CONFIG.stop_loss_pct, "take_profit_pct": CONFIG.take_profit_pct,
            "rel_volume_window": CONFIG.rel_volume_window,
            "intraday_timeframe": CONFIG.intraday_timeframe,
        },
        "signals": shown,
        "ranked": ranked,
        "structured": structured,
        "nlp_scores": nlp_scores,
        "pairs": pairs_data,
        "intraday": intraday_shown,
        "intraday_track": intraday_track,
        "orb": orb_payload,
        "learned": learned_payload,
        "analyst": analyst_payload,
        "market_brief": market_brief,
        "changes": changes,
        "calendar": calendar,
        "charts": {k: charts[k] for k in shown_syms if k in charts},
        "news": news,
    }


def _regime_html(reg: dict | None) -> str:
    if not reg:
        return ""
    palette = {"Risk-on": ("#15361f", "#7ee2a0"), "Neutral": ("#3a2e12", "#b5b5ba"),
               "Risk-off": ("#3a1e1e", "#ff9b9b")}
    bg, fg = palette.get(reg["label"], ("#1a212b", "var(--txt2)"))
    return (f'<div class="regime" style="background:{bg};">'
            f'<span class="rlabel" style="color:{fg};">Market: {reg["label"]}</span>'
            f'<span class="rdetail">{reg["breadth"]}% of {reg["total"]} scanned above trend &middot; '
            f'avg momentum {reg["avg_rsi"]}/100 &middot; {reg["buys"]} fresh buys</span>'
            f'<span class="rnote">{reg["note"]}</span></div>')


def _kpi_html(reg: dict | None, snap: dict) -> str:
    """A summary strip of KPI tiles up top — the 'what matters now' inverted pyramid."""
    sigs = snap.get("signals", [])
    n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
    n_short = sum(1 for s in sigs if s.get("action") == "SHORT")
    tone = {"Risk-on": "buy", "Neutral": "warn", "Risk-off": "sell"}.get((reg or {}).get("label"), "")
    tk = snap.get("track") or {}
    wr = tk.get("win_rate")
    wr_txt = f'{wr}%' if isinstance(wr, (int, float)) else "—"

    def tile(label, value, cls="", sub=""):
        v = f'<div class="kpi-v {cls}">{value}</div>'
        s = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return f'<div class="kpi"><div class="kpi-l">{label}</div>{v}{s}</div>'

    tiles = ""
    if reg:
        tiles += (f'<div class="kpi hero"><div class="kpi-l">Market regime</div>'
                  f'<div class="kpi-v {tone}">{reg.get("label", "—")}</div>'
                  f'<div class="kpi-sub">{reg.get("note", "")[:60]}</div></div>')
        tiles += tile("Breadth", f'{reg.get("breadth", "—")}%', "", f'of {reg.get("total","?")} above trend')
        tiles += tile("Avg momentum", f'{reg.get("avg_rsi", "—")}', "", "RSI, 0–100")
    tiles += tile("Fresh buys", str(n_buy), "buy" if n_buy else "", "new long setups")
    tiles += tile("Fresh shorts", str(n_short), "sell" if n_short else "", "new short setups")
    tiles += tile("Track record", wr_txt, "", f'{tk.get("resolved", 0)} calls resolved')
    return f'<div class="kpis">{tiles}</div>'


def _meganav() -> str:
    """xAI-style top navigation: a few top-level groups, each dropping a mega-menu of pages with a
    one-line description and a tiny visual mockup of that page. Items proxy-click the (hidden) sidebar
    buttons so routing is untouched. Pure-CSS hover open; a small JS delegate handles the click."""
    M = {
        "rows": "<rect x='10' y='12' width='40' height='3' rx='1.5' fill='#8a8a8f'/><rect x='10' y='23' width='34' height='3' rx='1.5' fill='#5a5a60'/><rect x='10' y='34' width='44' height='3' rx='1.5' fill='#5a5a60'/><circle cx='62' cy='13' r='3' fill='#5ed6a6'/>",
        "candles": "<rect x='16' y='18' width='4' height='16' fill='#5ed6a6'/><rect x='28' y='12' width='4' height='24' fill='#5ed6a6'/><rect x='40' y='22' width='4' height='14' fill='#f0797f'/><rect x='52' y='16' width='4' height='20' fill='#5ed6a6'/>",
        "donut": "<circle cx='38' cy='24' r='13' fill='none' stroke='#5a5a60' stroke-width='5'/><circle cx='38' cy='24' r='13' fill='none' stroke='#5ed6a6' stroke-width='5' stroke-dasharray='40 82' transform='rotate(-90 38 24)'/>",
        "chartup": "<polyline points='10,34 24,28 34,30 48,18 64,12' fill='none' stroke='#5ed6a6' stroke-width='2'/>",
        "text": "<rect x='10' y='14' width='48' height='3' rx='1.5' fill='#8a8a8f'/><rect x='10' y='23' width='40' height='3' rx='1.5' fill='#5a5a60'/><rect x='10' y='32' width='52' height='3' rx='1.5' fill='#5a5a60'/>",
        "nodes": "<line x1='38' y1='14' x2='22' y2='34' stroke='#5a5a60'/><line x1='38' y1='14' x2='54' y2='34' stroke='#5a5a60'/><circle cx='38' cy='14' r='4' fill='#8a8a8f'/><circle cx='22' cy='34' r='4' fill='#5ed6a6'/><circle cx='54' cy='34' r='4' fill='#8a8a8f'/>",
        "tv": "<rect x='18' y='12' width='40' height='24' rx='3' fill='none' stroke='#8a8a8f'/><path d='M34 19 L44 24 L34 29 Z' fill='#f0797f'/>",
    }
    groups = [
        ("Trading", [("signals", "Signals", "Live conviction-scored ideas", "rows"),
                     ("control", "Control", "Tune the engine · accept/reject", "donut"),
                     ("markets", "Markets", "Charts, sectors, macro", "candles"),
                     ("portfolio", "Portfolio", "Positions & allocation", "donut"),
                     ("premium", "Premium selling", "Where selling options premium pays", "candles")]),
        ("Research", [("intel", "Intel", "Data-driven signals", "text"),
                      ("news", "News", "Market-moving headlines", "text"),
                      ("track", "Track record", "How past calls did", "chartup"),
                      ("analytics", "Edge explorer", "Waffle heatmaps of what wins", "donut"),
                      ("analyst", "Analyst", "Nightly self-review", "text")]),
        ("AI", [("brain", "Engine brain", "See the mechanics visually", "nodes"),
                ("agents", "Agents", "The agent ecosystem", "nodes")]),
        ("More", [("livetv", "Live TV", "Financial news streams", "tv"),
                  ("whatsnew", "What's new", "Latest changes", "text"),
                  ("about", "About", "How it works", "text"),
                  ("system", "System", "Config & health", "text")]),
    ]
    def mock(k):
        return f"<div class='mg-mock'><svg viewBox='0 0 76 48' xmlns='http://www.w3.org/2000/svg'>{M.get(k, M['text'])}</svg></div>"
    out = ['<nav class="meganav" id="megaNav">']
    for gname, items in groups:
        links = "".join(
            f'<a class="mg-link" data-go="{area}">{mock(mk)}'
            f'<div><div class="mg-t">{title}</div><div class="mg-d">{desc}</div></div></a>'
            for area, title, desc, mk in items)
        out.append(f'<div class="mg-item"><span class="mg-lbl">{gname} <span class="mg-car">&#9662;</span></span>'
                   f'<div class="mg-panel">{links}</div></div>')
    out.append('</nav>')
    return "".join(out)


def _signals_hero(snap: dict) -> str:
    """xAI-style Signals hero: a big thin headline, one-line thesis, two pill actions, and a KPI trio
    wired to real snapshot data (actionable count, realised win rate, current regime). Never raises."""
    sigs = snap.get("signals") or []
    n_live = sum(1 for s in sigs if s.get("action") in ("BUY", "SHORT"))
    tk = snap.get("track") or {}
    # Shorts are cut, so the honest headline is the LONG (active) book — not the blend that still
    # averages in the dead short trades. Fall back to blended only if shorts are re-enabled.
    _ldir = (tk.get("by_direction") or {}).get("LONG") or {}
    if not getattr(CONFIG, "allow_shorts", False) and isinstance(_ldir.get("win_rate"), (int, float)):
        wr = _ldir.get("win_rate")
        resolved = _ldir.get("n", tk.get("resolved", 0))
        wr_label = f"Win rate · {resolved} long"
    else:
        wr = tk.get("win_rate")
        resolved = tk.get("resolved", 0)
        wr_label = f"Win rate · {resolved} resolved"
    wr_txt = f'{wr}%' if isinstance(wr, (int, float)) else "—"
    def _is_long(s): return s.get("direction") == "LONG" or s.get("action") in ("BUY", "HOLD LONG", "WATCH LONG")
    def _is_short(s): return s.get("direction") == "SHORT" or s.get("action") in ("SHORT", "HOLD SHORT", "WATCH SHORT")
    n_long = sum(1 for s in sigs if _is_long(s))
    n_short = sum(1 for s in sigs if _is_short(s))
    _cv = [(s.get("conviction") or {}).get("score_pct") for s in sigs]
    _cv = [c for c in _cv if isinstance(c, (int, float))]
    avg_conv = round(sum(_cv) / len(_cv)) if _cv else "—"
    reg = snap.get("regime") or {}
    mp = snap.get("macro_posture") or {}
    ranked = snap.get("ranked") or []
    _breadth = reg.get("breadth")
    breadth_txt = f'{_breadth}%' if _breadth is not None else "—"
    rsi_txt = f'{reg.get("avg_rsi")}' if reg.get("avg_rsi") is not None else "—"
    _expo = mp.get("exposure_mult")
    expo_txt = f'{_expo:.2f}×' if _expo else "—"
    # extra stat-board metrics (fill the hero + reflect the learned engine)
    _pw = [s.get("p_win") for s in sigs if isinstance(s.get("p_win"), (int, float))]
    avg_pwin = f'{round(100 * sum(_pw) / len(_pw))}%' if _pw else "—"
    _pwc = _tone_pct(round(100 * sum(_pw) / len(_pw)) if _pw else None)
    _tg = [(s.get("plan") or {}).get("target_pct") for s in sigs]
    _tg = [t for t in _tg if isinstance(t, (int, float))]
    avg_tgt = f'+{round(sum(_tg) / len(_tg), 1)}%' if _tg else "—"
    scanned_txt = str(snap.get("scanned") or len(sigs))
    n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
    n_watch = sum(1 for s in sigs if str(s.get("action") or "").startswith("WATCH LONG"))

    def _lv(x):
        a = x.get("action") or ""
        if a in ("FLAT", "EXIT", "AVOID"):
            return False
        if (x.get("days_old") or 0) > 14:
            return False
        p = x.get("quote_price") if x.get("quote_price") is not None else x.get("price")
        pl = x.get("plan") or {}
        if p is not None and pl.get("target") is not None and pl.get("stop") is not None:
            lng = x.get("direction") != "SHORT"
            if lng and (p >= pl["target"] or p <= pl["stop"]):
                return False
            if (not lng) and (p <= pl["target"] or p >= pl["stop"]):
                return False
        return True
    _top = max((x for x in sigs if _lv(x)), key=lambda x: (x.get("rank_score") or 0), default=None)
    top_txt = (_top.get("symbol") if _top else None) or "—"
    reg_lab = reg.get("label", "—")
    tone = {"Risk-on": "buy", "Neutral": "warn", "Risk-off": "sell"}.get(reg_lab, "")
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return (
        '<div class="hero-x"><div class="hx-eyebrow">Live · US equities</div><div class="hx-title">Signals</div>'
        '<p class="hx-sub">A self-adapting read on US equities — conviction scored from every check, '
        'graded against real outcomes, with a live social pulse from Grok.</p>'
        '<div class="hx-btns">'
        '<button class="hx-btn solid" onclick="location.reload()">Refresh &rarr;</button>'
        '<button class="hx-btn ghost" onclick="window._showPage&&_showPage(\'markets\')">Watchlist</button>'
        '</div>'
        '<div class="hx-kpis">'
        f'<div class="hx-kpi"><div class="v">{n_live}</div><div class="k">Live today</div></div>'
        f'<div class="hx-kpi hint" data-tiphtml="{_esc_attr(_callout("Win rate — resolved calls", [("Win rate", (wr if isinstance(wr,(int,float)) else None), wr_txt, _tone_pct(wr if isinstance(wr,(int,float)) else None))], note=f"{resolved} calls resolved"))}"><div class="v">{wr_txt}</div><div class="k">{wr_label}</div></div>'
        f'<div class="hx-kpi"><div class="v {tone}">{esc(reg_lab)}</div><div class="k">Regime</div></div>'
        f'<div class="hx-kpi"><div class="v">{n_buy}</div><div class="k">Fresh buys</div></div>'
        f'<div class="hx-kpi"><div class="v">{avg_conv}</div><div class="k">Avg conviction</div></div>'
        f'<div class="hx-kpi hint" data-tiphtml="{_esc_attr(_callout("Model win-probability", [("Avg P(win)", (round(100*sum(_pw)/len(_pw)) if _pw else None), avg_pwin, _pwc)], note="learned meta-label read on live longs"))}"><div class="v" style="color:var(--accent);">{avg_pwin}</div><div class="k">Avg P(win)</div></div>'
        f'<div class="hx-kpi hint" data-tiphtml="{_esc_attr(_callout("Market breadth", [("Above trend", (_breadth if isinstance(_breadth,(int,float)) else None), breadth_txt, _tone_pct(_breadth if isinstance(_breadth,(int,float)) else None, 55))], note="share of scanned names in an uptrend"))}"><div class="v">{breadth_txt}</div><div class="k">Breadth</div></div>'
        f'<div class="hx-kpi"><div class="v">{rsi_txt}</div><div class="k">Avg momentum</div></div>'
        f'<div class="hx-kpi"><div class="v">{avg_tgt}</div><div class="k">Avg target</div></div>'
        f'<div class="hx-kpi"><div class="v">{expo_txt}</div><div class="k">Position sizing</div></div>'
        f'<div class="hx-kpi"><div class="v">{scanned_txt}</div><div class="k">Scanned today</div></div>'
        f'<div class="hx-kpi"><div class="v">{top_txt}</div><div class="k">Top opportunity</div></div>'
        '</div></div>'
    )


def _stat_strip(snap: dict) -> str:
    """xAI-style stat callout strip — big number + small muted label, in a row.
    Uses facets NOT shown in the hero KPIs or bento (to avoid duplicate metrics)."""
    sigs = snap.get("signals") or []
    n_total = len(sigs)
    def _is_long(s):
        return s.get("direction") == "LONG" or s.get("action") in ("BUY", "HOLD LONG", "WATCH LONG")
    def _is_short(s):
        return s.get("direction") == "SHORT" or s.get("action") in ("SHORT", "HOLD SHORT", "WATCH SHORT")
    n_long = sum(1 for s in sigs if _is_long(s))
    n_short = sum(1 for s in sigs if _is_short(s))
    convs = [(s.get("conviction") or {}).get("score_pct") for s in sigs]
    convs = [c for c in convs if isinstance(c, (int, float))]
    avg_conv = round(sum(convs) / len(convs)) if convs else None
    tgts = [(s.get("plan") or {}).get("target_pct") for s in sigs]
    tgts = [t for t in tgts if isinstance(t, (int, float))]
    avg_tgt = round(sum(tgts) / len(tgts)) if tgts else None
    items = [
        (str(n_total), "signals scanned"),
        (f"{n_long} / {n_short}", "long / short"),
        (f"{avg_conv}" if avg_conv is not None else "—", "avg conviction"),
        (f"{avg_tgt}%" if avg_tgt is not None else "—", "avg target"),
    ]
    cells = "".join(
        f'<div class="sx-stat"><div class="sx-stat-v">{v}</div><div class="sx-stat-k">{k}</div></div>'
        for v, k in items
    )
    return f'<div class="sx-statstrip">{cells}</div>'


def _showcase(snap: dict) -> str:
    """xAI-style 2x2 showcase of the main sections — quiet cards: icon · label · one stat · Explore."""
    sigs = snap.get("signals") or []

    def _lv(x):
        a = x.get("action") or ""
        return not (a in ("FLAT", "EXIT", "AVOID") or (x.get("days_old") or 0) > 14)
    live = sorted((x for x in sigs if _lv(x)), key=lambda x: -(x.get("rank_score") or 0))
    n_live = len(live)
    top = live[0].get("symbol") if live else "—"

    reg = snap.get("regime") or {}
    reg_lab = reg.get("label", "—")
    tone = {"Risk-on": "up", "Neutral": "warn", "Risk-off": "dn"}.get(reg_lab, "")
    breadth = reg.get("breadth")
    tk = snap.get("track") or {}

    def card(icon, label, sub, act):
        return (f'<div class="sc-card" onclick="{act}"><div class="sc-ic">{_svg(icon, 19)}</div>'
                f'<div class="sc-t">{label}</div><div class="sc-s">{sub}</div>'
                f'<div class="sc-ex">Explore &rarr;</div></div>')

    sig_sub = f'{n_live} live &middot; top pick {top}'
    mkt_sub = f'<span class="{tone}">{reg_lab}</span>' + (f' &middot; {breadth}% breadth' if breadth is not None else "")
    tv_sub = "On air &middot; Yahoo Finance"
    tr_sub = f'{tk.get("win_rate", "—")}% win &middot; {tk.get("resolved", 0)} resolved'

    return ('<div class="showcase">'
            + card("sparkle", "Signals", sig_sub, "document.getElementById('cards').scrollIntoView()")
            + card("chart", "Markets", mkt_sub, "window._showPage&&_showPage('markets')")
            + card("tv", "Live TV", tv_sub, "window._showPage&&_showPage('livetv')")
            + card("receipt", "Track record", tr_sub, "window._showPage&&_showPage('track')")
            + '</div>')


def _bento_home(snap: dict) -> str:
    """The Signals home as a true HUD bento — a grid of varied-size tiles: a regime hero, the key
    metrics, the market brief and what-changed, all as modular boxes."""
    reg = snap.get("regime") or {}
    mp = snap.get("macro_posture") or {}
    sigs = snap.get("signals", [])
    n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
    n_short = sum(1 for s in sigs if s.get("action") == "SHORT")
    tk = snap.get("track") or {}
    wr = tk.get("win_rate")
    wr_txt = f'{wr}%' if isinstance(wr, (int, float)) else "—"
    ranked = snap.get("ranked") or []
    tone = {"Risk-on": "buy", "Neutral": "warn", "Risk-off": "sell"}.get(reg.get("label"), "")

    def t(label, val, cls="", sub=""):
        sb = f'<div class="bt-sub">{sub}</div>' if sub else ""
        return f'<div class="bt"><div class="bt-l">{label}</div><div class="bt-v {cls}">{val}</div>{sb}</div>'

    if not reg and not sigs:
        return ""
    # NB: regime, live count and win rate now live in the Signals hero — don't duplicate them here.
    # Breadth / Avg momentum / Position sizing / Top opportunity now live in the Signals hero
    # as naked stat callouts (single stat primitive). The bento keeps only the narrative tiles.
    tiles = ""
    brief = (snap.get("market_brief") or "").strip()
    # drop a redundant leading "**Market Brief**" title line — the tile already labels it
    if brief.lstrip("*# ").lower().startswith("market brief"):
        _nl = brief.find("\n")
        brief = brief[_nl + 1:].lstrip() if _nl != -1 else ""
    if brief:
        tiles += (f'<div class="bt wide"><div class="bt-l ai-ident">{_svg("ai",14)} Market brief</div>'
                  f'{_brief_panels(brief, snap.get("sectors"))}</div>')
    changes = snap.get("changes") or []
    if changes:
        tiles += (f'<div class="bt wide"><div class="bt-l">{_svg("bolt",14)} What changed since last build</div>'
                  '<ul class="bt-list">' + "".join(f"<li>{c}</li>" for c in changes) + "</ul></div>")
    return f'<div class="bento">{tiles}</div>'


def _allweather_html(aw: dict | None) -> str:
    intro = (_strat_badge("All-seasons allocation · static, ~yearly rebalance") +
             '<p style="color:var(--muted);font-size:13px;margin:0 0 14px;max-width:760px;">'
             "Ray Dalio's <b>All Weather</b> portfolio is a static, <b>risk-balanced</b> allocation built to "
             "hold up across all four economic environments — rising and falling growth, rising and falling "
             "inflation — instead of betting on which is next. It's a <b>buy-and-hold</b> mix you rebalance "
             "about once a year, <i>not</i> a trading signal. The trade-off: lower returns than all-stocks in "
             "a bull run, but much shallower drawdowns and steadier compounding.</p>")
    if not aw:
        return intro + '<p style="color:var(--muted);font-size:13px;">All Weather data unavailable.</p>'

    # allocation table
    body = ""
    colors = ["#2ea043", "#58a6ff", "#7aa2f7", "#b5b5ba", "#c08457"]
    bar = ""
    for i, t in enumerate(aw.get("targets", [])):
        px = f'${t["price"]:,.2f}' if t.get("price") is not None else "—"
        body += (f'<tr><td><b>{t["symbol"]}</b> <span style="color:var(--muted);font-weight:400;">{t["name"]}</span></td>'
                 f'<td style="color:var(--muted);">{t["role"]}</td>'
                 f'<td style="color:var(--muted);">{t["env"]}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{px}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;font-weight:700;">{t["weight"]}%</td></tr>')
        bar += (f'<div title="{t["symbol"]} {t["weight"]}%" style="width:{t["weight"]}%;'
                f'background:{colors[i % len(colors)]};">{t["weight"] if t["weight"] >= 7 else ""}</div>')
    table = ('<table class="trackrec"><thead><tr><th>Asset</th><th>Role</th>'
             '<th>Best environment</th><th style="text-align:right;">Price</th>'
             '<th style="text-align:right;">Target weight</th></tr></thead>'
             f'<tbody>{body}</tbody></table>')
    bar_html = (f'<div style="display:flex;height:26px;border-radius:6px;overflow:hidden;'
                f'margin:4px 0 18px;font-size:10px;color:#fff;font-weight:700;text-align:center;'
                f'line-height:26px;">{bar}</div>')

    # backtest vs SPY
    bt_html = ""
    bt = aw.get("backtest")
    if bt:
        a, s = bt["allweather"], bt["spy"]
        def _row(label, m, hot=False):
            w = 'font-weight:700;' if hot else ''
            ddc = 'color:var(--buy);' if m["maxdd"] > s["maxdd"] else ''
            return (f'<tr style="{w}"><td>{label}</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["ret"]:.0f}%</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["cagr"]:.1f}%</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["sharpe"]:.2f}</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;{ddc}">{m["maxdd"]:.1f}%</td></tr>')
        bt_html = (f'<h3 style="margin:18px 0 6px;">Backtest vs S&amp;P 500 — {bt["years"]} years '
                   f'({bt["start"]} → {bt["end"]}, monthly rebalance)</h3>'
                   '<table class="trackrec"><thead><tr><th>Portfolio</th>'
                   '<th style="text-align:right;">Total return</th><th style="text-align:right;">CAGR</th>'
                   '<th style="text-align:right;" title="risk-adjusted return — higher is better">Sharpe</th>'
                   '<th style="text-align:right;" title="worst peak-to-trough drop — closer to zero is better">Max drawdown</th>'
                   '</tr></thead><tbody>'
                   + _row("All Weather", a, hot=True) + _row("S&amp;P 500 (buy &amp; hold)", s)
                   + '</tbody></table>'
                   '<p style="color:var(--muted);font-size:12px;margin:8px 0 0;">The point isn\'t to beat the '
                   "S&amp;P on raw return — it usually won't in a bull market. It's the <b>shallower max drawdown</b> "
                   "and steadier ride (often a comparable or better Sharpe). If you can't stomach a deep stock-market "
                   "fall, that smoother path is the whole appeal.</p>")
    else:
        bt_html = ('<p style="color:var(--muted);font-size:12px;margin-top:10px;">'
                   f'Backtest unavailable ({aw.get("data_src", "no data")}). Allocation shown above.</p>')

    why = ('<div class="deskread" style="margin-top:16px;border-left-color:#58a6ff;">'
           '<b>Why these five?</b> Each sleeve is the asset that tends to do best in one environment, so '
           'something is usually working: stocks for rising growth, long Treasuries for falling growth/'
           'deflation, intermediate Treasuries as ballast, gold and commodities for rising inflation. They '
           'are weighted so no single one dominates the portfolio\'s <i>risk</i> — which is why bonds get a '
           'big nominal slice (they swing less than stocks).</div>')
    caveat = ('<p style="color:var(--muted);font-size:12px;margin-top:12px;">Educational only, not advice. '
              'Backtest ignores fees, taxes and fund expense ratios; past performance isn\'t a forecast. '
              'The 2022 simultaneous stock+bond drawdown was a notably hard stretch for this mix.</p>')
    return intro + bar_html + table + bt_html + why + caveat


def _ipo_html(ipos: list[dict], ipo_news: list[dict]) -> str:
    """Upcoming-IPO calendar + general headlines mentioning pre-IPO names (SpaceX etc.)."""
    if ipos:
        rows = ""
        for r in ipos[:30]:
            price = r.get("price") or "—"
            val = r.get("value")
            try:
                valtxt = f'${float(val) / 1e6:,.0f}M' if val else "—"
            except Exception:  # noqa: BLE001
                valtxt = "—"
            rows += (f'<tr><td>{r.get("date","")}</td><td><b>{r.get("name","")}</b></td>'
                     f'<td>{r.get("symbol","") or "—"}</td><td>{r.get("exchange","") or ""}</td>'
                     f'<td style="text-align:right;">{price}</td>'
                     f'<td style="text-align:right;color:var(--muted);">{valtxt}</td>'
                     f'<td>{r.get("status","") or ""}</td></tr>')
        cal = ('<table class="trackrec"><thead><tr><th>Date</th><th>Company</th><th>Ticker</th>'
               '<th>Exchange</th><th style="text-align:right;">Price</th>'
               '<th style="text-align:right;">Deal size</th><th>Status</th></tr></thead>'
               f'<tbody>{rows}</tbody></table>')
    else:
        cal = ('<p style="color:var(--muted);font-size:13px;">No companies have formally filed to list in the next '
               '~90 days. Rumoured deals like SpaceX appear in the buzz feed below until they file an S-1.</p>')
    if ipo_news:
        items = ""
        for n in ipo_news:
            t = (f'<a href="{n["url"]}" target="_blank" rel="noopener">{n["headline"]}</a>'
                 if n.get("url") else f'<span class="h">{n["headline"]}</span>')
            items += (f'<li>{t}<div class="src">{n.get("source","")} {n.get("created_at","")} '
                      f'&middot; <span class="chip mini neutral">{n.get("match","")}</span></div></li>')
        news = f'<ul class="news">{items}</ul>'
    else:
        news = '<p style="color:var(--muted);font-size:13px;">No pre-IPO headlines matched right now.</p>'
    return (f'<div class="sech" style="margin-top:0;">Upcoming IPO calendar</div>{cal}'
            '<div class="sech">Pre-IPO buzz — including private names like SpaceX</div>'
            '<p style="color:var(--muted);font-size:12.5px;margin:0 0 8px;">'
            'Private companies have no ticker, so they can\'t be scanned or charted — these are '
            'general-market headlines mentioning notable pre-IPO names and IPO filings.</p>'
            f'{news}')


def _has_bad_bar(cl: list, jump: float = 0.50) -> bool:
    """True if the close series contains a spike-and-revert bad print: a single-day move
    larger than `jump` that is undone (mostly) the very next day. A genuine split or a real
    trend move does NOT immediately reverse, so this isolates corrupt IEX bars without
    flagging legitimate big movers. One such bar anywhere in the ~12-month window can wreck
    the momentum base, so any occurrence disqualifies the name from the leaderboard."""
    for i in range(1, len(cl) - 1):
        p0, p1, p2 = cl[i - 1], cl[i], cl[i + 1]
        if not p0 or not p1:
            continue
        r1 = p1 / p0 - 1.0          # move into the suspect bar
        if abs(r1) <= jump:
            continue
        if not p2:
            return True             # huge move with no valid confirmation bar — distrust it
        r2 = p2 / p1 - 1.0          # move out of it
        # reverses if the next day undoes most of the spike (opposite sign, similar size)
        if r1 * r2 < 0 and abs(r2) >= jump * 0.6:
            return True
    return False


def _momentum_rank(charts: dict, top: int = 15, per_sector: int = 3) -> list[dict]:
    """Dual-momentum leaderboard: 12-1 momentum, kept only if positive AND above the
    200-day average. Then capped to ``per_sector`` names per sector (diversification)
    and assigned inverse-volatility suggested weights (risk-parity, like factor funds)."""
    max_mom = getattr(CONFIG, "max_momentum_pct", 200.0)
    jump = getattr(CONFIG, "bad_bar_jump_pct", 50.0) / 100.0
    cand = []
    for sym, ch in charts.items():
        cl = [c for c in (ch.get("close") or []) if c is not None]
        n = len(cl)
        if n < 230:
            continue
        # Bad-bar guard: a single-day move > `jump` that immediately reverses (spike-and-
        # revert) is the signature of a corrupt IEX print, not a real move or a split. Such
        # a bad bar ~12 months back deflates the momentum base and balloons the score
        # (this is what produced MU +591% / INTC +479%). Drop the name rather than trust it.
        if _has_bad_bar(cl, jump):
            continue
        lb = min(252, n - 1)
        sk = 21 if n > 257 else 0
        base = cl[-lb]
        recent = cl[-1 - sk] if sk else cl[-1]
        if not base:
            continue
        score = recent / base - 1
        sma200 = sum(cl[-200:]) / 200 if n >= 200 else sum(cl) / n
        if not (score > 0 and cl[-1] > sma200):
            continue
        # Score sanity cap: a 12-1 momentum above `max_mom`% on a scanned large/mid-cap is
        # almost certainly a leftover data artifact (a bad base the spike check missed), not
        # a tradeable winner. Drop it so the leaderboard never publishes impossible numbers.
        if score * 100 > max_mom:
            continue
        rets = [cl[i] / cl[i - 1] - 1 for i in range(max(1, n - 21), n) if cl[i - 1]]
        vol = (sum((x - sum(rets) / len(rets)) ** 2 for x in rets) / len(rets)) ** 0.5 if len(rets) > 1 else 0.0
        ext = (cl[-1] / sma200 - 1) * 100 if sma200 else 0.0
        r1m = (cl[-1] / cl[-22] - 1) * 100 if n >= 22 and cl[-22] else None
        cand.append({"symbol": sym, "score": round(score * 100, 1),
                     "price": round(cl[-1], 2), "sector": scanner.sector_of(sym),
                     "ext": round(ext, 1), "r1m": round(r1m, 1) if r1m is not None else None,
                     "_vol": vol})
    cand.sort(key=lambda x: -x["score"])
    out, cnt = [], {}
    for c in cand:
        if cnt.get(c["sector"], 0) >= per_sector:
            continue
        out.append(c)
        cnt[c["sector"]] = cnt.get(c["sector"], 0) + 1
        if len(out) >= top:
            break
    invs = [(1.0 / c["_vol"] if c["_vol"] > 1e-9 else 0.0) for c in out]
    tot = sum(invs) or 1.0
    for c, iv in zip(out, invs):
        c["weight"] = round(iv / tot * 100, 1)
        c.pop("_vol", None)
    return out


def _portfolio(rows: list[dict]) -> dict:
    """Aggregate the actionable signals into a hypothetical book — what you'd be holding
    if you took every BUY/SHORT/HOLD at the model's position size (risk-based, on
    starting_cash). Net/gross exposure, sector mix, total $ at risk, per-position list."""
    longs = [r for r in rows if r.get("action") in ("BUY", "HOLD LONG")]
    shorts = [r for r in rows if r.get("action") in ("SHORT", "HOLD SHORT")]
    def expo(r): return (r.get("plan") or {}).get("exposure") or 0.0
    def risk(r): return (r.get("plan") or {}).get("dollar_risk") or 0.0
    long_e = sum(expo(r) for r in longs)
    short_e = sum(expo(r) for r in shorts)
    sec: dict = {}
    for r in longs:
        sec[r.get("sector", "Other")] = sec.get(r.get("sector", "Other"), 0.0) + expo(r)
    for r in shorts:
        sec[r.get("sector", "Other")] = sec.get(r.get("sector", "Other"), 0.0) - expo(r)
    positions = [{
        "symbol": r["symbol"], "name": r.get("name", ""), "action": r["action"],
        "direction": r.get("direction"), "sector": r.get("sector", ""),
        "shares": (r.get("plan") or {}).get("shares"),
        "exposure": round(expo(r)), "risk": round(risk(r)),
        "conviction": (r.get("conviction") or {}).get("score_pct"),
    } for r in longs + shorts]
    positions.sort(key=lambda p: -(p["conviction"] or 0))
    return {
        "n_long": len(longs), "n_short": len(shorts),
        "long_exposure": round(long_e), "short_exposure": round(short_e),
        "gross": round(long_e + short_e), "net": round(long_e - short_e),
        "at_risk": round(sum(risk(r) for r in longs + shorts)),
        "starting_cash": CONFIG.starting_cash,
        "sectors": sorted(sec.items(), key=lambda kv: -abs(kv[1]))[:10],
        "positions": positions,
    }


def _portfolio_html(p: dict | None) -> str:
    badge = _strat_badge("Hypothetical book · every actionable signal at model size")
    if not p or not p.get("positions"):
        return badge + ('<p style="color:var(--muted);font-size:13px;">No actionable positions right '
                        'now — nothing to assemble into a book.</p>')
    cash = p.get("starting_cash") or 100000
    pct = lambda v: f'{v/cash*100:.0f}%'
    money = lambda v: f'${v:,.0f}'
    def tile(label, value, sub="", cls=""):
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        return f'<div class="stat"><div class="l">{label}</div><div class="v {cls}">{value}</div>{sub_html}</div>'
    netcls = "buy" if p["net"] >= 0 else "sell"
    tiles = (tile("Positions", f'{p["n_long"]}L / {p["n_short"]}S', "long / short") +
             tile("Gross exposure", money(p["gross"]), pct(p["gross"]) + " of book") +
             tile("Net exposure", ("+" if p["net"] >= 0 else "") + money(p["net"]), pct(p["net"]) + " net " + ("long" if p["net"] >= 0 else "short"), netcls) +
             tile("$ at risk", money(p["at_risk"]), pct(p["at_risk"]) + " if all stop out", "sell"))
    # sector mix bar (green long / red short, width by |exposure| share of gross)
    gross = p["gross"] or 1
    secbar = ""
    for name, val in p["sectors"]:
        w = abs(val) / gross * 100
        col = "var(--buy)" if val >= 0 else "var(--sell)"
        secbar += (f'<div class="secrow"><span class="secname">{name}</span>'
                   f'<div class="secbarwrap"><div class="secbarfill" style="width:{w:.0f}%;background:{col};"></div></div>'
                   f'<span class="secval" style="color:{col};">{("+" if val>=0 else "−")}{money(abs(val))}</span></div>')
    rows = ""
    for q in p["positions"]:
        acol = "var(--sell)" if q["direction"] == "SHORT" else "var(--buy)"
        _ew = abs(q["exposure"]) / gross * 100
        _rw = abs(q["risk"]) / (p["at_risk"] or 1) * 100
        _cv = q["conviction"]
        callout = _callout(f'{q["symbol"]} — {q["action"]}',
                           [("Exposure", _ew, pct(q["exposure"]), "up" if q["direction"] != "SHORT" else "dn"),
                            ("$ at risk", _rw, money(q["risk"]), "dn")],
                           note=f'{q["sector"]} · conviction {_cv if _cv is not None else "—"}')
        rows += (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}">'
                 f'<td><b>{q["symbol"]}</b> <span style="color:var(--muted);font-weight:400;">{q["name"][:22]}</span></td>'
                 f'<td style="color:{acol};">{q["action"]}</td><td style="color:var(--muted);">{q["sector"]}</td>'
                 f'<td style="text-align:right;">{q["shares"] or "—"}</td>'
                 f'<td style="text-align:right;">{money(q["exposure"])}</td>'
                 f'<td style="text-align:right;">{money(q["risk"])}</td>'
                 f'<td style="text-align:right;">{_cv if _cv is not None else "—"}</td></tr>')
    table = ('<table class="trackrec"><thead><tr><th>Position</th><th>Side</th><th>Sector</th>'
             '<th style="text-align:right;">Shares</th><th style="text-align:right;">Exposure</th>'
             '<th style="text-align:right;">$ risk</th><th style="text-align:right;">Conv</th></tr></thead>'
             f'<tbody>{rows}</tbody></table>')
    note = ('<p style="color:var(--muted);font-size:12px;margin-top:10px;">Hypothetical: assumes you take '
            'every actionable signal at the model\'s risk-based size on a '
            f'{money(cash)} book. Not advice; sizing is illustrative. Net exposure = long − short.</p>')
    return (badge + f'<div class="trackstats">{tiles}</div>'
            + '<div class="sech">Sector tilt <span style="text-transform:none;color:var(--muted);font-weight:400;">— net exposure by sector (green long · red short)</span></div>'
            + f'<div class="secmix">{secbar}</div>'
            + '<div class="sech">Positions</div>' + table + note)


def _strat_badge(value: str) -> str:
    """A small, consistent 'Strategy type: …' pill so every tab self-labels its approach."""
    return (f'<div class="strat-badge"><span class="k">Strategy type</span>'
            f'<span class="v">{value}</span></div>')


def _mom_detail(momentum_rows: list[dict], rows_by_sym: dict, shown: list[dict]) -> dict:
    """Full analysis row per momentum leader, keyed by symbol, so the leaderboard rows can
    open the same rich detail modal (chart, info, reasoning, conviction) as the signal cards.
    Prefers the already-enriched `shown` row (has research) and prepends a momentum-context line."""
    shown_by_sym = {r["symbol"]: r for r in shown}
    out = {}
    for m in momentum_rows:
        sym = m["symbol"]
        row = shown_by_sym.get(sym) or rows_by_sym.get(sym)
        if not row:
            continue
        d = dict(row)
        d["name"] = scanner.name_of(sym, d.get("name", ""))
        r1m = m.get("r1m")
        r1m_txt = f"{'+' if (r1m or 0) >= 0 else ''}{r1m}%" if r1m is not None else "—"
        mom_note = (f"{_svg('trend-up',13)} Momentum leader — 12-1 momentum +{m['score']}% (its return over the last ~12 "
                    f"months, skipping the most recent). It's +{m.get('ext', 0)}% above its 200-day "
                    f"average and did {r1m_txt} over the past month. Suggested risk-parity weight "
                    f"{m.get('weight', '—')}% in a monthly-rebalanced leaders basket — a positional "
                    f"hold, not a day-trade.")
        d["reasons"] = [mom_note] + list(d.get("reasons") or [])
        out[sym] = d
    return out


def _momentum_bt_html(bt: dict | None) -> str:
    """Honest, survivorship-bias-FREE momentum backtest (fixed ETF universe) vs SPY."""
    if not bt or not bt.get("strategy") or not bt.get("spy"):
        return ""
    m, s = bt["strategy"], bt["spy"]
    ddc = 'color:var(--buy);' if m["maxdd"] > s["maxdd"] else ''
    shc = 'color:var(--buy);' if m["sharpe"] > s["sharpe"] else ''
    rows = (f'<tr><td><b>Dual-momentum (ETFs)</b></td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["ret"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["cagr"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;{shc}">{m["sharpe"]:.2f}</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;{ddc}">{m["maxdd"]:.1f}%</td></tr>'
            f'<tr><td>SPY buy &amp; hold</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["ret"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["cagr"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["sharpe"]:.2f}</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["maxdd"]:.1f}%</td></tr>')
    uni = ", ".join(bt.get("universe", [])[:18])
    return ('<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">' + _svg('ruler',14) + ' Honest backtest — '
            'survivorship-bias-free</div>'
            f'<table class="trackrec" style="margin-top:8px;"><thead><tr><th>Strategy</th>'
            '<th style="text-align:right;">Total return</th><th style="text-align:right;">CAGR</th>'
            '<th style="text-align:right;" title="risk-adjusted return — higher is better">Sharpe</th>'
            f'<th style="text-align:right;">Max drawdown</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p style="color:var(--muted);font-size:12px;margin:10px 0 0;">Same dual-momentum rules run on a '
            f'<b>fixed universe of {bt.get("n_universe","~16")} broad ETFs</b> that existed for the whole '
            f'{bt.get("months","")}-month window and never delist ({uni}) — so the result can\'t be flattered '
            'by hindsight stock-picking. This is the honest performance read; the single-stock ranking below is '
            'for idea generation (and is survivorship-biased — it\'s today\'s winners).</p></div>')


def _walkforward_html(wf: dict | None) -> str:
    """Walk-forward / out-of-sample validation panel: IS vs OOS, per-fold, sensitivity sweep."""
    if not wf or wf.get("error") or not wf.get("per_symbol"):
        return ""
    grade = wf.get("grade", "marginal")
    gcol = {"holds up": "var(--buy)", "marginal": "var(--warn)", "fragile": "var(--sell)"}.get(grade, "var(--muted)")
    oos = wf.get("oos_avg_pct")
    pos = wf.get("oos_pos_folds_pct")
    folds = wf.get("oos_total_folds")
    syms = ", ".join(wf.get("symbols", []))

    # per-symbol OOS row table
    prows = ""
    for s in wf.get("per_symbol", []):
        oc = "buy" if (s.get("oos_avg_fold_pct") or 0) > 0 else "sell"
        prows += (f'<tr><td><b>{s["symbol"]}</b></td>'
                  f'<td style="text-align:right;">{s.get("is_avg_sharpe","—")}</td>'
                  f'<td style="text-align:right;" class="{oc}">{s.get("oos_avg_fold_pct",0):+.2f}%</td>'
                  f'<td style="text-align:right;">{s.get("oos_win_folds",0)}/{s.get("oos_n_folds",0)}</td></tr>')
    ptable = ('<table class="tbl" style="margin-top:8px;"><thead><tr><th>Symbol</th>'
              '<th style="text-align:right;" title="avg Sharpe on the in-sample windows the params were tuned on">IS Sharpe</th>'
              '<th style="text-align:right;" title="avg per-fold return on unseen out-of-sample windows">OOS return/fold</th>'
              '<th style="text-align:right;" title="profitable out-of-sample windows">OOS wins</th></tr></thead>'
              f'<tbody>{prows}</tbody></table>')

    # sensitivity sweep
    srows = ""
    for r in wf.get("sensitivity", []):
        srows += (f'<tr><td>{r["params"]}</td>'
                  f'<td style="text-align:right;">{r["total_return_pct"]:+.1f}%</td>'
                  f'<td style="text-align:right;">{r["sharpe"]:.2f}</td>'
                  f'<td style="text-align:right;">{r["max_dd_pct"]:.1f}%</td></tr>')
    stable = ('<h4 style="font-size:13px;margin:14px 0 4px;">Parameter sensitivity (full sample)</h4>'
              '<table class="tbl"><thead><tr><th title="fast/slow moving-average pair">MA pair</th>'
              '<th style="text-align:right;">Return</th><th style="text-align:right;">Sharpe</th>'
              '<th style="text-align:right;">Max DD</th></tr></thead>'
              f'<tbody>{srows}</tbody></table>') if srows else ""

    return ('<div class="ovbox" style="margin:0 0 16px;border-left:4px solid ' + gcol + ';">'
            '<div class="ovhead">' + _svg('microscope',14) + ' Walk-forward / out-of-sample validation — '
            f'<span style="color:{gcol};text-transform:capitalize;">{grade}</span></div>'
            f'<div style="font-size:12px;color:var(--muted);margin:6px 0 8px;">'
            f'OOS return/fold <b>{oos:+.2f}%</b> &nbsp;·&nbsp; profitable unseen windows <b>{pos}%</b> '
            f'({folds} folds across {len(wf.get("symbols",[]))} names) &nbsp;·&nbsp; basket: {syms}</div>'
            f'<p style="color:var(--txt2);font-size:13px;margin:0 0 4px;">{wf.get("verdict","")}</p>'
            + ptable + stable +
            '<p style="color:var(--muted);font-size:11px;margin:10px 0 0;">Each fold tunes the moving-average pair '
            'on past data only, then trades the next unseen window with those frozen settings — so OOS is a fair test '
            'of the edge, not a curve-fit. Net of modeled slippage. Educational; not investment advice.</p></div>')


def _momentum_html(rows: list[dict]) -> str:
    intro = (_strat_badge("Dual momentum · positional, ~monthly rebalance") +
             '<p style="color:var(--muted);font-size:13px;margin:0 0 12px;max-width:680px;">'
             'Ranked by <b>12-1 momentum</b> (return over the last ~12 months, skipping the most '
             'recent month), keeping only names in their own uptrend (above the 200-day average). '
             'This is the dual-momentum approach factor funds use — the one strategy that beat the '
             'index on risk-adjusted terms across a full cycle in our backtest.</p>')
    if not rows:
        return intro + ('<p style="color:var(--muted);font-size:13px;">Not enough price history to '
                        'rank momentum right now.</p>')
    body = ""
    for i, m in enumerate(rows, 1):
        nm = f' <span style="color:var(--muted);font-weight:400;">{m.get("name","")}</span>' if m.get("name") else ""
        r1m = m.get("r1m")
        if r1m is None:
            r1m_cell = '<td style="text-align:right;color:var(--muted);">—</td>'
        else:
            r1m_cell = (f'<td style="text-align:right;font-variant-numeric:tabular-nums;" '
                        f'class="{"win" if r1m >= 0 else "loss"}">{"+" if r1m >= 0 else ""}{r1m}%</td>')
        new = ' <span class="chip mini bull" style="font-size:9px;padding:0 5px;">NEW</span>' if m.get("is_new") else ""
        _r1 = m.get("r1m")
        _mcall = _callout(f'{m["symbol"]} — momentum leader',
                          [("12-1 momentum", min(100, m["score"]), f'+{m["score"]}%', "up"),
                           ("vs 200-day", min(100, m.get("ext", 0)), f'+{m.get("ext", 0)}%', "mut"),
                           ("1-month", (min(100, abs(_r1)) if isinstance(_r1, (int, float)) else None),
                            (f'{"+" if (_r1 or 0) >= 0 else ""}{_r1}%' if _r1 is not None else "—"),
                            "up" if (_r1 or 0) >= 0 else "dn")],
                          note=f'{m.get("sector","")} · suggested weight {m.get("weight","—")}%')
        body += (f'<tr class="momrow hint" data-sym="{m["symbol"]}" style="cursor:pointer;" '
                 f'data-tiphtml="{_esc_attr(_mcall)}">'
                 f'<td>{i}</td><td><b>{m["symbol"]}</b>{nm}{new}</td>'
                 f'<td style="color:var(--muted);">{m.get("sector","")}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;">${m["price"]:,.2f}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;" class="win">+{m["score"]}%</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);">+{m.get("ext",0)}%</td>'
                 f'{r1m_cell}'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;font-weight:600;">{m.get("weight","—")}%</td></tr>')
    table = ('<table class="trackrec"><thead><tr><th>#</th><th>Stock</th><th>Sector</th>'
             '<th style="text-align:right;">Price</th>'
             '<th style="text-align:right;" title="return over ~12 months, skipping the last month">12-1 momentum</th>'
             '<th style="text-align:right;" title="how far above its 200-day average — bigger = more extended">vs 200d</th>'
             '<th style="text-align:right;" title="last ~1 month return — negative means the leader is cooling off">1-mo</th>'
             '<th style="text-align:right;" title="suggested risk-parity weight: more to steadier names, less to volatile ones">Wt</th>'
             f'</tr></thead><tbody>{body}</tbody></table>'
             '<p style="color:var(--muted);font-size:12px;margin:8px 0 0;">'
             'Columns: <b>12-1 momentum</b> = ranking signal · <b>vs 200d</b> = how extended (high = chasing risk) · '
             '<b>1-mo</b> = recent month (negative = losing steam) · <b>Wt</b> = suggested inverse-volatility weight. '
             'Capped at 3 names per sector. Re-rank monthly; rotate out names that drop off. '
             '<span class="chip mini bull" style="font-size:9px;padding:0 5px;">NEW</span> = entered the list this run.</p>')
    caveats = ('<div class="deskread" style="margin-top:16px;border-left-color:#b5b5ba;">'
               '<b>Read before using.</b> This is a monthly-rebalanced approach (hold the leaders, '
               're-rank ~monthly) — not a day-trade list. In backtest it earned a higher Sharpe than '
               'the index over ~9 years <i>including</i> the 2022 bear, but with a deeper ~31% drawdown, '
               'and the figures are flattered by survivorship bias (this watchlist is today\'s winners). '
               'Expect a smaller real edge and real drawdowns. Educational only — not financial advice.</div>')
    return intro + table + caveats


def _sectors_html(secs: list[dict]) -> str:
    if not secs:
        return ""
    def _srow(s):
        pu = s["pct_up"]
        callout = _callout(f'{s["sector"]} — sector breadth',
                           [("Trending up", pu, f"{pu}%", _tone_pct(pu, 55))],
                           note=f'{s["count"]} names tracked')
        return (f'<div class="secrow hint" data-tiphtml="{_esc_attr(callout)}">'
                f'<span class="secname">{s["sector"]}</span>'
                f'<div class="secbar"><div class="secfill" style="width:{pu}%;"></div></div>'
                f'<span class="secpct">{pu}% up · {s["count"]}</span></div>')
    rows = "".join(_srow(s) for s in secs)
    return ('<div class="ovbox"><div class="ovhead">' + _svg('compass',14) + ' Sector strength '
            '<span style="font-weight:400;color:var(--muted);font-size:12px;">— share of each sector trending up</span></div>'
            f'{rows}</div>')


def _ranked_html(ranked: list | None, top: int = 12) -> str:
    """Adaptive allocation ranking — the best setups for capital, with a labelled factor breakdown."""
    if not ranked:
        return ""

    def fcell(v, cls=""):
        v = max(0, min(100, int(v or 0)))
        c = "var(--buy)" if v >= 67 else "var(--warn)" if v >= 40 else "var(--sell)"
        return (f'<td class="{cls}" style="min-width:62px;vertical-align:middle;">'
                f'<div style="font-size:12px;color:var(--txt2);margin-bottom:3px;text-align:center;">{v}</div>'
                f'<div style="height:6px;border-radius:3px;background:color-mix(in srgb,var(--accent) 12%,transparent);">'
                f'<div style="height:100%;width:{v}%;border-radius:3px;background:{c};"></div></div></td>')
    rows = ""
    for i, r in enumerate(ranked[:top], 1):
        f = r.get("factors", {})
        d = r.get("direction", "LONG")
        dcol = "buy" if d == "LONG" else "sell"
        nm = (r.get("name") or "")
        nm = (nm[:22] + "…") if len(nm) > 23 else nm
        pr = r.get("price")
        prc = f'${pr:,.2f}' if isinstance(pr, (int, float)) else ""
        rows += (
            f'<tr><td style="text-align:right;color:var(--muted);">{i}</td>'
            f'<td style="min-width:170px;"><b>{r["symbol"]}</b> '
            f'<span style="color:var(--muted);font-size:12px;">{nm}</span>'
            f'<div style="color:var(--muted);font-size:11px;">{prc}</div></td>'
            f'<td class="{dcol}" style="white-space:nowrap;">{r.get("action","")}</td>'
            f'<td style="text-align:center;"><b style="font-size:16px;">{r.get("rank_score","")}</b></td>'
            + fcell(f.get("quality")) + fcell(f.get("vreward")) + fcell(f.get("macrofit"))
            + fcell(f.get("liquidity"), "rkf-sm") + fcell(f.get("momentum"), "rkf-sm") + '</tr>'
        )
    th = ('<th style="text-align:center;font-size:11px;">{}</th>')
    thh = ('<th class="rkf-sm" style="text-align:center;font-size:11px;">{}</th>')
    return (
        '<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">' + _svg('target',14) + ' Top opportunities '
        '<span style="font-weight:400;color:var(--muted);font-size:12px;">— adaptive allocation rank: '
        'where limited capital should go first</span></div>'
        '<table class="tbl" style="margin-top:8px;width:100%;"><thead><tr>'
        '<th style="text-align:right;">#</th><th>Stock</th><th>Action</th>'
        '<th style="text-align:center;" title="0–100 composite allocation score">Rank</th>'
        + th.format("Quality") + th.format("Reward") + th.format("Macro&nbsp;fit")
        + thh.format("Liquidity") + thh.format("Momentum") + '</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:12px;margin:12px 0 0;line-height:1.6;">The <b>Rank</b> blends the five '
        'factors into one 0–100 allocation score, and capital (paper entries) goes to the highest first. '
        '<b>Quality</b> = conviction; <b>Reward</b> = volatility-adjusted reward:risk; <b>Macro fit</b> = how well the '
        'trade suits the current regime; <b>Liquidity</b> = how cleanly it trades; <b>Momentum</b> = trend strength. '
        'Greener bars are stronger. Educational; not advice.</p></div>'
    )


def _notrade_html(nt: dict | None) -> str:
    """Unified no-trade panel: the conditions that pause new entries, each ok / caution / block."""
    if not nt or not nt.get("checks"):
        return ""
    blocked = nt.get("block_new")
    head_col = "var(--sell)" if blocked else ("var(--warn)" if nt.get("cautions") else "var(--buy)")
    head_txt = ("Standing down — not opening new positions" if blocked
                else "Caution — trading with reservations" if nt.get("cautions")
                else "Clear to trade — no blocking conditions")
    icon = {"ok": f'<span style="color:var(--buy);">{_svg("dot",11)}</span>',
            "caution": f'<span style="color:var(--warn);">{_svg("dot",11)}</span>',
            "block": f'<span style="color:var(--sell);">{_svg("dot",11)}</span>'}
    rows = ""
    for c in nt.get("checks", []):
        rows += (f'<tr><td style="white-space:nowrap;">{icon.get(c["status"],"")} {c["name"]}</td>'
                 f'<td style="color:var(--txt2);font-size:13px;">{c["detail"]}</td></tr>')
    return (
        f'<div class="ovbox" style="border-left:4px solid {head_col};margin:0 0 16px;">'
        f'<div class="ovhead">{_svg("traffic",14)} No-trade check — <span style="color:{head_col};">{head_txt}</span></div>'
        f'<table class="tbl" style="margin-top:8px;"><tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">The bot sits on its hands when conditions '
        'are poor, even if a signal fires. A red status pauses <b>new</b> entries this run (open positions keep their '
        'stops/targets); amber means trade smaller / be selective. It never overrides the risk engine.</p></div>'
    )


def _nlp_html(scores: dict | None) -> str:
    """LLM structured news-read panel: named text scores per stock (LLM converts text → numbers)."""
    if not scores:
        return ""
    dims = [("guidance", "Guidance"), ("demand_strength", "Demand"), ("management_confidence", "Mgmt"),
            ("margin_pressure", "Margins"), ("regulatory_risk", "Reg risk"),
            ("balance_sheet_concern", "Balance sht"), ("earnings_quality_risk", "Earn qual")]

    def chip(v):
        v = int(v or 0)
        c = "var(--buy)" if v > 0 else "var(--sell)" if v < 0 else "var(--muted)"
        return f'<span style="color:{c};font-weight:600;">{v:+d}</span>'
    rows = ""
    for sym, d in sorted(scores.items(), key=lambda kv: -(kv[1].get("net") or 0)):
        net = d.get("net", 0)
        ncol = "var(--buy)" if net > 0.15 else "var(--sell)" if net < -0.15 else "var(--muted)"
        cells = "".join(f'<td style="text-align:center;">{chip(d.get(k,0))}</td>' for k, _ in dims)
        rows += (f'<tr><td><b>{sym}</b></td>'
                 f'<td style="text-align:center;color:{ncol};font-weight:700;">{net:+.2f}</td>'
                 f'{cells}'
                 f'<td style="color:var(--txt2);font-size:12px;">{d.get("note","")}</td></tr>')
    heads = "".join(f'<th style="text-align:center;" title="{lab}">{lab}</th>' for _, lab in dims)
    return (
        '<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead ai-ident">' + _svg('ai',14) + ' AI news read '
        '<span style="font-weight:400;color:var(--muted);font-size:12px;">— the LLM turns recent headlines into '
        'structured scores (−2…+2); it never decides the trade, it just feeds the meta-model</span></div>'
        '<table class="tbl" style="margin-top:8px;"><thead><tr><th>Symbol</th>'
        '<th style="text-align:center;" title="average across dimensions">Net</th>'
        f'{heads}<th>Note</th></tr></thead><tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:11px;margin:10px 0 0;">+ is favourable for the stock, − is a risk '
        'flag (for risk rows, − means the risk is elevated). Grounded only in the headlines shown on each card. '
        'A strongly opposing read makes the meta-model trim size. Educational; not advice.</p></div>'
    )


def _structured_html(items: list | None, top: int = 14) -> str:
    """Structured signal output: one row per actionable trade with the full signal contract."""
    if not items:
        return ""
    order = {"reject": 0, "delay": 1, "reduce": 2, "accept": 3}
    items = sorted(items, key=lambda s: (-(s.get("confidence") or 0)))
    ucol = {"low": "var(--buy)", "moderate": "var(--warn)", "high": "var(--sell)"}
    dcol = {"accept": "var(--buy)", "reduce": "var(--warn)", "delay": "var(--muted)", "reject": "var(--sell)"}
    rows = ""
    for s in items[:top]:
        rr = s.get("return_range") or {}
        up, dn = rr.get("upside_pct"), rr.get("downside_pct")
        rng = (f'<span class="buy">+{up:.0f}%</span> / <span class="sell">{dn:.0f}%</span>'
               if (up is not None and dn is not None) else "—")
        d = s.get("direction", "LONG")
        ddec = s.get("meta_decision", "accept")
        rows += (
            f'<tr><td><b>{s["symbol"]}</b></td>'
            f'<td class="{"buy" if d=="LONG" else "sell"}">{s.get("action","")}</td>'
            f'<td style="text-align:right;">{s.get("confidence","—")}</td>'
            f'<td style="text-align:right;">{rng}</td>'
            f'<td style="text-align:right;">{("%+.1f%%" % s["expected_value_pct"]) if s.get("expected_value_pct") is not None else "—"}</td>'
            f'<td style="text-align:right;">{s.get("expected_hold_days","—")}d</td>'
            f'<td style="text-align:right;">{s.get("risk_score","—")}</td>'
            f'<td style="text-align:right;color:{ucol.get(s.get("uncertainty_band"),"var(--muted)")};">{s.get("uncertainty","—")}</td>'
            f'<td style="text-align:center;color:{dcol.get(ddec,"var(--muted)")};font-weight:600;">{ddec}</td>'
            f'<td style="text-align:center;">{s.get("size_recommendation","—")}</td></tr>'
        )
    return (
        '<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">' + _svg('receipt',14) + ' Structured signals '
        '<span style="font-weight:400;color:var(--muted);font-size:12px;">— the full signal contract per trade: '
        'confidence, expected range, risk, uncertainty and the meta verdict</span></div>'
        '<table class="tbl" style="margin-top:8px;"><thead><tr>'
        '<th>Symbol</th><th>Action</th>'
        '<th style="text-align:right;" title="confidence score 0–100">Conf</th>'
        '<th style="text-align:right;" title="target upside / stop downside">Range</th>'
        '<th style="text-align:right;" title="probability-weighted expected return">EV</th>'
        '<th style="text-align:right;" title="expected holding period (sessions)">Hold</th>'
        '<th style="text-align:right;" title="risk score 0–100 (volatility + illiquidity)">Risk</th>'
        '<th style="text-align:right;" title="uncertainty 0–100 (disagreement / mixed macro / thin liquidity)">Unc</th>'
        '<th style="text-align:center;" title="meta-model verdict">Verdict</th>'
        '<th style="text-align:center;" title="recommended size">Size</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:11px;margin:10px 0 0;">EV and the range are rough, '
        'probability-weighted estimates (confidence as win-odds), not promises. High uncertainty is what makes the '
        'meta-model reduce or skip size — so a "reduce/Half" or "delay/Skip" verdict is the system being selective. '
        'Educational; not advice.</p></div>'
    )


def _macro_posture_html(mp: dict | None) -> str:
    """Macro regime → exposure panel: composite posture, exposure multiplier, and the drivers."""
    if not mp:
        return ""
    col = {"Risk-on": "var(--buy)", "Neutral": "var(--muted)", "Risk-off": "var(--sell)"}.get(mp.get("label"), "var(--muted)")
    em = mp.get("exposure_mult", 1.0)
    tilt = mp.get("cash_tilt_pct", 0)
    em_txt = (f"{em:.2f}× sizing" + (f" · ~{tilt}% more cash" if tilt else ""))
    thr = mp.get("entry_threshold")
    thr_txt = (f" · entry bar {thr}%" if thr else "")
    chips = ""
    for d in mp.get("drivers", []):
        dc = "var(--buy)" if d["score"] > 0.1 else "var(--sell)" if d["score"] < -0.1 else "var(--muted)"
        chips += (f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 9px;border-radius:999px;'
                  f'background:color-mix(in srgb,{dc} 14%,transparent);color:{dc};font-size:12px;" '
                  f'title="{d["read"]}">{d["name"]} {d["score"]:+.1f}</span>')
    # secondary regime tags (high-vol / recessionary / inflationary / liquidity-driven)
    tag_html = ""
    for t in mp.get("tags", []):
        tag_html += (f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 10px;border-radius:6px;'
                     f'background:color-mix(in srgb,#b5b5ba 16%,transparent);color:#b5b5ba;font-size:12px;font-weight:600;" '
                     f'title="{t["why"]}">{t["tag"]}</span>')
    tags_row = (f'<div style="margin:2px 0 8px;">{tag_html}</div>') if tag_html else ""
    # strategy bias (favoured vs caution)
    sb = mp.get("strategy_bias") or {}
    bias_html = ""
    if sb.get("favored") or sb.get("caution"):
        fav = " · ".join(sb.get("favored", []))
        cau = " · ".join(sb.get("caution", []))
        bias_html = (
            '<div style="font-size:12px;margin:6px 0 0;line-height:1.7;">'
            f'<div><span style="color:var(--buy);">▲ Favour:</span> <span style="color:var(--txt2);">{fav}</span></div>'
            f'<div><span style="color:var(--sell);">▼ Ease off:</span> <span style="color:var(--txt2);">{cau}</span></div>'
            '</div>')
    return (
        f'<div class="ovbox" style="border-left:4px solid {col};margin:0 0 16px;">'
        f'<div class="ovhead">{_svg("compass",14)} Macro regime &rarr; exposure: <span style="color:{col};">{mp.get("label")}</span> '
        f'<span style="font-weight:400;color:var(--muted);font-size:12px;">— composite {mp.get("score"):+.2f}, '
        f'<b style="color:{col};">{em_txt}</b>{thr_txt}</span></div>'
        f'{tags_row}'
        f'<p style="color:var(--txt2);font-size:13px;margin:6px 0 8px;">{mp.get("posture","")}</p>'
        f'<div>{chips}</div>'
        f'{bias_html}'
        '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">Macro sets <b>exposure</b> and <b>strategy emphasis</b>, '
        'not direction: it scales position size and tilts which strategies to lean on — it never directly buys or sells, '
        'and the rules-based risk engine always has the final say. Hover a chip/tag for the read behind it.</p></div>'
    )


def _timing_html(tp: dict | None) -> str:
    """Market-timing panel: O'Neil Follow-Through-Day state + distribution-day count on SPY/QQQ."""
    if not tp:
        return ""
    state = tp.get("state")
    col = {"confirmed": "var(--buy)", "neutral": "var(--muted)", "rally_attempt": "#b5b5ba",
           "pressure": "var(--sell)", "correction": "var(--sell)"}.get(state, "var(--muted)")
    em = tp.get("exposure_mult_blended", tp.get("exposure_mult", 1.0))
    # per-index chips (SPY / QQQ): FTD state + distribution-day count
    chips = ""
    for name, ix in (tp.get("indexes") or {}).items():
        dc = {"normal": "var(--muted)", "caution": "#b5b5ba", "pressure": "var(--sell)",
              "correction": "var(--sell)"}.get(ix.get("dd_risk"), "var(--muted)")
        dd = ix.get("dd", 0)
        off = ix.get("off_high")
        off_txt = f", {off:.1f}% off high" if isinstance(off, (int, float)) and off else ""
        chips += (f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 10px;border-radius:6px;'
                  f'background:color-mix(in srgb,{dc} 14%,transparent);color:{dc};font-size:12px;" '
                  f'title="{name}: FTD {ix.get("state")}, {dd} distribution days{off_txt}">'
                  f'{name} · {dd} DD{off_txt}</span>')
    ftd_note = ""
    if state == "confirmed" and tp.get("ftd_quality"):
        ftd_note = f' · FTD quality {tp.get("ftd_quality")}/100 (day {tp.get("ftd_day")})'
    elif tp.get("dd_total"):
        ftd_note = f' · {tp.get("dd_total")} distribution days across indexes'
    # Self-validation study (from timing_backtest, refreshed each post-close CI run): does the
    # signal actually precede the returns it claims? Show the verdict + a compact per-state table.
    study_html = ""
    study = tp.get("study") or {}
    if study.get("verdict"):
        hz = (study.get("horizons") or [5, 10, 20])
        hcol = hz[1] if len(hz) > 1 else hz[0]
        order = [("confirmed", "Confirmed FTD", "var(--buy)"), ("neutral", "Neutral", "var(--muted)"),
                 ("pressure", "Pressure", "#b5b5ba"), ("correction", "Correction", "var(--sell)")]
        rows = ""
        for key, lbl, rc in order:
            cell = (study.get("states") or {}).get(key, {}).get(str(hcol))
            if not cell or cell.get("mean_pct") is None:
                continue
            mp = cell["mean_pct"]
            mc = "var(--buy)" if mp > 0 else "var(--sell)" if mp < 0 else "var(--muted)"
            rows += (f'<tr><td style="color:{rc};padding:2px 10px 2px 0;">{lbl}</td>'
                     f'<td style="text-align:right;color:{mc};font-family:var(--mono,monospace);">{mp:+.2f}%</td>'
                     f'<td style="text-align:right;color:var(--muted);padding-left:10px;">{cell.get("hit_rate")}% up</td>'
                     f'<td style="text-align:right;color:var(--muted);padding-left:10px;">n={cell.get("n")}</td></tr>')
        gen = study.get("generated_at", "")
        study_html = (
            f'<p style="color:var(--txt2);font-size:12px;margin:10px 0 4px;"><b>Signal check:</b> {study.get("verdict")}</p>'
            f'<table style="font-size:12px;border-collapse:collapse;margin:2px 0 0;"><thead><tr>'
            f'<td style="color:var(--muted);padding-right:10px;">State</td>'
            f'<td style="color:var(--muted);text-align:right;">avg {hcol}d fwd</td>'
            f'<td style="color:var(--muted);text-align:right;padding-left:10px;">hit</td>'
            f'<td style="color:var(--muted);text-align:right;padding-left:10px;">samples</td></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            f'<p style="color:var(--muted);font-size:10px;margin:4px 0 0;">Walk-forward on SPY+QQQ history, no look-ahead · {gen}</p>')
    return (
        f'<div class="ovbox" style="border-left:4px solid {col};margin:0 0 16px;">'
        f'<div class="ovhead">{_svg("regime",14)} Market timing (O\'Neil): '
        f'<span style="color:{col};">{tp.get("label")}</span> '
        f'<span style="font-weight:400;color:var(--muted);font-size:12px;">— <b style="color:{col};">{em:.2f}× exposure</b>{ftd_note}</span></div>'
        f'<p style="color:var(--txt2);font-size:13px;margin:6px 0 8px;">{tp.get("note","")}</p>'
        f'<div>{chips}</div>'
        f'{study_html}'
        '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">A <b>Follow-Through Day</b> (high-volume '
        '&ge;1.25% index gain in the day 4-10 rally window) confirms a new uptrend; a cluster of '
        '<b>distribution days</b> (index down on rising volume) warns of institutional selling. This tilts '
        'exposure alongside the macro backdrop — it never buys or sells directly.</p></div>'
    )


def _book_risk_html(br: dict | None) -> str:
    """Open-book risk panel: portfolio heat (total open risk-to-stop) + parametric 95% VaR/CVaR on
    the positions held right now. Pulled from the finance-skills risk recipe (steps 3 + 4)."""
    if not br or not br.get("n"):
        return ""
    heat = br.get("heat_pct")
    cap = br.get("heat_cap_pct", 6.0)
    hc = "var(--sell)" if heat >= cap * 1.5 else "#b5b5ba" if heat >= cap else "var(--buy)"
    sig = int(round(br.get("sigma_assumed", 0.02) * 100))

    def tile(label, val, color="var(--txt)", sub=""):
        s = f'<div style="font-size:10px;color:var(--muted);margin-top:1px;">{sub}</div>' if sub else ""
        return (f'<div style="min-width:0;"><div style="font-size:11px;color:var(--muted);">{label}</div>'
                f'<div style="font-size:17px;font-weight:700;color:{color};font-family:var(--mono,monospace);">{val}</div>{s}</div>')
    clusters = br.get("correlated_clusters") or []
    eff = br.get("effective_bets")
    eff_tile = (tile("Effective bets", f'{eff}', "#b5b5ba" if clusters else "var(--buy)",
                     f'{br.get("n")} names, {len(clusters)} cluster{"s" if len(clusters) != 1 else ""}')
                if eff is not None else
                tile("Gross exposure", f'{br.get("gross_exposure_pct")}%', "var(--txt)", f'{br.get("n")} positions'))
    grid = "".join([
        tile("Portfolio heat", f'{heat:.1f}%', hc, f'total risk-to-stop · cap {cap:.0f}%'),
        eff_tile,
        tile("VaR 95% (1d)", f'${br.get("var95_usd"):,}', "var(--sell)", f'{br.get("var95_pct")}% of equity'),
        tile("CVaR 95%", f'${br.get("cvar95_usd"):,}', "var(--sell)", "mean worst 5% day"),
    ])
    cluster_html = ""
    if clusters:
        rows = "".join(f'<div style="font-size:12px;color:var(--txt2);margin:2px 0;">{_svg("octagon",11)} '
                       f'<b>{" + ".join(c)}</b> move together — count as one bet</div>' for c in clusters)
        cluster_html = (f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--glass-bd,rgba(255,255,255,.08));">'
                        f'{rows}</div>')
    return (
        '<div class="ovbox glass" style="border-left:4px solid var(--accent,#d0d0d3);margin:0 0 16px;">'
        f'<div class="ovhead">{_svg("octagon",14)} Open-book risk '
        f'<span style="font-weight:400;color:var(--muted);font-size:12px;">— the book you’re holding right now</span></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px 16px;margin-top:8px;">{grid}</div>'
        f'{cluster_html}'
        f'<p style="color:var(--muted);font-size:10px;margin:10px 0 0;"><b>Heat</b> sums every open position’s risk to '
        f'its stop — caps total risk-on, not just per-trade size. <b>Effective bets</b> collapses names that move '
        f'together into one. <b>VaR/CVaR</b> are a parametric estimate '
        f'(assumes ~{sig}%/name daily vol with a correlation haircut) of a rough day’s loss on the current book — '
        f'an estimate, not a promise.</p></div>'
    )


def _changelog_html(entries: list | None) -> str:
    """'What's new' view — the running log of features/additions, newest first, grouped by day."""
    if not entries:
        return ""
    cat_col = {"Signals": "var(--accent,#d0d0d3)", "Risk": "var(--sell,#f0596b)", "AI": "var(--ai,#8b5cf6)",
               "Timing": "var(--muted,#868c9a)", "Stats": "var(--muted,#868c9a)", "Analytics": "var(--buy,#22c98a)",
               "Validation": "var(--buy,#22c98a)", "Alerts": "var(--ai,#8b5cf6)", "Data": "var(--muted,#868c9a)",
               "UI": "var(--muted,#868c9a)"}
    from collections import OrderedDict
    by_day = OrderedDict()
    for e in entries:
        by_day.setdefault(e.get("date", ""), []).append(e)
    blocks = ""
    for day, items in by_day.items():
        rows = ""
        for e in items:
            c = cat_col.get(e.get("cat"), "var(--muted,#868c9a)")
            rows += (
                '<div style="display:flex;gap:12px;padding:12px 0;border-top:1px solid var(--glass-bd,rgba(255,255,255,.07));min-width:0;">'
                f'<span style="flex:0 0 auto;align-self:flex-start;font-size:10.5px;font-weight:700;color:{c};'
                f'background:color-mix(in srgb,{c} 14%,transparent);padding:3px 9px;border-radius:999px;">{e.get("cat","")}</span>'
                '<div style="min-width:0;">'
                f'<div style="font-weight:700;font-size:14px;color:var(--txt);">{e.get("title","")}</div>'
                f'<div style="font-size:12.5px;color:var(--txt2);margin-top:2px;line-height:1.5;">{e.get("note","")}</div>'
                '</div></div>')
        blocks += (f'<div style="margin-bottom:18px;"><div style="font-size:12px;font-weight:700;color:var(--muted);'
                   f'text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px;">{day}</div>{rows}</div>')
    return (
        f'<div class="sec-head"><span class="sh-ico">{_svg("news",15)}</span>'
        '<h2>What\'s new</h2><span class="sh-sub">recent changes &amp; additions to the bot</span></div>'
        f'<div class="card glass" style="padding:16px 18px;">{blocks}'
        '<p style="font-size:10.5px;color:var(--muted);margin:6px 0 0;">Curated highlights — the engine ships changes most days.</p></div>'
    )


def _agent_web_html(snap: dict) -> str:
    """Interactive 'ecosystem web' of the agents, organised on the PEER cycle (Plan → Execute →
    Express → Review, borrowed from agentUniverse), with the Review→Plan feedback loop drawn in.
    Nodes are clickable → a drawer shows that agent's FULL latest reasoning. Self-contained SVG +
    a tiny script; never raises."""
    import json as _json
    cfg = CONFIG
    signals = snap.get("signals") or []
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def bullets(items):
        return "".join(f'<li style="margin:3px 0;">{i}</li>' for i in items if i)

    # ---- gather full reasoning per agent (real data) ----
    cs = next((s.get("committee") for s in signals if s.get("committee")), None)
    who = next((s["symbol"] for s in signals if s.get("committee") is cs), "") if cs else ""
    if cs:
        roles = cs.get("roles") or {}
        if roles:
            rd = bullets([f'<b>{k.title()}</b> — {v.get("lean","?")}: {esc(v.get("note",""))}' for k, v in roles.items()])
        elif cs.get("models"):
            rd = bullets([f'<b>{esc(m)}</b>: {esc(v)}' for m, v in cs["models"].items()])
        else:
            rd = ""
        comm_detail = (f'<p>On <b>{who}</b> the verdict is <b>{cs.get("verdict")}</b> '
                       f'({cs.get("confidence")}% confidence).</p><ul>{rd}</ul>'
                       f'<p style="color:var(--muted);">{esc(cs.get("summary",""))}</p>')
    else:
        comm_detail = "<p>No committee verdict this build — it runs on the top actionable names.</p>"

    top = max(signals, key=lambda s: (s.get("conviction") or {}).get("score_pct") or 0, default=None)
    if top:
        checks = (top.get("conviction") or {}).get("checks") or []
        ic = {"pass": "✅", "warn": "🟡", "fail": "❌"}
        conv_detail = (f'<p><b>{top.get("symbol")}</b> scored <b>{(top.get("conviction") or {}).get("score_pct")}%</b> '
                       f'({(top.get("conviction") or {}).get("label")}). The checklist:</p><ul>'
                       + bullets([f'{ic.get(c.get("status"),"•")} {esc(c.get("label"))}' for c in checks[:14]]) + "</ul>")
    else:
        conv_detail = "<p>No actionable signals right now.</p>"

    tp = snap.get("timing") or {}
    tim_detail = (f'<p>Tape: <b>{tp.get("label","—")}</b>. {esc(tp.get("note",""))}</p>'
                  + "<ul>" + bullets([f'<b>{k}</b>: {v.get("state")} — {v.get("dd")} distribution days'
                                      for k, v in (tp.get("indexes") or {}).items()]) + "</ul>") if tp else "<p>No timing read.</p>"

    ss = _load_json_safe("setups_study.json") or {}
    if ss.get("setups"):
        H = ss.get("primary_horizon", 10)
        setu_detail = f'<p>Walk-forward edge vs baseline over {H} days:</p><ul>' + bullets([
            f'<b>{v.get("label")}</b>: {v.get("edge_pct"):+}% edge (n={v.get("n")})'
            for v in ss["setups"].values() if v.get("edge_pct") is not None]) + "</ul>"
    else:
        setu_detail = "<p>Validation runs after the close.</p>"

    br = snap.get("book_risk") or {}
    if br:
        cl = br.get("correlated_clusters") or []
        risk_detail = ("<ul>" + bullets([
            f'Portfolio heat {br.get("heat_pct")}% of {br.get("heat_cap_pct")}% cap',
            f'Effective bets {br.get("effective_bets", br.get("n"))} (of {br.get("n")} positions)',
            f'1-day VaR ${br.get("var95_usd"):,} ({br.get("var95_pct")}% of equity)',
            (" + ".join(cl[0]) + " move together") if cl else None]) + "</ul>")
    else:
        risk_detail = "<p>Flat book — no open risk.</p>"

    pa = snap.get("paper_acct") or {}
    exec_detail = (f'<ul>' + bullets([
        f'Equity ${pa.get("equity"):,}' if pa.get("equity") else None,
        f'Today {pa.get("day_pl_pct")}%' if pa.get("day_pl_pct") is not None else None,
        f'{pa.get("n_open", 0)} open positions']) + "</ul>") if pa else "<p>Paper account off.</p>"

    ar = _load_json_safe("analyst_report.json") or {}
    finds = ar.get("findings") or []
    _sc = ar.get("self_confidence") or {}
    if _sc.get("hit_rate") is not None:
        _conf = f'{_sc.get("hit_rate")}% hit rate ({_sc.get("improved")}/{_sc.get("graded")} acted-on buckets improved)'
    elif _sc.get("pending"):
        _conf = f'building — {_sc.get("pending")} action(s) awaiting enough post-change trades to grade'
    else:
        _conf = "building — no graded actions yet"
    an_detail = (f'<p>{ar.get("n_actions",0)} action items · self-check: {_conf} '
                 f'({ar.get("generated_at","")}):</p><ul>'
                 + bullets([f'<b>[{f.get("severity")}]</b> {esc(f.get("proposal",""))}' for f in finds[:6]]) + "</ul>") if ar else "<p>First report pending.</p>"

    lw = ((snap.get("learned") or {}).get("daily") or {}).get("weights") or {}
    retired = (snap.get("learned") or {}).get("retired") or []
    ups = sorted([(k, v) for k, v in lw.items() if v > 1.0], key=lambda x: -x[1])[:5]
    downs = sorted([(k, v) for k, v in lw.items() if 0 < v < 1.0], key=lambda x: x[1])[:5]
    attr_detail = ("<p>What the checks have earned from real outcomes:</p><ul>"
                   + bullets([f'⬆ {esc(k)} ({v}×)' for k, v in ups] + [f'⬇ {esc(k)} ({v}×)' for k, v in downs]
                             + ([f'🚫 retired: {esc(", ".join(retired))}'] if retired else [])) + "</ul>") if lw else "<p>Gathering trades.</p>"

    mem = _load_json_safe("analyst_memory.json") or {}
    mem_detail = f'<p>Graded {len(mem.get("ledger") or [])} past proposals against what actually happened, feeding the analyst\'s self-confidence.</p>'

    ch = _load_json_safe("changelog.json") or []
    express_detail = "<p>Turns the decision into what you see: the dashboard signal cards, the desk read, and the Telegram/phone/email alerts on fresh high-conviction names.</p>"
    build_detail = ("<p>The build layer — the Claude orchestrator + Explore/Plan/General sub-agents that research, write, verify and ship changes into the runtime.</p>"
                    + (f'<p style="color:var(--muted);">Latest shipped: {esc(ch[0].get("title",""))} ({ch[0].get("date","")}).</p>' if ch else ""))

    swarm = bool(getattr(cfg, "committee_swarm_enabled", False) and getattr(cfg, "openrouter_api_key", ""))
    # id, name, role, PEER column (0-3) or 4=build, enabled, detail
    NODES = [
        ("committee", "Committee", "swarm vote" if swarm else "4 roles + chair", 0, bool(getattr(cfg, "llm_enabled", False) and getattr(cfg, "committee_enabled", True)), comm_detail),
        ("timing", "Timing", "FTD / distribution", 0, bool(getattr(cfg, "timing_gate_enabled", True)), tim_detail),
        ("setups", "Setup edge", "walk-forward", 0, True, setu_detail),
        ("conviction", "Conviction", "scores the trade", 0, True, conv_detail),
        ("risk", "Risk engine", "heat · VaR · sizing", 1, bool(getattr(cfg, "risk_engine_enabled", True)), risk_detail),
        ("paper", "Paper book", "places + holds", 1, bool((snap.get("paper_acct") or {}).get("enabled")), exec_detail),
        ("express", "Dashboard + alerts", "tells you", 2, True, express_detail),
        ("analyst", "Analyst", "proposes changes", 3, bool(getattr(cfg, "llm_enabled", False)), an_detail),
        ("attribution", "Attribution", "re-weights checks", 3, bool(getattr(cfg, "adaptive_weights_enabled", True)), attr_detail),
        ("memory", "Memory", "grades itself", 3, True, mem_detail),
        ("build", "Build agents", "orchestrator + subagents", 4, True, build_detail),
    ]
    cols = {0: "PLAN", 1: "EXECUTE", 2: "EXPRESS", 3: "REVIEW", 4: "BUILD"}
    colx = {0: 90, 1: 300, 2: 500, 3: 710, 4: 400}
    V, G = "#8b5cf6", "#d0d0d3"
    # y layout per column
    byc = {}
    for n in NODES:
        byc.setdefault(n[3], []).append(n)
    node_svg, detail_map, first_id = [], {}, NODES[0][0]
    NW, NH = 150, 44
    for c, ns in byc.items():
        cx = colx[c]
        y0 = 70 if c != 4 else 470
        for i, (nid, name, role, _c, on, detail) in enumerate(ns):
            y = y0 + i * 66 if c != 4 else 470
            x = (cx - NW // 2) if c != 4 else (60 + len(node_svg) * 0)  # build laid out separately below
            detail_map[nid] = {"name": name, "role": role, "detail": detail}
    # place build nodes in a bottom row
    bx = 60
    positions = {}
    for c, ns in byc.items():
        if c == 4:
            continue
        cx = colx[c]
        for i, (nid, *_r) in enumerate(ns):
            positions[nid] = (cx - NW // 2, 78 + i * 66)
    for i, n in enumerate(byc.get(4, [])):
        positions[n[0]] = (70 + i * 200, 500)

    acc = {0: V, 1: "#4aa3ff", 2: G, 3: "#22c98a", 4: "#868c9a"}
    for (nid, name, role, c, on, detail) in NODES:
        x, y = positions[nid]
        col = acc[c]
        dotc = "#22c98a" if on else "#868c9a"
        node_svg.append(
            f'<g class="aunode" data-id="{nid}" onclick="auShow(\'{nid}\')" style="cursor:pointer;">'
            f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="10" fill="var(--card,#12141a)" '
            f'stroke="{col}" stroke-width="1.5"/>'
            f'<circle cx="{x+13}" cy="{y+14}" r="4" fill="{dotc}"/>'
            f'<text x="{x+24}" y="{y+18}" fill="var(--txt,#e9ecf2)" font-size="12.5" font-weight="700">{name}</text>'
            f'<text x="{x+12}" y="{y+34}" fill="var(--muted,#868c9a)" font-size="9.5">{role}</text></g>')
    # column headers
    heads = "".join(f'<text x="{colx[c]}" y="52" fill="{acc[c]}" font-size="12" font-weight="800" '
                    f'text-anchor="middle" letter-spacing="1">{cols[c]}</text>' for c in (0, 1, 2, 3))
    # stage flow arrows (Plan→Execute→Express→Review) + feedback + build→plan
    edges = (
        '<path class="auflow" d="M175 250 L285 250" stroke="#5b6270" stroke-width="1.6" marker-end="url(#aar)"/>'
        '<path class="auflow" d="M385 250 L485 160" stroke="#5b6270" stroke-width="1.6" marker-end="url(#aar)"/>'
        '<path class="auflow" d="M575 160 L695 200" stroke="#5b6270" stroke-width="1.6" marker-end="url(#aar)"/>'
        '<path class="auflowg" d="M710 320 C 500 430, 250 430, 90 300" fill="none" stroke="#22c98a" stroke-width="1.6" '
        'marker-end="url(#aarg)"/>'
        '<text x="400" y="428" fill="#22c98a" font-size="11" text-anchor="middle" font-weight="600">learns from outcomes → re-weights Plan</text>'
        '<path class="auflowa" d="M160 500 L120 340" fill="none" stroke="#d0d0d3" stroke-width="1.5" marker-end="url(#aara)"/>'
        '<text x="70" y="470" fill="#d0d0d3" font-size="10.5" font-weight="600">ships ↑</text>')
    # a live signal pulse that circulates the full PEER loop, + an amber pulse feeding up from Build
    _loop = "M120 250 L300 250 L500 160 L710 190 C 500 430, 250 430, 90 300 L120 250"
    pulses = (
        '<g class="aupulse">'
        f'<circle r="10" fill="#a78bfa" opacity="0.18"><animateMotion dur="7s" repeatCount="indefinite" path="{_loop}"/></circle>'
        f'<circle r="4.5" fill="#c4b5fd"><animateMotion dur="7s" repeatCount="indefinite" path="{_loop}"/>'
        '<animate attributeName="r" values="3.5;5.5;3.5" dur="1.1s" repeatCount="indefinite"/></circle>'
        '<circle r="3.5" fill="#d0d0d3" opacity="0.9"><animateMotion dur="2.6s" repeatCount="indefinite" '
        'path="M160 500 L120 340"/></circle>'
        '</g>')
    svg = (
        '<svg viewBox="0 0 900 560" xmlns="http://www.w3.org/2000/svg" role="img" '
        'font-family="Inter,system-ui,sans-serif" style="width:100%;height:auto;">'
        '<defs>'
        '<marker id="aar" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0L5 3L0 6Z" fill="#5b6270"/></marker>'
        '<marker id="aarg" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0L5 3L0 6Z" fill="#22c98a"/></marker>'
        '<marker id="aara" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0L5 3L0 6Z" fill="#d0d0d3"/></marker>'
        '</defs>' + edges + pulses + heads + "".join(node_svg) + '</svg>')
    drawer = ('<div id="au-detail" class="glass" style="border-radius:12px;padding:14px 16px;margin-top:10px;'
              'min-height:120px;font-size:13px;line-height:1.55;color:var(--txt2);"></div>')
    style = ('<style>.aunode rect{transition:filter .15s}.aunode:hover rect{filter:brightness(1.25)}'
             '.aunode.sel rect{stroke-width:2.5px}#au-detail ul{margin:6px 0;padding-left:18px}'
             '#au-detail .aud-h{font-size:15px;font-weight:800;color:var(--txt);margin-bottom:6px}'
             '#au-detail .aud-h span{font-weight:500;color:var(--muted);font-size:12px}'
             '@keyframes auflow{to{stroke-dashoffset:-22 } }'
             '.auflow{stroke-dasharray:6 5;animation:auflow .9s linear infinite}'
             '.auflowg{stroke-dasharray:5 4;animation:auflow 1.5s linear infinite}'
             '.auflowa{stroke-dasharray:4 3;animation:auflow 1.1s linear infinite}'
             '@media (prefers-reduced-motion:reduce){.auflow,.auflowg,.auflowa{animation:none }'
             '.aupulse{display:none } }</style>')
    script = ("<script>(function(){var AU=" + _json.dumps(detail_map)
              + ";window.auShow=function(id){var d=AU[id];if(!d)return;"
              "document.querySelectorAll('#page-agents .aunode').forEach(function(n){n.classList.toggle('sel',n.getAttribute('data-id')===id);});"
              "var el=document.getElementById('au-detail');"
              "el.innerHTML='<div class=\\'aud-h\\'>'+d.name+' <span>· '+d.role+'</span></div>'+d.detail;};"
              "auShow('" + first_id + "');})();</script>")
    return (
        f'<div class="sec-head"><span class="sh-ico">{_svg("ai",15)}</span><h2>Agent ecosystem</h2>'
        '<span class="sh-sub">the PEER cycle — Plan → Execute → Express → Review → (feedback). Tap a node for its full reasoning.</span></div>'
        f'<div class="card glass" style="padding:14px;overflow-x:auto;">{style}{svg}{drawer}{script}</div>'
    )


def _agent_universe_html(snap: dict) -> str:
    """Live 'Agent Universe' — every AI in the system as a card with its ACTUAL latest output.
    Two zones: RUNTIME (the trading brain, live from the snapshot) and BUILD (how it's made).
    Extensible: add an agent by appending one card. Never raises (defensive .get everywhere)."""
    def dot(on):
        c = "var(--buy)" if on else "var(--muted)"
        return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};flex:0 0 auto;"></span>'

    def card(icon, name, role, on, line, accent="var(--ai,#8b5cf6)"):
        return (
            f'<div class="glass" style="border-left:3px solid {accent};border-radius:12px;padding:13px 15px;min-width:0;">'
            f'<div style="display:flex;align-items:center;gap:8px;">{dot(on)}'
            f'<span style="color:{accent};display:inline-flex;">{_svg(icon,14)}</span>'
            f'<span style="font-weight:700;font-size:13.5px;color:var(--txt);">{name}</span>'
            f'<span style="margin-left:auto;font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;">{"live" if on else "off"}</span></div>'
            f'<div style="font-size:11px;color:var(--muted);margin:3px 0 6px;">{role}</div>'
            f'<div style="font-size:12.5px;color:var(--txt2);line-height:1.5;min-width:0;">{line}</div></div>')

    V, G = "var(--ai,#8b5cf6)", "var(--accent,#d0d0d3)"
    cfg = CONFIG
    signals = snap.get("signals") or []

    # --- Committee / chair (or swarm) ---
    swarm = bool(getattr(cfg, "committee_swarm_enabled", False) and getattr(cfg, "openrouter_api_key", ""))
    comm_on = bool(getattr(cfg, "llm_enabled", False) and getattr(cfg, "committee_enabled", True))
    cs = next((s.get("committee") for s in signals if s.get("committee")), None)
    if cs:
        who = next((s["symbol"] for s in signals if s.get("committee") is cs), "top pick")
        models = cs.get("models")
        detail = (f'{cs.get("n_models", len(models))} models: ' + ", ".join(sorted(set(models.values())))) if models else f'{cs.get("support",0)}/4 analysts agree'
        comm_line = f'Latest verdict on <b>{who}</b>: <b>{cs.get("verdict","?")}</b> ({detail}, {cs.get("confidence","?")}% conf).'
    else:
        comm_line = "No verdict yet this build — runs on the top actionable names."
    comm_role = ("Multi-model swarm — 3 models vote" if swarm else "4 analyst roles + a chair vote")

    # --- Conviction engine ---
    top = max(signals, key=lambda s: (s.get("conviction") or {}).get("score_pct") or 0, default=None)
    if top:
        conv = top.get("conviction") or {}
        conv_line = f'Top pick <b>{top.get("symbol")}</b> at <b>{conv.get("score_pct")}%</b> ({conv.get("label")}) across {conv.get("total", len(conv.get("checks") or []))} checks.'
    else:
        conv_line = "No actionable signals right now."

    # --- Attribution (self-learning weights) ---
    lw = ((snap.get("learned") or {}).get("daily") or {}).get("weights") or {}
    retired = (snap.get("learned") or {}).get("retired") or []
    up = sum(1 for m in lw.values() if isinstance(m, (int, float)) and m > 1.0)
    down = sum(1 for m in lw.values() if isinstance(m, (int, float)) and 0 < m < 1.0)
    if lw:
        _best = max(lw.items(), key=lambda kv: kv[1]) if lw else None
        attr_line = f'Up-weighted {up}, down-weighted {down}, retired {len(retired)} checks.' + (f' Strongest: “{_best[0]}” ({_best[1]}×).' if _best else "")
    else:
        attr_line = "Still gathering resolved trades before it adjusts weights."

    # --- Nightly analyst ---
    ar = _load_json_safe("analyst_report.json") or {}
    if ar:
        acts = ar.get("n_actions", 0)
        top_find = next((f for f in (ar.get("findings") or []) if f.get("severity") == "act"), None) or (ar.get("findings") or [None])[0]
        an_line = f'{acts} action item{"s" if acts != 1 else ""} in the last review ({ar.get("generated_at","")}).' + (f' Top: {top_find.get("proposal","")[:90]}' if top_find else "")
    else:
        an_line = "Runs after the close — first report pending."

    # --- Memory loop ---
    mem = _load_json_safe("analyst_memory.json") or {}
    ledger = mem.get("ledger") or []
    mem_line = (f'Graded {len(ledger)} of its own past proposals against outcomes.' if ledger
                else "Will start grading once the analyst has a history.")

    # --- Market timing ---
    tp = snap.get("timing") or {}
    tim_on = bool(getattr(cfg, "timing_gate_enabled", True))
    tim_line = (f'Tape read: <b>{tp.get("label")}</b>.' + (f' {tp.get("dd_total")} distribution days.' if tp.get("dd_total") else "")) if tp else "No index-timing read this build."

    # --- Risk engine ---
    br = snap.get("book_risk") or {}
    risk_on = bool(getattr(cfg, "risk_engine_enabled", True))
    if br:
        risk_line = f'Open book: heat {br.get("heat_pct")}% of {br.get("heat_cap_pct")}% cap, {br.get("effective_bets", br.get("n"))} effective bets, VaR ${br.get("var95_usd"):,}.'
    else:
        risk_line = "Flat book — no open risk to watch."

    # --- Setup validation ---
    ss = _load_json_safe("setups_study.json") or {}
    val_line = ss.get("verdict", "Walk-forward validation runs after the close.")[:150]

    runtime = "".join([
        card("bank" if not swarm else "ai", "Trade committee", comm_role, comm_on, comm_line, V),
        card("target", "Conviction engine", "scores every trade from all checks", True, conv_line, V),
        card("ai", "Attribution", "learns which checks predict wins", bool(getattr(cfg, "adaptive_weights_enabled", True)), attr_line, V),
        card("regime", "Market timing", "O'Neil FTD / distribution on SPY+QQQ", tim_on, tim_line, V),
        card("octagon", "Risk engine", "heat, VaR, drawdown, cooldown", risk_on, risk_line, V),
        card("chart", "Setup validation", "walk-forward edge of each setup", True, val_line, V),
        card("news", "Nightly analyst", "reviews the book, proposes changes", bool(getattr(cfg, "llm_enabled", False)), an_line, V),
        card("compass", "Memory loop", "grades the analyst's own past calls", True, mem_line, V),
    ])

    # --- Build AI ---
    cl = _load_json_safe("changelog.json") or []
    build_line = (f'{len(cl)} tracked additions; latest {cl[0].get("date","")} — {cl[0].get("title","")}.' if cl else "Ships changes to the runtime AI.")
    build = "".join([
        card("gear", "Orchestrator", "the main Claude that plans + coordinates", True, "Turns your prompts into verified, shipped changes. " + build_line, G),
        card("search", "Explore agent", "fast fan-out code/search", True, "Spawned to sweep the repo and skills libraries when a task needs broad reading.", G),
        card("compass", "Plan agent", "designs the implementation", True, "Spawned to architect bigger changes before code is written.", G),
        card("gear", "General agent", "deep research + build", True, "Spawned for heavy research (e.g. reading external skill repos) and multi-step builds.", G),
    ])

    return (
        f'<div class="sec-head"><span class="sh-ico">{_svg("ai",15)}</span><h2>Agent universe</h2>'
        '<span class="sh-sub">every AI in the system — and its latest output</span></div>'
        '<div style="margin:0 0 8px;font-size:12px;font-weight:800;letter-spacing:.5px;color:var(--ai,#8b5cf6);">◈ RUNTIME AI — THE TRADING BRAIN</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;">{runtime}</div>'
        '<div style="margin:20px 0 8px;font-size:12px;font-weight:800;letter-spacing:.5px;color:var(--accent,#d0d0d3);">◆ BUILD AI — HOW IT\'S MADE</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;">{build}</div>'
        '<p style="color:var(--muted);font-size:10.5px;margin:14px 0 0;">Green dot = live now · grey = off or awaiting data. Runtime agents read this build\'s snapshot; the build agents ship changes into it.</p>'
    )


def _perf_metrics():
    """Risk-adjusted performance of the traded book (metrics.py). Best-effort; None on too little data."""
    try:
        import metrics
        return metrics.from_file("track_record.json", min_n=20)
    except Exception:  # noqa: BLE001
        return None


def _performance_html(p: dict | None) -> str:
    """Institutional performance & risk panel: expectancy, SQN, profit factor, drawdown, tail risk —
    computed from how each thesis actually resolved (R-multiples). Pulled from finance-skills."""
    if not p:
        return ""
    sqn = p.get("sqn")
    # SQN quality bands (Van Tharp): <1.6 below-avg, 1.6-2.0 avg, 2.0-2.5 good, 2.5+ excellent
    if sqn is None:
        sc, sq = "var(--muted)", "—"
    elif sqn >= 2.5:
        sc, sq = "var(--buy)", "excellent"
    elif sqn >= 1.6:
        sc, sq = "var(--buy)", "tradeable"
    elif sqn >= 1.0:
        sc, sq = "#b5b5ba", "marginal"
    else:
        sc, sq = "var(--sell)", "no clear edge"
    exp = p.get("expectancy_r")
    ec = "var(--buy)" if (exp or 0) > 0 else "var(--sell)" if (exp or 0) < 0 else "var(--muted)"

    def tile(label, val, color="var(--txt)", sub=""):
        subhtml = f'<div style="font-size:10px;color:var(--muted);margin-top:1px;">{sub}</div>' if sub else ""
        return (f'<div style="min-width:0;"><div style="font-size:11px;color:var(--muted);">{label}</div>'
                f'<div style="font-size:17px;font-weight:700;color:{color};font-family:var(--mono,monospace);">{val}</div>'
                f'{subhtml}</div>')
    pf = p.get("profit_factor")
    grid = "".join([
        tile("Expectancy", f'{exp:+.3f}R' if exp is not None else "—", ec, "avg per trade"),
        tile("System Quality", f'{sqn:.2f}' if sqn is not None else "—", sc, sq),
        tile("Profit factor", f'{pf:.2f}' if pf is not None else "—",
             "var(--buy)" if (pf or 0) >= 1 else "var(--sell)"),
        tile("Payoff", f'{p.get("payoff")}×' if p.get("payoff") else "—", "var(--txt)", "avg win / avg loss"),
        tile("Sortino", f'{p.get("sortino")}' if p.get("sortino") is not None else "—"),
        tile("Max drawdown", f'{p.get("max_drawdown_r")}R', "var(--sell)", "peak-to-trough"),
        tile("VaR 95%", f'{p.get("var95_r")}R', "var(--sell)", "typical bad trade"),
        tile("CVaR 95%", f'{p.get("cvar95_r")}R', "var(--sell)", "mean worst 5%"),
    ])
    # Kelly advisory + Monte Carlo forward-risk read
    k = p.get("kelly_pct")
    hk = p.get("half_kelly_pct")
    if k is None:
        kelly_html = ""
    elif k <= 0:
        kelly_html = ('<div style="font-size:12px;color:var(--txt2);margin-top:12px;">'
                      f'<b style="color:var(--sell);">Kelly: 0%</b> — no measured edge, so the math says '
                      "don't size up; work on selection before adding risk.</div>")
    else:
        kelly_html = ('<div style="font-size:12px;color:var(--txt2);margin-top:12px;">'
                      f'<b style="color:var(--buy);">Kelly: risk {k}%</b> of equity per trade for full growth · '
                      f'<b>half-Kelly {hk}%</b> (the practical, lower-variance choice).</div>')
    mc = p.get("montecarlo") or {}
    mc_html = ""
    if mc:
        pl = mc.get("prob_losing_r_pct")
        plc = "var(--sell)" if (pl or 0) >= 40 else "#b5b5ba" if (pl or 0) >= 20 else "var(--buy)"
        mc_html = (
            '<div style="font-size:12px;color:var(--txt2);margin-top:8px;border-top:1px solid var(--glass-bd,rgba(255,255,255,.08));padding-top:8px;">'
            f'{_svg("regime",12)} <b>Monte Carlo</b> — {mc.get("sims"):,} bootstrapped paths of the next '
            f'{mc.get("horizon")} trades: typical drawdown <b>{mc.get("median_maxdd_r")}R</b>, bad-case (95th) '
            f'<b style="color:var(--sell);">{mc.get("p95_maxdd_r")}R</b>; ending equity median '
            f'<b>{mc.get("median_terminal_r")}R</b> (5th pct {mc.get("p05_terminal_r")}R); '
            f'chance of a net-losing run <b style="color:{plc};">{pl}%</b>.</div>')
    return (
        '<div class="ovbox glass" style="border-left:4px solid var(--accent,#d0d0d3);margin:0 0 16px;">'
        f'<div class="ovhead">{_svg("target",14)} Performance &amp; risk '
        f'<span style="font-weight:400;color:var(--muted);font-size:12px;">— {p.get("n")} resolved theses, '
        f'in R (units of risk)</span></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px 16px;margin-top:8px;">{grid}</div>'
        f'{kelly_html}{mc_html}'
        '<p style="color:var(--muted);font-size:10px;margin:10px 0 0;">Each thesis resolves as an R-multiple '
        '(target hit = +reward:risk, stop hit = −1R). <b>SQN</b> = expectancy ÷ std × √N (Van Tharp): the '
        'trade-native quality score. <b>Kelly</b> is the growth-optimal risk fraction from win rate + payoff. '
        '<b>Monte Carlo</b> bootstraps your own trade distribution forward. Not annualized — a high-throughput, '
        'many-concurrent book makes annualized Sharpe misleading.</p></div>'
    )


def _setup_study_html(st: dict | None) -> str:
    """Setup edge-validation panel: walk-forward hit rate + edge-over-baseline for the ported
    entry setups (Momentum Burst, Episodic Pivot), refreshed each post-close CI run."""
    if not st or not st.get("setups"):
        return ""
    H = st.get("primary_horizon", 10)
    base = (st.get("baseline") or {}).get(str(H)) or {}
    rows = ""
    for key in ("burst", "ep", "vcp", "pshort"):
        s = (st.get("setups") or {}).get(key)
        if not s:
            continue
        cell = (s.get("stats") or {}).get(str(H)) or {}
        mp, edge, nn = cell.get("mean_pct"), s.get("edge_pct"), s.get("n", 0)
        _dir = "↓ short" if s.get("direction") == "short" else ""
        _lbl = f'{s.get("label")} <span style="color:var(--muted);font-size:10px;">{_dir}</span>' if _dir else s.get("label")
        if mp is None:
            rows += (f'<tr><td style="padding:2px 10px 2px 0;color:var(--txt2);">{_lbl}</td>'
                     f'<td colspan="3" style="color:var(--muted);">gathering samples (n={nn})</td></tr>')
            continue
        ec = "var(--buy)" if (edge or 0) > 0 else "var(--sell)" if (edge or 0) < 0 else "var(--muted)"
        rows += (f'<tr><td style="padding:2px 10px 2px 0;color:var(--txt2);">{_lbl}</td>'
                 f'<td style="text-align:right;font-family:var(--mono,monospace);">{mp:+.2f}%</td>'
                 f'<td style="text-align:right;padding-left:10px;color:{ec};font-family:var(--mono,monospace);">'
                 f'{edge:+.2f}% edge</td>'
                 f'<td style="text-align:right;padding-left:10px;color:var(--muted);">hit {cell.get("hit_rate")}% · n={nn}</td></tr>')
    basetxt = f'{base.get("mean_pct"):+.2f}%' if base.get("mean_pct") is not None else "—"
    return (
        '<div class="ovbox" style="border-left:4px solid var(--accent,#d0d0d3);margin:0 0 16px;">'
        f'<div class="ovhead">{_svg("target",14)} Setup edge check '
        f'<span style="font-weight:400;color:var(--muted);font-size:12px;">— forward {H}-day return, walk-forward on '
        f'{st.get("names", 0)} liquid names</span></div>'
        f'<p style="color:var(--txt2);font-size:12px;margin:6px 0 6px;">{st.get("verdict","")}</p>'
        f'<table style="font-size:12px;border-collapse:collapse;"><tbody>{rows}'
        f'<tr><td style="padding-top:4px;color:var(--muted);">Baseline (all days)</td>'
        f'<td style="text-align:right;color:var(--muted);font-family:var(--mono,monospace);padding-top:4px;">{basetxt}</td>'
        f'<td colspan="2"></td></tr></tbody></table>'
        f'<p style="color:var(--muted);font-size:10px;margin:5px 0 0;">Edge = setup mean − baseline mean; positive means '
        f'the setup selects better-than-average windows. EP validated technical-only (no historical news) · {st.get("generated_at","")}</p></div>'
    )


def _macro_html(m: dict | None) -> str:
    if not m:
        return ""
    def cell(label, val):
        return (f'<div class="stat"><div class="l">{label}</div>'
                f'<div class="v" style="font-size:15px;">{val}</div></div>')
    cells = ""
    if m.get("vix") is not None:
        _vt = f' ({m["vix_trend"]})' if m.get("vix_trend") else ''
        cells += cell("VIX (fear gauge)", f'{m["vix"]}{_vt}')
    if m.get("dxy") is not None:
        cells += cell("US dollar index", f'{m["dxy"]}')
    if m.get("oil") is not None:
        cells += cell("WTI crude oil", f'${m["oil"]}')
    if m.get("hy_oas") is not None:
        _ht = f' ({m["hy_trend"]})' if m.get("hy_trend") else ''
        cells += cell("Credit spread (HY)", f'{m["hy_oas"]}%{_ht}')
    if m.get("y10") is not None:
        cells += cell("10-yr yield", f'{m["y10"]}%')
    if m.get("curve") is not None:
        cells += cell("Yield curve (10y-2y)", f'{m["curve"]:+.2f}')
    if m.get("cpi_yoy") is not None:
        cells += cell("Inflation (CPI)", f'{m["cpi_yoy"]}%')
    if m.get("unemployment") is not None:
        cells += cell("Unemployment", f'{m["unemployment"]}%')
    if m.get("fed_funds") is not None:
        cells += cell("Fed funds rate", f'{m["fed_funds"]}%')
    if m.get("nfci") is not None:
        _nt = f' ({m["nfci_trend"]})' if m.get("nfci_trend") else ''
        cells += cell("Financial conditions", f'{m["nfci"]:+.2f}{_nt}')
    if m.get("curve_3m") is not None:
        cells += cell("Yield curve (10y-3mo)", f'{m["curve_3m"]:+.2f}')
    if m.get("infl_expectations") is not None:
        cells += cell("Inflation expectations", f'{m["infl_expectations"]}%')
    return ('<div class="ovbox"><div class="ovhead">' + _svg('globe',14) + ' Macro backdrop '
            '<span style="font-weight:400;color:var(--muted);font-size:12px;">'
            '— the underlying readings feeding the posture above</span></div>'
            f'<div class="trackstats">{cells}</div></div>')


def _calendar_html(cal: dict | None) -> str:
    if not cal:
        return ""
    ew, ec = cal.get("earnings") or [], cal.get("econ") or []
    if not ew and not ec:
        return ""

    def chip(t):
        return f'<span class="chip mini">{t}</span> '
    e = "".join(chip(f'{x["symbol"]} · {"today" if x["days"] == 0 else str(x["days"]) + "d"}') for x in ew) \
        or '<span style="color:var(--muted);font-size:12px;">none in the next week</span>'
    mm = "".join(chip(f'{x["date"][5:]} · {x["name"]}') for x in ec) \
        or '<span style="color:var(--muted);font-size:12px;">none flagged</span>'
    return ('<div class="ovbox" style="margin-top:14px;"><div class="ovhead">' + _svg('calendar',14) + ' Event calendar '
            '<span style="font-weight:400;color:var(--muted);font-size:12px;">— event risk: avoid fresh entries right before these</span></div>'
            f'<div style="margin:8px 0 4px;"><div class="l" style="margin-bottom:5px;">Earnings this week</div>{e}</div>'
            f'<div style="margin:10px 0 2px;"><div class="l" style="margin-bottom:5px;">Key macro releases</div>{mm}</div></div>')


def _paper_spark(history: dict | None) -> str:
    """Tiny inline SVG equity curve from the paper account's portfolio history."""
    pts = (history or {}).get("points") or []
    vals = [p["v"] for p in pts if p.get("v")]
    if len(vals) < 2:
        return ""
    # drop implausible glitch prints (e.g. a near-zero equity tick) that spike the line
    import statistics as _st
    med = _st.median(vals)
    clean = [v for v in vals if med * 0.5 <= v <= med * 2]
    vals = clean if len(clean) >= 2 else vals
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    w, h = 320, 56
    step = w / (len(vals) - 1)
    coords = [(i * step, h - (v - lo) / rng * (h - 8) - 4) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = f"0,{h} " + line + f" {w},{h}"
    up = vals[-1] >= vals[0]
    col = "var(--buy)" if up else "var(--sell)"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'style="width:100%;max-width:340px;height:56px;display:block;">'
            f'<polygon points="{area}" fill="{col}" opacity="0.10"/>'
            f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def _esc_attr(s) -> str:
    """Escape a string for safe embedding inside a double-quoted HTML attribute (e.g. data-tiphtml)."""
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _waffle(pct, cls="win", cells=100):
    """A waffle heatmap: `cells` squares, first pct% filled with `cls`. Shares .an-waffle styling."""
    try:
        p = max(0, min(cells, int(round((pct or 0) * cells / 100.0))))
    except (TypeError, ValueError):
        p = 0
    return ('<div class="an-waffle">'
            + "".join(f'<div class="an-sq {cls if i < p else ""}"></div>' for i in range(cells))
            + "</div>")


def _wmini(pct, cls="win", cells=20):
    """Compact single-row mini waffle for inline / table-cell use (each cell ≈ 100/cells %)."""
    try:
        p = max(0, min(cells, int(round((pct or 0) * cells / 100.0))))
    except (TypeError, ValueError):
        p = 0
    return ('<span class="wmini">'
            + "".join(f'<i class="{cls if i < p else ""}"></i>' for i in range(cells))
            + "</span>")


def _callout(title, rows=None, note="", sub=""):
    """Build the dark hover-callout markup (returned RAW, not attr-escaped). `rows` is a list of
    (label, pct_or_None, value_str, tone) → mini win-rate bars. Wrap with _esc_attr for data-tiphtml."""
    parts = [f"<div class='cb-wrap'><div class='cb-h'>{title}</div>"]
    if sub:
        parts.append(f"<div class='cb-line'>{sub}</div>")
    for label, pct, val, tone in (rows or []):
        col = {"up": "var(--buy)", "dn": "var(--sell)", "mut": "var(--muted)"}.get(tone, "var(--buy)")
        w = 6 if pct is None else max(0, min(100, pct))
        parts.append(f"<div class='cb-row'><span class='cb-nm'>{label}</span>"
                     f"<span class='cb-bar'><i style='width:{w}%;background:{col};'></i></span>"
                     f"<span class='cb-v'>{val}</span></div>")
    if note:
        parts.append(f"<div class='cb-cm'>{note}</div>")
    parts.append("</div>")
    return "".join(parts)


def _tone_pct(pct, hi=50):
    """Green/red tone class for a percentage (win-rate style)."""
    if pct is None:
        return "mut"
    return "up" if pct >= hi else "dn"


def _control_html(snap: dict) -> str:
    """Control panel — tune the engine in plain terms + review your accept/reject decisions, then export
    a file the engine reads next build. All the interactivity is client-side JS; this is the shell."""
    return ('<div class="sec-eyebrow">Control</div>'
            f'<div class="sec-head"><span class="sh-ico">{_svg("scale", 15)}</span><h2>Control panel</h2>'
            '<span class="sh-sub">tune it &middot; accept / reject &middot; apply</span></div>'
            '<p style="color:var(--muted);font-size:13px;margin:2px 0 16px;max-width:680px;">Adjust the engine '
            'in plain terms and it updates what you see instantly. Hit <b>Accept</b>/<b>Reject</b> on any signal '
            'card. When ready, <b>Apply to engine</b> saves your choices to a file the bot reads on its next build.</p>'
            '<div class="ctrl-grid">'
            '<div class="ctrl-card"><h3>Settings</h3>'
            '<div class="ctrl-row"><div class="ctrl-lbl">Minimum win-probability to BUY <b id="cFloorV">52%</b></div>'
            '<input type="range" id="cFloor" min="40" max="75" step="1"></div>'
            '<div class="ctrl-hint">Higher = fewer, higher-quality trades. ~55% targets a 70% win rate.</div>'
            '<div class="ctrl-row"><div class="ctrl-lbl">Max fresh BUYs per day <b id="cCapV">6</b></div>'
            '<input type="range" id="cCap" min="1" max="20" step="1"></div>'
            '<div class="ctrl-hint">Caps the actionable list to the top N by win-probability.</div>'
            '<label class="ctrl-tog"><input type="checkbox" id="cExt"> <span>No chasing — demote stretched entries (extension gate)</span></label>'
            '<label class="ctrl-tog"><input type="checkbox" id="cVol"> <span>Calm only — demote too-jumpy entries (volatility gate)</span></label>'
            '<label class="ctrl-tog"><input type="checkbox" id="cShorts"> <span>Allow shorts (off = long-only)</span></label>'
            '<div class="ctrl-preview" id="cPreview">—</div>'
            '</div>'
            '<div class="ctrl-card"><h3>Your decisions</h3>'
            '<div class="ctrl-dec-h">Accepted</div><div class="ctrl-chips" id="cAccepted"></div>'
            '<div class="ctrl-dec-h" style="margin-top:14px;">Rejected</div><div class="ctrl-chips" id="cRejected"></div>'
            '<button class="ctrl-clear" id="cClear">Clear all decisions</button>'
            '</div></div>'
            '<div class="ctrl-apply">'
            '<button class="ctrl-apply-btn" id="cApply">Apply to engine &darr;</button>'
            '<button class="ctrl-copy" id="cCopy">Copy</button>'
            '<span class="ctrl-apply-note">Downloads <code>dashboard_controls.json</code> — drop it in your '
            '<code>trading_bot</code> folder and push (or let the watcher sync). The preview above is instant; '
            'the engine applies it on the next build.</span></div>'
            '<div class="ctrl-reset"><button id="cReset">Reset settings to defaults</button></div>')


def _ticker_tape_html() -> str:
    """A full-width TradingView ticker tape — live scrolling quotes for indices + the core names.
    Reliable embed (no rotating video IDs like the old Live-TV panel had)."""
    import json as _jt
    cfg = {
        "symbols": [
            {"description": "S&P 500", "proxy": "AMEX:SPY"},
            {"description": "Nasdaq 100", "proxy": "NASDAQ:QQQ"},
            {"description": "Dow", "proxy": "AMEX:DIA"},
            {"description": "Russell 2000", "proxy": "AMEX:IWM"},
            {"description": "Apple", "proxy": "NASDAQ:AAPL"},
            {"description": "Microsoft", "proxy": "NASDAQ:MSFT"},
            {"description": "Nvidia", "proxy": "NASDAQ:NVDA"},
            {"description": "Amazon", "proxy": "NASDAQ:AMZN"},
            {"description": "Meta", "proxy": "NASDAQ:META"},
            {"description": "Tesla", "proxy": "NASDAQ:TSLA"},
            {"description": "Gold", "proxy": "TVC:GOLD"},
            {"description": "Bitcoin", "proxy": "CRYPTO:BTCUSD"},
        ],
        "showSymbolLogo": True, "isTransparent": True, "displayMode": "adaptive",
        "colorTheme": "dark", "locale": "en",
    }
    return ('<div class="ticker-band"><div class="tradingview-widget-container">'
            '<div class="tradingview-widget-container__widget"></div>'
            '<script type="text/javascript" '
            'src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>'
            + _jt.dumps(cfg) + '</script></div></div>')


def _brain_html(snap: dict) -> str:
    """The 'Engine brain' — a visual, animated flow of every stage and skill in the pipeline: how raw
    data becomes a scored, gated, sized signal, and how the nightly loop makes it learn. Mostly a fixed
    schematic, with a few live numbers wired in so it breathes."""
    reg = (snap.get("regime") or {}).get("label", "—")
    n_sig = sum(1 for s in (snap.get("signals") or []) if s.get("action") in ("BUY", "WATCH LONG", "HOLD LONG"))
    _mh = _load_json_safe("meta_history.json") or []
    auc = _mh[-1].get("auc_meta") if _mh else None
    auc_txt = f"AUC {auc}" if isinstance(auc, (int, float)) else "training"

    # (icon, name, one-liner, status, hover-detail)
    stages = [
        ("01", "Ingest", "live data in", [
            ("bank", "Market data", "Alpaca bars, quotes, movers", "live",
             "Settled daily bars for signals + live quotes/movers intraday. The factual backbone."),
            ("text", "News & catalysts", "multi-feed headlines + ideas", "live",
             "Alpaca/Benzinga + free feeds (Google/Yahoo). Fresh catalysts get pinned into the scan."),
            ("ai", "Grok · X pulse", "live social read + buzz scan", "live",
             "Grok searches X + web in real time: per-name sentiment, and a 'where's the buzz' discovery scan."),
            ("chat", "StockTwits trending", "retail crowd, most-active", "new",
             "Ported from fintwit-bot: the most-discussed retail names right now, seeded into the scan."),
            ("compass", "Macro / FRED", "regime, NFCI, curve", "live",
             "Risk-on/off regime, financial conditions (NFCI), yield curve — sets the whole-book posture."),
        ]),
        ("02", "Scan & setup", "find the shape", [
            ("brick", "7 strategies", "independent methods → confluence", "live",
             "Trend, momentum, breakout, mean-reversion, VCP, episodic pivot, parabolic — how many agree."),
            ("target", "Trade plan", "entry · ATR stop · target · R:R", "live",
             "ATR-based stops, structural targets, and dead-zone stop-widening to escape the R:R 3-4 trap."),
            ("scale", "Screens", "liquidity + earnings", "live",
             "Liquidity tiering and an earnings-window check before anything is taken seriously."),
        ]),
        ("03", "Score", "how good is it?", [
            ("receipt", "Conviction checks", "~27 checks → score", "legacy",
             "The hand-tuned checklist. Diagnostic showed it's near-random (AUC 0.23) — now superseded by the learned model."),
            ("nodes", "AI committee", "multi-model vote", "live",
             "An LLM swarm each votes accept/reduce/reject; the tally tilts size, graded by outcomes."),
        ]),
        ("04", "The learned brain", "what actually wins", [
            ("donut", "Meta-label P(win)", f"walk-forward model · {auc_txt}", "learn",
             "López de Prado meta-labeling: a model trained on your real outcomes predicts P(this long wins). "
             "OOS it ranks winners far better than the checklist (0.77 vs 0.23). Drives filter + size."),
            ("chartup", "Calibration audit", "is 70% really 70%?", "planned",
             "From Quant-toolkit: checks whether the model's stated probability matches reality before we trust it."),
        ]),
        ("05", "Gates", "discipline that keeps you alive", [
            ("octagon", "Shorts cut", "long-only in this regime", "gate",
             "Shorts won 7% at −4% expectancy — cut at the source. Downtrends show as AVOID, not tradeable."),
            ("octagon", "Extension gate", "no chasing", "gate",
             "Stretched-above-trend BUYs demoted to Watch — the edge is in bases/pullbacks, not rips."),
            ("octagon", "Volatility gate", "calm entries only", "gate",
             "Too-jumpy names demoted — calm names won 60% vs 15%."),
            ("octagon", "Hype-risk guard", "no coordinated pumps", "gate",
             "Grok flags pump/promo buzz; high-hype names are never seeded as trades."),
        ]),
        ("06", "Size & output", "act with discipline", [
            ("chartup", "Adaptive rank", "best setups first", "live",
             "Ranks by quality/reward/macro-fit/liquidity/momentum so limited capital goes to the best names."),
            ("donut", "Position sizing", "risk% × P(win) × Kelly × regime", "learn",
             "Base risk-per-trade, tilted by the meta-label P(win), fractional-Kelly, and macro exposure."),
            ("rows", "Signals out", f"{n_sig} live · BUY / Watch", "live",
             "The surfaced calls. Paper execution is advisory; you place any real trade yourself."),
        ]),
    ]
    stcls = {"live": "Live", "new": "New", "learn": "Learns", "gate": "Gate",
             "legacy": "Legacy", "planned": "Planned", "advisory": "Advisory"}
    # each skill card links to the page where it actually lives, so the diagram doubles as navigation
    GO = {
        "Market data": "markets", "News & catalysts": "news", "Grok · X pulse": "signals",
        "StockTwits trending": "signals", "Macro / FRED": "markets", "7 strategies": "analytics",
        "Trade plan": "signals", "Screens": "signals", "Conviction checks": "analytics",
        "AI committee": "agents", "Meta-label P(win)": "system", "Calibration audit": "system",
        "Shorts cut": "system", "Extension gate": "system", "Volatility gate": "system",
        "Hype-risk guard": "signals", "Adaptive rank": "portfolio", "Position sizing": "portfolio",
        "Signals out": "signals", "Premium selling": "premium", "Portfolio book": "portfolio",
        "All-weather": "allweather",
    }

    def node(ic, nm, desc, st, tip):
        go = GO.get(nm)
        click = f' onclick="window._showPage&&_showPage(\'{go}\')"' if go else ""
        arrow = '<span class="bn-arrow">&rarr;</span>' if go else ""
        return (f'<div class="bn bn-{st} hint{" bn-go" if go else ""}" data-tip="{_esc_attr(tip)}"{click}>'
                f'<div class="bn-top"><span class="bn-ic">{_svg(ic, 15)}</span>'
                f'<span class="bn-badge bn-b-{st}">{stcls.get(st, st)}</span></div>'
                f'<div class="bn-nm">{nm}{arrow}</div><div class="bn-desc">{desc}</div></div>')

    body = ""
    for num, name, sub, nodes in stages:
        cards = "".join(node(*n) for n in nodes)
        body += (f'<div class="bstage"><div class="bstage-h"><span class="bstage-num">{num}</span>'
                 f'<div><div class="bstage-nm">{name}</div><div class="bstage-sub">{sub}</div></div></div>'
                 f'<div class="bstage-nodes">{cards}</div></div>'
                 f'<div class="bconn"><span class="bflow"></span></div>')

    loop = ('<div class="bloop"><div class="bloop-h"><span class="bn-ic">' + _svg("chartup", 16)
            + '</span> Learns every night</div>'
            '<div class="bloop-body">Every call is logged and graded against real prices → the '
            '<b>nightly analyst</b> reviews what worked → the <b>meta-label model retrains</b> + the '
            '<b>system diagnostic</b> runs → conviction weights and P(win) update. The brain gets sharper '
            'as your track record grows — it tunes itself, you approve the direction.</div>'
            '<div class="bloop-steps"><span>Track record</span><span>&rarr;</span><span>Nightly analyst</span>'
            '<span>&rarr;</span><span>Meta-label retrain</span><span>&rarr;</span><span>Diagnostic</span>'
            '<span>&rarr;</span><span>Weights + P(win) update</span><span class="bloop-back">&#8630; back into the brain</span></div></div>')

    advisory = ('<div class="badv"><div class="bstage-nm" style="margin-bottom:10px;">Advisory branches '
                '<span class="bstage-sub" style="font-weight:400;">— read-outs, never auto-executed</span></div>'
                '<div class="bstage-nodes">'
                + node("scale", "Premium selling", "options income edge 0–100", "advisory",
                       "Theta Harvest-style: scores where selling options premium is favourable, with hard gates. You place any trade.")
                + node("donut", "Portfolio book", "hypothetical allocation", "advisory",
                       "Assembles every actionable signal into a risk-sized book so you can see exposure + concentration.")
                + node("candles", "All-weather", "defensive sleeve", "advisory",
                       "A classic all-weather allocation for the steadier, lower-drawdown part of a portfolio.")
                + '</div></div>')

    legend = ('<div class="blegend">'
              + "".join(f'<span class="bn-badge bn-b-{k}">{v}</span>' for k, v in
                        [("live", "Live"), ("new", "New"), ("learn", "Learns"), ("gate", "Gate"),
                         ("legacy", "Legacy"), ("planned", "Planned"), ("advisory", "Advisory")])
              + '</div>')

    return ('<div class="sec-eyebrow">Under the hood</div>'
            f'<div class="sec-head"><span class="sh-ico">{_svg("nodes", 15)}</span><h2>Engine brain</h2>'
            f'<span class="sh-sub">regime: {reg} &middot; the full pipeline, live</span></div>'
            '<p style="color:var(--muted);font-size:13px;margin:2px 0 16px;max-width:720px;">How a raw ticker '
            'becomes a scored, gated, sized signal — every skill in the stack, in the order it fires. Hover any '
            'block for what it does. Data flows top to bottom; the brain retrains on the loop at the end.</p>'
            + legend + '<div class="brain">' + body + '</div>' + loop + advisory)


def _premium_selling_html(snap: dict) -> str:
    """Premium-selling advisory (Theta Harvest-style) — a 0-100 read on WHERE selling options premium
    is favourable, with hard gates. Advisory only; never a trade instruction."""
    ps = snap.get("premium_selling") or {}
    names = ps.get("names") or []
    reg = ps.get("regime") or "—"
    _regcol = {"THE FINALS": "var(--buy)", "THE PLAYOFFS": "var(--txt)",
               "REGULAR SEASON": "var(--warn)", "OFF SEASON": "var(--sell)"}.get(reg, "var(--muted)")
    _regnote = {"THE FINALS": "widest edge — conditions favour selling premium",
                "THE PLAYOFFS": "normal harvesting conditions",
                "REGULAR SEASON": "elevated vol — defined-risk only",
                "OFF SEASON": "go to cash — realised vol can outrun implied"}.get(reg, "")
    head = ('<div class="sec-eyebrow">Options income</div>'
            f'<div class="sec-head"><span class="sh-ico">{_svg("scale", 15)}</span><h2>Premium selling</h2>'
            '<span class="sh-sub">where selling options premium pays &mdash; advisory only</span></div>'
            '<p style="color:var(--muted);font-size:13px;margin:2px 0 14px;max-width:720px;">Scores how '
            'favourable it is to <b>sell</b> options premium on each name (0&ndash;100) by fusing volatility '
            'signals &mdash; is implied vol genuinely rich vs realised (the core edge), expensive vs its own '
            'history, and does the term structure support harvesting? Hard gates veto dangerous setups. It '
            'measures conditions; it does <b>not</b> place trades or predict direction.</p>')
    if not names:
        return head + ('<div class="ovbox"><div class="gp-empty">' + _svg("scale", 15)
                       + ' No premium-selling scores yet. This needs live <b>options (implied-vol)</b> data '
                       'and populates on a live build &mdash; if your Alpaca plan includes options data. '
                       'Realised-vol and the scoring engine are ready; it lights up as implied vol flows.</div></div>')
    banner = (f'<div class="ps-regime" style="border-color:{_regcol};"><span class="ps-reg-l">Market regime</span>'
              f'<span class="ps-reg-v" style="color:{_regcol};">{reg}</span>'
              f'<span class="ps-reg-n">{_regnote}</span></div>')
    vcol = {"GO": "var(--buy)", "OK": "var(--warn)", "SKIP": "var(--muted)", "AVOID": "var(--sell)"}
    rows = ""
    for r in names:
        sc = r.get("score", 0)
        vd = r.get("verdict", "")
        vc = vcol.get(vd, "var(--muted)")
        _iv = r.get("iv"); _rv = r.get("realized"); _vrp = r.get("vrp"); _ivp = r.get("iv_pct")
        gate = (r.get("gates") or [None])[0]
        callout = _callout(f'{r["symbol"]} &middot; {vd}',
                           [("Edge score", sc, f"{sc:.0f}", "up" if sc >= 65 else ("dn" if sc < 50 else "mut"))],
                           note=(gate or f'IV {(_iv or 0)*100:.0f}% vs realised {(_rv or 0)*100:.0f}% &middot; VRP {(_vrp or 0)*100:+.0f} bp'))
        rows += (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}"><td><b>{r["symbol"]}</b></td>'
                 f'<td style="text-align:right;"><span class="wrcell">{_wmini(sc, "win")}{sc:.0f}</span></td>'
                 f'<td><span class="ps-verd" style="color:{vc};border-color:{vc};">{vd}</span></td>'
                 f'<td style="text-align:right;">{(_iv or 0)*100:.0f}%</td>'
                 f'<td style="text-align:right;">{(_rv or 0)*100:.0f}%</td>'
                 f'<td style="text-align:right;color:{"var(--buy)" if (_vrp or 0) > 0 else "var(--sell)"};">{(_vrp or 0)*100:+.0f}</td>'
                 f'<td style="text-align:right;">{(f"{_ivp:.0f}" if _ivp is not None else "&mdash;")}</td>'
                 f'<td style="color:var(--muted);font-size:12px;">{gate or ""}</td></tr>')
    table = ('<table class="trackrec"><thead><tr><th>Name</th><th style="text-align:right;">Edge</th><th>Verdict</th>'
             '<th style="text-align:right;">IV</th><th style="text-align:right;">Realised</th>'
             '<th style="text-align:right;" title="volatility risk premium (IV − realised), basis points">VRP</th>'
             '<th style="text-align:right;" title="IV vs its own 1-yr history">IV %ile</th><th>Gate</th></tr></thead>'
             f'<tbody>{rows}</tbody></table>')
    disc = ('<p style="color:var(--muted);font-size:11.5px;margin-top:12px;">Advisory only &mdash; not a '
            'recommendation to trade. Selling options carries risk that can exceed the premium collected. '
            'IV percentile bootstraps as history accrues. Verify in your broker; you place any trade yourself.</p>')
    return head + banner + table + disc


def _analytics_html(snap: dict) -> str:
    """Edge explorer — Anthropic-EconIndex-style left-rail tabs + waffle heatmaps of what actually
    wins: strategy hit-rates, market breadth, and conviction tiers, from real data."""
    reg = snap.get("regime") or {}
    tk = (snap.get("track") or {}).get("by_conviction") or {}
    setups = (_load_json_safe("setups_study.json") or {}).get("setups") or {}

    def waffle(pct, cls="win"):
        p = max(0, min(100, int(round(pct or 0))))
        return '<div class="an-waffle">' + "".join(
            f'<div class="an-sq {cls if i < p else ""}"></div>' for i in range(100)) + '</div>'

    cards = ""
    for k, d in sorted(setups.items(), key=lambda kv: -(((kv[1].get("stats") or {}).get("10") or {}).get("hit_rate") or 0)):
        st = (d.get("stats") or {}).get("10") or {}
        hit, n, edge = st.get("hit_rate"), st.get("n"), d.get("edge_pct")
        if hit is None:
            continue
        tone = "anup" if (edge or 0) > 0 else "andn"
        cards += (f'<div class="an-card"><div class="an-jt">{d.get("label", k)}</div>'
                  f'<div class="an-js">{hit:.0f}% hit &middot; edge <span class="{tone}">{(edge or 0):+.2f}%</span> &middot; {n} fires</div>'
                  f'{waffle(hit)}</div>')
    strat_view = (f'<div class="an-legend"><span><i class="win"></i>hit</span><span><i class="muted"></i>miss</span></div>'
                  f'<div class="an-grid">{cards}</div>') if cards else "<div class='an-empty'>Setup study not available yet.</div>"

    tiles = ""
    for s in (snap.get("sectors") or [])[:12]:
        pu = s.get("pct_up", 0)
        c = "anup" if pu >= 70 else ("andn" if pu < 50 else "anmut")
        tiles += f'<span class="an-sect">{s.get("sector","")} <b class="{c}">{pu}%</b></span>'
    br = reg.get("breadth")
    sector_view = (f'<div class="an-card"><div class="an-jt">Market breadth</div>'
                   f'<div class="an-js">{br}% of {reg.get("total","?")} scanned names above trend &middot; {reg.get("label","")}</div>'
                   f'{waffle(br)}</div><div class="an-sects">{tiles}</div>') if br is not None else "<div class='an-empty'>No breadth data.</div>"

    cc = ""
    for tier in ("High", "Medium", "Low"):
        d = tk.get(tier) or {}
        wr, n = d.get("win_rate"), d.get("n", 0)
        if not n:
            continue
        tone = "anup" if (wr or 0) >= 50 else "andn"
        cc += (f'<div class="an-card"><div class="an-jt">{tier} conviction</div>'
               f'<div class="an-js"><span class="{tone}">{wr:.0f}% win</span> &middot; {n} trades &middot; avg {d.get("avg_return",0):+.2f}%</div>'
               f'{waffle(wr)}</div>')
    conv_view = (f'<div class="an-legend"><span><i class="win"></i>win</span><span><i class="muted"></i>loss</span></div>'
                 f'<div class="an-grid">{cc}</div>') if cc else "<div class='an-empty'>No conviction data yet.</div>"

    report = ('<div class="an-report"><div class="an-report-ic">' + _svg("receipt", 20) + '</div>'
              '<div class="an-report-t">System diagnostic</div>'
              "<div class=\"an-report-d\">The nightly read on what's actually working — direction, R:R, meta-label AUC.</div>"
              '<div class="an-report-lk" onclick="window._showPage&&_showPage(\'system\')">View report &rarr;</div></div>')

    return (f'<div class="sec-eyebrow">Analytics</div>'
            f'<div class="sec-head"><span class="sh-ico">{_svg("brick", 15)}</span><h2>Edge explorer</h2>'
            '<span class="sh-sub">waffle heatmaps of what wins</span></div>'
            '<div class="an-lay"><nav class="an-rail">'
            '<button class="an-tab on" data-anview="strat">Strategy record</button>'
            '<button class="an-tab" data-anview="sector">Sector map</button>'
            '<button class="an-tab" data-anview="conv">Conviction</button>'
            f'{report}</nav><div class="an-main">'
            f'<div class="an-view on" data-anview="strat">{strat_view}</div>'
            f'<div class="an-view" data-anview="sector">{sector_view}</div>'
            f'<div class="an-view" data-anview="conv">{conv_view}</div>'
            '</div></div>')


def _system_health_html(diag: dict | None) -> str:
    """The nightly self-diagnostic surfaced on the dashboard: direction split, where targets get
    reached by R:R, the most/least predictive checks (longs-only), and the setup/timing verdicts."""
    if not diag or not diag.get("by_direction"):
        return ""
    bd = diag["by_direction"]

    def tile(lab, s):
        if not s:
            return ""
        exp = s.get("exp", 0)
        wr = round(s.get("wr", 0))
        tone = "buy" if exp > 0 else ("sell" if exp < 0 else "")
        callout = _callout(f"{lab} — resolved trades",
                           [("Win rate", wr, f"{wr}%", _tone_pct(wr))],
                           note=f"expectancy {exp:+.2f}% · n={s.get('n', 0)}")
        return (f'<div class="stat hint" data-tiphtml="{_esc_attr(callout)}"><div class="l">{lab}</div>'
                f'<div class="v {tone}">{wr}%</div>'
                f'<div style="margin:6px 0 4px;">{_wmini(wr, "win", 22)}</div>'
                f'<div class="sub">win &middot; exp {exp:+.2f}% &middot; n={s.get("n", 0)}</div></div>')
    dirg = (f'<div class="plangrid">{tile("Longs", bd.get("LONG"))}'
            f'{tile("Shorts", bd.get("SHORT"))}{tile("All", bd.get("ALL"))}</div>')

    rr_rows = ""
    for k, b in (diag.get("rr_buckets_longs") or {}).items():
        ret = b.get("avg_ret", 0)
        cls = "buy" if ret >= 0 else "sell"
        th = b.get("target_hit_pct", 0)
        callout = _callout(f"Reward:risk {k}",
                           [("Target hit", th, f"{th}%", _tone_pct(th, 45))],
                           note=f"{b.get('n', 0)} longs · avg {ret:+.2f}%")
        rr_rows += (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}"><td>R:R {k}</td>'
                    f'<td style="text-align:right;"><span class="wrcell">{_wmini(th, "win")}{th}%</span></td>'
                    f'<td style="text-align:right;color:var(--muted);">{b.get("n", 0)}</td>'
                    f'<td style="text-align:right;" class="{cls}">{ret:+.2f}%</td></tr>')
    rr_tbl = (f'<table class="tbl"><thead><tr><th>Reward:risk</th><th style="text-align:right;">Target hit</th>'
              f'<th style="text-align:right;">n</th><th style="text-align:right;">Avg</th></tr></thead>'
              f'<tbody>{rr_rows}</tbody></table>') if rr_rows else ""

    checks = diag.get("checks_longs_only") or {}
    items = sorted(checks.items(), key=lambda kv: kv[1].get("win_delta", 0))
    sel = (items[:3] + items[-3:][::-1]) if len(items) >= 6 else items
    ck_rows = ""
    for lab, c in sel:
        wd = c.get("win_delta", 0)
        cls = "buy" if wd > 0 else "sell"
        pw, npw = round(c.get("pass_wr", 0)), round(c.get("notpass_wr", 0))
        callout = _callout(f"{lab} — predictiveness",
                           [("Passed", pw, f"{pw}%", _tone_pct(pw)),
                            ("Not passed", npw, f"{npw}%", _tone_pct(npw))],
                           note=f"edge {wd:+.0f}pt {c.get('sig', '')}")
        ck_rows += (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}"><td>{lab}</td>'
                    f'<td style="text-align:right;">{pw}% vs {npw}%</td>'
                    f'<td style="text-align:right;" class="{cls}">{wd:+.0f}pt {c.get("sig", "")}</td></tr>')
    ck_tbl = (f'<table class="tbl"><thead><tr><th>Check (longs)</th><th style="text-align:right;">pass vs not</th>'
              f'<th style="text-align:right;">edge</th></tr></thead><tbody>{ck_rows}</tbody></table>') if ck_rows else ""

    verd = ""
    if diag.get("setups_verdict"):
        verd += f'<p class="shverd"><b>Setups:</b> {diag["setups_verdict"]}</p>'
    if diag.get("timing_verdict"):
        verd += f'<p class="shverd"><b>Timing:</b> {diag["timing_verdict"]}</p>'

    return ('<div class="sec-eyebrow">Self-check</div>'
            f'<div class="sec-head"><span class="sh-ico">{_svg("scale", 15)}</span><h2>System health</h2>'
            f'<span class="sh-sub">nightly diagnostic &middot; {diag.get("resolved", 0)} trades</span></div>'
            f'<div class="ovbox" style="padding:18px 20px;">{dirg}'
            '<div class="shgrid">'
            f'<div><div class="mdh">Where targets get reached (longs)</div>{rr_tbl}</div>'
            f'<div><div class="mdh">Most &amp; least predictive checks (longs)</div>{ck_tbl}</div>'
            f'</div>{verd}</div>')


def _metalabel_html(hist: list | None) -> str:
    """Meta-label model health: current out-of-sample AUC vs the conviction baseline, the top-20%
    win-rate lift, and a trend sparkline so you can watch it stabilise before it drives sizing."""
    if not hist:
        return ""
    last = hist[-1]
    am, ab = last.get("auc_meta"), last.get("auc_baseline")
    if not isinstance(am, (int, float)):
        return ""
    tone = "buy" if am >= 0.60 else ("warn" if am >= 0.55 else "")
    verdict = ("useful — beating the checklist" if am >= 0.60 else
               "marginal — keep watching" if am >= 0.55 else "no edge yet")
    xs = [h.get("auc_meta") for h in hist if isinstance(h.get("auc_meta"), (int, float))]
    spark = ""
    if len(xs) >= 2:
        mn, mx = min(xs), max(xs)
        rng = (mx - mn) or 1
        w, h = 150, 28
        pts = " ".join(f"{i / (len(xs) - 1) * w:.1f},{h - ((v - mn) / rng) * (h - 4) - 2:.1f}"
                       for i, v in enumerate(xs))
        spark = (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="vertical-align:middle;">'
                 f'<polyline points="{pts}" fill="none" stroke="var(--buy)" stroke-width="1.5"/></svg>')
    t20m, t20b = last.get("top20_meta_wr"), last.get("top20_base_wr")
    tiles = (f'<div class="stat"><div class="l">Meta-label AUC (out-of-sample)</div>'
             f'<div class="v {tone}">{am:.2f}</div>'
             f'<div class="sub">vs checklist baseline {ab:.2f} &middot; {verdict}</div></div>'
             f'<div class="stat"><div class="l">Top-20% win rate</div>'
             f'<div class="v buy">{t20m}%</div>'
             f'<div class="sub">model vs checklist {t20b}% &middot; n={last.get("n_oos","—")}</div></div>')
    return (f'<div class="ovbox" style="padding:18px 20px;margin-top:14px;">'
            f'<div class="mdh" style="margin-top:0;display:flex;align-items:center;gap:12px;">'
            f'Meta-label model — learned P(win) {spark}</div>'
            f'<div class="plangrid" style="margin-top:10px;">{tiles}</div>'
            '<p class="shverd">A learned P(win) that ranks winners better than the hand-tuned checklist. '
            'Wire it into sizing once the AUC holds above ~0.60 across a few months of data.</p></div>')


def _system_html(sysd: dict | None) -> str:
    """Live 'under the hood' status: every feed, AI layer, engine feature, execution toggle,
    scraper, alert channel and piece of infra — with an ON/OFF state read from the real config."""
    if not sysd:
        return ""
    groups = [
        (_svg("satellite", 14), "Data feeds", sysd.get("feeds"), "Where the numbers come from."),
        (_svg("ai", 14), "AI layers", sysd.get("ai"), "Anthropic LLM passes (need API credits)."),
        (_svg("gear", 14), "Signal engine", sysd.get("engine"), "How signals are generated + scored."),
        (_svg("target", 14), "Execution", sysd.get("execution"), "What actually places/manages paper orders (mostly opt-in)."),
        (_svg("search", 14), "Alt-data scrapers", sysd.get("scrapers"), "Extra inputs feeding conviction."),
        (_svg("bell", 14), "Alert delivery", sysd.get("delivery"), "How you get notified."),
        (_svg("brick", 14), "Infrastructure", sysd.get("infra"), "What hosts and rebuilds the site."),
    ]
    blocks = ""
    for gico, title, items, lead in groups:
        if not items:
            continue
        rows = ""
        for it in items:
            on = it.get("on")
            pill = (f'<span class="syspill on">{_svg("dot",10)} ON</span>' if on else f'<span class="syspill off">{_svg("dot",10)} off</span>')
            note = f'<span class="sysnote">{it.get("note","")}</span>' if it.get("note") else ""
            rows += (f'<div class="sysrow"><span class="sysname">{it["name"]}</span>{note}{pill}</div>')
        blocks += (f'<div class="ovbox" style="margin:0 0 14px;"><div class="ovhead">{gico} {title} '
                   f'<span style="font-weight:400;color:var(--muted);text-transform:none;font-size:12px;">— {lead}</span></div>'
                   f'<div class="sysgrid">{rows}</div></div>')
    intro = (f'<div class="sec-head"><span class="sh-ico">{_svg("gear",15)}</span><h2>System</h2>'
             '<span class="sh-sub">what\'s wired in and running right now</span></div>'
             '<p style="color:var(--muted);font-size:13px;margin:0 0 16px;">A live readout of every integration and '
             'feature, read straight from the current config each build. <b>ON</b> = active this run; <b>off</b> = '
             'not configured or deliberately disabled. No secrets shown.</p>')
    return intro + blocks


def _news_ideas_html(ideas: list[dict] | None) -> str:
    """Server-rendered 'News-driven ideas' block: the LLM's read of recent headlines."""
    if not ideas:
        return ""
    rows = ""
    for i in ideas:
        d = i.get("direction", "")
        tone = "buy" if d == "bullish" else "sell"
        arrow = "↑" if d == "bullish" else "↓"
        rows += (f'<div class="nidea {tone}"><div class="nidea-top"><b>{i.get("ticker","")}</b> '
                 f'<span class="{tone}" style="font-weight:700;">{arrow} {d}</span> '
                 f'<span class="nidea-conf">{i.get("confidence","")} confidence</span></div>'
                 f'<div class="nidea-why">{i.get("reason","")}</div>'
                 f'<div class="nidea-src">from: “{i.get("headline","")}”</div></div>')
    return ('<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">' + _svg('news',14) + ' News-driven ideas '
            '<span style="font-weight:400;color:var(--muted);text-transform:none;">— an AI read of recent '
            'headlines (sentiment, not the confluence engine)</span></div>'
            f'<div class="nideas">{rows}</div>'
            '<p style="color:var(--muted);font-size:11.5px;margin:9px 0 0;">Extracted by an LLM from recent '
            'news text — directional reads, not verified signals. Treat as leads to research. Tickers also in '
            "today's scan get a small conviction nudge.</p></div>")


def _altdata_html(snap: dict) -> str:
    """Aggregate + explain what the alt-data scrapers (SEC insiders, analyst ratings, StockTwits
    buzz) found across today's signals, and how each is meant to be read."""
    sigs = snap.get("signals", []) or []

    def _dir_tone(s):
        return "buy" if s.get("direction") != "SHORT" else "sell"

    def _rowlink(s, detail):
        nm = (s.get("name") or "")[:26]
        return (f'<tr><td><b>{s["symbol"]}</b> <span style="color:var(--muted);">{nm}</span></td>'
                f'<td><span class="{_dir_tone(s)}" style="font-weight:700;">{s.get("action","")}</span></td>'
                f'<td>{detail}</td></tr>')

    def _block(title, lead, header3, rows, empty):
        body = (f'<table class="trackrec" style="margin-top:8px;"><thead><tr><th>Ticker</th><th>Signal</th>'
                f'<th>{header3}</th></tr></thead><tbody>{rows}</tbody></table>') if rows else \
               f'<p style="color:var(--muted);font-size:13px;margin:6px 0 0;">{empty}</p>'
        return (f'<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">{title}</div>'
                f'<p style="color:var(--muted);font-size:12.5px;margin:6px 0 0;">{lead}</p>{body}</div>')

    # Insider (SEC Form 4)
    ins_rows = ""
    for s in sigs:
        i = s.get("insider") or {}
        if i.get("cluster_buy"):
            ins_rows += _rowlink(s, f'{_svg("bank",13)} {i["buys"]} open-market purchase(s), '
                                    f'{(i.get("buy_shares") or 0):,} shares (last {i.get("last_date","")})')
    ins = _block(_svg("bank",14) + " Insider buying (SEC Form 4)",
                 "Insiders (officers/directors) must file a Form 4 within 2 business days of trading their own "
                 "stock. Clusters of <b>open-market purchases</b> are a well-studied bullish tell — they're "
                 "spending real money, unlike option grants. We raise a long's conviction (and cut a short's) when "
                 "we see one.", "Finding", ins_rows,
                 "No insider open-market buy clusters across today's signals (most names have none on a given day).")

    # Analyst rating changes (Finnhub)
    rat_rows = ""
    for s in sigs:
        aa = (s.get("fundamentals") or {}).get("analyst_actions") or {}
        lt = aa.get("latest") or {}
        if lt.get("action") in ("up", "down"):
            arrow = _svg("arrow-up",13) if lt["action"] == "up" else _svg("arrow-dn",13)
            rat_rows += _rowlink(s, f'{arrow} {lt.get("firm","")}: {lt.get("from","") or "?"} &rarr; '
                                    f'{lt.get("to","")} ({lt.get("date","")}) · 60d net '
                                    f'{aa.get("n_up",0)}&uarr;/{aa.get("n_down",0)}&darr;')
    rat = _block(_svg("chart",14) + " Analyst rating changes (Finnhub)",
                 "Recent upgrades/downgrades and the firm behind them, over the last 60 days. A fresh upgrade is a "
                 "supportive catalyst for a long (headwind for a short); net downgrades lean the other way. It's one "
                 "input, not gospel — analysts lag as often as they lead.", "Latest action", rat_rows,
                 "No analyst rating changes in the last 60 days across today's signals (or Finnhub's free tier "
                 "didn't return them).")

    # Retail buzz (StockTwits)
    buzz_rows = ""
    for s in sigs:
        b = s.get("buzz") or {}
        if b.get("lean"):
            lean = {"bull": "Bullish", "bear": "Bearish", "mixed": "Mixed"}.get(b["lean"], b["lean"])
            buzz_rows += _rowlink(s, f'{_svg("chat",13)} {lean} — {b.get("sentiment_pct","?")}% bullish of tagged, '
                                     f'{b.get("n","?")} recent posts')
    buzz = _block(_svg("chat",14) + " Retail buzz (StockTwits)",
                  "Crowd chatter: how many recent posts mention the ticker and the Bull/Bear split among those the "
                  "author tagged. Treat it as a <b>contrarian-tinted attention gauge</b>, not a signal — it's noisy "
                  "and the crowd is often late. We weight it gently.", "Buzz", buzz_rows,
                  "No tickers cleared the buzz threshold (≥5 sentiment-tagged posts) across today's signals.")

    note = ('<p style="color:var(--muted);font-size:12px;margin-top:4px;">These feed the conviction checklist on '
            'each signal (open any card) and show as badges on the Cards/Terminal layouts. Data appears only on live '
            'runs, and is sparse by design — a quiet day here is normal, not a bug. Sources: SEC EDGAR, Finnhub, StockTwits.</p>')
    intro = (f'<div class="sec-head"><span class="sh-ico">{_svg("satellite",15)}</span><h2>Data signals</h2>'
             '<span class="sh-sub">what the scrapers found, and how to read it</span></div>')
    return intro + ins + rat + buzz + note


def _pairs_html(data: dict | None) -> str:
    """Render the pairs / mean-reversion diversifier tab: spread z-scores, signals, validation."""
    intro = (f'<div class="sec-head"><span class="sh-ico">{_svg("scale",15)}</span><h2>Pairs &amp; mean-reversion</h2>'
             '<span class="sh-sub">market-neutral spread bets; a diversifier for trendless tape</span></div>')
    explainer = (
        '<details class="ovbox" style="margin:0 0 16px;" open><summary style="cursor:pointer;font-weight:700;'
        'font-size:14px;list-style:none;">' + _svg('book',14) + ' What is pairs trading? <span style="font-weight:400;color:var(--muted);'
        'font-size:12px;">(tap to hide)</span></summary>'
        '<div style="margin-top:10px;font-size:13px;line-height:1.7;color:var(--txt2);">'
        '<p style="margin:0 0 8px;">Two stocks in the same business — say <b>Coca-Cola (KO)</b> and <b>Pepsi (PEP)</b> '
        '— normally move together. <b>Pairs trading</b> ignores whether the market goes up or down and instead bets that '
        'when the <i>gap</i> between such a pair stretches unusually wide, it will snap back to normal. You '
        '<b>buy the cheap one and short the expensive one</b> in equal dollar amounts, so you only profit from the gap '
        'closing — not from the market\'s direction. That\'s why it\'s a useful diversifier: it can work when the trend-following '
        'engine is struggling in a sideways market.</p>'
        '<p style="margin:0 0 6px;"><b>How to read the table:</b></p>'
        '<ul style="margin:0 0 8px;padding-left:18px;">'
        '<li><b>Spread z</b> — how far the gap is from normal, in standard deviations. <b>±2σ</b> = unusually stretched '
        '(actionable, marked ' + _svg('star-fill',12) + '). 0 = at its normal level.</li>'
        '<li><b>Signal</b> — <span class="buy">Long spread</span> = buy the first name, short the second; '
        '<span class="sell">Short spread</span> = the reverse; <b>Watch</b> = not stretched enough yet.</li>'
        '<li><b>β (beta)</b> — the hedge ratio: how many shares of the second name to trade per share of the first so the '
        'two legs cancel out market risk.</li>'
        '<li><b>Corr</b> — how tightly the two normally move together (closer to 1.0 = more reliable pair).</li>'
        '<li><b>Half-life</b> — roughly how many days the gap has historically taken to revert to normal.</li></ul>'
        '<p style="margin:0;"><b>How a trade works:</b> enter when the gap hits ±2σ, take it off as the gap reverts toward '
        '0, and stop out if it stretches past ±3σ (a sign the relationship may have broken). It\'s a paper-money '
        'diversifier here — educational, not investment advice.</p></div></details>')
    if not data or not data.get("pairs"):
        msg = (data or {}).get("note") or "No pairs qualified right now — none are stretched or stably mean-reverting."
        return (intro + explainer + '<div class="ovbox"><div class="ovhead">No active pairs</div>'
                f'<p style="color:var(--muted);font-size:13px;margin:8px 0 0;">{msg}</p>'
                '<p style="color:var(--muted);font-size:12px;margin:8px 0 0;">A pair only appears once its two legs '
                'are correlated, the spread is genuinely mean-reverting (sane half-life), and it has stretched toward '
                '±2σ. Enter at ±2σ, exit toward 0, stop beyond ±3σ.</p></div>')

    fit = data.get("regime_fit")
    fit_col = "var(--buy)" if fit else "var(--muted)"
    fit_txt = data.get("note") or ""
    banner = (f'<div class="ovbox" style="border-left:4px solid {fit_col};margin:0 0 16px;">'
              f'<div class="ovhead">Regime fit: <span style="color:{fit_col};">'
              f'{"favourable" if fit else "lower priority"}</span></div>'
              f'<p style="color:var(--txt2);font-size:13px;margin:6px 0 0;">{fit_txt}</p></div>')

    sig_style = {
        "LONG_SPREAD": ("buy", "Long spread"), "SHORT_SPREAD": ("sell", "Short spread"),
        "STOP": ("sell", "Stop / broken"), "WATCH": ("", "Watch"),
        "FLAT": ("", "At fair value"),
    }
    rows = ""
    hi = ' style="background:color-mix(in srgb,var(--accent) 7%,transparent);"'
    for p in data["pairs"]:
        cls, lab = sig_style.get(p["signal"], ("", p["signal"]))
        star = (f'<span style="color:var(--accent);">{_svg("star-fill",12)}</span> ') if p.get("actionable") else ""
        zc = "sell" if abs(p["z"]) >= p.get("stop_z", 3) else ("buy" if p.get("actionable") else "")
        tr_attr = hi if p.get("actionable") else ""
        rows += (
            f'<tr{tr_attr}>'
            f'<td><b>{p["a"]} / {p["b"]}</b><div style="color:var(--muted);font-size:11px;">'
            f'${p["price_a"]} vs ${p["price_b"]}</div></td>'
            f'<td class="{cls}">{star}{lab}</td>'
            f'<td style="text-align:right;" class="{zc}"><b>{p["z"]:+.2f}σ</b></td>'
            f'<td style="text-align:right;">{p["beta"]:.2f}</td>'
            f'<td style="text-align:right;">{p["corr"]:.2f}</td>'
            f'<td style="text-align:right;">{p["half_life"]:.0f}d</td>'
            f'<td style="color:var(--txt2);font-size:12px;">{p["note"]}</td></tr>'
        )
    table = (
        '<table class="tbl"><thead><tr>'
        '<th>Pair</th><th>Signal</th>'
        '<th style="text-align:right;" title="how many standard deviations the spread sits from its mean">Spread z</th>'
        '<th style="text-align:right;" title="hedge ratio: shares of B per share of A for a neutral spread">β</th>'
        '<th style="text-align:right;" title="return correlation of the two legs">Corr</th>'
        '<th style="text-align:right;" title="how fast the spread reverts to its mean">Half-life</th>'
        '<th>Read</th></tr></thead><tbody>' + rows + '</tbody></table>'
    )
    legend = ('<p style="color:var(--muted);font-size:12px;margin:12px 0 0;">'
              + _svg('star-fill',12) + ' = actionable now (|z| ≥ 2σ). Enter at ±2σ, exit as the spread reverts toward 0, '
              'stop if it stretches past ±3σ (the relationship may have broken). Dollar-neutral: trade β shares of '
              'the second leg per share of the first. Diversifier only — not a core directional position. '
              'Paper money / educational; not investment advice.</p>')
    return intro + explainer + banner + table + legend


def _risk_html(risk: dict | None) -> str:
    """Book-level risk-engine status banner: state, drawdown, day P&L, and any active limits."""
    if not risk or not risk.get("enabled"):
        return ""
    state = risk.get("state", "normal")
    palette = {
        "normal": ("var(--buy)", _svg("check", 15), "Normal", "Within all book-level risk limits."),
        "derisk": ("var(--warn)", _svg("warn", 15), "De-risking", "Drawdown elevated — new positions sized at half."),
        "halt":   ("var(--sell)", _svg("octagon", 15), "Halted", "A book-level limit was hit — no new positions this session."),
        "killed": ("var(--sell)", _svg("octagon", 15), "Kill switch", "Trading paused after repeated run failures."),
        "off":    ("var(--muted)", _svg("dot", 15), "Off", "Risk engine not evaluated this run."),
    }
    col, dot, lab, default_msg = palette.get(state, palette["normal"])
    dd = risk.get("drawdown_pct")
    dpl = risk.get("day_pl_pct")
    mpp = risk.get("max_position_pct")
    bits = []
    if dd is not None:
        bits.append(f'<span title="peak-to-now equity drawdown">Drawdown <b>{dd:.1f}%</b></span>')
    if dpl is not None:
        bits.append(f'<span title="today\'s P&amp;L vs prior close">Day P&amp;L <b>{dpl:+.1f}%</b></span>')
    if mpp is not None:
        bits.append(f'<span title="max single-position size">Concentration cap <b>{mpp:.0f}%</b></span>')
    metrics = ' &nbsp;·&nbsp; '.join(bits)
    msgs = (risk.get("reasons") or []) + (risk.get("warnings") or [])
    msg = " · ".join(msgs) if msgs else default_msg
    return (
        f'<div class="ovbox" style="border-left:4px solid {col};margin:0 0 16px;">'
        f'<div class="ovhead" style="display:flex;align-items:center;gap:8px;">'
        f'<span style="color:{col};display:inline-flex;">{dot}</span>'
        f'<span>Portfolio risk engine — <span style="color:{col};">{lab}</span></span></div>'
        f'<div style="font-size:12px;color:var(--muted);margin:6px 0 8px;">{metrics}</div>'
        f'<p style="color:var(--txt2);font-size:13px;margin:0;">{msg}</p></div>'
    )


def _paper_html(p: dict | None) -> str:
    """Server-rendered REAL paper-account block (opt-in). Honest, fills-based — distinct from
    the hypothetical tracker."""
    if not p:
        return ""
    intro = (f'<div class="sec-head"><span class="sh-ico">{_svg("receipt",15)}</span><h2>Paper account</h2>'
             '<span class="sh-sub">a real, fills-based record from an Alpaca paper account</span></div>')
    if not p.get("enabled"):
        return (intro + '<div class="ovbox"><div class="ovhead">Auto paper-trading is off.</div>'
                f'<p style="color:var(--muted);font-size:13px;margin:8px 0 0;">{p.get("reason","Set PAPER_TRADE=true to enable.")}'
                ' Once enabled, fresh High-conviction signals are auto-submitted to your <b>paper</b> account '
                'as bracket orders, and this page shows the real equity, P&amp;L and open positions.</p></div>')

    def tile(label, value, tone="", sub=""):
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v {tone}">{value}</div>{sub_html}</div>'

    dp = p.get("day_pl", 0) or 0
    tone = "buy" if dp > 0 else "sell" if dp < 0 else ""
    rz = p.get("realized") or {}
    wr = rz.get("win_rate")
    wr_v = "—" if wr is None else f"{wr:.0f}%"
    _wr_sub = ((f"{rz.get('n_trades', 0)} closed since {rz.get('since')}" if rz.get('since')
                else f"over {rz.get('n_trades', 0)} closed trades")
               + (f'<div style="margin-top:6px;">{_wmini(wr, "win", 22)}</div>' if isinstance(wr, (int, float)) else ""))
    tiles = (tile("Equity", f"${p.get('equity',0):,.0f}", "", "paper account")
             + tile("Day P&amp;L", f"${dp:+,.0f}", tone, f"{p.get('day_pl_pct',0):+.2f}%")
             + tile("Open positions", str(p.get("n_open", 0)), "", f"of {p.get('tracked_total', 0)} tracked")
             + tile("Realized win rate", wr_v, "", _wr_sub))
    spark = _paper_spark(p.get("history"))
    spark_html = f'<div style="margin:6px 0 16px;">{spark}</div>' if spark else ""

    # open positions table (live unrealized P&L)
    rows = ""
    for pos in p.get("positions", []):
        pl = pos.get("unrealized_pl", 0) or 0
        c = "buy" if pl > 0 else "sell" if pl < 0 else ""
        rows += (f'<tr><td><b>{pos["symbol"]}</b></td><td>{pos.get("side","") or ""}</td>'
                 f'<td style="text-align:right;">{pos.get("qty","")}</td>'
                 f'<td style="text-align:right;">${pos.get("avg_entry",0):,.2f}</td>'
                 f'<td style="text-align:right;">${pos.get("price",0):,.2f}</td>'
                 f'<td style="text-align:right;" class="{c}">${pl:+,.0f} ({pos.get("unrealized_plpc",0):+.1f}%)</td></tr>')
    postable = (f'<table class="trackrec"><thead><tr><th>Symbol</th><th>Side</th><th style="text-align:right;">Qty</th>'
                f'<th style="text-align:right;">Entry</th><th style="text-align:right;">Last</th>'
                f'<th style="text-align:right;">Unrealized</th></tr></thead><tbody>{rows}</tbody></table>'
                if rows else '<p style="color:var(--muted);font-size:13px;">No open paper positions right now.</p>')

    # --- realized performance: closed round-trips matched from actual fills ---
    def _pct(v, sign=True):
        if v is None:
            return "—"
        return f"{'+' if (sign and v > 0) else ''}{v}%"
    rstats = ""
    if rz.get("n_trades"):
        tp = rz.get("total_pl", 0) or 0
        _since = rz.get("since")
        rstats = ('<div class="kpis" style="margin-top:4px;">'
                  + tile("Closed trades", str(rz.get("n_trades", 0)), "",
                         (f"strategy record since {_since}" if _since else "matched round-trips (incl. any manual closes)"))
                  + tile("Avg return / trade", _pct(rz.get("avg_return_pct")),
                         "buy" if (rz.get("avg_return_pct") or 0) > 0 else "sell" if (rz.get("avg_return_pct") or 0) < 0 else "",
                         "per closed trade")
                  + tile("Avg win", _pct(rz.get("avg_win")), "buy", "winners only")
                  + tile("Avg loss", _pct(rz.get("avg_loss")), "sell", "losers only")
                  + tile("Realized P&amp;L", f"${tp:+,.0f}", "buy" if tp > 0 else "sell" if tp < 0 else "", "all closed trades")
                  + '</div>')
        trows = ""
        for t in rz.get("recent", []):
            ret = t.get("return_pct")
            c = "buy" if (ret or 0) > 0 else "sell" if (ret or 0) < 0 else ""
            when = (t.get("exit_time") or "")[:10]
            pl = t.get("pl", 0) or 0
            trows += (f'<tr><td><b>{t.get("symbol","")}</b></td><td>{t.get("direction","")}</td>'
                      f'<td style="text-align:right;">{t.get("qty","")}</td>'
                      f'<td style="text-align:right;">${t.get("entry_price",0):,.2f}</td>'
                      f'<td style="text-align:right;">${t.get("exit_price",0):,.2f}</td>'
                      f'<td style="text-align:right;" class="{c}">{_pct(ret)}</td>'
                      f'<td style="text-align:right;" class="{c}">${pl:+,.0f}</td>'
                      f'<td style="text-align:right;color:var(--muted);">{when}</td></tr>')
        rtable = (f'<table class="trackrec"><thead><tr><th>Symbol</th><th>Side</th>'
                  f'<th style="text-align:right;">Qty</th><th style="text-align:right;">Entry</th>'
                  f'<th style="text-align:right;">Exit</th><th style="text-align:right;">Return</th>'
                  f'<th style="text-align:right;">P&amp;L</th><th style="text-align:right;">Closed</th>'
                  f'</tr></thead><tbody>{trows}</tbody></table>')
        realized_html = ('<h3 style="font-size:15px;margin:18px 0 8px;">Realized performance '
                         '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
                         '— closed round-trips matched from real fills</span></h3>'
                         + rstats + rtable)
    else:
        realized_html = ('<h3 style="font-size:15px;margin:18px 0 8px;">Realized performance</h3>'
                         '<p style="color:var(--muted);font-size:13px;">No closed round-trips yet — '
                         'win rate and per-trade return appear here once paper positions are opened and exited.</p>')

    sub = []
    if p.get("submitted_now"):
        sub.append("Opened this run: " + ", ".join(f'{r["symbol"]} ({r["action"]}, {r["qty"]}sh)' for r in p["submitted_now"]))
    if not p.get("market_open"):
        sub.append("Market is closed — orders fire during market hours.")
    for n in p.get("notes", []):
        sub.append(n)
    subline = ('<p style="color:var(--muted);font-size:12px;margin-top:12px;">' + " · ".join(sub) + "</p>") if sub else ""

    return (intro
            + '<p style="color:var(--muted);font-size:13px;margin:0 0 14px;">These are <b>real fills</b> on a '
            'paper account — actual entry prices, slippage and timing — so they reflect how the calls truly play out, '
            'unlike the hypothetical tracker. Not investment advice; paper money only.</p>'
            + _risk_html(p.get("risk"))
            + f'<div class="kpis">{tiles}</div>{spark_html}'
            + '<h3 style="font-size:15px;margin:6px 0 8px;">Open positions</h3>' + postable
            + realized_html + subline)


def _attribution_html(rep: list[dict] | None) -> str:
    """Panel: which conviction checks actually predicted wins (pass vs fail win rate)."""
    intro = ('<h3 style="font-size:15px;margin:18px 0 8px;">Which checks earn their keep '
             '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
             '— win rate when each conviction check passed vs failed, on resolved calls</span></h3>')
    if not rep:
        return intro + ('<p style="color:var(--muted);font-size:13px;">Still accruing — per-check '
                        'win rates appear here once enough tracked calls resolve.</p>')
    rows = ""
    for r in rep:
        wp = "—" if r["win_rate_pass"] is None else f'{r["win_rate_pass"]:.0f}%'
        wf = "—" if r["win_rate_fail"] is None else f'{r["win_rate_fail"]:.0f}%'
        edge = r["edge"]
        ec = "buy" if (edge or 0) > 0 else "sell" if (edge or 0) < 0 else ""
        es = "—" if edge is None else f'{"+" if edge > 0 else ""}{edge:.0f} pts'
        rows += (f'<tr><td>{r["label"]}</td>'
                 f'<td style="text-align:right;">{r["n_pass"]}/{r["n_fail"]}</td>'
                 f'<td style="text-align:right;">{wp}</td><td style="text-align:right;">{wf}</td>'
                 f'<td style="text-align:right;" class="{ec}">{es}</td></tr>')
    return (intro + '<table class="trackrec"><thead><tr><th>Check</th>'
            '<th style="text-align:right;">Pass/Fail n</th><th style="text-align:right;">Win% pass</th>'
            '<th style="text-align:right;">Win% fail</th><th style="text-align:right;">Edge</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>')


def _analyst_html(analyst: dict | None) -> str:
    """Panel: the latest autonomous-analyst review — narrative + prioritised proposed changes."""
    if not analyst or not (analyst.get("findings") or analyst.get("narrative")):
        return ""
    sev = {"act": ("var(--sell)", "ACT"), "watch": ("var(--accent)", "WATCH"), "info": ("var(--muted)", "INFO")}
    intro = (f'<h3 style="font-size:15px;margin:22px 0 6px;"><span class="ai-ident">{_svg("ai",15)}</span> Autonomous analyst '
             '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
             f'— nightly self-review, {analyst.get("generated_at","")}. '
             f'{analyst.get("n_actions",0)} action item(s). It proposes; you approve.</span></h3>')
    narr = ""
    if analyst.get("narrative"):
        narr = (f'<div class="ai-box" style="margin:2px 0 12px;line-height:1.6;">'
                f'<span class="ai-h">{_svg("ai",13)} Read</span> {_md_inline(analyst["narrative"])}</div>')
    rows = ""
    for f in (analyst.get("findings") or [])[:14]:
        col, lab = sev.get(f.get("severity"), ("var(--muted)", "—"))
        rows += (f'<tr><td style="white-space:nowrap;color:{col};font-weight:700;font-size:11px;">{lab}</td>'
                 f'<td><b>{f.get("area","")}</b><div style="color:var(--muted);font-size:12px;margin-top:2px;">'
                 f'{f.get("observation","")}</div><div style="font-size:12.5px;margin-top:3px;">→ {f.get("proposal","")}</div></td></tr>')
    tbl = (f'<table class="trackrec"><tbody>{rows}</tbody></table>' if rows else
           '<p style="color:var(--muted);font-size:13px;">No action items — strategies within tolerance.</p>')
    return intro + narr + tbl


def _learned_html(learned: dict | None) -> str:
    """Panel: what the bot has learned, PER STRATEGY. Daily/swing and intraday learn from their own
    resolved books (they're different strategies), so each gets its own section: the per-check edge
    (win rate when the check passed vs failed) and the weight multiplier the conviction engine now
    applies as a result. Up-weighted = has predicted winners; down-weighted = hasn't earned its keep."""
    if not learned:
        return ""
    min_n = learned.get("min_n", 12)
    intro = ('<h3 style="font-size:15px;margin:22px 0 6px;">What the bot has learned '
             '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
             '— each strategy adapts from its own resolved trades. A check is up- or down-weighted by '
             'how it has actually separated winners from losers.</span></h3>')
    blocks = ""
    for key, title, sub in (("daily", "Daily / swing signals", "daily bars · longer holds"),
                            ("intraday", "Intraday signals", "5-min bars · hours, not days"),
                            ("orb", "ORB day-trade signals", "opening-range breakout · flat by 15:45")):
        sect = learned.get(key) or {}
        rep = sect.get("report") or []
        weights = sect.get("weights") or {}
        nadj = len(weights)
        head = (f'<div class="sech" style="margin-top:16px;">{title} '
                f'<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:11px;">'
                f'— {sub} · {nadj} check{"" if nadj==1 else "s"} adapted</span></div>')
        if not rep:
            blocks += head + ('<p style="color:var(--muted);font-size:13px;margin:4px 0;">Still '
                              f'gathering — a check adapts once it has ≥{min_n} decided trades on both '
                              'sides. Until then this strategy runs on its default weights.</p>')
            continue
        body = ""
        for r in rep:
            w = weights.get(r["label"])
            if w is None:
                ws, wc = '<span style="color:var(--muted);">default</span>', ""
            else:
                ws = f'×{w:.2f}'
                wc = "buy" if w > 1 else "sell" if w < 1 else ""
            wp = "—" if r["win_rate_pass"] is None else f'{r["win_rate_pass"]:.0f}%'
            wf = "—" if r["win_rate_fail"] is None else f'{r["win_rate_fail"]:.0f}%'
            edge = r["edge"]
            ec = "buy" if (edge or 0) > 0 else "sell" if (edge or 0) < 0 else ""
            es = "—" if edge is None else f'{"+" if edge > 0 else ""}{edge:.0f} pts'
            body += (f'<tr><td>{r["label"]}</td>'
                     f'<td style="text-align:right;">{r["n_pass"]}/{r["n_fail"]}</td>'
                     f'<td style="text-align:right;">{wp}</td><td style="text-align:right;">{wf}</td>'
                     f'<td style="text-align:right;" class="{ec}">{es}</td>'
                     f'<td style="text-align:right;" class="{wc}">{ws}</td></tr>')
        blocks += (head + '<table class="trackrec"><thead><tr><th>Check</th>'
                   '<th style="text-align:right;">Pass/Fail n</th>'
                   '<th style="text-align:right;">Win% pass</th>'
                   '<th style="text-align:right;">Win% fail</th>'
                   '<th style="text-align:right;">Edge</th>'
                   '<th style="text-align:right;">Weight now</th>'
                   '</tr></thead><tbody>' + body + '</tbody></table>')
    return intro + blocks


def _track_html(track: dict | None) -> str:
    """Server-rendered track-record block (works without JS)."""
    if not track:
        return ""
    def stat(label, value, cls="", sub=""):
        sb = f'<div style="font-size:11px;color:var(--muted);margin-top:3px;line-height:1.35;">{sub}</div>' if sub else ""
        return (f'<div class="stat"><div class="l">{label}</div>'
                f'<div class="v {cls}">{value}</div>{sb}</div>')
    def _pct(v):
        return "—" if v is None else f"{'+' if v > 0 else ''}{v}%"
    wr = "—" if track["win_rate"] is None else f"{track['win_rate']}%"
    _res, _adv = track["resolved"], track["advised"]
    _wr_sub = (f"{_res} of {_adv} resolved"
               + (" · still maturing — winners take longer to reach target, so early reads skew low"
                  if (_adv and _res < _adv * 0.5) else ""))
    stats = (
        stat("Calls advised", track["advised"]) +
        stat("Resolved", track["resolved"]) +
        stat("Still open", track["open"]) +
        stat("Win rate", wr, "", _wr_sub) +
        stat("Expectancy", _pct(track.get("expectancy")), "buy" if (track.get("expectancy") or 0) > 0 else ("sell" if (track.get("expectancy") or 0) < 0 else "")) +
        stat("Avg win", _pct(track.get("avg_win")), "win") +
        stat("Avg loss", _pct(track.get("avg_loss")), "loss")
    )
    # per-direction / per-conviction breakdown (the live-performance read, as data accrues)
    def _brk(title, d, keys):
        cells = ""
        _dim = title.split()[-1].lower()
        for k in keys:
            g = (d or {}).get(k) or {}
            wr_v = g.get("win_rate")
            wrk = "—" if wr_v is None else f"{wr_v}%"
            n = g.get("n", 0)
            avg = g.get("avg_return")
            wbar = _wmini(wr_v, "win") if wr_v is not None else ""
            callout = _callout(f"{k} — {_dim}",
                               [("Win rate", wr_v, wrk, _tone_pct(wr_v))],
                               note=f"{n} resolved · avg return {_pct(avg)}")
            cells += (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}"><td>{k}</td>'
                      f'<td style="text-align:right;">{n}</td>'
                      f'<td style="text-align:right;"><span class="wrcell">{wbar}{wrk}</span></td>'
                      f'<td style="text-align:right;">{_pct(avg)}</td></tr>')
        return (f'<div class="sech" style="margin-top:14px;">{title}</div>'
                '<table class="trackrec"><thead><tr><th>'+title.split()[-1]+'</th>'
                '<th style="text-align:right;">Resolved</th><th style="text-align:right;">Win rate</th>'
                '<th style="text-align:right;">Avg return</th></tr></thead><tbody>'+cells+'</tbody></table>')
    breakdown = ""
    if track.get("resolved"):
        breakdown = (_brk("By direction", track.get("by_direction"), ["LONG", "SHORT"]) +
                     _brk("By conviction", track.get("by_conviction"), ["High", "Medium", "Low"]))
        _byreg = track.get("by_regime") or {}
        if _byreg:
            breakdown += _brk("By macro regime", _byreg, list(_byreg.keys()))
        tv = track.get("by_tv") or {}
        if (tv.get("agree", {}) or {}).get("n") or (tv.get("not_agree", {}) or {}).get("n"):
            def _tvrow(label, g):
                g = g or {}
                wr_v = g.get("win_rate")
                wrk = "—" if wr_v is None else f"{wr_v}%"
                wbar = _wmini(wr_v, "win") if wr_v is not None else ""
                callout = _callout(f"TradingView {label.lower()}",
                                   [("Win rate", wr_v, wrk, _tone_pct(wr_v))],
                                   note=f"{g.get('n',0)} resolved · avg {_pct(g.get('avg_return'))}")
                return (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}"><td>{label}</td>'
                        f'<td style="text-align:right;">{g.get("n",0)}</td>'
                        f'<td style="text-align:right;"><span class="wrcell">{wbar}{wrk}</span></td>'
                        f'<td style="text-align:right;">{_pct(g.get("avg_return"))}</td></tr>')
            breakdown += ('<div class="sech" style="margin-top:14px;">Does TradingView help? '
                          '<span style="text-transform:none;color:var(--muted);font-weight:400;">'
                          '— win rate when the TradingView cross-check agreed vs didn\'t</span></div>'
                          '<table class="trackrec"><thead><tr><th>TradingView</th>'
                          '<th style="text-align:right;">Resolved</th><th style="text-align:right;">Win rate</th>'
                          '<th style="text-align:right;">Avg return</th></tr></thead><tbody>'
                          + _tvrow("Agreed", tv.get("agree")) + _tvrow("Disagreed / mixed", tv.get("not_agree"))
                          + '</tbody></table>')
    # Your REAL journal record (from Obsidian via journal_sync) shown beside the engine's hypothetical
    # one — your actual trading vs what the bot flagged. Display only; never fed into the model.
    journal_rec = ""
    _jr = (_load_json_safe("journal_overrides.json") or {}).get("journal_record") or {}
    if _jr.get("n"):
        _jwr, _jav = _jr.get("win_rate"), _jr.get("avg_return")
        _jc = "var(--buy)" if (_jwr or 0) >= 50 else ("var(--warn)" if (_jwr or 0) >= 40 else "var(--sell)")
        _eng_wr = track.get("win_rate")
        _cmp = (f' &middot; engine (hypothetical): {_eng_wr}% over {track.get("resolved",0)}'
                if isinstance(_eng_wr, (int, float)) else "")
        journal_rec = (
            '<div class="jrec"><div class="jrec-h">' + _svg("receipt", 14)
            + ' Your journal — real trades you logged <span class="jrec-sub">(from Obsidian)</span></div>'
            f'<div class="jrec-row"><div class="jrec-stat"><div class="v" style="color:{_jc};">{_jwr}%</div><div class="k">win rate</div></div>'
            f'<div class="jrec-stat"><div class="v">{_jr.get("n")}</div><div class="k">closed &middot; {_jr.get("open",0)} open</div></div>'
            f'<div class="jrec-stat"><div class="v" style="color:{"var(--buy)" if (_jav or 0)>0 else "var(--sell)"};">{"+" if (_jav or 0)>0 else ""}{_jav}%</div><div class="k">avg return</div></div>'
            f'<div class="jrec-stat"><div class="v" style="color:var(--buy);">+{_jr.get("avg_win")}%</div><div class="k">avg win</div></div>'
            f'<div class="jrec-stat"><div class="v" style="color:var(--sell);">{_jr.get("avg_loss")}%</div><div class="k">avg loss</div></div></div>'
            f'<div class="jrec-note">Your own record{_cmp}. Log trades in Obsidian; the watcher syncs them here. Not fed into the model.</div></div>')

    # win-rate rate-hero: the headline number beside a full waffle of resolved outcomes
    wr_hero = ""
    _wrv = track.get("win_rate")
    if _wrv is not None and track.get("resolved"):
        _wcol = "var(--buy)" if _wrv >= 50 else ("var(--warn)" if _wrv >= 40 else "var(--sell)")
        wr_hero = (f'<div class="rate-hero"><div><div class="rh-v" style="color:{_wcol};">{_wrv}%</div>'
                   f'<div class="rh-k">win rate · {track.get("resolved",0)} resolved calls</div></div>'
                   f'{_waffle(_wrv, "win")}</div>')
    rows = ""
    icon = {"win": f'<span class="win" style="color:var(--buy);">{_svg("check",13)} hit target</span>',
            "loss": f'<span class="loss" style="color:var(--sell);">{_svg("x",13)} hit stop</span>',
            "expired": f'<span class="exp" style="color:var(--muted);">{_svg("clock",13)} expired</span>'}
    _outcome_word = {"win": "Hit target", "loss": "Hit stop", "expired": "Expired"}
    for t in track.get("recent", []):
        ret = t.get("return_pct")
        ret_s = "—" if ret is None else f"{'+' if ret > 0 else ''}{ret}%"
        _st = t.get("status")
        _tone = "up" if (ret or 0) > 0 else ("dn" if (ret or 0) < 0 else "mut")
        _rmag = min(100, abs(ret) * 5) if isinstance(ret, (int, float)) else 6
        callout = _callout(
            f"{t.get('symbol','')} — {_outcome_word.get(_st, _st or 'open')}",
            [("Return", _rmag, ret_s, _tone)],
            note=f"Advised {t.get('advised_date','')} · held {t.get('days_held','—')}d")
        rows += (f'<tr class="hint" data-tiphtml="{_esc_attr(callout)}"><td>{t.get("symbol","")}</td>'
                 f"<td>{t.get('advised_date','')}</td>"
                 f"<td>{icon.get(_st, _st or '')}</td>"
                 f"<td>{ret_s}</td><td>{t.get('days_held','—')}d</td></tr>")
    if rows:
        table = (f'<table class="trackrec"><tr><th>Stock</th><th>Advised</th><th>Outcome</th>'
                 f'<th>Return</th><th>Held</th></tr>{rows}</table>')
    elif track.get("advised"):
        table = (f'<p style="color:var(--muted);font-size:13px;">{track["advised"]} call'
                 f'{"s" if track["advised"] != 1 else ""} logged and still open — results appear here as each '
                 'hits its target or stop (usually within a few days).</p>')
    else:
        table = ('<p style="color:var(--muted);font-size:13px;">Building your track record — every BUY the screen '
                 'flags gets logged as it runs each weekday, and the first resolved results land within a day or two '
                 'as trades play out. Nothing to show yet.</p>')
    return f"""
  <div class="track">
    <div class="sec-head"><span class="sh-ico">{_svg('chart',15)}</span><h2>Track record</h2><span class="sh-sub">how past BUY calls have done</span></div>
    <p style="color:var(--muted);font-size:13px;margin:2px 0 0;">Every BUY the tool flags is logged, then
    checked against real prices: did it reach its target ({_svg('check',12)}) or hit its stop first ({_svg('x',12)})? This builds up
    over time into an honest read on how reliable the calls are. It's a hypothetical record — no fees or
    slippage — so treat it as a rough guide, not a brokerage statement.</p>
    <div class="trackstats">{stats}</div>
    {journal_rec}
    {wr_hero}
    {table}
    {breakdown}
  </div>"""


def _orb_backtest_cached(syms, cfg, _data, _orb, _inplay, _replace, cand_by):
    """Deep cost-modeled ORB backtest across windows, run AT MOST once per ET day and only when the
    market is closed (so it never burdens the 10-min market-hours builds). Cached to a json file;
    intraday builds reuse the latest. Returns the aggregate payload (or None). Fail-silent."""
    import json as _json
    import os as _os
    import pandas as _pd
    path = _os.getenv("ORB_BACKTEST_FILE", "orb_backtest.json")
    try:
        et = _pd.Timestamp.now(tz="America/New_York")
        today_et = str(et.date())
        mins = et.hour * 60 + et.minute
        market_open = et.weekday() < 5 and (9 * 60 + 30) <= mins < 16 * 60
    except Exception:  # noqa: BLE001
        today_et, market_open = "", True
    cache = None
    try:
        with open(path) as f:
            cache = _json.load(f)
    except Exception:  # noqa: BLE001
        cache = None
    # fresh today, or market open (never recompute intraday) -> reuse what we have
    if cache and (cache.get("date") == today_et or market_open):
        return cache
    # recompute (market closed + stale): deep fetch just for the backtest, off the critical path
    try:
        deep_lb = max(20, getattr(cfg, "orb_lookback_days", 45))
        bcfg = _replace(cfg, timeframe="5Min", lookback_days=deep_lb)
        try:
            spy = _data.get_bars("SPY", bcfg)
        except Exception:  # noqa: BLE001
            spy = None
        bt_by_window = {w: [] for w in getattr(cfg, "orb_windows", (5, 15, 30))}
        n = 0
        for sym in syms:
            try:
                df = _data.get_bars(sym, bcfg)
            except Exception:  # noqa: BLE001
                continue
            if df is None or getattr(df, "empty", True):
                continue
            n += 1
            bw = _orb.best_window(sym, df, spy, cfg)
            for w, st in (bw.get("by_window") or {}).items():
                if w in bt_by_window:
                    bt_by_window[w].append(st)
        agg = _orb.aggregate_backtest(bt_by_window)
        out = {"date": today_et, "by_window": agg, "lookback_days": deep_lb, "names": n}
        try:
            with open(path, "w") as f:
                _json.dump(out, f, indent=2)
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:  # noqa: BLE001
        return cache


def _run_orb(rows, idea_map, nlp_scores, regime, cfg, live, today, equity):
    """Stocks-in-play ORB pass (its own bucket). Ranks in-play names, fetches 5-min bars, scores
    breakouts with the ORB learning weights, grades + logs to the ORB shadow tracker, and enforces
    the hard day-trading caps. Returns the snap['orb'] payload. Fail-silent — never breaks build."""
    out = {"enabled": bool(getattr(cfg, "orb_enabled", False)), "signals": [], "inplay": [],
           "risk_state": {}, "track": {}, "learned": None, "scanned": 0, "note": ""}
    if not (live and getattr(cfg, "orb_enabled", False)):
        out["note"] = "ORB runs on live market data only (needs intraday bars + quotes)."
        return out
    try:
        from dataclasses import replace as _replace
        import attribution as _attr
        import data as _data
        import inplay as _inplay
        import orb as _orb
        import orb_track as _orbt

        learned = {}
        if getattr(cfg, "adaptive_weights_enabled", True):
            try:
                learned = _attr.learned_weights(scope="orb", min_n=getattr(cfg, "adaptive_min_n", 12))
            except Exception:  # noqa: BLE001
                learned = {}

        def _tier(r):
            liq = r.get("liquidity")
            return (liq.get("tier") if isinstance(liq, dict) else liq) or ""

        def _catalyst(sym):
            if sym in (idea_map or {}):
                return 75.0
            n = (nlp_scores or {}).get(sym) or {}
            net = n.get("net")
            return float(min(90.0, 50.0 + abs(net) * 8.0)) if net is not None else 0.0

        cands = [{"symbol": r.get("symbol"), "rel_volume": r.get("rel_volume"),
                  "gap_pct": r.get("gap_pct"), "liquidity_tier": _tier(r),
                  "has_news": (r.get("symbol") in (idea_map or {})) or (r.get("symbol") in (nlp_scores or {})),
                  "catalyst_score": _catalyst(r.get("symbol"))}
                 for r in rows if r.get("symbol")]
        # Always include fresh IPOs (SpaceX-style) as stocks-in-play. They usually aren't in the
        # daily scan rows yet (too little history for the swing engine), but ORB only needs a
        # session or two of intraday bars — and an IPO is a textbook catalyst-driven mover.
        try:
            import scanner as _scn
            _have = {c["symbol"] for c in cands}
            for _sym in _scn.recent_listings(cfg):
                if _sym and _sym not in _have:
                    cands.append({"symbol": _sym, "rel_volume": None, "gap_pct": None,
                                  "liquidity_tier": "high", "has_news": True, "catalyst_score": 85.0})
                    _have.add(_sym)
        except Exception:  # noqa: BLE001
            pass
        # Pick which names to FETCH bars for: top N by a LOOSE pre-score (no min cut). The overnight
        # gap isn't known until we have bars, so gating on the full in-play score here would starve
        # the universe — instead take the most active/liquid names and let the real gap re-rank them
        # afterwards. This widens the backtest sample too.
        pre = _inplay.rank(cands, cfg, top=getattr(cfg, "orb_inplay_top", 40), min_score=0.0)
        out["inplay"] = pre
        syms = [s["symbol"] for s in pre]
        out["scanned"] = len(syms)
        if not syms:
            out["note"] = "No stocks in play (no liquid, active names today)."
            return out

        # SHALLOW fetch on EVERY build — the live signal + gap only need the last couple of sessions,
        # so this stays cheap even at 40 names and the 10-min market-hours build finishes comfortably.
        sig_lb = max(4, getattr(cfg, "orb_signal_lookback_days", 6))
        icfg = _replace(cfg, timeframe="5Min", lookback_days=sig_lb)
        try:
            spy_df = _data.get_bars("SPY", icfg)
        except Exception:  # noqa: BLE001
            spy_df = None
        try:
            quotes = _data.get_latest_quotes(syms, cfg)
        except Exception:  # noqa: BLE001
            quotes = {}
        ip_by = {s["symbol"]: s for s in pre}
        cand_by = {c["symbol"]: c for c in cands}
        bars_by, signals = {}, []
        for sym in syms:
            try:
                df = _data.get_bars(sym, icfg)
            except Exception:  # noqa: BLE001
                continue
            if df is None or getattr(df, "empty", True):
                continue
            bars_by[sym] = df
            g = _orb.gap_pct(df)             # real overnight gap (replaces the 0% placeholder)
            if g is not None and sym in cand_by:
                cand_by[sym]["gap_pct"] = g
            ip = ip_by.get(sym) or {}
            ctx = {"rel_volume": ip.get("rel_volume"), "catalyst_score": _catalyst(sym),
                   "liquidity_tier": ip.get("liquidity_tier"),
                   "spread_pct": (quotes.get(sym) or {}).get("spread_pct")}
            sig = _orb.build(sym, df, spy_df, cfg, equity=equity, ctx=ctx, learned=learned)
            if sig:
                sig["gap_pct"] = g
                sig["name"] = next((r.get("name") for r in rows if r.get("symbol") == sym), sym)
                signals.append(sig)

        # re-rank in-play now that real gaps are in, and tag each signal with its in-play score
        ranked2 = _inplay.rank([cand_by[s] for s in syms if s in cand_by], cfg,
                               top=getattr(cfg, "orb_inplay_top", 40), min_score=0.0)
        out["inplay"] = ranked2 or pre
        ip2 = {s["symbol"]: s for s in (ranked2 or pre)}
        for s in signals:
            s["in_play"] = (ip2.get(s["symbol"]) or {}).get("in_play")

        # DEEP backtest — heavy (deep history per name), so run it at most once per ET day and ONLY
        # when the market is closed (pre-market / after-close builds run on slow crons, never the
        # 10-min cadence). Cached to orb_backtest.json; intraday builds just reuse the latest.
        out["backtest"] = _orb_backtest_cached(syms, cfg, _data, _orb, _inplay, _replace, cand_by)

        track = _orbt.run(signals, bars_by, today, cfg, regime=regime)
        rs = track.get("risk_state", {})
        if rs.get("blocked"):
            for s in signals:
                if s.get("recommended_action") == "paper_trade":
                    s["risk_blocked"] = True
                    s["risk_block_reason"] = ("daily ORB trade cap reached" if rs.get("trades_capped")
                                              else "bucket halted after consecutive losses")
        try:
            out["learned"] = {"weights": learned, "report": _attr.report(scope="orb")}
        except Exception:  # noqa: BLE001
            out["learned"] = None
        signals.sort(key=lambda s: -(s.get("score") or 0))
        out["signals"] = signals
        out["risk_state"] = rs
        out["track"] = {k: v for k, v in track.items() if k != "risk_state"}
        out["spy_ok"] = bool(spy_df is not None and not getattr(spy_df, "empty", True))
        if not signals:
            out["note"] = f"{len(syms)} in-play names scanned; no qualifying breakouts in the 09:45–10:30 window yet."
    except Exception as e:  # noqa: BLE001
        out["note"] = f"ORB pass skipped: {e}"
    return out


def _orb_details(summary: str, body: str, open_: bool = False) -> str:
    """A collapsible section so the ORB tab leads with cards, with the tables tucked beneath."""
    if not body:
        return ""
    op = " open" if open_ else ""
    return (f'<details class="orb-sec"{op}><summary>{summary}</summary>'
            f'<div style="padding-top:4px;">{body}</div></details>')


def _orb_html(orb: dict | None) -> str:
    """Server-rendered ORB page: leads with the signal cards, then collapsible backtest + in-play."""
    head = ('<h2 style="margin-top:0;">Opening Range Breakout '
            '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
            '— stocks-in-play day-trade: break the first 15-min range, confirmed by VWAP + the '
            'market, scored 0–100, day-traded flat by 15:45. Its own strategy &amp; learning bucket.</span></h2>')
    if not orb or not orb.get("enabled"):
        return head + '<p style="color:var(--muted);font-size:13px;">ORB is disabled (set ORB_ENABLED).</p>'
    note = orb.get("note") or ""
    rs = orb.get("risk_state") or {}
    # risk-state banner
    rb = ""
    if rs:
        chips = (f'<span class="pill">{rs.get("trades_today",0)}/{rs.get("max_trades_per_day","–")} trades today</span> '
                 f'<span class="pill">{rs.get("open_today",0)}/{rs.get("max_concurrent","–")} open</span> '
                 f'<span class="pill">{rs.get("consec_losses",0)} loss streak</span>')
        if rs.get("blocked"):
            why = "daily trade cap reached" if rs.get("trades_capped") else "halted after consecutive losses"
            chips += f' <span class="pill" style="color:var(--sell);border-color:var(--sell);">{_svg("octagon",12)} new trades blocked — {why}</span>'
        rb = f'<div style="margin:6px 0 14px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">{chips}</div>'
    # signals table
    # signals render as cards (same look as Signals/Intraday) via JS from DATA.orb; this is the
    # no-JS fallback note only.
    sig_tbl = f'<div id="orbCards"><p style="color:var(--muted);font-size:13px;">{note or "Loading ORB signals…"}</p></div>'
    # in-play list
    ip = orb.get("inplay") or []
    ip_html = ""
    if ip:
        ir = ""
        for c in ip[:20]:
            comp = c.get("components") or {}
            ir += (f'<tr><td><b>{c.get("symbol")}</b></td>'
                   f'<td style="text-align:right;">{c.get("in_play")}</td>'
                   f'<td>{c.get("in_play_band")}</td>'
                   f'<td style="text-align:right;">{c.get("gap_pct")}%</td>'
                   f'<td style="text-align:right;">{c.get("rel_volume")}x</td>'
                   f'<td style="text-align:right;">{int(comp.get("catalyst",0))}</td>'
                   f'<td style="text-align:right;">{int(comp.get("liquid",0))}</td></tr>')
        _ip_tbl = ('<table class="trackrec"><thead><tr><th>Sym</th>'
                   '<th style="text-align:right;">In-play</th><th>Band</th>'
                   '<th style="text-align:right;">Gap</th><th style="text-align:right;">RVOL</th>'
                   '<th style="text-align:right;">Catalyst</th><th style="text-align:right;">Liq</th>'
                   '</tr></thead><tbody>' + ir + '</tbody></table>')
        ip_html = _orb_details(
            'Stocks in play <span style="text-transform:none;font-weight:400;color:var(--muted);'
            'font-size:11px;">— today\'s ranked watchlist (gap · RVOL · catalyst · liquidity)</span>',
            _ip_tbl)
    tk = orb.get("track") or {}
    tk_html = ""
    if tk:
        tk_html = (f'<p style="color:var(--muted);font-size:12px;margin-top:12px;">Shadow record: '
                   f'{tk.get("advised",0)} logged · {tk.get("resolved",0)} resolved · '
                   f'{tk.get("open",0)} open · win rate {tk.get("win_rate") if tk.get("win_rate") is not None else "—"}%</p>')
    # cost-modeled backtest across windows (the real edge test, net of costs)
    bt = orb.get("backtest") or {}
    bt_html = ""
    byw = bt.get("by_window") or {}
    if byw:
        br = ""
        for w in sorted(byw, key=lambda x: int(x)):
            st = byw[w] or {}
            if not st.get("n"):
                br += f'<tr><td>{w}-min</td><td colspan="5" style="color:var(--muted);">no trades in sample</td></tr>'
                continue
            exp = st.get("expectancy_pct")
            ec = "buy" if (exp or 0) > 0 else "sell" if (exp or 0) < 0 else ""
            pf = st.get("profit_factor")
            pfc = "buy" if (pf or 0) >= 1.3 else "sell" if (pf is not None and pf < 1) else ""
            br += (f'<tr><td>{w}-min</td><td style="text-align:right;">{st.get("n")}</td>'
                   f'<td style="text-align:right;">{st.get("win_rate")}%</td>'
                   f'<td style="text-align:right;" class="{ec}">{("+" if (exp or 0) > 0 else "")}{exp}%</td>'
                   f'<td style="text-align:right;">{st.get("avg_r")}R</td>'
                   f'<td style="text-align:right;" class="{pfc}">{pf if pf is not None else "—"}</td></tr>')
        _bt_tbl = ('<table class="trackrec"><thead><tr><th>OR window</th>'
                   '<th style="text-align:right;">Trades</th><th style="text-align:right;">Win%</th>'
                   '<th style="text-align:right;">Expectancy</th><th style="text-align:right;">Avg R</th>'
                   '<th style="text-align:right;">Profit factor</th></tr></thead><tbody>' + br + '</tbody></table>'
                   '<p style="color:var(--muted);font-size:11px;margin:4px 0 0;">Small, recent sample — a '
                   'directional read, not proof. Walk-forward validation comes as the shadow record grows.</p>')
        bt_html = _orb_details(
            'Backtest <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:11px;">'
            f'— {bt.get("names",0)} in-play names, ~{bt.get("lookback_days","?")} days of 5-min bars, net of '
            'costs. Expectancy &gt; 0 &amp; profit factor &gt; 1.3 is the bar.</span>', _bt_tbl)
    cap = ('<p style="color:var(--muted);font-size:11px;margin-top:10px;">Spread is a conservative '
           'IEX top-of-book estimate (runs a touch wider than the true NBBO). Backtest uses a fixed '
           'bps cost model. Long-only v1; shadow signals only — no orders.</p>')
    # lead with the cards (with risk-state chips), then collapsible backtest + in-play + shadow record
    return head + rb + sig_tbl + tk_html + bt_html + ip_html + cap


def render_html(snap: dict) -> str:
    data_json = json.dumps(snap)
    icon_js = _icon_js_object()
    mode = snap["mode"]
    mode_note = {
        "LIVE": "Live account data. Real money is at risk if you act on these.",
        "PAPER": "Alpaca paper data and account.",
        "SYNTHETIC": "Synthetic data — NOT real prices or news. Add Alpaca keys for the real thing.",
    }[mode]
    track_html = _track_html(snap.get("track"))
    if track_html:
        try:
            track_html = _book_risk_html(snap.get("book_risk")) + _performance_html(snap.get("performance")) + track_html
            track_html += _learned_html(snap.get("learned"))
        except Exception:  # noqa: BLE001 - panels are additive; never break the build
            pass
    # Dedicated Analyst tab — the autonomous nightly self-review.
    _an_body = ""
    try:
        _an_body = _analyst_html(snap.get("analyst"))
    except Exception:  # noqa: BLE001
        _an_body = ""
    if not _an_body:
        _an_body = (f'<h3 style="font-size:15px;margin:0 0 6px;"><span class="ai-ident">{_svg("ai",15)}</span> Autonomous analyst</h3>'
                    '<p style="color:var(--muted);font-size:13px;">The analyst runs in the cloud after '
                    'each close, reviews every strategy bucket, and posts prioritised proposed changes '
                    'here. Nothing yet — the first report lands after the next nightly run.</p>')
    analyst_html = _an_body
    _paper_acct = snap.get("paper_acct")
    paper_html = _paper_html(_paper_acct)
    paper_nav = '<button data-page="paper">Paper account</button>' if _paper_acct else ''
    paper_section = f'<section class="page" id="page-paper">{paper_html}</section>' if _paper_acct else ''
    _pairs_data = snap.get("pairs") or {}
    pairs_html = _pairs_html(_pairs_data)
    orb_html = _orb_html(snap.get("orb"))
    pairs_nav = '<button data-page="pairs">Pairs</button>' if _pairs_data.get("pairs") else ''
    altdata_html = _altdata_html(snap)
    news_ideas_html = _news_ideas_html(snap.get("news_ideas"))
    system_html = _system_html(snap.get("system"))
    sysdiag_html = _system_health_html(_load_json_safe("system_diagnostic.json"))
    metalabel_html = _metalabel_html(_load_json_safe("meta_history.json"))
    analytics_html = _analytics_html(snap)
    premium_html = _premium_selling_html(snap)
    brain_html = _brain_html(snap)
    control_html = _control_html(snap)
    whatsnew_html = _changelog_html(snap.get("changelog"))
    agents_html = _agent_web_html(snap) + _agent_universe_html(snap)
    regime_html = _regime_html(snap.get("regime"))
    # Compact regime pill for the top app bar (colour-coded dot + label).
    _reg = snap.get("regime") or {}
    _reg_lab = _reg.get("label")
    _reg_col = {"Risk-on": "var(--buy)", "Neutral": "var(--muted)",
                "Risk-off": "var(--sell)"}.get(_reg_lab, "var(--muted)")
    regime_pill = (f'<span class="regime-pill" title="Market regime — {_reg.get("breadth","")}% of '
                   f'{_reg.get("total","")} scanned above trend" style="color:{_reg_col};">'
                   f'<svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true">'
                   f'<circle cx="4.5" cy="4.5" r="4.5" fill="currentColor"/></svg>{_reg_lab}</span>'
                   ) if _reg_lab else ""
    _pd = snap.get("price_drops") or []
    pdrop_html = (f' &middot; <span style="color:var(--muted);" title="{(" | ".join(_pd))[:300].replace(chr(34), chr(39))}">'
                  f'{len(_pd)} dropped (bad feed price)</span>') if _pd else ""
    kpi_html = _kpi_html(snap.get("regime"), snap)
    bento_home_html = _bento_home(snap)
    signals_hero_html = _signals_hero(snap)
    ticker_tape_html = _ticker_tape_html()
    showcase_html = _showcase(snap)
    meganav_html = _meganav()
    _brief = (snap.get("market_brief") or "").strip()
    brief_html = (f'<div class="ai-box" style="margin:2px 0 18px;line-height:1.6;">'
                  f'<span class="ai-h">{_svg("ai",13)} Market brief</span> {_md_inline(_brief)}</div>') if _brief else ""
    _changes = snap.get("changes") or []
    changes_html = ((f'<div class="ai-box" style="margin:0 0 18px;border-color:color-mix(in srgb,#b5b5ba 32%,transparent);'
                     f'background:color-mix(in srgb,#b5b5ba 11%,transparent);">'
                     f'<span class="ai-h" style="color:#b5b5ba;">{_svg("bolt",13)} What changed since last build</span>'
                     f'<ul style="margin:7px 0 0;padding-left:18px;line-height:1.8;">'
                     + "".join(f"<li>{_c}</li>" for _c in _changes) + "</ul></div>") if _changes else "")
    momentum_html = (_momentum_bt_html(snap.get("momentum_bt"))
                     + _walkforward_html(snap.get("walkforward"))
                     + _momentum_html(snap.get("momentum") or []))
    allweather_html = _allweather_html(snap.get("allweather"))
    portfolio_html = (_ranked_html(snap.get("ranked")) + _structured_html(snap.get("structured"))
                      + _nlp_html(snap.get("nlp_scores")) + _portfolio_html(snap.get("portfolio")))
    ipo_html = _ipo_html(snap.get("ipos") or [], snap.get("ipo_news") or [])
    sectors_html = _sectors_html(snap.get("sectors"))
    macro_html = (_notrade_html(snap.get("notrade"))
                  + _macro_posture_html(snap.get("macro_posture"))
                  + _timing_html(snap.get("timing"))
                  + _setup_study_html(snap.get("setups_study"))
                  + _macro_html(snap.get("macro")) + _calendar_html(snap.get("calendar")))
    dh = snap.get("data_health")
    if not dh:
        health_html = ""
    else:
        n_err = dh.get("n_err", 0)
        n_warn = dh.get("n_warn", 0)
        if n_err:
            tip = " | ".join(dh.get("errors", []))[:400].replace('"', "'")
            health_html = (f' &middot; <span style="color:var(--sell);" title="{tip}">'
                           f'{_svg("warn",12)} data check · {n_err} to review</span>')
        elif n_warn:
            tip = ("Extreme but likely-real movers (volatile names): "
                   + " | ".join(dh.get("warnings", []))[:380]).replace('"', "'")
            health_html = (f' &middot; <span style="color:#2ea043;" title="{tip}">{_svg("check",12)} data check</span>'
                           f' <span style="color:var(--muted);font-size:12px;" title="{tip}">'
                           f'· {n_warn} volatile</span>')
        else:
            health_html = (f' &middot; <span style="color:#2ea043;" '
                           f'title="{dh.get("checks",0)} integrity checks passed">{_svg("check",12)} data check</span>')
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Trading Signals Dashboard</title>
<meta name="theme-color" content="#0d1117">
<link rel="manifest" href="manifest.webmanifest">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Signal Desk">
<link rel="apple-touch-icon" href="icon-180.png">
<link rel="icon" type="image/png" href="icon-192.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://s3.tradingview.com/tv.js"></script>
<script src="chart_engine.js"></script>
<style>
  /* Light "Capital IQ Pro" palette is the default; dark is a toggle. */
  :root {{ --bg:#f5f7fa; --card:#ffffff; --line:#e4e8ed; --txt:#16202c;
    --muted:#5b6776; --txt2:#3d4757; --buy:#0a7d44; --sell:#d1242f; --hold:#0b5cad; --flat:#6b7785;
    --short:#c2410c; --watch:#475569; --exit:#b45309; --avoid:#6b7280; --warn:#b7791f;
    --accent:#0b5cad; --grid:rgba(120,130,145,0.16); --cross:rgba(60,70,85,0.4);
    --inset:#f1f4f8; --hover:#eef2f7; --ring:rgba(11,92,173,.40);
    --tape:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;
    --mono:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;
    --hud-edge:color-mix(in srgb,var(--accent) 46%,var(--line));
    --shadow:0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06);
    --shadow-lg:0 6px 20px rgba(16,24,40,0.10);
    --acc2:#b5b5ba; --amb-ground:#eef1f6; --glass-bg:rgba(255,255,255,.72); --glass-bd:rgba(16,24,40,.09); --glass-blur:14px;
    --ai:#6d3fd4; --ai-soft:rgba(109,63,212,.12); }}
  /* Warm-gold glass system (v1 — see DESIGN_SPEC.md). Dark-first. */
  /* xAI-style: pure-black canvas, monochrome chrome, green/red kept only for P&L + direction. */
  html[data-theme="dark"] {{ --bg:#0a0a0a; --amb-ground:#0a0a0a; --card:#1a1a1a; --line:rgba(255,255,255,.10); --txt:#f5f5f5;
    --muted:#8a8a8f; --txt2:#b5b5ba; --buy:#5ed6a6; --sell:#f0797f; --hold:#c7c7cc; --flat:#6e7681;
    --short:#f0797f; --watch:#a8a8ad; --exit:#d0d0d3; --avoid:#6e7681; --warn:#d0d0d3;
    --accent:#f5f5f5; --acc2:#ffffff; --grid:rgba(255,255,255,.05); --cross:rgba(160,160,166,0.4);
    --inset:#242428; --hover:#2b2b30; --ring:rgba(255,255,255,.30);
    --glass-bg:#1a1a1a; --glass-bd:rgba(255,255,255,.10); --glass-blur:0px;
    --hud-edge:rgba(255,255,255,.12); --ai:#c9c9cf; --ai-soft:rgba(255,255,255,.06);
    --shadow:none; --shadow-lg:none; }}
  * {{ box-sizing:border-box; }}
  /* ---- inline SVG icon set (Tabler/Lucide 1.5-stroke, inherits colour) ---- */
  .ico {{ display:inline-block; vertical-align:-.16em; flex:0 0 auto; }}
  .ico-b {{ vertical-align:-.22em; }}
  .sech-ico {{ vertical-align:-.16em; margin-right:6px; opacity:.85; }}
  /* ---- global polish: motion, focus, numerals, scrollbars ---- */
  button, select, .card, .wl, summary, .tabs button, .ctlgrp button, .ctlbtn, .tc-seg button {{
    transition:background-color .15s ease, border-color .15s ease, color .15s ease,
               transform .15s ease, box-shadow .15s ease; }}
  button {{ font-family:inherit; }}
  :focus-visible {{ outline:2px solid var(--ring); outline-offset:2px; border-radius:6px; }}
  a {{ color:var(--accent); text-underline-offset:2px; }}
  .px, .stat .v, .kv span:last-child, .trackrec td, .secpct, .readout .rprice,
  .wl-px, .wl-chg, .convbadge {{ font-variant-numeric:tabular-nums; font-feature-settings:'tnum' 1; }}
  /* HUD: tabular data figures (Inter, tabular-nums) */
  .kpi-v, .stat .v, .px, .wl-px, .convbadge, .num {{ font-family:var(--mono); }}
  ::-webkit-scrollbar {{ width:10px; height:10px; }}
  ::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:6px; border:2px solid transparent;
    background-clip:padding-box; }}
  ::-webkit-scrollbar-thumb:hover {{ background:var(--muted); background-clip:padding-box; }}
  ::-webkit-scrollbar-track {{ background:transparent; }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ transition:none !important; animation:none !important; scroll-behavior:auto !important; }}
    .card:hover {{ transform:none; }} }}
  html, body {{ max-width:100%; overflow-x:hidden; }}
  body {{ margin:0; font:14px/1.55 'Inter',-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--txt);
    -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale; text-rendering:optimizeLegibility; }}
  /* warm-gold drifting ambient glow behind everything (F2 / softer) */
  body::before {{ content:''; position:fixed; inset:0; z-index:-2; pointer-events:none;
    background:#000000; }}
  html[data-theme="dark"] body::after {{ content:none; }}
  @keyframes ambDrift {{ 0%,100%{{ transform:translate(0,0); }} 33%{{ transform:translate(11vw,7vh); }} 66%{{ transform:translate(-6vw,11vh); }} }}
  @media (prefers-reduced-motion: reduce) {{ html[data-theme="dark"] body::after {{ animation:none; }} }}
  /* ===== scrolling ticker tape (terminal marquee) ===== */
  .tickertape {{ position:relative; overflow:hidden; white-space:nowrap; margin:0 0 12px;
    border:1px solid var(--glass-bd); border-radius:12px; background:var(--glass-bg);
    backdrop-filter:blur(var(--glass-blur)); -webkit-backdrop-filter:blur(var(--glass-blur));
    box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,.05); }}
  .tickertape::before, .tickertape::after {{ content:''; position:absolute; top:0; bottom:0; width:46px; z-index:2; pointer-events:none; }}
  .tickertape::before {{ left:0; background:linear-gradient(90deg, var(--bg), transparent); }}
  .tickertape::after  {{ right:0; background:linear-gradient(270deg, var(--bg), transparent); }}
  html[data-theme="dark"] .tickertape::before {{ background:linear-gradient(90deg, rgba(6,6,7,.92), transparent); }}
  html[data-theme="dark"] .tickertape::after  {{ background:linear-gradient(270deg, rgba(6,6,7,.92), transparent); }}
  .tkt-track {{ display:inline-flex; align-items:center; padding:8px 0; animation:tktScroll 74s linear infinite; }}
  .tickertape:hover .tkt-track {{ animation-play-state:paused; }}
  @keyframes tktScroll {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
  .tkt-it {{ display:inline-flex; align-items:baseline; gap:7px; margin:0 16px; font-family:var(--tape);
    font-size:12px; font-variant-numeric:tabular-nums; }}
  .tkt-it .tlogo {{ width:17px; height:17px; border-radius:4px; object-fit:contain; background:#fff;
    align-self:center; flex:0 0 auto; }}
  .tkt-it .tlogo-mono {{ display:inline-flex; align-items:center; justify-content:center; background:var(--inset);
    color:var(--txt2); font-size:8px; font-weight:800; letter-spacing:.01em; text-transform:uppercase; }}
  .tkt-it .sym {{ color:var(--txt); font-weight:700; letter-spacing:.02em; }}
  .tkt-it .px {{ color:var(--txt2); }}
  .tkt-it .chg.up {{ color:var(--buy); }} .tkt-it .chg.dn {{ color:var(--sell); }}
  .tkt-it .dir {{ display:inline-flex; align-items:center; vertical-align:-.12em; }}
  .tkt-it .dir.up {{ color:var(--buy); }} .tkt-it .dir.dn {{ color:var(--sell); }}
  .tkt-sep {{ display:inline-block; width:1px; height:11px; margin:0 2px; background:var(--line); vertical-align:-1px; }}
  @media (prefers-reduced-motion: reduce) {{ .tkt-track {{ animation:none; }} }}
  /* ===== section headers with SVG icons + dividers ===== */
  .sec-head {{ display:flex; align-items:center; gap:10px; margin:30px 0 14px; padding-bottom:0;
    border-bottom:0; }}
  .sec-head:first-child {{ margin-top:6px; }}
  .sec-head .sh-ico {{ display:inline-grid; place-items:center; width:22px; height:22px; border-radius:6px;
    background:transparent; color:var(--txt); flex:0 0 auto; }}
  .sec-head .sh-ico.ai {{ background:transparent; color:var(--txt); }}
  .sec-head h2 {{ margin:0; font-size:18px; font-weight:600; letter-spacing:-.01em; text-transform:none;
    color:var(--txt); }}
  .sec-head .sh-sub {{ margin-left:auto; font-size:13px; color:var(--muted); font-weight:400; }}
  .sec-rule {{ height:1px; background:var(--line); border:0; margin:22px 0; }}
  /* AI identity: violet accent for any AI/analyst surface */
  .ai-ident {{ color:var(--ai); }}
  html[data-theme="dark"] .ai-box, html[data-theme="dark"] .ai-read {{ border-left:2px solid var(--ai) !important; }}
  .ai-h, .ai-read-h {{ color:var(--ai); }}
  /* reusable frosted glass panel */
  .glass {{ background:var(--glass-bg); border:1px solid var(--glass-bd); border-radius:14px; box-shadow:none; }}
  /* Apply the glass system to EVERY surface tile (dark), so the whole dashboard feels the design. */
  html[data-theme="dark"] .card, html[data-theme="dark"] .bento .bt, html[data-theme="dark"] .bento-tile,
  html[data-theme="dark"] .bento-regime, html[data-theme="dark"] .bento-feat, html[data-theme="dark"] .kpi,
  html[data-theme="dark"] .stat, html[data-theme="dark"] .wl, html[data-theme="dark"] .featured,
  html[data-theme="dark"] .lane, html[data-theme="dark"] .tk-panel, html[data-theme="dark"] .secbar,
  html[data-theme="dark"] .trackrec, html[data-theme="dark"] details.tvwidget {{
    background:var(--card) !important;
    border:1px solid var(--glass-bd) !important; border-radius:14px !important;
    box-shadow:none !important; }}
  /* clickable float on interactive tiles */
  html[data-theme="dark"] .card, html[data-theme="dark"] .bento .bt, html[data-theme="dark"] .kpi,
  html[data-theme="dark"] .stat, html[data-theme="dark"] .lane {{
    cursor:pointer; transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }}
  html[data-theme="dark"] .card:hover, html[data-theme="dark"] .bento .bt:hover, html[data-theme="dark"] .kpi:hover,
  html[data-theme="dark"] .stat:hover, html[data-theme="dark"] .lane:hover {{
    transform:none;
    border-color:rgba(255,255,255,.28) !important;
    box-shadow:none !important; }}
  .wrap {{ width:100%; max-width:1480px; margin:0 auto;
    padding:0 max(24px, env(safe-area-inset-right)) calc(60px + env(safe-area-inset-bottom)) max(24px, env(safe-area-inset-left)); }}
  .grid-stack {{ width:100%; }}
  h1 {{ font-size:30px; font-weight:300; letter-spacing:-.025em; margin:0 0 5px; }}
  h2 {{ font-size:13px; margin:30px 0 12px; color:var(--muted); font-weight:700;
    text-transform:uppercase; letter-spacing:.06em; }}
  .meta {{ color:var(--muted); font-size:13px; margin-bottom:6px; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:12px; font-weight:600; }}
  .m-LIVE {{ background:#5a1e1e; color:#ff9b9b; }}
  .m-PAPER {{ background:#15361f; color:#7ee2a0; }}
  .m-SYNTHETIC {{ background:#3a2e12; color:#b5b5ba; }}
  .note {{ color:var(--muted); font-size:13px; margin:10px 0 8px;
    display:inline-block; padding:5px 11px; border-radius:9px;
    border:1px solid var(--line); background:var(--inset); }}
  .grid {{ display:flex; gap:14px; overflow-x:auto; overflow-y:hidden; align-items:stretch;
    padding:2px 2px 14px; scroll-snap-type:x proximity; scrollbar-width:thin; }}
  .grid > .card {{ flex:0 0 340px; scroll-snap-align:start; }}
  .grid::-webkit-scrollbar {{ height:8px; }}
  .grid::-webkit-scrollbar-thumb {{ background:var(--inset); border-radius:8px; }}
  .grid::-webkit-scrollbar-track {{ background:transparent; }}
  @media (max-width:600px) {{ .grid > .card {{ flex-basis:82vw; }} }}
  /* live-signals scroll controls + expand-all */
  .cards-ctl {{ margin-left:auto; display:flex; align-items:center; gap:8px; }}
  .cards-ctl .sh-sub {{ margin-left:0; }}
  .cbtn {{ display:inline-flex; align-items:center; justify-content:center; min-width:30px; height:30px;
    border-radius:999px; border:1px solid rgba(255,255,255,.16); background:none; color:var(--muted);
    font-size:15px; line-height:1; cursor:pointer; transition:color .15s ease, border-color .15s ease; }}
  .cbtn:hover {{ color:var(--txt); border-color:var(--txt); }}
  .cbtn-txt {{ padding:0 14px; font-size:12.5px; font-weight:500; }}
  #cards.expanded .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
    gap:14px; overflow:visible; scroll-snap-type:none; }}
  .card {{ background:var(--glass-bg); border:1px solid var(--glass-bd); border-radius:14px;
    padding:16px; cursor:pointer; box-shadow:none; }}
  .card:hover {{ border-color:rgba(255,255,255,.28); transform:none; box-shadow:none; }}
  .ladder {{ margin-top:12px; border:0.5px solid var(--line); border-radius:8px; overflow:hidden; }}
  .lad-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:6px 11px; font-size:13px; }}
  .lad-row > span:first-child {{ color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; }}
  .lad-row > span:last-child {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  .lad-row em {{ font-style:normal; font-size:11px; font-weight:500; margin-left:7px; }}
  .lad-row.ent {{ background:var(--inset); }}
  .lad-row.tgt > span:last-child, .lad-row.tgt em {{ color:var(--buy); }}
  .lad-row.stp > span:last-child, .lad-row.stp em {{ color:var(--sell); }}
  .lad-rr {{ padding:5px 11px; border-top:0.5px solid var(--line); font-size:11px; color:var(--muted); text-align:right; }}
  .card-warn {{ margin-top:9px; font-size:11.5px; color:var(--sell); font-weight:500; }}
  /* scraped alt-data badges (insider / analyst rating / retail buzz) */
  .altrow {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
  .altpill {{ font-size:11px; font-weight:700; padding:3px 9px; border-radius:7px;
    border:1px solid currentColor; background:color-mix(in srgb, currentColor 10%, transparent);
    display:inline-flex; align-items:center; gap:4px; cursor:help; line-height:1.3; }}
  .bbalt {{ margin-top:4px; font-size:10.5px; font-weight:700; display:flex; gap:9px; flex-wrap:wrap; }}
  .bbalt span {{ cursor:help; }}
  .conc-warn {{ margin:0 0 16px; padding:10px 14px; font-size:12.5px; line-height:1.5; border-radius:10px;
    background:color-mix(in srgb, #6e7681 12%, transparent); border:1px solid color-mix(in srgb, #6e7681 38%, transparent);
    color:var(--txt); cursor:help; }}
  .hcell {{ cursor:help; text-decoration:underline dotted var(--muted); text-underline-offset:3px;
    text-decoration-thickness:1px; }}
  .hint {{ cursor:help; }}
  #tip {{ position:fixed; z-index:9999; display:none; max-width:300px; padding:9px 12px;
    background:var(--card); color:var(--txt); border:1px solid var(--line);
    border-radius:8px; box-shadow:var(--shadow-lg); font-size:12px; line-height:1.5;
    pointer-events:none; }}
  html[data-theme="dark"] #tip {{ background:#0f0f12; border-color:rgba(255,255,255,.14); }}
  #tip.rich {{ max-width:280px; padding:12px 14px; }}
  .cb-wrap {{ min-width:190px; }}
  .cb-h {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:9px; }}
  .cb-row {{ display:grid; grid-template-columns:1fr 66px 32px; align-items:center; gap:8px; margin:5px 0; font-size:11.5px; }}
  .cb-nm {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--txt2); }}
  .cb-bar {{ height:6px; background:var(--inset); border-radius:3px; overflow:hidden; }}
  .cb-bar i {{ display:block; height:100%; border-radius:3px; }}
  .cb-v {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--txt); }}
  .cb-empty {{ font-size:11.5px; color:var(--muted); }}
  .cb-line {{ font-size:11.5px; line-height:1.5; color:var(--txt2); }}
  .cb-line b {{ color:var(--txt); font-weight:600; }}
  .cb-stack {{ display:flex; height:8px; border-radius:4px; overflow:hidden; background:var(--inset); margin-bottom:8px; }}
  .cb-stack i {{ display:block; height:100%; }}
  .cb-leg {{ display:flex; gap:12px; font-size:11px; color:var(--muted); }}
  .cb-leg b {{ color:var(--txt); font-variant-numeric:tabular-nums; }}
  .cb-cm {{ margin-top:8px; padding-top:8px; border-top:1px solid var(--line); font-size:11px; color:var(--txt2); }}
  /* inline mini-waffle for table cells / rate metrics */
  .wmini {{ display:inline-flex; gap:2px; vertical-align:middle; }}
  .wmini i {{ width:6px; height:10px; border-radius:2px; background:var(--inset); display:inline-block; }}
  .wmini i.win {{ background:var(--buy); }}
  .wmini i.loss {{ background:var(--sell); }}
  .wmini i.warn {{ background:var(--warn); }}
  .wmini i.neu {{ background:var(--txt2); }}
  .wrcell {{ display:inline-flex; align-items:center; gap:8px; justify-content:flex-end;
    font-variant-numeric:tabular-nums; }}
  tr.hint {{ cursor:help; }}
  /* section rate-hero: a big number beside a full waffle (Track record / System) */
  .rate-hero {{ display:flex; align-items:center; gap:22px; background:var(--card); border:1px solid var(--line);
    border-radius:14px; padding:18px 20px; margin:14px 0; flex-wrap:wrap; }}
  .rate-hero .rh-v {{ font-size:34px; font-weight:600; line-height:1; font-variant-numeric:tabular-nums; }}
  .rate-hero .rh-k {{ font-size:12px; color:var(--muted); margin-top:6px; }}
  .rate-hero .an-waffle {{ flex:1; min-width:210px; max-width:260px; }}
  /* Your real journal record (from Obsidian) */
  .jrec {{ background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 7%,var(--card)),var(--card));
    border:1px solid color-mix(in srgb,var(--accent) 28%,var(--line)); border-radius:14px; padding:16px 18px; margin:14px 0; }}
  .jrec-h {{ font-size:13px; font-weight:700; color:var(--txt); display:flex; align-items:center; gap:7px; }}
  .jrec-h .ico {{ color:var(--accent); }}
  .jrec-sub {{ font-weight:400; color:var(--muted); font-size:11.5px; }}
  .jrec-row {{ display:flex; flex-wrap:wrap; gap:26px; margin:12px 0 6px; }}
  .jrec-stat .v {{ font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1; }}
  .jrec-stat .k {{ font-size:11px; color:var(--muted); margin-top:2px; }}
  .jrec-note {{ font-size:11.5px; color:var(--muted); }}
  /* strategy-mix waffle on signal cards */
  .sx-conf {{ margin-top:14px; cursor:help; }}
  .sx-conf-h {{ display:flex; justify-content:space-between; font-size:10.5px; text-transform:uppercase;
    letter-spacing:.05em; color:var(--muted); margin-bottom:8px; }}
  .sx-conf-n {{ font-variant-numeric:tabular-nums; color:var(--txt2); }}
  .sw-waffle {{ display:flex; flex-wrap:wrap; gap:5px; }}
  .sw-sq {{ width:20px; height:20px; border-radius:5px; background:var(--inset);
    transition:transform .12s ease; }}
  .sx-conf:hover .sw-sq {{ transform:none; }}
  .sw-sq.sw-on {{ background:color-mix(in srgb,var(--buy) 50%,transparent); }}
  .sw-sq.sw-fresh {{ background:var(--buy); }}
  .sx-conf.short .sw-sq.sw-on {{ background:color-mix(in srgb,var(--sell) 50%,transparent); }}
  .sx-conf.short .sw-sq.sw-fresh {{ background:var(--sell); }}
  #newbuild {{ position:fixed; left:50%; transform:translateX(-50%); bottom:18px; z-index:9998;
    background:var(--accent); color:#fff; border:0; border-radius:999px; cursor:pointer;
    padding:9px 16px; font-size:13px; font-weight:600; box-shadow:var(--shadow-lg); }}
  #newbuild:hover {{ filter:brightness(1.08); }}
  /* ---- alternate layouts ---- */
  .cat-chip {{ margin-top:10px; display:inline-block; font-size:11.5px; font-weight:600;
    color:#6e7681; background:color-mix(in srgb, #b5b5ba 16%, transparent);
    border:1px solid color-mix(in srgb, #b5b5ba 36%, transparent); padding:3px 9px; border-radius:999px; }}
  .ai-tag {{ color:var(--accent); font-weight:700; }}
  .ai-box {{ margin-top:10px; padding:9px 11px; border-radius:8px; font-size:12px; line-height:1.5;
    color:var(--txt); background:color-mix(in srgb, #9b59b6 12%, transparent);
    border:1px solid color-mix(in srgb, #9b59b6 32%, transparent); }}
  .ai-h {{ color:#9b59b6; font-weight:700; margin-right:5px; }}
  .tv-chip {{ margin-top:8px; display:inline-block; font-size:11.5px; font-weight:600;
    color:var(--accent); background:color-mix(in srgb, var(--accent) 12%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 30%, transparent); padding:3px 9px; border-radius:999px; }}
  .secmix {{ display:flex; flex-direction:column; gap:5px; margin:4px 0 8px; }}
  .secrow {{ display:flex; align-items:center; gap:10px; font-size:12px; }}
  .secname {{ width:120px; color:var(--muted); }}
  .secbarwrap {{ flex:1; height:8px; background:var(--inset); border-radius:4px; overflow:hidden; }}
  .secbarfill {{ height:100%; }}
  .secval {{ width:80px; text-align:right; font-variant-numeric:tabular-nums; }}
  .card-spark {{ margin:8px 0 2px; }}
  .card-spark svg {{ width:100% !important; height:42px; display:block; opacity:.9; }}
  .mono2 {{ display:inline-flex; align-items:center; justify-content:center; border-radius:5px;
    color:#fff; font-weight:700; overflow:hidden; position:relative; flex:none; }}
  .mono2 img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#fff; }}
  .bbwrap {{ background:#000; border:1px solid #2a2a17; border-radius:8px; overflow:hidden;
    font-family:var(--mono); }}
  .bbhead {{ display:flex; align-items:center; gap:18px; padding:8px 12px; background:#13130a;
    border-bottom:1px solid #2a2a17; font-size:11px; color:#8a8a6a; letter-spacing:1px;
    white-space:nowrap; overflow-x:auto; }}
  .bbtitle {{ color:#d0d0d3; font-weight:700; }} .bbst b {{ color:#d0d0d3; }}
  .bbclock {{ margin-left:auto; color:#5a5a45; }}
  .bbgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:1px; background:#2a2a17; }}
  .bbtile {{ background:#000; padding:9px 11px; cursor:pointer; }}
  .bbtile:hover {{ background:#0c0c06; }}
  .bbtop {{ display:flex; align-items:center; gap:7px; }}
  .bbsym {{ color:#fff; font-weight:700; }} .bbact {{ margin-left:auto; font-size:11px; font-weight:700; }}
  .bbpx {{ color:#fff; font-size:18px; font-weight:700; margin:6px 0 2px; font-variant-numeric:tabular-nums; }}
  .bbmeta {{ color:#8a8a6a; font-size:10.5px; }}
  .bblv {{ color:#8a8a6a; font-size:10.5px; margin-top:3px; font-variant-numeric:tabular-nums; }}
  .bbtv {{ color:#8a8a6a; font-size:10px; margin-top:2px; letter-spacing:.02em; }}
  .gtv {{ color:var(--muted); font-size:10px; margin-top:2px; }}
  .tvwrap {{ position:relative; width:100%; max-width:1100px; padding-bottom:min(56.25%, 620px); height:0;
    border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#000; }}
  .tvwrap iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
  /* compact Bloomberg live widget on the Signals page */
  .tvwidget {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:10px 14px; margin:0 0 16px; max-width:460px; box-shadow:var(--shadow); }}
  .tvwidget summary {{ cursor:pointer; font-weight:700; font-size:13px; display:flex; align-items:center;
    gap:8px; list-style:none; }}
  .tvwidget summary::-webkit-details-marker {{ display:none; }}
  .tvwidget .tvw-open {{ margin-left:auto; font-size:11.5px; font-weight:600; color:var(--muted); text-decoration:none; }}
  .tvwidget .tvw-open:hover {{ text-decoration:underline; }}
  .tvw-frame {{ position:relative; width:100%; padding-bottom:56.25%; height:0; margin-top:10px;
    border-radius:8px; overflow:hidden; background:#000; }}
  .tvw-frame iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
  /* solo TV (S&P chart removed) — a big panel that fills the hero's right column */
  .hero-tv-solo {{ display:flex; flex-direction:column; }}
  .hero-tv-solo .tvw-frame {{ padding-bottom:0; height:540px; margin-top:12px; flex:1 1 auto; min-height:420px; }}
  @media (max-width:900px) {{ .hero-tv-solo .tvw-frame {{ height:0; padding-bottom:56.25%; min-height:0; }} }}
  .lanes {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; align-items:start; }}
  .lanehd {{ font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }}
  .lcard {{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--muted);
    border-radius:8px; padding:9px 10px; margin-bottom:8px; cursor:pointer; }}
  .lcard:hover {{ background:var(--hover); }}
  .lcard-t {{ display:flex; align-items:center; gap:7px; }} .lsym {{ font-weight:600; }}
  .lconv {{ margin-left:auto; color:var(--muted); font-size:12px; }} .lsub {{ color:var(--muted); font-size:11px; margin-top:3px; }}
  .gauges {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(100px,1fr)); gap:16px; }}
  .gauge {{ text-align:center; cursor:pointer; }}
  .gsvg {{ width:64px; height:64px; display:block; margin:0 auto; }}
  .gnum {{ fill:var(--txt); font-size:17px; font-weight:700; font-family:inherit; }}
  .glab {{ display:flex; align-items:center; justify-content:center; gap:5px; margin-top:7px; font-weight:600; font-size:13px; }}
  .gact {{ font-size:11px; }}
  .feedwrap {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:2px 14px; }}
  .feeditem {{ display:flex; align-items:center; gap:11px; padding:11px 0; border-bottom:1px solid var(--line); cursor:pointer; }}
  .feeditem:last-child {{ border-bottom:0; }} .feedtxt {{ font-size:13px; flex:1; min-width:0; }}
  .feedsub {{ color:var(--muted); font-size:11px; margin-top:2px; }}
  .feedspark {{ flex:none; opacity:.9; }}
  .bento {{ display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:10px; }}
  .bento-regime {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px;
    grid-row:span 2; display:flex; flex-direction:column; justify-content:center; }}
  .bento-feat {{ grid-column:span 2; background:var(--card); border:1px solid var(--line); border-left:3px solid var(--buy);
    border-radius:10px; padding:12px; display:flex; align-items:center; gap:10px; cursor:pointer; }}
  .bento-tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px;
    display:flex; align-items:center; gap:8px; cursor:pointer; }}
  .blab {{ font-size:10px; text-transform:uppercase; color:var(--muted); letter-spacing:.04em; }}
  .bval {{ font-size:20px; font-weight:700; }} .btk {{ font-weight:600; font-size:13px; }}
  .mag {{ display:grid; grid-template-columns:1.5fr 1fr; gap:12px; align-items:start; }}
  .mag-hero {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:13px; cursor:pointer; }}
  .mag-hero-t {{ display:flex; align-items:center; gap:9px; margin-bottom:6px; }}
  .mag-side {{ display:flex; flex-direction:column; gap:6px; }}
  .magrow {{ display:flex; align-items:center; gap:8px; background:var(--card); border:1px solid var(--line);
    border-radius:8px; padding:7px 9px; cursor:pointer; }}
  .magrk {{ color:var(--muted); font-size:11px; width:14px; }} .magsym {{ font-weight:600; }} .magc {{ margin-left:auto; font-size:12px; }}
  .tktape {{ overflow:hidden; background:var(--inset); border:1px solid var(--line); border-radius:8px; white-space:nowrap; }}
  .tktape-in {{ display:inline-block; padding:8px 0; animation:tkscroll 45s linear infinite; }}
  .tkitem {{ margin:0 18px; font-variant-numeric:tabular-nums; font-size:13px; }}
  @keyframes tkscroll {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
  .tkbody {{ margin-top:10px; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  .tkrow {{ display:flex; align-items:center; gap:10px; padding:9px 13px; border-bottom:1px solid var(--line); cursor:pointer; }}
  .tkrow:last-child {{ border-bottom:0; }} .tkrow:hover {{ background:var(--hover); }}
  .tksym {{ font-weight:600; width:58px; }} .tkpx {{ width:88px; font-variant-numeric:tabular-nums; }}
  .tkspark {{ flex:none; opacity:.9; }}
  .tkfam {{ color:var(--muted); font-size:12px; width:120px; }} .tklv {{ margin-left:auto; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
  .splitwrap {{ display:grid; grid-template-columns:172px 1fr; gap:12px; align-items:start; }}
  .splitlist {{ display:flex; flex-direction:column; gap:3px; max-height:540px; overflow:auto; }}
  .splititem {{ display:flex; align-items:center; gap:7px; padding:7px 8px; border-radius:8px; cursor:pointer; }}
  .splititem:hover {{ background:var(--hover); }} .splititem.on {{ background:var(--inset); }}
  .splitsym {{ font-weight:600; font-size:13px; }} .splitact {{ margin-left:auto; font-size:10px; white-space:nowrap; }}
  .splitdetail {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:13px; min-height:160px; }}
  .sd-top {{ display:flex; align-items:center; gap:9px; margin-bottom:8px; }}
  .sd-full {{ margin-top:11px; background:var(--accent); color:#fff; border:0; border-radius:7px; padding:7px 13px; font-size:12px; font-weight:600; cursor:pointer; }}
  @media (max-width:760px) {{ .lanes,.bento,.mag,.splitwrap {{ grid-template-columns:1fr; }} }}
  .card-why {{ margin-top:11px; padding:9px 11px; background:var(--inset);
    border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:8px; }}
  .why-h {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.04em;
    font-weight:800; color:var(--muted); margin-bottom:7px; }}
  .why-fam {{ display:inline-block; max-width:100%; box-sizing:border-box;
    font-weight:700; font-size:11px; margin:0 0 9px; white-space:normal;
    color:var(--accent); background:color-mix(in srgb, var(--accent) 14%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 32%, transparent);
    padding:2px 9px; border-radius:999px; }}
  .strat-badge {{ display:inline-flex; align-items:center; gap:9px; margin:0 0 14px;
    padding:6px 13px; border-radius:999px; background:color-mix(in srgb, var(--accent) 12%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 30%, transparent); }}
  .strat-badge .k {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em;
    font-weight:800; color:var(--muted); }}
  .strat-badge .v {{ font-size:13px; font-weight:700; color:var(--accent); }}
  .why-chips {{ display:flex; flex-wrap:wrap; gap:5px; }}
  .why-chip {{ font-size:11.5px; padding:3px 9px; border-radius:999px; line-height:1.35;
    border:1px solid var(--line); background:var(--card); color:var(--txt); white-space:nowrap; }}
  .why-chip.trig {{ background:var(--accent); color:#fff; border-color:var(--accent); font-weight:700; }}
  .why-chip.more {{ color:var(--muted); }}
  .why-txt {{ font-size:12.5px; color:var(--txt); line-height:1.45; }}
  .more {{ color:var(--muted); font-size:12px; margin-top:10px;
    border-top:1px solid var(--line); padding-top:8px; }}
  .sym {{ font-size:18px; font-weight:700; }}
  .logo {{ width:20px; height:20px; border-radius:4px; vertical-align:middle;
    margin-right:7px; background:#fff; object-fit:contain; }}
  .cname {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .act {{ float:right; padding:2px 10px; border-radius:6px; font-size:12px; font-weight:700; color:#fff; }}
  .a-BUY {{ background:var(--buy); }} .a-SELL {{ background:var(--sell); }}
  .a-HOLDLONG {{ background:var(--hold); }} .a-FLAT {{ background:var(--flat); }}
  .a-SHORT {{ background:var(--short); }} .a-HOLDSHORT {{ background:var(--short); opacity:.82; }}
  .a-WATCHLONG {{ background:var(--watch); }} .a-WATCHSHORT {{ background:var(--watch); }}
  .a-EXIT {{ background:var(--exit); }} .a-AVOID {{ background:var(--avoid); }}
  .px {{ font-size:26px; font-weight:700; margin:8px 0 2px; }}
  .kv {{ display:flex; justify-content:space-between; font-size:13px;
    color:var(--muted); padding:3px 0; }}
  .kv span:last-child {{ color:var(--txt); }}
  .hot span:last-child {{ color:#b5b5ba; font-weight:700; }}
  select {{ background:var(--card); color:var(--txt); border:1px solid var(--line);
    border-radius:8px; padding:6px 10px; font-size:14px; }}
  .chartbox {{ background:var(--card); border:1px solid var(--line);
    border-radius:12px; padding:16px; margin-top:14px; }}
  .news a, .news span.h {{ color:var(--txt); text-decoration:none; }}
  .news li {{ margin-bottom:10px; }}
  .news .src {{ color:var(--muted); font-size:12px; }}
  .disclaimer {{ color:var(--muted); font-size:12px; margin-top:36px;
    border-top:1px solid var(--line); padding-top:16px; }}
  /* ===== signal detail modal — glass-terminal redesign ===== */
  .overlay {{ display:none; position:fixed; inset:0; z-index:50; padding:24px; overflow:auto;
    background:rgba(6,6,9,.62); backdrop-filter:blur(9px) saturate(1.1);
    -webkit-backdrop-filter:blur(9px) saturate(1.1); }}
  .overlay.open {{ display:block; }}
  .modal {{ max-width:760px; margin:22px auto; position:relative;
    background:var(--card);
    border:1px solid var(--line); border-radius:16px; padding:0 24px 24px;
    box-shadow:0 24px 60px rgba(0,0,0,.55); }}
  /* header: logo tile · ticker · direction pill · live price · %chg · close */
  .mhead {{ display:flex; align-items:center; gap:14px; position:sticky; top:0; z-index:2;
    margin:0 -24px 4px; padding:18px 22px 15px; border-bottom:1px solid var(--line);
    border-radius:16px 16px 0 0; background:var(--card); }}
  .mhead-id {{ display:flex; align-items:center; gap:12px; min-width:0; flex:1 1 auto; }}
  .mhead-logo {{ position:relative; flex:0 0 auto; width:42px; height:42px; border-radius:11px;
    overflow:hidden; display:grid; place-items:center; font-weight:800; font-size:15px; letter-spacing:.02em;
    color:var(--txt2); background:var(--inset);
    border:1px solid var(--line); box-shadow:none; }}
  .mhead-logo img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
  .mhead-init {{ position:relative; z-index:0; }}
  .mhead-idtext {{ min-width:0; }}
  .mhead-tickrow {{ display:flex; align-items:center; gap:9px; min-width:0; }}
  .mhead-tick {{ font-family:var(--mono); font-size:24px; font-weight:600; letter-spacing:-.01em;
    line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .mhead-pill {{ flex:0 0 auto; display:inline-flex; align-items:center; gap:5px;
    padding:3px 10px 3px 8px; border-radius:999px; font-size:11.5px; font-weight:800;
    letter-spacing:.03em; text-transform:uppercase; color:#fff; white-space:nowrap; }}
  .mhead-pill .ico {{ width:13px; height:13px; }}
  .mhead-pill.a-BUY, .mhead-pill.a-HOLDLONG {{ background:var(--buy); }}
  .mhead-pill.a-SHORT, .mhead-pill.a-HOLDSHORT {{ background:var(--sell); }}
  .mhead-pill.a-HOLD, .mhead-pill.a-WATCH {{ background:var(--muted); }}
  .mhead-name {{ color:var(--muted); font-size:12.5px; margin-top:3px; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis; max-width:100%; }}
  .mhead-quote {{ flex:0 0 auto; text-align:right; margin-left:auto; }}
  .mhead-px {{ font-family:var(--mono); font-size:19px; font-weight:600; line-height:1.1;
    white-space:nowrap; }}
  .mhead-chg {{ display:inline-flex; align-items:center; gap:3px; justify-content:flex-end;
    font-family:var(--mono); font-size:12.5px; font-weight:700; margin-top:3px; color:var(--muted); }}
  .mhead-chg.up {{ color:var(--buy); }} .mhead-chg.dn {{ color:var(--sell); }}
  .mhead-chg .ico {{ width:12px; height:12px; }}
  .mclose {{ flex:0 0 auto; align-self:flex-start; display:grid; place-items:center;
    width:34px; height:34px; border-radius:10px; cursor:pointer; color:var(--muted);
    background:color-mix(in srgb,var(--inset) 70%, transparent); border:1px solid var(--glass-bd);
    transition:color .15s ease, border-color .15s ease, background .15s ease; }}
  .mclose:hover {{ color:var(--accent); border-color:color-mix(in srgb,var(--accent) 40%,var(--glass-bd));
    background:color-mix(in srgb,var(--accent) 12%, var(--inset)); }}
  .reasons {{ list-style:none; padding:0; margin:14px 0; }}
  .reasons li {{ position:relative; padding:8px 0 8px 24px; font-size:14px;
    border-bottom:1px solid var(--line); overflow-wrap:anywhere; }}
  .reasons li:before {{ content:'›'; position:absolute; left:6px; color:var(--accent); }}
  .modal .sech {{ color:var(--muted); font-weight:600; text-transform:none; font-size:13px;
    letter-spacing:0; margin:22px 0 10px; padding-bottom:8px;
    border-bottom:1px solid var(--line); }}
  .mk-view > .sech:first-child {{ margin-top:2px; }}
  .modal .summary {{ font-size:14.5px; margin:14px 0 2px; color:var(--txt2); line-height:1.5;
    overflow-wrap:anywhere; }}
  .modal .chartbox {{ margin-top:0; }}
  .plangrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(0,1fr));
    gap:10px; }}
  @supports (grid-template-columns:repeat(auto-fit,minmax(150px,1fr))) {{
    .plangrid {{ grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr)); }} }}
  /* trade-plan tiles — inset glass chips with a tier accent bar */
  .stat {{ position:relative; min-width:0; background:var(--inset);
    border:1px solid var(--line); border-radius:12px; padding:15px 16px; overflow:hidden;
    box-shadow:none; }}
  .stat::before {{ display:none; }}
  .stat .l {{ color:var(--muted); font-size:10px; text-transform:uppercase; font-weight:600;
    letter-spacing:.05em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .stat .v {{ font-size:21px; font-weight:600; letter-spacing:-.015em; font-variant-numeric:tabular-nums; margin-top:5px; overflow-wrap:anywhere; }}
  /* Direction 2 — strategy row-cards with inline win-rate bars */
  .strows {{ display:flex; flex-direction:column; }}
  .strow {{ display:grid; grid-template-columns:8px 1fr 90px 42px 30px 52px; align-items:center; gap:11px;
    padding:12px 4px; border-top:1px solid var(--line); cursor:help; }}
  .strow:first-child {{ border-top:0; }}
  .st-dot {{ width:7px; height:7px; border-radius:50%; }}
  .st-nm {{ font-size:13px; font-weight:500; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .st-kind {{ color:var(--muted); font-weight:400; margin-left:8px; font-size:11.5px; }}
  .st-bar {{ height:6px; background:var(--inset); border-radius:4px; overflow:hidden; }}
  .st-bar i {{ display:block; height:100%; border-radius:4px; }}
  .st-wr {{ text-align:right; font-size:12.5px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .st-tr {{ text-align:right; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
  .st-ret {{ text-align:right; font-size:12.5px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .st-ret.win {{ color:var(--buy); }} .st-ret.loss {{ color:var(--sell); }}
  @media (max-width:600px) {{ .strow {{ grid-template-columns:8px 1fr 40px 52px; }} .st-bar, .st-tr {{ display:none; }} }}
  /* Direction 2 — committee analyst rows */
  .crows {{ margin-top:12px; }}
  .crow {{ display:grid; grid-template-columns:132px 76px 1fr; gap:12px; align-items:start;
    padding:11px 2px; border-top:1px solid var(--line); font-size:12.5px; }}
  .crow:first-child {{ border-top:0; }}
  .cr-role {{ color:var(--muted); }}
  .cr-lean {{ text-transform:capitalize; font-weight:600; }}
  .cr-note {{ color:var(--txt2); line-height:1.5; }}
  @media (max-width:600px) {{ .crow {{ grid-template-columns:1fr; gap:3px; }} }}
  .stat .v.buy {{ color:var(--buy); }} .stat .v.sell {{ color:var(--sell); }}
  .stat .sub {{ color:var(--muted); font-size:11px; margin-top:2px; overflow-wrap:anywhere; }}
  /* Option E — verdict strip + plan summary (modal overview) */
  .msparkwrap {{ background:var(--inset); border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin-bottom:14px; }}
  .msparkwrap:empty {{ display:none; }}
  .mverdict {{ display:flex; align-items:center; gap:14px; background:var(--inset); border:1px solid var(--line);
    border-radius:12px; padding:13px 16px; margin-bottom:14px; }}
  .mverdict:empty, .mplan-strip:empty {{ display:none; }}
  .mv-badge {{ font-size:12px; font-weight:700; letter-spacing:.03em; text-transform:uppercase; padding:4px 12px;
    border-radius:999px; border:1px solid var(--line); color:var(--txt2); white-space:nowrap; }}
  .mv-badge.up {{ color:var(--buy); border-color:color-mix(in srgb,var(--buy) 40%,var(--line)); }}
  .mv-badge.dn {{ color:var(--sell); border-color:color-mix(in srgb,var(--sell) 40%,var(--line)); }}
  .mv-badge.warn {{ color:var(--warn); border-color:color-mix(in srgb,var(--warn) 40%,var(--line)); }}
  .mv-meter {{ flex:1 1 auto; height:6px; border-radius:4px; background:var(--card); overflow:hidden; }}
  .mv-meter i {{ display:block; height:100%; border-radius:4px; }}
  .mv-score {{ font-size:13px; font-weight:600; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .mplan-strip {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:16px; }}
  .pt {{ background:var(--inset); border:1px solid var(--line); border-radius:12px; padding:12px 13px; }}
  .pt-l {{ font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }}
  .pt-v {{ font-size:17px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
  .pt-v.up {{ color:var(--buy); }} .pt-v.dn {{ color:var(--sell); }}
  @media (max-width:560px) {{ .mplan-strip {{ grid-template-columns:1fr 1fr; }} }}
  .stat.buy::before {{ background:var(--buy); opacity:1; }}
  .stat.sell::before {{ background:var(--sell); opacity:1; }}
  .stat.gold::before {{ background:var(--accent); opacity:1; }}
  /* tint the plan tiles by tier: entry neutral, target green, stop red, R:R gold */
  #mPlan .stat:nth-child(2)::before {{ background:var(--sell); opacity:1; }}
  #mPlan .stat:nth-child(3)::before {{ background:var(--buy); opacity:1; }}
  #mPlan .stat:nth-child(4)::before {{ background:var(--accent); opacity:1; }}
  /* target scenarios (conservative / base / stretch) */
  .scen {{ grid-column:1/-1; margin-top:10px; border-top:1px solid var(--line); padding-top:13px; }}
  .scen-h {{ display:flex; align-items:center; gap:7px; font-size:12.5px; font-weight:800; margin-bottom:10px;
    flex-wrap:wrap; }}
  .scen-h span {{ font-weight:400; color:var(--muted); text-transform:none; overflow-wrap:anywhere; }}
  .scen-h .ico {{ color:var(--accent); }}
  .scen-row {{ padding:11px 14px; border:1px solid var(--line); border-radius:12px; margin-bottom:8px;
    background:var(--inset); }}
  .scen-row.higherodds {{ border-left:3px solid var(--buy); }}
  .scen-row.medium {{ border-left:3px solid var(--muted); }}
  .scen-row.lowerodds {{ border-left:3px solid #6e7681; }}
  .scen-top {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap; min-width:0; }}
  .scen-top > b {{ min-width:0; overflow-wrap:anywhere; }}
  .scen-px {{ font-family:var(--mono); font-variant-numeric:tabular-nums; font-weight:800; white-space:nowrap; }}
  .scen-px em {{ color:var(--muted); font-style:normal; font-weight:500; font-size:12px; }}
  .scen-why {{ color:var(--muted); font-size:12px; margin-top:4px; line-height:1.45; overflow-wrap:anywhere; }}
  /* signal-input detail cards (modal Signals sub-tab) */
  .sigdet {{ background:var(--inset); border:1px solid var(--line); border-left:3px solid var(--line);
    border-radius:9px; padding:10px 12px; margin-bottom:9px; }}
  .sigdet.good {{ border-left-color:var(--buy); }}
  .sigdet.bad {{ border-left-color:var(--sell); }}
  .sigdet.warn {{ border-left-color:#6e7681; }}
  .sigdet-h {{ display:flex; align-items:baseline; gap:8px; font-size:13.5px; flex-wrap:wrap; }}
  .sigdet-v {{ margin-left:auto; font-weight:700; font-variant-numeric:tabular-nums; font-size:12.5px; }}
  .sigdet-why {{ color:var(--muted); font-size:12px; margin-top:4px; line-height:1.45; }}
  /* news-driven ideas (Market news tab) */
  .nideas {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:10px; margin-top:10px; }}
  .nidea {{ background:var(--inset); border:1px solid var(--line); border-left:3px solid var(--line);
    border-radius:9px; padding:10px 12px; }}
  .nidea.buy {{ border-left-color:var(--buy); }}
  .nidea.sell {{ border-left-color:var(--sell); }}
  .nidea-top {{ display:flex; align-items:baseline; gap:8px; font-size:14px; flex-wrap:wrap; }}
  .nidea-conf {{ margin-left:auto; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .nidea-why {{ font-size:12.5px; margin-top:4px; line-height:1.45; }}
  .nidea-src {{ color:var(--muted); font-size:11px; margin-top:4px; font-style:italic; }}
  /* system status tab */
  .sysgrid {{ margin-top:8px; }}
  .sysrow {{ display:flex; align-items:center; gap:10px; padding:7px 2px; border-bottom:1px solid var(--line); }}
  .sysrow:last-child {{ border-bottom:0; }}
  .sysname {{ font-size:13px; font-weight:600; }}
  .sysnote {{ color:var(--muted); font-size:11.5px; margin-left:auto; text-align:right; }}
  .syspill {{ font-size:10.5px; font-weight:800; padding:2px 8px; border-radius:999px; white-space:nowrap;
    margin-left:10px; letter-spacing:.03em; }}
  .syspill.on {{ color:var(--buy); background:color-mix(in srgb, var(--buy) 14%, transparent); }}
  .syspill.off {{ color:var(--muted); background:var(--inset); }}
  .deskread {{ background:var(--inset); border:1px solid var(--line); border-left:3px solid rgba(255,255,255,.16);
    border-radius:12px; padding:13px 15px; font-size:14px; line-height:1.6; margin:12px 0;
    overflow-wrap:anywhere; }}
  .deskread.ai-read {{ border-left-color:var(--muted); background:var(--inset); }}
  .deskread p {{ margin:0 0 8px; }} .deskread p:last-child {{ margin-bottom:0; }}
  .convbadge {{ font-size:12.5px; font-weight:800; padding:3px 11px; border-radius:999px; color:#fff;
    letter-spacing:.02em; }}
  .conv-High {{ background:var(--buy); }} .conv-Medium {{ background:#9e6a1e; }}
  .conv-Low {{ background:var(--sell); }}
  .checks {{ list-style:none; padding:0; margin:8px 0; display:flex; flex-direction:column; gap:6px; }}
  .checks li {{ display:flex; gap:10px; align-items:flex-start; padding:10px 13px; min-width:0;
    background:var(--inset); border:1px solid var(--line); border-radius:12px; font-size:13px; }}
  .checks li.pass {{ border-left:3px solid var(--buy); }}
  .checks li.warn {{ border-left:3px solid var(--muted); }}
  .checks li.fail {{ border-left:3px solid var(--sell); }}
  .checks li > span:last-child {{ min-width:0; overflow-wrap:anywhere; line-height:1.45; }}
  .checks .ic {{ flex:0 0 16px; display:inline-flex; }}
  .checks .pass .ic {{ color:var(--buy); }} .checks .warn .ic {{ color:var(--muted); }}
  .checks .fail .ic {{ color:var(--sell); }}
  .checks .ck-l {{ font-weight:600; }} .checks .ck-n {{ color:var(--muted); }}
  .chartkey {{ color:var(--muted); font-size:12px; margin-top:8px; line-height:1.6; }}
  .reasons li {{ font-size:14px; line-height:1.5; }}
  .readout {{ display:flex; align-items:baseline; gap:12px; min-height:30px; margin:2px 0 10px; }}
  .readout .rprice {{ font-size:24px; font-weight:700; }}
  .readout .rchg {{ font-size:15px; font-weight:600; }}
  .readout .rdate {{ color:var(--muted); font-size:13px; margin-left:auto; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:7px; }}
  .chip {{ font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px;
    border:1px solid var(--line); }}
  .chip.bull {{ background:color-mix(in srgb, var(--buy) 14%, transparent); color:var(--buy);
    border-color:color-mix(in srgb, var(--buy) 32%, transparent); }}
  .chip.bear {{ background:color-mix(in srgb, var(--sell) 14%, transparent); color:var(--sell);
    border-color:color-mix(in srgb, var(--sell) 32%, transparent); }}
  .chip.neutral {{ background:var(--inset); color:var(--txt2); border-color:var(--line); }}
  .chip.mini {{ font-size:10.5px; padding:1px 7px; }}
  /* primary nav: brand-aligned horizontal scroller flanked by arrow controls */
  .tabsbar {{ display:flex; align-items:stretch; gap:2px; }}
  .tabscroll {{ flex:none; display:none; align-items:center; justify-content:center;
    width:28px; padding:0 0 2px; background:none; border:none; cursor:pointer; color:var(--muted);
    font-size:20px; line-height:1; border-bottom:2px solid transparent;
    transition:color .15s ease, opacity .15s ease; }}
  .tabscroll:hover {{ color:var(--txt); }}
  .tabsbar.scrollable .tabscroll {{ display:inline-flex; }}
  .tabsbar.at-start .tabscroll.left {{ opacity:0; pointer-events:none; }}
  .tabsbar.at-end .tabscroll.right {{ opacity:0; pointer-events:none; }}
  .tabs {{ flex:1 1 auto; min-width:0; display:flex; gap:1px; flex-wrap:nowrap; margin:0; padding:0;
    overflow-x:auto; overflow-y:hidden; scrollbar-width:none; -ms-overflow-style:none;
    -webkit-overflow-scrolling:touch; scroll-snap-type:x proximity; scroll-behavior:smooth; }}
  .tabsbar.scrollable .tabs {{
    -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 calc(100% - 18px),transparent 100%);
            mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 calc(100% - 18px),transparent 100%); }}
  .tabsbar.scrollable.at-start .tabs {{ -webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 18px),transparent 100%);
            mask-image:linear-gradient(90deg,#000 calc(100% - 18px),transparent 100%); }}
  .tabsbar.scrollable.at-end .tabs {{ -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 100%);
            mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 100%); }}
  .tabs::-webkit-scrollbar {{ height:0; display:none; }}
  .tabs button {{ background:none; border:none; color:var(--muted); font-size:14px;
    font-weight:600; padding:9px 14px 11px; cursor:pointer; white-space:nowrap; flex:none;
    letter-spacing:-.004em; border-bottom:2px solid transparent; scroll-snap-align:start; }}
  .tabs button.on {{ color:var(--txt); border-bottom-color:var(--accent); }}
  .tabs button:hover {{ color:var(--txt); }}
  .ctlbtn:hover, .ctlgrp button:hover {{ color:var(--txt); background:var(--hover); }}
  .page {{ display:none; }} .page.on {{ display:block; }}
  .secthead {{ font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase;
    letter-spacing:.05em; margin:22px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .secthead:first-child {{ margin-top:4px; }}
  .ovbox {{ background:var(--card); border:1px solid var(--hud-edge); border-radius:6px;
    padding:14px 16px; margin-bottom:22px; }}
  .ovhead {{ font-weight:600; font-size:13px; margin-bottom:8px; text-transform:uppercase; letter-spacing:.08em; }}
  .ovwrap {{ display:flex; gap:14px; align-items:stretch; }}
  .ovchart {{ flex:1; min-width:0; }}
  .ovboard {{ width:150px; max-height:300px; overflow-y:auto; border-left:1px solid var(--line); padding-left:10px; }}
  .ovrow {{ display:flex; align-items:center; gap:6px; font-size:12px; padding:3px 2px; cursor:pointer; color:var(--muted); border-radius:4px; }}
  .ovrow:hover {{ color:var(--txt); background:var(--inset); }}
  .ovrow.on {{ color:var(--txt); font-weight:700; }}
  .ovdot {{ width:8px; height:8px; border-radius:50%; flex:0 0 8px; }}
  .ovsym {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .viewctl {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:16px; }}
  .regime {{ border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin:14px 0 4px;
    display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; }}
  .regime .rlabel {{ font-weight:700; font-size:15px; }}
  .regime .rdetail {{ color:var(--txt); font-size:13px; }}
  .regime .rnote {{ color:var(--muted); font-size:12px; width:100%; }}
  .secrow {{ display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }}
  .secname {{ width:130px; color:var(--txt); }}
  .secbar {{ flex:1; height:8px; background:var(--inset); border-radius:5px; overflow:hidden; }}
  .secfill {{ height:100%; background:linear-gradient(90deg,#388bfd,#2ea043); }}
  .secpct {{ width:90px; text-align:right; color:var(--muted); }}
  .track {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px; margin:18px 0; }}
  .track h2 {{ margin:0 0 4px; }}
  .trackstats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr));
    gap:10px; margin:12px 0; }}
  .trackrec {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
  .trackrec th, .trackrec td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }}
  .trackrec th {{ color:var(--muted); font-weight:600; }}
  .orb-sec {{ margin-top:14px; border-top:1px solid var(--line); }}
  .orb-sec > summary {{ cursor:pointer; list-style:none; padding:12px 2px 6px; font-size:14px;
    font-weight:600; color:var(--txt); display:flex; align-items:center; gap:7px; }}
  .orb-sec > summary::-webkit-details-marker {{ display:none; }}
  .orb-sec > summary::before {{ content:'▸'; color:var(--accent); font-size:11px; transition:transform .15s; }}
  .orb-sec[open] > summary::before {{ transform:rotate(90deg); }}
  .win {{ color:var(--buy); }} .loss {{ color:var(--sell); }} .exp {{ color:var(--muted); }}
  /* shared table style for the intelligence/data panels (was previously unstyled) */
  .tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .tbl th, .tbl td {{ text-align:left; padding:9px 11px; border-bottom:1px solid var(--line); vertical-align:middle; font-variant-numeric:tabular-nums; }}
  .tbl thead th {{ color:var(--muted); font-weight:600; font-size:11px; white-space:nowrap;
    text-transform:uppercase; letter-spacing:.05em; position:sticky; top:0; background:var(--card); }}
  .tbl tbody tr:hover {{ background:var(--hover); }}
  .tbl tbody tr:last-child td {{ border-bottom:none; }}
  .tbl .buy {{ color:var(--buy); }} .tbl .sell {{ color:var(--sell); }}
  details.ovbox summary {{ list-style:none; }}
  details.ovbox summary::-webkit-details-marker {{ display:none; }}
  .chartctl {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:10px; }}
  .ctlgrp {{ display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  .ctlgrp button {{ background:var(--card); color:var(--muted); border:none; padding:5px 11px;
    font-size:13px; cursor:pointer; border-right:1px solid var(--line); }}
  .ctlgrp button:last-child {{ border-right:none; }}
  .ctlgrp button.on {{ background:var(--hold); color:#fff; }}
  .ctltog {{ font-size:13px; color:var(--muted); cursor:pointer; }}
  .ctlbtn {{ background:var(--card); color:var(--muted); border:1px solid var(--line);
    border-radius:8px; padding:5px 11px; font-size:13px; cursor:pointer; }}
  .method {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:4px 18px; margin:18px 0 8px; }}
  .method summary {{ cursor:pointer; font-weight:700; font-size:15px; padding:12px 0;
    list-style:none; }}
  .method summary::-webkit-details-marker {{ display:none; }}
  .method summary:before {{ content:'▸ '; color:var(--hold); }}
  .method[open] summary:before {{ content:'▾ '; }}
  .method h4 {{ margin:16px 0 6px; font-size:14px; color:var(--txt); }}
  .method p, .method li {{ font-size:14px; color:var(--txt2); line-height:1.6; }}
  .method ol, .method ul {{ padding-left:20px; margin:6px 0; }}
  .method .pill {{ display:inline-block; background:var(--inset); border:1px solid var(--line);
    border-radius:6px; padding:1px 7px; font-size:13px; color:var(--txt); }}
  /* ---- theme toggle ---- */
  .themebtn {{ background:var(--card); color:var(--muted);
    border:1px solid var(--line); border-radius:8px; padding:6px 12px; font-size:13px; cursor:pointer;
    box-shadow:var(--shadow); }}
  .themebtn:hover {{ color:var(--txt); }}
  /* ---- accent colour picker ---- */
  .accent-wrap {{ position:relative; display:inline-block; }}
  .accent-pop {{ position:absolute; right:0; top:calc(100% + 6px); background:var(--card);
    border:1px solid var(--line); border-radius:10px; padding:11px; display:flex; flex-wrap:wrap;
    gap:7px; width:184px; z-index:60; box-shadow:var(--shadow-lg); }}
  .accent-pop[hidden] {{ display:none; }}
  .accent-pop .acsw {{ width:26px; height:26px; border-radius:50%; border:2px solid transparent;
    cursor:pointer; padding:0; }}
  .accent-pop .acsw.on {{ border-color:var(--txt); }}
  .accent-pop .accustom {{ display:flex; align-items:center; gap:6px; font-size:11px;
    color:var(--muted); width:100%; margin-top:2px; }}
  .accent-pop .accustom input {{ width:26px; height:26px; padding:0; border:none; background:none; cursor:pointer; }}
  .accent-pop .acreset {{ width:100%; font-size:11.5px; color:var(--muted); background:var(--inset);
    border:1px solid var(--line); border-radius:7px; padding:6px; cursor:pointer; }}
  .accent-pop .acreset:hover {{ color:var(--txt); }}
  /* ---- sidebar + top-tab shell ---- */
  .shell {{ display:flex; gap:0; align-items:flex-start; }}
  .sidebar {{ width:162px; flex:0 0 162px; position:sticky; top:8px; display:flex; flex-direction:column;
    gap:3px; padding:8px; background:var(--glass-bg); backdrop-filter:blur(var(--glass-blur)); -webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-bd); border-radius:14px; box-shadow:var(--shadow-lg), inset 0 1px 0 rgba(255,255,255,.06);
    max-height:calc(100vh - 16px); overflow-y:auto; overflow-x:hidden; scrollbar-width:thin; }}
  .sidebar::-webkit-scrollbar {{ width:6px; }}
  .sidebar::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:3px; }}
  /* ---- sidebar active-signals list ---- */
  .side-sig {{ margin-top:9px; padding-top:9px; border-top:1px solid var(--line); }}
  .side-h {{ display:flex; align-items:center; gap:6px; font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:.06em; color:var(--muted); padding:2px 4px 6px; }}
  .side-h svg {{ width:12px; height:12px; flex:0 0 auto; }}
  .side-sig-list {{ display:flex; flex-direction:column; gap:1px; }}
  .side-sig-row {{ display:flex; align-items:center; gap:7px; padding:5px 5px; border-radius:8px;
    cursor:pointer; }}
  .side-sig-row:hover {{ background:var(--hover); }}
  .side-sig-row .ss-sym {{ font-size:11.5px; font-weight:700; color:var(--txt); font-family:var(--mono);
    letter-spacing:.02em; flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; }}
  .side-sig-row .ss-conv {{ font-size:11.5px; font-weight:800; font-variant-numeric:tabular-nums;
    flex:0 0 auto; }}
  .side-sig-row .ss-dir {{ display:inline-flex; flex:0 0 auto; }}
  .side-sig-row .ss-dir svg {{ width:12px; height:12px; }}
  /* ---- sidebar alpha / status footer ---- */
  .side-foot {{ margin-top:9px; padding:9px 10px; border-radius:11px; background:var(--inset);
    border:1px solid var(--line); box-shadow:inset 0 1px 0 rgba(255,255,255,.05); }}
  .side-foot .sf-l {{ font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
    color:var(--muted); margin-bottom:2px; }}
  .side-foot .sf-v {{ font-size:17px; font-weight:800; font-variant-numeric:tabular-nums; line-height:1.1; }}
  .side-foot .sf-sub {{ font-size:10.5px; color:var(--muted); margin-top:3px; font-variant-numeric:tabular-nums; }}
  .sidebar button {{ display:flex; align-items:center; gap:10px; text-align:left; background:none;
    border:none; border-left:2px solid transparent; color:var(--muted); font-size:12px; font-weight:600; padding:9px 11px; border-radius:9px;
    cursor:pointer; text-transform:uppercase; letter-spacing:.06em; }}
  .sidebar button svg {{ width:15px; height:15px; flex:0 0 auto; }}
  .sidebar button:hover {{ background:var(--hover); color:var(--txt); }}
  .sidebar button.on {{ background:linear-gradient(90deg,color-mix(in srgb,var(--accent) 18%,transparent),transparent);
    color:var(--accent); box-shadow:inset 2px 0 0 var(--accent), inset 0 0 22px color-mix(in srgb,var(--accent) 9%,transparent); }}
  .maincol {{ flex:1; min-width:0; padding-left:16px; }}
  .toptabs {{ display:flex; gap:3px; flex-wrap:wrap; border-bottom:1px solid var(--line); margin:0 0 14px; }}
  .toptabs button {{ background:none; border:none; border-bottom:2px solid transparent; color:var(--muted);
    font-size:12px; font-weight:600; padding:8px 13px; margin-bottom:-1px; cursor:pointer;
    text-transform:uppercase; letter-spacing:.07em; }}
  .toptabs button:hover {{ color:var(--txt); }}
  .toptabs button.on {{ color:var(--accent); border-bottom-color:var(--accent); }}
  .toptabs button.on::before {{ content:"\\25B8  "; }}
  /* HUD corner ticks on tiles + panels (re-skin with the accent) */
  .bt, .kpi, .ovbox {{ position:relative; }}
  .bt::after, .kpi::after, .ovbox::after {{ content:""; position:absolute; top:-1px; right:-1px;
    width:11px; height:11px; border-top:2px solid var(--accent); border-right:2px solid var(--accent);
    border-top-right-radius:6px; opacity:.6; pointer-events:none; }}
  .bt::before, .kpi::before {{ content:""; position:absolute; bottom:-1px; left:-1px;
    width:11px; height:11px; border-bottom:2px solid var(--accent); border-left:2px solid var(--accent);
    border-bottom-left-radius:6px; opacity:.35; pointer-events:none; }}
  @media (max-width:760px) {{
    .shell {{ flex-direction:column; }}
    .sidebar {{ flex-direction:row; width:auto; flex:none; position:sticky; top:0; z-index:40;
      overflow-x:auto; overflow-y:visible; max-height:none; background:var(--bg); border-bottom:1px solid var(--line); padding:6px 0; gap:2px; }}
    .sidebar button {{ white-space:nowrap; padding:8px 12px; }}
    .side-sig, .side-foot {{ display:none; }}
    .maincol {{ padding-left:0; border-left:none; width:100%; }}
  }}
  /* ---- featured chart panel + watchlist ---- */
  .featured {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px;
    margin:6px 0 18px; box-shadow:var(--shadow); }}
  .feat-grid {{ display:grid; grid-template-columns:1fr 256px; gap:16px; }}
  @media (max-width:840px) {{ .feat-grid {{ grid-template-columns:1fr; }} .feat-watch {{ max-height:220px; }} }}
  .feat-wtitle {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
    font-weight:700; margin-bottom:6px; }}
  .feat-watch {{ overflow-y:auto; max-height:560px; border-left:1px solid var(--line); padding-left:12px; }}
  .wl {{ display:flex; align-items:center; gap:9px; padding:6px 8px; border-radius:7px; cursor:pointer; }}
  .wl:hover {{ background:var(--line); }}
  .wl.on {{ background:color-mix(in srgb, var(--accent) 16%, transparent); }}
  .wl-logo {{ position:relative; flex:0 0 auto; width:28px; height:28px; border-radius:6px; color:#fff;
    font-size:10px; font-weight:800; display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .wl-logo img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#fff; }}
  .wl-main {{ display:flex; flex-direction:column; min-width:0; flex:1 1 auto; }}
  .wl-sym {{ font-weight:700; font-size:13px; line-height:1.2; }}
  .wl-name {{ color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .wl-r {{ display:flex; flex-direction:column; align-items:flex-end; flex:0 0 auto; }}
  .wl-px {{ font-variant-numeric:tabular-nums; font-size:12px; }}
  .wl-chg {{ font-variant-numeric:tabular-nums; font-size:11px; }}
  /* ---- TradeChart component ---- */
  .tc {{ width:100%; }}
  .tc-bar {{ display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; margin-bottom:8px; }}
  .tc-seg {{ display:inline-flex; align-items:center; gap:2px; background:var(--bg); border:1px solid var(--line);
    border-radius:8px; padding:2px; }}
  .tc-seglab {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
    padding:0 6px; font-weight:700; }}
  .tc-seg button {{ background:none; border:none; color:var(--muted); font-size:12px; padding:4px 9px;
    border-radius:6px; cursor:pointer; font-variant-numeric:tabular-nums; }}
  .tc-seg button:hover {{ color:var(--txt); }}
  .tc-seg button.on {{ background:var(--accent); color:#fff; }}
  .tc-cmp {{ background:var(--bg); border:1px solid var(--line); border-radius:7px; color:var(--txt);
    padding:5px 9px; font-size:12px; }}
  .tc-clr {{ background:var(--bg); border:1px solid var(--line); color:var(--muted); border-radius:7px;
    padding:5px 10px; font-size:12px; cursor:pointer; }}
  .tc-readout {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; min-height:24px; margin-bottom:4px; }}
  .tc-readout .tc-sym {{ font-weight:800; font-size:15px; }}
  .tc-readout .tc-price-lg {{ font-weight:700; font-size:18px; font-variant-numeric:tabular-nums; }}
  .tc-readout .tc-chg {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .tc-readout .tc-ohlc {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
  .tc-readout .tc-date {{ color:var(--muted); font-size:12px; margin-left:auto; }}
  .tc-wrap {{ position:relative; height:380px; }}
  .tc-compact .tc-wrap {{ height:300px; }}
  /* TradingView widget containers */
  .tv-wrap {{ position:relative; height:520px; width:100%; }}
  .tv-wrap.tv-compact {{ height:420px; }}
  @media (max-width:760px) {{ .tv-wrap {{ height:380px; }} }}
  .tc-sub {{ position:relative; height:84px; margin-top:6px; }}
  .tc-sublab {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:2px; }}
  .tc-key {{ color:var(--muted); font-size:12px; margin-top:8px; line-height:1.6; }}
  .tc-chip {{ display:inline-block; background:var(--line); border-radius:10px; padding:1px 8px; cursor:pointer;
    font-size:11px; color:var(--txt); }}
  /* ---- app shell ---- */
  /* unified sticky header: brand + status on top, primary tab nav directly beneath */
  .appbar {{ position:sticky; top:0; z-index:30; margin:0 0 18px;
    padding-top:env(safe-area-inset-top);
    background:color-mix(in srgb, var(--bg) 86%, transparent);
    backdrop-filter:saturate(1.4) blur(12px); -webkit-backdrop-filter:saturate(1.4) blur(12px);
    border-bottom:1px solid var(--line); }}
  .appbar-top {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:13px 2px 9px; }}
  .brand {{ display:flex; align-items:center; gap:10px; font-family:var(--mono); font-weight:700;
    font-size:15px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); }}
  .brand-mark {{ display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px;
    border-radius:9px; color:#fff; font-size:15px; box-shadow:0 2px 8px color-mix(in srgb, var(--accent) 45%, transparent);
    background:linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 50%, #5ed6a6)); }}
  .appbar-right {{ display:flex; align-items:center; gap:10px; }}
  .livepill {{ font-size:12px; color:var(--muted); }}
  /* ---- app-bar centre cluster: search + clock + regime ---- */
  .appbar-mid {{ display:flex; align-items:center; gap:12px; flex:1 1 auto; min-width:0;
    justify-content:flex-start; margin:0 6px; }}
  .appsearch {{ display:flex; align-items:center; gap:6px; background:var(--glass-bg);
    backdrop-filter:blur(var(--glass-blur)); -webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-bd); border-radius:9px; padding:5px 10px; color:var(--muted);
    box-shadow:var(--shadow); min-width:0; max-width:220px; }}
  .appsearch:focus-within {{ border-color:color-mix(in srgb,var(--accent) 55%,transparent);
    color:var(--txt); box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 18%,transparent); }}
  .appsearch svg {{ width:14px; height:14px; flex:0 0 auto; }}
  .appsearch input {{ background:none; border:none; outline:none; color:var(--txt); font-size:12.5px;
    font-family:var(--mono); letter-spacing:.03em; width:118px; min-width:0; padding:0; }}
  .appsearch input::placeholder {{ color:var(--muted); text-transform:none; letter-spacing:0; }}
  .appclock {{ font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums;
    color:var(--muted); white-space:nowrap; flex:0 0 auto; }}
  .regime-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:700;
    text-transform:uppercase; letter-spacing:.05em; white-space:nowrap; flex:0 0 auto;
    padding:4px 10px; border-radius:999px; border:1px solid var(--glass-bd); background:var(--glass-bg);
    backdrop-filter:blur(var(--glass-blur)); -webkit-backdrop-filter:blur(var(--glass-blur)); }}
  .regime-pill svg {{ flex:0 0 auto; }}
  @media (max-width:1080px) {{ .appclock {{ display:none; }} }}
  @media (max-width:920px) {{ .appsearch {{ display:none; }} }}
  @media (max-width:640px) {{ .appbar-mid {{ display:none; }} }}
  .subhead {{ color:var(--muted); font-size:12.5px; margin:0 0 16px; }}
  .stale-banner {{ border-radius:10px; padding:10px 14px; margin:0 0 12px; font-size:13px; line-height:1.45;
    border:1px solid; }}
  .stale-banner.red {{ background:color-mix(in srgb, var(--sell) 14%, transparent);
    border-color:var(--sell); color:var(--sell); }}
  .stale-banner.amber {{ background:color-mix(in srgb, #6e7681 12%, transparent);
    border-color:color-mix(in srgb, #6e7681 45%, transparent); color:var(--txt); }}
  /* ---- KPI summary strip ---- */
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:0 0 18px; }}
  .kpi {{ background:var(--card); border:1px solid var(--hud-edge); border-radius:6px; padding:12px 14px; }}
  .kpi.hero {{ grid-column:span 2; border-top:2px solid var(--accent);
    display:flex; flex-direction:column; justify-content:center; }}
  .kpi.hero .kpi-v {{ font-size:34px; }}
  @media (max-width:600px) {{ .kpi.hero {{ grid-column:span 2; }} .kpi.hero .kpi-v {{ font-size:26px; }} }}
  /* ---- bento home grid ---- */
  .bento {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); grid-auto-flow:row dense;
    gap:10px; margin:0 0 18px; align-items:start; }}
  .bento .bt {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:18px 20px; min-width:0; display:flex; flex-direction:column; }}
  .bento .bt.hero {{ grid-column:span 2; }}
  .bento .bt.wide {{ grid-column:1 / -1; }}
  .bt-l {{ font-size:10px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-weight:600; margin-bottom:3px; }}
  .bt-v {{ font-family:var(--mono); font-weight:600; font-size:22px; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
  .bt-v.buy {{ color:var(--buy); }} .bt-v.sell {{ color:var(--sell); }} .bt-v.warn {{ color:var(--warn); }}
  .bt-sub {{ font-size:11px; color:var(--muted); margin-top:3px; }}
  .bt-chip {{ display:inline-block; margin-top:9px; font-size:11px; padding:3px 10px; border-radius:999px;
    background:color-mix(in srgb,var(--accent) 16%,transparent); color:var(--accent); align-self:flex-start; }}
  .bt-body {{ font-size:14px; line-height:1.7; color:var(--txt2); max-width:74ch; }}
  .bt-body p {{ margin:0 0 14px; }}
  .bt-body p:last-child {{ margin-bottom:0; }}
  .bt-body p > strong:first-child {{ display:block; color:var(--txt); font-weight:600; letter-spacing:-.01em; margin-bottom:3px; }}
  .bt.wide {{ padding:22px 24px; }}
  .bt.wide .bt-list {{ max-width:74ch; line-height:1.7; }}
  /* market brief — 2x2 quadrant panels + coloured sector strip */
  .mb-quad {{ display:grid; grid-template-columns:1fr 1fr; grid-auto-rows:1fr; gap:12px; }}
  .mbp {{ background:var(--inset); border-radius:12px; padding:15px 17px; min-width:0; }}
  .mbp-h {{ font-size:13px; font-weight:600; display:flex; align-items:center; gap:7px; margin-bottom:6px; color:var(--txt); }}
  .mbp-h .ico {{ color:var(--muted); }}
  .mbp-b {{ font-size:13px; line-height:1.55; color:var(--txt2); }}
  .mbp-b strong {{ color:var(--txt); font-weight:600; }}
  .mbsecs {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
  .mbsec {{ font-size:12px; background:var(--inset); color:var(--txt2); padding:4px 11px; border-radius:999px;
    display:inline-flex; align-items:center; gap:6px; }}
  .mbsec b {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .mbsec b.up {{ color:var(--buy); }} .mbsec b.dn {{ color:var(--sell); }} .mbsec b.flat {{ color:var(--muted); }}
  .mbp-b .mbhl.up {{ color:var(--buy); font-weight:500; }}
  .mbp-b .mbhl.dn {{ color:var(--sell); font-weight:500; }}
  @media (max-width:640px) {{ .mb-quad {{ grid-template-columns:1fr; }} }}

  /* Premium selling — regime banner + verdict pills */
  .ps-regime {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; background:var(--card);
    border:1px solid var(--line); border-left-width:3px; border-radius:12px; padding:14px 18px; margin-bottom:16px; }}
  .ps-reg-l {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .ps-reg-v {{ font-size:16px; font-weight:700; letter-spacing:.01em; }}
  .ps-reg-n {{ font-size:12.5px; color:var(--muted); }}
  .ps-verd {{ display:inline-block; font-size:11px; font-weight:700; letter-spacing:.03em;
    border:1px solid; border-radius:999px; padding:2px 10px; }}
  /* ===== Engine brain — visual pipeline ===== */
  .brain {{ margin-top:8px; }}
  .bstage {{ display:grid; grid-template-columns:150px minmax(0,1fr); gap:20px; align-items:start; }}
  .bstage-h {{ display:flex; align-items:center; gap:12px; position:sticky; }}
  .bstage-num {{ font-size:26px; font-weight:800; color:var(--muted); opacity:.5;
    font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
  .bstage-nm {{ font-size:15px; font-weight:700; color:var(--txt); }}
  .bstage-sub {{ font-size:11.5px; color:var(--muted); }}
  .bstage-nodes {{ display:flex; flex-wrap:wrap; gap:12px; }}
  .bn {{ width:190px; background:var(--card); border:1px solid var(--line); border-radius:13px;
    padding:14px 15px; cursor:help; transition:border-color .15s ease, transform .15s ease; position:relative; }}
  .bn:hover {{ border-color:rgba(255,255,255,.28); transform:translateY(-2px); }}
  .bn-go {{ cursor:pointer; }}
  .bn-arrow {{ opacity:0; margin-left:5px; color:var(--muted); transition:opacity .15s ease; }}
  .bn-go:hover .bn-arrow {{ opacity:1; }}
  .bn-go:hover {{ border-color:color-mix(in srgb,var(--accent) 45%,var(--line)); }}
  .bn-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:9px; }}
  .bn-ic {{ color:var(--txt2); display:inline-flex; }}
  .bn-nm {{ font-size:14px; font-weight:650; color:var(--txt); }}
  .bn-desc {{ font-size:12px; color:var(--muted); line-height:1.45; margin-top:3px; }}
  .bn-badge {{ font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
    padding:2px 7px; border-radius:999px; border:1px solid; }}
  .bn-b-live {{ color:var(--buy); border-color:color-mix(in srgb,var(--buy) 40%,transparent); }}
  .bn-b-new {{ color:#6ea8ff; border-color:color-mix(in srgb,#6ea8ff 45%,transparent); }}
  .bn-b-learn {{ color:var(--accent); border-color:color-mix(in srgb,var(--accent) 45%,transparent); }}
  .bn-b-gate {{ color:var(--sell); border-color:color-mix(in srgb,var(--sell) 40%,transparent); }}
  .bn-b-legacy {{ color:var(--muted); border-color:var(--line); }}
  .bn-b-planned {{ color:var(--muted); border-color:var(--line); border-style:dashed; }}
  .bn-b-advisory {{ color:var(--txt2); border-color:var(--line); }}
  .bn-gate {{ border-left:2px solid color-mix(in srgb,var(--sell) 45%,var(--line)); }}
  .bn-learn {{ border-left:2px solid color-mix(in srgb,var(--accent) 55%,var(--line)); }}
  .bn-new {{ border-left:2px solid color-mix(in srgb,#6ea8ff 55%,var(--line)); }}
  .bconn {{ grid-column:1 / -1; height:26px; margin-left:74px; position:relative; }}
  .bconn::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:2px;
    background:linear-gradient(var(--line), var(--line)); }}
  .bflow {{ position:absolute; left:-3px; width:8px; height:8px; border-radius:50%;
    background:var(--accent); box-shadow:0 0 8px var(--accent); animation:bflow 2.2s linear infinite; opacity:.85; }}
  @keyframes bflow {{ 0% {{ top:-4px; opacity:0; }} 15% {{ opacity:.9; }} 85% {{ opacity:.9; }} 100% {{ top:26px; opacity:0; }} }}
  @media (prefers-reduced-motion: reduce) {{ .bflow {{ animation:none; opacity:0; }} }}
  .blegend {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }}
  .bloop {{ margin-top:12px; background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 8%,var(--card)),var(--card));
    border:1px solid color-mix(in srgb,var(--accent) 30%,var(--line)); border-radius:14px; padding:18px 20px; }}
  .bloop-h {{ display:flex; align-items:center; gap:8px; font-size:14px; font-weight:700; color:var(--txt); }}
  .bloop-h .bn-ic {{ color:var(--accent); }}
  .bloop-body {{ font-size:13px; line-height:1.55; color:var(--txt2); margin:9px 0 12px; max-width:760px; }}
  .bloop-steps {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; font-size:11.5px; color:var(--muted); }}
  .bloop-steps span {{ background:var(--inset); border-radius:999px; padding:3px 10px; }}
  .bloop-steps span:nth-child(even) {{ background:none; padding:0; }}
  .bloop-back {{ color:var(--accent) !important; background:none !important; font-weight:600; }}
  .badv {{ margin-top:20px; padding-top:18px; border-top:1px solid var(--line); }}
  @media (max-width:760px) {{ .bstage {{ grid-template-columns:1fr; gap:10px; }}
    .bconn {{ margin-left:20px; }} .bn {{ width:100%; }} }}
  /* Edge explorer — left-rail tabs + waffle heatmaps (Anthropic EconIndex style) */
  .an-lay {{ display:grid; grid-template-columns:200px 1fr; gap:28px; margin-top:8px; align-items:start; }}
  .an-rail {{ display:flex; flex-direction:column; position:sticky; top:66px; }}
  .an-tab {{ text-align:left; background:none; border:0; border-top:1px solid var(--line); color:var(--txt2);
    font-family:'Inter',-apple-system,sans-serif; font-size:14.5px; padding:13px 4px; cursor:pointer; transition:color .15s ease; }}
  .an-tab:first-child {{ border-top:0; }}
  .an-tab:hover {{ color:var(--txt); }}
  .an-tab.on {{ color:var(--txt); font-weight:600; background:rgba(255,255,255,.05); border-radius:8px;
    padding-left:12px; border-top-color:transparent; }}
  .an-report {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; margin-top:18px; }}
  .an-report-ic {{ color:var(--muted); }}
  .an-report-t {{ font-size:14px; font-weight:600; margin:8px 0 4px; color:var(--txt); }}
  .an-report-d {{ font-size:12.5px; color:var(--muted); line-height:1.5; }}
  .an-report-lk {{ font-size:12.5px; color:var(--txt2); margin-top:10px; cursor:pointer; }}
  .an-report-lk:hover {{ color:var(--txt); }}
  .an-view {{ display:none; }} .an-view.on {{ display:block; }}
  .an-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:16px; }}
  .an-card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; }}
  .an-jt {{ font-size:15px; font-weight:600; color:var(--txt); }}
  .an-js {{ font-size:12px; color:var(--muted); margin:2px 0 12px; }}
  .an-waffle {{ display:grid; grid-template-columns:repeat(10,1fr); gap:4px; max-width:220px; }}
  .an-sq {{ aspect-ratio:1; border-radius:3px; background:var(--inset); }}
  .an-sq.win {{ background:var(--buy); }}
  .an-legend {{ display:flex; gap:16px; font-size:12px; color:var(--muted); margin-bottom:14px; }}
  .an-legend i {{ display:inline-block; width:9px; height:9px; border-radius:2px; vertical-align:-1px; margin-right:5px; }}
  .an-legend i.win {{ background:var(--buy); }} .an-legend i.muted {{ background:var(--inset); }}
  .an-sects {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
  .an-sect {{ font-size:12px; background:var(--inset); border-radius:999px; padding:4px 11px; color:var(--txt2); }}
  .an-sect b {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .anup {{ color:var(--buy); }} .andn {{ color:var(--sell); }} .anmut {{ color:var(--muted); }}
  .an-empty {{ color:var(--muted); font-size:13px; padding:20px 0; }}
  @media (max-width:760px) {{ .an-lay {{ grid-template-columns:1fr; }}
    .an-rail {{ position:static; flex-direction:row; flex-wrap:wrap; gap:6px; }}
    .an-tab {{ border-top:0; }} }}
  /* system health panel */
  .shgrid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:16px; }}
  .shverd {{ color:var(--txt2); font-size:13px; margin:10px 0 0; line-height:1.6; }}
  @media (max-width:760px) {{ .shgrid {{ grid-template-columns:1fr; }} }}
  /* xAI-style 2x2 section showcase — quiet cards */
  .showcase {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:6px 0 4px; }}
  .sc-card {{ display:flex; flex-direction:column; background:var(--card); border:1px solid var(--line);
    border-radius:16px; padding:18px 20px; cursor:pointer; transition:border-color .15s ease; }}
  .sc-card:hover {{ border-color:rgba(255,255,255,.24); }}
  .sc-ic {{ color:var(--muted); }}
  .sc-t {{ font-size:16px; font-weight:600; letter-spacing:-.01em; margin-top:10px; }}
  .sc-s {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
  .sc-s .up {{ color:var(--buy); }} .sc-s .dn {{ color:var(--sell); }} .sc-s .warn {{ color:var(--warn); }}
  .sc-ex {{ font-size:12px; color:var(--muted); margin-top:12px; }}
  .sc-card:hover .sc-ex {{ color:var(--txt); }}
  @media (max-width:760px) {{ .showcase {{ grid-template-columns:1fr; }} }}
  .bt-list {{ margin:4px 0 0; padding-left:16px; font-size:13px; line-height:1.7; color:var(--txt2); }}
  .bt-logo {{ position:relative; width:26px; height:26px; border-radius:6px; flex:0 0 auto;
    display:inline-flex; align-items:center; justify-content:center; font-size:10px; font-weight:700;
    color:#fff; background:var(--accent); border:1px solid var(--hud-edge); overflow:hidden; }}
  .bt-logo img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#fff; }}
  @media (max-width:760px) {{
    .bento {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .bento .bt.hero {{ grid-column:span 2; grid-row:auto; }}
    .bento .bt.wide {{ grid-column:span 2; }}
  }}
  .kpi-l {{ font-size:10px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); font-weight:600; }}
  .kpi-v {{ font-size:24px; font-weight:600; margin-top:4px; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
  .kpi-v.buy {{ color:var(--buy); }} .kpi-v.sell {{ color:var(--sell); }} .kpi-v.warn {{ color:#6e7681; }}
  .kpi-sub {{ font-size:11px; color:var(--muted); margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  /* ---- redesigned signal card ---- */
  .card-top {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
  .card-mono {{ width:36px; height:36px; border-radius:9px; flex:0 0 auto; display:flex; align-items:center;
    justify-content:center; color:#fff; font-weight:800; font-size:12px; position:relative; overflow:hidden; }}
  .card-id {{ min-width:0; flex:1 1 auto; }}
  .card-id .s {{ font-size:16px; font-weight:800; line-height:1.15; }}
  .card-id .n {{ font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .card-age {{ font-size:10.5px; color:var(--muted); margin-top:3px; }}
  .card-age.fresh {{ color:var(--buy); font-weight:700; }}
  .card-age.held {{ color:var(--accent); font-weight:700; }}
  .card-px-row {{ display:flex; align-items:baseline; gap:10px; margin:6px 0 4px; }}
  .card-px {{ font-size:24px; font-weight:800; letter-spacing:-.015em; font-variant-numeric:tabular-nums; }}
  .card-day {{ font-size:13px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .conv-wrap {{ margin:12px 0 10px; }}
  .conv-row {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted);
    margin-bottom:5px; text-transform:uppercase; letter-spacing:.04em; font-weight:700; }}
  .conv-meter {{ height:6px; background:var(--inset); border-radius:4px; overflow:hidden; }}
  .conv-fill {{ height:100%; border-radius:4px; transition:width .3s ease; }}
  .card-stats {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 16px; font-size:12.5px; margin-top:6px; }}
  .card-stat {{ display:flex; justify-content:space-between; color:var(--muted); }}
  .card-stat b {{ color:var(--txt); font-weight:600; font-variant-numeric:tabular-nums; }}
  /* ---- signal card v2: direction pill, conviction meter, plan chips, meta chips ---- */
  .dir-pill {{ display:inline-flex; align-items:center; gap:5px; padding:3px 10px 3px 8px; border-radius:999px;
    font-size:11.5px; font-weight:800; letter-spacing:.02em; line-height:1; white-space:nowrap;
    border:1px solid currentColor; background:color-mix(in srgb,currentColor 14%,transparent); }}
  .dir-pill .ico {{ width:13px; height:13px; }}
  .dp-long, .dp-buy {{ color:var(--buy); }} .dp-short {{ color:var(--sell); }}
  .dp-hold {{ color:var(--hold); }} .dp-watch {{ color:var(--watch); }}
  .dp-exit {{ color:var(--exit); }} .dp-avoid {{ color:var(--avoid); }} .dp-flat {{ color:var(--flat); }}
  .conv2 {{ display:flex; align-items:center; gap:12px; margin:12px 0 11px; }}
  .conv2-ring {{ flex:0 0 auto; position:relative; width:52px; height:52px; }}
  .conv2-ring svg {{ transform:rotate(-90deg); width:52px; height:52px; }}
  .conv2-ring .cv {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-weight:700; font-size:14px; font-variant-numeric:tabular-nums; }}
  .conv2-meta {{ min-width:0; flex:1 1 auto; }}
  .conv2-lab {{ font-size:10px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-weight:700; }}
  .conv2-tier {{ font-size:14px; font-weight:800; margin-top:1px; }}
  .conv2-bar {{ height:5px; border-radius:3px; background:var(--inset); overflow:hidden; margin-top:6px; }}
  .conv2-bar i {{ display:block; height:100%; border-radius:3px; }}
  .plan-chips {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; margin:11px 0 2px; }}
  .plan-chip {{ background:var(--inset); border:1px solid var(--line); border-radius:9px; padding:6px 7px;
    text-align:center; min-width:0; }}
  .plan-chip .pc-l {{ font-size:9px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:700; }}
  .plan-chip .pc-v {{ font-family:var(--mono); font-size:12.5px; font-weight:700; margin-top:2px;
    font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .plan-chip.tgt .pc-v {{ color:var(--buy); }} .plan-chip.stp .pc-v {{ color:var(--sell); }}
  .plan-chip.tgt {{ border-color:color-mix(in srgb,var(--buy) 28%,var(--line)); }}
  .plan-chip.stp {{ border-color:color-mix(in srgb,var(--sell) 28%,var(--line)); }}
  .meta-chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
  .meta-chip {{ display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:600;
    padding:3px 9px; border-radius:999px; border:1px solid var(--line); background:var(--inset); color:var(--muted); }}
  .meta-chip .ico {{ width:12px; height:12px; }}
  .meta-chip.fresh {{ color:var(--buy); border-color:color-mix(in srgb,var(--buy) 30%,var(--line)); }}
  .meta-chip.held {{ color:var(--accent); border-color:color-mix(in srgb,var(--accent) 30%,var(--line)); }}
  .meta-chip.bad {{ color:var(--sell); border-color:color-mix(in srgb,var(--sell) 30%,var(--line)); }}
  .meta-chip.ai {{ color:var(--txt2); border-color:var(--line); }}
  @media (max-width:400px) {{ .plan-chips {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  /* ---- SpaceX-style signal card (monochrome, label/value driven) ---- */
  .sxcard {{ position:relative; display:flex; flex-direction:column; background:var(--card);
    border:1px solid var(--line); border-radius:16px; padding:22px 24px 20px; box-shadow:none; }}
  .sxcard:hover {{ border-color:rgba(255,255,255,.20); transform:none; box-shadow:none; }}
  .sxcard .favbtn {{ position:absolute; top:15px; right:15px; z-index:2; }}
  .sx-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:14px; }}
  .sx-hl {{ display:flex; align-items:center; gap:11px; min-width:0; }}
  .sx-hid {{ min-width:0; }}
  .sxcard .card-mono {{ width:34px; height:34px; border-radius:8px; flex:0 0 auto; }}
  .sx-title {{ font-size:21px; font-weight:800; letter-spacing:-.01em; line-height:1.12; color:var(--txt); }}
  .sx-meta {{ text-align:right; white-space:nowrap; padding-right:26px; }}
  .sx-meta-l {{ font-size:13px; color:var(--muted); }}
  .sx-meta-v {{ font-size:15px; font-weight:700; color:var(--txt2); font-variant-numeric:tabular-nums; }}
  .sx-meta-chg {{ font-size:12px; margin-top:2px; font-variant-numeric:tabular-nums; }}
  .sx-sub {{ font-size:13px; color:var(--muted); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .sx-desc {{ font-size:14px; line-height:1.5; color:var(--muted); margin-top:13px;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
  .sx-pills {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:15px; }}
  .sx-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:500;
    padding:5px 12px; border-radius:999px; background:var(--inset); color:var(--txt2); line-height:1.2; }}
  .sx-dot {{ width:6px; height:6px; border-radius:50%; flex:0 0 auto; }}
  .sx-dot.l {{ background:var(--buy); }} .sx-dot.s {{ background:var(--sell); }}
  .sx-div {{ height:1px; background:var(--line); margin:18px 0 14px; }}
  .sxcard .sx-div {{ margin-top:auto; }}
  .sx-rows {{ display:flex; flex-direction:column; gap:10px; }}
  .sx-row {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; }}
  .sx-l {{ font-size:14px; color:var(--muted); }}
  .sx-v {{ font-size:14px; font-weight:600; color:var(--txt); font-variant-numeric:tabular-nums; }}
  .sx-up {{ color:var(--buy); }} .sx-dn {{ color:var(--sell); }}
  .sxcard .card-warn {{ margin-top:14px; }}
  /* ---- Markets sub-tab layout ---- */
  .mkt {{ display:grid; grid-template-columns:190px minmax(0,1fr); gap:18px; align-items:start; }}
  .mkt-side {{ display:flex; flex-direction:column; position:sticky; top:66px; }}
  .mkt-side button {{ text-align:left; background:none; border:0; border-top:1px solid var(--line);
    color:var(--txt2); font-family:'Inter',-apple-system,sans-serif; font-size:14px; font-weight:400;
    padding:11px 4px; cursor:pointer; transition:color .15s ease; }}
  .mkt-side button:first-child {{ border-top:0; }}
  .mkt-side button:hover {{ color:var(--txt); }}
  .mkt-side button.on {{ color:var(--txt); font-weight:600; background:rgba(255,255,255,.05);
    border-radius:8px; padding-left:12px; border-top-color:transparent; }}
  .mkt-view {{ display:none; }} .mkt-view.on {{ display:block; }}
  @media (max-width:760px) {{ .mkt {{ grid-template-columns:1fr; }}
    .mkt-side {{ flex-direction:row; flex-wrap:wrap; position:static; }}
    .mkt-side button {{ border-top:0; }} }}
  /* ---- modal sub-tab layout ---- */
  .modal-wide {{ max-width:880px; }}
  .mk-top {{ display:flex; gap:7px; flex-wrap:wrap; margin:18px 0 0; padding-bottom:2px; }}
  .mk-top button {{ display:inline-flex; align-items:center; gap:6px; background:none;
    border:1px solid rgba(255,255,255,.14); color:var(--muted); font-size:13px; font-weight:500;
    padding:7px 14px; cursor:pointer; border-radius:999px;
    transition:color .15s ease, background .15s ease, border-color .15s ease; white-space:nowrap; }}
  .mk-top button .ico {{ width:14px; height:14px; opacity:.8; }}
  .mk-top button:hover {{ color:var(--txt); border-color:var(--txt); }}
  .mk-top button.on {{ color:#000; background:var(--txt); border-color:var(--txt); }}
  .mk-top button.on .ico {{ opacity:1; }}
  .mk {{ display:grid; grid-template-columns:158px minmax(0,1fr); gap:18px; margin-top:14px; align-items:start; }}
  .mk-side {{ display:flex; flex-direction:column; position:sticky; top:0; }}
  .mk-side button {{ text-align:left; background:none; border:0; border-top:1px solid var(--line);
    color:var(--txt2); font-family:'Inter',-apple-system,sans-serif; font-size:13.5px; font-weight:400;
    padding:10px 4px; cursor:pointer; transition:color .15s ease; }}
  .mk-side button:first-child {{ border-top:0; }}
  .mk-side button:hover {{ color:var(--txt); }}
  .mk-side button.on {{ color:var(--txt); font-weight:600; background:rgba(255,255,255,.05);
    border-radius:8px; padding-left:11px; border-top-color:transparent; }}
  .mk-view {{ display:none; }} .mk-view.on {{ display:block; }}
  @media (max-width:680px) {{ .mk {{ grid-template-columns:1fr; }}
    .mk-side {{ flex-direction:row; flex-wrap:wrap; position:static; }}
    .mk-side button {{ border-top:0; }} }}
  /* favorites star */
  .favbtn {{ background:none; border:none; color:var(--flat); cursor:pointer; font-size:17px;
    line-height:1; padding:0 2px; flex:0 0 auto; }}
  .favbtn:hover, .favbtn.on {{ color:var(--txt); }}
  /* ---- mobile / small screens ---- */
  @media (max-width:600px) {{
    /* rank table: drop to the 3 key factors on a narrow screen */
    .rkf-sm {{ display:none; }}
    /* data/intelligence tables: denser, and let a wide one scroll inside its panel */
    .tbl {{ font-size:12px; }}
    .tbl th, .tbl td {{ padding:5px 6px; }}
    .ovbox {{ overflow-x:auto; }}
    .wrap {{ padding:16px 12px 48px; }}
    h1 {{ font-size:21px; }}
    .appbar-top {{ gap:8px; padding:11px 2px 7px; }}
    .brand {{ font-size:16.5px; }}
    .tabs button {{ padding:9px 12px 11px; font-size:13.5px; }}
    /* tighter KPI tiles so signals aren't pushed a full screen down */
    .kpis {{ grid-template-columns:repeat(2, minmax(0,1fr)); gap:8px; margin-bottom:16px; }}
    .kpi {{ padding:9px 11px; }}
    .kpi-l {{ font-size:10px; }}
    .kpi-v {{ font-size:19px; margin-top:2px; }}
    .kpi-sub {{ font-size:10px; }}
    .trackrec {{ font-size:12px; }}
    .trackrec th, .trackrec td {{ padding:5px 6px; }}
    /* use more of the screen for the detail popup */
    .overlay {{ padding:8px; }}
    .modal {{ padding:0 14px 16px; margin:6px auto; border-radius:14px; }}
    .modal h3 {{ font-size:19px; }}
    .mhead {{ margin:0 -14px 4px; padding:14px 14px 12px; gap:10px; border-radius:14px 14px 0 0; }}
    .mhead-logo {{ width:36px; height:36px; border-radius:9px; font-size:13px; }}
    .mhead-tick {{ font-size:20px; }}
    .mhead-px {{ font-size:16px; }}
    .mk-top {{ overflow-x:auto; flex-wrap:nowrap; scrollbar-width:none; }}
    .mk-top::-webkit-scrollbar {{ display:none; }}
    .tv-wrap {{ height:340px; }}
  }}
  /* =====================================================================
     DESIGN SYSTEM v1 — warm-gold glass, applied across EVERY component (dark).
     Loaded last so it harmonises the whole dashboard. See DESIGN_SPEC.md.
     ===================================================================== */
  html[data-theme="dark"] h1, html[data-theme="dark"] h2, html[data-theme="dark"] h3,
  html[data-theme="dark"] .bbtitle, html[data-theme="dark"] .why-h, html[data-theme="dark"] .ai-h {{ letter-spacing:-.012em; }}
  /* nested inner surfaces — translucent fills (no extra blur, avoids glass-on-glass muddiness) */
  html[data-theme="dark"] .deskread, html[data-theme="dark"] .scen, html[data-theme="dark"] .sigdet,
  html[data-theme="dark"] .readout, html[data-theme="dark"] .ai-box, html[data-theme="dark"] .nidea,
  html[data-theme="dark"] .ladder, html[data-theme="dark"] .ovbox, html[data-theme="dark"] .feeditem,
  html[data-theme="dark"] .lcard, html[data-theme="dark"] .splititem, html[data-theme="dark"] .tkitem,
  html[data-theme="dark"] .sysrow, html[data-theme="dark"] .method, html[data-theme="dark"] .kv,
  html[data-theme="dark"] .card-stat, html[data-theme="dark"] .plangrid {{
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:11px; }}
  html[data-theme="dark"] .deskread {{ border-left:3px solid rgba(255,255,255,.16); }}
  html[data-theme="dark"] .deskread.ai-read, html[data-theme="dark"] #mAI {{ border-left:3px solid var(--ai) !important; }}
  /* controls — glass pill buttons */
  html[data-theme="dark"] .themebtn, html[data-theme="dark"] .ctlbtn, html[data-theme="dark"] .favbtn,
  html[data-theme="dark"] .more, html[data-theme="dark"] .tc-seg button, html[data-theme="dark"] .viewctl button,
  html[data-theme="dark"] .chartctl button,
  html[data-theme="dark"] .ctlgrp button, html[data-theme="dark"] .tc-cmp, html[data-theme="dark"] .tc-clr {{
    background:rgba(255,255,255,.04); border:1px solid var(--glass-bd); border-radius:10px; color:var(--txt2);
    transition:background-color .15s ease, border-color .15s ease, color .15s ease; }}
  html[data-theme="dark"] .themebtn:hover, html[data-theme="dark"] .ctlbtn:hover, html[data-theme="dark"] .more:hover,
  html[data-theme="dark"] .tc-seg button:hover, html[data-theme="dark"] .viewctl button:hover {{
    background:rgba(255,255,255,.08); color:var(--txt);
    border-color:color-mix(in srgb,var(--accent) 34%,var(--glass-bd)); }}
  /* active segmented / side-view controls → gold */
  html[data-theme="dark"] .tc-seg button.on, html[data-theme="dark"] .viewctl button.on,
  html[data-theme="dark"] .ctltog.on {{
    background:linear-gradient(90deg,color-mix(in srgb,var(--accent) 22%,transparent),color-mix(in srgb,var(--accent) 7%,transparent));
    color:var(--accent); border-color:color-mix(in srgb,var(--accent) 42%,transparent); }}
  /* chips / pills / small tags */
  html[data-theme="dark"] .chip, html[data-theme="dark"] .altpill, html[data-theme="dark"] .cat-chip,
  html[data-theme="dark"] .why-chip, html[data-theme="dark"] .tv-chip, html[data-theme="dark"] .bt-chip,
  html[data-theme="dark"] .ai-tag, html[data-theme="dark"] .nidea-src, html[data-theme="dark"] .strat-badge {{
    background:rgba(255,255,255,.05); border:1px solid var(--glass-bd); color:var(--txt2); border-radius:999px; }}
  /* page tabs + top tabs → gold active */
  html[data-theme="dark"] .tabs button.on {{ color:var(--accent); border-bottom-color:var(--accent); }}
  html[data-theme="dark"] .toptabs button.on {{ color:var(--accent); border-bottom-color:var(--accent); }}
  /* data tables */
  html[data-theme="dark"] .tbl th, html[data-theme="dark"] .trackrec th {{ color:var(--muted); border-color:var(--glass-bd); }}
  html[data-theme="dark"] .tbl td, html[data-theme="dark"] .trackrec td {{ border-color:rgba(255,255,255,.05); }}
  html[data-theme="dark"] .tbl tr:hover td, html[data-theme="dark"] .trackrec tr:hover td {{ background:rgba(255,255,255,.035); }}
  /* form controls */
  html[data-theme="dark"] select, html[data-theme="dark"] input[type=text], html[data-theme="dark"] input[type=search],
  html[data-theme="dark"] input[type=number] {{
    background:rgba(255,255,255,.04); border:1px solid var(--glass-bd); color:var(--txt); border-radius:9px; }}
  /* modal: richer close + heading, and give inner sections breathing room */
  html[data-theme="dark"] .modal .close:hover {{ color:var(--accent); }}
  html[data-theme="dark"] .modal .summary {{ color:var(--txt2); }}
  /* conviction meter → gold track */
  html[data-theme="dark"] .conv-meter {{ background:rgba(255,255,255,.10); }}
  /* soften the diagnostic note so it doesn't read like an error */
  html[data-theme="dark"] .note {{ color:var(--muted);
    background:rgba(255,255,255,.04); border:1px solid var(--glass-bd); }}
  /* --- SIGNAL CARD internals: premium treatment --- */
  html[data-theme="dark"] .card-mono {{ background:var(--inset);
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.06); }}
  html[data-theme="dark"] .card-id .s {{ font-size:17px; letter-spacing:-.01em; }}
  html[data-theme="dark"] .act {{ padding:3px 12px !important; border-radius:999px !important; font-size:11px !important;
    font-weight:800 !important; letter-spacing:.05em; box-shadow:0 3px 10px rgba(0,0,0,.4); }}
  html[data-theme="dark"] .conv-wrap {{ margin:13px 0 11px; }}
  html[data-theme="dark"] .conv-row {{ font-size:10.5px; }}
  html[data-theme="dark"] .conv-meter {{ height:8px; background:rgba(255,255,255,.09); border-radius:6px;
    box-shadow:inset 0 1px 2px rgba(0,0,0,.45); }}
  html[data-theme="dark"] .conv-fill {{ border-radius:6px; box-shadow:inset 0 1px 0 rgba(255,255,255,.5), 0 0 12px rgba(0,0,0,.2); }}
  html[data-theme="dark"] .card-px, html[data-theme="dark"] .card-stat b {{ font-family:var(--mono); }}
  html[data-theme="dark"] .card-stats {{ border-top:1px solid rgba(255,255,255,.06); padding-top:10px; margin-top:11px; }}
  html[data-theme="dark"] .card-spark svg {{ opacity:1; }}
  html[data-theme="dark"] .more {{ color:var(--accent); font-weight:600; }}
  html[data-theme="dark"] .card-why {{ border-left:3px solid color-mix(in srgb,var(--accent) 55%,transparent); }}
  /* --- final cohesion: type, AI identity, headers, links (whole app) --- */
  html[data-theme="dark"] .mdh {{ display:block; color:var(--acc2); font-size:12.5px; font-weight:700;
    letter-spacing:.02em; margin:12px 0 5px; }}
  html[data-theme="dark"] .mdh:first-child {{ margin-top:0; }}
  html[data-theme="dark"] p {{ margin:0 0 9px; }} html[data-theme="dark"] p:last-child {{ margin-bottom:0; }}
  html[data-theme="dark"] .ai-h {{ color:var(--txt2); font-weight:700; }}
  html[data-theme="dark"] .ai-box {{ background:rgba(255,255,255,.04) !important;
    border:1px solid var(--line) !important; border-radius:12px !important; }}
  html[data-theme="dark"] h1, html[data-theme="dark"] h2, html[data-theme="dark"] h3 {{ font-weight:700; }}
  html[data-theme="dark"] .subhead, html[data-theme="dark"] .why-h, html[data-theme="dark"] .kpi-l,
  html[data-theme="dark"] .bt-l, html[data-theme="dark"] .conv-row {{ color:var(--muted); }}
  html[data-theme="dark"] a {{ color:var(--acc2); }}
  html[data-theme="dark"] ::selection {{ background:color-mix(in srgb,var(--accent) 40%,transparent); color:#fff; }}
  /* sector-strength bars — glassy track */
  html[data-theme="dark"] .secbar, html[data-theme="dark"] .secbarwrap {{ background:rgba(255,255,255,.06); border-radius:6px; }}
  /* keep charts/candles crisp over glass (no blur bleed on chart canvases) */
  html[data-theme="dark"] #featuredChart, html[data-theme="dark"] .tc-wrap, html[data-theme="dark"] .chartbox {{ backdrop-filter:none; }}

  /* =====================================================================
     OVERFLOW SAFETY v1 — global, robust rules so text always fits its box.
     Loaded last so it harmonises every component. Two strategies:
       (a) single-line clip (ellipsis) for identity fields — tickers, names;
       (b) wrap (break long tokens) for prose — briefs, notes, reasons, values.
     ===================================================================== */
  /* let grid/flex children actually shrink instead of forcing overflow */
  .kpis, .bento, .plangrid, .card-stats, .chips, .nideas {{ min-width:0; }}
  .kpi, .bt, .card, .stat, .chip, .sigdet, .nidea, .scen-row, .checks li,
  .why-chip, .altpill, .cat-chip, .bt-chip, .card-why, .deskread {{ min-width:0; }}
  /* grids that used a fixed minmax floor can now shrink to fit narrow columns */
  .kpis {{ grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr)); }}
  /* long KPI / bento / card numbers wrap instead of spilling past the rounded edge */
  .kpi-v, .bt-v, .card-px {{ overflow-wrap:anywhere; word-break:break-word; }}
  .kpi, .bt, .card, .stat, .kpi-v, .bt-v, .card-px, .card-day {{ min-width:0; }}
  /* prose blocks wrap long tokens (URLs, tickers) rather than overflow */
  .bt-body, .bt-list, .bt-sub, .kpi-sub, .card-warn, .scen-why, .sigdet-why,
  .nidea-why, .nidea-src, .reasons li, .checks li, .summary, .deskread,
  .news li, .news li a {{ overflow-wrap:anywhere; word-break:break-word; }}
  /* chips wrap their own text when a label is very long */
  .chip, .why-chip, .altpill, .cat-chip, .syspill, .bt-chip {{ overflow-wrap:anywhere; }}
  /* section headers: title clips, sub stays put, nothing spills */
  .sec-head {{ min-width:0; }}
  .sec-head h2 {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .sec-head .sh-sub {{ flex:0 0 auto; white-space:nowrap; }}
  .sec-head .sh-ico {{ flex:0 0 auto; }}
  /* ---- CONTENT WRAPS, IDENTITY CLIPS ---------------------------------
     Content text (labels, chips, quotes, disclaimer) must be fully readable:
     it WRAPS and its box grows. Only true single-line identity fields clip. */
  /* macro backdrop tiles: labels wrap to 2+ lines, tile grows to fit */
  .trackstats .stat {{ overflow:visible; height:auto; }}
  .trackstats .stat .l {{ white-space:normal; overflow:visible; text-overflow:clip;
    overflow-wrap:anywhere; }}
  /* event-calendar chips: multi-line, grow to show the full release name */
  .chip.mini {{ white-space:normal; height:auto; border-radius:10px; overflow:visible;
    text-overflow:clip; overflow-wrap:anywhere; display:inline-block; }}
  /* news-idea quote boxes: wrap fully and let the box grow */
  .nidea, .nidea-src, .nidea-why {{ overflow:visible; white-space:normal;
    overflow-wrap:anywhere; word-break:break-word; }}
  /* bottom strategy/disclaimer paragraph: stay inside its container, wrap */
  .disclaimer {{ max-width:100%; overflow-wrap:anywhere; word-break:break-word;
    white-space:normal; overflow:visible; }}
  /* identity rows stay single-line + ellipsis (already partly set — reinforce).
     Ellipsis is restricted to ONLY these fields + the modal header ticker. */
  .card-id .s, .wl-sym, .mhead-tick {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .card-id, .wl-main {{ min-width:0; }}
  .card-top, .card-px-row, .wl {{ min-width:0; }}
  /* any wide table inside a panel scrolls horizontally instead of blowing out */
  .trackrec, .tbl {{ min-width:0; }}
  .tbl-scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; max-width:100%; }}
  .ovbox {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
  /* modal: never let content punch outside the rounded card; scroll inside on small screens.
     Use overflow-x:clip (not hidden) so it doesn't create a scroll box that breaks the
     sticky header. */
  .modal {{ overflow-x:clip; }}
  @supports not (overflow:clip) {{ .modal {{ overflow-x:hidden; }} }}
  .mk-main {{ min-width:0; }}
  .mk-view {{ min-width:0; }}
  .mk-view table {{ max-width:100%; }}
  #mStrategies, #mSignals, #mResearch, #mMeta, #mRegimeFit, #mNewsRead, #mRank {{
    min-width:0; overflow-wrap:anywhere; }}
  #mStrategies .trackrec {{ display:block; overflow-x:auto; -webkit-overflow-scrolling:touch; }}
  /* ============================================================
     xAI RESURFACE — stage 1 chrome: xAI-console-style sectioned sidebar
     ============================================================ */
  .appbar {{ background:rgba(0,0,0,.72) !important; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border-bottom:1px solid var(--line); }}
  .brand {{ color:var(--txt); letter-spacing:.02em; }}
  .brand-mark {{ background:none !important; border:1px solid var(--txt); color:var(--txt) !important;
    box-shadow:none !important; border-radius:8px; }}
  /* sidebar: flat, wider, xAI-console look (kept vertical) */
  .sidebar {{ width:216px; flex:0 0 216px; background:transparent !important; backdrop-filter:none !important;
    -webkit-backdrop-filter:none !important; border:0 !important; border-radius:0 !important;
    box-shadow:none !important; padding:4px 8px; gap:1px; }}
  .sidebar button {{ border-left:0 !important; border-radius:9px; text-transform:none; letter-spacing:0;
    font-size:14px; font-weight:400; color:var(--muted); padding:9px 12px; gap:12px; }}
  .sidebar button svg {{ width:17px; height:17px; opacity:.8; }}
  .sidebar button:hover {{ background:rgba(255,255,255,.05) !important; color:var(--txt); }}
  .sidebar button.on {{ background:rgba(255,255,255,.07) !important; color:var(--txt) !important;
    box-shadow:none !important; border-left:0 !important; }}
  .sidebar button.on svg {{ opacity:1; }}
  /* section labels between nav groups */
  .side-sect {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--faint);
    font-weight:500; padding:17px 12px 6px; }}
  .side-sig, .side-foot {{ display:none !important; }}
  /* kill the gold HUD corner ticks */
  .bt::after, .bt::before, .kpi::after, .kpi::before, .ovbox::after {{ display:none !important; }}
  /* secondary tab bar -> white underline */
  .toptabs button.on {{ color:var(--txt) !important; border-bottom-color:var(--txt) !important; }}
  .toptabs button.on::before {{ content:"" !important; }}
  /* accent is now white -> pills/CTAs on an accent bg need dark text */
  .sd-full, .why-chip.trig, .tc-seg button.on, button.sf-run, .sd-cta, #newbuild, .bt-logo {{ color:#000 !important; }}
  /* ---- stage 2: Signals hero ---- */
  .hero-x {{ display:flex; flex-direction:column; padding:8px 0 16px; }}
  .hx-title {{ font-size:62px; line-height:.95; font-weight:300; letter-spacing:-.03em; margin:0 0 14px; }}
  .hx-sub {{ font-size:16px; line-height:1.55; color:var(--muted); max-width:440px; margin:0 0 22px; font-weight:300; }}
  .hx-btns {{ display:flex; gap:12px; margin:0 0 26px; }}
  .hx-btn {{ border-radius:999px; padding:12px 24px; font-size:14px; font-weight:400; cursor:pointer;
    border:1px solid rgba(255,255,255,.16); background:none; color:var(--txt); font-family:inherit; transition:.15s; }}
  .hx-btn.solid {{ background:var(--txt); color:#000; border-color:var(--txt); }}
  .hx-btn.solid:hover {{ background:#d8d8d8; border-color:#d8d8d8; }}
  .hx-btn.ghost:hover {{ border-color:var(--txt); }}
  .hx-kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:26px 30px; margin-top:28px;
    flex:1 1 auto; align-content:space-between; }}
  /* full-width hero (S&P chart + Live-TV removed → ticker tape band below) */
  .hero-row.hero-solo {{ display:block; }}
  .hero-solo .hx-kpis {{ grid-template-columns:repeat(6,minmax(0,1fr)); }}
  @media (max-width:1100px) {{ .hero-solo .hx-kpis {{ grid-template-columns:repeat(4,minmax(0,1fr)); }} }}
  @media (max-width:680px) {{ .hero-solo .hx-kpis {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
  /* full-width TradingView ticker tape band */
  .ticker-band {{ margin:22px 0 8px; border:1px solid var(--line); border-radius:12px; overflow:hidden;
    background:var(--card); }}
  .ticker-band .tradingview-widget-container {{ width:100%; }}
  /* ===== Accept / Reject on cards ===== */
  .ar-btns {{ position:absolute; top:14px; right:44px; display:flex; gap:5px; opacity:0; transition:opacity .15s ease; z-index:3; }}
  .sxcard:hover .ar-btns, .sxcard.accepted .ar-btns, .sxcard.rejected .ar-btns {{ opacity:1; }}
  .ar-btn {{ width:26px; height:26px; border-radius:7px; border:1px solid var(--line); background:var(--card);
    color:var(--muted); cursor:pointer; display:inline-flex; align-items:center; justify-content:center; padding:0; }}
  .ar-yes:hover, .sxcard.accepted .ar-yes {{ color:var(--buy); border-color:color-mix(in srgb,var(--buy) 45%,var(--line)); background:color-mix(in srgb,var(--buy) 12%,var(--card)); }}
  .ar-no:hover, .sxcard.rejected .ar-no {{ color:var(--sell); border-color:color-mix(in srgb,var(--sell) 45%,var(--line)); background:color-mix(in srgb,var(--sell) 12%,var(--card)); }}
  .sxcard.accepted {{ border-color:color-mix(in srgb,var(--buy) 45%,var(--line)); box-shadow:0 0 0 1px color-mix(in srgb,var(--buy) 30%,transparent); }}
  .sxcard.rejected {{ border-color:color-mix(in srgb,var(--sell) 40%,var(--line)); }}
  .sxcard.ctrl-dim {{ opacity:.38; filter:grayscale(.3); }}
  .sxcard.accepted.ctrl-dim {{ opacity:1; filter:none; }}
  /* ===== Control panel ===== */
  .ctrl-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:8px; align-items:start; }}
  .ctrl-card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 22px; }}
  .ctrl-card h3 {{ margin:0 0 14px; font-size:15px; font-weight:700; }}
  .ctrl-row {{ margin-top:16px; }}
  .ctrl-lbl {{ font-size:13.5px; color:var(--txt2); display:flex; justify-content:space-between; margin-bottom:7px; }}
  .ctrl-lbl b {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
  .ctrl-hint {{ font-size:11.5px; color:var(--muted); margin-top:5px; }}
  .ctrl-card input[type=range] {{ width:100%; accent-color:var(--accent); cursor:pointer; }}
  .ctrl-tog {{ display:flex; align-items:flex-start; gap:9px; margin-top:15px; font-size:13px; color:var(--txt2); cursor:pointer; line-height:1.4; }}
  .ctrl-tog input {{ margin-top:2px; accent-color:var(--accent); cursor:pointer; }}
  .ctrl-preview {{ margin-top:18px; padding-top:14px; border-top:1px solid var(--line); font-size:12.5px; color:var(--txt); font-weight:600; }}
  .ctrl-dec-h {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:8px; }}
  .ctrl-chips {{ display:flex; flex-wrap:wrap; gap:7px; }}
  .ctrl-chip {{ display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:600; border-radius:999px; padding:3px 6px 3px 11px; }}
  .ctrl-chip.acc {{ color:var(--buy); background:color-mix(in srgb,var(--buy) 14%,transparent); }}
  .ctrl-chip.rej {{ color:var(--sell); background:color-mix(in srgb,var(--sell) 14%,transparent); }}
  .ctrl-chip button {{ background:none; border:0; color:inherit; cursor:pointer; font-size:14px; line-height:1; opacity:.7; padding:0 2px; }}
  .ctrl-chip button:hover {{ opacity:1; }}
  .ctrl-none {{ font-size:12px; color:var(--muted); }}
  .ctrl-clear {{ margin-top:18px; background:none; border:1px solid var(--line); color:var(--muted); border-radius:8px;
    padding:7px 13px; font-size:12px; cursor:pointer; }}
  .ctrl-clear:hover {{ color:var(--txt); border-color:var(--txt); }}
  .ctrl-apply {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-top:20px; }}
  .ctrl-apply-btn {{ background:var(--txt); color:#000; border:0; border-radius:999px; font-size:14px; font-weight:600;
    padding:11px 22px; cursor:pointer; }}
  .ctrl-apply-btn:hover {{ filter:brightness(1.1); }}
  .ctrl-copy {{ background:none; border:1px solid var(--line); color:var(--txt2); border-radius:999px; padding:10px 16px; font-size:13px; cursor:pointer; }}
  .ctrl-copy:hover {{ border-color:var(--txt); color:var(--txt); }}
  .ctrl-apply-note {{ font-size:11.5px; color:var(--muted); flex:1; min-width:200px; line-height:1.5; }}
  .ctrl-apply-note code {{ font-family:var(--tape); background:var(--inset); padding:1px 5px; border-radius:4px; }}
  .ctrl-reset {{ margin-top:14px; }}
  .ctrl-reset button {{ background:none; border:0; color:var(--muted); font-size:11.5px; cursor:pointer; text-decoration:underline; }}
  @media (max-width:760px) {{ .ctrl-grid {{ grid-template-columns:1fr; }} }}
  .hx-kpi .v {{ font-size:32px; font-weight:600; letter-spacing:-.02em; line-height:1; font-variant-numeric:tabular-nums; }}
  .hx-kpi .k {{ font-size:12px; color:var(--muted); margin-top:8px; }}
  .hx-kpi .v.buy {{ color:var(--buy); }} .hx-kpi .v.sell {{ color:var(--sell); }} .hx-kpi .v.warn {{ color:var(--warn); }}
  @media (max-width:760px) {{ .hx-title {{ font-size:58px; }} .hx-kpis {{ grid-template-columns:repeat(3,minmax(0,1fr)); gap:24px 22px; }} }}
  /* hero right column: chart + Live TV */
  .hero-row {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(400px,520px); gap:32px; align-items:stretch; }}
  .hero-row .hero-x {{ padding-right:0; }}
  .hero-side {{ display:flex; flex-direction:column; gap:14px; padding-top:6px; min-width:0; align-self:stretch; }}
  .hero-chart {{ border:1px solid var(--line); border-radius:14px; background:var(--card); overflow:hidden; }}
  .hero-chart-h {{ font-size:12.5px; color:var(--muted); padding:12px 14px 8px; display:flex; align-items:center; gap:6px; }}
  .hero-chart-frame {{ width:100%; height:190px; display:block; border:0; }}
  @media (max-width:900px) {{ .hero-row {{ grid-template-columns:1fr; gap:10px; }} .hero-side {{ padding-top:2px; }} }}
  /* xAI eyebrow label + stat callout strip */
  .hx-eyebrow {{ font-size:12px; font-weight:500; color:var(--muted); margin:0 0 16px; }}
  .sec-eyebrow {{ font-size:12px; font-weight:500; color:var(--muted); margin:32px 0 6px; }}
  .sec-eyebrow + .sec-head {{ margin-top:4px; }}
  .sx-statstrip {{ display:flex; flex-wrap:wrap; gap:44px; padding:22px 0; margin:6px 0 2px;
    border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
  .sx-stat-v {{ font-size:32px; font-weight:500; letter-spacing:-.02em; font-variant-numeric:tabular-nums; line-height:1; }}
  .sx-stat-k {{ font-size:12px; color:var(--muted); margin-top:8px; }}
  @media (max-width:600px) {{ .sx-statstrip {{ gap:28px 32px; }} .sx-stat-v {{ font-size:26px; }} }}
  /* Grok · X pulse feed */
  .gp-empty {{ color:var(--muted); font-size:13.5px; line-height:1.6; padding:14px 0 4px; max-width:640px; }}
  .gp-empty code {{ font-family:var(--tape); background:var(--inset); padding:1px 6px; border-radius:5px; font-size:12px; }}
  .gp-list {{ display:flex; flex-direction:column; }}
  .gp-row {{ display:grid; grid-template-columns:66px 96px 1fr auto; align-items:center; gap:14px;
    padding:13px 6px; border-top:1px solid var(--line); cursor:pointer; transition:background .15s ease; }}
  .gp-row:hover {{ background:rgba(255,255,255,.02); }}
  .gp-sym {{ font-weight:600; letter-spacing:-.01em; }}
  .gp-row .sx-pill {{ justify-self:start; text-transform:capitalize; }}
  .gp-up {{ color:var(--buy); }} .gp-dn {{ color:var(--sell); }} .gp-mut {{ color:var(--muted); }}
  .gp-note {{ color:var(--txt2); font-size:13px; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .gp-cat {{ color:var(--muted); }}
  .gp-meta {{ color:var(--muted); font-size:12px; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .gp-mom-mini {{ font-variant-numeric:tabular-nums; }}
  .gp-sigbadge {{ font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.03em;
    color:var(--buy); border:1px solid color-mix(in srgb,var(--buy) 38%,transparent); border-radius:999px;
    padding:1px 6px; margin-left:7px; vertical-align:middle; }}
  .gp-trend {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:14px;
    padding-top:14px; border-top:1px solid var(--line); }}
  .gp-trend-l {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
    display:inline-flex; align-items:center; gap:6px; margin-right:4px; }}
  .gp-trend-chip {{ font-size:12px; font-weight:600; background:var(--inset); color:var(--txt2);
    border-radius:999px; padding:4px 11px; }}
  .gp-trend-chip.sig {{ color:var(--buy); cursor:pointer; }}
  .gp-trend-chip.sig:hover {{ background:color-mix(in srgb,var(--buy) 14%,var(--inset)); }}
  @media (max-width:600px) {{ .gp-row {{ grid-template-columns:56px 84px 1fr; }} .gp-meta {{ display:none; }} }}
  /* ---- Grok deep-dive modal: what X is saying about a name right now ---- */
  .gk-modal {{ max-width:560px; padding:26px 28px; }}
  .gk-head {{ display:flex; align-items:center; gap:12px; }}
  .gk-tick {{ font-size:22px; font-weight:800; letter-spacing:-.01em; }}
  .gk-modal .sx-pill {{ text-transform:capitalize; }}
  .gk-mom {{ font-size:12.5px; font-variant-numeric:tabular-nums; }}
  .gk-sub {{ font-size:13px; color:var(--muted); margin:5px 0 0; }}
  .gk-cat {{ font-size:13.5px; color:var(--txt2); margin-top:15px; }}
  .gk-cat b {{ color:var(--txt); font-weight:600; }}
  .gk-when {{ color:var(--muted); }}
  .gk-fresh {{ font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:var(--buy);
    border:1px solid color-mix(in srgb,var(--buy) 40%,transparent); border-radius:999px; padding:1px 7px; margin-left:4px; }}
  .gk-lead {{ font-size:14.5px; line-height:1.5; color:var(--txt); margin-top:14px; }}
  .gk-sec {{ margin-top:20px; }}
  .gk-h {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
    margin-bottom:10px; display:flex; align-items:center; gap:6px; }}
  .gk-h.up {{ color:var(--buy); }} .gk-h.dn {{ color:var(--sell); }}
  .gk-chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .gk-chip {{ font-size:12.5px; background:var(--inset); color:var(--txt2); border-radius:999px; padding:5px 12px; }}
  .gk-args {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:20px; }}
  .gk-ul {{ margin:0; padding-left:16px; }}
  .gk-ul li {{ font-size:13px; line-height:1.55; color:var(--txt2); margin-bottom:5px; }}
  .gk-none {{ color:var(--muted); list-style:none; margin-left:-16px; }}
  .gk-hype {{ margin-top:14px; font-size:12.5px; line-height:1.5; color:var(--sell);
    background:color-mix(in srgb,var(--sell) 10%,transparent); border:1px solid color-mix(in srgb,var(--sell) 30%,transparent);
    border-radius:10px; padding:10px 13px; }}
  .gk-hype.med {{ color:var(--warn); background:color-mix(in srgb,var(--warn) 10%,transparent);
    border-color:color-mix(in srgb,var(--warn) 30%,transparent); }}
  .gk-hype b {{ font-weight:700; }}
  .gk-watch {{ margin-top:20px; background:var(--inset); border-radius:12px; padding:14px 16px;
    font-size:13.5px; line-height:1.5; color:var(--txt2); }}
  .gk-foot {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:22px;
    padding-top:16px; border-top:1px solid var(--line); flex-wrap:wrap; }}
  .gk-src {{ font-size:11.5px; color:var(--muted); display:inline-flex; align-items:center; gap:6px; }}
  .gk-open {{ background:var(--txt); color:#000; border:0; border-radius:999px; font-size:12.5px; font-weight:600;
    padding:8px 15px; cursor:pointer; }}
  .gk-open:hover {{ filter:brightness(1.1); }}
  @media (max-width:560px) {{ .gk-args {{ grid-template-columns:1fr; gap:14px; }} }}
  /* ---- stage 3: hairline signal rows ---- */
  .rowswrap {{ margin-top:2px; }}
  .rowsig {{ display:grid; grid-template-columns:30px 28px 1.5fr 96px 170px 1fr 20px; align-items:center;
    gap:16px; padding:17px 6px; border-top:1px solid var(--line); cursor:pointer; transition:.15s; }}
  .rowsig:hover {{ padding-left:9px; background:rgba(255,255,255,.02); }}
  .rowsig .rk {{ font-family:var(--mono); font-size:13px; color:var(--faint); }}
  .rowsig .nm {{ min-width:0; }}
  .rowsig .sym {{ font-size:16px; font-weight:500; letter-spacing:.01em; }}
  .rowsig .co {{ font-size:12.5px; color:var(--muted); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .rowsig .rtag {{ font-size:10px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
    border:1px solid rgba(255,255,255,.16); border-radius:999px; padding:2px 8px; margin-left:6px; vertical-align:2px; }}
  .rowsig .rconv {{ font-size:13px; color:var(--muted); }}
  .rowsig .rconv b {{ font-family:var(--mono); font-weight:500; font-size:16px; color:var(--txt); }}
  .rowsig .rpx {{ font-family:var(--mono); font-size:14px; text-align:right; }}
  .rowsig .rpx .chg {{ font-size:12px; margin-top:2px; }}
  .rowsig .rarr {{ color:var(--faint); font-size:16px; text-align:right; transition:.15s; }}
  .rowsig:hover .rarr {{ color:var(--txt); transform:translateX(3px); }}
  .rowsig .rspark canvas, .rowsig .rspark svg {{ display:block; }}
  @media (max-width:760px) {{ .rowsig {{ grid-template-columns:24px 26px 1.4fr auto 20px; gap:11px; }}
    .rowsig .rspark, .rowsig .rpx {{ display:none; }} }}
  /* ============================================================
     xAI RESURFACE — stage 4: fonts, section headers, cards, pills, buttons (all pages)
     ============================================================ */
  .brand {{ font-family:'Inter',-apple-system,sans-serif !important; text-transform:none !important;
    letter-spacing:.01em; font-weight:400; }}
  /* section headers -> quiet label + hairline icon (every page) */
  .sec-head {{ border-bottom:0; margin:34px 0 14px; }}
  .sec-head h2 {{ font-weight:600 !important; color:var(--txt) !important; letter-spacing:-.01em; font-size:18px; text-transform:none; }}
  .sec-head .sh-ico {{ background:none !important; border:0; color:var(--txt) !important; border-radius:6px; }}
  .sec-head .sh-ico.ai {{ background:none !important; color:var(--txt) !important; }}
  .sec-head .sh-sub {{ color:var(--muted); }}
  /* uniform hairline card everywhere */
  .card, .bt, .kpi, .stat, .lane, .ovbox, .glass, .featured, .tk-panel, .secbar, .trackrec,
  .bento-tile, .bento-feat, details.tvwidget, .wl {{
    border:1px solid var(--line) !important; border-radius:14px !important; background:var(--card) !important;
    box-shadow:none !important; }}
  .card {{ padding:18px 20px; }}
  /* pills / chips -> subtle rounded (keep semantic colour where it carries meaning) */
  .badge, .bt-chip, .why-chip, .tag, .metaChip, .convbadge {{ border-radius:999px !important; }}
  .why-chip, .bt-chip {{ background:rgba(255,255,255,.05) !important; border:1px solid var(--line) !important;
    color:var(--muted) !important; }}
  /* control + header buttons -> ghost pills; active = white pill */
  .themebtn, .ctlgrp button, .ctlbtn {{ border-radius:999px !important;
    border:1px solid rgba(255,255,255,.14) !important; background:none !important; color:var(--muted) !important;
    font-weight:400 !important; }}
  .themebtn:hover, .ctlgrp button:hover, .ctlbtn:hover {{ color:var(--txt) !important; border-color:var(--txt) !important; }}
  .ctlgrp button.on {{ background:var(--txt) !important; color:#000 !important; border-color:var(--txt) !important; }}
  /* ---- Live TV: channel card grid ---- */
  .tvwrap {{ border:1px solid var(--line); border-radius:14px; overflow:hidden; aspect-ratio:16/9; background:#000; }}
  .tvwrap iframe {{ width:100%; height:100%; display:block; border:0; }}
  .tvcards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; margin-top:4px; }}
  .tvcard {{ border:1px solid var(--line); border-radius:14px; overflow:hidden; cursor:pointer;
    background:var(--card); transition:border-color .15s; }}
  .tvcard:hover {{ border-color:rgba(255,255,255,.28); }}
  .tvcard.on {{ border-color:var(--txt); }}
  .tvthumb {{ position:relative; aspect-ratio:16/9; background:#0a0a0b; overflow:hidden; }}
  .tvthumb img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  .tvlabel {{ position:absolute; top:10px; left:10px; display:inline-flex; align-items:center; gap:6px;
    font-size:11px; letter-spacing:.02em; color:var(--txt); background:rgba(0,0,0,.6);
    border:1px solid rgba(255,255,255,.14); border-radius:999px; padding:4px 10px; }}
  .tvlabel .dot {{ width:6px; height:6px; border-radius:50%; background:var(--sell); }}
  .tvbody {{ padding:14px 16px 16px; }}
  .tvname {{ font-size:16px; font-weight:500; }}
  .tvdesc {{ font-size:13px; color:var(--muted); margin-top:4px; line-height:1.5; }}
  .tvtags {{ display:flex; gap:7px; margin-top:12px; flex-wrap:wrap; }}
  .tvtag {{ font-size:11px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:3px 10px; }}
  /* ---- top mega-nav (replaces the sidebar) ---- */
  .sidebar {{ display:none !important; }}
  .shell {{ display:block; }}
  .maincol {{ padding-left:0 !important; border-left:0 !important; width:100%; }}
  .meganav {{ display:flex; align-items:center; gap:2px; margin-left:14px; }}
  .mg-item {{ position:relative; }}
  .mg-lbl {{ display:inline-flex; align-items:center; gap:6px; padding:8px 13px; font-size:14px;
    color:var(--muted); cursor:default; border-radius:8px; white-space:nowrap; }}
  .mg-item:hover .mg-lbl {{ color:var(--txt); background:rgba(255,255,255,.05); }}
  .mg-car {{ font-size:9px; opacity:.7; }}
  .mg-panel {{ position:absolute; top:100%; left:0; margin-top:8px; background:#0b0b0c;
    border:1px solid var(--line); border-radius:14px; padding:8px; display:none;
    grid-template-columns:repeat(2,minmax(230px,1fr)); gap:4px; z-index:60;
    box-shadow:0 24px 60px rgba(0,0,0,.65); }}
  .mg-item:last-child .mg-panel {{ left:auto; right:0; }}
  .mg-item:hover .mg-panel {{ display:grid; }}
  .mg-panel::before {{ content:""; position:absolute; top:-8px; left:0; right:0; height:8px; }}
  .mg-link {{ display:flex; gap:12px; align-items:center; padding:10px; border-radius:10px; cursor:pointer; }}
  .mg-link:hover {{ background:rgba(255,255,255,.06); }}
  .mg-mock {{ width:76px; height:48px; border:1px solid var(--line); border-radius:7px; background:#050506;
    flex:0 0 auto; overflow:hidden; }}
  .mg-mock svg {{ width:100%; height:100%; display:block; }}
  .mg-t {{ font-size:14px; font-weight:500; color:var(--txt); }}
  .mg-d {{ font-size:12px; color:var(--muted); margin-top:2px; line-height:1.4; }}
  @media (max-width:900px) {{ .meganav {{ display:none; }} }}
  /* ---- stage 4b: one typeface (Inter) for ALL text incl. big numbers; mono only in the ticker ---- */
  .kpi-v, .bt-v, .stat .v, .hx-kpi .v, .px, .wl-px, .convbadge, .num, .mono2,
  .rowsig .rpx, .rowsig .rk, .rowsig .rconv b, .sig .conv b, .mhead-tick, .mhead-px, .mhead-chg,
  .scen-px, .brand, .appsearch input, .side-sig-row .ss-sym, .lv, .lad-row > span:last-child {{
    font-family:'Inter',-apple-system,Segoe UI,Roboto,sans-serif !important; font-variant-numeric:tabular-nums; }}
  .sec-head .sh-sub {{ font-family:'Inter',-apple-system,sans-serif !important; }}
  /* strategy badge -> quiet, not coral */
  .strat-badge {{ background:none !important; border:1px solid var(--line) !important; }}
  .strat-badge .v {{ color:var(--txt) !important; font-weight:500 !important; }}
  /* ---- stage 4c: signal cards -> grok-clean (keep target-green / stop-red + charts) ---- */
  .card-why {{ border:1px solid var(--line) !important; border-left:1px solid var(--line) !important;
    background:none !important; border-radius:10px !important; }}
  .why-h {{ color:var(--muted) !important; }}
  .why-fam {{ background:rgba(255,255,255,.05) !important; border:1px solid var(--line) !important;
    color:var(--txt) !important; border-radius:999px !important; }}
  .plan-chip {{ background:none !important; border-radius:10px !important; }}
  .plan-chip .pc-v {{ font-family:'Inter',-apple-system,sans-serif !important; }}
  .meta-chip {{ background:rgba(255,255,255,.04) !important; }}
  .meta-chip.ai {{ color:var(--muted) !important; border-color:var(--line) !important; }}
  .meta-chip.held, .card-age.held {{ color:var(--muted) !important; }}
  .cat-chip {{ border-radius:999px !important; background:rgba(255,255,255,.05) !important;
    border:1px solid var(--line) !important; color:var(--muted) !important; }}
  .conv2-ring .cv {{ font-family:'Inter',-apple-system,sans-serif !important; }}
  .card-id .s {{ font-weight:600 !important; }}
  /* home tiles a touch more padded, grok-style */
  .bt {{ padding:15px 17px; }} .bt-l {{ letter-spacing:.05em; }}
</style></head>
<body><div class="wrap">
  <header class="appbar">
    <div class="appbar-top">
      <div class="brand"><span class="brand-mark">◢</span><span>Signal Desk</span></div>
      {meganav_html}
      <div class="appbar-mid">
        <div class="appsearch">
          {_svg('search',14)}
          <input id="tickerSearch" type="text" autocomplete="off" spellcheck="false"
                 placeholder="Search ticker…" aria-label="Search ticker">
        </div>
        <span class="appclock" id="barClock" aria-label="Market clock"></span>
        {regime_pill}
      </div>
      <div class="appbar-right">
        <span class="badge m-{mode}">{mode}</span>
        <span class="livepill" id="liveStatus"></span>
        <button class="themebtn" title="Reload for the latest published build" onclick="location.reload()">{_svg('refresh',14)} Refresh</button>
        <div class="accent-wrap">
          <button id="accentBtn" class="themebtn" title="Accent colour" aria-label="Accent colour">{_svg('palette',15)}</button>
          <div id="accentPop" class="accent-pop" hidden>
            <button class="acsw" data-accent="#58a6ff" style="background:#58a6ff;" aria-label="Blue"></button>
            <button class="acsw" data-accent="#2dd4bf" style="background:#2dd4bf;" aria-label="Cyan"></button>
            <button class="acsw" data-accent="#8b5cf6" style="background:#8b5cf6;" aria-label="Violet"></button>
            <button class="acsw" data-accent="#46d08a" style="background:#46d08a;" aria-label="Green"></button>
            <button class="acsw" data-accent="#b5b5ba" style="background:#b5b5ba;" aria-label="Amber"></button>
            <button class="acsw" data-accent="#ff7a59" style="background:#ff7a59;" aria-label="Coral"></button>
            <button class="acsw" data-accent="#ec4899" style="background:#ec4899;" aria-label="Pink"></button>
            <label class="accustom">Custom <input type="color" id="accentCustom" value="#58a6ff"></label>
            <button id="accentReset" class="acreset">Reset to default</button>
          </div>
        </div>
        <button id="themeToggle" class="themebtn">{_svg('moon',14)} Dark</button>
      </div>
    </div>
  </header>
  <div class="shell">
    <aside class="sidebar" id="sideNav">
      <button data-area="signals" class="on"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1 8h3l2-5 3 10 2-5h4"/></svg> Signals</button>
      <button data-area="markets"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="8" cy="8" r="6.3"/><path d="M1.7 8h12.6M8 1.7c2.4 1.8 2.4 10.8 0 12.6M8 1.7c-2.4 1.8-2.4 10.8 0 12.6"/></svg> Markets</button>
      <button data-area="portfolio"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="5" width="12.4" height="8.5" rx="1.2"/><path d="M5.5 5V3.7a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1V5"/></svg> Portfolio</button>
      <div class="side-sect">Research</div>
      <button data-area="intel"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><rect x="4.5" y="4.5" width="7" height="7" rx="1"/><path d="M6.5 1.8v2.7M9.5 1.8v2.7M6.5 11.5v2.7M9.5 11.5v2.7M1.8 6.5h2.7M1.8 9.5h2.7M11.5 6.5h2.7M11.5 9.5h2.7"/></svg> Intel</button>
      <button data-area="news"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="3" width="12.4" height="10" rx="1.2"/><path d="M4.3 6h7.4M4.3 8.3h7.4M4.3 10.6h4.5"/></svg> News</button>
      <button data-area="track"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M1.5 14V2M1.5 14h13"/><path d="M4 11l3-3 2.4 2L14 4.5"/><path d="M11 4.5h3v3"/></svg> Track record</button>
      <button data-area="analyst"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="7" cy="7" r="4.3"/><path d="M10.1 10.1 14 14"/></svg> Analyst</button>
      <div class="side-sect">System</div>
      <button data-area="livetv"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="4.3" width="12.4" height="8.4" rx="1.4"/><path d="M6 1.6 8 4l2-2.4"/></svg> Live TV</button>
      <button data-area="about"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="8" cy="8" r="6.3"/><path d="M8 7.3v4"/><path d="M8 4.9h.01"/></svg> About</button>
      <button data-area="system"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="2.3"/><path d="M8 1.4v2M8 12.6v2M1.4 8h2M12.6 8h2M3.3 3.3l1.4 1.4M11.3 11.3l1.4 1.4M12.7 3.3l-1.4 1.4M4.7 11.3l-1.4 1.4"/></svg> System</button>
      <button data-area="agents"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="4" r="2.2"/><circle cx="3.6" cy="11.5" r="2.2"/><circle cx="12.4" cy="11.5" r="2.2"/><path d="M8 6.2v3M6.7 5.6 4.9 9.6M9.3 5.6l1.8 4"/></svg> Agents</button>
      <button data-area="whatsnew"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1.5l1.6 3.6 3.9.4-2.9 2.6.8 3.8L8 12.6 4.6 14.5l.8-3.8L2.5 8.1l3.9-.4z"/></svg> What's new</button>
      <div class="side-sig" id="sideSignals" hidden>
        <div class="side-h">{_svg('bolt',12)} Active signals</div>
        <div class="side-sig-list" id="sideSigList"></div>
      </div>
      <div class="side-foot" id="sideFoot" hidden></div>
    </aside>
    <div class="maincol">
      <nav class="toptabs" id="topTabs"></nav>
      <div class="tickertape" aria-label="Live ticker tape"><div class="tkt-track" id="tapeTrack"></div></div>
      <div id="staleBanner" class="stale-banner" style="display:none;"></div>
      <div class="subhead">Built {snap['generated_at']} <span id="builtAgo" style="opacity:.7;"></span> &middot; scanned {snap['scanned']} symbols{health_html}{pdrop_html}</div>
      <div class="subhead" id="marketClock" style="margin-top:-9px;opacity:.85;"></div>
      <div class="note" style="margin-top:0;">{mode_note}</div>
      <div id="diag"></div>

  <section class="page" id="page-markets">
    <div class="mkt">
      <nav class="mkt-side" id="mktNav">
        <button data-mview="chart" class="on">Chart</button>
        <button data-mview="sectors">Sector strength</button>
        <button data-mview="macro">Macro backdrop</button>
      </nav>
      <div class="mkt-main">
        <div class="mkt-view on" id="mview-chart">
          <div class="featured">
            <div class="feat-grid">
              <div class="feat-main"><div id="featuredChart"></div></div>
              <aside class="feat-watch"><div class="feat-wtitle">Watchlist · click to load</div><div id="featWatch"></div></aside>
            </div>
          </div>
        </div>
        <div class="mkt-view" id="mview-sectors">{sectors_html}</div>
        <div class="mkt-view" id="mview-macro">{macro_html}</div>
      </div>
    </div>
  </section>

  <section class="page on" id="page-signals">
    <div class="hero-row">
      {signals_hero_html}
      <div class="hero-side">
        <details class="tvwidget hero-tv-solo" open>
          <summary>{_svg('tv',15)} Live market TV
            <span class="ctlgrp wtvgrp">
              <button class="on" onclick="event.preventDefault();event.stopPropagation();_heroTV(this,'UCEAZeUIeJs0IjQiqTCdVSIg');">Yahoo Finance</button>
              <button onclick="event.preventDefault();event.stopPropagation();_heroTV(this,'UCIALMKvObZNtJ6AmdCLP7Lg');">Bloomberg</button>
            </span>
            <a class="tvw-open" href="#" onclick="event.preventDefault();event.stopPropagation();_gotoTab('livetv');return false;">more channels →</a></summary>
          <div class="tvw-frame"><iframe id="heroTVFrame" src="https://www.youtube.com/embed/live_stream?channel=UCEAZeUIeJs0IjQiqTCdVSIg&amp;autoplay=1&amp;mute=1" title="Live market TV" frameborder="0" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
        </details>
      </div>
    </div>
    {ticker_tape_html}
    {showcase_html}
    {bento_home_html}
    <div class="strat-badge"><span class="k">Strategy type</span><span class="v">Swing · long-only, meta-label filtered · partial at 2R, then trail the winner</span></div>
    <div id="concWarn"></div>
    <div class="sec-eyebrow">Live social</div>
    <div class="sec-head"><span class="sh-ico">{_svg('sparkle',15)}</span><h2>Grok · X pulse</h2><span class="sh-sub">real-time X + news read on the top names</span></div>
    <div id="grokPulse"></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Layout:</span>
      <span class="ctlgrp" id="layoutBtns"></span></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Sort:</span>
      <span class="ctlgrp" id="sortBtns"></span></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Show:</span>
      <span class="ctlgrp" id="filterBtns"></span></div>
    <div class="sec-eyebrow">Signals engine</div>
    <div class="sec-head"><span class="sh-ico">{_svg('sparkle',15)}</span><h2>Live signals</h2>
      <span class="cards-ctl"><span class="sh-sub" id="cardsCount"></span>
      <button class="cbtn" id="cardsPrev" aria-label="Scroll left">&larr;</button>
      <button class="cbtn" id="cardsNext" aria-label="Scroll right">&rarr;</button>
      <button class="cbtn cbtn-txt" id="cardsExpand">Expand all</button></span></div>
    <div id="cards"></div>
  </section>

  <section class="page" id="page-momentum">
    <div class="sec-eyebrow">Rankings</div>
    <div class="sec-head"><span class="sh-ico">{_svg('trend-up',15)}</span><h2>Momentum leaders</h2><span class="sh-sub">dual-momentum ranking · best backtested strategy</span></div>
{momentum_html}
  </section>

  <section class="page" id="page-intraday">
    <div class="sec-eyebrow">Intraday</div>
    <div class="sec-head"><span class="sh-ico">{_svg('bolt',15)}</span><h2>Intraday signals</h2><span class="sh-sub">same engine on intraday bars — faster, noisier</span></div>
    <div id="intradayCards"></div>
  </section>

  <section class="page" id="page-orb">
{orb_html}
  </section>

  <section class="page" id="page-heatmap">
    <div class="sec-eyebrow">Markets</div>
    <div class="sec-head"><span class="sh-ico">{_svg('brick',15)}</span><h2>Market heatmap</h2><span class="sh-sub">whole market by sector, sized by cap, coloured by today's move</span></div>
    <div id="heatmapHost" style="height:78vh;min-height:520px;width:100%;"></div>
  </section>

  <section class="page" id="page-pairs">
{pairs_html}
  </section>

  <section class="page" id="page-portfolio">
    <div class="sec-eyebrow">Book</div>
    <div class="sec-head"><span class="sh-ico">{_svg('briefcase',15)}</span><h2>Portfolio</h2><span class="sh-sub">hypothetical book from today's actionable signals</span></div>
{portfolio_html}
  </section>

  <section class="page" id="page-allweather">
    <div class="sec-head"><span class="sh-ico">{_svg('layers',15)}</span><h2>All Weather</h2><span class="sh-sub">Ray Dalio's risk-balanced all-seasons portfolio</span></div>
{allweather_html}
  </section>

  <section class="page" id="page-ipos">
    <div class="sec-head"><span class="sh-ico">{_svg('sparkle',15)}</span><h2>IPO watch</h2><span class="sh-sub">upcoming listings + pre-IPO buzz</span></div>
{ipo_html}
  </section>

  <section class="page" id="page-track">
{track_html}
  </section>

  <section class="page" id="page-analyst">
{analyst_html}
  </section>

  <section class="page" id="page-altdata">
{altdata_html}
  </section>

  {paper_section}

  <section class="page" id="page-analytics">
{analytics_html}
  </section>

  <section class="page" id="page-premium">
{premium_html}
  </section>

  <section class="page" id="page-brain">
{brain_html}
  </section>

  <section class="page" id="page-control">
{control_html}
  </section>

  <section class="page" id="page-system">
{sysdiag_html}
{metalabel_html}
{system_html}
  </section>

  <section class="page" id="page-agents">
{agents_html}
  </section>

  <section class="page" id="page-whatsnew">
{whatsnew_html}
  </section>

  <section class="page" id="page-method">
    <div class="sec-head"><span class="sh-ico">{_svg('book',15)}</span><h2>How it works</h2><span class="sh-sub">the method, end to end</span></div>
    <div class="method">
      <h4>The big picture</h4>
      <p>This page is an automated <b>stock screen</b>. Every weekday (after the US close) it scans a
      curated list of major stocks plus the day's biggest movers, and flags the ones that look like
      they're <b>starting to trend upward</b>, using a simple, well-known momentum strategy. It's a
      research tool to tell you <i>where to look</i> — not a tip service.</p>

      <h4>What we're looking for</h4>
      <p>Stocks where a short-term price trend is overtaking the longer-term trend — the classic early
      sign of a move higher — and where momentum and trading activity back that up.</p>

      <h4>The strategy: multi-strategy confluence, both directions</h4>
      <p>Rather than wait for one rare event (a single moving-average crossover), the engine runs a
      panel of well-known, independent strategies on each stock and asks <b>how many agree right now</b>,
      then filters by the trend regime and a conviction floor. This surfaces real setups far more
      often while keeping only the strong ones labelled actionable.</p>
      <ol>
        <li><b>Scan</b> — curated large-caps plus the day's most-active stocks and biggest movers.</li>
        <li><b>Buy signal</b> — price is in an <b>uptrend</b> (above its 200-day average) and <b>3+ independent
        strategies</b> line up long, and the setup clears a Medium-or-better conviction score. Weaker
        setups appear as <span class="pill">Watch</span> rather than a buy.</li>
        <li><b>Short signal</b> — the mirror image: price in a <b>downtrend</b> with 3+ strategies lined up
        short and conviction clearing the bar. Shorts profit if the stock falls — and carry higher risk,
        so they're gated the same way. There are also <span class="pill">Exit</span> (sell a long that
        just rolled over) and <span class="pill">Avoid</span> (weak, stay away) alerts.</li>
        <li><b>Risk first</b> — every setup gets a <b>stop-loss</b> (~{snap['params']['stop_loss_pct']:.0%} the wrong
        way) and an <b>honest target</b> — the nearest real level price has to clear (recent swing high/low),
        bounded by the analyst price target and a volatility-reachable distance, and never more than {snap['params']['take_profit_pct']:.0%} — sized so a stop-out
        costs only about {snap['params']['risk_per_trade']:.0%} of the account. For a long the stop sits below entry and
        the target above; for a short it's inverted.</li>
      </ol>

      <h4>How each signal is graded (multi-factor confluence)</h4>
      <p>A good trade rarely rests on one signal. Each stock is scored on several factors, the way a
      desk trader weighs confluence:</p>
      <ul>
        <li><b>Trend</b> — short-term average above the long-term one (direction).</li>
        <li><b>Momentum</b> — RSI (overbought/oversold) and MACD (is momentum building?).</li>
        <li><b>Trend strength</b> — ADX, to tell a real trend from chop.</li>
        <li><b>Volume</b> — heavier-than-usual trading confirms a move.</li>
        <li><b>Where price sits</b> — Bollinger band position, distance from 1-year highs/lows, and
        whether it's stretched (chasing) or pulling back to the trend.</li>
        <li><b>Risk : reward</b> — the target must pay enough for the risk taken.</li>
        <li><b>Historical edge</b> — we <i>backtest this exact strategy on that stock's own history</i>
        and factor in how often it has actually worked there.</li>
        <li><b>Strategy confluence</b> — the core of the engine. We run seven <i>independent</i> strategies
        in each direction: long (trend crossover, golden cross, Donchian breakout, MACD momentum, RSI-2
        dip-buy, Bollinger squeeze breakout, EMA momentum stack) and their bearish mirrors (death cross,
        breakdowns, RSI-2 rip-sell, etc.). When 3+ agree <i>and</i> price is in the matching trend,
        the setup becomes actionable; 2 agreeing is a Watch. The detail panel shows which are firing.</li>
        <li><b>Independent cross-check (TradingView)</b> — TradingView's own aggregate technical rating
        (daily + weekly), as a second opinion that's separate from our engine.</li>
        <li><b>News &amp; analysts</b> — recent news tone, the analyst consensus and average price target, plus
        <b>recent rating changes</b> (upgrades/downgrades and the firm behind them).</li>
        <li><b>Earnings momentum &amp; quality</b> — EPS and revenue growth, margins and leverage. Growing
        fundamentals back a long (and fight a short); shrinking ones do the reverse.</li>
        <li><b>Liquidity / execution quality</b> — average dollar turnover and an estimated spread. A name
        that's too thin to fill cleanly is flagged, and in paper trading its size is capped (or skipped)
        so the trade is actually practical — microstructure improving <i>execution</i>, not selection.</li>
        <li><b>Insider activity (SEC Form 4)</b> — clusters of open-market insider <i>purchases</i> raise a
        long's conviction (and lower a short's); heavy insider selling leans the other way.</li>
        <li><b>Retail buzz (StockTwits)</b> — crowd chatter and Bull/Bear sentiment, weighted gently since
        it's noisy and often contrarian.</li>
        <li><b>Short interest / squeeze risk (Yahoo)</b> — how heavily a name is shorted (% of float, days-to-cover).
        A crowded short is squeeze fuel for a long (a tailwind) and a real danger for a fresh short.</li>
        <li><b>Retail / social attention (Reddit &amp; WSB, via ApeWisdom)</b> — names the retail crowd is piling into.
        The lightest nudge of all: a mention spike adds momentum and volatility, so it gently helps a long and
        warns a short. Never a primary driver.</li>
        <li><b>Market alignment</b> — is the trade running with the broad tape (Risk-on/off) or against it?
        Counter-trend setups lose points.</li>
        <li><b>Earnings gate</b> — a fresh entry within ~2 days of an earnings report is held back (capped
        out of High conviction): a binary report can gap straight through the stop.</li>
      </ul>
      <p>The Signals page also warns when <b>too many fresh signals cluster in one sector</b> (often the same
      macro bet in disguise), and the <b>Data signals</b> tab explains and lists what the insider / rating /
      buzz scrapers found today.</p>
      <p>The detail panel also flags <b>chart patterns</b> (golden cross, breakouts, pullbacks, MACD
      crosses, oversold bounces…) and reads the <b>market backdrop</b> — overall breadth (how many
      stocks are trending up) and which <b>sectors</b> are strongest — because signals work better when
      the broader tape agrees.</p>

      <h4>How to use it</h4>
      <p>Each card shows the action and a <b>conviction score</b> (how well it fits the rules). Click any
      card for the full breakdown: a plain-English explanation, the trade plan (entry, stop, target,
      risk:reward), a chart marking where the strategy would have bought/sold, and recent news.</p>

      <h4>Proving it out — paper trading &amp; honest backtests</h4>
      <p>The <b>Track record</b> tab logs every call and grades it against real prices (hypothetical — no fees).
      The <b>Paper account</b> tab goes further: when enabled, fresh High-conviction signals are auto-submitted
      as bracket orders to a real Alpaca <b>paper</b> account, so you see actual fills, slippage and P&amp;L — the
      honest counterpart to the hypothetical log. And the <b>Momentum</b> tab leads with a
      <b>survivorship-bias-free</b> backtest (run on a fixed universe of always-alive ETFs) so its headline
      Sharpe/return can't be flattered by today's winners.</p>

      <h4>Trusting the backtest — walk-forward / out-of-sample validation</h4>
      <p>A single backtest sees the whole history, so any setting that happened to fit the past looks good — that's
      curve-fitting. The <b>Momentum</b> tab now also runs a <b>walk-forward</b> test: it tunes the strategy on a
      slice of <i>past</i> data, then trades the <i>next, unseen</i> slice with those frozen settings, and repeats
      rolling forward. The result is an honest <b>out-of-sample</b> read plus a verdict — <i>holds up</i>,
      <i>marginal</i>, or <i>fragile</i> — and a parameter-sensitivity sweep that shows whether the edge depends on
      one lucky setting. If a strategy only shines in-sample, this is where it gets exposed.</p>

      <h4>The intelligence layer — regimes, ranking &amp; learning</h4>
      <p>On top of the per-stock signals sit a few adaptive layers. The <b>macro regime classifier</b> (Markets tab)
      reads the backdrop and labels it risk-on / neutral / risk-off plus secondary tags — <i>high-volatility,
      recessionary, inflationary, liquidity-driven</i> — and uses that to set an exposure dial <i>and</i> tilt which
      strategies to lean on (momentum in risk-on, mean-reversion/pairs when it's choppy). The <b>adaptive ranking</b>
      (Portfolio tab → "Top opportunities") scores every actionable name 0–100 for where capital should go first,
      blending conviction quality, volatility-adjusted reward, macro fit, liquidity and momentum — so a high-conviction
      but illiquid or poorly-paying setup is correctly ranked below a cleaner one. And a <b>feedback loop</b> tags every
      logged trade with the macro regime and score at entry, so the Track record tab can show which regimes each
      strategy actually works in as results accrue. There's also a <b>no-trade layer</b> (Markets tab &rarr; "No-trade
      check") that makes the bot sit on its hands when conditions are poor — a major data release due that day, panic-level
      volatility, a deteriorating track record, or a drawdown breach — even if a signal fires. It's all transparent rules
      today; that logged history is also the foundation for adding machine-learning scoring later — and even then, every
      decision still passes through the rules-based risk engine, which always has the final say.</p>
      <p>Two more layers sit on top. A <b>meta-signal model</b> gives every candidate a second opinion —
      <i>accept, reduce, delay</i> or <i>reject</i> — weighing regime fit, liquidity, conflicting signals and how
      that regime has paid off before; a "reduce" halves the size, a "delay/reject" skips it. Each trade is then
      written out as a <b>structured record</b> (Portfolio tab) with its confidence, expected return range, holding
      period, risk and <b>uncertainty</b> scores — and when uncertainty is high (signals disagree, macro mixed,
      liquidity thin) the meta-model is what trims or skips it. Finally, an <b>AI news read</b> uses the language
      model to turn recent headlines into structured scores (guidance, demand, margins, regulatory risk and so on);
      it never places a trade — it just feeds the meta-model, so genuinely bad news quietly shrinks position size.
      The whole point is to be <i>more selective, not more active</i> — fewer, better trades.</p>

      <h4>Macro sets the exposure dial (not the trades)</h4>
      <p>Macro data — the VIX, the yield curve, credit spreads, the dollar, plus overall market breadth —
      never directly buys or sells anything. Instead it's blended into one <b>risk-on / neutral / risk-off</b>
      posture that sets an <b>exposure multiplier</b>: in a risk-on backdrop new positions are sized a little
      larger and lean into momentum; in risk-off they're sized smaller, with more cash and a defensive tilt.
      You can see the posture, the multiplier, and the drivers behind it on the <b>Markets</b> tab. The core
      principle: macro controls <i>how much</i> you deploy, security-level data controls <i>what</i> you pick,
      and liquidity controls <i>whether the trade is practical</i>.</p>

      <h4>Protecting the whole book — the risk engine &amp; kill switch</h4>
      <p>Stops protect a single trade; the <b>risk engine</b> protects the whole account. Before any new paper order
      it checks the book and can throttle or stand down:</p>
      <ul>
        <li><b>Daily loss limit</b> — once the day is down ~3%, it stops opening new positions (open trades keep their brackets).</li>
        <li><b>Drawdown control</b> — at ~8% peak-to-now drawdown it <b>halves</b> new-position size; at ~10% it
        <b>halts</b> new entries until equity recovers.</li>
        <li><b>Concentration cap</b> — no single position is allowed to exceed ~15% of equity.</li>
        <li><b>Kill switch</b> — repeated run failures (broker/data outages) flip a hard stop, which auto-resets after
        a few clean runs — so a glitch can never trigger runaway trading.</li>
      </ul>
      <p>Its current state shows as a colour-coded banner at the top of the <b>Paper account</b> tab
      (green normal, amber de-risking, red halt, or the kill switch).</p>

      <h4>A diversifier for flat markets — pairs &amp; mean-reversion</h4>
      <p>The core engine trades <i>direction</i> (trend + momentum), which struggles when the market goes sideways.
      The <b>Pairs</b> tab adds a market-neutral complement: it watches economically-related, liquid pairs
      (e.g. KO/PEP, GS/MS, V/MA), and when the price <b>spread</b> between two normally-linked names stretches
      unusually far — about <b>2 standard deviations</b> from its norm — it flags a bet that the gap closes again
      (long the cheap leg, short the rich one). It only lists a pair once the two legs are genuinely correlated and
      the spread is reliably <i>mean-reverting</i>; it exits as the spread reverts toward normal and stops out if the
      relationship breaks (past ~3σ). It leans in when the broad tape is trendless and steps back when there's a
      strong trend to ride. When a spread reaches its entry band you also get a <b>phone alert</b>, just like signal alerts.</p>

      <h4>Honest limits</h4>
      <p>This is an <b>educational tool, not financial advice</b>. Signals are often wrong, the data is
      free and slightly delayed, and the numbers ignore fees and slippage. The extra inputs above (insider,
      buzz, ratings) are <i>context, not certainty</i> — they tilt conviction, they don't guarantee anything.
      Treat it all as a starting point for your own research — never risk money you can't afford to lose.</p>
    </div>
  </section>

  <section class="page" id="page-news">
    <div class="sec-eyebrow">News</div>
    <div class="sec-head"><span class="sh-ico">{_svg('news',15)}</span><h2>Market news</h2><span class="sh-sub">recent headlines across the scanned stocks</span></div>
{news_ideas_html}
    <ul class="news" id="news"></ul>
  </section>

  <section class="page" id="page-livetv">
    <div class="sec-eyebrow">Watch</div>
    <div class="sec-head"><span class="sh-ico">{_svg('tv',15)}</span><h2>Live TV</h2><span class="sh-sub">live financial news streams</span></div>
    <div class="tvwrap"><iframe id="tvFrame" title="Live financial news" frameborder="0" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <p style="color:var(--muted);font-size:12px;margin-top:10px;">Start muted; unmute in the player. If a stream is blank (it restarted with a new ID) or you want the full experience, <a id="tvLink" href="https://www.youtube.com/@markets/live" target="_blank" rel="noopener">open it on YouTube ↗</a>. Not affiliated with these networks; embedded for convenience.</p>
    <div class="sec-head"><span class="sh-ico">{_svg('tv',15)}</span><h2>Channels</h2><span class="sh-sub">tap to play</span></div>
    <div class="tvcards" id="tvCards"></div>
  </section>

  <div class="disclaimer">
    Strategy: multi-strategy confluence (7 long + 7 short), trend-gated (200-day) with a
    conviction floor; {snap['params']['fast_ma']}/{snap['params']['slow_ma']} SMA + RSI({snap['params']['rsi_period']}) is one input.
    Risk {snap['params']['risk_per_trade']:.0%}/trade, stop {snap['params']['stop_loss_pct']:.0%}, target = nearest structure (bounded by fundamentals &amp; volatility, capped {snap['params']['take_profit_pct']:.0%}).
    Shorts profit if price falls and carry higher risk.
    "Rel vol" = today's volume vs its {snap['params']['rel_volume_window']}-day average — a free
    proxy for unusual activity, NOT real institutional/options order flow.<br>
    Educational tool only. Not financial advice. Signals can be wrong; backtests ignore
    fees and slippage. Verify before acting and never risk money you can't afford to lose.
  </div>
    </div>
  </div>
</div>

<div class="overlay gk-overlay" id="grokOverlay">
  <div class="modal gk-modal" role="dialog" aria-modal="true">
    <button class="mclose" id="grokClose" aria-label="Close">{_svg('x',18)}</button>
    <div id="grokBody"></div>
  </div>
</div>

<div class="overlay" id="overlay">
  <div class="modal modal-wide" role="dialog" aria-modal="true">
    <header class="mhead" id="mHead">
      <div class="mhead-id">
        <div class="mhead-logo" id="mLogo"></div>
        <div class="mhead-idtext">
          <div class="mhead-tickrow">
            <span class="mhead-tick" id="mTick"></span>
            <span class="mhead-pill" id="mPill"></span>
          </div>
          <div class="mhead-name" id="mName"></div>
        </div>
      </div>
      <div class="mhead-quote">
        <div class="mhead-px" id="mPx"></div>
        <div class="mhead-chg" id="mChg"></div>
      </div>
      <button class="mclose" id="modalClose" aria-label="Close">{_svg('x',18)}</button>
    </header>
    <h3 id="mTitle" hidden></h3>
    <div class="summary" id="mSummary"></div>
    <nav class="mk-top" id="mkTop">
      <button data-top="overview" class="on">{_svg('clipboard',14)} Overview</button>
      <button data-top="chart">{_svg('chart',14)} Chart</button>
      <button data-top="trade">{_svg('target',14)} Trade</button>
      <button data-top="intel">{_svg('ai',14)} Intelligence</button>
      <button data-top="research">{_svg('search',14)} Research</button>
    </nav>
    <div class="mk">
      <nav class="mk-side" id="mkNav">
        <button data-top="overview" data-mkview="overview" class="on">Summary</button>
        <button data-top="chart" data-mkview="chart">Chart</button>
        <button data-top="trade" data-mkview="plan">Plan</button>
        <button data-top="trade" data-mkview="risk">Risk &amp; sizing</button>
        <button data-top="trade" data-mkview="exec">Execution</button>
        <button data-top="intel" data-mkview="meta">Meta verdict</button>
        <button data-top="intel" data-mkview="regimefit">Regime fit</button>
        <button data-top="intel" data-mkview="newsread">AI news read</button>
        <button data-top="intel" data-mkview="rank">Adaptive rank</button>
        <button data-top="research" data-mkview="strategies">Strategies</button>
        <button data-top="research" data-mkview="signals">Signal inputs</button>
        <button data-top="research" data-mkview="research">Fundamentals &amp; news</button>
      </nav>
      <div class="mk-main">
        <div class="mk-view on" id="mkview-overview">
          <div class="msparkwrap" id="mSpark"></div>
          <div class="mverdict" id="mVerdict"></div>
          <div class="mplan-strip" id="mPlanTop"></div>
          <div class="sech ai-ident" id="mAIHead" style="display:none;">{_svg('ai',13)} In plain English (AI)</div>
          <div class="deskread ai-read" id="mAI" style="display:none;border-left-color:var(--ai);"></div>
          <div class="sech">The bottom line</div>
          <div class="deskread" id="mDesk"></div>
          <div class="sech">Should you take it? <span id="mConvScore"></span></div>
          <ul class="checks" id="mChecks"></ul>
          <div class="sech">Patterns spotted</div>
          <div class="chips" id="mPatterns"></div>
        </div>
        <div class="mk-view" id="mkview-chart">
          <div id="modalChart"></div>
        </div>
        <div class="mk-view" id="mkview-plan">
          <div class="sech" style="margin-top:0;">The trade plan <span id="mPlanNote" style="text-transform:none;color:var(--muted);"></span></div>
          <div class="plangrid" id="mPlan"></div>
          <div class="sech">How this strategy has done on this stock <span style="text-transform:none;color:var(--muted);">(backtest)</span></div>
          <div class="plangrid" id="mEdge"></div>
          <div class="sech">Market context</div>
          <div class="plangrid" id="mContext"></div>
        </div>
        <div class="mk-view" id="mkview-strategies">
          <div class="sech" style="margin-top:0;">Strategies in play <span style="text-transform:none;color:var(--muted);">— independent methods + their track record here</span></div>
          <div id="mStrategies"></div>
        </div>
        <div class="mk-view" id="mkview-signals">
          <div class="sech" style="margin-top:0;">Signal inputs <span style="text-transform:none;color:var(--muted);">— the extra data feeding this call, in detail</span></div>
          <div id="mSignals"></div>
        </div>
        <div class="mk-view" id="mkview-research">
          <div class="sech" style="margin-top:0;">Analysts, fundamentals &amp; news tone</div>
          <div class="plangrid" id="mResearch"></div>
          <div class="sech">Latest news on this stock</div>
          <ul class="news" id="mNews"></ul>
          <div class="sech">The details, explained</div>
          <ul class="reasons" id="mReasons"></ul>
        </div>
        <div class="mk-view" id="mkview-risk">
          <div class="sech" style="margin-top:0;">Risk &amp; sizing <span style="text-transform:none;color:var(--muted);">— the structured signal contract</span></div>
          <div class="plangrid" id="mRisk"></div>
          <div class="sech">Kill conditions</div>
          <div id="mKill" style="font-size:13px;color:var(--txt2);"></div>
        </div>
        <div class="mk-view" id="mkview-exec">
          <div class="sech" style="margin-top:0;">Execution quality <span style="text-transform:none;color:var(--muted);">— can this be traded cleanly?</span></div>
          <div class="plangrid" id="mExec"></div>
        </div>
        <div class="mk-view" id="mkview-meta">
          <div class="sech" style="margin-top:0;">Meta-signal verdict <span style="text-transform:none;color:var(--muted);">— the second opinion on this trade</span></div>
          <div id="mMeta"></div>
        </div>
        <div class="mk-view" id="mkview-regimefit">
          <div class="sech" style="margin-top:0;">Macro &amp; regime fit</div>
          <div id="mRegimeFit"></div>
        </div>
        <div class="mk-view" id="mkview-newsread">
          <div class="sech" style="margin-top:0;">AI news read <span style="text-transform:none;color:var(--muted);">— headlines turned into structured scores</span></div>
          <div id="mNewsRead"></div>
        </div>
        <div class="mk-view" id="mkview-rank">
          <div class="sech" style="margin-top:0;">Adaptive allocation rank</div>
          <div id="mRank"></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const LIVE_URL = "{CONFIG.live_quotes_url}";
let LIVE = {{}};  // latest live prices (declared early so renderCards can read it safely)
let featTC = null, modalTC = null;   // Capital IQ-style chart engine instances
window.__APP = {{ DATA: DATA, LIVE_URL: LIVE_URL }};
// --- inline SVG icon library (mirrors the Python _ICON_PATHS map) ---
const ICON = {icon_js};
function _ico(name, size, cls) {{
  const p = ICON[name] || ICON.dot;
  size = size || 16;
  return `<svg class="ico ${{cls||''}}" width="${{size}}" height="${{size}}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${{p}}</svg>`;
}}
// read a CSS theme variable (so the overview chart flips with light/dark)
function _cv(n, f) {{ try {{ const v = getComputedStyle(document.documentElement).getPropertyValue(n).trim(); return v || f; }} catch (e) {{ return f; }} }}
function _shortDate(s) {{ try {{ return new Date(s + 'T00:00:00').toLocaleDateString([], {{month:'short', day:'numeric'}}); }} catch (e) {{ return s; }} }}
let _curSort = 'sector';
try {{ _curSort = localStorage.getItem('sort') || 'sector'; }} catch(e) {{}}
let _curFilter = 'all';
try {{ _curFilter = localStorage.getItem('filter') || 'all'; }} catch(e) {{}}
let _searchTerm = '';  // live ticker-search term from the app-bar search box
let FAVS = new Set();
try {{ FAVS = new Set(JSON.parse(localStorage.getItem('tb-favs') || '[]')); }} catch (e) {{}}
function _toggleFav(sym) {{
  if (FAVS.has(sym)) FAVS.delete(sym); else FAVS.add(sym);
  try {{ localStorage.setItem('tb-favs', JSON.stringify([...FAVS])); }} catch (e) {{}}
}}
// ---- Control panel state (browser-side): settings + accept/reject decisions ----
const _CTRL_DEFAULTS = {{ floor: 52, cap: 6, ext: true, vol: true, shorts: false }};
let _CONTROLS = Object.assign({{}}, _CTRL_DEFAULTS);
let _DEC = {{ accepted: [], rejected: [] }};
try {{ _CONTROLS = Object.assign({{}}, _CTRL_DEFAULTS, JSON.parse(localStorage.getItem('tb-controls') || '{{}}')); }} catch (e) {{}}
try {{ _DEC = Object.assign({{accepted:[],rejected:[]}}, JSON.parse(localStorage.getItem('tb-decisions') || '{{}}')); }} catch (e) {{}}
function _saveCtrl() {{ try {{ localStorage.setItem('tb-controls', JSON.stringify(_CONTROLS)); }} catch (e) {{}} }}
function _saveDec() {{ try {{ localStorage.setItem('tb-decisions', JSON.stringify(_DEC)); }} catch (e) {{}} }}
function _decide(sym, verdict) {{
  _DEC.accepted = _DEC.accepted.filter(s => s !== sym);
  _DEC.rejected = _DEC.rejected.filter(s => s !== sym);
  if (verdict === 'accept') _DEC.accepted.push(sym);
  else if (verdict === 'reject') _DEC.rejected.push(sym);
  _saveDec();
  document.querySelectorAll('[data-sym="' + sym + '"]').forEach(el => {{
    el.classList.toggle('accepted', verdict === 'accept');
    el.classList.toggle('rejected', verdict === 'reject');
  }});
  if (window._renderControl) _renderControl();
}}
// Plain-English explanations shown on hover for every strategy + type, across the app.
const STRAT_INFO = {{
  'Trend crossover': 'A short-term average price crosses ABOVE a longer-term one — a classic early sign an uptrend is starting.',
  'Golden cross': 'The 50-day average rises above the 200-day — a slow, big-picture signal the long-term trend has turned up.',
  'Donchian breakout': 'Price pushes above its highest level of the last 20 days — buyers breaking it out to fresh short-term highs.',
  'MACD momentum': 'A popular momentum gauge turns positive — the upward speed of the move is building.',
  'Dip buy (RSI-2)': 'Inside an existing uptrend, price dips hard for a day or two — a chance to buy the pullback before it resumes.',
  'Squeeze breakout': 'After a quiet, low-volatility stretch, price pops out of its range — pent-up energy releasing into a move up.',
  'EMA momentum stack': 'Fast averages line up above slow ones (8 > 21 > 50) — a tidy, healthy uptrend with momentum behind it.',
  'Trend cross-down': 'A short-term average crosses BELOW a longer-term one — a classic early sign a downtrend is starting.',
  'Death cross': 'The 50-day average falls below the 200-day — a slow, big-picture signal the long-term trend has turned down.',
  'Donchian breakdown': 'Price breaks below its lowest level of the last 20 days — sellers pushing it to fresh short-term lows.',
  'MACD momentum (down)': 'The momentum gauge turns negative — the downward speed of the move is building.',
  'Rip-sell (RSI-2)': 'Inside a downtrend, price spikes up sharply for a day or two — a chance to short the bounce before it rolls back over.',
  'Squeeze breakdown': 'After a quiet stretch, price drops out of its range to the downside — pent-up energy releasing into a fall.',
  'EMA momentum stack (down)': 'Fast averages line up below slow ones (8 < 21 < 50) — a clean downtrend with momentum behind it.',
}};
const TYPE_INFO = {{
  'trend': 'Trend-following: aims to ride a sustained move in one direction. Great in trending markets, whipsaws in choppy ones.',
  'momentum': 'Momentum: bets that recent strength (or weakness) keeps going a while longer.',
  'breakout': 'Breakout: acts when price escapes its recent range to a new high, expecting the move to continue.',
  'breakdown': 'Breakdown: acts when price escapes its recent range to a new low, expecting the fall to continue.',
  'mean-reversion': 'Mean-reversion: bets a short, sharp move snaps back toward the average — buy dips in uptrends, sell spikes in downtrends.',
}};
const FAMILY_INFO = {{
  'Trend-following': TYPE_INFO['trend'], 'Momentum': TYPE_INFO['momentum'],
  'Breakout': 'Breakout: acts when price escapes its recent range (a new high for longs, new low for shorts), expecting the move to continue.',
  'Mean-reversion': TYPE_INFO['mean-reversion'], 'Trend filter': TYPE_INFO['trend'],
}};
const _esc = t => String(t||'').replace(/"/g, '&quot;');
// render the LLM markdown subset (**bold**, *italic*, # headings, paragraphs) to safe HTML
function _md(t) {{
  let s = String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s = s.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  s = s.replace(/^\\s*#{{1,6}}\\s*(.+)$/gm, '<strong class="mdh">$1</strong>');
  s = s.replace(/(^|[^*])\\*([^*\\n]+?)\\*(?!\\*)/g, '$1<em>$2</em>');
  return s.split(/\\n\\s*\\n/).map(p => '<p>'+p.replace(/\\n/g,'<br>')+'</p>').join('');
}}
const _mdStrip = t => String(t||'').replace(/[#*`>_]/g,'').replace(/\\s+/g,' ').trim();
function _leadSentence(md) {{
  let t = String(md||'').replace(/\\r/g,'');
  t = t.replace(/^\\s*#{{1,6}}[^\\n]*\\n+/, '');          // drop leading markdown heading line
  t = t.replace(/^\\s*\\*\\*[^*]+\\*\\*:?\\s*/, '');       // drop a leading **Label:** prefix
  t = t.replace(/[#*`>_]/g,'').replace(/\\s+/g,' ').trim();
  const m = t.match(/^(.+?[.!?])(?:\\s|$)/);           // first real sentence
  let out = m ? m[1] : t;
  if (out.length > 165) out = out.slice(0,165).replace(/\\s+\\S*$/,'') + '…';
  return out;
}}
// estimated time-to-play-out, labelled for the signal's timeframe (intraday = minutes/hours, daily = sessions)
function _holdTxt(p) {{
  if (!p || p.hold_lo == null || p.hold_hi == null) return '—';
  const mm = ({{'1Min':1,'5Min':5,'15Min':15,'30Min':30,'1Hour':60}})[p.hold_tf || '1Day'];
  if (mm) {{
    const lo = p.hold_lo * mm, hi = p.hold_hi * mm;
    return hi < 90 ? ('~' + lo + '–' + hi + ' min') : ('~' + Math.round(lo/60) + '–' + Math.round(hi/60) + ' hrs');
  }}
  return '~' + p.hold_lo + '–' + p.hold_hi + ' sessions';
}}
// --- ORB day-trade: cards (same look as Signals/Intraday) + shared detail modal ---
const _ORB_FAC = {{breakout:'Clean breakout candle', rvol:'Unusual volume', vwap:'Confirmed by VWAP',
  catalyst:'Real catalyst', market:'Market aligned', liquidity:'Liquid enough', volatility:'OR width tradable'}};
function _orbBars(comp) {{
  comp = comp || {{}};
  return Object.keys(_ORB_FAC).map(k => {{
    const v = Math.max(0, Math.min(100, Math.round(comp[k] || 0)));
    const c = v >= 60 ? 'var(--buy)' : v < 40 ? 'var(--sell)' : 'var(--accent)';
    return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px;">`
      + `<span style="flex:0 0 150px;color:var(--muted);">${{_ORB_FAC[k]}}</span>`
      + `<span style="flex:1;height:7px;background:var(--line);border-radius:4px;overflow:hidden;"><i style="display:block;height:100%;width:${{v}}%;background:${{c}};"></i></span>`
      + `<span style="flex:0 0 26px;text-align:right;font-variant-numeric:tabular-nums;">${{v}}</span></div>`;
  }}).join('');
}}
function makeOrbCard(s) {{
  const el = document.createElement('div'); el.className = 'card';
  const dir = s.direction, cls = dir === 'LONG' ? 'BUY' : 'SHORT';
  const sc = s.score || 0, ccol = sc >= 75 ? 'var(--buy)' : sc >= 65 ? 'var(--accent)' : 'var(--muted)';
  const logo = `<img class="logo" src="https://assets.parqet.com/logos/symbol/${{s.symbol}}?format=png" alt="" onerror="this.style.display='none'">`;
  const ladder = `<div class="ladder">`
    + `<div class="lrow"><span>Entry</span><b>$${{(s.entry||0).toLocaleString()}}</b></div>`
    + `<div class="lrow"><span>Stop</span><b class="sell">$${{(s.stop||0).toLocaleString()}}</b></div>`
    + `<div class="lrow"><span>Target</span><b class="buy">$${{(s.target||0).toLocaleString()}}</b></div>`
    + `<div class="lrow"><span>Reward : risk</span><b>${{s.rr}} : 1</b></div></div>`;
  const blocked = s.risk_blocked ? `<div class="ed-warn">${{_ico('octagon',12)}} ${{s.risk_block_reason||'risk-blocked'}}</div>` : '';
  el.innerHTML = `
    <div class="card-top">${{logo}}
      <div class="card-id"><div class="s">${{s.symbol}}</div><div class="n">${{s.name||''}}</div></div>
      <span class="act a-${{cls}}">ORB ${{dir}}</span></div>
    <div class="card-px-row"><span class="card-px">$${{(s.entry||0).toLocaleString()}}</span> <span style="color:var(--muted);font-size:12px;">entry · ${{s.window_min}}m OR</span></div>
    <div class="conv-wrap"><div class="conv-row"><span>Signal score · ${{s.score_band||''}}</span><span style="color:${{ccol}};font-weight:700;">${{sc}}</span></div>
      <div class="conv-meter"><div class="conv-fill" style="width:${{sc}}%;background:${{ccol}};"></div></div></div>
    ${{ladder}}
    <div class="card-why"><div class="why-h">${{_ico('clipboard',12)}} Why this breakout</div><div class="why-txt">${{(s.reasons||[]).join(' · ')}}</div></div>
    ${{blocked}}
    <div class="more">click for the full 7-factor breakdown ${{_ico('arrow-rt',12)}}</div>`;
  el.addEventListener('click', () => openOrbModal(s));
  return el;
}}
function renderOrb() {{
  const host = document.getElementById('orbCards'); if (!host) return;
  const o = (DATA && DATA.orb) || {{}}, list = o.signals || [];
  if (!list.length) {{
    const msg = o.note || 'Watching for breakouts in the 09:45–10:30 ET window — none yet today.';
    host.innerHTML = `<div style="border:1px dashed var(--line);border-radius:12px;padding:26px 18px;text-align:center;color:var(--muted);font-size:13px;">`
      + `<div style="font-size:22px;margin-bottom:6px;opacity:.6;">◷</div>${{msg}}</div>`;
    return;
  }}
  const grid = document.createElement('div'); grid.className = 'grid';
  list.forEach(s => grid.appendChild(makeOrbCard(s)));
  host.innerHTML = ''; host.appendChild(grid);
}}
function openOrbModal(s) {{
  const top = document.getElementById('mkTop'), nav = document.getElementById('mkNav');
  if (top) top.style.display = 'none';
  if (nav) nav.style.display = 'none';
  document.querySelectorAll('.mk-view').forEach(v => v.classList.remove('on'));
  let ov = document.getElementById('mkview-orb');
  if (!ov) {{ ov = document.createElement('div'); ov.className = 'mk-view'; ov.id = 'mkview-orb';
    const main = document.querySelector('.mk-main'); if (main) main.appendChild(ov); }}
  ov.classList.add('on');
  const dir = s.direction, cls = dir === 'LONG' ? 'BUY' : 'SHORT';
  // populate the shared glass header (same as openModal)
  const _oInit = (s.symbol.replace(/[^A-Za-z]/g,'').slice(0,2) || s.symbol.slice(0,2)).toUpperCase();
  document.getElementById('mLogo').innerHTML =
    `<span class="mhead-init">${{_oInit}}</span>`
    + `<img src="https://assets.parqet.com/logos/symbol/${{s.symbol}}?format=png" alt="" loading="lazy" onerror="this.remove()">`;
  document.getElementById('mTick').textContent = s.symbol;
  document.getElementById('mPill').className = 'mhead-pill a-' + cls;
  document.getElementById('mPill').innerHTML =
    (dir === 'LONG' ? _ico('trend-up',13) : _ico('trend-dn',13)) + `<span>ORB ${{dir}}</span>`;
  document.getElementById('mName').textContent = s.name || '';
  document.getElementById('mPx').innerHTML =
    `<span>$${{Number(s.entry||0).toLocaleString(undefined,{{maximumFractionDigits:2}})}}</span>`;
  document.getElementById('mChg').className = 'mhead-chg'; document.getElementById('mChg').textContent = '';
  document.getElementById('mSummary').textContent =
    `Opening-range breakout · ${{s.window_min}}-min range · VWAP + market confirmed · day-trade, flat by 15:45 ET.`;
  const sc = s.score || 0, ccol = sc >= 75 ? 'var(--buy)' : sc >= 65 ? 'var(--accent)' : 'var(--muted)';
  const stat = (l, v, c) => `<div class="stat"><div class="l">${{l}}</div><div class="v ${{c||''}}" style="font-size:15px;">${{v}}</div></div>`;
  const plan = `<div class="plangrid">`
    + stat('Entry', '$' + (s.entry||0).toLocaleString())
    + stat('Stop', '$' + (s.stop||0).toLocaleString(), 'sell')
    + stat('Target', '$' + (s.target||0).toLocaleString(), 'buy')
    + stat('Reward : risk', s.rr + ' : 1')
    + stat('Risk / share', s.risk_per_share != null ? ('$' + s.risk_per_share) : '—')
    + stat('Risk', s.risk_pct != null ? (s.risk_pct + '%') : '—')
    + stat('Shares', s.qty != null ? s.qty.toLocaleString() : '—')
    + stat('Est. round-trip cost', s.est_cost_pct != null ? (s.est_cost_pct + '%') : '—') + `</div>`;
  const ctx = `<div class="plangrid">`
    + stat('Opening range', `${{s.or_low}} – ${{s.or_high}}`)
    + stat('VWAP at entry', s.vwap_at_entry)
    + stat('ATR (5-min)', s.atr != null ? s.atr : '—')
    + stat('OR width / ATR', s.or_width_atr != null ? s.or_width_atr : '—')
    + stat('Spread', s.spread_pct != null ? (s.spread_pct + '%') : '—')
    + stat('In-play score', s.in_play != null ? s.in_play : '—')
    + stat('Market', s.market_bias_note || '—') + `</div>`;
  const blocked = s.risk_blocked ? `<div class="ed-warn" style="margin-top:10px;">${{_ico('octagon',12)}} ${{s.risk_block_reason||'risk-blocked'}}</div>` : '';
  ov.innerHTML = `
    <div class="sech" style="margin-top:0;">Signal score <span style="text-transform:none;color:var(--muted);">— 0–100, ≥75 = eligible · this is ${{sc}} (${{s.score_band}})</span></div>
    <div class="conv-meter" style="margin:2px 0 4px;"><div class="conv-fill" style="width:${{sc}}%;background:${{ccol}};"></div></div>
    <div class="sech">The 7 factors</div>${{_orbBars(s.score_components)}}
    <div class="sech">The trade plan</div>${{plan}}
    <div class="sech">Levels &amp; context</div>${{ctx}}
    <div class="sech">Why this breakout</div>
    <ul class="reasons">${{(s.reasons||[]).map(r => `<li>${{_esc(r)}}</li>`).join('')}}</ul>${{blocked}}
    <p style="color:var(--muted);font-size:11px;margin-top:10px;">Opening-range breakout, stocks-in-play. Long-only v1, shadow signal (no orders), flat by 15:45 ET. Its own strategy &amp; learning bucket.</p>`;
  overlay.classList.add('open');
  try {{ history.replaceState(null, '', '#' + s.symbol); }} catch (e) {{}}
}}

// Custom tooltip: reliable + instant (native title= is slow and easy to miss). Any element
// with a non-empty data-tip shows it on hover, positioned by the cursor.
const _tipEl = document.createElement('div'); _tipEl.id = 'tip'; document.body.appendChild(_tipEl);
function _placeTip(e) {{
  const pad = 14, w = _tipEl.offsetWidth, h = _tipEl.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
  _tipEl.style.left = Math.max(8, x) + 'px';
  _tipEl.style.top = Math.max(8, y) + 'px';
}}
document.addEventListener('mouseover', e => {{
  const t = e.target.closest && e.target.closest('[data-tip],[data-tiphtml]');
  if (!t) return;
  const h = t.getAttribute('data-tiphtml');
  if (h) {{ _tipEl.innerHTML = h; _tipEl.classList.add('rich'); _tipEl.style.display = 'block'; _placeTip(e); return; }}
  const txt = t.getAttribute('data-tip');
  if (txt) {{ _tipEl.textContent = txt; _tipEl.classList.remove('rich'); _tipEl.style.display = 'block'; _placeTip(e); }}
}});
document.addEventListener('mousemove', e => {{ if (_tipEl.style.display === 'block') _placeTip(e); }});
document.addEventListener('mouseout', e => {{
  if (e.target.closest && e.target.closest('[data-tip],[data-tiphtml]')) _tipEl.style.display = 'none';
}});

const diag = document.getElementById('diag');
if ((DATA.diagnostics||[]).length) {{
  const _hasSignals = (DATA.signals||[]).length > 0;
  if (_hasSignals) {{
    // Signals exist — a few symbols just failed to fetch (e.g. delisted/units). Show a
    // muted, collapsible note instead of a scary "no signals" banner.
    const n = DATA.diagnostics.length;
    diag.innerHTML = '<details style="background:var(--card,#1a1a1a);border:1px solid var(--bd,#333);'
      + 'color:var(--muted,#9aa);border-radius:10px;padding:10px 14px;margin:8px 0 14px;font-size:12px;">'
      + '<summary style="cursor:pointer;">&#9888; ' + n + ' symbol' + (n>1?'s':'')
      + ' skipped this run (data unavailable)</summary><div style="margin-top:8px;">'
      + DATA.diagnostics.map(e => '&bull; '+e).join('<br>') + '</div></details>';
  }} else {{
    diag.innerHTML = '<div style="background:#3a1e1e;border:1px solid #5a1e1e;color:#ff9b9b;'
      + 'border-radius:10px;padding:14px;margin:8px 0 18px;font-size:13px;">'
      + '<b>No signals to show.</b> Diagnostic:<br>'
      + DATA.diagnostics.map(e => '&bull; '+e).join('<br>') + '</div>';
  }}
}}
const cards = document.getElementById('cards');
// direction pill (SVG + word), colour-coded to the action
function _dirPill(action) {{
  const a = action || '';
  const map = {{
    'BUY': ['buy','trend-up','Buy'], 'SHORT': ['short','trend-dn','Short'],
    'HOLD LONG': ['hold','arrow-up','Hold long'], 'HOLD SHORT': ['hold','arrow-dn','Hold short'],
    'WATCH LONG': ['watch','search','Watch long'], 'WATCH SHORT': ['watch','search','Watch short'],
    'EXIT': ['exit','arrow-rt','Exit'], 'AVOID': ['avoid','octagon','Avoid'], 'FLAT': ['flat','dot','Flat']
  }};
  const m = map[a] || ['flat','dot', a || '—'];
  return `<span class="dir-pill dp-${{m[0]}}">${{_ico(m[1],13)}}${{m[2]}}</span>`;
}}
// meta chip (SVG icon + label), tone: '', 'fresh', 'held', 'bad', 'ai'
function _metaChip(icon, label, tone, tip) {{
  return `<span class="meta-chip ${{tone||''}}"${{tip?` data-tip="${{_esc(tip)}}"`:''}}>${{_ico(icon,12)}}${{label}}</span>`;
}}
// Small "first seen" age chip (powered by the persisted first_seen date).
function _ageBit(s) {{
  if (!s.first_seen) return '';
  const txt = s.is_fresh ? 'New today' : (s.days_old === 1 ? '1 day old' : (s.days_old||0) + ' days old');
  return _metaChip('clock', txt, s.is_fresh ? 'fresh' : '', 'First flagged ' + s.first_seen);
}}
function _alertBit(s) {{
  return s.alerted ? _metaChip('bell', 'Alerted', 'fresh', 'An ntfy alert fired for this name today — pinned here so your alerts and the dashboard stay in line') : '';
}}
const HELD = new Set((typeof DATA !== 'undefined' && DATA.paper_held) || []);
function _heldBit(s) {{
  return HELD.has(s.symbol) ? _metaChip('briefcase', 'In paper book', 'held', "You hold this in the paper book — the bot won't open a second position in the same name") : '';
}}
function _cmteBit(s) {{
  const cm = s.committee; if (!cm) return '';
  const ic = cm.verdict === 'accept' ? 'check' : cm.verdict === 'reject' ? 'x' : 'warn';
  const tone = cm.verdict === 'accept' ? 'fresh' : cm.verdict === 'reject' ? 'bad' : '';
  return `<span class="meta-chip ${{tone}}" data-tip="${{_esc('AI trade committee — ' + cm.support + '/4 analysts support: ' + (cm.summary||''))}}">${{_ico('bank',12)}}Committee ${{_ico(ic,11)}} ${{cm.verdict}}</span>`;
}}
function _intradayBit(s) {{
  if (!s.intraday_confirm || s.intraday_confirm === 'none') return '';
  const ok = s.intraday_confirm === 'agree';
  const tf = (DATA.params && DATA.params.intraday_timeframe) || '5m';
  return `<span class="meta-chip ${{ok ? 'fresh' : 'bad'}}" data-tip="${{_esc('Lower-timeframe (' + tf + ') momentum ' + (ok ? 'agrees with' : 'is against') + ' this trade')}}">${{_ico('bolt',12)}}Intraday ${{_ico(ok ? 'check' : 'x',11)}}</span>`;
}}
function makeCard(s) {{
  const el = document.createElement('div'); el.className='card sxcard';
  el.dataset.sym = s.symbol; el.dataset.action = s.action || '';
  el.dataset.dir = s.direction || ''; if (typeof s.p_win === 'number') el.dataset.pwin = s.p_win;
  if (_DEC.rejected.includes(s.symbol)) el.classList.add('rejected');
  if (_DEC.accepted.includes(s.symbol)) el.classList.add('accepted');
  const cls = (s.action||'').replace(' ','');
  const conv = s.conviction || {{}};
  const cpct = conv.score_pct || 0;
  const ccol = _rag(cpct);   // RAG: green ≥70, amber ≥50, red below
  // Prefer Yahoo's consolidated price + previous close so the card matches Google.
  const _px = (s.quote_price != null) ? s.quote_price : s.price;
  const _base = (s.prev_close != null) ? s.prev_close : s.price;
  const _hasQ = (s.quote_price != null && s.prev_close);
  const _dc = _hasQ ? (s.quote_price / s.prev_close - 1) * 100 : (s.context && s.context.day_change_pct);
  const _lab = _hasQ ? 'today' : ('on ' + _shortDate(s.as_of));
  // refreshLive() recomputes this live vs the previous close and relabels it "today".
  const dchg = (_dc != null)
    ? `<span class="card-day" data-chg="${{s.symbol}}" data-base="${{_base}}" style="color:${{_dc>=0?'var(--buy)':'var(--sell)'}};">${{_dc>=0?'+':''}}${{_dc.toFixed(2)}}% ${{_lab}}</span>`
    : '';
  const initials = (s.symbol.replace(/[^A-Za-z]/g,'').slice(0,2) || s.symbol.slice(0,2)).toUpperCase();
  const logo = `<span class="card-mono" style="background:var(--inset);color:var(--txt2);">${{initials}}`
    + `<img src="https://financialmodelingprep.com/image-stock/${{s.symbol}}.png" alt="" loading="lazy" onerror="this.remove()" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#fff;">`
    + `</span>`;
  const _isShort = (s.direction === 'SHORT');
  const _cn = (s.strategies&&s.strategies.now) ? s.strategies.now : null;
  const _cs = (s.strategies&&s.strategies.short) ? s.strategies.short : null;
  const ed = (s.fundamentals||{{}}).earnings_days;
  const edGated = (s.conviction||{{}}).earnings_gated;
  const edWarn = edGated
    ? `<div class="card-warn">${{_ico('octagon',13)}} Earnings in ${{ed}}d — held back from a fresh entry (a report this close can gap through the stop)</div>`
    : (ed!=null && ed<=7)
    ? `<div class="card-warn">${{_ico('warn',13)}} Earnings in ${{ed}}d — event risk around the report</div>` : '';
  // direction-aware plan chips: Entry / Target / Stop / R:R as inset tiles.
  const _p = s.plan || {{}};
  let ladder = '';
  if (_p.entry!=null && _p.stop!=null && _p.target!=null) {{
    const _m = v => '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}});
    const ent = `<div class="plan-chip ent"><div class="pc-l">Entry</div><div class="pc-v">${{_m(_p.entry)}}</div></div>`;
    const tgt = `<div class="plan-chip tgt hint" data-tip="${{_esc('Target basis: ' + (_p.target_basis || 'nearest structural level, bounded by fundamentals & volatility'))}}"><div class="pc-l">Target</div><div class="pc-v">${{_isShort?'−':'+'}}${{_p.target_pct}}%</div></div>`;
    const stp = `<div class="plan-chip stp"><div class="pc-l">Stop</div><div class="pc-v">${{_isShort?'+':'−'}}${{_p.stop_pct}}%</div></div>`;
    const rr = `<div class="plan-chip"><div class="pc-l">R : R</div><div class="pc-v">${{_p.rr!=null?('1:'+_p.rr):'—'}}</div></div>`;
    ladder = `<div class="plan-chips">${{ent+tgt+stp+rr}}</div>`;
  }}
  // "Why this signal" — a clear panel naming the strategies behind the decision.
  // Trigger strategies (the catalyst) are filled pills; supporting ones are outlined.
  const _co = _isShort ? _cs : _cn;
  const _actWord = (s.action==='SHORT'||s.action==='HOLD SHORT'||s.action==='WATCH SHORT') ? 'short'
                 : (s.action==='BUY'||s.action==='HOLD LONG'||s.action==='WATCH LONG') ? 'buy' : 'signal';
  // strategy FAMILY (approach) behind this card — derived from the firing strategies' kind.
  const _FAMILY = {{trend:'Trend-following', momentum:'Momentum', breakout:'Breakout',
                    breakdown:'Breakout', 'mean-reversion':'Mean-reversion'}};
  let whyBody = '', famLabel = '';
  if (_co) {{
    const agree = (_isShort ? (_co.short||[]) : (_co.long||[]));
    const fresh = _co.fresh || [];
    if (agree.length) {{
      const pills = agree.slice(0,6).map(n => {{
        const tip = (STRAT_INFO[n]||'') + (fresh.includes(n) ? '  (this is the fresh trigger)' : '');
        return `<span class="why-chip${{fresh.includes(n)?' trig':''}} hint" data-tip="${{_esc(tip)}}">${{n}}</span>`;
      }});
      const extra = agree.length>6 ? `<span class="why-chip more">+${{agree.length-6}}</span>` : '';
      whyBody = `<div class="why-chips">${{pills.join('')}}${{extra}}</div>`;
      // collect the families of the agreeing strategies (most common first)
      const counts = {{}};
      if (_co.results) Object.values(_co.results).forEach(r => {{
        if (_isShort ? r.short : r.long) {{ const f=_FAMILY[r.kind]||r.kind; counts[f]=(counts[f]||0)+1; }}
      }});
      famLabel = Object.keys(counts).sort((a,b)=>counts[b]-counts[a]).slice(0,2).join(' + ') || 'Multi-strategy';
    }}
  }}
  if (!whyBody && s.action==='EXIT') {{ whyBody = `<div class="why-txt">Trend break — its uptrend just rolled over.</div>`; famLabel='Trend-following'; }}
  if (!whyBody && s.action==='AVOID') {{ whyBody = `<div class="why-txt">Below trend with a weak/bearish setup — stay away.</div>`; famLabel='Trend filter'; }}
  const famTip = (famLabel || '').split(' + ').map(f => FAMILY_INFO[f]).filter(Boolean).join('  •  ')
                 || 'the strategy approach behind this signal';
  const famTag = famLabel ? `<span class="why-fam hint" data-tip="${{_esc(famTip)}}">${{famLabel}}</span>` : '';
  const whyHtml = whyBody
    ? `<div class="card-why"><div class="why-h">${{_ico('clipboard',12)}} Why this ${{_actWord}}</div>${{famTag}}${{whyBody}}</div>` : '';
  const nNews = (s.news||[]).length;
  // conviction ring (SVG donut), tier-coloured
  const _cvC = 2 * Math.PI * 20, _cvOff = _cvC * (1 - cpct/100);
  const convRing = conv.label ? `<div class="conv2">
      <div class="conv2-ring"><svg viewBox="0 0 52 52"><circle cx="26" cy="26" r="20" fill="none" stroke="var(--inset)" stroke-width="5"/><circle cx="26" cy="26" r="20" fill="none" stroke="${{ccol}}" stroke-width="5" stroke-linecap="round" stroke-dasharray="${{_cvC}}" stroke-dashoffset="${{_cvOff}}"/></svg><span class="cv" style="color:${{ccol}};">${{cpct}}</span></div>
      <div class="conv2-meta"><div class="conv2-lab">Conviction</div><div class="conv2-tier" style="color:${{ccol}};">${{conv.label}}</div>
        <div class="conv2-bar"><i style="width:${{cpct}}%;background:${{ccol}};"></i></div></div>
    </div>` : '';
  const metaRow = _ageBit(s)+_heldBit(s)+_cmteBit(s)+_alertBit(s)+_intradayBit(s);
  // ---- SpaceX-style card body: header · description · pills · hairline · label/value rows ----
  const _m2 = v => '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}});
  const _agree = _co ? (_isShort ? (_co.short||[]) : (_co.long||[])) : [];
  let _dirWord = _isShort ? 'Short' : 'Long';
  if (s.action==='EXIT') _dirWord='Exit';
  else if (s.action==='AVOID') _dirWord='Avoid';
  else if ((s.action||'').indexOf('WATCH')===0) _dirWord = _isShort ? 'Watch short' : 'Watch';
  let _desc = '';
  if (s.ai_read) {{ _desc = _leadSentence(s.ai_read); }}
  else if (famLabel) {{ _desc = famLabel + ' setup' + (_agree.length ? ' — ' + _agree.length + ' strateg' + (_agree.length>1?'ies':'y') + ' in agreement' : ''); }}
  else if (s.catalyst) {{ _desc = _mdStrip(s.catalyst.headline).slice(0,140); }}
  const _pills = [`<span class="sx-pill sx-dir"><i class="sx-dot ${{_isShort?'s':'l'}}"></i>${{_dirWord}}</span>`];
  if (famLabel) _pills.push(`<span class="sx-pill">${{_esc(famLabel)}}</span>`);
  _agree.slice(0,2).forEach(n => _pills.push(`<span class="sx-pill">${{_esc(n)}}</span>`));
  if (s.catalyst) _pills.push(`<span class="sx-pill">Catalyst</span>`);
  // ---- rich hover callouts for the plan rows (dark box + mini bars, Anthropic-style) ----
  const _entryTip = (_p.entry!=null)
    ? `<div class='cb-wrap'><div class='cb-h'>Entry</div><div class='cb-line'>Suggested fill near <b>${{_m2(_p.entry)}}</b>. Last <b>$${{_px.toLocaleString()}}</b>.</div></div>` : '';
  const _tgtTip = (_p.target_pct!=null)
    ? `<div class='cb-wrap'><div class='cb-h'>Target &middot; ${{_isShort?'−':'+'}}${{_p.target_pct}}%</div><div class='cb-line'>${{_esc(_p.target_basis || 'Nearest structural level, bounded by volatility & fundamentals.')}}</div><div class='cb-cm'>Swing mode: reaching this <b>tightens the trailing stop</b> rather than exiting — winners keep running.</div></div>` : '';
  const _stpTip = (_p.stop_pct!=null)
    ? `<div class='cb-wrap'><div class='cb-h'>Stop &middot; ${{_isShort?'+':'−'}}${{_p.stop_pct}}%</div><div class='cb-line'>${{_esc(_p.stop_basis || 'Below the invalidation level — where the thesis breaks.')}}</div></div>` : '';
  // reward:risk visualised as two stacked mini bars (reward green vs 1 unit of risk red)
  let _rrTip = '';
  if (_p.rr!=null) {{
    const _rw = Math.min(100, (Number(_p.rr)||0) / (Math.max(1, Number(_p.rr)||1) + 1) * 100);
    _rrTip = `<div class='cb-wrap'><div class='cb-h'>Reward : risk &middot; 1:${{_p.rr}}</div>`
      + `<div class='cb-row'><span class='cb-nm'>Reward</span><span class='cb-bar'><i style='width:${{_rw}}%;background:var(--buy);'></i></span><span class='cb-v'>${{_p.rr}}R</span></div>`
      + `<div class='cb-row'><span class='cb-nm'>Risk</span><span class='cb-bar'><i style='width:${{100/(Math.max(1,Number(_p.rr)||1)+1)}}%;background:var(--sell);'></i></span><span class='cb-v'>1R</span></div></div>`;
  }}
  // conviction callout: pass / warn / fail as a stacked mini bar + committee support
  let _convTip = '';
  if (conv.label) {{
    const _ck = conv.checks || [], _tot = _ck.length || 1;
    const _np = _ck.filter(c=>c.status==='pass').length, _nw = _ck.filter(c=>c.status==='warn').length, _nf = _ck.filter(c=>c.status==='fail').length;
    _convTip = `<div class='cb-wrap'><div class='cb-h'>Conviction &middot; ${{cpct}} ${{_esc(conv.label)}}</div>`
      + `<div class='cb-stack'><i style='width:${{100*_np/_tot}}%;background:var(--buy);'></i><i style='width:${{100*_nw/_tot}}%;background:var(--warn);'></i><i style='width:${{100*_nf/_tot}}%;background:var(--sell);'></i></div>`
      + `<div class='cb-leg'><span><b>${{_np}}</b> pass</span><span><b>${{_nw}}</b> warn</span><span><b>${{_nf}}</b> fail</span></div>`
      + (s.committee ? `<div class='cb-cm'>Committee &middot; ${{s.committee.support}}/4 support</div>` : '')
      + `</div>`;
  }}
  const _rowVal = (v, tip) => tip ? `<span class="hint" data-tiphtml="${{_esc(tip)}}">${{v}}</span>` : v;
  const _rows = [];
  if (_p.entry!=null) _rows.push(['Entry', _rowVal(_m2(_p.entry), _entryTip)]);
  if (_p.t1_pct!=null) {{
    const _t1Tip = `<div class='cb-wrap'><div class='cb-h'>Partial exit &middot; T1</div><div class='cb-line'>Book <b>${{Math.round((_p.t1_frac||0.5)*100)}}%</b> at <b>${{_m2(_p.t1)}}</b> (+${{_p.t1_pct}}%), then move the stop to breakeven and let the rest run to target.</div><div class='cb-cm'>Turns a round-trip into a booked win — far targets get noise-stopped ~60% of the time.</div></div>`;
    _rows.push(['Take partial', _rowVal(`<span class="sx-up">${{_isShort?'−':'+'}}${{_p.t1_pct}}%</span> <span style="color:var(--muted);font-weight:400;">· ${{Math.round((_p.t1_frac||0.5)*100)}}%</span>`, _t1Tip)]);
  }}
  if (_p.target_pct!=null) _rows.push(['Target', _rowVal(`<span class="sx-up">${{_isShort?'−':'+'}}${{_p.target_pct}}%</span>`, _tgtTip)]);
  if (_p.stop_pct!=null) _rows.push(['Stop', _rowVal(`<span class="sx-dn">${{_isShort?'+':'−'}}${{_p.stop_pct}}%</span>`, _stpTip)]);
  if (_p.rr!=null) _rows.push(['R : R', _rowVal('1:'+_p.rr, _rrTip)]);
  if (conv.label) _rows.push(['Conviction', _rowVal(`<span style="color:${{ccol}}; font-weight:700;">${{cpct}} · ${{_esc(conv.label)}}</span>`, _convTip)]);
  // Model win-probability (meta-label) — the learned P(this long wins), OOS-validated to rank winners
  // far better than the conviction score. Shown for longs only (the model is long-only).
  if (typeof s.p_win === 'number') {{
    const _pwp = Math.round(s.p_win*100);
    const _pwc = _pwp>=55 ? 'var(--buy)' : (_pwp>=45 ? 'var(--warn)' : 'var(--sell)');
    const _pwTip = `<div class='cb-wrap'><div class='cb-h'>Model win-probability</div>`
      + `<div class='cb-row'><span class='cb-nm'>P(win)</span><span class='cb-bar'><i style='width:${{_pwp}}%;background:${{_pwc}};'></i></span><span class='cb-v'>${{_pwp}}%</span></div>`
      + `<div class='cb-cm'>Learned meta-label (OOS AUC 0.77). ${{s.meta_gated ? 'Below the quality floor — shown as Watch.' : 'Ranks winners better than the conviction score.'}}</div></div>`;
    _rows.push(['Model P(win)', _rowVal(`<span style="color:${{_pwc}}; font-weight:700;">${{_pwp}}%</span>`, _pwTip)]);
  }}
  const _rightMeta = `<div class="sx-meta"><span class="sx-meta-l">Last</span> <span class="sx-meta-v" data-px="${{s.symbol}}">$${{_px.toLocaleString()}}</span>${{dchg?`<div class="sx-meta-chg">${{dchg}}</div>`:''}}</div>`;
  // ---- strategy-mix waffle: one square per independent strategy, filled = agrees with this signal.
  // Hovering the strip pops a dark callout with a mini win-rate bar breakdown of the agreeing methods.
  let confWaffle = '';
  if (_co && _co.results) {{
    const _res = _co.results, _keys = Object.keys(_res);
    if (_keys.length) {{
      const _fresh = _co.fresh || [];
      const _edges = ((s.strategies||{{}}).edges||{{}}).by || {{}};
      const _agreeK = _keys.filter(k => _isShort ? _res[k].short : _res[k].long);
      const _sq = _keys.map(k => {{
        const r = _res[k], agrees = _isShort ? r.short : r.long;
        const cls = agrees ? (_fresh.includes(r.label) ? 'sw-fresh' : 'sw-on') : 'sw-off';
        const st = agrees ? (_fresh.includes(r.label) ? 'fresh trigger' : (_isShort ? 'short' : 'long')) : 'flat';
        return `<span class="sw-sq ${{cls}}" data-tip="${{_esc(r.label + ' — ' + st)}}"></span>`;
      }}).join('');
      let cbRows = '';
      _agreeK.forEach(k => {{
        const r = _res[k], e = _edges[k] || {{}}, wr = (e.win_rate==null) ? null : e.win_rate;
        const bcol = (wr==null) ? 'var(--muted)' : (wr>=50 ? 'var(--buy)' : 'var(--sell)');
        cbRows += `<div class="cb-row"><span class="cb-nm">${{_esc(r.label)}}</span>`
          + `<span class="cb-bar"><i style="width:${{wr==null?6:wr}}%;background:${{bcol}};"></i></span>`
          + `<span class="cb-v">${{wr==null?'–':wr+'%'}}</span></div>`;
      }});
      const callout = `<div class='cb-wrap'><div class='cb-h'>Strategy confluence &middot; ${{_agreeK.length}}/${{_keys.length}} agree</div>`
        + (cbRows || "<div class='cb-empty'>Per-strategy backtests pending.</div>") + `</div>`;
      confWaffle = `<div class="sx-conf${{_isShort?' short':''}} hint" data-tiphtml="${{_esc(callout)}}">`
        + `<div class="sx-conf-h"><span>Strategy confluence</span><span class="sx-conf-n">${{_agreeK.length}}/${{_keys.length}}</span></div>`
        + `<div class="sw-waffle">${{_sq}}</div></div>`;
    }}
  }}
  el.innerHTML = `
    <button class="favbtn ${{FAVS.has(s.symbol)?'on':''}}" title="Save to favorites" aria-label="Save to favorites">${{_ico(FAVS.has(s.symbol)?'star-fill':'star',17)}}</button>
    <div class="ar-btns"><button class="ar-btn ar-yes" title="Accept — keep this trade" aria-label="Accept">${{_ico('check',15)}}</button><button class="ar-btn ar-no" title="Reject — suppress this trade" aria-label="Reject">${{_ico('x',15)}}</button></div>
    <div class="sx-head"><div class="sx-hl">${{logo}}<div class="sx-hid"><div class="sx-title">${{s.symbol}}</div><div class="sx-sub">${{s.name||s.exchange||''}}</div></div></div>${{_rightMeta}}</div>
    ${{_desc ? `<div class="sx-desc">${{_esc(_desc)}}</div>` : ''}}
    ${{_pills.length ? `<div class="sx-pills">${{_pills.join('')}}</div>` : ''}}
    ${{_rows.length ? `<div class="sx-div"></div><div class="sx-rows">${{_rows.map(r=>`<div class="sx-row"><span class="sx-l">${{r[0]}}</span><span class="sx-v">${{r[1]}}</span></div>`).join('')}}</div>` : ''}}
    ${{confWaffle}}
    ${{edWarn}}`;
  const _fb = el.querySelector('.favbtn');
  if (_fb) _fb.addEventListener('click', (e) => {{
    e.stopPropagation(); _toggleFav(s.symbol);
    _fb.innerHTML = _ico(FAVS.has(s.symbol) ? 'star-fill' : 'star', 17); _fb.classList.toggle('on', FAVS.has(s.symbol));
    if (_curFilter === 'favs') renderCards();
  }});
  const _ay = el.querySelector('.ar-yes'), _an = el.querySelector('.ar-no');
  if (_ay) _ay.addEventListener('click', (e) => {{ e.stopPropagation(); _decide(s.symbol, _DEC.accepted.includes(s.symbol) ? 'clear' : 'accept'); }});
  if (_an) _an.addEventListener('click', (e) => {{ e.stopPropagation(); _decide(s.symbol, _DEC.rejected.includes(s.symbol) ? 'clear' : 'reject'); }});
  el.addEventListener('click', () => openModal(s));
  return el;
}}
// --- views: filter / sort the signal cards ---
const _ACT_ORDER = {{'BUY':0, 'SHORT':1, 'HOLD LONG':2, 'HOLD SHORT':3, 'EXIT':4,
                     'WATCH LONG':5, 'WATCH SHORT':6, 'AVOID':7, 'SELL':7, 'FLAT':8}};
const _conv = s => (s.conviction ? s.conviction.score_pct : -1);

// ===== Alternate layouts (view switcher) ===========================================
let _layout = 'cards';
try {{ _layout = localStorage.getItem('layout2') || 'cards'; }} catch(e) {{}}
function _pxOf(s) {{ return (s.quote_price != null) ? s.quote_price : s.price; }}
function _rag(pct) {{ pct = pct||0; return pct>=70 ? 'var(--buy)' : (pct>=50 ? '#c08a1e' : 'var(--sell)'); }}
function _ragT(pct) {{ pct = pct||0; return pct>=70 ? '#33d17a' : (pct>=50 ? '#d0d0d3' : '#ff5c4d'); }}
function _tvBit(s) {{ return (s.tv && s.tv.d) ? (' · TV ' + s.tv.d) : ''; }}  // short tail for compact rows

// Jump to a top-level tab programmatically (used by clickable alt-data badges).
function _gotoTab(p) {{ if (window._showPage) window._showPage(p); }}

// Signals-page TV widget: switch the embedded player between reliably-embeddable channels.
function _wtvSet(btn) {{
  const f = document.getElementById('wtvFrame');
  if (f) f.src = `https://www.youtube.com/embed/${{btn.dataset.wtv}}?autoplay=1&mute=1`;
  document.querySelectorAll('.wtvgrp button').forEach(b => b.classList.toggle('on', b === btn));
}}
// Hero Live TV — channel-based live stream (always plays the channel's CURRENT live, so no dead video IDs).
function _heroTV(btn, channel) {{
  const f = document.getElementById('heroTVFrame');
  if (f) f.src = `https://www.youtube.com/embed/live_stream?channel=${{channel}}&autoplay=1&mute=1`;
  const grp = btn.parentElement;
  if (grp) grp.querySelectorAll('button').forEach(b => b.classList.toggle('on', b === btn));
}}

// Scraped alt-data (SEC insiders / analyst rating change / StockTwits buzz) as a normalised list.
function _altData(s) {{
  const out = [];
  const ins = s.insider;
  if (ins && ins.cluster_buy)
    out.push({{icon:_ico('bank',12), txt:'Insider buys', extra:ins.buys, col:'var(--buy)',
      tip:`SEC Form 4: ${{ins.buys}} recent open-market insider purchase(s), ${{(ins.buy_shares||0).toLocaleString()}} shares.`}});
  const aa = (s.fundamentals||{{}}).analyst_actions, lt = aa && aa.latest;
  if (lt && (lt.action==='up' || lt.action==='down'))
    out.push({{icon: _ico(lt.action==='up'?'arrow-up':'arrow-dn',12), txt: lt.action==='up'?'Upgrade':'Downgrade', extra: lt.firm||'',
      col: lt.action==='up'?'var(--buy)':'var(--sell)',
      tip:`Analyst: ${{_esc(lt.firm||'')}} ${{lt.from?lt.from+' → ':''}}${{_esc(lt.to||'')}} on ${{lt.date}}.`}});
  const b = s.buzz;
  if (b && b.lean)
    out.push({{icon:_ico('chat',12), txt: b.lean==='bull'?'Bullish buzz':b.lean==='bear'?'Bearish buzz':'Mixed buzz',
      extra: b.sentiment_pct!=null?b.sentiment_pct+'%':'',
      col: b.lean==='bull'?'var(--buy)':b.lean==='bear'?'var(--sell)':'var(--muted)',
      tip:`StockTwits: ${{b.n}} recent posts, ${{b.sentiment_pct}}% bullish of tagged. Crowd sentiment — noisy/contrarian.`}});
  const rs = (s.factors||{{}}).rs;
  if (rs && rs.pct!=null)
    out.push({{icon:_ico('trend-up',12), txt:'RS '+rs.pct, extra:'', nojump:true,
      col: rs.pct>=70?'var(--buy)':rs.pct<=40?'var(--sell)':'var(--muted)',
      tip:`Relative-strength percentile ${{rs.pct}} vs the market over recent months — higher = leading, lower = lagging.`}});
  return out;
}}
// Clear, labelled pills for the Cards layout.
function _altPills(s) {{
  const d = _altData(s); if (!d.length) return '';
  return '<div class="altrow">' + d.map(x => x.nojump
    ? `<span class="altpill hint" style="color:${{x.col}};" data-tip="${{x.tip}}">${{x.icon}} ${{x.txt}}${{x.extra!==''&&x.extra!=null?' '+x.extra:''}}</span>`
    : `<span class="altpill hint" style="color:${{x.col}};cursor:pointer;" data-tip="${{x.tip}} · Click for all findings →" onclick="event.stopPropagation();_gotoTab('altdata')">${{x.icon}} ${{x.txt}}${{x.extra!==''&&x.extra!=null?' '+x.extra:''}}</span>`
  ).join('') + '</div>';
}}
// Compact coloured icon strip for dense layouts (Terminal etc.).
function _altMini(s) {{
  const d = _altData(s); if (!d.length) return '';
  return '<div class="bbalt">' + d.map(x => x.nojump
    ? `<span class="hint" style="color:${{x.col}};" data-tip="${{x.tip}}">${{x.icon}} ${{x.extra!==''&&x.extra!=null?x.extra:x.txt}}</span>`
    : `<span class="hint" style="color:${{x.col}};cursor:pointer;" data-tip="${{x.tip}} · Click for all findings →" onclick="event.stopPropagation();_gotoTab('altdata')">${{x.icon}} ${{x.extra!==''&&x.extra!=null?x.extra:x.txt}}</span>`
  ).join('') + '</div>';
}}
function _dirCol(s) {{
  if (s.direction === 'SHORT') return 'var(--sell)';
  if (s.action === 'BUY' || s.action === 'HOLD LONG' || s.action === 'WATCH LONG') return 'var(--buy)';
  return 'var(--muted)';
}}
function _logo2(sym, px) {{
  const ini = (sym.replace(/[^A-Za-z]/g,'').slice(0,2) || sym.slice(0,2)).toUpperCase();
  return `<span class="mono2" style="width:${{px}}px;height:${{px}}px;font-size:${{Math.round(px*0.4)}}px;background:hsl(${{_symHue(sym)}},42%,42%);">${{ini}}`
    + `<img src="https://financialmodelingprep.com/image-stock/${{sym}}.png" alt="" loading="lazy" onerror="this.remove()">` + `</span>`;
}}
function _spark2(sym, color, w, h) {{
  const ch = (DATA.charts||{{}})[sym]; const c = (ch && ch.close ? ch.close : []).filter(x=>x!=null).slice(-40);
  if (c.length < 3) return `<svg viewBox="0 0 ${{w}} ${{h}}" style="width:${{w}}px;height:${{h}}px;"></svg>`;
  const mn=Math.min(...c), mx=Math.max(...c), rng=(mx-mn)||1;
  const pts = c.map((v,i)=>`${{(i/(c.length-1)*w).toFixed(1)}},${{(h-((v-mn)/rng)*(h-3)-1.5).toFixed(1)}}`).join(' ');
  return `<svg viewBox="0 0 ${{w}} ${{h}}" style="width:${{w}}px;height:${{h}}px;"><polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="1.5"/></svg>`;
}}
function _mChart(s) {{
  const el = document.getElementById('mSpark'); if (!el) return;
  const ch = (DATA.charts||{{}})[s.symbol];
  let c = (ch && ch.close ? ch.close : []).filter(x=>x!=null);
  const p = s.plan||{{}};
  if (c.length < 5) {{ el.innerHTML = ''; return; }}
  c = c.slice(-60);
  const W=600, H=150, padR=48, plotW=W-padR;
  const lv=[p.entry,p.target,p.stop].filter(x=>x!=null);
  const mn=Math.min(Math.min(...c),...lv), mx=Math.max(Math.max(...c),...lv), rng=(mx-mn)||1;
  const X=i=>(i/(c.length-1))*plotW, Y=v=>H-((v-mn)/rng)*(H-14)-7;
  const up=c[c.length-1]>=c[0], col=up?'var(--buy)':'var(--sell)', hex=up?'#5ed6a6':'#f0797f';
  const ln=c.map((v,i)=>X(i).toFixed(1)+','+Y(v).toFixed(1)).join(' ');
  const ar=ln+' '+plotW.toFixed(1)+','+H+' 0,'+H;
  const fmt=v=>(+v).toLocaleString(undefined,{{maximumFractionDigits:(v<10?2:0)}});
  const lvln=(v,color,lab)=>v==null?'':(
    '<line x1="0" y1="'+Y(v).toFixed(1)+'" x2="'+plotW+'" y2="'+Y(v).toFixed(1)+'" stroke="'+color+'" stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>'
    +'<text x="'+(W-4)+'" y="'+(Y(v)+3).toFixed(1)+'" text-anchor="end" fill="'+color+'" font-size="9.5" font-weight="600">'+lab+' '+fmt(v)+'</text>');
  el.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto;display:block;">'
    +'<defs><linearGradient id="mspg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="'+hex+'" stop-opacity=".16"/><stop offset="1" stop-color="'+hex+'" stop-opacity="0"/></linearGradient></defs>'
    +'<polygon points="'+ar+'" fill="url(#mspg)"/>'
    +lvln(p.target,'var(--buy)','T')+lvln(p.entry,'var(--muted)','E')+lvln(p.stop,'var(--sell)','S')
    +'<polyline points="'+ln+'" fill="none" stroke="'+col+'" stroke-width="2"/>'
    +'<circle cx="'+plotW.toFixed(1)+'" cy="'+Y(c[c.length-1]).toFixed(1)+'" r="3" fill="'+col+'"/>'
    +'</svg>';
}}
function _famOf(s) {{
  const co = (s.direction==='SHORT') ? (s.strategies&&s.strategies.short) : (s.strategies&&s.strategies.now);
  const F={{trend:'Trend',momentum:'Momentum',breakout:'Breakout',breakdown:'Breakout','mean-reversion':'Mean-rev'}};
  if(!co||!co.results) return '';
  const cnt={{}}; Object.values(co.results).forEach(r=>{{ if(s.direction==='SHORT'?r.short:r.long){{const f=F[r.kind]||r.kind;cnt[f]=(cnt[f]||0)+1;}}}});
  return Object.keys(cnt).sort((a,b)=>cnt[b]-cnt[a]).slice(0,2).join(' + ');
}}
function _levelsInline(s) {{
  const p=s.plan||{{}}; if(p.entry==null||p.stop==null||p.target==null) return '';
  return `${{p.entry}} · <span style="color:var(--sell);">${{p.stop}}</span> · <span style="color:var(--buy);">${{p.target}}</span>`;
}}
function _empty() {{ return '<div style="color:var(--muted);padding:14px;">Nothing matches this view right now.</div>'; }}
function _seenTs(s) {{ return Date.parse((s.first_seen || s.as_of || '') + 'T00:00:00') || 0; }}
function _applyFilter(list, f) {{
  if (f==='favs') return list.filter(s=>FAVS.has(s.symbol));
  if (f==='buys') return list.filter(s=>s.action==='BUY'||s.action==='HOLD LONG');
  if (f==='shorts') return list.filter(s=>s.action==='SHORT'||s.action==='HOLD SHORT');
  if (f==='watch') return list.filter(s=>s.action==='WATCH LONG'||s.action==='WATCH SHORT');
  if (f==='actionable') return list.filter(s=>['BUY','SHORT','HOLD LONG','HOLD SHORT','EXIT'].includes(s.action));
  return list;  // 'all'
}}
function _applySort(list, sort) {{
  if (sort==='conviction') list.sort((a,b)=>_conv(b)-_conv(a));
  else if (sort==='movers') list.sort((a,b)=>(b.rel_volume||0)-(a.rel_volume||0));
  else if (sort==='newest') list.sort((a,b)=>(_seenTs(b)-_seenTs(a))||(_conv(b)-_conv(a)));
  else list.sort((a,b)=>(_ACT_ORDER[a.action]-_ACT_ORDER[b.action])||(_conv(b)-_conv(a)));  // 'order' / 'sector' groups
  return list;
}}
function _reapplyLive() {{
  document.querySelectorAll('[data-px]').forEach(el => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[el.dataset.px] : null;
    if (p != null) el.textContent = _fmtPx(p);
  }});
}}
function _bindAll(container, list) {{
  const bySym = {{}}; list.forEach(s=>bySym[s.symbol]=s);
  container.querySelectorAll('[data-open]').forEach(el => {{
    el.addEventListener('click', () => {{ const s = bySym[el.getAttribute('data-open')]; if (s) openModal(s); }});
  }});
  return container;
}}
const _wrap = (cls, html) => {{ const d=document.createElement('div'); d.className=cls; d.innerHTML=html||_empty(); return d; }};

function _bbAct(s) {{
  if (s.direction==='SHORT') return '#ff5c4d';
  if (['BUY','HOLD LONG','WATCH LONG'].includes(s.action)) return '#33d17a';
  return '#9a9a78';
}}
function L_terminal(list, grouped) {{
  const r = DATA.regime || {{}};
  const buys = list.filter(s=>['BUY','HOLD LONG'].includes(s.action)).length;
  const shorts = list.filter(s=>['SHORT','HOLD SHORT'].includes(s.action)).length;
  const head = grouped ? '' : (`<div class="bbhead"><span class="bbtitle">SIGNAL DESK ▮</span>`
    + `<span class="bbst">REGIME <b>${{(r.label||'—').toUpperCase()}}</b></span>`
    + `<span class="bbst">BREADTH <b style="color:#fff;">${{r.breadth!=null?r.breadth+'%':'—'}}</b></span>`
    + `<span class="bbst">BUYS <b style="color:#33d17a;">${{buys}}</b></span>`
    + `<span class="bbst">SHORTS <b style="color:#ff5c4d;">${{shorts}}</b></span>`
    + `<span class="bbclock" id="bbclock"></span></div>`);
  const tiles = list.map(s=>{{
    const p = s.plan||{{}};
    const dc = (s.quote_price!=null && s.prev_close) ? (s.quote_price/s.prev_close-1)*100 : (s.context&&s.context.day_change_pct);
    const dcs = (dc!=null) ? `<span style="color:${{dc<0?'#ff5c4d':'#33d17a'}};">${{dc>=0?'+':''}}${{dc.toFixed(1)}}%</span>` : '';
    const lv = (p.entry!=null)
      ? `<span style="color:#33d17a;">T${{Math.round(p.target)}}</span> <span style="color:#cfcfcf;">E${{Math.round(p.entry)}}</span> <span style="color:#ff5c4d;">S${{Math.round(p.stop)}}</span>`
      : '—';
    return `<div class="bbtile" data-open="${{s.symbol}}"><div class="bbtop">${{_logo2(s.symbol,24)}}`
      + `<span class="bbsym">${{s.symbol}}</span><span class="bbact" style="color:${{_bbAct(s)}};">${{s.action}}</span></div>`
      + `<div class="bbpx"><span data-px="${{s.symbol}}">${{_pxOf(s).toLocaleString()}}</span> ${{dcs}}</div>`
      + `<div class="bbmeta">CONV <b style="color:${{_conv(s)>=0?_ragT(_conv(s)):'#8a8a6a'}};">${{_conv(s)>=0?_conv(s):'—'}}</b> · ${{_famOf(s)||'—'}}</div>`
      + `<div class="bblv">${{lv}}</div>`
      + (s.tv && s.tv.d ? `<div class="bbtv">TV <span style="color:#b5b5ba;">${{s.tv.d}}</span> · 1W ${{s.tv.w||'—'}}</div>` : '')
      + _altMini(s)
      + `</div>`;
  }}).join('');
  return _bindAll(_wrap('bbwrap', head + `<div class="bbgrid">${{tiles}}</div>`), list);
}}
function _laneTip(s) {{
  const p = s.plan||{{}}, conv=_conv(s);
  const reason = ((s.reasons||[]).find(r=>!/^<svg/.test(r)) || (s.reasons||[])[0] || '').replace(/<[^>]+>/g,'').trim();
  const lvl = (p.entry!=null) ? `Entry ${{p.entry}} · Stop ${{p.stop}} · Target ${{p.target}}${{p.rr!=null?' · R:R 1:'+p.rr:''}}` : '';
  return `<div style='font-weight:600;'>${{s.symbol}} · <span style='color:${{_dirCol(s)}};'>${{s.action}}</span></div>`
    + `<div style='color:var(--muted);font-size:11px;margin-bottom:5px;'>${{s.name||''}}</div>`
    + `<div>$${{_pxOf(s).toLocaleString()}} · Conviction ${{conv>=0?conv+'%':'—'}}</div>`
    + (_famOf(s)?`<div style='color:var(--muted);margin:3px 0;'>${{_famOf(s)}}</div>`:'')
    + (lvl?`<div>${{lvl}}</div>`:'')
    + (reason?`<div style='margin-top:5px;color:var(--muted);'>${{String(reason).slice(0,150)}}</div>`:'');
}}
function L_lanes(list) {{
  const lane = (title, col, items) => `<div class="lane"><div class="lanehd" style="color:${{col}};">${{title}} · ${{items.length}}</div>`
    + (items.map(s=>`<div class="lcard hint" data-open="${{s.symbol}}" data-tiphtml="${{_esc(_laneTip(s))}}" style="border-left-color:${{col}};">`
      + `<div class="lcard-t">${{_logo2(s.symbol,18)}}<span class="lsym">${{s.symbol}}</span><span class="lconv" style="color:${{_conv(s)>=0?_rag(_conv(s)):'var(--muted)'}};font-weight:700;">${{_conv(s)>=0?_conv(s)+'%':'—'}}</span></div>`
      + `${{_spark2(s.symbol, col, 150, 26)}}`
      + `<div class="lsub">${{_pxOf(s).toLocaleString ? '$'+_pxOf(s).toLocaleString() : ''}} · ${{_famOf(s)}}${{_tvBit(s)}}</div></div>`).join('') || '<div class="lsub" style="padding:6px;">—</div>') + '</div>';
  const buys = list.filter(s=>['BUY','HOLD LONG'].includes(s.action));
  const shorts = list.filter(s=>['SHORT','HOLD SHORT'].includes(s.action));
  const watch = list.filter(s=>['WATCH LONG','WATCH SHORT','EXIT','AVOID','FLAT'].includes(s.action));
  return _bindAll(_wrap('lanes', lane('↑ Long','var(--buy)',buys)+lane('↓ Short','var(--sell)',shorts)+lane('◷ Watch','var(--muted)',watch)), list);
}}
function L_gauges(list) {{
  const R=26, C=2*Math.PI*R;
  const cells = list.map(s=>{{
    const raw=_conv(s); const pc=Math.max(0,Math.min(100, raw>=0?raw:0)); const col=_rag(pc);
    const dash=(C*pc/100).toFixed(1);
    return `<div class="gauge" data-open="${{s.symbol}}"><svg viewBox="0 0 64 64" class="gsvg">`
      + `<circle cx="32" cy="32" r="${{R}}" fill="none" stroke="var(--inset)" stroke-width="6"/>`
      + `<circle cx="32" cy="32" r="${{R}}" fill="none" stroke="${{col}}" stroke-width="6" stroke-linecap="round" stroke-dasharray="${{dash}} ${{(C-dash).toFixed(1)}}" transform="rotate(-90 32 32)"/>`
      + `<text x="32" y="38" text-anchor="middle" class="gnum">${{raw>=0?pc:'—'}}</text></svg>`
      + `<div class="glab">${{_logo2(s.symbol,16)}}<span>${{s.symbol}}</span></div>`
      + `<div class="gact" style="color:${{_dirCol(s)}};">${{s.action}}</div>`
      + (s.tv && s.tv.d ? `<div class="gtv">TV ${{s.tv.d}}</div>` : '') + `</div>`;
  }}).join('');
  return _bindAll(_wrap('gauges', cells), list);
}}
function L_feed(list) {{
  const rows = list.map(s=>{{
    const co=(s.direction==='SHORT')?(s.strategies&&s.strategies.short):(s.strategies&&s.strategies.now);
    const trig=(co&&co.fresh&&co.fresh[0])||_famOf(s)||'multiple methods';
    const verb = s.action.indexOf('WATCH')>=0?'is building a':(s.action.indexOf('HOLD')>=0?'is holding a':'triggered a');
    return `<div class="feeditem" data-open="${{s.symbol}}">${{_logo2(s.symbol,26)}}`
      + `<div class="feedtxt"><div><b>${{s.symbol}}</b> ${{verb}} <span style="color:${{_dirCol(s)}};font-weight:600;">${{s.action}}</span> — ${{trig}}</div>`
      + `<div class="feedsub">${{_conv(s)>=0?_conv(s)+'% conviction · ':''}}$${{_pxOf(s).toLocaleString()}}${{_tvBit(s)}} · as of ${{s.as_of||''}}</div></div>`
      + `<span class="feedspark">${{_spark2(s.symbol,_dirCol(s),96,30)}}</span></div>`;
  }}).join('');
  return _bindAll(_wrap('feedwrap', rows), list);
}}
function L_ticker(list, grouped) {{
  const tape = list.map(s=>`<span class="tkitem"><b>${{s.symbol}}</b> <span data-px="${{s.symbol}}" style="color:${{_dirCol(s)}};">$${{_pxOf(s).toLocaleString()}}</span></span>`).join('');
  const rows = list.map(s=>`<div class="tkrow" data-open="${{s.symbol}}">${{_logo2(s.symbol,22)}}`
    + `<span class="tksym">${{s.symbol}}</span><span style="color:${{_dirCol(s)}};font-weight:600;width:80px;">${{s.action}}</span>`
    + `<span class="tkpx" data-px="${{s.symbol}}">$${{_pxOf(s).toLocaleString()}}</span>`
    + `<span class="tkspark">${{_spark2(s.symbol,_dirCol(s),72,22)}}</span>`
    + `<span class="tkfam">${{_famOf(s)}}${{s.tv&&s.tv.d?' · TV '+s.tv.d:''}}</span><span class="tklv">${{_levelsInline(s)}}</span></div>`).join('');
  const tapeHtml = grouped ? '' : `<div class="tktape"><div class="tktape-in">${{tape}}${{tape}}</div></div>`;
  return _bindAll(_wrap('', `${{tapeHtml}}<div class="tkbody">${{rows}}</div>`), list);
}}
function L_rows(list) {{
  const rows = list.map((s,i)=>{{
    const conv=_conv(s);
    const dc=(s.quote_price!=null && s.prev_close)?(s.quote_price/s.prev_close-1)*100:(s.context&&s.context.day_change_pct);
    const dcs=(dc!=null)?`<div class="chg" style="color:${{dc<0?'var(--sell)':'var(--buy)'}};">${{dc>=0?'+':''}}${{dc.toFixed(1)}}%</div>`:'';
    const dir=s.action.indexOf('SHORT')>=0?'short':(s.action.indexOf('WATCH')>=0?'watch':'long');
    return `<div class="rowsig" data-open="${{s.symbol}}">`
      + `<div class="rk">${{String(i+1).padStart(2,'0')}}</div>`
      + `${{_logo2(s.symbol,28)}}`
      + `<div class="nm"><div class="sym">${{s.symbol}} <span class="rtag">${{dir}}</span></div>`
      + `<div class="co">${{s.name||_famOf(s)||''}}</div></div>`
      + `<div class="rconv">${{conv>=0?'<b>'+conv+'</b> <span>/100</span>':'—'}}</div>`
      + `<span class="rspark">${{_spark2(s.symbol,_dirCol(s),160,26)}}</span>`
      + `<div class="rpx">$${{_pxOf(s).toLocaleString()}}${{dcs}}</div>`
      + `<div class="rarr">&rarr;</div></div>`;
  }}).join('');
  return _bindAll(_wrap('rowswrap', rows), list);
}}
const LAYOUT_RENDER = {{rows:L_rows, terminal:L_terminal, lanes:L_lanes, gauges:L_gauges, feed:L_feed, ticker:L_ticker}};
// ===================================================================================

function _renderConcWarn() {{
  const el = document.getElementById('concWarn'); if (!el) return;
  const c = DATA.concentration;
  if (!c) {{ el.innerHTML = ''; return; }}
  el.innerHTML = `<div class="conc-warn" data-tip="${{c.symbols.join(', ')}}">`
    + `${{_ico('warn',13)}} Concentration: ${{c.n}} of ${{c.total}} fresh ${{c.word}} are in <b>${{c.sector}}</b> (${{c.pct}}%). `
    + `These can be the same macro bet in disguise — sizing them as separate trades understates your real risk.</div>`;
}}
function _isLive(s) {{
  const a = s.action || '';
  if (a === 'FLAT' || a === 'EXIT' || a === 'AVOID') return false;   // not a live setup
  if ((s.days_old || 0) > 14) return false;                          // time has run out
  const p = (s.quote_price != null) ? s.quote_price : s.price;
  const pl = s.plan || {{}};
  if (p != null && pl.target != null && pl.stop != null) {{
    const long = s.direction !== 'SHORT';
    if (long && (p >= pl.target || p <= pl.stop)) return false;      // target hit / stopped → done
    if (!long && (p <= pl.target || p >= pl.stop)) return false;
  }}
  return true;
}}
function renderCards() {{
  _renderConcWarn();
  cards.innerHTML = '';
  const useLayout = _layout && _layout !== 'cards' && LAYOUT_RENDER[_layout];
  let base = _applyFilter(DATA.signals.filter(_isLive), _curFilter);
  if (_searchTerm) {{
    const q = _searchTerm.toLowerCase();
    base = base.filter(s => (s.symbol||'').toLowerCase().includes(q)
      || (s.name||'').toLowerCase().includes(q));
  }}
  const _cc = document.getElementById('cardsCount');
  if (_cc) _cc.textContent = base.length + (base.length === 1 ? ' signal' : ' signals');
  const emptyMsg = (_curFilter === 'favs')
    ? `No favorites yet — tap the ${{_ico('star',13)}} on any card to save it here.`
    : 'Nothing matches this view right now.';
  const flatGrid = (list) => {{
    const grid = document.createElement('div'); grid.className = 'grid';
    if (!list.length) grid.innerHTML = '<div style="color:var(--muted);">' + emptyMsg + '</div>';
    list.forEach(s => grid.appendChild(makeCard(s)));
    return grid;
  }};
  if (_curSort === 'sector') {{
    const by = {{}}, order = [];
    base.forEach(s => {{ const sec = s.sector || 'Other / Movers';
      if (!by[sec]) {{ by[sec] = []; order.push(sec); }} by[sec].push(s); }});
    if (!order.length) {{
      cards.appendChild(useLayout ? LAYOUT_RENDER[_layout]([]) : flatGrid([]));
    }}
    order.forEach(sec => {{
      const grp = _applySort(by[sec].slice(), 'order');
      const h = document.createElement('div'); h.className = 'secthead';
      h.textContent = sec + ' · ' + grp.length; cards.appendChild(h);
      cards.appendChild(useLayout ? LAYOUT_RENDER[_layout](grp, true) : flatGrid(grp));
    }});
  }} else {{
    const list = _applySort(base, _curSort);
    cards.appendChild(useLayout ? LAYOUT_RENDER[_layout](list) : flatGrid(list));
  }}
  // re-apply any live prices to the freshly rendered cards
  document.querySelectorAll('[data-px]').forEach(el => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[el.dataset.px] : null;
    if (p != null) el.textContent = _fmtPx(p);
  }});
  if (window._applyControls) try {{ window._applyControls(); }} catch (e) {{}}
}}
(function setupViews() {{
  // Sort = how the SAME set of cards is ordered (sector grouping, conviction, etc.)
  const sortBar = document.getElementById('sortBtns');
  const sorts = [['sector','By sector'],['order','Actionable first'],['newest','Newest'],
                 ['conviction','Highest conviction'],['movers','Biggest movers']];
  sorts.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.sort = v;
    if (v === _curSort) b.className = 'on';
    b.onclick = () => {{
      _curSort = v;
      try {{ localStorage.setItem('sort', v); }} catch(e) {{}}
      sortBar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.sort === v));
      renderCards();
    }};
    sortBar.appendChild(b);
  }});
  // Show = which cards to include (narrows the set); composes with Sort
  const filterBar = document.getElementById('filterBtns');
  const filters = [['all','All'],['buys','Longs'],['shorts','Shorts'],['watch','Watch'],
                   ['actionable','Actionable'],['favs',_ico('star-fill',12)+' Favorites']];
  filters.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.innerHTML = lab; b.dataset.filter = v;
    if (v === _curFilter) b.className = 'on';
    b.onclick = () => {{
      _curFilter = v;
      try {{ localStorage.setItem('filter', v); }} catch(e) {{}}
      filterBar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.filter === v));
      renderCards();
    }};
    filterBar.appendChild(b);
  }});
  // --- layout switcher (visual form: cards / terminal / lanes / …) ---
  const lbar = document.getElementById('layoutBtns');
  const layouts = [['rows','Rows'],['cards','Cards'],['lanes','Lanes'],['terminal','Terminal'],
                   ['gauges','Gauges'],['feed','Feed'],['ticker','Ticker']];
  layouts.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.layout = v;
    if (v === _layout) b.className = 'on';
    b.onclick = () => {{
      _layout = v;
      try {{ localStorage.setItem('layout2', v); }} catch(e) {{}}
      lbar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.layout === v));
      renderCards();
    }};
    lbar.appendChild(b);
  }});
  renderCards();
}})();

// --- Edge explorer: left-rail tab switching ---
(function setupEdgeTabs() {{
  document.querySelectorAll('.an-tab').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      const v = btn.dataset.anview;
      document.querySelectorAll('.an-tab').forEach(function(b) {{ b.classList.toggle('on', b.dataset.anview === v); }});
      document.querySelectorAll('.an-view').forEach(function(x) {{ x.classList.toggle('on', x.dataset.anview === v); }});
    }});
  }});
}})();

// ---- Control panel: settings sliders + accept/reject + apply-to-engine export ----
(function setupControl() {{
  const $ = id => document.getElementById(id);
  const floor = $('cFloor'); if (!floor) return;
  function paint() {{
    $('cFloor').value = _CONTROLS.floor; $('cFloorV').textContent = _CONTROLS.floor + '%';
    $('cCap').value = _CONTROLS.cap; $('cCapV').textContent = _CONTROLS.cap;
    $('cExt').checked = _CONTROLS.ext; $('cVol').checked = _CONTROLS.vol; $('cShorts').checked = _CONTROLS.shorts;
  }}
  function chips(el, arr, kind) {{
    el.innerHTML = arr.length ? arr.map(s => '<span class="ctrl-chip ' + kind + '">' + s + '<button data-sym="' + s + '">&times;</button></span>').join('') : '<span class="ctrl-none">none yet</span>';
    el.querySelectorAll('button').forEach(b => b.addEventListener('click', () => _decide(b.dataset.sym, 'clear')));
  }}
  function apply() {{
    const cards = [...document.querySelectorAll('#cards [data-sym]')];
    const floorF = _CONTROLS.floor / 100;
    let buys = cards.filter(c => c.dataset.action === 'BUY' && c.dataset.dir !== 'SHORT');
    buys.sort((a, b) => (parseFloat(b.dataset.pwin) || 0) - (parseFloat(a.dataset.pwin) || 0));
    cards.forEach(c => c.classList.remove('ctrl-dim'));
    let kept = 0;
    buys.forEach(c => {{
      const pw = parseFloat(c.dataset.pwin);
      const rej = _DEC.rejected.includes(c.dataset.sym), acc = _DEC.accepted.includes(c.dataset.sym);
      const ok = !rej && (acc || ((isNaN(pw) || pw >= floorF) && kept < _CONTROLS.cap));
      if (ok) kept++; else c.classList.add('ctrl-dim');
    }});
    cards.forEach(c => {{ if (_DEC.rejected.includes(c.dataset.sym)) c.classList.add('ctrl-dim'); }});
    const pv = $('cPreview'); if (pv) pv.textContent = 'Preview: ' + kept + ' actionable BUY' + (kept === 1 ? '' : 's') + ' under these settings (of ' + buys.length + ' flagged today).';
  }}
  window._applyControls = apply;
  window._renderControl = function() {{ paint(); chips($('cAccepted'), _DEC.accepted, 'acc'); chips($('cRejected'), _DEC.rejected, 'rej'); apply(); }};
  floor.addEventListener('input', () => {{ _CONTROLS.floor = +floor.value; $('cFloorV').textContent = floor.value + '%'; _saveCtrl(); apply(); }});
  $('cCap').addEventListener('input', () => {{ _CONTROLS.cap = +$('cCap').value; $('cCapV').textContent = $('cCap').value; _saveCtrl(); apply(); }});
  [['cExt', 'ext'], ['cVol', 'vol'], ['cShorts', 'shorts']].forEach(([id, k]) => $(id).addEventListener('change', e => {{ _CONTROLS[k] = e.target.checked; _saveCtrl(); apply(); }}));
  $('cClear').addEventListener('click', () => {{ _DEC = {{accepted: [], rejected: []}}; _saveDec(); document.querySelectorAll('.accepted, .rejected').forEach(el => el.classList.remove('accepted', 'rejected')); window._renderControl(); }});
  $('cReset').addEventListener('click', () => {{ _CONTROLS = Object.assign({{}}, _CTRL_DEFAULTS); _saveCtrl(); window._renderControl(); }});
  function payload() {{
    return {{ settings: {{ meta_pwin_floor: +(_CONTROLS.floor / 100).toFixed(2), meta_buy_cap: _CONTROLS.cap,
      extension_gate_enabled: _CONTROLS.ext, volatility_gate_enabled: _CONTROLS.vol, allow_shorts: _CONTROLS.shorts }},
      accepted: _DEC.accepted, rejected: _DEC.rejected, generated: new Date().toISOString() }};
  }}
  $('cApply').addEventListener('click', () => {{
    const blob = new Blob([JSON.stringify(payload(), null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'dashboard_controls.json'; a.click();
  }});
  $('cCopy').addEventListener('click', () => {{ try {{ navigator.clipboard.writeText(JSON.stringify(payload(), null, 2)); $('cCopy').textContent = 'Copied'; setTimeout(() => $('cCopy').textContent = 'Copy', 1200); }} catch (e) {{}} }});
  window._renderControl();
}})();

// --- Live-signals horizontal scroll arrows + expand-all toggle ---
(function setupCardScroll() {{
  const wrap = document.getElementById('cards');
  const prev = document.getElementById('cardsPrev');
  const next = document.getElementById('cardsNext');
  const exp = document.getElementById('cardsExpand');
  if (!wrap) return;
  const grid = () => wrap.querySelector('.grid');
  const scroll = d => {{ const el = grid(); if (el) el.scrollBy({{left: d * Math.min(560, el.clientWidth * 0.85), behavior: 'smooth'}}); }};
  if (prev) prev.onclick = () => scroll(-1);
  if (next) next.onclick = () => scroll(1);
  if (exp) exp.onclick = () => {{
    const on = wrap.classList.toggle('expanded');
    exp.textContent = on ? 'Collapse' : 'Expand all';
    if (prev) prev.style.display = on ? 'none' : '';
    if (next) next.style.display = on ? 'none' : '';
  }};
}})();

// --- Grok · X pulse: live social/news read on top names (populated on live builds) ---
const _GK_BUZZ = {{}};   // symbol -> buzz item, so the modal can find it on click
function renderGrokPulse() {{
  const el = document.getElementById('grokPulse'); if (!el) return;
  // Primary: the live BUZZ feed (most talked-about names right now). Fallback: sentiment on our signals.
  let items = (DATA.xai_buzz || []).slice();
  if (!items.length) {{
    items = (DATA.signals || []).filter(s => s.xai_sentiment)
      .map(s => Object.assign({{symbol: s.symbol, is_signal: true, signal_action: s.action, p_win: s.p_win}}, s.xai_sentiment));
  }}
  if (!items.length) {{
    const st = (DATA.xai_status || 'empty');
    const msg = (st === 'off') ? 'Grok live sentiment is off — set the <code>XAI_LIVE_SENTIMENT</code> repo variable to true.'
      : (st === 'no_key') ? 'No <code>XAI_API_KEY</code> in this build — add it as a repo secret.'
      : (st === 'not_live') ? 'Synthetic/offline build — Grok runs only on live builds with market data.'
      : 'No buzz landed this build. It populates on a <b>live build during market hours</b> — the most talked-about names on X right now.';
    el.innerHTML = '<div class="gp-empty">' + _ico('ai',15) + ' ' + msg + '</div>';
    return;
  }}
  const cls = {{bullish:'gp-up', bearish:'gp-dn', mixed:'gp-mut', quiet:'gp-mut'}};
  const momMap = {{rising:'&#8599; rising', steady:'&#8594; steady', fading:'&#8600; fading'}};
  for (const k in _GK_BUZZ) delete _GK_BUZZ[k];
  el.innerHTML = '<div class="gp-list">' + items.slice(0,12).map((x, i) => {{
    _GK_BUZZ[x.symbol] = x;
    const st = (x.stance || 'quiet');
    const conf = (x.confidence != null) ? x.confidence : '';
    const vol = x.social_volume ? (x.social_volume + ' volume') : '';
    const cat = x.catalyst ? ('<span class="gp-cat">' + _ico('bolt',11) + ' ' + _esc(x.catalyst) + '</span>') : '';
    const note = x.note ? _esc(x.note) : '';
    const momc = x.momentum === 'rising' ? 'gp-up' : (x.momentum === 'fading' ? 'gp-dn' : 'gp-mut');
    const mom = x.momentum ? ('<span class="gp-mom-mini ' + momc + '">' + momMap[x.momentum] + '</span>') : '';
    const badge = x.is_signal ? ('<span class="gp-sigbadge" title="Also one of our signals">' + _esc(x.signal_action || 'signal') + '</span>') : '';
    return '<div class="gp-row" data-sym="' + x.symbol + '">'
      + '<span class="gp-sym">' + x.symbol + badge + '</span>'
      + '<span class="sx-pill ' + (cls[st] || 'gp-mut') + '">' + st + '</span>'
      + '<span class="gp-note">' + note + ' ' + cat + '</span>'
      + '<span class="gp-meta">' + mom + ' ' + vol + (conf !== '' ? (' · ' + conf) : '') + '</span>'
      + '</div>';
  }}).join('') + '</div>';
  // Independent crowd source: the most-discussed retail names on StockTwits right now (ported feed).
  const trend = (DATA.retail_trending || []);
  if (trend.length) {{
    el.innerHTML += '<div class="gp-trend"><span class="gp-trend-l">' + _ico('chat',12)
      + ' Also trending on StockTwits</span>'
      + trend.slice(0,12).map(t => {{
          const isSig = (DATA.signals || []).some(s => s.symbol === t.symbol);
          return '<span class="gp-trend-chip' + (isSig ? ' sig' : '') + '" data-sym="' + t.symbol + '">' + t.symbol + '</span>';
        }}).join('') + '</div>';
  }}
  el.querySelectorAll('.gp-row').forEach(r => r.addEventListener('click', () => {{
    const it = _GK_BUZZ[r.dataset.sym]; if (it) openGrokModal(it);
  }}));
  el.querySelectorAll('.gp-trend-chip').forEach(c => c.addEventListener('click', () => {{
    const sig = (DATA.signals || []).find(s => s.symbol === c.dataset.sym); if (sig) openModal(sig);
  }}));
}}
try {{ renderGrokPulse(); }} catch (e) {{}}

// --- Grok deep-dive modal: what X is really saying about a name RIGHT NOW ---
const _gkOverlay = document.getElementById('grokOverlay');
function openGrokModal(item) {{
  const body = document.getElementById('grokBody'); if (!body || !_gkOverlay) return;
  // `item` is either a buzz entry (fields at top level) or a signal (fields under .xai_sentiment).
  const x = item.xai_sentiment || item || {{}};
  const sym = item.symbol;
  const sig = (DATA.signals || []).find(s => s.symbol === sym);
  const dispName = (sig && sig.name) || item.name || '';
  const st = x.stance || 'quiet';
  const scls = {{bullish:'gp-up', bearish:'gp-dn', mixed:'gp-mut', quiet:'gp-mut'}}[st] || 'gp-mut';
  const mom = {{rising:'&#8599; rising attention', steady:'&#8594; steady attention', fading:'&#8600; fading attention'}}[x.momentum] || '';
  const momc = x.momentum === 'rising' ? 'gp-up' : (x.momentum === 'fading' ? 'gp-dn' : 'gp-mut');
  const meta = [x.social_volume ? (x.social_volume + ' volume') : '', (x.confidence != null ? x.confidence + '% confidence' : '')].filter(Boolean).join(' &middot; ');
  const chips = (x.themes || []).map(t => `<span class="gk-chip">${{_esc(t)}}</span>`).join('');
  const bull = (x.bull || []).map(b => `<li>${{_esc(b)}}</li>`).join('');
  const bear = (x.bear || []).map(b => `<li>${{_esc(b)}}</li>`).join('');
  const cat = x.catalyst ? `<div class="gk-cat">${{_ico('bolt',13)}} <b>Catalyst:</b> ${{_esc(x.catalyst)}}${{x.catalyst_time ? ` <span class="gk-when">&middot; ${{_esc(x.catalyst_time)}}</span>` : ''}}${{x.fresh_catalyst ? ' <span class="gk-fresh">fresh</span>' : ''}}</div>` : '';
  const asof = (DATA.generated_at || DATA.as_of || '');
  body.innerHTML =
      `<div class="gk-head"><span class="gk-tick">${{sym}}</span>`
    + `<span class="sx-pill ${{scls}}">${{st}}</span>`
    + (mom ? `<span class="gk-mom ${{momc}}">${{mom}}</span>` : '')
    + (item.is_signal ? `<span class="gp-sigbadge">${{_esc(item.signal_action || (sig && sig.action) || 'signal')}}</span>` : '')
    + `</div>`
    + `<div class="gk-sub">${{_esc(dispName)}}${{meta ? ` &middot; ${{meta}}` : ''}}</div>`
    + cat
    + (x.note ? `<div class="gk-lead">${{_esc(x.note)}}</div>` : '')
    + (x.hype_risk === 'high' ? `<div class="gk-hype">${{_ico('octagon',13)}} <b>High hype risk</b> — buzz looks like a coordinated pump/promotion, not genuine news. Not seeded as a trade.</div>`
        : (x.hype_risk === 'medium' ? `<div class="gk-hype med">${{_ico('warn',13)}} Medium hype risk — some promotional noise; weigh the source.</div>` : ''))
    + (chips ? `<div class="gk-sec"><div class="gk-h">${{_ico('chat',13)}} What people are talking about now</div><div class="gk-chips">${{chips}}</div></div>` : '')
    + ((bull || bear) ? `<div class="gk-args"><div class="gk-col"><div class="gk-h up">Bull chatter</div><ul class="gk-ul">${{bull || '<li class="gk-none">nothing notable</li>'}}</ul></div>`
        + `<div class="gk-col"><div class="gk-h dn">Bear chatter</div><ul class="gk-ul">${{bear || '<li class="gk-none">nothing notable</li>'}}</ul></div></div>` : '')
    + (x.watch ? `<div class="gk-watch"><div class="gk-h">${{_ico('target',13)}} Watch for entry</div><div>${{_esc(x.watch)}}</div></div>` : '')
    + `<div class="gk-foot"><span class="gk-src">${{_ico('ai',12)}} Grok &middot; live X + web read${{asof ? ' &middot; as of ' + _esc(asof) : ''}}</span>`
    + (sig ? `<button class="gk-open">Open full signal &rarr;</button>` : '')
    + `</div>`;
  const ob = body.querySelector('.gk-open');
  if (ob && sig) ob.addEventListener('click', () => {{ closeGrokModal(); openModal(sig); }});
  _gkOverlay.classList.add('open');
}}
function closeGrokModal() {{ if (_gkOverlay) _gkOverlay.classList.remove('open'); }}
if (_gkOverlay) {{
  const gc = document.getElementById('grokClose');
  if (gc) gc.addEventListener('click', closeGrokModal);
  _gkOverlay.addEventListener('click', e => {{ if (e.target === _gkOverlay) closeGrokModal(); }});
  document.addEventListener('keydown', e => {{ if (e.key === 'Escape' && _gkOverlay.classList.contains('open')) closeGrokModal(); }});
}}

// --- Intraday tab: render the intraday-bar signals (reuses the same card UI) ---
function renderIntraday() {{
  const el = document.getElementById('intradayCards'); if (!el) return;
  const list = DATA.intraday || [];
  const tf = (DATA.params && DATA.params.intraday_timeframe) || '5Min';
  const it = DATA.intraday_track || {{}};
  const rec = it.resolved
    ? `Shadow record: <b>${{it.win_rate ?? '—'}}%</b> win over ${{it.resolved}} resolved · expectancy ${{(it.expectancy >= 0 ? '+' : '')}}${{it.expectancy ?? '—'}}% · ${{it.open || 0}} open`
    : `Shadow record: building — grades these ${{tf}} calls against real prices over the next few days (no orders placed)`;
  el.innerHTML = `<div class="strat-badge"><span class="k">Layer</span><span class="v">Intraday · ${{tf}} bars — faster &amp; noisier than the daily signals; confirm before acting</span></div>`
    + `<div class="note" style="margin:0 0 12px;">${{_ico('chart',13)}} ${{rec}}</div>`;
  const grid = document.createElement('div'); grid.className = 'grid';
  if (!list.length) {{
    grid.innerHTML = '<div style="color:var(--muted);font-size:13px;">No intraday signals this build (or intraday data was unavailable — it falls back silently, so the daily view is never affected).</div>';
  }} else {{
    list.forEach(s => grid.appendChild(makeCard(s)));
  }}
  el.appendChild(grid);
  document.querySelectorAll('#page-intraday [data-px]').forEach(elp => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[elp.dataset.px] : null;
    if (p != null) elp.textContent = _fmtPx(p);
  }});
}}
renderIntraday();
renderOrb();

// --- momentum leaderboard rows open the same rich detail modal as the cards ---
(function bindMomentumRows() {{
  const det = DATA.mom_detail || {{}};
  document.querySelectorAll('tr.momrow').forEach(tr => {{
    tr.style.cursor = det[tr.dataset.sym] ? 'pointer' : 'default';
    tr.addEventListener('click', () => {{
      const s = det[tr.dataset.sym];
      if (s) openModal(s);
    }});
  }});
}})();

// ---- live prices (via Cloudflare Worker proxy) ----
const LIVE_SYMS = [...new Set(DATA.signals.map(s => s.symbol).concat('SPY'))];
function _fmtPx(v) {{ return '$' + Number(v).toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}}); }}
async function refreshLive() {{
  if (!LIVE_URL) return;
  const st = document.getElementById('liveStatus');
  try {{
    const r = await fetch(LIVE_URL + '?symbols=' + encodeURIComponent(LIVE_SYMS.join(',')));
    if (!r.ok) throw new Error('bad');
    const d = await r.json();
    LIVE = d.prices || {{}};
    document.querySelectorAll('[data-px]').forEach(el => {{
      const p = LIVE[el.dataset.px];
      if (p != null) el.textContent = _fmtPx(p);
    }});
    // live, correctly-labelled "today" change: live quote vs the last daily close
    document.querySelectorAll('[data-chg]').forEach(el => {{
      const p = LIVE[el.dataset.chg], base = parseFloat(el.dataset.base);
      if (p != null && base) {{
        const pct = (p / base - 1) * 100;
        el.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '% today';
        el.style.color = pct >= 0 ? 'var(--buy)' : 'var(--sell)';
      }}
    }});
    if (featTC) featTC.onLive(LIVE);
    if (modalTC) modalTC.onLive(LIVE);
    if (typeof renderTape === 'function') renderTape();
    const tm = new Date(d.at || Date.now()).toLocaleTimeString('en-GB', {{timeZone:'GMT', hour12:false}}) + ' GMT';
    st.innerHTML = '&middot; <span style="color:#2ea043;">● Live</span> <span style="color:#8b97a6;">'+tm+'</span>';
  }} catch (e) {{
    st.innerHTML = '&middot; <span style="color:#8b97a6;">live prices unavailable</span>';
  }}
}}
if (LIVE_URL) {{ refreshLive(); setInterval(refreshLive, 15000); }}
// ===== scrolling ticker tape: SYM  price  ▲/▼ +x.x%  (mono, direction-coloured) =====
function _tapeItems() {{
  const seen = new Set(), out = [];
  const push = (sym, px, base, dir) => {{
    if (!sym || seen.has(sym)) return; seen.add(sym);
    out.push({{ sym, px: (px != null ? Number(px) : null), base: (base != null ? Number(base) : null), dir }});
  }};
  (DATA.signals || []).forEach(s => {{
    const px = (s.quote_price != null) ? s.quote_price : s.price;
    const base = (s.prev_close != null) ? s.prev_close : s.price;
    push(s.symbol, px, base, s.direction === 'SHORT' ? 'S' : 'L');
  }});
  (DATA.watchlist || []).forEach(w => {{ if (typeof w === 'string') push(w, null, null, ''); else if (w && w.symbol) push(w.symbol, w.price, w.prev_close, ''); }});
  return out.slice(0, 28);
}}
function _tapeItemHtml(it) {{
  let pxv = it.px, chg = null;
  if (LIVE[it.sym] != null) pxv = LIVE[it.sym];
  if (pxv != null && it.base) chg = (pxv / it.base - 1) * 100;
  const up = (chg != null) ? chg >= 0 : (it.dir === 'L');
  const pxTxt = (pxv != null) ? ('$' + Number(pxv).toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}})) : '';
  const chgTxt = (chg != null) ? `<span class="chg ${{up?'up':'dn'}}">${{chg>=0?'+':''}}${{chg.toFixed(2)}}%</span>` : '';
  const dirI = (chg != null || it.dir) ? `<span class="dir ${{up?'up':'dn'}}">${{_ico(up?'trend-up':'trend-dn',12)}}</span>` : '';
  const logo = `<img class="tlogo" src="https://assets.parqet.com/logos/symbol/${{it.sym}}?format=png" alt="" loading="lazy" onerror="this.outerHTML='<span class=\\'tlogo tlogo-mono\\'>'+String(this.getAttribute('data-i')||'?')+'</span>'" data-i="${{it.sym.slice(0,2)}}">`;
  return `<span class="tkt-it" data-tsym="${{it.sym}}">${{logo}}<span class="sym">${{it.sym}}</span>`
       + (pxTxt?`<span class="px">${{pxTxt}}</span>`:'') + chgTxt + dirI + `</span><span class="tkt-sep"></span>`;
}}
function renderTape() {{
  const el = document.getElementById('tapeTrack'); if (!el) return;
  const items = _tapeItems(); if (!items.length) {{ el.innerHTML = ''; return; }}
  const one = items.map(_tapeItemHtml).join('');
  el.innerHTML = one + one;  // duplicated for a seamless -50% loop
}}
renderTape();
// "last built X min ago" ticker for the build time (shown in GMT in the subhead).
(function builtAgo() {{
  const el = document.getElementById('builtAgo');
  const ts = (typeof DATA !== 'undefined' && DATA.generated_ts) ? DATA.generated_ts * 1000 : null;
  if (!el || !ts) return;
  const banner = document.getElementById('staleBanner');
  function mktOpen() {{
    try {{
      const et = new Date(new Date().toLocaleString('en-US', {{timeZone:'America/New_York'}}));
      const d = et.getDay(), m = et.getHours()*60 + et.getMinutes();
      return d >= 1 && d <= 5 && m >= 540 && m < 990;   // ~9:00–16:30 ET (incl. pre/post buffer)
    }} catch (e) {{ return false; }}
  }}
  function fmtAge(m) {{
    if (m < 60) return m + ' min';
    const h = Math.floor(m/60), mm = m%60;
    return h + 'h' + (mm ? ' ' + mm + 'm' : '');
  }}
  function upd() {{
    const m = Math.max(0, Math.round((Date.now() - ts) / 60000));
    el.textContent = '· ' + (m < 1 ? 'just now' : (m === 1 ? '1 min ago' : m + ' min ago'));
    if (!banner) return;
    // Builds run ~every 30 min on a trading day. >90 min during market hours = scheduler likely
    // skipped runs → the page is STALE and shouldn't be trusted as live. >12h any time = very stale.
    const open = mktOpen();
    if (open && m > 90) {{
      banner.className = 'stale-banner red';
      banner.style.display = '';
      banner.innerHTML = `${{_ico('warn',14)}} <b>Data is ${{fmtAge(m)}} old</b> — the rebuild scheduler looks stuck (it should refresh every ~30 min while the market is open). Prices/signals below may be out of date. Trigger a manual run from the repo's Actions tab to refresh.`;
    }} else if (m > 720) {{
      banner.className = 'stale-banner amber';
      banner.style.display = '';
      banner.innerHTML = `${{_ico('warn',14)}} Data is ${{fmtAge(m)}} old (last build shown above). The market may be closed; this will refresh on the next scheduled run.`;
    }} else {{
      banner.style.display = 'none';
    }}
  }}
  upd(); setInterval(upd, 30000);
}})();
// Live clock: current time in GMT and GMT+4, plus the US market window + open/closed status.
(function marketClock() {{
  const el = document.getElementById('marketClock');
  const bar = document.getElementById('barClock');
  if (!el && !bar) return;
  const t = (tz) => new Date().toLocaleTimeString('en-GB', {{timeZone:tz, hour12:false, hour:'2-digit', minute:'2-digit'}});
  const ts = (tz) => new Date().toLocaleTimeString('en-GB', {{timeZone:tz, hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'}});
  function isOpen() {{
    try {{
      const et = new Date(new Date().toLocaleString('en-US', {{timeZone:'America/New_York'}}));
      const d = et.getDay(), m = et.getHours()*60 + et.getMinutes();
      return d >= 1 && d <= 5 && m >= 570 && m < 960;   // Mon–Fri, 09:30–16:00 ET
    }} catch (e) {{ return false; }}
  }}
  function upd() {{
    const open = isOpen();
    if (el) el.innerHTML = _ico('clock',13) + ' ' + t('GMT') + ' GMT &middot; ' + t('Asia/Dubai') + ' GMT+4'
      + ' &middot; NYSE 09:30–16:00 ET ('
      + (open ? '<span style="color:var(--buy);font-weight:600;">open</span>'
              : '<span style="color:var(--muted);font-weight:600;">closed</span>') + ')';
    if (bar) bar.innerHTML = _ico('clock',12) + ' ' + ts('America/New_York') + ' ET '
      + (open ? '<span style="color:var(--buy);font-weight:700;">•</span>'
              : '<span style="color:var(--muted);font-weight:700;">•</span>');
  }}
  upd(); setInterval(upd, 1000);
}})();
// live ET clock for the terminal header (only present when Terminal layout is active)
setInterval(() => {{
  const c = document.getElementById('bbclock');
  if (c) try {{ c.textContent = new Date().toLocaleTimeString('en-US', {{hour12:false, timeZone:'America/New_York'}}) + ' ET'; }} catch(e) {{}}
}}, 1000);

// Notice when a newer build has been published (the Action rebuilds every ~30 min in market
// hours) and offer a one-click refresh — never reloads from under the user.
(function watchForNewBuild() {{
  const cur = (typeof DATA !== 'undefined' && DATA.generated_at) || '';
  async function check() {{
    if (document.hidden) return;
    try {{
      const r = await fetch('signals.json?cb=' + Date.now(), {{cache: 'no-store'}});
      if (!r.ok) return;
      const d = await r.json();
      if (d.generated_at && d.generated_at !== cur && !document.getElementById('newbuild')) {{
        const b = document.createElement('button');
        b.id = 'newbuild';
        b.textContent = '↻ New data available — refresh';
        b.onclick = () => location.reload();
        document.body.appendChild(b);
      }}
    }} catch (e) {{}}
  }}
  setInterval(check, 300000);  // check every 5 minutes
}})();

// ---- detail modal ----
const overlay = document.getElementById('overlay');
// Detailed, readable breakdown of every alt-data input behind a signal (for the Signals sub-tab).
function _signalsDetail(s) {{
  const short = s.direction === 'SHORT';
  const cards = [];
  const card = (icon, title, value, tone, why) =>
    `<div class="sigdet ${{tone||''}}"><div class="sigdet-h">${{icon}} <b>${{title}}</b>`
    + `<span class="sigdet-v">${{value}}</span></div><div class="sigdet-why">${{why}}</div></div>`;

  // Relative strength
  const rs = (s.factors||{{}}).rs;
  if (rs && rs.pct!=null) {{
    const lead = rs.pct>=70, lag = rs.pct<=40;
    cards.push(card(_ico('trend-up',14),'Relative strength', 'RS '+rs.pct+' percentile',
      lead?'good':lag?'bad':'warn',
      lead ? 'Outrunning most of the market — leadership, which tends to persist.'
           : lag ? 'Lagging the market — relative weakness.'
                 : 'Middle of the pack versus the market — no strong lead either way.'));
  }}
  // TradingView
  if (s.tv && (s.tv.d||s.tv.w)) {{
    const agree = (!short && /Buy/.test(s.tv.d||'')) || (short && /Sell/.test(s.tv.d||''));
    cards.push(card(_ico('scale',14),'TradingView rating', (s.tv.d||'—')+' daily · '+(s.tv.w||'—')+' weekly',
      agree?'good':'warn',
      'An independent technical read (≈26 indicators), separate from our engine. '
      + (agree ? 'It lines up with this '+(short?'short':'long')+'.' : 'It does not strongly confirm this side — weigh it.')));
  }}
  // Insider (SEC Form 4)
  const ins = s.insider;
  if (ins && ins.n_filings) {{
    if (ins.cluster_buy)
      cards.push(card(_ico('bank',14),'Insider activity (SEC Form 4)', ins.buys+' open-market buy(s)', short?'bad':'good',
        ins.buys+' insider purchase(s) of '+(ins.buy_shares||0).toLocaleString()+' shares recently — real money down. '
        + (short?'A headwind for a short.':'A bullish vote of confidence.')));
    else if (ins.sells>=2)
      cards.push(card(_ico('bank',14),'Insider activity (SEC Form 4)', ins.sells+' sale(s)', short?'good':'warn',
        ins.sells+' insider sale(s) recently' + (short?' — supports the short.':' and little buying — mild caution.')));
    else
      cards.push(card(_ico('bank',14),'Insider activity (SEC Form 4)', 'no clear cluster', '',
        'No notable cluster of insider buys or sells in recent filings.'));
  }}
  // Analyst rating changes
  const aa = (s.fundamentals||{{}}).analyst_actions, lt = aa && aa.latest;
  if (lt && (lt.action==='up'||lt.action==='down')) {{
    const up = lt.action==='up';
    cards.push(card(_ico(up?'arrow-up':'arrow-dn',14),'Analyst rating change',
      (lt.firm||'analyst')+': '+(lt.from?lt.from+' &rarr; ':'')+(lt.to||''), up?(short?'bad':'good'):(short?'good':'warn'),
      '60-day net: '+(aa.n_up||0)+' upgrades / '+(aa.n_down||0)+' downgrades. Latest on '+(lt.date||'')+'. '
      + (up?'Fresh upgrades are a supportive catalyst for a long.':'Net downgrades lean bearish.')));
  }}
  // Retail buzz
  const b = s.buzz;
  if (b && b.lean) {{
    const lean = b.lean==='bull'?'Bullish':b.lean==='bear'?'Bearish':'Mixed';
    cards.push(card(_ico('chat',14),'Retail buzz (StockTwits)', lean+' · '+(b.sentiment_pct)+'% bullish',
      b.lean==='mixed'?'warn':'',
      (b.n)+' recent posts tagged; '+(b.sentiment_pct)+'% bullish. Crowd sentiment is noisy and often contrarian — weighted gently.'));
  }}
  // News-driven idea (LLM read of headlines)
  const ni = s.news_idea;
  if (ni && ni.direction) {{
    const aligns = (ni.direction==='bearish') === short;
    cards.push(card(_ico('news',14),'News-driven read', ni.direction+' · '+(ni.confidence||'')+' conf',
      aligns?'good':'warn',
      _esc(ni.reason||'') + (ni.headline?` (from: “${{_esc(ni.headline)}}”)`:'')));
  }}
  // Catalyst (fresh news)
  if (s.catalyst) {{
    cards.push(card(_ico('bolt',14),'News catalyst', (s.catalyst.source||'news'),
      '', 'Fresh headline driving attention: “'+_esc(s.catalyst.headline||'')+'”.'));
  }}
  // News tone
  const sent = s.sentiment;
  if (sent && sent.label && sent.label!=='Neutral') {{
    cards.push(card(_ico('news',14),'News tone', sent.label, sent.label==='Positive'?(short?'bad':'good'):(short?'good':'warn'),
      'Overall tone across '+(sent.n||'recent')+' headlines reads '+sent.label.toLowerCase()+'.'));
  }}
  return cards.length ? cards.join('')
    : '<div class="sigdet"><div class="sigdet-why">No extra signal data for this name on this run '
      + '(insider/analyst/buzz data is sparse and only appears on live runs).</div></div>';
}}

// ---- per-headline news implication (heuristic keyword read; no API cost) ----
const _NEWS_POS = ['beat','beats','surge','surges','soar','soars','rally','rallies','gain','gains','jump','jumps',
  'upgrade','upgraded','upgrades','outperform','growth','grow','grows','profit','profits','record','strong',
  'bullish','rebound','rebounds','firepower','tops','wins','win','approval','approved',
  'expansion','breakthrough','momentum','boost','boosts','optimistic','upside','accelerate','rise','rises',
  'soaring','green','greenlight','demand','partnership','buyback','dividend'];
const _NEWS_NEG = ['miss','misses','plunge','plunges','fall','falls','drop','drops','decline','declines','sink',
  'sinks','slump','downgrade','downgraded','downgrades','lawsuit','probe','investigation','fraud','defeat','risk',
  'risks','loss','losses','weak','bearish','warning','warns','recall','halt','ban','fine','fined','slash','slashes',
  'layoff','layoffs','bankruptcy','default','concern','concerns','fears','plummet','tumble','selloff','sue','sued',
  'delay','delays','disappoint','disappoints','crash','slows','slowing','cuts','subpoena','dispute'];
function _newsLean(h) {{
  const words = (h||'').toLowerCase().replace(/[^a-z0-9\\s-]/g,' ').split(/\\s+/);
  let p=0,q=0; const hit=[];
  words.forEach(w => {{ if (_NEWS_POS.indexOf(w)>=0) {{ p++; if (hit.indexOf(w)<0 && hit.length<3) hit.push(w); }}
                       else if (_NEWS_NEG.indexOf(w)>=0) {{ q++; if (hit.indexOf(w)<0 && hit.length<3) hit.push(w); }} }});
  return {{ lean: p>q ? 'bull' : q>p ? 'bear' : 'flat', hit }};
}}
function _newsSent(h) {{ const l=_newsLean(h).lean;
  return l==='bull'?{{t:'Bullish',c:'var(--buy)'}}:l==='bear'?{{t:'Bearish',c:'var(--sell)'}}:{{t:'Neutral',c:'var(--muted)'}}; }}
function _newsImpact(h, s) {{ const l=_newsLean(h).lean, short=s.direction==='SHORT';
  if (l==='flat') return {{t:'Neutral for your',c:'var(--muted)',g:'•'}};
  const helps = (l==='bull') !== short;  // bullish helps a long; bearish helps a short
  return helps ? {{t:'Supports your',c:'var(--buy)',g:'▲'}} : {{t:'Works against your',c:'var(--sell)',g:'▼'}}; }}
function _newsTip(n, s) {{
  const r=_newsLean(n.headline), sym=s.symbol, short=s.direction==='SHORT', dirw=short?'short':'long';
  let lead, rel;
  if (r.lean==='bull') {{ lead=`Reads bullish for ${{sym}} — positive coverage / potential catalyst.`;
    rel = short ? `Bullish news pushes the price up, which works against your short.` : `Bullish news supports your long.`; }}
  else if (r.lean==='bear') {{ lead=`Reads bearish for ${{sym}} — negative coverage / risk flagged.`;
    rel = short ? `Bearish news pushes the price down, which backs your short.` : `Bearish news is a headwind for your long.`; }}
  else {{ lead=`Neutral / unclear for ${{sym}} — context, not a clear catalyst.`; rel=`No clear push for or against this ${{dirw}}.`; }}
  const flags = r.hit.length ? ` Flags: ${{r.hit.join(', ')}}.` : '';
  return `${{lead}}${{flags}} ${{rel}} (Heuristic read of the headline text — verify before acting.)`;
}}
function openModal(s) {{
  // restore the standard tabbed modal chrome (an ORB modal may have hidden it)
  const _t = document.getElementById('mkTop'), _n = document.getElementById('mkNav');
  if (_t) _t.style.display = ''; if (_n) _n.style.display = '';
  const _ovOrb = document.getElementById('mkview-orb'); if (_ovOrb) {{ _ovOrb.classList.remove('on'); _ovOrb.innerHTML = ''; }}
  const cls = (s.action||'').replace(' ','');
  // ---- glass header: logo tile · ticker · direction pill · live price · %chg ----
  const _short = (s.direction === 'SHORT');
  const _initials = (s.symbol.replace(/[^A-Za-z]/g,'').slice(0,2) || s.symbol.slice(0,2)).toUpperCase();
  document.getElementById('mLogo').innerHTML =
    `<span class="mhead-init">${{_initials}}</span>`
    + `<img src="https://assets.parqet.com/logos/symbol/${{s.symbol}}?format=png" alt="" loading="lazy" onerror="this.remove()">`;
  document.getElementById('mTick').textContent = s.symbol;
  const _dirIco = _short ? _ico('trend-dn',13) : _ico('trend-up',13);
  document.getElementById('mPill').className = 'mhead-pill a-' + cls;
  document.getElementById('mPill').innerHTML = _dirIco + '<span>' + s.action + '</span>';
  document.getElementById('mName').textContent =
    (s.name || '') + (s.exchange ? (s.name ? ' · ' : '') + s.exchange : '');
  document.getElementById('mPx').innerHTML =
    `<span data-px="${{s.symbol}}">$${{Number(s.price).toLocaleString(undefined,{{maximumFractionDigits:2}})}}</span>`;
  const _dc = s.context && s.context.day_change_pct;
  const _chgEl = document.getElementById('mChg');
  if (_dc != null) {{
    _chgEl.className = 'mhead-chg ' + (_dc >= 0 ? 'up' : 'dn');
    _chgEl.innerHTML = (_dc >= 0 ? _ico('arrow-up',12) : _ico('arrow-dn',12))
      + `<span>${{_dc >= 0 ? '+' : ''}}${{_dc.toFixed(2)}}%</span>`;
  }} else {{ _chgEl.className = 'mhead-chg'; _chgEl.textContent = ''; }}
  document.getElementById('mSummary').textContent = s.summary || '';
  document.getElementById('mDesk').textContent = s.desk_read || '';
  const pel = document.getElementById('mPatterns');
  pel.innerHTML = (s.patterns||[]).length
    ? (s.patterns||[]).map(p => `<span class="chip ${{p.kind}}">${{p.label}}</span>`).join('')
    : '<span style="color:var(--muted);font-size:13px;">No standout chart patterns right now.</span>';
  // research: analysts, fundamentals, news tone
  const rel = document.getElementById('mResearch');
  const short = s.direction === 'SHORT';
  // color by whether a signal HELPS this trade's direction, not by raw bullishness:
  // on a short, bullish news/ratings/upside are headwinds (red), bearish are supportive (green).
  const dirCls = (bullish) => bullish === null ? '' : ((bullish !== short) ? 'buy' : 'sell');
  const fu = s.fundamentals || {{}}, an = fu.analysts, sen = s.sentiment;
  let rcells = '';
  const statc = (l,v,cls)=>`<div class="stat"><div class="l">${{l}}</div><div class="v ${{cls||''}}" style="font-size:15px;">${{v}}</div></div>`;
  if (an) {{
    const cc = dirCls(an.consensus==='Buy'?true:an.consensus==='Sell'?false:null);
    rcells += statc('Analyst consensus', an.consensus, cc) + statc('Buy / Hold / Sell', `${{an.buy}} / ${{an.hold}} / ${{an.sell}}`);
  }}
  if (fu.target_mean) {{
    const up = ((fu.target_mean/s.price-1)*100);
    rcells += statc('Avg price target', '$'+fu.target_mean.toLocaleString(), dirCls(up>=0))
            + statc('Upside to target', (up>=0?'+':'')+up.toFixed(0)+'%', dirCls(up>=0));
  }}
  if (fu.pe) rcells += statc('P/E ratio', fu.pe);
  if (fu.earnings_date) {{
    const ed = fu.earnings_days;
    rcells += statc('Next earnings', fu.earnings_date + (ed!=null?` (${{ed}}d)`:''), (ed!=null && ed<=7)?'sell':'');
  }}
  if (sen && sen.label) rcells += statc('News tone', sen.label, dirCls(sen.label==='Positive'?true:sen.label==='Negative'?false:null));
  rel.innerHTML = rcells || '<div style="color:var(--muted);font-size:13px;">No analyst/fundamental data available'
    + (sen ? '' : ' (add a Finnhub key to enable it)') + '.</div>';
  const eel = document.getElementById('mEdge'), e = s.edge;
  if (e && e.n_trades) {{
    const money = v => (v==null?'–':(v>0?'+':'')+v+'%');
    eel.innerHTML =
      `<div class="stat"><div class="l">Win rate</div><div class="v">${{e.win_rate==null?'–':e.win_rate+'%'}}</div></div>`
      + `<div class="stat"><div class="l">Past trades</div><div class="v">${{e.n_trades}}</div></div>`
      + `<div class="stat"><div class="l">Total return</div><div class="v ${{e.total_return>=0?'buy':'sell'}}">${{money(e.total_return)}}</div></div>`
      + `<div class="stat"><div class="l">Worst drawdown</div><div class="v sell">${{money(e.max_drawdown)}}</div></div>`;
  }} else {{
    eel.innerHTML = '<div style="color:var(--muted);font-size:13px;">Not enough past trades on this stock to measure an edge yet.</div>';
  }}
  // strategies in play: which independent methods are long now + their edge here
  const sel = document.getElementById('mStrategies'), sd = s.strategies || {{}};
  if (sel) {{
    const now = sd.now || {{}}, res = now.results || {{}}, edges = ((sd.edges || {{}}).by) || {{}};
    let chips = '';
    Object.keys(res).forEach(k => {{
      const r = res[k]; const cls = r.long ? (r.fresh ? 'bull' : 'neutral') : '';
      chips += `<span class="chip mini ${{cls}} hint" data-tip="${{_esc(STRAT_INFO[r.label] || r.kind)}}">${{r.long ? '●' : '○'}} ${{r.label}}</span>`;
    }});
    let rows = '';
    Object.keys(edges).forEach(k => {{
      const e = edges[k];
      const wr = (e.win_rate == null) ? null : e.win_rate;
      const ret = (e.total_return >= 0 ? '+' : '') + e.total_return + '%';
      const dcol = e.side === 'short' ? 'var(--sell)' : 'var(--buy)';
      const bcol = (wr == null) ? 'var(--muted)' : (wr >= 50 ? 'var(--buy)' : 'var(--sell)');
      rows += `<div class="strow hint" data-tip="${{_esc(STRAT_INFO[e.label] || '')}}">`
        + `<span class="st-dot" style="background:${{dcol}};"></span>`
        + `<span class="st-nm">${{e.label}}<span class="st-kind">${{e.kind}}</span></span>`
        + `<span class="st-bar"><i style="width:${{wr == null ? 0 : wr}}%;background:${{bcol}};"></i></span>`
        + `<span class="st-wr">${{wr == null ? '–' : wr + '%'}}</span>`
        + `<span class="st-tr">${{e.n_trades}}t</span>`
        + `<span class="st-ret ${{e.total_return >= 0 ? 'win' : 'loss'}}">${{ret}}</span></div>`;
    }});
    const head = `<div style="margin-bottom:12px;font-size:13px;"><b>${{now.count || 0}}</b> of ${{now.total || 0}} strategies are long here now (● long · ○ flat). <span style="color:var(--muted);">Hover a name for what it means.</span></div>`
      + `<div class="chips" style="margin-bottom:14px;">${{chips}}</div>`;
    const table = rows
      ? `<div class="strows">${{rows}}</div>`
      : '<div style="color:var(--muted);font-size:13px;">Per-strategy backtests are computed for the shown signals.</div>';
    sel.innerHTML = head + table;
  }}
  const aiHead = document.getElementById('mAIHead'), aiBox = document.getElementById('mAI');
  if (s.ai_read) {{
    aiBox.innerHTML = _md(s.ai_read); aiBox.style.display = 'block'; aiHead.style.display = 'block';
  }} else {{
    aiBox.style.display = 'none'; aiHead.style.display = 'none';
  }}

  const conv = s.conviction || {{}};
  document.getElementById('mConvScore').innerHTML = conv.label
    ? `<span class="convbadge conv-${{conv.label}}">${{conv.passes}}/${{conv.total}} checks passed</span>`
    : '';
  const icon = {{pass:_ico('check',13), warn:_ico('warn',13), fail:_ico('x',13)}};
  document.getElementById('mChecks').innerHTML = (conv.checks||[]).map(c =>
    `<li class="${{c.status}}"><span class="ic">${{icon[c.status]}}</span>`
    + `<span><span class="ck-l">${{c.label}}</span> — <span class="ck-n">${{c.note}}</span></span></li>`
  ).join('');

  const p = s.plan || {{}}, ctx = s.context || {{}};
  const money = v => (v==null ? '–' : '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}}));
  const pct = v => (v==null ? '–' : (v>0?'+':'')+v+'%');
  const stat = (label, value, sub, cls) =>
    `<div class="stat"><div class="l">${{label}}</div><div class="v ${{cls||''}}">${{value}}</div>${{sub?`<div class="sub">${{sub}}</div>`:''}}</div>`;
  const _active = (s.action==='BUY'||s.action==='HOLD LONG'||s.action==='SHORT'||s.action==='HOLD SHORT');
  const _dirWord = _short ? 'short' : 'long';
  document.getElementById('mPlanNote').textContent =
    _active ? `(${{_dirWord}} — active)` : `— levels if you took this ${{_dirWord}}`;
  try {{ _mChart(s); }} catch (e) {{}}
  // Option E — verdict strip + plan summary at the top of the overview
  (function() {{
    const cpct = conv.score_pct || 0;
    const rag = cpct >= 70 ? 'var(--buy)' : (cpct >= 50 ? 'var(--warn)' : 'var(--sell)');
    const cm = s.committee;
    let vword, vtone = '', vsup = '';
    if (cm) {{
      vword = ({{accept:'STRONG BUY', reduce:'REDUCE', reject:'AVOID'}})[cm.verdict] || String(cm.verdict||'').toUpperCase();
      vtone = ({{accept:'up', reduce:'warn', reject:'dn'}})[cm.verdict] || '';
      vsup = cm.support != null ? (' · ' + cm.support + '/4') : '';
    }} else {{
      const a = s.action || 'Signal';
      vword = a;
      vtone = _short ? 'dn' : ((a.indexOf('BUY') >= 0 || a.indexOf('LONG') >= 0) ? 'up' : '');
    }}
    const vEl = document.getElementById('mVerdict');
    if (vEl) vEl.innerHTML = conv.label ? (
      '<span class="mv-badge ' + vtone + '">' + vword + vsup + '</span>'
      + '<div class="mv-meter"><i style="width:' + cpct + '%;background:' + rag + ';"></i></div>'
      + '<span class="mv-score" style="color:' + rag + ';">' + cpct + ' · ' + conv.label + '</span>'
    ) : '';
    const pt = document.getElementById('mPlanTop');
    if (pt) pt.innerHTML = (p.entry != null) ? (
      '<div class="pt"><div class="pt-l">Entry</div><div class="pt-v">' + money(p.entry) + '</div></div>'
      + '<div class="pt"><div class="pt-l">Target</div><div class="pt-v up">' + (_short ? '−' : '+') + p.target_pct + '%</div></div>'
      + '<div class="pt"><div class="pt-l">Stop</div><div class="pt-v dn">' + (_short ? '+' : '−') + p.stop_pct + '%</div></div>'
      + '<div class="pt"><div class="pt-l">R : R</div><div class="pt-v">' + (p.rr != null ? ('1:' + p.rr) : '–') + '</div></div>'
    ) : '';
  }})();
  const _scen = (Array.isArray(p.targets) && p.targets.length)
    ? `<div class="scen"><div class="scen-h">${{_ico('target',13)}} Target scenarios <span>— the order uses the Base case; the others are where you could scale out or run it</span></div>`
      + p.targets.map(t => {{
          const cls = (t.odds||'').replace(/ /g,'');
          return `<div class="scen-row ${{cls}}"><div class="scen-top"><b>${{t.label}}</b>`
            + `<span class="scen-px">${{money(t.price)}} <em>${{_short?'−':'+'}}${{Math.abs(t.pct)}}% · ${{t.r}}R · ${{t.odds}}</em></span></div>`
            + `<div class="scen-why">${{t.basis}}</div></div>`;
        }}).join('') + `</div>`
    : '';
  document.getElementById('mPlan').innerHTML =
    stat('Entry', money(p.entry), _short ? 'short here' : 'current price') +
    stat('Stop-loss', money(p.stop), `${{_short?'+':'−'}}${{p.stop_pct}}%  ·  ATR-based`, 'sell') +
    stat(_short ? 'Cover target' : 'Take-profit', money(p.target), `${{_short?'−':'+'}}${{p.target_pct}}%  ·  ${{p.target_basis||'base target'}}`, 'buy') +
    stat('Risk : Reward', p.rr!=null ? ('1 : '+p.rr) : '–', 'reward per $1 risked') +
    stat('Est. time to play out', _holdTxt(p), 'to reach the target at typical pace') +
    stat('Position size', (p.shares||0)+' sh', money(p.exposure)+' exposure') +
    stat('$ at risk', money(p.dollar_risk), `${{p.shares||0}} sh to stop`, 'sell') +
    _scen;
  document.getElementById('mContext').innerHTML =
    stat('Today', pct(ctx.day_change_pct)) +
    stat('Volatility (ATR)', money(ctx.atr), (ctx.atr_pct!=null?ctx.atr_pct+'% of price':'')) +
    stat('Vs trend line', pct(ctx.vs_slow_ma_pct), 'price vs slow MA') +
    stat('From recent high', pct(ctx.pct_from_high), money(ctx.period_high)) +
    stat('From recent low', pct(ctx.pct_from_low), money(ctx.period_low)) +
    stat('History', (ctx.history_bars||0)+' bars', 'data depth');

  document.getElementById('mReasons').innerHTML =
    (s.reasons||[]).map(r => `<li>${{r}}</li>`).join('') || '<li>No details available.</li>';
  const nl = document.getElementById('mNews');
  nl.innerHTML = (s.news||[]).length
    ? (s.news||[]).map(n => {{
        const t = n.url ? `<a href="${{n.url}}" target="_blank" rel="noopener">${{n.headline}}</a>`
                        : `<span class="h">${{n.headline}}</span>`;
        const se = _newsSent(n.headline), im = _newsImpact(n.headline, s);
        return `<li class="hint" data-tip="${{_esc(_newsTip(n, s))}}">${{t}}`
          + `<div class="src"><span style="color:${{se.c}};font-weight:600;">${{se.t}} for ${{s.symbol}}</span>`
          + ` &middot; <span style="color:${{im.c}};font-weight:600;">${{im.g}} ${{im.t}} ${{s.direction==='SHORT'?'short':'long'}}</span>`
          + ` &middot; ${{n.source||''}} ${{n.created_at||''}}</div></li>`;
      }}).join('')
    : '<li class="src">No recent news tagged for this symbol.</li>';
  // ---- Signals sub-tab: every alt-data input in detail, with plain-English reasoning ----
  document.getElementById('mSignals').innerHTML = _signalsDetail(s);

  // ===== Intelligence + Trade sub-views (meta / structured / nlp / rank / liquidity) =====
  const so = s.structured || {{}};
  const _rng = so.return_range || {{}};
  const _fmtPct = v => (v==null?'—':((v>0?'+':'')+v+'%'));
  // Risk & sizing
  const mRisk = document.getElementById('mRisk');
  if (mRisk) mRisk.innerHTML =
    stat('Confidence', (so.confidence!=null?so.confidence:'—'), '0–100 conviction') +
    stat('Expected value', _fmtPct(so.expected_value_pct), 'probability-weighted') +
    stat('Return range', (_rng.upside_pct!=null? (_fmtPct(_rng.upside_pct)+' / '+_fmtPct(_rng.downside_pct)) : '—'), 'target / stop') +
    stat('Reward : risk', (so.rr!=null?('1 : '+so.rr):'—'), '') +
    stat('Hold (est.)', (so.expected_hold_days!=null?(so.expected_hold_days+' sessions'):'—'), 'to target at typical move') +
    stat('Risk score', (so.risk_score!=null?(so.risk_score+'/100'):'—'), 'volatility + illiquidity', (so.risk_score>=66?'sell':'')) +
    stat('Uncertainty', (so.uncertainty!=null?(so.uncertainty+'/100 · '+(so.uncertainty_band||'')):'—'), 'disagreement / mixed macro / thin liq', (so.uncertainty_band==='high'?'sell':'')) +
    stat('Size rec.', (so.size_recommendation||'—'), 'after meta + regime', (so.size_recommendation==='Skip'?'sell':so.size_recommendation==='Full'?'buy':''));
  const kc = so.kill_conditions || {{}};
  const mKill = document.getElementById('mKill');
  if (mKill) mKill.innerHTML = 'Exit if the <b>stop</b> ('+(kc.stop_pct!=null?kc.stop_pct+'%':'—')+') is hit. The whole book de-risks at <b>'
    + (kc.book_drawdown_halt_pct!=null?kc.book_drawdown_halt_pct:'—')+'% drawdown</b> or a <b>'+(kc.daily_loss_limit_pct!=null?kc.daily_loss_limit_pct:'—')
    + '% daily loss</b>, and the kill switch halts trading after repeated run failures.';
  // Execution / liquidity
  const lq = s.liquidity || {{}};
  const _dv = lq.dollar_volume;
  const mExec = document.getElementById('mExec');
  if (mExec) mExec.innerHTML =
    stat('Liquidity tier', (lq.tier||'—'), 'by daily $ turnover', (lq.tier==='illiquid'||lq.tier==='thin'?'sell':'')) +
    stat('Avg $ volume', (_dv!=null? ('$'+(_dv>=1e9?(_dv/1e9).toFixed(1)+'B':(_dv/1e6).toFixed(0)+'M')+'/day') : '—'), 'how much trades hands') +
    stat('Est. spread', (lq.spread_bps!=null?(lq.spread_bps+' bps'):'—'), 'modeled half-spread') +
    stat('Liquidity score', (so.liquidity_score!=null?(so.liquidity_score+'/100'):'—'), 'execution quality');
  // Meta verdict
  const mv = s.meta;
  const mMeta = document.getElementById('mMeta');
  if (mMeta) {{
    if (!mv) mMeta.innerHTML = '<div class="deskread">No meta verdict for this name on this run.</div>';
    else {{
      const _dc = ({{accept:'var(--buy)',reduce:'var(--warn)',delay:'var(--muted)',reject:'var(--sell)'}})[mv.decision] || 'var(--muted)';
      mMeta.innerHTML = '<div class="deskread" style="border-left-color:'+_dc+';"><b style="color:'+_dc+';text-transform:capitalize;">'+mv.decision+'</b>'
        + ((mv.decision==='reduce'&&mv.size_factor!=null)?(' — size × '+mv.size_factor):'')
        + '<ul style="margin:8px 0 0;padding-left:18px;line-height:1.7;">'+(mv.reasons||[]).map(r=>'<li>'+r+'</li>').join('')+'</ul></div>';
    }}
    const cm = s.committee;
    if (cm) {{
      const vc = ({{accept:'var(--buy)',reduce:'var(--warn)',reject:'var(--sell)'}})[cm.verdict] || 'var(--muted)';
      const leanc = l => l==='support' ? 'var(--buy)' : l==='against' ? 'var(--sell)' : 'var(--muted)';
      const RL = {{technicals:'Technicals',fundamentals:'Fundamentals',news:'News / catalyst',macro:'Macro / regime'}};
      const rolesH = Object.keys(RL).map(k => {{ const rv = (cm.roles||{{}})[k] || {{lean:'neutral',note:''}};
        return '<div class="crow"><span class="cr-role">'+RL[k]+'</span>'
          + '<span class="cr-lean" style="color:'+leanc(rv.lean)+';">'+rv.lean+'</span>'
          + '<span class="cr-note">'+_esc(rv.note||'')+'</span></div>'; }}).join('');
      mMeta.innerHTML += '<div class="sech ai-ident" style="margin-top:14px;">'+_ico('bank',13)+' Trade committee <span style="text-transform:none;color:var(--muted);font-size:12px;">— four AI analysts debate the setup; the chair rules</span></div>'
        + '<div class="deskread" style="border-left-color:'+vc+';"><b style="color:'+vc+';text-transform:uppercase;">'+cm.verdict+'</b> · confidence '+cm.confidence+'% · '+cm.support+'/4 support, '+cm.against+'/4 against'
        + (cm.summary ? (' — '+_esc(cm.summary)) : '') + '</div><div class="crows">' + rolesH + '</div>'
        + '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">Advisory second opinion. The rules risk engine keeps final authority.</p>';
    }}
  }}
  // Macro & regime fit
  const mp = DATA.macro_posture || {{}};
  const _fit = (s.rank_factors||{{}}).macrofit;
  const mRF = document.getElementById('mRegimeFit');
  if (mRF) {{
    let g = '<div class="plangrid">'
      + stat('Macro regime', (mp.label||'—'), (mp.score!=null?('composite '+mp.score):''))
      + stat('Exposure dial', (mp.exposure_mult!=null?(mp.exposure_mult+'×'):'—'), 'new-position sizing')
      + stat('This trade’s fit', (_fit!=null?(_fit+'/100'):'—'), 'direction vs regime', (_fit!=null&&_fit<40?'sell':_fit>=70?'buy':''))
      + (mp.entry_threshold?stat('Entry bar', mp.entry_threshold+'%', 'raised by regime'):'')
      + '</div>';
    const _tags = (mp.tags||[]).map(t=>'<span class="chip" title="'+_esc(t.why||'')+'">'+t.tag+'</span>').join(' ');
    if (_tags) g += '<div class="sech">Regime tags</div><div class="chips">'+_tags+'</div>';
    const _sb = mp.strategy_bias || {{}};
    if (_sb.favored) g += '<div class="sech">Favoured now</div><div style="font-size:13px;color:var(--txt2);">'+_sb.favored.join(' · ')+'</div>';
    mRF.innerHTML = g;
  }}
  // AI news read (LLM structured scores)
  const nlp = s.nlp;
  const mNR = document.getElementById('mNewsRead');
  if (mNR) {{
    if (!nlp) mNR.innerHTML = '<div class="deskread">No AI news read for this name this run (top actionable names only, live runs).</div>';
    else {{
      const dims = [['guidance','Guidance'],['demand_strength','Demand'],['management_confidence','Mgmt confidence'],['margin_pressure','Margin pressure'],['regulatory_risk','Regulatory risk'],['balance_sheet_concern','Balance-sheet'],['earnings_quality_risk','Earnings quality']];
      const cells = dims.map(d => {{ const v = nlp[d[0]]||0; const c = v>0?'var(--buy)':v<0?'var(--sell)':'var(--muted)';
        return '<div class="stat"><div class="l">'+d[1]+'</div><div class="v" style="color:'+c+';font-size:17px;">'+(v>0?'+':'')+v+'</div></div>'; }}).join('');
      const net = nlp.net; const nc = net>0.15?'var(--buy)':net<-0.15?'var(--sell)':'var(--muted)';
      mNR.innerHTML = '<div class="deskread">Net read: <b style="color:'+nc+';">'+(net>0?'+':'')+net+'</b>'+(nlp.note?(' — '+_esc(nlp.note)):'')+'</div>'
        + '<div class="plangrid">'+cells+'</div>'
        + '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">+ favourable, − a risk flag. Built from headlines only; it feeds the meta-model, it never places the trade.</p>';
    }}
  }}
  // Adaptive rank
  const rf = s.rank_factors, rs = s.rank_score;
  const mRk = document.getElementById('mRank');
  if (mRk) {{
    if (rs==null || !rf) mRk.innerHTML = '<div class="deskread">Not ranked (only actionable names get an allocation rank).</div>';
    else {{
      const bar = (lab,v) => {{ v=Math.max(0,Math.min(100,Math.round(v||0)));
        return '<div style="margin:7px 0;"><div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);"><span>'+lab+'</span><span>'+v+'</span></div>'
          + '<div style="height:7px;border-radius:4px;background:color-mix(in srgb,var(--accent) 14%,transparent);"><div style="height:100%;width:'+v+'%;border-radius:4px;background:var(--accent);"></div></div></div>'; }};
      mRk.innerHTML = '<div class="deskread">Allocation rank <b>'+rs+'</b>/100'+(s.rank?(' · #'+s.rank+' today'):'')+'</div>'
        + bar('Quality',rf.quality)+bar('Vol-adjusted reward',rf.vreward)+bar('Macro fit',rf.macrofit)+bar('Liquidity',rf.liquidity)+bar('Momentum',rf.momentum);
    }}
  }}

  // load this symbol into the modal's Capital IQ-style chart engine
  if (modalTC) modalTC.setSymbol(s.symbol, s.plan || {{}});
  if (window._mkShow) window._mkShow('overview');   // every open starts on Overview
  overlay.classList.add('open');
  try {{ history.replaceState(null, '', '#' + s.symbol); }} catch (e) {{}}   // shareable deep link
}}

// ---- Capital IQ-style chart engine: featured panel + watchlist + theme ----
function _symHue(s) {{ let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360; return h; }}
function _logoHTML(sym) {{
  const initials = (sym.replace(/[^A-Za-z]/g, '').slice(0, 2) || sym.slice(0, 2)).toUpperCase();
  const bg = `hsl(${{_symHue(sym)}},42%,42%)`;
  return `<span class="wl-logo" style="background:${{bg}};">${{initials}}`
    + `<img src="https://financialmodelingprep.com/image-stock/${{sym}}.png" alt="" loading="lazy" onerror="this.remove()">`
    + `</span>`;
}}
function buildWatchlist() {{
  const box = document.getElementById('featWatch'); if (!box) return;
  box.innerHTML = '';
  (DATA.signals || []).forEach((s, i) => {{
    const row = document.createElement('div');
    row.className = 'wl' + (i === 0 ? ' on' : ''); row.dataset.sym = s.symbol;
    const px = s.price != null ? ('$' + Number(s.price).toFixed(2)) : '';
    const dc = s.context && s.context.day_change_pct;
    const chg = (dc != null) ? `<span class="wl-chg" style="color:${{dc >= 0 ? 'var(--buy)' : 'var(--sell)'}};">${{dc >= 0 ? '+' : ''}}${{dc.toFixed(1)}}%</span>` : '';
    row.innerHTML = _logoHTML(s.symbol)
      + `<span class="wl-main"><span class="wl-sym">${{s.symbol}}</span>`
      + `<span class="wl-name">${{s.name || ''}}</span></span>`
      + `<span class="wl-r"><span class="wl-px" data-px="${{s.symbol}}">${{px}}</span>${{chg}}</span>`;
    row.onclick = () => {{
      box.querySelectorAll('.wl').forEach(x => x.classList.remove('on'));
      row.classList.add('on');
      if (featTC) featTC.setSymbol(s.symbol, s.plan || {{}});
    }};
    box.appendChild(row);
  }});
}}
function _initCharts() {{
  if (!window.TradeChart) {{ console.warn('chart engine not loaded'); return; }}
  const fEl = document.getElementById('featuredChart');
  const mEl = document.getElementById('modalChart');
  if (fEl) featTC = new TradeChart(fEl, {{ app: window.__APP, range: '6M', type: 'candle' }});
  if (mEl) modalTC = new TradeChart(mEl, {{ app: window.__APP, range: '6M', type: 'candle', compact: true }});
  buildWatchlist();
  const first = DATA.signals && DATA.signals[0];
  if (featTC && first) featTC.setSymbol(first.symbol, first.plan || {{}});
}}
// ---- light / dark theme ----
(function themeSetup() {{
  const KEY = 'tb-theme-v3';  // bumped for warm-gold redesign: drops stale 'light' prefs so dark is the default
  const btn = document.getElementById('themeToggle');
  function apply(t) {{
    document.documentElement.dataset.theme = t;
    if (btn) btn.innerHTML = (t === 'dark') ? (_ico('sun',14)+' Light') : (_ico('moon',14)+' Dark');
    if (featTC) featTC.applyTheme();
    if (modalTC) modalTC.applyTheme();
  }}
  let cur = 'dark';
  try {{ cur = localStorage.getItem(KEY) || 'dark'; }} catch (e) {{}}
  document.documentElement.dataset.theme = cur;
  if (btn) {{
    btn.innerHTML = (cur === 'dark') ? (_ico('sun',14)+' Light') : (_ico('moon',14)+' Dark');
    btn.onclick = () => {{
      const next = (document.documentElement.dataset.theme === 'dark') ? 'light' : 'dark';
      try {{ localStorage.setItem(KEY, next); }} catch (e) {{}}
      apply(next);
    }};
  }}
}})();
// ---- accent colour picker (persists; overrides --accent for both themes) ----
(function accentSetup() {{
  const KEY = 'tb-accent';
  const root = document.documentElement;
  const pop = document.getElementById('accentPop');
  const btn = document.getElementById('accentBtn');
  const cust = document.getElementById('accentCustom');
  if (!pop || !btn) return;
  function apply(c) {{ if (c) root.style.setProperty('--accent', c); else root.style.removeProperty('--accent'); }}
  function mark(c) {{ pop.querySelectorAll('.acsw').forEach(x => x.classList.toggle('on', x.dataset.accent === c)); }}
  let saved = null;
  try {{ saved = localStorage.getItem(KEY); }} catch (e) {{}}
  if (saved) {{ apply(saved); if (cust) cust.value = saved; mark(saved); }}
  btn.onclick = (e) => {{ e.stopPropagation(); pop.hidden = !pop.hidden; }};
  document.addEventListener('click', (e) => {{ if (!pop.hidden && !pop.contains(e.target) && e.target !== btn) pop.hidden = true; }});
  pop.querySelectorAll('.acsw').forEach(s => s.onclick = () => {{
    apply(s.dataset.accent); try {{ localStorage.setItem(KEY, s.dataset.accent); }} catch (e) {{}}
    if (cust) cust.value = s.dataset.accent; mark(s.dataset.accent); pop.hidden = true;
  }});
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') pop.hidden = true; }});
  if (cust) cust.oninput = () => {{ apply(cust.value); try {{ localStorage.setItem(KEY, cust.value); }} catch (e) {{}} mark(cust.value); }};
  const rst = document.getElementById('accentReset');
  if (rst) rst.onclick = () => {{ apply(null); try {{ localStorage.removeItem(KEY); }} catch (e) {{}} mark(null); pop.hidden = true; }};
}})();
// resize the featured chart when its panel becomes visible
function _refitCharts() {{
  try {{ if (featTC) featTC.resize(); }} catch (e) {{}}
}}
_initCharts();

function closeModal() {{ overlay.classList.remove('open'); try {{ history.replaceState(null, '', location.pathname + location.search); }} catch (e) {{}} }}
(function deepLink() {{
  function openFromHash() {{
    const h = (location.hash || '').replace('#', '').trim().toUpperCase();
    if (!h) return;
    const s = (DATA.signals || []).find(x => x.symbol === h);
    if (s) openModal(s);
  }}
  window.addEventListener('hashchange', openFromHash);
  openFromHash();   // open a shared #SYMBOL link on load
}})();
document.getElementById('modalClose').addEventListener('click', closeModal);
overlay.addEventListener('click', e => {{ if (e.target === overlay) closeModal(); }});
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

// ---- Live TV (pinned embeds) ----
// [key, label, videoId, watch-link]. We embed a SPECIFIC video id directly (no channel
// auto-resolve) because that reliably plays the exact feed we want — Bloomberg runs several
// concurrent streams, so auto-resolve grabbed the wrong one. Trade-off: a 24/7 stream that
// restarts gets a new id, so the embed can go blank until the id is refreshed -> the
// "open on YouTube" link below always works as the escape hatch.
const TV_CHANNELS = [
  ['bloomberg', 'Bloomberg', 'iEpJwprxDdk', 'https://www.youtube.com/@markets/live', 'Global markets, macro and business news, around the clock.', ['Markets','24/7']],
  ['yahoo', 'Yahoo Finance', 'KQp-e_XQnDE', 'https://www.youtube.com/@YahooFinance/live', 'US market coverage, earnings and single-stock news.', ['Stocks','Earnings']],
  ['schwab', 'Schwab Network', 'vKOd3v8VTYo', 'https://www.youtube.com/@SchwabNetwork/live', 'Trader-focused analysis, technicals and the open/close.', ['Trading','Technicals']],
  ['cnbc', 'CNBC', '', 'https://www.youtube.com/@CNBC/live', 'Breaking business news and market coverage (opens on YouTube).', ['News','Live']],
];
let _tvCur = 'bloomberg';
try {{ _tvCur = localStorage.getItem('tvch') || 'bloomberg'; }} catch (e) {{}}
function _tvSet(key) {{
  const ch = TV_CHANNELS.find(c => c[0] === key) || TV_CHANNELS[0];
  _tvCur = ch[0];
  const f = document.getElementById('tvFrame');
  // No pinned id (e.g. CNBC, whose live is login-gated) -> blank the player; the link below covers it.
  if (f) f.src = ch[2] ? `https://www.youtube.com/embed/${{ch[2]}}?autoplay=1&mute=1` : 'about:blank';
  const lk = document.getElementById('tvLink'); if (lk) lk.href = ch[3];
  document.querySelectorAll('#tvCards .tvcard').forEach(b => b.classList.toggle('on', b.dataset.tv === _tvCur));
  try {{ localStorage.setItem('tvch', _tvCur); }} catch (e) {{}}
}}
let _tvLoaded = false;
// Market sector heatmap — official TradingView Stock Heatmap widget, lazy-loaded on first open.
let _heatmapLoaded = false;
function _heatmapInit() {{
  if (_heatmapLoaded) return;
  const host = document.getElementById('heatmapHost');
  if (!host) return;
  _heatmapLoaded = true;
  const cont = document.createElement('div');
  cont.className = 'tradingview-widget-container';
  cont.style.height = '100%';
  const widget = document.createElement('div');
  widget.className = 'tradingview-widget-container__widget';
  widget.style.height = '100%';
  cont.appendChild(widget);
  const s = document.createElement('script');
  s.type = 'text/javascript';
  s.async = true;
  s.src = 'https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js';
  s.innerHTML = JSON.stringify({{
    dataSource: 'SPX500', exchanges: [], grouping: 'sector',
    blockSize: 'market_cap_basic', blockColor: 'change', locale: 'en',
    hasTopBar: true, isDataSetEnabled: true, isZoomEnabled: true, hasSymbolTooltip: true,
    colorTheme: (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'),
    width: '100%', height: '100%'
  }});
  cont.appendChild(s);
  host.appendChild(cont);
}}
function _tvInit() {{
  const grid = document.getElementById('tvCards');
  if (!grid || grid.childElementCount) return;
  TV_CHANNELS.forEach(c => {{
    const thumb = c[2] ? `https://img.youtube.com/vi/${{c[2]}}/hqdefault.jpg` : '';
    const tags = (c[5]||[]).map(t => `<span class="tvtag">${{t}}</span>`).join('');
    const label = c[2] ? '<span class="dot"></span> Live' : 'YouTube &rarr;';
    const el = document.createElement('div');
    el.className = 'tvcard' + (c[0] === _tvCur ? ' on' : '');
    el.dataset.tv = c[0];
    el.innerHTML = `<div class="tvthumb">${{thumb ? `<img src="${{thumb}}" loading="lazy" onerror="this.remove()">` : ''}}`
      + `<span class="tvlabel">${{label}}</span></div>`
      + `<div class="tvbody"><div class="tvname">${{c[1]}}</div><div class="tvdesc">${{c[4]||''}}</div>`
      + `<div class="tvtags">${{tags}}</div></div>`;
    el.onclick = () => _tvSet(c[0]);
    grid.appendChild(el);
  }});
}}

// ---- top mega-nav proxies clicks to the (hidden) sidebar buttons ----
(function setupMegaNav() {{
  document.querySelectorAll('#megaNav .mg-link[data-go]').forEach(a => {{
    a.addEventListener('click', () => {{
      const b = document.querySelector('#sideNav [data-area="' + a.dataset.go + '"]');
      if (b) b.click();
      else if (window._showPage && document.getElementById('page-' + a.dataset.go)) window._showPage(a.dataset.go);
    }});
  }});
}})();
// ---- tab navigation ----
(function setupTabs() {{
  // primary areas (sidebar) -> pages (top tabs). Pages not present in the DOM are filtered out.
  const AREAS = [
    ['signals', [['signals','Signals'],['control','Control'],['intraday','Intraday'],['orb','ORB day-trade'],['pairs','Pairs']]],
    ['markets', [['markets','Markets'],['heatmap','Heatmap'],['momentum','Momentum']]],
    ['portfolio', [['portfolio','Portfolio'],['paper','Paper account'],['allweather','All Weather']]],
    ['intel', [['altdata','Data signals']]],
    ['track', [['track','Track record'],['analytics','Edge explorer']]],
    ['premium', [['premium','Premium selling']]],
    ['analyst', [['analyst','Analyst']]],
    ['news', [['news','Market news'],['ipos','IPO watch']]],
    ['livetv', [['livetv','Live TV']]],
    ['about', [['method','How it works']]],
    ['system', [['system','System']]],
    ['agents', [['brain','Engine brain'],['agents','Agent universe']]],
    ['whatsnew', [['whatsnew',"What's new"]]]
  ];
  AREAS.forEach(a => a[1] = a[1].filter(p => document.getElementById('page-' + p[0])));
  const sideNav = document.getElementById('sideNav');
  const topTabs = document.getElementById('topTabs');
  if (!sideNav || !topTabs) return;
  const areaOf = page => AREAS.find(a => a[1].some(p => p[0] === page)) || AREAS[0];
  function renderTop(area) {{
    topTabs.innerHTML = area[1].map(p => `<button data-page="${{p[0]}}">${{p[1]}}</button>`).join('');
  }}
  function show(page) {{
    const area = areaOf(page);
    if (!area[1].some(p => p[0] === page)) page = (area[1][0] || ['signals'])[0];
    sideNav.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.area === area[0]));
    if (!topTabs.querySelector(`[data-page="${{page}}"]`)) renderTop(area);
    topTabs.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.page === page));
    document.querySelectorAll('.page').forEach(s => s.classList.toggle('on', s.id === 'page-' + page));
    try {{ localStorage.setItem('tab', page); }} catch (e) {{}}
    window.scrollTo(0, 0);
    if (page === 'markets') setTimeout(() => {{ try {{ window.dispatchEvent(new Event('resize')); }} catch (e) {{}} _refitCharts(); }}, 60);
    if (page === 'livetv') {{ _tvInit(); if (!_tvLoaded) {{ _tvLoaded = true; _tvSet(_tvCur); }} }}
    if (page === 'heatmap') _heatmapInit();
  }}
  window._showPage = show;
  sideNav.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {{
    const area = AREAS.find(a => a[0] === b.dataset.area);
    if (area && area[1].length) {{ renderTop(area); show(area[1][0][0]); }}
  }}));
  topTabs.addEventListener('click', e => {{ const b = e.target.closest('button'); if (b && b.dataset.page) show(b.dataset.page); }});
  let saved = 'signals';
  try {{ saved = localStorage.getItem('tab') || 'signals'; }} catch (e) {{}}
  if (!document.getElementById('page-' + saved)) saved = 'signals';
  renderTop(areaOf(saved));
  show(saved);
}})();

// ---- sidebar: live "Active signals" list + alpha/status footer ----
(function sidebarExtras() {{
  const list = document.getElementById('sideSigList');
  const wrap = document.getElementById('sideSignals');
  const foot = document.getElementById('sideFoot');
  const sigs = (DATA.signals || []).slice();
  // Top ~5 by conviction; direction marker from action/direction.
  if (list && wrap && sigs.length) {{
    const conv = s => ((s.conviction || {{}}).score_pct) || 0;
    const top = sigs.sort((a, b) => conv(b) - conv(a)).slice(0, 5);
    top.forEach(s => {{
      const cp = conv(s);
      const isShort = (s.direction === 'SHORT')
        || /SHORT/.test(s.action || '') || (s.action === 'SELL') || (s.action === 'EXIT');
      const dirIco = isShort
        ? `<span class="ss-dir" style="color:var(--sell);">${{_ico('trend-dn', 12)}}</span>`
        : `<span class="ss-dir" style="color:var(--buy);">${{_ico('trend-up', 12)}}</span>`;
      const row = document.createElement('div');
      row.className = 'side-sig-row';
      row.title = (s.name || s.symbol) + ' · ' + ((s.conviction || {{}}).label || '') + ' conviction';
      row.innerHTML = _logo2(s.symbol, 20)
        + `<span class="ss-sym">${{s.symbol}}</span>`
        + `<span class="ss-conv" style="color:${{_rag(cp)}};">${{cp}}</span>`
        + dirIco;
      row.addEventListener('click', () => openModal(s));
      list.appendChild(row);
    }});
    wrap.hidden = false;
  }}
  // Alpha / status footer: benchmark excess if present, else win rate + signal count.
  if (foot) {{
    const tk = DATA.track || {{}};
    const wr = (typeof tk.win_rate === 'number') ? tk.win_rate : null;
    const ex = (DATA.benchmark && typeof DATA.benchmark.avg_excess === 'number') ? DATA.benchmark.avg_excess : null;
    let mainLab, mainVal, mainCol, sub;
    if (ex != null) {{
      mainLab = 'Long α vs SPY';
      mainVal = (ex >= 0 ? '+' : '') + ex.toFixed(1) + '%';
      mainCol = ex >= 0 ? 'var(--buy)' : 'var(--sell)';
      sub = (wr != null ? wr.toFixed(0) + '% win rate' : '') + ' · ' + sigs.length + ' live';
    }} else if (wr != null) {{
      mainLab = 'Win rate vs SPY';
      mainVal = wr.toFixed(0) + '%';
      mainCol = wr >= 50 ? 'var(--buy)' : 'var(--sell)';
      const ar = (typeof tk.avg_return === 'number') ? ((tk.avg_return >= 0 ? '+' : '') + tk.avg_return.toFixed(1) + '% avg') : '';
      sub = (ar ? ar + ' · ' : '') + sigs.length + ' live signals';
    }} else {{
      mainLab = 'Active';
      mainVal = sigs.length;
      mainCol = 'var(--accent)';
      sub = 'live signals';
    }}
    foot.innerHTML = `<div class="sf-l">${{mainLab}}</div>`
      + `<div class="sf-v" style="color:${{mainCol}};">${{mainVal}}</div>`
      + `<div class="sf-sub">${{sub}}</div>`;
    foot.hidden = false;
  }}
}})();

// ---- app-bar ticker search: live-filter the signal cards; Enter opens an exact match ----
(function tickerSearch() {{
  const inp = document.getElementById('tickerSearch');
  if (!inp) return;
  const findExact = q => (DATA.signals || []).find(s => (s.symbol || '').toLowerCase() === q);
  inp.addEventListener('input', () => {{
    _searchTerm = inp.value.trim();
    // Only re-render the cards grid; if not on the signals page, jump there so the filter is visible.
    if (_searchTerm && window._showPage) {{
      const active = document.querySelector('.page.on');
      if (!active || active.id !== 'page-signals') window._showPage('signals');
    }}
    try {{ renderCards(); }} catch (e) {{}}
  }});
  inp.addEventListener('keydown', e => {{
    if (e.key === 'Enter') {{
      const q = inp.value.trim().toLowerCase();
      const hit = q && (findExact(q) || (DATA.signals || []).find(s => (s.symbol || '').toLowerCase().startsWith(q)));
      if (hit) {{ openModal(hit); }}
    }} else if (e.key === 'Escape') {{
      inp.value = ''; _searchTerm = ''; try {{ renderCards(); }} catch (err) {{}} inp.blur();
    }}
  }});
}})();

// ---- modal sub-views (left rail) ----
(function setupModalViews() {{
  const top = document.getElementById('mkTop');
  const nav = document.getElementById('mkNav');
  const mk = document.querySelector('.mk');
  if (!top || !nav) return;
  const topBtns = top.querySelectorAll('button');
  const sideBtns = nav.querySelectorAll('button');
  function showView(v) {{
    sideBtns.forEach(b => b.classList.toggle('on', b.dataset.mkview === v));
    document.querySelectorAll('.mk-view').forEach(p => p.classList.toggle('on', p.id === 'mkview-' + v));
    if (v === 'chart') setTimeout(() => {{ try {{ if (modalTC) modalTC.resize(); }} catch (e) {{}} }}, 50);
  }}
  function showTop(t) {{
    topBtns.forEach(b => b.classList.toggle('on', b.dataset.top === t));
    let first = null, count = 0;
    sideBtns.forEach(b => {{
      const inGroup = b.dataset.top === t;
      b.style.display = inGroup ? '' : 'none';
      if (inGroup) {{ count++; if (!first) first = b; }}
    }});
    // single-view top tabs (Overview / Chart) go full width with no side rail
    if (nav) nav.style.display = count > 1 ? '' : 'none';
    if (mk) mk.style.gridTemplateColumns = count > 1 ? '' : '1fr';
    if (first) showView(first.dataset.mkview);
  }}
  topBtns.forEach(b => b.addEventListener('click', () => showTop(b.dataset.top)));
  sideBtns.forEach(b => b.addEventListener('click', () => showView(b.dataset.mkview)));
  // _mkShow(viewId): jump straight to a sub-view, activating its parent top tab too
  window._mkShow = function(v) {{
    const btn = nav.querySelector('[data-mkview="' + v + '"]');
    const t = btn ? btn.dataset.top : 'overview';
    showTop(t);
    showView(v);
  }};
}})();

// ---- Markets sub-views (left rail) ----
(function setupMarketViews() {{
  const nav = document.getElementById('mktNav'); if (!nav) return;
  const btns = nav.querySelectorAll('button');
  function show(v) {{
    btns.forEach(b => b.classList.toggle('on', b.dataset.mview === v));
    document.querySelectorAll('.mkt-view').forEach(p => p.classList.toggle('on', p.id === 'mview-' + v));
    setTimeout(() => {{ try {{ window.dispatchEvent(new Event('resize')); }} catch (e) {{}} _refitCharts(); }}, 50);
  }}
  btns.forEach(b => b.addEventListener('click', () => show(b.dataset.mview)));
}})();

const news = document.getElementById('news');
(DATA.news||[]).forEach(n => {{
  const li=document.createElement('li');
  const title = n.url ? `<a href="${{n.url}}" target="_blank" rel="noopener">${{n.headline}}</a>`
                      : `<span class="h">${{n.headline}}</span>`;
  const tag = (n.symbols&&n.symbols.length)?` [${{n.symbols.join(', ')}}]`:'';
  li.innerHTML = `${{title}}<div class="src">${{n.source}} ${{n.created_at}}${{tag}}</div>`;
  news.appendChild(li);
}});
if (!(DATA.news||[]).length) news.innerHTML = '<li class="src">No news for flagged symbols.</li>';

// build the overview chart last so nothing else can be blocked by it
</script></body></html>"""


def main() -> None:
    snap = build_snapshot()
    # Self-audit the data and record a health badge for the dashboard.
    try:
        import audit
        checks, flags = audit.audit_data(snap)
        errs = [f["msg"] for f in flags if f.get("level") == "error"]
        warns = [f["msg"] for f in flags if f.get("level") == "warn"]
        snap["data_health"] = {"ok": not errs, "checks": checks,
                               "errors": errs[:20], "warnings": warns[:20],
                               "n_err": len(errs), "n_warn": len(warns)}
        # Categorise the errors so we can see at a glance WHAT is failing (and surface it
        # early in the JSON via audit_summary so a truncated fetch still shows it).
        def _cat(msg):
            m = msg.lower()
            if "one-day jump" in m or "split" in m: return "split/jump >50%"
            if "ohlc" in m: return "OHLC violation"
            if "timestamp" in m: return "timestamps"
            if "bollinger" in m: return "bollinger order"
            if "3mo return" in m or "return" in m: return "window return"
            if "macro" in m: return "macro range"
            return "field/range"
        by_type = {}
        for m in errs:
            k = _cat(m); by_type[k] = by_type.get(k, 0) + 1
        snap["audit_summary"] = {"n_err": len(errs), "n_warn": len(warns),
                                 "by_type": by_type, "errors": errs[:25]}
        if errs or warns:
            print(f"DATA AUDIT: {len(errs)} error(s), {len(warns)} warning(s) over {checks} checks:")
            for fl in (errs + warns)[:25]:
                print("  -", fl)
        else:
            print(f"DATA AUDIT: clean ({checks} checks).")
    except Exception as exc:  # noqa: BLE001
        snap["data_health"] = None
        print("DATA AUDIT: skipped —", exc)
    with open("signals.json", "w") as f:
        json.dump(snap, f, indent=2)
    with open("dashboard.html", "w") as f:
        f.write(render_html(snap))
    print(f"[{snap['mode']}] scanned {snap['scanned']}, showing {len(snap['signals'])} "
          f"-> dashboard.html / signals.json @ {snap['generated_at']}")
    for s in snap["signals"]:
        rv = f" relvol {s['rel_volume']}x" if s.get("rel_volume") else ""
        print(f"  {s['symbol']}: {s['action']} @ ${s['price']} (RSI {s['rsi']}){rv}")


if __name__ == "__main__":
    main()
