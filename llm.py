"""Optional AI analyst layer (Anthropic Claude via REST).

Generates a short, natural-language analyst note for a signal from the numbers
we already computed. Entirely optional: if no ANTHROPIC_API_KEY is set, or the
call fails, callers fall back to the deterministic desk read. We never invent
data — the model is told to reason only from the facts we pass in.
"""
from __future__ import annotations

import requests

from config import Config

_URL = "https://api.anthropic.com/v1/messages"


def market_brief(regime, signals, sectors, momentum, news_ideas, macro, cfg: Config, timeout: int = 25):
    """A plain-English 'what's happening / what to watch' market brief, built ONLY from the data
    we already computed (no invented facts). Returns text or None — never raises."""
    if not getattr(cfg, "llm_enabled", False):
        return None
    try:
        sg = signals or []
        fresh_b = sum(1 for s in sg if s.get("is_fresh") and s.get("action") == "BUY")
        fresh_s = sum(1 for s in sg if s.get("is_fresh") and s.get("action") == "SHORT")
        top = sorted([s for s in sg if s.get("action") in ("BUY", "SHORT")],
                     key=lambda s: -((s.get("conviction") or {}).get("score_pct") or 0))[:6]

        def _sl(s):
            c = (s.get("conviction") or {}).get("score_pct")
            return f"{s.get('symbol')} {s.get('action')}" + (f" ({c}%)" if c is not None else "")
        secs = sorted(sectors or [], key=lambda x: -(x.get("pct_up") or 0))
        strong = ", ".join(f"{x['sector']} ({x['pct_up']}% up)" for x in secs[:3]) or "n/a"
        weak = ", ".join(f"{x['sector']} ({x['pct_up']}% up)" for x in secs[-3:]) or "n/a"
        mom = ", ".join(f"{m['symbol']} +{m.get('score')}%" for m in (momentum or [])[:5]) or "n/a"
        cats = [i for i in (news_ideas or []) if i.get("confidence") == "high"][:5]
        cat_txt = "; ".join(f"{i.get('ticker')} {i.get('direction')} — {i.get('reason')}" for i in cats) or "none flagged"
        reg = regime or {}
        macro_txt = (f"{macro.get('backdrop')} — VIX {macro.get('vix')} ({macro.get('vix_trend', 'flat')}), "
                     f"10y {macro.get('y10')}%, dollar idx {macro.get('dxy')}, WTI oil ${macro.get('oil')}, "
                     f"CPI {macro.get('cpi_yoy')}% YoY, unemployment {macro.get('unemployment')}%") if macro else "n/a"
        data = (
            f"Regime: {reg.get('label', 'n/a')} — breadth {reg.get('breadth', '?')}% of scanned stocks above trend, "
            f"avg momentum {reg.get('avg_rsi', '?')}/100.\n"
            f"Fresh today: {fresh_b} new long signals, {fresh_s} new shorts.\n"
            f"Top-conviction setups: {', '.join(_sl(s) for s in top) or 'none'}.\n"
            f"Strongest sectors: {strong}. Weakest: {weak}.\n"
            f"Momentum leaders (12-1): {mom}.\n"
            f"Notable news catalysts: {cat_txt}.\n"
            f"Macro backdrop: {macro_txt}."
        )
        prompt = (
            "You are a markets strategist writing a quick brief for a smart beginner. Using ONLY the "
            "data below (never invent numbers, names, or events), write 3-4 short, plain-English "
            "sentences: (1) what the market is doing right now and the mood; (2) what's driving it — "
            "name the strong/weak sectors and any real news catalysts; (3) what the screen is flagging "
            "today; (4) one practical 'what to watch'. Briefly define any jargon. No hype, and do not "
            "tell the reader to buy or sell.\n\nDATA:\n" + data
        )
        r = requests.post(
            _URL,
            headers={"x-api-key": cfg.anthropic_api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": cfg.llm_model, "max_tokens": 320,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        parts = [b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text"]
        return "".join(parts).strip() or None
    except Exception:  # noqa: BLE001
        return None


def _prompt(sig: dict, regime: dict | None = None, macro: dict | None = None) -> str:
    p, ctx, conv = sig.get("plan", {}), sig.get("context", {}), sig.get("conviction", {})
    f, edge = sig.get("factors", {}) or {}, sig.get("edge") or {}
    news = sig.get("news", []) or []
    headlines = "; ".join(n.get("headline", "") for n in news[:4]) or "none"
    pats = ", ".join(f"{pt['label']} ({pt['kind']})" for pt in sig.get("patterns", [])) or "none"
    sent = sig.get("sentiment") or {}
    fund = sig.get("fundamentals") or {}
    an = fund.get("analysts") or {}
    research_txt = "; ".join(filter(None, [
        f"news tone {sent.get('label')}" if sent.get("label") else None,
        f"analysts {an.get('consensus')} ({an.get('buy')}/{an.get('hold')}/{an.get('sell')} b/h/s)" if an else None,
        f"avg price target ${fund.get('target_mean')}" if fund.get("target_mean") else None,
        f"P/E {fund.get('pe')}" if fund.get("pe") else None,
    ])) or "none available"
    macro_txt = (f"{macro.get('backdrop')} — 10y {macro.get('y10')}%, curve {macro.get('curve')}, "
                 f"CPI {macro.get('cpi_yoy')}% YoY, unemployment {macro.get('unemployment')}%") if macro else "n/a"
    mh = f.get("macd_hist")
    macd_txt = "n/a" if mh is None else ("positive (momentum up)" if mh > 0 else "negative (momentum down)")
    edge_txt = (f"{edge.get('win_rate')}% win rate over {edge.get('n_trades')} past trades"
                if edge.get("n_trades") else "not enough past trades to judge")
    reg_txt = f"{regime.get('label')} ({regime.get('breadth')}% of scanned stocks above trend)" if regime else "n/a"
    return f"""You are a seasoned desk trader explaining a setup to a smart beginner. Write 3-5 short
sentences in plain, everyday English (briefly define any term like RSI/MACD). Reason like a trader:
weigh the CONFLUENCE of signals, name where the risk is, state the invalidation level, and note
whether the market backdrop helps or hurts. Use ONLY the data below — you may cite the analyst
consensus, price target, fundamentals, news tone and macro that ARE provided, but never invent any
numbers, catalysts, or earnings not shown here. Do NOT tell them to buy or sell; explain the setup and its risks.

Ticker: {sig.get('symbol')} ({sig.get('name','')})
Signal: {sig.get('action')}  | Price: ${sig.get('price')}
Why flagged: {' '.join('- '+r for r in sig.get('reasons', []))}
Chart patterns: {pats}
Indicators: MACD momentum {macd_txt}; ADX trend strength {f.get('adx')}; Bollinger position {f.get('bb_pct')} (0=low band,1=high band).
Trade plan: entry ${p.get('entry')}, stop ${p.get('stop')} (-{p.get('stop_pct')}%),
target ${p.get('target')} (+{p.get('target_pct')}%), risk:reward 1:{p.get('rr')}.
Context: today {ctx.get('day_change_pct')}%, daily swing {ctx.get('atr_pct')}%, {ctx.get('vs_slow_ma_pct')}% vs trend line,
{ctx.get('pct_from_high')}% from 1-yr high.
This strategy's history on this stock: {edge_txt}.
Research: {research_txt}.
Macro backdrop: {macro_txt}.
Rule-based conviction: {conv.get('label')} ({conv.get('score_pct')}%).
Market (stocks scanned): {reg_txt}.
Recent news: {headlines}
"""


def analyst_note(sig: dict, cfg: Config, regime: dict | None = None,
                 macro: dict | None = None, timeout: int = 20) -> str | None:
    if not cfg.llm_enabled:
        return None
    try:
        r = requests.post(
            _URL,
            headers={
                "x-api-key": cfg.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": cfg.llm_model,
                "max_tokens": 320,
                "messages": [{"role": "user", "content": _prompt(sig, regime, macro)}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        text = "".join(parts).strip()
        return text or None
    except Exception:  # noqa: BLE001 - any failure -> silent fallback
        return None


def news_ideas(news: list[dict], cfg: Config, universe: set | None = None, timeout: int = 25) -> list[dict]:
    """Read recent news HEADLINES and extract concrete, single-stock actionable ideas via one LLM
    call (cheap — once per run, not per symbol). Returns a list of:
        {ticker, direction: 'bullish'|'bearish', confidence: 'high'|'medium'|'low', reason, headline}
    Grounded only in the headlines provided; never invents prices/targets. [] on no key/failure.
    This is a NEWS-SENTIMENT read, explicitly separate from the confluence engine."""
    if not cfg.llm_enabled or not news:
        return []
    import json as _json
    items = [n for n in news if (n.get("headline") or "").strip()][:25]
    if not items:
        return []
    lines = "\n".join(f"- {n['headline']}" + (f"  [{n.get('source')}]" if n.get("source") else "")
                      for n in items)
    prompt = (
        "You are a markets analyst. From the news headlines below, extract concrete, actionable "
        "SINGLE-STOCK ideas. Rules:\n"
        "- Only US-listed individual stocks; give the ticker symbol. Skip macro/index/crypto/rates-only news.\n"
        "- direction: 'bullish' or 'bearish' for that stock, based only on the headline.\n"
        "- confidence: 'high' only for clear, material, company-specific catalysts (M&A, guidance, "
        "approvals, big beats/misses); 'low' for vague or already-priced news.\n"
        "- reason: one short sentence grounded ONLY in the headline. Do NOT invent numbers or targets.\n"
        "- At most 10 ideas; skip headlines with no clear single-stock implication.\n"
        'Return ONLY a JSON array, e.g. [{"ticker":"NVDA","direction":"bullish","confidence":"high",'
        '"reason":"...","headline":"..."}]. No prose.\n\nHEADLINES:\n' + lines)
    try:
        r = requests.post(
            _URL,
            headers={"x-api-key": cfg.anthropic_api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": cfg.llm_model, "max_tokens": 900,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        txt = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        # strip code fences / pull out the JSON array
        if "```" in txt:
            txt = txt.split("```")[1].lstrip("json").strip() if len(txt.split("```")) > 1 else txt
        a, b = txt.find("["), txt.rfind("]")
        if a < 0 or b < 0:
            return []
        raw = _json.loads(txt[a:b + 1])
    except Exception:  # noqa: BLE001
        return []
    out, seen = [], set()
    for it in raw if isinstance(raw, list) else []:
        try:
            tk = str(it.get("ticker", "")).upper().strip().lstrip("$")
            d = str(it.get("direction", "")).lower()
            c = str(it.get("confidence", "")).lower()
            if not tk or tk in seen or d not in ("bullish", "bearish"):
                continue
            if universe and tk not in universe and not (1 <= len(tk) <= 5 and tk.isalpha()):
                continue
            seen.add(tk)
            out.append({"ticker": tk, "direction": d,
                        "confidence": c if c in ("high", "medium", "low") else "medium",
                        "reason": str(it.get("reason", ""))[:200],
                        "headline": str(it.get("headline", ""))[:200]})
        except Exception:  # noqa: BLE001
            continue
    # strongest first
    _rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda x: _rank.get(x["confidence"], 1))
    return out[:10]


_NLP_DIMS = ("guidance", "margin_pressure", "demand_strength", "regulatory_risk",
             "balance_sheet_concern", "management_confidence", "earnings_quality_risk")


def structured_scores(rows: list[dict], cfg: Config, max_names: int = 6,
                      timeout: int = 30) -> dict:
    """Turn recent per-stock NEWS HEADLINES into structured, named scores via ONE batched LLM call
    (cheap — only the top actionable names). Each dimension is a signed effect on the thesis from
    -2 (clearly negative) to +2 (clearly positive). The model converts text → numbers; it never
    decides a trade. Returns {SYMBOL: {dim: score..., note, net}}; {} on no key / failure.

    Dimensions: guidance, margin_pressure, demand_strength, regulatory_risk, balance_sheet_concern,
    management_confidence, earnings_quality_risk — each signed so + is good for the stock and − is a
    risk flag (e.g. high regulatory risk → negative)."""
    if not cfg.llm_enabled or not rows:
        return {}
    import json as _json
    # pick the strongest actionable names that actually have news to read
    cand = [r for r in rows
            if r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT") and r.get("news")]
    cand.sort(key=lambda r: -((r.get("conviction") or {}).get("score_pct") or 0))
    cand = cand[:max_names]
    if not cand:
        return {}
    blocks = []
    for r in cand:
        heads = "; ".join((n.get("headline") or "").strip() for n in (r.get("news") or [])[:5] if n.get("headline"))
        if heads:
            blocks.append(f'{r["symbol"]} ({r.get("name","")}): {heads}')
    if not blocks:
        return {}
    prompt = (
        "You are an equity analyst. For each stock below, read ONLY the provided headlines and rate "
        "seven dimensions as a signed integer from -2 to +2 for their effect on the investment thesis "
        "(+2 clearly positive for the stock, 0 neutral/none, -2 clearly negative/high-risk):\n"
        "guidance, margin_pressure, demand_strength, regulatory_risk, balance_sheet_concern, "
        "management_confidence, earnings_quality_risk.\n"
        "Note: for risk dimensions (margin_pressure, regulatory_risk, balance_sheet_concern, "
        "earnings_quality_risk), a NEGATIVE number means the risk is ELEVATED (bad); 0 means no signal.\n"
        "If the headlines say nothing about a dimension, score it 0. Do NOT invent facts. Add a 'note' "
        "of at most 12 words grounded only in the headlines.\n"
        'Return ONLY a JSON object keyed by ticker, e.g. {"NVDA":{"guidance":1,"margin_pressure":0,'
        '"demand_strength":2,"regulatory_risk":0,"balance_sheet_concern":0,"management_confidence":1,'
        '"earnings_quality_risk":0,"note":"..."}}. No prose.\n\nSTOCKS:\n' + "\n".join(blocks))
    try:
        r = requests.post(
            _URL,
            headers={"x-api-key": cfg.anthropic_api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": cfg.llm_model, "max_tokens": 1100,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        txt = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        if "```" in txt:
            parts = txt.split("```")
            txt = parts[1].lstrip("json").strip() if len(parts) > 1 else txt
        a, b = txt.find("{"), txt.rfind("}")
        if a < 0 or b < 0:
            return {}
        raw = _json.loads(txt[a:b + 1])
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for sym, d in (raw.items() if isinstance(raw, dict) else []):
        try:
            rec = {}
            for k in _NLP_DIMS:
                v = d.get(k, 0)
                rec[k] = int(max(-2, min(2, int(v)))) if isinstance(v, (int, float)) else 0
            rec["note"] = str(d.get("note", ""))[:140]
            rec["net"] = round(sum(rec[k] for k in _NLP_DIMS) / len(_NLP_DIMS), 2)
            out[str(sym).upper().strip().lstrip("$")] = rec
        except Exception:  # noqa: BLE001
            continue
    return out


def analyst_review(report: dict, cfg: Config, timeout: int = 30) -> str | None:
    """A concise nightly analyst narrative built ONLY from the computed findings — what's working,
    what isn't, and the 2-3 highest-priority changes to consider. Advisory; never invents data.
    Returns text or None (no key / failure)."""
    if not getattr(cfg, "llm_enabled", False):
        return None
    try:
        bl = []
        for scope, b in (report.get("buckets") or {}).items():
            s = b.get("stats") or {}
            bl.append(f"- {b.get('label', scope)}: {s.get('decided', 0)} decided trades, "
                      f"win rate {s.get('win_rate')}%, {s.get('open', 0)} open; "
                      f"learned weights {s.get('learned_weights') or '{}'}; "
                      f"by-regime {s.get('by_regime') or '{}'}.")
        fl = [f"- [{f['severity']}] {f['area']}: {f['observation']} -> {f['proposal']}"
              for f in (report.get("findings") or [])[:14]]
        prompt = (
            "You are a quantitative trading systems analyst reviewing a paper-trading bot's own "
            "performance. Using ONLY the computed findings below (never invent numbers or events), "
            "write a tight nightly note for the system's owner: (1) one sentence on overall state per "
            "strategy bucket; (2) the 2-3 highest-priority changes you'd make and why; (3) one risk to "
            "watch. Be concrete and reference the specific checks/regimes/windows named. Plain English, "
            "no hype, do not give personal investment advice. Keep under 180 words.\n\n"
            "BUCKETS:\n" + "\n".join(bl) + "\n\nFINDINGS (already prioritised):\n" + "\n".join(fl))
        r = requests.post(
            _URL,
            headers={"x-api-key": cfg.anthropic_api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": cfg.llm_model, "max_tokens": 420,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        txt = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        return txt or None
    except Exception:  # noqa: BLE001
        return None


def diagnose(cfg: Config, timeout: int = 15) -> dict:
    """One tiny probe call so we can see WHY the analyst layer is/ isn't producing notes
    (used to populate a diagnostic field in signals.json). Never raises."""
    if not cfg.llm_enabled:
        return {"enabled": False, "reason": "ANTHROPIC_API_KEY not set"}
    try:
        r = requests.post(
            _URL,
            headers={"x-api-key": cfg.anthropic_api_key,
                     "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": cfg.llm_model, "max_tokens": 8,
                  "messages": [{"role": "user", "content": "Reply with the word ok."}]},
            timeout=timeout,
        )
        if r.status_code == 200:
            return {"enabled": True, "ok": True, "model": cfg.llm_model}
        return {"enabled": True, "ok": False, "model": cfg.llm_model,
                "status": r.status_code, "body": (r.text or "")[:240]}
    except Exception as e:  # noqa: BLE001
        return {"enabled": True, "ok": False, "model": cfg.llm_model, "error": str(e)[:240]}
