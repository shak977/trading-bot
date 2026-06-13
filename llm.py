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
