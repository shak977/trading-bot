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


def _prompt(sig: dict) -> str:
    p, ctx, conv = sig.get("plan", {}), sig.get("context", {}), sig.get("conviction", {})
    news = sig.get("news", []) or []
    headlines = "; ".join(n.get("headline", "") for n in news[:4]) or "none"
    reasons = " ".join(f"- {r}" for r in sig.get("reasons", []))
    return f"""You are a senior desk trader and risk strategist. Write a concise (3-5 sentence)
read on the setup below for an experienced trader. Focus on capital protection, edge clarity,
and the invalidation level. Reason ONLY from the data given — do not invent numbers, catalysts,
or fundamentals. Do not give a recommendation to buy or sell; describe the setup and its risks.

Ticker: {sig.get('symbol')}
Signal: {sig.get('action')}  | Price: ${sig.get('price')}
Strategy reasoning:
{reasons}
Trade plan: entry ${p.get('entry')}, stop ${p.get('stop')} (-{p.get('stop_pct')}%),
target ${p.get('target')} (+{p.get('target_pct')}%), risk:reward 1:{p.get('rr')},
size {p.get('shares')} sh, ${p.get('dollar_risk')} at risk.
Context: today {ctx.get('day_change_pct')}%, ATR {ctx.get('atr_pct')}% of price,
{ctx.get('vs_slow_ma_pct')}% vs trend line, {ctx.get('pct_from_high')}% from recent high.
Conviction (rule-based): {conv.get('label')} ({conv.get('score_pct')}%).
Recent news headlines: {headlines}
"""


def analyst_note(sig: dict, cfg: Config, timeout: int = 20) -> str | None:
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
                "messages": [{"role": "user", "content": _prompt(sig)}],
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
