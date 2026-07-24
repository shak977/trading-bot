"""xAI (Grok) integration — the one model in the stack with LIVE access to X (Twitter) and the
real-time web. We use it for exactly what the other committee models can't do: a live social + news
read on a candidate, and an overnight catalyst sweep before the open.

Two entry points, both opt-in (need XAI_API_KEY) and fail-silent (never raise, never block a build):

  live_sentiment(symbol, name, direction, cfg) -> {stance, fresh_catalyst, catalyst, social_volume,
      confidence, note} | None. Fed into conviction as a normal check, so the attribution loop grades
      it and self-weights it from real outcomes — it has to earn its influence like everything else.

  premarket_catalysts(cfg, limit) -> [{symbol, stance, catalyst, note}]. Fresh overnight movers from
      X/news to seed the scan (via news_candidates.json) so they get a full technical + episodic read.

xAI's API is OpenAI-compatible; we enable the server-side x_search / web_search tools so Grok answers
from real-time data, not stale training. A per-run call cap and a tiny in-process cache bound cost.
Temperature 0 for stability. Only ever surfaces structured JSON we can trust — no free-form price
predictions are requested or used.
"""
from __future__ import annotations

import json as _json
import os
import sys as _sys

import requests

_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
_calls = {"n": 0}      # per-process call counter (a fresh process per build, so this is per-run)
_cache: dict = {}      # symbol -> sentiment (or None), so we never pay twice for the same name in a build


def _key(cfg) -> str:
    return getattr(cfg, "xai_api_key", "") or os.getenv("XAI_API_KEY", "")


def _budget_ok(cfg) -> bool:
    cap = int(getattr(cfg, "xai_daily_call_cap", 40) or 40)
    return _calls["n"] < cap


def reset_budget() -> None:
    """Reset the per-run call counter + cache (tests; and any long-lived process)."""
    _calls["n"] = 0
    _cache.clear()


def _extract_text(resp: dict) -> str:
    """Pull the assistant's text out of a Responses-API reply (skips reasoning items), with
    fallbacks to a flattened output_text and to the legacy chat/completions shape."""
    out = resp.get("output")
    if isinstance(out, list):
        texts = []
        for item in out:
            if not isinstance(item, dict):
                continue
            for c in (item.get("content") or []):
                if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"):
                    texts.append(c["text"])
        if texts:
            return "\n".join(texts).strip()
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"].strip()
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _chat(prompt: str, cfg, *, live: bool = True, timeout: int = 45, max_tokens: int = 1800) -> dict:
    """One JSON call to Grok via the xAI Responses API with real-time X + web search. Returns a parsed
    dict, or {} on any failure / no key / spent budget. Never raises — but logs the HTTP status/error
    to stderr so a build's logs reveal WHY a read came back empty instead of hiding it."""
    key = _key(cfg)
    if not key or not _budget_ok(cfg):
        return {}
    model = getattr(cfg, "xai_model", "grok-4.3")
    # Responses API: `input` (not `messages`), `max_output_tokens`; reasoning models reject temperature.
    body = {"model": model, "max_output_tokens": max_tokens,
            "input": [{"role": "user", "content": prompt}]}
    if live:
        # Server-side real-time search over X + the web. These tool types are ONLY supported on the
        # Responses API (/v1/responses), not the legacy /v1/chat/completions endpoint.
        body["tools"] = [{"type": "x_search"}, {"type": "web_search"}]
    try:
        _calls["n"] += 1
        r = requests.post(f"{_BASE}/responses",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=body, timeout=timeout)
        if r.status_code >= 400:
            print(f"[xai] HTTP {r.status_code} from /responses (model={model}): {r.text[:240]}",
                  file=_sys.stderr)
            return {}
        return _parse_json(_extract_text(r.json()))
    except Exception as e:  # noqa: BLE001
        print(f"[xai] request failed (model={model}): {e}", file=_sys.stderr)
        return {}


def _parse_json(txt: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences + surrounding prose)."""
    try:
        if "```" in txt:
            parts = txt.split("```")
            txt = parts[1].lstrip("json").strip() if len(parts) > 1 else txt
        a, b = txt.find("{"), txt.rfind("}")
        return _json.loads(txt[a:b + 1]) if (a >= 0 and b >= 0) else {}
    except Exception:  # noqa: BLE001
        return {}


def normalize_sentiment(d: dict) -> dict | None:
    """Coerce a raw Grok reply into our sentiment shape. None if it isn't usable. (Split out so tests
    can exercise the parsing without the network.)"""
    if not isinstance(d, dict) or not d.get("stance"):
        return None
    stance = str(d.get("stance", "")).lower().strip()
    if stance not in ("bullish", "bearish", "mixed", "quiet"):
        return None
    return {
        "stance": stance,
        "fresh_catalyst": bool(d.get("fresh_catalyst")),
        "catalyst": str(d.get("catalyst", ""))[:80],
        "social_volume": str(d.get("social_volume", "normal")).lower().strip() or "normal",
        "confidence": int(max(0, min(100, int(d.get("confidence", 50) or 50)))),
        "note": str(d.get("note", ""))[:140],
        "source": "grok-live-x",
    }


def live_sentiment(symbol: str, name: str, direction: str, cfg) -> dict | None:
    """Real-time X/web social+news read on one candidate. None when disabled / no key / no answer."""
    if not getattr(cfg, "xai_live_sentiment_enabled", False) or not _key(cfg):
        return None
    sym = str(symbol).upper().strip().lstrip("$")
    if sym in _cache:
        return _cache[sym]
    side = "short" if str(direction).upper() == "SHORT" else "long"
    prompt = (
        f"Using ONLY real-time data from X (Twitter) and the web from the LAST 3 DAYS, assess the live "
        f"social + news picture for {sym} ({name}). We are weighing a {side} trade. Do NOT predict "
        "price or invent events — report only what is actually being said and whether there is a fresh, "
        "specific catalyst. Return ONLY JSON: "
        '{"stance":"bullish|bearish|mixed|quiet","fresh_catalyst":true|false,'
        '"catalyst":"<=10 words or empty","social_volume":"high|normal|low",'
        '"confidence":0-100,"note":"<=16 words"}')
    out = normalize_sentiment(_chat(prompt, cfg))
    _cache[sym] = out
    return out


def normalize_catalysts(d: dict, limit: int = 8) -> list[dict]:
    """Coerce a raw premarket reply into a clean symbol list (test-friendly, no network)."""
    names = d.get("names") if isinstance(d, dict) else None
    out: list[dict] = []
    if not isinstance(names, list):
        return out
    seen = set()
    for it in names:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol", "")).upper().strip().lstrip("$")
        if not sym or not sym.isalpha() or len(sym) > 5 or sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "stance": str(it.get("stance", "")).lower().strip(),
                    "catalyst": str(it.get("catalyst", ""))[:80], "note": str(it.get("note", ""))[:140],
                    "source": "grok-premarket"})
        if len(out) >= limit:
            break
    return out


def premarket_catalysts(cfg, limit: int = 8) -> list[dict]:
    """Overnight/pre-open sweep of X + news for US-listed names with a fresh, specific catalyst.
    [] when disabled / no key. Names are seeds only — the scan's technicals still decide."""
    if not getattr(cfg, "xai_premarket_scan_enabled", False) or not _key(cfg):
        return []
    prompt = (
        "Using ONLY real-time data from X (Twitter) and the web from the LAST 24 HOURS, list US-listed "
        "stocks with a FRESH, specific catalyst driving unusual attention right now (earnings surprise, "
        "guidance change, FDA/regulatory, M&A, major product or contract news). Exclude vague rumors "
        "with no source. Do NOT predict price. Return ONLY JSON: "
        '{"names":[{"symbol":"AAPL","stance":"bullish|bearish","catalyst":"<=10 words",'
        '"note":"<=16 words"}]}. '
        f"At most {limit} names, highest-conviction first.")
    return normalize_catalysts(_chat(prompt, cfg, max_tokens=900), limit=limit)
