"""Diagnose the Anthropic AI layer. Prints masked key + the real API response."""
from __future__ import annotations

import requests
from config import CONFIG


def mask(s: str) -> str:
    return f"{s[:7]}…{s[-4:]} (len {len(s)})" if s else "(empty / not set)"


print("ANTHROPIC_API_KEY:", mask(CONFIG.anthropic_api_key))
print("LLM model        :", CONFIG.llm_model)
print("llm_enabled      :", CONFIG.llm_enabled)

if not CONFIG.llm_enabled:
    print("\n-> No key loaded. Add ANTHROPIC_API_KEY to your .env to test locally.")
    raise SystemExit

print("\nCalling Anthropic...")
try:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": CONFIG.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CONFIG.llm_model,
            "max_tokens": 60,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        },
        timeout=20,
    )
    print("HTTP status:", r.status_code)
    if r.status_code == 200:
        txt = "".join(b.get("text", "") for b in r.json().get("content", []))
        print("Model replied:", repr(txt.strip()))
        print("\n✅ Working — your weekly run will now generate AI notes.")
    else:
        print("Response body:", r.text[:400])
        print("\n❌ Call rejected — see the message above (common: bad model name,"
              " no billing credit, or wrong key).")
except Exception as exc:  # noqa: BLE001
    print("Request failed:", type(exc).__name__, "-", str(exc)[:200])
