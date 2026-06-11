"""Diagnose Alpaca credentials. Prints masked info only — never your secret."""
from __future__ import annotations

from config import CONFIG


def mask(s: str) -> str:
    if not s:
        return "(empty)"
    return f"{s[:4]}…{s[-4:]} (len {len(s)})"


print("== .env values as loaded ==")
print("ALPACA_API_KEY   :", mask(CONFIG.api_key))
print("ALPACA_SECRET_KEY:", mask(CONFIG.secret_key))
print("ALPACA_PAPER     :", CONFIG.paper)
# whitespace / quote traps
for name, val in [("api_key", CONFIG.api_key), ("secret_key", CONFIG.secret_key)]:
    if val != val.strip():
        print(f"  !! {name} has leading/trailing whitespace")
    if val and (val[0] in "\"'" or val[-1] in "\"'"):
        print(f"  !! {name} has surrounding quotes — remove them")

print("\n== Test 1: Trading API (account) ==")
try:
    from alpaca.trading.client import TradingClient
    acct = TradingClient(CONFIG.api_key, CONFIG.secret_key, paper=CONFIG.paper).get_account()
    print("  OK — account status:", acct.status, "| equity:", acct.equity)
    trading_ok = True
except Exception as exc:  # noqa: BLE001
    print("  FAILED:", type(exc).__name__, "-", str(exc)[:200])
    trading_ok = False

print("\n== Test 2: Market Data API (IEX bars) ==")
try:
    from data import get_bars
    df = get_bars("AAPL", CONFIG)
    print(f"  OK — got {len(df)} bars for AAPL")
except Exception as exc:  # noqa: BLE001
    print("  FAILED:", type(exc).__name__, "-", str(exc)[:200])

print("\n== Verdict ==")
if not trading_ok:
    print("  Keys are invalid for trading too -> the key/secret are wrong.")
    print("  Fix: regenerate PAPER keys at")
    print("  https://app.alpaca.markets/paper/dashboard/overview (top-right, 'Generate')")
    print("  and paste BOTH the new Key ID and Secret into .env.")
else:
    print("  Trading works but data doesn't -> a market-data entitlement issue, not the keys.")
