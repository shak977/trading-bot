"""IBKR (Interactive Brokers) data client — OPT-IN, fails soft, read-only.

Talks to an authenticated Client Portal Web API gateway (kept alive by IBeam on your VPS).
Nothing here logs in, places orders, or sees your password — it only READS data from a
gateway you authenticate. Disabled unless ``cfg.ibkr_enabled`` and a gateway URL are set.

Every public function returns None / [] on any problem and NEVER raises, so a gateway hiccup
can't break the dashboard build. Wire-in to scanner/strategies happens later, behind the flag.

Requires the optional dependency ``ibind`` (only imported when enabled):  pip install ibind
See docs/IBKR_INTEGRATION.md for the VPS + IBeam setup.
"""
from __future__ import annotations

import os
from functools import lru_cache

from config import Config


def _conf(cfg: Config) -> dict:
    return {
        "enabled": bool(getattr(cfg, "ibkr_enabled", False)) or os.getenv("IBKR_ENABLED", "").lower() == "true",
        "url": (getattr(cfg, "ibkr_gateway_url", "") or os.getenv("IBKR_GATEWAY_URL", "")).strip(),
        "token": os.getenv("IBKR_GATEWAY_TOKEN", "").strip(),
        "account": (getattr(cfg, "ibkr_account_id", "") or os.getenv("IBKR_ACCOUNT_ID", "")).strip(),
        "timeout": int(getattr(cfg, "ibkr_timeout", 12) or 12),
    }


def enabled(cfg: Config) -> bool:
    c = _conf(cfg)
    return bool(c["enabled"] and c["url"])


@lru_cache(maxsize=1)
def _client_cached(url: str, token: str, timeout: int):
    """Build the ibind REST client once. Returns None if ibind isn't installed or init fails."""
    try:
        from ibind import IbkrClient  # optional dependency
    except Exception:  # noqa: BLE001 - not installed / import error
        return None
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else None
        # NB: exact IbkrClient kwargs depend on your ibind version + proxy — validate live.
        return IbkrClient(url=url, timeout=timeout, extra_headers=headers)
    except Exception:  # noqa: BLE001
        return None


def _client(cfg: Config):
    if not enabled(cfg):
        return None
    c = _conf(cfg)
    return _client_cached(c["url"], c["token"], c["timeout"])


def diagnose(cfg: Config) -> dict:
    """Connectivity + auth self-test for the System tab. Never raises."""
    c = _conf(cfg)
    if not c["enabled"]:
        return {"ok": False, "state": "off", "msg": "IBKR disabled (set IBKR_ENABLED + gateway URL)."}
    if not c["url"]:
        return {"ok": False, "state": "misconfigured", "msg": "IBKR_GATEWAY_URL not set."}
    cli = _client(cfg)
    if cli is None:
        return {"ok": False, "state": "no-client", "msg": "ibind not installed or client init failed."}
    try:
        # CP Web API: /iserver/auth/status reports session + authenticated flags.
        st = cli.check_health() if hasattr(cli, "check_health") else cli.authentication_status()
        ok = bool(getattr(st, "data", st))
        return {"ok": ok, "state": "authenticated" if ok else "not-authenticated",
                "msg": "Gateway reachable." if ok else "Gateway up but session not authenticated (re-login on the VPS)."}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "state": "unreachable", "msg": f"Gateway unreachable: {type(exc).__name__}"}


def available(cfg: Config) -> bool:
    return bool(diagnose(cfg).get("ok"))


# ---- data reads (all fail soft) -----------------------------------------------------------

def find_contract(cfg: Config, symbol: str, sec_type: str = "STK"):
    """Resolve a symbol to an IBKR conid for STK / FUT / CASH (FX) / IND. Returns dict or None."""
    cli = _client(cfg)
    if cli is None:
        return None
    try:
        res = cli.search_contract_by_symbol(symbol=symbol, sec_type=sec_type)
        data = getattr(res, "data", res)
        return (data[0] if isinstance(data, list) and data else data) or None
    except Exception:  # noqa: BLE001
        return None


def intraday_bars(cfg: Config, symbol: str, bar: str = "5min", lookback: str = "2d"):
    """Finer-grained price history than the daily bars. Returns a list of {t,o,h,l,c,v} or None."""
    cli = _client(cfg)
    if cli is None:
        return None
    try:
        con = find_contract(cfg, symbol)
        conid = con and (con.get("conid") or con.get("conidEx"))
        if not conid:
            return None
        res = cli.marketdata_history_by_conid(conid=conid, period=lookback, bar=bar, outside_rth=False)
        data = getattr(res, "data", res) or {}
        rows = data.get("data") if isinstance(data, dict) else data
        return [{"t": r.get("t"), "o": r.get("o"), "h": r.get("h"),
                 "l": r.get("l"), "c": r.get("c"), "v": r.get("v")} for r in (rows or [])] or None
    except Exception:  # noqa: BLE001
        return None


def option_summary(cfg: Config, symbol: str):
    """ATM implied vol + a light greeks snapshot from the chain. Returns dict or None.
    NB: the CP Web API options flow (secdef/strikes + snapshot fields) is multi-step — validate
    the exact field ids against your gateway before relying on this."""
    cli = _client(cfg)
    if cli is None:
        return None
    try:
        con = find_contract(cfg, symbol)
        conid = con and con.get("conid")
        if not conid:
            return None
        # 7283 = implied vol, 7308/7309 = delta/gamma (CP Web API md field ids — confirm live).
        res = cli.live_marketdata_snapshot(conids=str(conid), fields=["7283", "7308", "7309"])
        data = getattr(res, "data", res)
        row = (data[0] if isinstance(data, list) and data else data) or {}
        return {"symbol": symbol, "iv": row.get("7283"), "delta": row.get("7308"),
                "gamma": row.get("7309")} or None
    except Exception:  # noqa: BLE001
        return None


def positions(cfg: Config):
    """Your real account holdings, for portfolio-aware signals. Returns list of dicts or None."""
    cli = _client(cfg)
    if cli is None:
        return None
    c = _conf(cfg)
    try:
        acct = c["account"] or (getattr(cli, "account_id", None))
        res = cli.positions(account_id=acct) if acct else cli.positions()
        data = getattr(res, "data", res) or []
        out = []
        for p in data:
            out.append({"symbol": p.get("contractDesc") or p.get("ticker"),
                        "position": p.get("position"), "avg_cost": p.get("avgCost"),
                        "mkt_value": p.get("mktValue"), "sec_type": p.get("assetClass")})
        return out or None
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":  # quick manual check: python ibkr.py
    cfg = Config()
    print("enabled:", enabled(cfg))
    print("diagnose:", diagnose(cfg))
