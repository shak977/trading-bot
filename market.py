"""Low-level Alpaca Market Data REST helpers (screener + news).

Kept as thin REST calls (via requests) rather than SDK objects so they don't
break across alpaca-py versions. All require valid keys.

Endpoints (Alpaca Market Data v1beta1):
  GET /v1beta1/screener/stocks/most-actives
  GET /v1beta1/screener/stocks/movers
  GET /v1beta1/news
"""
from __future__ import annotations

import requests

from config import Config

_DATA_BASE = "https://data.alpaca.markets"
_asset_cache: dict[str, dict] = {}


def _trading_base(cfg: Config) -> str:
    return "https://paper-api.alpaca.markets" if cfg.paper else "https://api.alpaca.markets"


def get_asset(symbol: str, cfg: Config) -> dict:
    """Company name + exchange for a ticker. Cached per run; never raises."""
    if symbol in _asset_cache:
        return _asset_cache[symbol]
    out = {"name": "", "exchange": ""}
    try:
        r = requests.get(f"{_trading_base(cfg)}/v2/assets/{symbol}",
                         headers=_headers(cfg), timeout=12)
        if r.status_code == 200:
            d = r.json()
            out = {"name": d.get("name", "") or "", "exchange": d.get("exchange", "") or ""}
    except Exception:  # noqa: BLE001
        pass
    _asset_cache[symbol] = out
    return out


def _headers(cfg: Config) -> dict:
    cfg.validate_for_live()
    return {
        "APCA-API-KEY-ID": cfg.api_key,
        "APCA-API-SECRET-KEY": cfg.secret_key,
        "accept": "application/json",
    }


def _get(path: str, cfg: Config, params: dict | None = None, timeout: int = 15) -> dict:
    r = requests.get(f"{_DATA_BASE}{path}", headers=_headers(cfg),
                     params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def most_actives(cfg: Config) -> list[str]:
    data = _get("/v1beta1/screener/stocks/most-actives", cfg,
                {"by": "volume", "top": cfg.scan_top})
    return [x["symbol"] for x in data.get("most_actives", [])]


def movers(cfg: Config) -> list[str]:
    data = _get("/v1beta1/screener/stocks/movers", cfg, {"top": cfg.scan_top})
    syms = [x["symbol"] for x in data.get("gainers", [])]
    syms += [x["symbol"] for x in data.get("losers", [])]
    return syms


def get_news(symbols: list[str], cfg: Config, limit: int | None = None) -> list[dict]:
    if not symbols:
        return []
    data = _get("/v1beta1/news", cfg, {
        "symbols": ",".join(symbols),
        "limit": limit or (cfg.news_per_symbol * max(len(symbols), 1)),
        "sort": "desc",
    })
    out = []
    for n in data.get("news", []):
        out.append({
            "headline": n.get("headline", ""),
            "source": n.get("source", ""),
            "created_at": n.get("created_at", ""),
            "url": n.get("url", ""),
            "symbols": n.get("symbols", []),
        })
    return out
