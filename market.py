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
