"""Premium-selling ("Theta Harvest"-style) advisory scorer — ADVISORY ONLY, never executes a trade.

Scores WHERE selling options premium is favorable, 0-100, by fusing volatility signals, with hard
gates that veto dangerous setups. It does NOT predict direction and does NOT place orders — it
measures how favourable *selling* conditions are right now and says NO as readily as GO.

The edge it's chasing is the Volatility Risk Premium: implied vol systematically overstates realised
vol (people overpay for protection), so selling that overpriced insurance earns the decay — but only
when the structure supports it. Sell into backwardation / an event / a vol spike and one trade erases
months of income, so the gates matter more than the score.

Data: implied vol from Alpaca option snapshots (fail-soft); realised vol from price history. The core
scoring math is pure + unit-testable and has no network dependency. The whole equity engine trades
shares long; this is a SEPARATE advisory layer for premium-sellers — execution stays in your broker.
"""
from __future__ import annotations

import math
import statistics

# Signal weights (sum to 100), mirroring the Theta Harvest framework. Purely additive — the number
# reads as "how much selling edge is present right now?"
_WEIGHTS = {"vrp": 30, "iv_pct": 25, "term": 20, "stability": 15, "liquidity": 10}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def realized_vol(closes: list[float], window: int = 20) -> float | None:
    """Annualised close-to-close realised volatility as a decimal (0.24 = 24%). None if too little
    data. Uses the last `window` daily log returns."""
    if not closes or len(closes) < 6:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    rets = rets[-window:]
    if len(rets) < 5:
        return None
    return statistics.pstdev(rets) * math.sqrt(252)


def iv_percentile(iv: float, iv_history: list[float]) -> float | None:
    """Where the current IV sits in its own trailing history, 0-100. Needs a reasonable history
    (>=20 obs) to mean anything; None otherwise (it bootstraps as iv_history.json accrues)."""
    hist = [h for h in (iv_history or []) if isinstance(h, (int, float)) and h > 0]
    if iv is None or len(hist) < 20:
        return None
    below = sum(1 for h in hist if h <= iv)
    return round(100.0 * below / len(hist), 1)


def score(iv: float | None, realized: float | None, *, iv_pct: float | None = None,
          term_ratio: float | None = None, stability: float | None = None,
          liquidity: float | None = None, earnings_soon: bool = False,
          regime: str | None = None) -> dict | None:
    """Fuse the signals into a 0-100 premium-selling edge score + verdict, applying the hard gates.

    iv, realized : annualised decimals (0.30 = 30%).
    iv_pct       : IV percentile 0-100 (optional).
    term_ratio   : front-expiry IV / longer-expiry IV. <1 = contango (healthy), >1 = backwardation.
    stability    : 0-1, how rangey/stable the underlying is (1 = calm, ideal for selling).
    liquidity    : 0-1, options liquidity (1 = tight, deep).
    Returns {score, verdict, vrp, signals, gates, note} or None if IV/realised missing.
    """
    if iv is None or realized is None or iv <= 0:
        return None
    vrp = iv - realized                                   # positive = insurance is overpriced = good

    sig: dict[str, float] = {}
    # VRP quality — how much richer IV is than realised (the core edge). ~40% richness = full marks.
    sig["vrp"] = _clamp01((vrp / iv) / 0.40) * _WEIGHTS["vrp"]
    # IV percentile — expensive vs its own year.
    if iv_pct is not None:
        sig["iv_pct"] = (iv_pct / 100.0) * _WEIGHTS["iv_pct"]
    # Term structure — contango (front cheaper than back) supports harvesting; backwardation is bad.
    if term_ratio is not None:
        sig["term"] = _clamp01((1.10 - term_ratio) / 0.20) * _WEIGHTS["term"]
    # Trend stability — calm/rangey names decay cleanly; trending knives don't.
    if stability is not None:
        sig["stability"] = _clamp01(stability) * _WEIGHTS["stability"]
    # Liquidity — you must be able to get filled at a fair price.
    if liquidity is not None:
        sig["liquidity"] = _clamp01(liquidity) * _WEIGHTS["liquidity"]

    # Re-weight to 0-100 over the signals we actually have (missing data shouldn't tank the score).
    have_w = sum(_WEIGHTS[k] for k in sig)
    raw = (sum(sig.values()) / have_w * 100.0) if have_w else 0.0
    sc = round(raw, 1)

    # ---- hard gates: a high score is only half the system ----
    gates: list[str] = []
    verdict = "GO" if sc >= 65 else ("OK" if sc >= 50 else "SKIP")
    if term_ratio is not None and term_ratio > 1.05:
        verdict = "AVOID"
        gates.append("Deep backwardation — the market is pricing a real event; don't sell into it.")
    if vrp < 0:
        sc = min(sc, 54.0)
        gates.append("Negative VRP — realised vol is beating implied; there's no edge to sell.")
        if verdict in ("GO", "OK"):
            verdict = "SKIP"
    if earnings_soon:
        gates.append("Earnings inside the expiry — event risk; use defined-risk or wait.")
        if verdict == "GO":
            verdict = "OK"
    if regime == "OFF SEASON":
        verdict = "AVOID"
        gates.append("Market regime OFF SEASON — go to cash.")

    return {
        "score": sc,
        "verdict": verdict,
        "vrp": round(vrp, 4),
        "iv": round(iv, 4),
        "realized": round(realized, 4),
        "iv_pct": iv_pct,
        "term_ratio": round(term_ratio, 3) if term_ratio is not None else None,
        "signals": {k: round(v, 1) for k, v in sig.items()},
        "gates": gates,
    }


def market_regime(vix: float | None, vix_far: float | None = None, breadth: float | None = None) -> str:
    """Whole-table posture (NBA-themed, per Theta Harvest). Uses VIX level + term structure if given,
    else falls back to breadth. THE FINALS (widest edge) · THE PLAYOFFS · REGULAR SEASON · OFF SEASON."""
    # Backwardation in the vol curve itself = danger → cash.
    if vix is not None and vix_far is not None and vix_far > 0 and vix / vix_far > 1.05:
        return "OFF SEASON"
    if vix is not None:
        if vix >= 30:
            return "OFF SEASON"          # crisis vol — realised can outrun implied fast
        if vix >= 22:
            return "REGULAR SEASON"      # elevated — defined-risk only
        if vix >= 15:
            return "THE PLAYOFFS"        # normal harvesting
        return "THE FINALS"              # calm, rich-enough premium — widest edge
    if breadth is not None:
        return "THE PLAYOFFS" if breadth >= 45 else "REGULAR SEASON"
    return "THE PLAYOFFS"


def atm_iv(cfg, symbol: str, spot: float | None = None, dte_target: int = 35) -> dict | None:
    """ATM implied vol near ~`dte_target` DTE + a longer expiry for term structure, from Alpaca option
    snapshots. Returns {iv, iv_far, term_ratio, expiry, dte} or None. Fail-soft; never raises."""
    try:
        from datetime import date, timedelta
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest
        key = getattr(cfg, "api_key", "") or ""
        sec = getattr(cfg, "secret_key", "") or ""
        if not key or not sec:
            return None
        cli = OptionHistoricalDataClient(key, sec)

        def _chain_iv(days: int):
            exp_gte = (date.today() + timedelta(days=max(1, days - 12))).isoformat()
            exp_lte = (date.today() + timedelta(days=days + 20)).isoformat()
            req = OptionChainRequest(underlying_symbol=symbol, expiration_date_gte=exp_gte,
                                     expiration_date_lte=exp_lte)
            snaps = cli.get_option_chain(req) or {}
            best = None  # (distance_to_spot, iv)
            for osym, snap in snaps.items():
                iv = getattr(snap, "implied_volatility", None)
                if iv is None:
                    continue
                strike = _strike_from_osym(osym)
                if strike is None:
                    continue
                dist = abs(strike - spot) if spot else 0
                if best is None or dist < best[0]:
                    best = (dist, float(iv))
            return best[1] if best else None

        iv_near = _chain_iv(dte_target)
        iv_far = _chain_iv(dte_target * 2)
        if iv_near is None:
            return None
        term_ratio = (iv_near / iv_far) if (iv_far and iv_far > 0) else None
        return {"iv": iv_near, "iv_far": iv_far, "term_ratio": term_ratio, "dte": dte_target}
    except Exception:  # noqa: BLE001
        return None


def _strike_from_osym(osym: str) -> float | None:
    """OCC option symbol: ROOT + YYMMDD + C/P + strike*1000 (8 digits). e.g. AAPL240119C00190000."""
    try:
        return int(osym[-8:]) / 1000.0
    except Exception:  # noqa: BLE001
        return None


# A curated, liquid, optionable universe (like Theta Harvest's 33) — deep, tight-spread names where
# selling premium is actually practical. Kept small so the per-build options fetch stays bounded.
PREMIUM_UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD",
    "TSLA", "JPM", "XLE", "GLD", "NFLX", "BAC", "DIS", "COIN", "PLTR", "SMCI",
]
_IV_HISTORY_FILE = "iv_history.json"


def _load_iv_history() -> dict:
    try:
        import json
        with open(_IV_HISTORY_FILE) as f:
            return json.load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def _save_iv_history(hist: dict) -> None:
    try:
        import json
        with open(_IV_HISTORY_FILE, "w") as f:
            json.dump(hist, f)
    except Exception:  # noqa: BLE001
        pass


def run(cfg, universe: list | None = None, breadth: float | None = None, max_names: int = 12) -> dict:
    """Score the premium-selling edge across the curated universe. Returns
    {regime, n, names:[scored...]} sorted best-first. Fail-soft per name; the IV-dependent parts need
    live Alpaca options data, so names without IV are skipped and IV percentile bootstraps over time
    as iv_history.json accrues. ADVISORY ONLY — never places a trade."""
    names = (universe or PREMIUM_UNIVERSE)[:max_names]
    regime = market_regime(None, breadth=breadth)
    iv_hist = _load_iv_history()
    out: list[dict] = []
    try:
        import data as _data
    except Exception:  # noqa: BLE001
        _data = None
    from datetime import date
    for sym in names:
        closes = []
        if _data is not None:
            try:
                df = _data.get_bars(sym, cfg)
                if df is not None and len(df):
                    closes = [float(x) for x in df["close"].tolist()]
            except Exception:  # noqa: BLE001
                closes = []
        rv = realized_vol(closes)
        spot = closes[-1] if closes else None
        ivd = atm_iv(cfg, sym, spot=spot)
        if not ivd or rv is None:
            continue
        iv = ivd["iv"]
        hist = iv_hist.get(sym, [])
        ivp = iv_percentile(iv, hist)
        sc = score(iv, rv, iv_pct=ivp, term_ratio=ivd.get("term_ratio"), regime=regime)
        if not sc:
            continue
        sc["symbol"] = sym
        sc["spot"] = round(spot, 2) if spot else None
        out.append(sc)
        # append today's IV to the rolling history (bootstraps the percentile over ~1yr)
        hist.append(round(iv, 4))
        iv_hist[sym] = hist[-260:]
    _save_iv_history(iv_hist)
    out.sort(key=lambda r: -r["score"])
    return {"regime": regime, "n": len(out), "names": out}
