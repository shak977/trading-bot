"""Self-validation for the ported entry setups (screens.py: Momentum Burst + Episodic Pivot).

Same idea as timing_backtest, applied to single-name setups. Walk a basket of liquid names day by
day (expanding window, no look-ahead); whenever a setup fires, record the forward 5/10/20-day
return. Compare each setup's average forward return + hit rate against the *baseline* (every day,
every name) so we can see the genuine edge-over-baseline — not just that stocks drift up.

  edge = setup_mean_forward_return − baseline_mean_forward_return

A positive, well-sampled edge means the setup actually selects better-than-average forward windows.
The result is written to `setups_study.json` and surfaced on the dashboard, and re-checked each
post-close CI run — so every ported edge reports its own hit rate, in the attribution spirit.

Note: the historical Episodic Pivot is evaluated in TECHNICAL mode only (has_news=False) — the
catalyst/news component can't be reconstructed from bars alone, so this validates the gap + volume
+ neglect skeleton, not the headline scoring. Momentum Burst is validated in full (pure price/vol).

Pure/offline core (`walk_symbol`) so it unit-tests on synthetic data; `study(cfg)` does fetching.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import screens

STUDY_PATH = "setups_study.json"
HORIZONS = (5, 10, 20)
WARMUP = 30          # setups need a prior base; skip the first ~30 bars of any series

# A curated, liquid, deep-history basket — large enough to be meaningful, small enough to be fast.
DEFAULT_BASKET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "NFLX",
    "JPM", "BAC", "GS", "V", "MA", "XOM", "CVX", "UNH", "LLY", "MRK",
    "HD", "COST", "WMT", "NKE", "DIS", "CRM", "ORCL", "ADBE", "INTC", "QCOM",
    "CAT", "BA", "GE", "UBER", "PYPL", "SBUX", "PFE", "T", "CSCO", "MU",
]


def walk_symbol(df, horizons=(5, 10, 20), warmup: int = WARMUP) -> dict:
    """Expanding-window walk over one name. For every day, record forward returns for the baseline,
    and — when a setup fires on the bars up to that day — for that setup too. Returns raw pooled
    lists per bucket. Pure; safe on any OHLCV DataFrame."""
    c = df["close"].to_numpy(float)
    n = len(c)
    hmax = max(horizons)
    buckets = {"baseline": {h: [] for h in horizons},
               "burst": {h: [] for h in horizons},
               "ep": {h: [] for h in horizons}}
    for i in range(warmup, n - hmax):
        if not c[i]:
            continue
        fwd = {h: c[i + h] / c[i] - 1.0 for h in horizons}
        for h in horizons:
            buckets["baseline"][h].append(fwd[h])
        window = df.iloc[:i + 1]
        try:
            if screens.momentum_burst(window).get("valid"):
                for h in horizons:
                    buckets["burst"][h].append(fwd[h])
        except Exception:  # noqa: BLE001
            pass
        try:
            if screens.episodic_pivot(window, has_news=False).get("valid"):
                for h in horizons:
                    buckets["ep"][h].append(fwd[h])
        except Exception:  # noqa: BLE001
            pass
    return buckets


def _pool(a: dict, b: dict, horizons) -> dict:
    """Concatenate two per-symbol bucket dicts."""
    out = {k: {h: list(a.get(k, {}).get(h, [])) + list(b.get(k, {}).get(h, [])) for h in horizons}
           for k in ("baseline", "burst", "ep")}
    return out


def _agg(vals):
    if not vals:
        return {"n": 0, "mean_pct": None, "hit_rate": None}
    x = np.array(vals, float)
    return {"n": len(x), "mean_pct": round(float(x.mean()) * 100, 3),
            "hit_rate": round(float((x > 0).mean()) * 100, 1)}


def summarize(buckets, horizons=HORIZONS) -> dict:
    """Turn pooled raw returns into per-setup stats + edge-over-baseline at the middle horizon."""
    H = horizons[1] if len(horizons) > 1 else horizons[0]
    base = {str(h): _agg(buckets["baseline"][h]) for h in horizons}
    out = {"baseline": base, "horizons": list(horizons), "primary_horizon": H, "setups": {}}
    for key, label in (("burst", "Momentum Burst"), ("ep", "Episodic Pivot (technical)")):
        stats = {str(h): _agg(buckets[key][h]) for h in horizons}
        setu = stats[str(H)].get("mean_pct")
        basu = base[str(H)].get("mean_pct")
        edge = round(setu - basu, 3) if (setu is not None and basu is not None) else None
        out["setups"][key] = {"label": label, "stats": stats, "edge_pct": edge,
                              "n": stats[str(H)].get("n", 0)}
    out["verdict"] = _verdict(out, H)
    return out


def _verdict(summary: dict, H: int) -> str:
    parts = []
    for key in ("burst", "ep"):
        s = summary["setups"].get(key, {})
        edge = s.get("edge_pct")
        n = s.get("n", 0)
        if edge is None or n < 20:
            parts.append(f"{s.get('label', key)}: still gathering samples (n={n})")
        else:
            verb = "beats" if edge > 0 else "lags"
            parts.append(f"{s.get('label', key)} {verb} baseline by {edge:+.2f}% over {H} days "
                         f"(hit {s['stats'][str(H)]['hit_rate']}%, n={n})")
    return " · ".join(parts) + "."


def study(cfg, symbols=None, horizons=HORIZONS, path: str = STUDY_PATH) -> dict | None:
    """Fetch real history for the basket and pool the walk across names. Writes setups_study.json.
    Returns the summary, or None if nothing could be fetched. Never raises."""
    try:
        import dataclasses
        from data import get_bars
        symbols = symbols or DEFAULT_BASKET
        pooled = {"baseline": {h: [] for h in horizons},
                  "burst": {h: [] for h in horizons},
                  "ep": {h: [] for h in horizons}}
        used = 0
        for sym in symbols:
            try:
                dcfg = dataclasses.replace(cfg, timeframe="1Day", lookback_days=760)
                df = get_bars(sym, dcfg)
            except Exception:  # noqa: BLE001
                df = None
            if df is None or len(df) < WARMUP + max(horizons) + 20:
                continue
            pooled = _pool(pooled, walk_symbol(df, horizons=horizons), horizons)
            used += 1
        if used == 0:
            return None
        out = summarize(pooled, horizons=horizons)
        out["names"] = used
        out["generated_at"] = _now()
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return out
    except Exception:  # noqa: BLE001 - advisory; never break the pipeline
        return None


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


if __name__ == "__main__":
    from config import Config
    r = study(Config())
    print(json.dumps(r, indent=2) if r else "no basket history available")
