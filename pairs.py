"""Pairs / mean-reversion diversifier — the brief's sideways-market complement.

The core book trades *direction* (trend + momentum). This module trades the *spread* between two
economically-related liquid names — a market-neutral bet that a temporarily stretched relationship
snaps back. It is deliberately a diversifier, not the core: it tends to earn when trend signals are
weak and the tape is choppy, which is exactly when the momentum book struggles.

Method (dependency-light, numpy only — no statsmodels needed):
  1. Hedge ratio β via least squares (regress A's price on B's price).
  2. Spread = A − β·B − α. Z-score the spread over a lookback window.
  3. Validate the relationship is genuinely mean-reverting:
       • legs' daily-return correlation ≥ pairs_min_corr (economically linked), and
       • the spread's AR(1) half-life sits in a sane band (reverts, but not just noise).
  4. Signal off the z-score: enter at |z| ≥ entry_z, exit toward 0, stop if |z| blows past stop_z.

Every pair is validated independently and the whole thing is wrapped so a data hiccup just yields
fewer (or zero) pairs — it can never break the daily build. Gated by cfg.pairs_enabled.
"""
from __future__ import annotations

import math

# Economically-related, highly-liquid candidate pairs (same sector / business model).
# A pair only ever trades if it ALSO passes the live correlation + mean-reversion validation.
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),    ("GS", "MS"),     ("V", "MA"),      ("HD", "LOW"),
    ("XOM", "CVX"),   ("JPM", "BAC"),   ("CAT", "DE"),    ("UPS", "FDX"),
    ("AMD", "NVDA"),  ("GOOGL", "META"),("LIN", "APD"),   ("WMT", "TGT"),
    ("LLY", "ABBV"),  ("MAR", "HLT"),   ("ADBE", "CRM"),  ("SBUX", "MCD"),
    ("C", "WFC"),     ("TGT", "COST"),  ("PANW", "CRWD"), ("QCOM", "AVGO"),
]


def _lstsq_beta(y, x):
    """OLS hedge ratio: y ≈ beta·x + alpha. Returns (beta, alpha)."""
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones(len(x))]).T
    beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(beta), float(alpha)


def _half_life(spread) -> float | None:
    """AR(1) half-life of mean reversion (in bars). None if the spread isn't reverting."""
    import numpy as np
    s = np.asarray(spread, float)
    if len(s) < 20:
        return None
    lag = s[:-1]
    delta = s[1:] - lag
    # delta = lambda*lag + c ; mean-reverting when lambda < 0
    A = np.vstack([lag, np.ones(len(lag))]).T
    lam = np.linalg.lstsq(A, delta, rcond=None)[0][0]
    if lam >= 0 or not math.isfinite(lam):
        return None
    return float(-math.log(2) / lam)


def _evaluate_pair(a: str, b: str, da, db, cfg) -> dict | None:
    """Validate + score one pair. Returns a pair dict or None if it doesn't qualify."""
    import numpy as np
    look = int(getattr(cfg, "pairs_lookback", 90))
    pa = da["close"].astype(float).tail(look)
    pb = db["close"].astype(float).tail(look)
    n = min(len(pa), len(pb))
    if n < max(40, look // 2):
        return None
    pa = pa.tail(n).reset_index(drop=True)
    pb = pb.tail(n).reset_index(drop=True)

    # return correlation — are the legs actually related?
    ra = pa.pct_change().dropna()
    rb = pb.pct_change().dropna()
    m = min(len(ra), len(rb))
    if m < 20:
        return None
    corr = float(np.corrcoef(ra.tail(m), rb.tail(m))[0, 1])
    if not math.isfinite(corr) or corr < float(getattr(cfg, "pairs_min_corr", 0.5)):
        return None

    beta, alpha = _lstsq_beta(pa.values, pb.values)
    if beta <= 0 or not math.isfinite(beta):
        return None  # inverse relationships aren't the clean co-movement we want here
    spread = pa.values - beta * pb.values - alpha
    mu, sd = float(np.mean(spread)), float(np.std(spread))
    if sd <= 0 or not math.isfinite(sd):
        return None
    z = float((spread[-1] - mu) / sd)

    hl = _half_life(spread)
    if hl is None:
        return None
    lo, hi = float(getattr(cfg, "pairs_min_halflife", 2)), float(getattr(cfg, "pairs_max_halflife", 40))
    if not (lo <= hl <= hi):
        return None

    entry_z = float(getattr(cfg, "pairs_entry_z", 2.0))
    exit_z = float(getattr(cfg, "pairs_exit_z", 0.5))
    stop_z = float(getattr(cfg, "pairs_stop_z", 3.0))

    # z > 0 => A rich vs B (short A / long B = "short the spread"); z < 0 => A cheap (long the spread)
    if z >= stop_z or z <= -stop_z:
        signal, side_txt = "STOP", "Spread stretched past the stop band — relationship may have broken; stand aside."
    elif z >= entry_z:
        signal, side_txt = "SHORT_SPREAD", f"Short {a} / long {b} (β {beta:.2f}) — {a} looks rich vs {b}; bet on convergence."
    elif z <= -entry_z:
        signal, side_txt = "LONG_SPREAD", f"Long {a} / short {b} (β {beta:.2f}) — {a} looks cheap vs {b}; bet on convergence."
    elif abs(z) <= exit_z:
        signal, side_txt = "FLAT", "Spread near fair value — no edge; this is where an open pair would be closed."
    else:
        signal, side_txt = "WATCH", f"Spread at {z:+.1f}σ — watching for a stretch to ±{entry_z:.0f}σ to act."

    return {
        "a": a, "b": b, "beta": round(beta, 3), "z": round(z, 2),
        "corr": round(corr, 2), "half_life": round(hl, 1),
        "signal": signal, "note": side_txt,
        "entry_z": entry_z, "exit_z": exit_z, "stop_z": stop_z,
        "price_a": round(float(pa.values[-1]), 2), "price_b": round(float(pb.values[-1]), 2),
        "actionable": signal in ("LONG_SPREAD", "SHORT_SPREAD"),
    }


def scan(cfg, live: bool, regime: dict | None = None, bars_fn=None,
         macro_posture: dict | None = None) -> dict:
    """Validate + score every candidate pair. Returns:
        {"pairs": [...sorted, most-actionable first...], "regime_fit": bool, "note": str}
    Never raises; returns an empty list on any failure or when disabled."""
    if not getattr(cfg, "pairs_enabled", False):
        return {"pairs": [], "regime_fit": False, "note": "Pairs module disabled."}
    try:
        if bars_fn is None:
            if live:
                from data import get_bars
                bars_fn = lambda s: get_bars(s, cfg)  # noqa: E731 - get_bars needs (symbol, cfg)
            else:
                from data import synthetic_bars
                bars_fn = lambda s: synthetic_bars(s, n=getattr(cfg, "lookback_days", 400))  # noqa: E731

        # pull bars once per unique symbol
        syms = sorted({s for pr in CANDIDATE_PAIRS for s in pr})
        cache: dict = {}
        for s in syms:
            try:
                df = bars_fn(s)
                if df is not None and len(df) >= 40:
                    cache[s] = df
            except Exception:  # noqa: BLE001 - skip a bad symbol, keep going
                continue

        out: list[dict] = []
        for a, b in CANDIDATE_PAIRS:
            if a not in cache or b not in cache:
                continue
            try:
                res = _evaluate_pair(a, b, cache[a], cache[b], cfg)
            except Exception:  # noqa: BLE001
                res = None
            if res:
                out.append(res)

        # most actionable first: actionable + biggest |z|, then watchers
        out.sort(key=lambda p: (p["actionable"], abs(p["z"])), reverse=True)

        # regime fit: pairs/mean-reversion shines when the tape is trendless (Neutral / weak breadth)
        # OR when the macro regime is High-volatility (mean-reversion thrives in choppy, fast tape).
        regime_fit = True
        rnote = "Mean-reversion suits the current trendless tape — a useful complement to the momentum book."
        high_vol = any(t.get("tag") == "High-volatility" for t in (macro_posture or {}).get("tags", []))
        if high_vol:
            regime_fit = True
            rnote = ("Macro regime is High-volatility — mean-reversion is favoured here; pairs move up the "
                     "priority list while the momentum book sizes down.")
        elif regime:
            lbl = regime.get("label")
            if lbl in ("Risk-on", "Risk-off"):
                regime_fit = False
                rnote = (f"Tape is strongly {lbl} — pairs are a lower-priority diversifier right now; "
                         "the directional book is in its element.")
        return {"pairs": out, "regime_fit": regime_fit, "note": rnote}
    except Exception as e:  # noqa: BLE001 - never break the build
        return {"pairs": [], "regime_fit": False, "note": f"Pairs scan skipped: {str(e)[:80]}"}
