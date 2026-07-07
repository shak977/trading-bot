"""Self-validation for the market-timing engine (timing.py).

Does the O'Neil FTD / distribution posture actually predict anything? This harness walks real
index history (SPY + NASDAQ), labels every day with the timing state that would have been shown
that day (expanding window — strictly no look-ahead), then measures the *forward* index return
that followed. If the engine has edge, the ordering should be:

    correction / pressure  →  weak (ideally negative) forward returns
    neutral                →  ~baseline
    confirmed (FTD)        →  strong forward returns

and, crucially, gating longs off in `correction` should lift the average of the days you stay
invested vs being always-in. The result is written to `timing_study.json` and a one-line edge
summary is surfaced on the dashboard's timing panel — so the block-longs-in-a-correction rule is
backed by numbers, and re-checked on every post-close CI run (same spirit as attribution.py).

Pure/offline core (`walk`) so it unit-tests on synthetic data; `study(cfg)` does the fetching.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import timing

STUDY_PATH = "timing_study.json"
HORIZONS = (5, 10, 20)
WARMUP = 70          # need enough history for the FTD swing/rally scan before labelling a day


def walk(df, horizons=(5, 10, 20), warmup: int = WARMUP) -> dict:
    """Expanding-window walk over one index. For each day i>=warmup, label the state from the bars
    up to and including i (no look-ahead), then record the forward h-day return. Returns per-state
    aggregates plus an always-in vs gated comparison. Pure; safe on any OHLCV DataFrame."""
    c = df["close"].to_numpy(float)
    n = len(c)
    hmax = max(horizons)
    by_state = defaultdict(lambda: {h: [] for h in horizons})
    seq = []                                            # (state, fwd_by_h) per labelled day
    for i in range(warmup, n - hmax):
        try:
            st = timing.state_at(df.iloc[:i + 1])
        except Exception:  # noqa: BLE001 - a bad slice must never abort the whole walk
            continue
        fwd = {}
        for h in horizons:
            if c[i]:
                r = c[i + h] / c[i] - 1.0
                by_state[st][h].append(r)
                fwd[h] = r
        seq.append((st, fwd))

    def agg(vals):
        if not vals:
            return {"n": 0, "mean_pct": None, "hit_rate": None}
        a = np.array(vals, float)
        return {"n": len(a), "mean_pct": round(float(a.mean()) * 100, 3),
                "hit_rate": round(float((a > 0).mean()) * 100, 1)}

    states = {st: {str(h): agg(by_state[st][h]) for h in horizons} for st in by_state}

    # Headline: on a chosen horizon, does skipping longs in 'correction' beat always-in?
    H = horizons[1] if len(horizons) > 1 else horizons[0]
    all_in = [f[H] for _, f in seq if H in f]
    gated = [f[H] for st, f in seq if H in f and st not in ("correction", "pressure")]
    headline = {
        "horizon": H,
        "always_in_mean_pct": round(float(np.mean(all_in)) * 100, 3) if all_in else None,
        "gated_mean_pct": round(float(np.mean(gated)) * 100, 3) if gated else None,
        "days_gated_out": len(all_in) - len(gated),
        "days_total": len(all_in),
    }
    return {"states": states, "headline": headline, "labelled_days": len(seq)}


def _merge(a: dict, b: dict, horizons) -> dict:
    """Combine two per-index walks by pooling raw counts weighted by n (means recombined)."""
    out_states = {}
    for st in set(a["states"]) | set(b["states"]):
        out_states[st] = {}
        for h in (str(x) for x in horizons):
            xa = a["states"].get(st, {}).get(h, {"n": 0, "mean_pct": None, "hit_rate": None})
            xb = b["states"].get(st, {}).get(h, {"n": 0, "mean_pct": None, "hit_rate": None})
            na, nb = xa["n"], xb["n"]
            if na + nb == 0:
                out_states[st][h] = {"n": 0, "mean_pct": None, "hit_rate": None}
                continue
            mean = ((xa["mean_pct"] or 0) * na + (xb["mean_pct"] or 0) * nb) / (na + nb)
            hit = ((xa["hit_rate"] or 0) * na + (xb["hit_rate"] or 0) * nb) / (na + nb)
            out_states[st][h] = {"n": na + nb, "mean_pct": round(mean, 3), "hit_rate": round(hit, 1)}
    return {"states": out_states}


def study(cfg, indexes=(("S&P 500", "SPY"), ("NASDAQ", "QQQ")), horizons=HORIZONS,
          path: str = STUDY_PATH) -> dict | None:
    """Fetch real index history and run the walk on each, pooling the result. Writes timing_study.json.
    Returns the study dict, or None if no index history could be fetched. Never raises."""
    try:
        import dataclasses
        from data import get_bars
        merged = None
        per_index = {}
        for name, sym in indexes:
            try:
                dcfg = dataclasses.replace(cfg, timeframe="1Day", lookback_days=760)
                df = get_bars(sym, dcfg)
            except Exception:  # noqa: BLE001
                df = None
            if df is None or len(df) < WARMUP + max(horizons) + 20:
                continue
            w = walk(df, horizons=horizons)
            per_index[name] = w["headline"]
            merged = w if merged is None else _merge(merged, w, horizons)
        if merged is None:
            return None
        # a friendly one-line verdict for the dashboard
        conf = merged["states"].get("confirmed", {}).get(str(horizons[1]), {})
        corr = merged["states"].get("correction", {}).get(str(horizons[1]), {})
        verdict = _verdict(conf, corr, horizons[1])
        out = {"states": merged["states"], "per_index": per_index, "verdict": verdict,
               "horizons": list(horizons), "generated_at": _now()}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return out
    except Exception:  # noqa: BLE001 - validation is advisory; never break the pipeline
        return None


def _verdict(conf: dict, corr: dict, h: int) -> str:
    cm = conf.get("mean_pct")
    rm = corr.get("mean_pct")
    if cm is None and rm is None:
        return "Not enough history yet to validate the timing signal."
    bits = []
    if rm is not None:
        bits.append(f"after a correction signal the index averaged {rm:+.2f}% over the next {h} days "
                    f"({corr.get('n', 0)} samples)")
    if cm is not None:
        bits.append(f"after a Follow-Through Day it averaged {cm:+.2f}% ({conf.get('n', 0)} samples)")
    edge = (cm is not None and rm is not None and cm > rm)
    lead = "Timing signal has separated outcomes: " if edge else "Timing signal (validating): "
    return lead + "; ".join(bits) + "."


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


if __name__ == "__main__":
    from config import Config
    r = study(Config())
    print(json.dumps(r, indent=2) if r else "no index history available")
