"""Live P(win) scorer — loads the walk-forward meta-label model (meta_model.json, produced nightly by
meta_label.py) and scores a live signal row's probability of winning, using the EXACT same entry-time
features + encodings the model was trained on. Long-only (the model is trained on resolved longs).

Fail-safe by design: if the model file is missing/unusable, p_win() returns None and callers leave the
engine untouched. Mirrors meta_label._vectorize so live scoring == training features. Pure stdlib.
"""
from __future__ import annotations

import json
import math

# Must match meta_label.py exactly.
_CHECK_ENC = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
_LIQ = {"illiquid": 0, "thin": 1, "moderate": 2, "high": 3, "very high": 4}
_TVA = {"agree": 1.0, "mixed": 0.0, "oppose": -1.0}
_CONV = {"High": 2.0, "Medium": 1.0, "Low": 0.0}

_model = None
_loaded = False


def load_model(path: str = "meta_model.json"):
    """Load + cache the model once. Returns the dict or None. Safe to call repeatedly."""
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    try:
        with open(path) as f:
            raw = json.load(f)
        # meta_label.py writes the trained model nested under "model"; accept a flat shape too.
        m = raw.get("model") if isinstance(raw.get("model"), dict) else raw
        if m.get("feature_names") and m.get("weight") and m.get("mean") and m.get("std"):
            _model = m
    except Exception:  # noqa: BLE001
        _model = None
    return _model


def _tv_align(row) -> str | None:
    """Derive TradingView alignment exactly as tracker.py records it (from the raw tv object)."""
    if row.get("tv_align") is not None:
        return row.get("tv_align")
    if row.get("tv"):
        try:
            import tradingview as _tv
            return _tv.alignment(row["tv"], row.get("direction", "LONG"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _features(row, names):
    """Build the feature vector in the model's feature order from a live signal row (mirrors
    meta_label._vectorize: leading block = sorted check labels; trailing = rr, liq, tv, conv tier)."""
    labs = names[:-4]
    cks = {c.get("label"): c.get("status")
           for c in ((row.get("conviction") or {}).get("checks") or [])}
    x = [_CHECK_ENC.get(cks.get(lab), 0.5) for lab in labs]
    rr = (row.get("plan") or {}).get("rr")
    x.append(min(float(rr or 0), 6.0))
    x.append(_LIQ.get((row.get("liquidity") or {}).get("tier"), 2))
    x.append(_TVA.get(_tv_align(row), 0.0))
    x.append(_CONV.get((row.get("conviction") or {}).get("label"), 1.0))
    return x


def p_win(row, path: str = "meta_model.json"):
    """Probability this LONG wins, per the meta-label model. Returns a float in (0,1) or None when
    there's no model / the row can't be scored. Only meaningful for LONG rows (model is long-only)."""
    m = load_model(path)
    if not m:
        return None
    if str(row.get("direction", "LONG")).upper() != "LONG":
        return None
    try:
        names, mean, std, w, b = (m["feature_names"], m["mean"], m["std"], m["weight"], m["bias"])
        x = _features(row, names)
        z = float(b)
        for i in range(len(w)):
            z += ((x[i] - mean[i]) / (std[i] or 1e-9)) * w[i]
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0
    except Exception:  # noqa: BLE001
        return None


def size_mult(pw, cfg=None) -> float:
    """Guardrailed position-size multiplier from P(win): 1.0 at p=0.5, scaled by k, clamped. Returns
    1.0 when pw is None so the engine is unchanged without a model."""
    if pw is None:
        return 1.0
    k = float(getattr(cfg, "meta_size_k", 2.0))
    lo = float(getattr(cfg, "meta_size_min", 0.6))
    hi = float(getattr(cfg, "meta_size_max", 1.5))
    return max(lo, min(hi, 1.0 + (pw - 0.5) * k))
