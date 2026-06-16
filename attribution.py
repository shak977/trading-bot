"""Factor attribution — which conviction checks actually predict wins?

Reads the resolved trades in track_record.json (each carries a snapshot of its
conviction checks) and, per check label, compares the win rate when that check
PASSED vs when it FAILED. The gap ("edge") tells you which checks earn their
weight and which are decoration. This is the loop that keeps a widening funnel
honest. Read-only; accumulates from the first run after the snapshot ships.
"""
from __future__ import annotations

import json
import os

PATH = os.getenv("TRACK_FILE", "track_record.json")
INTRADAY_PATH = os.getenv("TRACK_INTRADAY_FILE", "track_record_intraday.json")

_RESOLVED = ("win", "loss", "expired")


def _win_rate(items: list[dict]) -> float | None:
    decided = [t for t in items if t["status"] in ("win", "loss")]
    if not decided:
        return None
    return round(sum(1 for t in decided if t["status"] == "win") / len(decided) * 100, 1)


def attribute(log: list[dict]) -> list[dict]:
    """Per-check win-rate-when-pass vs win-rate-when-fail across resolved trades.

    Returns a list of {label, n_pass, n_fail, win_rate_pass, win_rate_fail, edge},
    sorted by edge (pass minus fail win rate) descending. `edge` is None when either
    side has no decided trades.
    """
    resolved = [t for t in log if t.get("status") in _RESOLVED and t.get("checks")]
    labels = []
    seen = set()
    for t in resolved:
        for c in t["checks"]:
            lbl = c.get("label")
            if lbl and lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)
    out = []
    for lbl in labels:
        passed = [t for t in resolved if any(c.get("label") == lbl and c.get("status") == "pass" for c in t["checks"])]
        failed = [t for t in resolved if any(c.get("label") == lbl and c.get("status") == "fail" for c in t["checks"])]
        wp, wf = _win_rate(passed), _win_rate(failed)
        edge = round(wp - wf, 1) if (wp is not None and wf is not None) else None
        out.append({"label": lbl, "n_pass": len(passed), "n_fail": len(failed),
                    "win_rate_pass": wp, "win_rate_fail": wf, "edge": edge})
    out.sort(key=lambda r: -(r["edge"] if r["edge"] is not None else -999))
    return out


def load(path: str | None = None) -> list[dict]:
    """Read the track-record log (best-effort; [] on any error)."""
    try:
        with open(path or PATH) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def load_all() -> list[dict]:
    """Daily + intraday resolved trades, pooled. Intraday resolves in hours, so it feeds the
    learning loop far more outcome volume — and the conviction checks are the same engine for both
    timeframes, so a check that predicts winners should do so across both. Each is tagged with `tf`
    so callers can split later if needed."""
    out = []
    for t in load(PATH):
        t.setdefault("tf", "daily")
        out.append(t)
    for t in load(INTRADAY_PATH):
        t = dict(t)
        t["tf"] = "intraday"
        out.append(t)
    return out


def report(path: str | None = None) -> list[dict]:
    return attribute(load(path) if path else load_all())


def learned_weights(path: str | None = None, min_n: int = 12, max_adj: float = 0.5) -> dict:
    """Turn the per-check edge into a {label: weight-multiplier} the conviction engine can use to
    ADAPT — checks that have historically predicted winners get up-weighted, those that haven't get
    down-weighted. This is the bot learning from its own resolved wins vs losses.

    Strictly gated: a check only earns an adjustment once it has at least `min_n` *decided* trades on
    BOTH the pass and fail sides (so the edge is real, not noise). Multiplier is bounded to
    1 ± max_adj so no single check can run away. Returns {} until there's enough data — so early on
    the engine runs on its transparent default weights, exactly as it does today."""
    out = {}
    for r in attribute(load(path) if path else load_all()):
        edge = r.get("edge")
        n = min(r.get("n_pass", 0), r.get("n_fail", 0))
        if edge is None or n < min_n:
            continue
        mult = 1.0 + max(-max_adj, min(max_adj, edge / 100.0))
        out[r["label"]] = round(mult, 2)
    return out
