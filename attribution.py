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


def report(path: str | None = None) -> list[dict]:
    return attribute(load(path))
