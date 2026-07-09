"""Factor attribution — which conviction checks actually predict wins?

Reads the resolved trades in track_record.json (each carries a snapshot of its
conviction checks) and, per check label, compares the win rate when that check
PASSED vs when it FAILED. The gap ("edge") tells you which checks earn their
weight and which are decoration. This is the loop that keeps a widening funnel
honest. Read-only; accumulates from the first run after the snapshot ships.
"""
from __future__ import annotations

import json
import math
import os


def _two_proportion_p(w1: int, n1: int, w2: int, n2: int) -> float | None:
    """Two-sided p-value for 'do these two win rates differ?' (pooled two-proportion z-test).
    Used to tell a real edge from a lucky one. Pure stdlib (no scipy). None if a side is empty."""
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))          # two-sided p-value

PATH = os.getenv("TRACK_FILE", "track_record.json")
INTRADAY_PATH = os.getenv("TRACK_INTRADAY_FILE", "track_record_intraday.json")
ORB_PATH = os.getenv("TRACK_ORB_FILE", "track_record_orb.json")

_RESOLVED = ("win", "loss", "expired", "eod")   # "eod" = ORB end-of-day flatten (a resolved timeout)


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
        # decided-only counts (win/loss) for the significance test
        dp = [t for t in passed if t["status"] in ("win", "loss")]
        dfl = [t for t in failed if t["status"] in ("win", "loss")]
        p_value = _two_proportion_p(sum(t["status"] == "win" for t in dp), len(dp),
                                    sum(t["status"] == "win" for t in dfl), len(dfl))
        out.append({"label": lbl, "n_pass": len(passed), "n_fail": len(failed),
                    "n_dec_pass": len(dp), "n_dec_fail": len(dfl),
                    "win_rate_pass": wp, "win_rate_fail": wf, "edge": edge, "p_value": p_value})
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


def _rows_for(scope: str) -> list[dict]:
    """Resolved-trade rows for a strategy scope. Daily/swing and intraday are DIFFERENT strategies,
    so each learns from its own outcomes; 'all' pools them only for a combined view."""
    if scope == "daily":
        return load(PATH)
    if scope == "intraday":
        return load(INTRADAY_PATH)
    if scope == "orb":
        return load(ORB_PATH)
    return load_all()


def report(scope: str = "all", path: str | None = None) -> list[dict]:
    return attribute(_tradeable_rows(scope, path))


def direction_edge(scope: str = "daily", path: str | None = None) -> dict:
    """Realised outcome per DIRECTION (LONG / SHORT) for a strategy scope.

    Returns {"LONG": {"n": decided, "win_rate": pct|None},
             "SHORT": {"n": decided, "win_rate": pct|None}}. This is how the bot notices a whole
    side of the book it can't trade (e.g. shorts getting run over in a bull tape) — the direction
    analogue of the per-check edge above. Read-only; None win_rate until there are decided trades."""
    rows = load(path) if path else _rows_for(scope)
    out = {}
    for d in ("LONG", "SHORT"):
        decided = [t for t in rows if (t.get("direction") or "LONG") == d and t.get("status") in ("win", "loss")]
        n = len(decided)
        wr = round(sum(1 for t in decided if t["status"] == "win") / n * 100, 1) if n else None
        out[d] = {"n": n, "win_rate": wr}
    return out


def suppressed_directions(scope: str = "daily", path: str | None = None,
                          min_n: int = 25, winrate_max: float = 35.0) -> dict:
    """Directions the book has PROVEN it can't currently trade: decided-n >= min_n AND realised
    win rate <= winrate_max. Returns {DIRECTION: {"n", "win_rate"}} (empty until the evidence is
    there). The engine uses this to stop surfacing new entries in that direction as actionable."""
    out = {}
    for d, s in direction_edge(scope, path).items():
        if s["n"] >= min_n and s["win_rate"] is not None and s["win_rate"] <= winrate_max:
            out[d] = s
    return out


def _tradeable_rows(scope: str, path: str | None = None) -> list[dict]:
    """Resolved rows for a scope, EXCLUDING any direction the book has proven it can't trade and is
    now suppressing (e.g. shorts getting run over in a bull tape). A losing, no-longer-surfaced side
    shouldn't teach the conviction engine what predicts a WINNER — otherwise a check gets judged
    'bad' only because it rode along on a losing short book, which mis-ranks the trades we DO take
    (this is what made high-conviction longs look worse than medium ones). When a direction earns
    its way back above the suppression bar, its trades rejoin the learning pool automatically."""
    rows = load(path) if path else _rows_for(scope)
    try:
        supp = suppressed_directions(scope, path)
        if supp:
            rows = [t for t in rows if (t.get("direction") or "LONG") not in supp]
    except Exception:  # noqa: BLE001
        pass
    return rows


# Payoff / expectancy checks (reward:risk). These INTENTIONALLY trade win rate for larger wins — a
# higher-R:R target sits farther away, so it's hit less often but pays more when it is. Grading them
# by win-rate edge is a category error: they will always look "anti-predictive" on win rate even
# when they add expectancy, so the win-rate loop must never down-weight or retire them. (Judging
# them properly needs realized R-multiples / expectancy, not hit rate.) Matched case-insensitively.
_EXPECTANCY_CHECKS = frozenset({"worth the risk?", "reward:risk worth it?"})


def learned_weights(scope: str = "all", path: str | None = None,
                    min_n: int = 12, max_adj: float = 0.5, retire_edge: float = -15.0,
                    bonferroni: bool = True, alpha: float = 0.05) -> dict:
    """Turn the per-check edge into a {label: weight-multiplier} the conviction engine can use to
    ADAPT — checks that have historically predicted winners get up-weighted, those that haven't get
    down-weighted, and checks that have proven ANTI-predictive get RETIRED (weight -> 0). This is
    the bot learning from its own resolved wins vs losses.

    `scope` selects the strategy: "daily" learns only from daily/swing outcomes, "intraday" only
    from intraday outcomes, "all" pools both. Daily and intraday are different strategies, so they
    should each learn from their own book — call with the matching scope per timeframe.

    Strictly gated: a check only earns an adjustment once it has at least `min_n` *decided* trades on
    BOTH the pass and fail sides (so the edge is real, not noise). Graded multipliers are bounded to
    1 ± max_adj so no single check can run away. A check whose edge is <= `retire_edge` (i.e. it wins
    LESS when it passes than when it fails, by a wide, well-sampled margin) is dropped to weight 0 —
    acting on the analyst's own 'down-weight or retire' finding instead of only reporting it. Returns
    {} until there's enough data — so early on the engine runs on its transparent default weights."""
    rows = attribute(_tradeable_rows(scope, path))
    # Bonferroni p-hack guard: we test many checks at once, so some will look predictive by pure
    # chance. Correct the significance bar for how many checks are eligible (alpha / m) — a check
    # only earns a weight change / retirement if its pass-vs-fail edge clears that stricter bar.
    # Without this, a lucky check gets rewarded and the engine chases noise.
    eligible = [r for r in rows if r.get("edge") is not None and min(r.get("n_pass", 0), r.get("n_fail", 0)) >= min_n]
    m = max(1, len(eligible))
    alpha_corr = alpha / m
    out = {}
    for r in eligible:
        if (r.get("label") or "").strip().lower() in _EXPECTANCY_CHECKS:
            continue                        # payoff check — never retire/adjust on win-rate edge (see note above)
        if bonferroni:
            p = r.get("p_value")
            if p is None or p >= alpha_corr:
                continue                    # not significant after correction -> treat as noise, leave at default
        edge = r["edge"]
        if edge <= retire_edge:
            out[r["label"]] = 0.0          # proven anti-predictive -> retire it
        else:
            mult = 1.0 + max(-max_adj, min(max_adj, edge / 100.0))
            out[r["label"]] = round(mult, 2)
    return out
