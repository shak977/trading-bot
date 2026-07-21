"""Meta-labeling prototype (López de Prado) — does a learned P(win) beat the hand-tuned conviction?

Trains a leakage-safe secondary model on resolved LONG trades (shorts are gated off) using only
ENTRY-TIME features (the conviction checks + rr + liquidity + TradingView + conviction tier) to
predict P(win). Validated with WALK-FORWARD (expanding, chronological, embargoed) splits so there's
no look-ahead. Compares the model's ranking of winners against the current baseline (how many checks
passed). Pure stdlib + numpy — no sklearn, no new deps.

Run:  python3 meta_label.py            # prints the report
      python3 meta_label.py --json     # also writes meta_model.json (refit on all data) + metrics

CAVEAT: with only ~4 weeks of history this tests the METHOD and the relative ranking, NOT a
deployable edge. Re-run as track_record.json grows; trust it once you have several months / regimes.
"""
from __future__ import annotations
import json
import sys
import numpy as np

CHECK_ENC = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
LIQ = {"illiquid": 0, "thin": 1, "moderate": 2, "high": 3, "very high": 4}
TVA = {"agree": 1.0, "mixed": 0.0, "oppose": -1.0}
CONV = {"High": 2.0, "Medium": 1.0, "Low": 0.0}


def _load_longs():
    tr = json.load(open("track_record.json"))
    lo = [t for t in tr if t.get("direction") == "LONG" and t.get("status") in ("win", "loss")
          and isinstance(t.get("return_pct"), (int, float)) and t.get("advised_date")]
    lo.sort(key=lambda t: t["advised_date"])
    return lo


def _feature_names(rows):
    labs = sorted({c["label"] for t in rows for c in (t.get("checks") or [])})
    return labs + ["rr", "liquidity", "tv_align", "conviction_tier"]


def _vectorize(rows, names):
    labs = names[:-4]
    X = np.zeros((len(rows), len(names)))
    y = np.zeros(len(rows))
    base = np.zeros(len(rows))            # baseline = # of passing checks (the current 'conviction')
    ret = np.zeros(len(rows))
    for i, t in enumerate(rows):
        cks = {c["label"]: c["status"] for c in (t.get("checks") or [])}
        for j, lab in enumerate(labs):
            X[i, j] = CHECK_ENC.get(cks.get(lab), 0.5)
        X[i, len(labs)] = min(float(t.get("rr") or 0), 6.0)
        X[i, len(labs) + 1] = LIQ.get(t.get("liquidity_tier"), 2)
        X[i, len(labs) + 2] = TVA.get(t.get("tv_align"), 0.0)
        X[i, len(labs) + 3] = CONV.get(t.get("conviction"), 1.0)
        y[i] = 1.0 if t["status"] == "win" else 0.0
        base[i] = sum(1 for v in cks.values() if v == "pass")
        ret[i] = t["return_pct"]
    return X, y, base, ret


def _fit(X, y, l2=2.0, iters=800, lr=0.3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * (g.mean())
    return w, b


def _auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def walk_forward(X, y, base, ret, folds=4, embargo=2):
    n = len(y)
    edges = [int(n * k / (folds + 1)) for k in range(1, folds + 2)]
    oof_p = np.full(n, np.nan)
    for i in range(folds):
        tr_end, te_end = edges[i], edges[i + 1]
        tr_idx = np.arange(0, max(0, tr_end - embargo))
        te_idx = np.arange(tr_end, te_end)
        if len(tr_idx) < 40 or len(te_idx) == 0:
            continue
        mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-9
        w, b = _fit((X[tr_idx] - mu) / sd, y[tr_idx])
        z = ((X[te_idx] - mu) / sd) @ w + b
        oof_p[te_idx] = 1.0 / (1.0 + np.exp(-z))
    m = ~np.isnan(oof_p)
    return oof_p, m


def _topfrac(score, y, ret, frac):
    k = max(1, int(len(score) * frac))
    idx = np.argsort(-score)[:k]
    return 100 * y[idx].mean(), ret[idx].mean(), k


def report():
    rows = _load_longs()
    names = _feature_names(rows)
    X, y, base, ret = _vectorize(rows, names)
    oof, m = walk_forward(X, y, base, ret)
    ym, pm, bm, rm = y[m], oof[m], base[m], ret[m]
    out = {"n_longs": len(rows), "n_oos": int(m.sum()),
           "base_winrate": round(100 * y.mean(), 1),
           "auc_meta": round(_auc(pm, ym), 3),
           "auc_baseline_passcount": round(_auc(bm, ym), 3),
           "date_range": [rows[0]["advised_date"], rows[-1]["advised_date"]]}
    L = [f"META-LABEL PROTOTYPE — {out['n_longs']} longs, {out['n_oos']} out-of-sample "
         f"({out['date_range'][0]}..{out['date_range'][1]})",
         f"  base win rate: {out['base_winrate']}%",
         f"  AUC  meta-label P(win): {out['auc_meta']}   (0.5 = no skill, >0.55 = useful)",
         f"  AUC  baseline (#checks passed): {out['auc_baseline_passcount']}",
         "", "  TOP-FRACTION by score — win rate / avg return (meta vs baseline):"]
    out["topfrac"] = {}
    for f in (0.2, 0.3, 0.5):
        mw, mr, k = _topfrac(pm, ym, rm, f)
        bw, br, _ = _topfrac(bm, ym, rm, f)
        out["topfrac"][f] = {"meta_wr": round(mw, 1), "meta_ret": round(mr, 2),
                             "base_wr": round(bw, 1), "base_ret": round(br, 2), "n": k}
        L.append(f"    top {int(f*100)}%:  meta {mw:4.1f}%/{mr:+.2f}%   vs   baseline {bw:4.1f}%/{br:+.2f}%")
    # refit on all + feature weights
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w, b = _fit((X - mu) / sd, y)
    order = np.argsort(-np.abs(w))
    L.append("\n  TOP FEATURES (standardized weight; + = raises P(win)):")
    for j in order[:10]:
        L.append(f"    {names[j][:26]:26} {w[j]:+.2f}")
    out["model"] = {"feature_names": names, "mean": mu.tolist(), "std": sd.tolist(),
                    "weight": w.tolist(), "bias": float(b)}
    return "\n".join(L), out


if __name__ == "__main__":
    txt, out = report()
    print(txt)
    if "--json" in sys.argv:
        import datetime
        import os
        json.dump(out, open("meta_model.json", "w"), indent=1)
        # Append a compact metrics record so we can watch AUC stabilise as data accumulates.
        hist = []
        if os.path.exists("meta_history.json"):
            try:
                hist = json.load(open("meta_history.json"))
            except Exception:  # noqa: BLE001
                hist = []
        tf20 = (out.get("topfrac") or {}).get(0.2, {})
        hist.append({"date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M GMT"),
                     "n_longs": out["n_longs"], "n_oos": out["n_oos"],
                     "auc_meta": out["auc_meta"], "auc_baseline": out["auc_baseline_passcount"],
                     "top20_meta_wr": tf20.get("meta_wr"), "top20_base_wr": tf20.get("base_wr"),
                     "range": out["date_range"]})
        json.dump(hist[-400:], open("meta_history.json", "w"), indent=1)
        print("\n[wrote meta_model.json + appended meta_history.json]")
