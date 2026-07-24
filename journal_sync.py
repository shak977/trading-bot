"""Journal → Engine bridge.

Reads your Obsidian "Trading Brain" ticker notes and distils YOUR manual judgment into
`journal_overrides.json`, which the engine reads to (a) seed your watchlist names into the scan and
(b) respect your avoid-list. This is how your human read of the market feeds the bot — the automated
learning (meta-label retrain / nightly analyst) already learns from outcomes; this adds *your* input.

Convention: each ticker note in `Journal/Tickers/` has frontmatter:
    ---
    symbol: NVDA
    bias: watch        # watch | bullish | bearish | avoid
    engine: true       # true = feed this name to the bot
    ---

Run it locally after editing ticker notes, then commit + push `journal_overrides.json`:
    python3 journal_sync.py                 # auto-finds ../Trading Brain (or ~/Desktop/Trading Brain)
    python3 journal_sync.py "/path/to/Trading Brain"
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys


def _frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip().lower()] = v.strip().strip('"').strip("'")
    return fm


def _num(v):
    try:
        return float(str(v).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_trades(vault: str) -> tuple[list, dict]:
    """Read Journal/Trades/*.md → your REAL trade log + a summary record. Closed trades with a numeric
    return count toward win rate / expectancy. Displayed on the dashboard vs the engine's own record —
    NOT fed into the meta-label training (small, human-selected sample would bias the model)."""
    tdir = os.path.join(vault, "Journal", "Trades")
    trades, rets = [], []
    if os.path.isdir(tdir):
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith(".md"):
                continue
            fm = _frontmatter(open(os.path.join(tdir, fn), encoding="utf-8").read())
            sym = (fm.get("symbol") or "").upper().strip().lstrip("$")
            if not sym:
                continue
            ret = _num(fm.get("return_pct"))
            outcome = (fm.get("outcome") or "").lower().strip()
            t = {"symbol": sym, "status": (fm.get("status") or "open").lower().strip(),
                 "direction": (fm.get("direction") or "long").lower().strip(),
                 "outcome": outcome, "return_pct": ret,
                 "opened": fm.get("opened", ""), "closed": fm.get("closed", "")}
            trades.append(t)
            if t["status"] == "closed" and ret is not None:
                rets.append(ret)
    rec = {}
    if rets:
        wins = sum(1 for r in rets if r > 0) + sum(1 for t in trades
                                                    if t["status"] == "closed" and t["return_pct"] is None
                                                    and t["outcome"] == "win")
        n = len(rets)
        rec = {"n": n, "wins": wins, "win_rate": round(100 * wins / n, 1),
               "avg_return": round(sum(rets) / n, 2),
               "avg_win": round(sum(r for r in rets if r > 0) / max(1, sum(1 for r in rets if r > 0)), 2),
               "avg_loss": round(sum(r for r in rets if r <= 0) / max(1, sum(1 for r in rets if r <= 0)), 2),
               "open": sum(1 for t in trades if t["status"] == "open")}
    return trades, rec


def sync(vault: str) -> dict:
    tdir = os.path.join(vault, "Journal", "Tickers")
    watchlist, avoid, bias = [], [], {}
    if os.path.isdir(tdir):
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith(".md"):
                continue
            fm = _frontmatter(open(os.path.join(tdir, fn), encoding="utf-8").read())
            sym = (fm.get("symbol") or os.path.splitext(fn)[0]).upper().strip().lstrip("$")
            if not sym.isalpha() or len(sym) > 5:
                continue
            b = (fm.get("bias") or "watch").lower()
            eng = str(fm.get("engine", "true")).lower() in ("true", "yes", "1")
            bias[sym] = b
            if b == "avoid":
                avoid.append(sym)
            elif eng and b in ("watch", "bullish"):
                watchlist.append(sym)
    trades, journal_record = _parse_trades(vault)
    out = {
        "watchlist": sorted(set(watchlist)),
        "avoid": sorted(set(avoid)),
        "bias": bias,
        "trades": trades,
        "journal_record": journal_record,
        "generated": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M GMT"),
    }
    with open("journal_overrides.json", "w") as f:
        json.dump(out, f, indent=2)
    _r = out["journal_record"]
    print(f"[journal_sync] {len(out['watchlist'])} watchlist, {len(out['avoid'])} avoid, "
          f"{len(out['trades'])} trades"
          + (f" ({_r['n']} closed, {_r['win_rate']}% win)" if _r else "")
          + " -> journal_overrides.json")
    return out


def _find_vault(arg: str | None) -> str | None:
    cands = [arg] if arg else []
    cands += [os.path.join("..", "Trading Brain"), os.path.expanduser("~/Desktop/Trading Brain")]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


if __name__ == "__main__":
    vault = _find_vault(sys.argv[1] if len(sys.argv) > 1 else None)
    if not vault:
        print("Vault not found. Pass the path: python3 journal_sync.py '/path/to/Trading Brain'")
        sys.exit(0)
    sync(vault)
