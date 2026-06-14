# Post-Earnings-Drift (PEAD) Setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-earnings-drift long/short setup that captures the durable drift after a strong, high-volume earnings reaction — an edge currently gated out entirely.

**Architecture:** PEAD is implemented as two bars-only strategies (`pead_long`, `pead_short`) that fit the existing `strategies.py` interface exactly (`fn(df, cfg) -> 0/1 Series`). The earnings event is **inferred from the bars** — a large move on a volume surge — so the strategy needs no external earnings-date feed, flows automatically into confluence + per-strategy backtests, and is fully testable offline.

**Tech Stack:** Python 3.13, pandas, numpy. Tests in `selftest.py` (`_ok` convention, `python3 selftest.py`).

---

## Design note — deviation from the spec (justified)

The spec proposed live signals from Finnhub's last-earnings date with a separate volume/gap *proxy* only for backtesting. That split would break the uniform bars-only strategy interface (`evaluate`, `analytics.strategy_edges`, `backtest.backtest_positions` all assume `fn(df, cfg)`), and earnings dates aren't reliably available on the free tier anyway. Instead, the **volume+move proxy is the single trigger everywhere** — uniform, backtestable, no feed dependency. The real Finnhub earnings date remains available via the existing earnings gate; no change needed there because that gate keys on the *next* earnings (~90 days out right after a report), so it never suppresses a PEAD entry.

**Note on synthetic data:** `data.synthetic_bars` sets `open[i] = close[i-1]` (no overnight gaps), so PEAD's reaction detector keys on the **close-to-close** move × a **volume surge**, not the open gap. Tests build explicit planted-reaction frames (as `test_short_engine` does), not random synthetics.

---

## File Structure

- **Modify** `config.py` — PEAD knobs.
- **Modify** `strategies.py` — `pead_long`, `pead_short`, a shared `_pead` helper, and registry entries.
- **Modify** `selftest.py` — tests for the strategy and its confluence/edge integration.
- Dashboard: **no change** — PEAD's label flows through the existing strategy-confluence display.

---

## Task 1: PEAD config knobs

**Files:**
- Modify: `config.py` (after the `regime_block_buys` line, within the Refinements block ~line 63)

- [ ] **Step 1: Add config fields**

```python
    # --- Post-earnings drift (PEAD) setup ---
    pead_enabled: bool = True       # include the post-earnings-drift strategy in confluence/backtests
    pead_window: int = 5            # bars after the reaction during which a drift entry may trigger
    pead_gap_min: float = 0.04      # reaction = close-to-close move >= this (4%) ...
    pead_vol_mult: float = 1.5      # ... AND volume >= this multiple of the median (earnings-like surge)
    pead_vol_window: int = 20       # median-volume lookback for the surge test
```

- [ ] **Step 2: Verify import**

Run: `python3 -c "from config import CONFIG; print(CONFIG.pead_enabled, CONFIG.pead_window, CONFIG.pead_gap_min, CONFIG.pead_vol_mult)"`
Expected: `True 5 0.04 1.5`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: add post-earnings-drift (PEAD) knobs"
```

---

## Task 2: `pead_long` / `pead_short` strategies + registry

**Files:**
- Modify: `strategies.py` — add `_pead` helper + two strategy fns + registry entries.
- Test: `selftest.py` (new `test_pead`)

- [ ] **Step 1: Write the failing test**

Add to `selftest.py` (above `def main():`):

```python
def test_pead():
    print("post-earnings drift:")
    import numpy as np
    import pandas as pd
    import strategies
    from config import CONFIG
    n = 60
    idx = pd.date_range("2024-01-01", periods=n)
    # Flat-ish base, then a big up reaction on a volume surge at bar 40, held afterwards.
    close = np.linspace(100, 102, n).astype(float)
    vol = np.full(n, 1_000_000.0)
    close[40] = close[39] * 1.08          # +8% reaction (earnings-like)
    vol[40] = 5_000_000.0                 # volume surge
    close[41:46] = close[40] * 1.01       # drift holds above the reaction close
    df = pd.DataFrame({"open": np.concatenate([[close[0]], close[:-1]]),
                       "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": vol}, index=idx)
    pos = strategies.pead_long(df, CONFIG)
    _ok("pead is a 0/1 series", set(pos.dropna().unique()).issubset({0.0, 1.0}) and len(pos) == n)
    _ok("pead long active during the held drift", bool(pos.iloc[44] == 1.0))
    _ok("pead long flat before the reaction", bool(pos.iloc[35] == 0.0))
    # break below the reaction low cancels the drift
    close2 = close.copy(); close2[47] = close[40] * 0.90
    df2 = df.copy(); df2["close"] = close2; df2["low"] = close2 * 0.99
    pos2 = strategies.pead_long(df2, CONFIG)
    _ok("pead long exits when the drift breaks down", bool(pos2.iloc[48] == 0.0))
    # disabled -> never fires
    import dataclasses
    off = dataclasses.replace(CONFIG, pead_enabled=False)
    _ok("pead disabled => all flat", float(strategies.pead_long(df, off).sum()) == 0.0)
    # short mirror: a big DOWN reaction, held below
    closed = np.linspace(100, 98, n).astype(float)
    closed[40] = closed[39] * 0.92; closed[41:46] = closed[40] * 0.99
    vold = np.full(n, 1_000_000.0); vold[40] = 5_000_000.0
    dfd = pd.DataFrame({"open": np.concatenate([[closed[0]], closed[:-1]]),
                        "high": closed * 1.01, "low": closed * 0.99,
                        "close": closed, "volume": vold}, index=idx)
    _ok("pead short active during a held down-drift", bool(strategies.pead_short(dfd, CONFIG).iloc[44] == 1.0))
```

Register `test_pead()` in `main()` (after `test_strategies()`).

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `module 'strategies' has no attribute 'pead_long'`.

- [ ] **Step 3: Implement the helper + strategies**

In `strategies.py`, after `ema_stack_down` (line 181) and before the `STRATEGIES` registry, add:

```python
# ---- post-earnings drift (PEAD): ride the drift after a strong, high-volume reaction ----

def _pead(df: pd.DataFrame, cfg: Config, *, short: bool) -> pd.Series:
    """Drift after an earnings-like reaction, inferred from price+volume (no date feed).

    A "reaction" bar = a close-to-close move of at least ``pead_gap_min`` in the trade's
    direction, on volume >= ``pead_vol_mult`` x the rolling-median volume (an earnings-style
    surprise). We then ride the drift while price holds the reaction (above its close for a
    long, below for a short) and within ``pead_window`` bars, exiting when price breaks back
    through the reaction bar's extreme. Bars-only, so it backtests like every other strategy.
    """
    if not getattr(cfg, "pead_enabled", True):
        return pd.Series(0.0, index=df.index, dtype="float64")
    c = df["close"]
    ret = c.pct_change()
    vol = df["volume"] if "volume" in df else pd.Series(1.0, index=df.index)
    med = vol.rolling(cfg.pead_vol_window, min_periods=5).median()
    surge = vol >= (cfg.pead_vol_mult * med)
    if short:
        reaction = (ret <= -cfg.pead_gap_min) & surge
    else:
        reaction = (ret >= cfg.pead_gap_min) & surge
    reaction = reaction.fillna(False)
    # carry forward the most recent reaction bar's close / extreme
    ref_close = c.where(reaction).ffill()
    ref_ext = (df["low"] if short else df["high"]).where(reaction).ffill()
    within = reaction.rolling(cfg.pead_window, min_periods=1).max().astype(bool)
    if short:
        holding = c <= ref_close
        entry = within & holding & reaction.cumsum().gt(0)
        exit_ = c > ref_ext
    else:
        holding = c >= ref_close
        entry = within & holding & reaction.cumsum().gt(0)
        exit_ = c < ref_ext
    return _state(entry.fillna(False), exit_.fillna(False))


def pead_long(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Long the drift after a strong, high-volume up reaction (post-earnings drift)."""
    return _pead(df, cfg, short=False)


def pead_short(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Short the drift after a strong, high-volume down reaction (mirror)."""
    return _pead(df, cfg, short=True)
```

Then register in the `STRATEGIES` dict (add as the last entry before the closing brace):

```python
    "pead": ("Post-earnings drift", pead_long, "event",
             "Rides the drift after a strong, high-volume earnings-style reaction."),
```

and in `SHORT_STRATEGIES`:

```python
    "pead_dn": ("Post-earnings drift (down)", pead_short, "event",
                "Rides the down-drift after a strong, high-volume negative reaction."),
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_pead` green. `test_strategies` still passes (it uses `len(STRATEGIES)` dynamically, now 8, and `strategy_edges` covers PEAD automatically).

- [ ] **Step 5: Commit**

```bash
git add strategies.py selftest.py
git commit -m "strategies: post-earnings-drift long/short (bars-only, fits confluence + backtest)"
```

---

## Task 3: Confirm PEAD flows through confluence + scan + dashboard

**Files:**
- Test: `selftest.py` (new `test_pead_in_pipeline`)

No new production code — this is an integration guard proving PEAD reaches confluence, per-strategy edges, and the scan row (and therefore the dashboard's strategy display).

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_pead_in_pipeline():
    print("pead in pipeline:")
    import strategies, analytics
    from data import synthetic_bars
    from config import CONFIG
    df = synthetic_bars("TEST", n=CONFIG.lookback_days)
    ev = strategies.evaluate(df, CONFIG)
    _ok("confluence total now counts PEAD", ev["total"] == len(strategies.STRATEGIES))
    _ok("PEAD registered long + short", "pead" in strategies.STRATEGIES and "pead_dn" in strategies.SHORT_STRATEGIES)
    se = analytics.strategy_edges(df, CONFIG)
    _ok("per-strategy edges include PEAD", "pead" in se["by"] and "pead_dn" in se["by"])
    _ok("PEAD edge tagged with a side", se["by"]["pead"]["side"] == "long" and se["by"]["pead_dn"]["side"] == "short")
```

Register `test_pead_in_pipeline()` in `main()`.

- [ ] **Step 2: Run to verify it fails (or passes)**

Run: `python3 selftest.py`
Expected: PASS once Task 2 is in (this is a guard; it should already hold). If `analytics.strategy_edges` doesn't pick PEAD up, that signals a registry gap to fix in Task 2.

- [ ] **Step 3: Commit**

```bash
git add selftest.py
git commit -m "selftest: guard that PEAD reaches confluence + per-strategy edges"
```

---

## Final verification

- [ ] **Run the whole suite**

Run: `python3 selftest.py`
Expected: ends with `ALL TESTS PASSED`.

- [ ] **Confirm PEAD surfaces in a built page (offline)**

PEAD's label "Post-earnings drift" flows through the existing strategy-confluence rendering; when a scanned name has a held reaction, the label appears among its strategies. No dashboard code change required.

---

## Self-Review notes (author)

- **Spec coverage:** the signal (PEAD long/short) → Task 2; earnings-date handling → resolved by the bars-only proxy (deviation documented); interaction with the earnings gate → no change needed (gate keys on *next* earnings, never overlaps a post-earnings entry); backtestability → automatic via the uniform interface (Task 3); dashboard tagging → via existing strategy labels.
- **Dropped from spec:** the separate Finnhub-live vs proxy-backtest split, and the drift-aware gate change — both unnecessary under the bars-only design. Recorded above.
- **Type consistency:** `pead_long(df, cfg)` / `pead_short(df, cfg)` match the `fn(df, cfg) -> pd.Series` contract used by `positions`, `evaluate`, `analytics.strategy_edges`. Registry keys `"pead"` / `"pead_dn"` are referenced identically in Tasks 2–3.
- **No placeholders:** every step shows real code; tests use planted-reaction frames (synthetic randoms can't gap by construction).
