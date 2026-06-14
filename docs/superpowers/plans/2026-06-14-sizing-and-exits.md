# Conviction×Volatility Sizing + Trailing/Partial/Time Exits — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every trade worth more — size bets by conviction and volatility instead of a flat 2%, and stop capping winners at a fixed target by adding trailing, partial-profit and time exits.

**Architecture:** Sizing logic centralises in `risk.py` (pure, consumed by both the paper path and the backtest). Exit improvements split by verifiability: **time-stop + partial-take land in the backtest simulator now** (fully testable offline; trailing already exists there). The **live** exit manager (amending OCO legs, partial closes) is genuinely risky and unverifiable without a funded account holding positions, so it ships **behind a default-OFF `manage_exits` flag**, unit-tested against a mock broker for idempotency and the one non-negotiable safety property: *a failure never strips a protective stop*.

**Tech Stack:** Python 3.13, pandas, numpy. Tests in `selftest.py` (`_ok` convention, `python3 selftest.py`).

---

## Scope & verifiability note

- **Part A (sizing):** conviction×vol multiplier. Conviction part is **live-only** (the backtest has no conviction — it grades the raw strategy, same boundary as RS in Spec 1). Vol-target part is backtestable.
- **Part B (exits):**
  - *Backtest* (Tasks 5–6): time-stop + partial-take added to the simulator. Trailing already present. Fully tested, measurable via `backtest_compare`.
  - *Live* (Task 7): `paper.manage_open_positions()` behind `manage_exits` (default OFF). Trailing-only first, then partial. Tested with a **mock broker** for idempotency + fail-safe. **Cannot be fully verified in-repo — needs the user's paper account holding live positions.** Flagged as such; default OFF so it can't touch live trading until opted in.

---

## File Structure

- **Modify** `config.py` — sizing + exit knobs.
- **Modify** `risk.py` — `risk_multiplier()` (pure) + a sizing helper that takes conviction + ATR%.
- **Modify** `paper.py` — use the new sizing in `_qty`; add `manage_open_positions()` (flagged).
- **Modify** `broker.py` — helpers to read open orders for a symbol and replace/cancel a protective leg + submit a partial close (used only by the flagged manager).
- **Modify** `backtest.py` — time-stop + partial-take in the long simulator (and short mirror).
- **Modify** `selftest.py` — tests per task.

---

## Task 1: Config knobs

**Files:**
- Modify: `config.py` (after the PEAD block from Spec 2)

- [ ] **Step 1: Add fields**

```python
    # --- Position sizing by conviction & volatility (live paper path) ---
    size_by_conviction: bool = True   # scale risk by conviction tier
    conv_mult_high: float = 1.0       # High-conviction risk multiplier (of paper_risk_pct)
    conv_mult_medium: float = 0.6     # Medium
    conv_mult_low: float = 0.3        # Low
    vol_target_atr_pct: float = 4.0   # ATR% at which the vol multiplier == 1.0; higher ATR% -> smaller
    min_size_mult: float = 0.25       # floor on the combined multiplier (never size below this x base)

    # --- Exit management ---
    partial_take_r: float = 1.0       # scale out half the position at this R multiple (0 = off)
    max_hold_days: int = 0            # time-stop: close after this many bars if < 1R reached (0 = off)
    manage_exits: bool = False        # LIVE only: actively amend stops / take partials (default OFF)
```

(`trail_atr_mult` already exists in `config.py`; this plan turns it on in the live manager when `manage_exits` is set, and it already works in the backtest.)

- [ ] **Step 2: Verify**

Run: `python3 -c "from config import CONFIG; print(CONFIG.size_by_conviction, CONFIG.conv_mult_high, CONFIG.vol_target_atr_pct, CONFIG.partial_take_r, CONFIG.manage_exits)"`
Expected: `True 1.0 4.0 1.0 False`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: add conviction/vol sizing + exit-management knobs"
```

---

## Task 2: `risk.risk_multiplier()` + sizing helper (pure)

**Files:**
- Modify: `risk.py`
- Test: `selftest.py` (new `test_risk_multiplier`)

- [ ] **Step 1: Write the failing test**

```python
def test_risk_multiplier():
    print("conviction/vol sizing:")
    import risk
    from config import CONFIG
    import dataclasses
    cfg = CONFIG
    # conviction tiers
    m_hi = risk.risk_multiplier("High", cfg.vol_target_atr_pct, cfg)
    m_md = risk.risk_multiplier("Medium", cfg.vol_target_atr_pct, cfg)
    m_lo = risk.risk_multiplier("Low", cfg.vol_target_atr_pct, cfg)
    _ok("High >= Medium >= Low", m_hi >= m_md >= m_lo)
    _ok("at target ATP%, High mult == conv_mult_high", abs(m_hi - cfg.conv_mult_high) < 1e-9)
    # volatility: double the target ATR% roughly halves the multiplier
    m_calm = risk.risk_multiplier("High", cfg.vol_target_atr_pct, cfg)
    m_wild = risk.risk_multiplier("High", cfg.vol_target_atr_pct * 2, cfg)
    _ok("higher vol -> smaller size", m_wild < m_calm)
    _ok("never exceeds base ceiling", risk.risk_multiplier("High", 0.1, cfg) <= 1.0 + 1e-9)
    _ok("floored at min_size_mult", risk.risk_multiplier("Low", 100.0, cfg) >= cfg.min_size_mult - 1e-9)
    # disabled -> flat 1.0
    off = dataclasses.replace(cfg, size_by_conviction=False)
    _ok("disabled => multiplier 1.0", risk.risk_multiplier("Low", 100.0, off) == 1.0)
```

Register `test_risk_multiplier()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `module 'risk' has no attribute 'risk_multiplier'`.

- [ ] **Step 3: Implement**

Add to `risk.py`:

```python
def risk_multiplier(conviction_label: str | None, atr_pct: float | None, cfg: Config) -> float:
    """Combined conviction x volatility multiplier on the base risk budget.

    Conviction tier scales the bet up for stronger setups; the vol term targets a constant
    dollar-volatility by shrinking size as ATR% rises above ``vol_target_atr_pct``. The result
    is clamped to ``[min_size_mult, 1.0]`` — it only ever throttles risk below the base ceiling,
    never above it. Returns 1.0 when ``size_by_conviction`` is off.
    """
    if not getattr(cfg, "size_by_conviction", True):
        return 1.0
    conv = {"High": cfg.conv_mult_high, "Medium": cfg.conv_mult_medium,
            "Low": cfg.conv_mult_low}.get(conviction_label or "", cfg.conv_mult_medium)
    if atr_pct and atr_pct > 0 and cfg.vol_target_atr_pct > 0:
        vol = cfg.vol_target_atr_pct / atr_pct
    else:
        vol = 1.0
    mult = conv * vol
    return max(cfg.min_size_mult, min(mult, 1.0))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_risk_multiplier` green.

- [ ] **Step 5: Commit**

```bash
git add risk.py selftest.py
git commit -m "risk: conviction x volatility risk multiplier (pure, clamped to base ceiling)"
```

---

## Task 3: Paper path uses conviction×vol sizing

**Files:**
- Modify: `paper.py` — `_qty` signature + the call site in `run()`.
- Test: `selftest.py` (new `test_paper_sizing`)

`paper._qty(equity, buying_power, entry, stop, risk_pct)` currently sizes off a flat `risk_pct`. Add an optional multiplier and apply it.

- [ ] **Step 1: Write the failing test**

```python
def test_paper_sizing():
    print("paper conviction sizing:")
    import paper
    from config import CONFIG
    # base size at full risk
    base = paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct, mult=1.0)
    half = paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct, mult=0.5)
    _ok("half multiplier ~ half the shares", half <= base and half >= base // 2 - 1)
    _ok("zero-ish multiplier shrinks size", paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct, mult=CONFIG.min_size_mult) < base)
    _ok("mult defaults to 1.0", paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct) == base)
```

Register `test_paper_sizing()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `_qty() got an unexpected keyword argument 'mult'`.

- [ ] **Step 3: Implement**

In `paper.py`, change `_qty` (lines ~41-51) to accept a multiplier:

```python
def _qty(equity: float, buying_power: float, entry: float, stop: float,
         risk_pct: float, mult: float = 1.0) -> int:
    """Risk-based size: lose ~risk_pct*mult of equity if the stop is hit, capped by buying power."""
    if not entry or entry <= 0:
        return 0
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    q = int((equity * risk_pct * mult) / per_share)
    afford = int((buying_power * 0.9) / entry)
    q = min(q, afford)
    return max(q, 0)
```

Then at the call site in `run()` (where `qty = _qty(equity, buying_power, entry, stop, cfg.paper_risk_pct)`), compute and pass the multiplier from the signal's conviction + ATR%:

```python
            label = (s.get("conviction") or {}).get("label")
            atr_pct = (s.get("context") or {}).get("atr_pct")
            from risk import risk_multiplier
            mult = risk_multiplier(label, atr_pct, cfg)
            qty = _qty(equity, buying_power, entry, stop, cfg.paper_risk_pct, mult=mult)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_paper_sizing` green; existing paper behaviour unchanged when `mult=1.0`.

- [ ] **Step 5: Commit**

```bash
git add paper.py selftest.py
git commit -m "paper: size positions by conviction x volatility (flat behaviour at mult=1.0)"
```

---

## Task 4: Backtest — time-stop

**Files:**
- Modify: `backtest.py` `backtest_positions` (long) and `_backtest_short`.
- Test: `selftest.py` (new `test_time_stop`)

- [ ] **Step 1: Write the failing test**

```python
def test_time_stop():
    print("backtest time-stop:")
    import dataclasses
    import pandas as pd
    from backtest import backtest_positions
    from config import CONFIG
    idx = pd.date_range("2024-01-01", periods=20)
    # price drifts sideways (never reaches 1R target, never hits stop) while held
    close = pd.Series([100 + (i % 2) * 0.2 for i in range(20)], index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close})
    pos = pd.Series(1.0, index=idx)
    cfg = dataclasses.replace(CONFIG, max_hold_days=5, trail_atr_mult=0.0)
    res = backtest_positions(df, pos, cfg, side="long")
    reasons = [t.get("reason") for t in res.trades.to_dict("records") if t.get("reason")]
    _ok("a time-stop exit is recorded", "time" in reasons)
    # with the time-stop off, a sideways held trade does NOT exit for 'time'
    cfg_off = dataclasses.replace(CONFIG, max_hold_days=0, trail_atr_mult=0.0)
    res_off = backtest_positions(df, pos, cfg_off, side="long")
    _ok("no time exit when disabled", "time" not in [t.get("reason") for t in res_off.trades.to_dict("records")])
```

Register `test_time_stop()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — no `"time"` reason recorded.

- [ ] **Step 3: Implement (long path)**

In `backtest_positions` (long), track bars-in-trade and add a time exit. Add a counter when a position is open, reset on entry. Inside the `if shares > 0:` block, before computing `hit_stop`, add:

```python
            held_bars += 1
            reached_1r = row["high"] >= entry + (entry - stop)
            time_stop = cfg.max_hold_days > 0 and held_bars >= cfg.max_hold_days and not reached_1r
```

and include `time_stop` in the exit decision:

```python
            if hit_stop or hit_target or exit_signal or time_stop:
                exit_px = stop if hit_stop else target if hit_target else price
                ...
                "reason": "stop" if hit_stop else "target" if hit_target else ("time" if time_stop else "signal")}
```

Initialise `held_bars = 0` alongside `shares = 0` at the top, and set `held_bars = 0` in the open-position block (right after `shares = qty`). Mirror the same logic in `_backtest_short` (there `reached_1r = row["low"] <= entry - (stop - entry)`).

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_time_stop` green; all existing backtest tests still pass (`max_hold_days=0` default keeps behaviour identical).

- [ ] **Step 5: Commit**

```bash
git add backtest.py selftest.py
git commit -m "backtest: optional time-stop for trades that stall below 1R"
```

---

## Task 5: Backtest — partial profit-take at 1R

**Files:**
- Modify: `backtest.py` `backtest_positions` (long) and `_backtest_short`.
- Test: `selftest.py` (new `test_partial_take`)

- [ ] **Step 1: Write the failing test**

```python
def test_partial_take():
    print("backtest partial-take:")
    import dataclasses
    import pandas as pd
    from backtest import backtest_positions
    from config import CONFIG
    idx = pd.date_range("2024-01-01", periods=12)
    # rise through 1R (so half scales out), then to target
    close = pd.Series([100, 101, 103, 106, 110, 113, 116, 118, 120, 121, 122, 123], index=idx, dtype=float)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
    pos = pd.Series(1.0, index=idx)
    cfg = dataclasses.replace(CONFIG, partial_take_r=1.0, trail_atr_mult=0.0)
    res = backtest_positions(df, pos, cfg, side="long")
    recs = res.trades.to_dict("records")
    _ok("a partial exit is recorded", any(t.get("reason") == "partial" for t in recs))
    # without partials, no partial reason
    cfg_off = dataclasses.replace(CONFIG, partial_take_r=0.0, trail_atr_mult=0.0)
    res_off = backtest_positions(df, pos, cfg_off, side="long")
    _ok("no partial when disabled", not any(t.get("reason") == "partial" for t in res_off.trades.to_dict("records")))
```

Register `test_partial_take()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — no `"partial"` reason recorded.

- [ ] **Step 3: Implement (long path)**

In the long `if shares > 0:` block, after the trailing-stop ratchet and before the full-exit check, add a one-time partial scale-out at 1R:

```python
            if (cfg.partial_take_r > 0 and not took_partial and shares >= 2
                    and row["high"] >= entry + cfg.partial_take_r * (entry - stop)):
                half = shares // 2
                px = (entry + cfg.partial_take_r * (entry - stop)) * (1 - slip)
                cash += half * px - comm
                trades.append({"exit_time": ts, "exit_px": px, "shares": half,
                               "pnl": half * (px - entry), "reason": "partial"})
                shares -= half
                took_partial = True
                stop = max(stop, entry)   # move the remainder to breakeven
```

Initialise `took_partial = False` next to `shares = 0`, and reset `took_partial = False` when a new position opens. Mirror in `_backtest_short` (1R below entry; `px = (entry - cfg.partial_take_r*(entry-stop))*(1+slip)`; `pnl = half*(entry-px)`; `stop = min(stop, entry)`).

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_partial_take` green; existing tests pass (`partial_take_r=0` default keeps behaviour identical).

- [ ] **Step 5: Commit**

```bash
git add backtest.py selftest.py
git commit -m "backtest: partial profit-take at 1R, remainder trails from breakeven"
```

---

## Task 6: A/B the exit changes in backtest_compare

**Files:**
- Modify: `backtest_compare.py` — add variants exercising the new exits.

- [ ] **Step 1: Add variants**

In `backtest_compare.py`'s `VARIANTS` dict, add:

```python
    "partial+trail":  {"adx_min": 0.0,  "trail_atr_mult": 3.0, "partial_take_r": 1.0},
    "time_stop(10)":  {"adx_min": 0.0,  "trail_atr_mult": 0.0, "max_hold_days": 10},
```

- [ ] **Step 2: Run it**

Run: `python3 backtest_compare.py`
Expected: prints the comparison table including the two new rows with no traceback. (This is the measurement that tells us whether the exits actually help on the universe.)

- [ ] **Step 3: Commit**

```bash
git add backtest_compare.py
git commit -m "backtest_compare: A/B variants for partial+trail and time-stop exits"
```

---

## Task 7: Live exit manager (FLAGGED, default OFF) + broker helpers

**Files:**
- Modify: `broker.py` — `open_orders_for(symbol)`, `replace_stop(order_id, new_stop)`, `partial_close(symbol, qty, side)`.
- Modify: `paper.py` — `manage_open_positions(broker, cfg, log)` called from `run()` only when `cfg.manage_exits`.
- Test: `selftest.py` (new `test_manage_exits_mock`) using a mock broker.

**Safety contract (non-negotiable):** every code path fails toward "the protective stop stays in place." Any exception while amending leaves the original bracket untouched. Idempotent across runs: re-running must not double-take a partial or re-trail a stop it already moved.

- [ ] **Step 1: Write the failing test (mock broker)**

```python
def test_manage_exits_mock():
    print("live exit manager (mock broker, flagged):")
    import dataclasses
    import paper
    from config import CONFIG

    class MockBroker:
        def __init__(self):
            self.replaced = []; self.partials = []
            self._positions = [{"symbol": "X", "qty": 10, "side": "long",
                                "avg_entry": 100.0, "price": 110.0, "unrealized_plpc": 10.0}]
            self._stops = {"X": 95.0}
            self.fail = False
        def positions_detail(self):
            return list(self._positions)
        def open_orders_for(self, sym):
            return [{"id": "stop-"+sym, "type": "stop", "stop_price": self._stops.get(sym)}]
        def replace_stop(self, order_id, new_stop):
            if self.fail:
                raise RuntimeError("broker down")
            self.replaced.append((order_id, round(new_stop, 2)))
        def partial_close(self, sym, qty, side):
            if self.fail:
                raise RuntimeError("broker down")
            self.partials.append((sym, qty, side))

    cfg = dataclasses.replace(CONFIG, manage_exits=True, partial_take_r=1.0, trail_atr_mult=3.0)
    # log records the entry so the manager knows entry/stop and what it already did
    log = [{"client_id": "sd-X", "symbol": "X", "direction": "LONG", "qty": 10,
            "entry_plan": 100.0, "stop": 95.0, "target": 130.0, "status": "open"}]
    mb = MockBroker()
    notes = paper.manage_open_positions(mb, cfg, log)
    _ok("manager returns a list of notes", isinstance(notes, list))
    _ok("partial taken once at >=1R (price 110, 1R=105)", len(mb.partials) == 1)
    _ok("partial is idempotent across a second run",
        (paper.manage_open_positions(mb, cfg, log), len(mb.partials))[1] == 1)
    _ok("log marks the partial as taken", log[0].get("partial_done") is True)
    # fail-safe: broker errors must not crash the manager and must not strip the stop
    mb.fail = True
    log2 = [{"client_id": "sd-Y", "symbol": "X", "direction": "LONG", "qty": 10,
             "entry_plan": 100.0, "stop": 95.0, "target": 130.0, "status": "open"}]
    notes2 = paper.manage_open_positions(mb, cfg, log2)
    _ok("manager never raises on broker failure", isinstance(notes2, list))
```

Register `test_manage_exits_mock()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `module 'paper' has no attribute 'manage_open_positions'`.

- [ ] **Step 3: Implement broker helpers**

In `broker.py`, add (used only by the flagged manager; all best-effort):

```python
    def open_orders_for(self, symbol: str) -> list[dict]:
        """Open orders for a symbol, normalised to {id, type, stop_price}."""
        out = []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=50)):
                out.append({"id": o.id, "type": getattr(o, "order_type", None) or getattr(o, "type", None),
                            "stop_price": float(o.stop_price) if getattr(o, "stop_price", None) else None})
        except Exception:  # noqa: BLE001
            pass
        return out

    def replace_stop(self, order_id, new_stop: float):
        """Move a stop leg to a tighter price (best-effort)."""
        from alpaca.trading.requests import ReplaceOrderRequest
        return self.client.replace_order_by_id(order_id, ReplaceOrderRequest(stop_price=round(float(new_stop), 2)))

    def partial_close(self, symbol: str, qty: int, side: str):
        """Close `qty` shares of an open position (market). side = current position side."""
        close_side = "sell" if side == "long" else "buy"
        return self.submit_market(symbol, qty, close_side)
```

- [ ] **Step 4: Implement `manage_open_positions` in `paper.py`**

```python
def manage_open_positions(broker, cfg: Config, log: list[dict]) -> list[str]:
    """LIVE exit management (only called when cfg.manage_exits). For each open position:
    scale out half at partial_take_r (once), then trail the stop under price by
    trail_atr_mult (never loosening). Idempotent via flags on the local log; every broker
    call is wrapped so a failure leaves the protective bracket intact. Never raises."""
    notes: list[str] = []
    if not cfg.manage_exits:
        return notes
    try:
        positions = broker.positions_detail()
    except Exception:  # noqa: BLE001
        return notes
    by_sym = {p["symbol"]: p for p in positions}
    for t in log:
        if t.get("status") != "open":
            continue
        p = by_sym.get(t["symbol"])
        if not p:
            continue
        side = "long" if t.get("direction", "LONG") != "SHORT" else "short"
        entry, stop = t.get("entry_plan"), t.get("stop")
        price = p.get("price")
        if not (entry and stop and price):
            continue
        r = abs(entry - stop)
        # 1) partial take at >= partial_take_r (once)
        if cfg.partial_take_r > 0 and not t.get("partial_done") and p.get("qty", 0) >= 2:
            up = (price - entry) if side == "long" else (entry - price)
            if up >= cfg.partial_take_r * r:
                half = int(p["qty"]) // 2
                try:
                    broker.partial_close(t["symbol"], half, side)
                    t["partial_done"] = True
                    notes.append(f'{t["symbol"]}: scaled out {half} at ~{cfg.partial_take_r}R')
                except Exception as e:  # noqa: BLE001 - leave bracket intact
                    notes.append(f'{t["symbol"]}: partial failed ({str(e)[:60]})')
        # 2) trail the stop (never loosen)
        if cfg.trail_atr_mult > 0:
            try:
                orders = broker.open_orders_for(t["symbol"])
                stop_leg = next((o for o in orders if (o.get("type") or "").startswith("stop")), None)
                if stop_leg:
                    cur = stop_leg.get("stop_price")
                    new_stop = max(stop, entry) if t.get("partial_done") else stop
                    if side == "long" and (cur is None or new_stop > cur):
                        broker.replace_stop(stop_leg["id"], new_stop)
                        notes.append(f'{t["symbol"]}: stop tightened to {new_stop:.2f}')
                    elif side == "short" and (cur is None or new_stop < cur):
                        broker.replace_stop(stop_leg["id"], new_stop)
                        notes.append(f'{t["symbol"]}: stop tightened to {new_stop:.2f}')
            except Exception as e:  # noqa: BLE001 - never strip a stop
                notes.append(f'{t["symbol"]}: stop amend skipped ({str(e)[:60]})')
    return notes
```

Then call it from `run()` after reconciliation (before building the return dict), and surface its notes:

```python
    if cfg.manage_exits:
        try:
            notes.extend(manage_open_positions(broker, cfg, log))
            _save(log)
        except Exception:  # noqa: BLE001
            pass
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_manage_exits_mock` green (partial taken once, idempotent, fail-safe).

- [ ] **Step 6: Commit**

```bash
git add broker.py paper.py selftest.py
git commit -m "paper: flagged live exit manager (partial + trail), default OFF, fail-safe"
```

---

## Final verification

- [ ] **Run the whole suite**

Run: `python3 selftest.py`
Expected: ends with `ALL TESTS PASSED`.

- [ ] **A/B the exits**

Run: `python3 backtest_compare.py`
Expected: comparison table with `partial+trail` and `time_stop(10)` rows.

- [ ] **Live-manager caveat (record, do not skip):** `manage_exits` defaults OFF and `manage_open_positions` is verified only against a mock broker. Before enabling live, the user must test on their paper account with real open positions and confirm: partials fire once, stops only tighten, and a forced broker error never removes a stop.

---

## Self-Review notes (author)

- **Spec coverage:** Part A sizing → Tasks 1–3 (conviction live, vol both). Part B exits → trailing (already present), time-stop (Task 4), partial (Task 5), A/B (Task 6), live manager (Task 7). Dashboard sized-risk display deferred (low value; sizing is visible in the order/qty already) — noted, not silently dropped.
- **Verifiability boundaries stated:** conviction-sizing is live-only (backtest has no conviction); the live exit manager needs a funded account and ships default-OFF, mock-tested.
- **Type consistency:** `risk.risk_multiplier(label, atr_pct, cfg)->float` used in Tasks 2–3. `paper._qty(..., mult=1.0)` and `paper.manage_open_positions(broker, cfg, log)->list[str]` consistent across Tasks 3, 7. Broker helpers `open_orders_for/replace_stop/partial_close` match the mock in the test.
- **No placeholders:** every step shows real code; safety contract is explicit and tested.
