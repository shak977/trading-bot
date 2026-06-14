# Spec 3 — Conviction×Volatility Sizing + Trailing/Partial Exits

**Date:** 2026-06-14
**Goal alignment:** secondary to "more opportunities," but the highest *expectancy-per-effort* lever
in the codebase (flagged by the planning pass). Makes every trade — existing and new — worth more.

## Problem

Two of the weakest links in the current system both sit *after* the signal:

1. **Flat sizing.** Every position risks `paper_risk_pct` (default 2%) of equity regardless of
   conviction or volatility (`_qty()` in `paper.py`, `risk.py`). A High-conviction, low-vol setup and
   a marginal, high-vol one get the same bet. That throws away the easiest expectancy gain available.
2. **Mechanical exits.** Entries are bracket OCO with a fixed % / ATR stop and a **fixed 15% target**
   (`take_profit_pct`). Every winner is capped at the same place; there is no trailing, no partial
   profit-take, no time-stop for trades that go nowhere. The `trail_atr_mult` config exists but is
   **off by default and only honored in the backtest**, never live.

## Non-goals

- No new signals or universe changes (Specs 1–2).
- No change to *which* trades we take — only *how big* and *how we exit*.
- No martingale / no averaging down. Risk per trade only ever scales **down** from the 2% ceiling.

## Design

### Part A — Conviction × volatility-targeted sizing

Replace flat `risk_pct` with a multiplier in `risk.py` (single source of truth, consumed by both
`backtest.py` and `paper.py`):

```
sized_risk_pct = base_risk_pct
               * conviction_mult(label)      # High 1.0, Medium 0.6, Low 0.3 (configurable)
               * vol_target_mult(atr_pct)     # scale toward a target volatility
```

- `conviction_mult`: bet more when the checklist is strong. Uses the label already computed.
- `vol_target_mult`: clamp so a name with double the ATR% gets ~half the risk — keeps **dollar
  volatility per position** roughly constant. Capped to `[min_mult, 1.0]` so we never *increase*
  risk above the 2% ceiling, only throttle it.
- Net effect: account naturally concentrates into high-conviction, well-behaved setups. Pure
  expectancy gain, no new alpha required.

### Part B — Exit management

Three additions, each independently toggleable:

1. **ATR trailing stop (live + backtest).** Wire the existing `trail_atr_mult` into live order
   management, not just the backtest. Once a position is up ≥1R, ratchet the stop to
   `price - trail_atr_mult * ATR` (mirror for shorts), never loosening.
2. **Partial profit-take.** Scale out half at 1R, let the remainder run under the trailing stop.
   This is the structural change that lifts expectancy without hurting win rate — you bank a winner
   *and* keep upside.
3. **Time-stop.** Close (or tighten to breakeven) a position that hasn't reached 1R after
   `max_hold_days`. Dead-money trades are a silent expectancy tax.

### The live-execution complication (this is why it's a separate spec)

Live, we submit a single **bracket OCO** (`broker.submit_bracket`): market entry + stop + target as
one unit. Partial scale-out and trailing **cannot** be expressed in that one order — they require
*managing legs over time*:

- Move from "fire and forget bracket" to a small **position manager** that, each run, reads open
  positions + current price/ATR and **amends/replaces** the stop and target legs (Alpaca
  `replace_order_by_id`), and submits the partial-close order at 1R.
- This is stateful order management across runs and is the bulk of the risk in this spec. It must be
  idempotent (re-runs must not double-close) and fail-safe (any error leaves the original protective
  bracket intact — never strip a stop).
- Backtest side is simpler: `backtest.py` already walks bars intrabar, so partial-take, trailing, and
  time-stop are added to the existing loop directly.

## Data flow

```
risk.size(equity, entry, stop, conviction_label, atr_pct) -> shares   # Part A, shared
paper.run():
  - existing entry path uses risk.size                                # Part A live
  - NEW manage_open_positions(): trail / partial / time-stop          # Part B live
backtest._simulate(): trail + partial + time-stop in the bar loop      # Part B backtest
```

## Files touched

- `risk.py` — conviction × vol sizing (single source of truth)
- `paper.py` — use new sizing; add `manage_open_positions()` exit manager
- `broker.py` — `replace_order_by_id` / partial-close helpers; read open order legs
- `backtest.py` — partial-take, trailing, time-stop in the simulator
- `config.py` — `conviction_mult_*`, `vol_target_atr_pct`, `min_size_mult`, `partial_take_r`,
  `trail_atr_mult` (default on), `max_hold_days`
- `dashboard.py` — show sized risk % and exit state (trailing / partial taken) per position
- `backtest_compare.py` — A/B variants: flat vs weighted sizing; bracket vs trail+partial+time

## Testing

- **Unit (risk.py):** sizing multipliers across conviction × ATP% grid; never exceeds base ceiling;
  clamps at `min_size_mult`.
- **Unit (exit logic, backtest):** planted series proving partial-take banks half at 1R, trail
  ratchets and never loosens, time-stop fires at the right bar.
- **Integration (paper, dry-run/mocked broker):** `manage_open_positions()` is idempotent across two
  consecutive runs; an injected broker error leaves the protective stop intact (the critical safety
  property).
- **Backtest A/B:** quantify expectancy, win rate, avg win/avg loss, and max drawdown for
  flat-vs-weighted sizing and bracket-vs-managed exits across the universe.

## Risks / open questions

- **Live order management is the real risk.** Partial fills, OCO leg replacement semantics, and race
  conditions between our amend and a stop already triggering. Mitigation: build behind a
  `manage_exits` flag, start with *trailing-only* (no partials) to prove the amend path, then add
  partial-take. Every code path must fail toward "protective stop stays."
- **Win-rate optics:** weighted sizing + let-winners-run *lowers raw win rate* even as expectancy
  rises (the goal you didn't pick, but worth stating). The dashboard should lead with expectancy and
  avg-win/avg-loss, not win rate, so the change doesn't *look* like a regression.
- **Backtest realism:** trailing/partial logic assumes intrabar fills at the level; gap-throughs
  aren't modeled. Keep slippage conservative and don't over-fit the multipliers.

## Rollout

1. Part A (sizing) first — pure function change, fully backtestable, no live order-management risk.
2. Part B trailing-only behind `manage_exits` — proves the amend path on the paper account.
3. Add partial-take + time-stop once trailing is trusted live.
