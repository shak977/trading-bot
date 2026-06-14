# Spec 1 — Relative-Strength Engine + Factor-Attribution Loop

**Date:** 2026-06-14
**Goal alignment:** "More & better opportunities" — widen the candidate funnel *and* tighten the
quality filter at the same time, using only the daily bars we already pull.

## Problem

Today the scan looks at ~61 core names plus live movers (capped at 90) and grades each with a
~13-check conviction model where **every check is weighted equally** (`_conviction()` in
`scanner.py`). Two consequences:

1. We have no notion of **relative strength** (RS) — how a name is performing *versus the market*.
   This is one of the most robust momentum factors in the literature and we compute none of it.
2. Because checks are equally weighted, simply widening the universe would add as much noise as
   signal. We also have a `track_record.json` that could tell us *which checks actually predict
   wins*, but nothing reads it back. The filter is static and unvalidated.

This spec adds RS as both a **discovery** mechanism (find better names) and a **heavily-weighted
conviction factor** (rank the names we found), and closes a **measurement loop** so the wider
funnel stays clean instead of getting noisier.

## Non-goals

- No new data vendor. SPY/benchmark bars come from the existing Alpaca path in `data.py`.
- No intraday data. Daily bars only.
- No change to position sizing or exits (that is Spec 3).
- No automatic re-weighting of conviction in v1 — we *surface* per-factor edge and apply a single
  hand-set RS weight. Auto-tuning weights from attribution is an explicit follow-up once we have
  enough resolved trades to trust it.

## Design

Three cooperating units, each independently testable.

### Unit A — `rs.py` (new, pure functions)

Responsibility: turn a price-history map into relative-strength scores and percentile ranks. No I/O.

```
rs_score(stock_closes, bench_closes, lookbacks=(21, 63, 126)) -> float | None
    # Blended trailing return of the stock minus the benchmark over each lookback,
    # weighted toward the longer windows (e.g. 0.2/0.3/0.5). None if insufficient history.

rank_universe(scores: dict[sym, float]) -> dict[sym, {"rs": float, "pct": int}]
    # Percentile-rank scores across the scanned set (0-100). Deterministic, ties averaged.
```

- What it does: measures momentum vs the market and ranks it cross-sectionally.
- How you use it: feed it closes, get a score + percentile per symbol.
- Depends on: nothing but the bars passed in (keeps it trivially unit-testable).

### Unit B — universe expansion + promotion (in `scanner.py`)

- Enlarge `CORE_WATCHLIST` into a larger liquid pool (target ~300–500 names; sourced once, checked
  in as a static list so the scan stays deterministic offline).
- **Do not** run heavy analysis on all of them. After cheap quality gates (`scanner.py:180` —
  price/ATR%/day-move/history), compute RS for every survivor, then **promote only the top-RS
  slice** (configurable `rs_promote_top`, e.g. 120) into the existing expensive path
  (strategy confluence, research, conviction, LLM). Breadth without quality dilution: RS does the
  pre-filtering that volume-movers alone can't.
- SPY (or `cfg.benchmark`, default `SPY`) is fetched once per run and reused for every RS calc.

### Unit C — RS conviction check + attribution panel

- **New conviction check** in `_conviction()`: "Relative strength" — pass if RS percentile is high
  (e.g. ≥70 for longs / ≤30 for shorts), warn mid, fail when the name is a market laggard you're
  trying to buy (or a leader you're trying to short). Given equal weighting today, we give RS a
  **2× weight** as the one deliberate exception until Unit D justifies a full re-weight.
- **Attribution module** `attribution.py` (new): read `track_record.json` (and the paper
  realized round-trips from `paper.py`), and for each conviction check compute **win rate and
  average return when that check passed vs. did not**. Output a ranked table: "which checks earn
  their keep." Render as a new panel on the dashboard (reuses the `.trackrec` table style).
  - This is the loop-closer. It needs the conviction *check results* stored on each tracked call.
    Today `track_record.json` stores the signal but not the per-check pass/warn/fail. **Add the
    conviction check snapshot to each tracker record** at log time (`tracker.py`) so attribution
    has something to read. Backfill is impossible for old records — attribution simply starts
    accumulating from first run after deploy (surfaced honestly in the panel: "N resolved trades
    with factor data").

## Data flow

```
data.fetch(universe + benchmark)            # daily closes, SPY included
  -> cheap quality gates (existing)
  -> rs.rs_score / rank_universe            # Unit A
  -> promote top-RS slice                   # Unit B
  -> existing confluence + research
  -> _conviction() incl. RS check (2x)      # Unit C
  -> tracker logs signal + conviction snapshot
  -> attribution.py reads resolved record   # Unit C panel
```

## Files touched

- `rs.py` — new, pure (Unit A)
- `attribution.py` — new, reads track record (Unit C)
- `scanner.py` — universe pool, RS promotion, RS conviction check
- `data.py` — ensure benchmark bars fetched/cached
- `tracker.py` — store conviction check snapshot on each logged call
- `dashboard.py` — RS percentile column on cards/terminal; factor-attribution panel
- `config.py` — `benchmark`, `rs_promote_top`, `rs_lookbacks`, `rs_weight`
- `backtest_compare.py` — add an RS-gate variant for A/B validation

## Testing

- **Unit (rs.py):** known synthetic series → exact RS scores; ranking ties; insufficient-history
  returns None; benchmark-flat case.
- **Unit (attribution.py):** hand-built resolved records → exact per-check win rates.
- **Backtest A/B:** `backtest_compare.py` variant — baseline vs "promote top-RS only" vs
  "RS-gated entries" across the universe; compare win%, avg return, Sharpe, drawdown. RS must
  *earn* its place or we ship it as display-only.
- **Determinism:** offline/synthetic run produces stable ranks (no `Date.now`/random in the path).

## Risks / open questions

- **Universe list provenance:** how do we pick the ~300–500 names? Proposal: S&P 500 + liquid
  ETFs, checked in as a static list, refreshed manually. Avoids a survivorship-biased dynamic pull.
- **Cost/latency:** more names = more bar fetches. Mitigation: RS uses only closes; heavy path is
  still capped by `rs_promote_top`. Need to confirm Alpaca rate limits at ~400 symbols/run.
- **Attribution sample size:** per-check win rates are noise until dozens of trades resolve. Panel
  must show counts and grey out low-N rows; we do **not** auto-tune weights until N is sane.

## Rollout

1. Ship `rs.py` + RS conviction check (display + 2× weight) behind no flag — pure additive.
2. Ship universe expansion + RS promotion behind `rs_promote_top` (set high initially = no-op).
3. Ship attribution panel (read-only) and let it accumulate.
4. Only after A/B + attribution agree, consider data-driven check weights (separate spec).
