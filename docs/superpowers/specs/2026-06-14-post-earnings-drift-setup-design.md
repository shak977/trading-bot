# Spec 2 — Post-Earnings-Drift (PEAD) Setup

**Date:** 2026-06-14
**Goal alignment:** "More & better opportunities" — add a *new setup type* that captures an edge we
currently gate out entirely.

## Problem

The conviction model treats earnings purely as **risk to avoid**: `_conviction()` fails the earnings
check at ≤2 days out and caps conviction at Medium, and the gate suppresses fresh entries into the
event. That is correct for *pre*-earnings — but it means we also ignore one of the most durable,
well-documented edges in equities: **post-earnings announcement drift (PEAD)** — stocks that gap and
hold on a strong beat tend to keep drifting in that direction for days to weeks. We never trade it.

This spec adds a PEAD setup that activates **after** a report, in the direction of a strong,
confirmed reaction — complementing (not conflicting with) the existing pre-earnings gate.

## Non-goals

- Not trading the earnings event itself (no holding into the print — the existing gate stays).
- No options/IV analysis (separate, paid-data territory).
- No fundamental modeling of the beat magnitude beyond what the price reaction already encodes.

## Design

### The signal (new strategy in `strategies.py`)

Mirror the existing strategy interface (each strategy returns a 0/1 position series + metadata, fed
into the confluence vote in `strategies.py`). Add `pead_long` and `pead_short`:

Entry conditions (long; short is the mirror):
1. An earnings report occurred **within the last `pead_window` trading days** (e.g. 1–5).
2. The **reaction bar** (first session after the report) gapped up and closed strong — gap ≥
   `pead_gap_min` (e.g. +3%) and closed in the top third of its range.
3. Price is **holding** the reaction — current close is above the reaction bar's close (drift not
   yet faded) and above the rising 20-MA.
4. Standard quality gates still apply (liquidity, ATR%, not over-extended vs trend).

Exit: existing bracket/stop logic, plus drift-specific invalidation — close back below the reaction
bar's low cancels the thesis.

### The hard part — earnings dates

PEAD needs to know *when* a company reported. Two regimes:

- **Live signal:** `research.py` already fetches an earnings date (days-out) from Finnhub. Extend it
  to expose the *most recent past* earnings date, not just the next one. This is enough to fire live
  signals. (No Finnhub key → PEAD silently disabled, like other research-gated features.)
- **Backtest:** Finnhub's calendar is forward-looking and historical earnings dates are sparse on
  the free tier — so a clean historical backtest is **not** straightforwardly available. We address
  this honestly rather than fake it:
  - **Proxy detector** for backtest only: infer probable earnings dates from the bar series itself —
    an isolated volume spike (≥ `earnings_vol_mult` × median) coincident with an outsized gap, on a
    roughly quarterly cadence. Flag these as *inferred* earnings events. This lets us backtest the
    drift mechanic on price/volume structure without a clean date feed.
  - The backtest report **labels PEAD results as proxy-based** and lower-confidence than the MA/RSI
    strategies whose entries are exact. We do not pretend the proxy equals real earnings dates.
  - Primary validation for PEAD is therefore **forward, on the paper account** (real fills, real
    earnings dates), with the backtest as a sanity check on the drift mechanic only.

### Interaction with the existing earnings gate

- The pre-earnings gate fires on the *next* earnings being ≤2 days out. PEAD fires on the *previous*
  earnings being ≤`pead_window` days ago. These windows do not overlap, but we add an explicit
  guard: PEAD is suppressed if the *next* earnings is also imminent (a name reporting again within
  the drift window — rare, but avoid buying drift straight into another print).
- A PEAD entry should **not** be capped at Medium by the earnings logic — the drift is the thesis.
  This requires the gate to distinguish "into earnings" (cap) from "out of earnings drift" (allow).

## Data flow

```
research.fetch_earnings(sym) -> {next_date, last_date}   # extend existing call
strategies.pead_long/short(bars, last_earnings_date)     # new, returns position series + meta
  -> confluence vote (existing)
  -> _conviction(): earnings check made drift-aware
  -> tracker / paper as usual
```

## Files touched

- `strategies.py` — `pead_long`, `pead_short` + register in the strategy list
- `research.py` — expose most-recent past earnings date
- `scanner.py` — pass last-earnings date through; make earnings conviction check drift-aware
- `backtest.py` / `analytics.py` — proxy earnings detector for backtest; label PEAD as proxy-based
- `config.py` — `pead_window`, `pead_gap_min`, `earnings_vol_mult`, `pead_enabled`
- `dashboard.py` — tag PEAD signals visibly ("post-earnings drift") so they're not confused with
  trend setups

## Testing

- **Unit (strategy):** synthetic bar series with a planted gap-and-hold → fires; gap-and-fade →
  doesn't; gap with weak close → doesn't; short mirror.
- **Unit (proxy detector):** series with a planted quarterly volume/gap spike → detected at the
  right bar; quiet series → nothing.
- **Backtest:** PEAD-only run on the universe (proxy dates), reported separately and labelled
  lower-confidence. Compare drift win rate vs random entry as a mechanic check.
- **Forward:** paper account is the real scorekeeper; the new realized-round-trip stats already
  segment by signal so PEAD win rate accrues honestly.

## Risks / open questions

- **Date accuracy is the whole ballgame.** Live PEAD is only as good as Finnhub's last-earnings
  date. We should spot-check a handful against reality before trusting it.
- **Proxy backtest is suggestive, not proof.** Stated plainly above; PEAD ships gated by
  `pead_enabled` and is validated forward.
- **Sample sparsity:** at ~quarterly cadence, PEAD produces far fewer signals than trend setups.
  It's a quality-additive setup, not a volume driver — set expectations accordingly.
