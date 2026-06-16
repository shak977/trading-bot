# Signal Desk — Intelligence Roadmap

*How the bot gets smarter at signals and execution. Grounded in current quant/ML/LLM-agent practice and the system's own track record. Paper-trading only; nothing here authorises live trading.*

Generated as a living plan. The **autonomous nightly analyst** (now live) reviews the bot's own trades each evening and re-prioritises this list with real evidence.

---

## How "autonomous" actually works here

There is no process grinding for hours in the background, and your Mac does not need to be on. Autonomy = **recurring cloud jobs on GitHub Actions**:

- The **scan/build** runs every ~10 min in market hours (signals, paper trades, shadow trackers, the bounded learned-weights loop).
- The **nightly analyst** (`analyst.py`, `analyst.yml`) runs after the close, reviews every strategy bucket, and writes a dated report of prioritised proposed changes — visible on the dashboard (Track record → 🤖 Autonomous analyst). It *proposes*; you *approve*.

This is the self-improving flywheel: the bot trades, grades itself, and tells you what to change — forever, unattended.

---

## The four workstreams

### 1. Signal intelligence (LLM + agentic)

The bot already has a rules meta-model, regime classifier, and LLM news scoring. Next steps, in order of evidence-to-effort:

- **Multi-agent "trade committee"** — specialised LLM roles (a fundamentals view, a technicals view, a news/catalyst view, a macro/regime view) debate each top candidate and return a structured accept / reduce / reject with reasons. Current research finds multi-agent debate improves factual validity and selectivity versus a single prompt. Keep it cheap: top names only, one batched call, headlines/levels we already computed — never raw price prediction.
- **Per-signal research agent** — for each actionable name, gather the latest filings/earnings/analyst actions into the structured signal contract the meta-model reads. (Builds on the existing `structured.py` / `nlp` layer.)
- **Point-in-time discipline** — any LLM/news feature must use only information available at the signal timestamp. Look-ahead bias is the #1 way these systems flatter themselves in backtests.

### 2. Execution quality (turning good signals into good trades)

Good signals lose to bad execution. Priorities:

- **Position sizing** — move from flat risk-% to **volatility-targeting** and **fractional Kelly** (¼–½ Kelly, capped), so size scales with edge and inverse-scales with volatility. Already have ATR and conviction; this is mostly arithmetic + guardrails.
- **Exits** — partial profit at +1R/+2R, trailing stop after profit, VWAP-loss and time-based stops (the ORB module already does EOD-flat; generalise to swing).
- **Entry timing & cost realism** — model spread/slippage/partial-fills in every backtest (ORB already does); extend the conservative cost model to the swing book so expectancy is net, not gross.

### 3. ML meta-model (meta-labeling)

The canonical upgrade, once enough labelled trades exist:

- **Triple-barrier labels** (López de Prado): label each past signal +1 / −1 / 0 by whether it hit target, stop, or timed out first — exactly the data the shadow trackers already produce.
- **Meta-labeling**: a secondary model (gradient boosting) learns *whether to act* on a primary signal and *how big* to size it — separating "side" (the rules engine) from "size/confidence" (the ML layer). This suppresses false positives without touching the strategy logic.
- **Gate strictly**: train only when there are enough decided trades per bucket; run advisory-first (shadow) before it influences sizing; walk-forward validate. The existing per-check attribution is the rules-based precursor to this.

### 4. Autonomous learning loop *(live now)*

- **Bounded auto-adaptation** *(live)* — per-check learned weights adjust within ±0.5 from the bot's own wins/losses, per strategy bucket.
- **Nightly analyst** *(live)* — reviews all buckets, flags checks that don't separate winners, hostile regimes, loss streaks, and the best ORB window; writes prioritised proposals + an LLM narrative.
- **Next**: let the analyst open a **pull request** with the concrete config/weight diff it proposes, so approving a change is one click — still human-gated, never auto-merged.

---

## Guardrails (non-negotiable)

- Paper-trading only; live execution stays disabled by default.
- The rules-based risk engine keeps final authority — no model confidence overrides max-loss, max-position, or no-trade gates.
- Every autonomous change is **advisory and reversible**; the analyst proposes, a human approves.
- No look-ahead, no survivorship bias, no optimising risk limits to improve returns.

---

## Sources

- Meta-labeling & triple-barrier — López de Prado: [Meta-Labeling (Wikipedia)](https://en.wikipedia.org/wiki/Meta-Labeling), [Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- LLM agents in trading (2022–2025 review): [LLMs in equity markets (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/), [QuantAgent multi-agent](https://arxiv.org/abs/2509.09995), [Look-ahead bias benchmark](https://arxiv.org/pdf/2601.13770)
