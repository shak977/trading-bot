"""Central configuration. Loads secrets from environment / .env file.

Defaults to PAPER trading. Never hard-code keys here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv optional; env vars still work
    pass


def _as_bool(val: str | None, default: bool = True) -> bool:
    if val is None or not val.strip():   # unset OR empty (an unset GitHub var passes "") -> default
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    # --- Credentials (read from env) ---
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    paper: bool = field(default_factory=lambda: _as_bool(os.getenv("ALPACA_PAPER"), True))

    # --- Universe & data ---
    # symbols is the FALLBACK / synthetic watchlist. In live mode the scanner
    # builds the universe dynamically (set scan_market=False to force this list).
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "SPY")
    timeframe: str = "1Day"          # 1Min, 5Min, 15Min, 1Hour, 1Day
    lookback_days: int = 400         # history pulled for indicators/backtest

    # --- Dynamic market scan ---
    scan_market: bool = True         # live: pull movers + most-active automatically
    scan_top: int = 30               # how many names to pull from each screener
    max_candidates: int = 210        # cap symbols analysed per run (fits core + wide pool + movers)
    buy_window: int = 5              # a crossover within this many days still counts as a fresh BUY/SELL
    min_price: float = 5.0           # ignore sub-$5 names
    max_atr_pct: float = 18.0        # skip hyper-volatile names (ATR > this % of price/day = penny chaos)
    max_day_move_pct: float = 25.0   # skip names that already moved more than this today (chasing/junk)
    max_momentum_pct: float = 200.0  # drop momentum-leaderboard names whose 12-1 score exceeds this — a
                                     # 12-month return >200% on a scanned large/mid-cap almost always means
                                     # a corrupt historical bar (bad IEX print) inflated the base, not real edge
    bad_bar_jump_pct: float = 50.0   # a single-day move > this that immediately reverses = bad print → drop the name
    rel_volume_window: int = 20      # days for the relative-volume (flow proxy) average
    news_per_symbol: int = 4         # headlines pulled per flagged symbol
    show_top: int = 40               # cards shown on the dashboard
    wide_universe: bool = field(default_factory=lambda: _as_bool(os.getenv("WIDE_UNIVERSE"), True))  # expanded liquid pool — more names = more signals + learning volume

    # --- Relative strength (vs benchmark) ---
    benchmark: str = "SPY"                            # name ranked against; fetched once per scan
    rs_lookbacks: tuple[int, ...] = (21, 63, 126)     # trading-day windows for blended RS
    rs_weights: tuple[float, ...] = (0.2, 0.3, 0.5)   # weight per lookback (sums to 1.0)
    rs_conviction_weight: float = 1.0                 # weight of the RS check in conviction (1.0 = one vote; raise once the attribution loop proves RS predicts wins)
    rs_pass_pct: int = 70                             # RS percentile >= this = pass (long); <= 100-this = pass (short)
    rs_fail_pct: int = 40                             # RS percentile <= this = fail (long); >= 100-this = fail (short)

    # --- Strategy params (MA crossover + RSI filter) ---
    fast_ma: int = 20
    slow_ma: int = 50
    rsi_period: int = 14

    # --- Intraday signal layer: runs the SAME engine on intraday bars over the scanned names,
    # shown in its own tab (and later nudges daily conviction). Fully gated + graceful: if the
    # intraday fetch fails/rate-limits, the daily build is untouched. ---
    intraday_enabled: bool = field(default_factory=lambda: os.getenv("INTRADAY_ENABLED", "true").lower() == "true")
    intraday_timeframe: str = "5Min"        # 1Min / 5Min / 15Min / 1Hour
    intraday_lookback_days: int = 15        # calendar days of intraday history to pull
    intraday_fast_ma: int = 9               # MA periods are in BARS at the intraday timeframe
    intraday_slow_ma: int = 20
    intraday_show_top: int = 60             # cap intraday cards shown

    # --- Opening Range Breakout (ORB) + VWAP day-trading strategy (its own learning bucket) ---
    # Built to the "stocks-in-play ORB + VWAP" brief: long-only v1, a standalone 0-100 signal score
    # with a hard threshold, a morning trade window, and hard day-trading risk caps. Kept SEPARATE
    # from the swing/intraday conviction stack — its own scoring, tracker and learning.
    orb_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("ORB_ENABLED"), True))
    orb_long_only: bool = True              # v1 is long-only (shorts deferred: borrow/locate realism)
    orb_primary_window: int = 15            # OR window (mins) used for the live signal
    orb_windows: tuple = (5, 15, 30)        # windows the backtest compares ("test several")
    orb_target_r: tuple = (1.0, 2.0)        # profit targets as R multiples (final = min reward:risk)
    orb_min_rr: float = 2.0                 # hard reward:risk floor (reject below)
    orb_score_threshold: float = 75.0       # 0-100 signal score needed to be tradable (brief default)
    orb_alert_threshold: float = 65.0       # 65-75 = alert-only band; <65 reject/watch
    orb_window_start: str = "09:45"         # primary trade window opens (ET) — after OR forms
    orb_window_end: str = "10:30"           # primary trade window closes (ET)
    orb_orw_atr_min: float = 0.3            # OR-width / ATR floor (too narrow = noise)
    orb_orw_atr_max: float = 3.0            # OR-width / ATR cap (too wide = stop too far, RR collapses)
    orb_max_spread_pct: float = 0.25        # reject if live bid/ask spread exceeds this % (IEX quote)
    orb_half_spread_bps: float = 2.0        # backtest transaction-cost model: half-spread (bps)
    orb_slippage_bps: float = 3.0           # backtest transaction-cost model: slippage per side (bps)
    # hard day-trading risk caps (brief): final authority, model confidence can't override
    orb_max_trades_per_day: int = 4         # total ORB trades/day across the book
    orb_max_concurrent: int = 3             # max simultaneous ORB positions
    orb_consec_loss_halt: int = 2           # halt the bucket after N consecutive full-stop losses
    orb_risk_pct: float = 0.005             # risk per ORB trade (fraction of equity to the stop)
    orb_show_top: int = 40                  # cap ORB cards shown
    orb_inplay_top: int = 40                # how many stocks-in-play to fetch+scan for ORB
    orb_signal_lookback_days: int = 6       # SHALLOW 5-min history pulled EVERY build (live signal+gap)
    #   The deep backtest history is fetched separately, once/day when the market is closed, so the
    #   10-min market-hours build stays fast regardless of how many names we scan.
    orb_lookback_days: int = 45             # DEEP 5-min history for the once-daily cached backtest

    # Multi-agent LLM trade committee (advisory second opinion on the top actionable signals)
    committee_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("COMMITTEE_ENABLED"), True))
    committee_max_names: int = 6            # cap names sent to the committee (one batched call)
    committee_conviction_enabled: bool = True   # the AI committee's vote counts toward conviction (and is graded by attribution)
    committee_conviction_weight: float = 1.0    # base weight of the committee check (the learned loop re-weights it from outcomes)
    # Real multi-model swarm: ask several DIFFERENT models (via OpenRouter) to each vote, then tally
    # agreement — instead of one model playing all four roles. Opt-in (needs a key + a little cost);
    # falls back to the single-model committee when off / no key.
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    committee_swarm_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("COMMITTEE_SWARM"), False))
    committee_models: tuple = ("anthropic/claude-3.5-sonnet", "deepseek/deepseek-chat",
                               "qwen/qwen-2.5-72b-instruct", "x-ai/grok-4.1-fast")

    # --- xAI / Grok: the one model with LIVE X (Twitter) + real-time web access (xai.py) ---
    # Direct xAI API (OpenAI-compatible) — needed for the server-side x_search tool that OpenRouter
    # doesn't expose. All opt-in + fail-silent; nothing runs without a key.
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    xai_model: str = field(default_factory=lambda: os.getenv("XAI_MODEL", "grok-4.1-fast"))  # cheap, real-time
    xai_daily_call_cap: int = 40            # hard per-run cap on Grok calls (bounds cost)
    xai_max_names: int = 6                  # live-sentiment only the top N actionable names per build
    # (1) live X/web social read as a self-grading conviction check
    xai_live_sentiment_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("XAI_LIVE_SENTIMENT"), True))
    xai_sentiment_conviction_weight: float = 1.0   # base weight; the learned loop re-weights it from outcomes
    # (2) pre-market catalyst sweep -> seeds the next scan via news_candidates.json
    xai_premarket_scan_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("XAI_PREMARKET_SCAN"), False))

    # --- Real-time ORB executor (always-on runner; PAPER only by default, DISABLED by default) ---
    live_executor_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("LIVE_EXECUTOR_ENABLED"), False))
    executor_allow_real: bool = field(default_factory=lambda: _as_bool(os.getenv("LIVE_EXECUTOR_ALLOW_REAL"), False))
    executor_poll_secs: int = 45            # how often the runner re-checks the watchlist for breakouts
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # --- News / market-development weighting in conviction (heavier = more news-driven). Tunable. ---
    news_conviction_weight: float = 1.6    # weight of the news-tone + news-idea checks
    catalyst_boost_weight: float = 2.0     # weight when a FRESH, aligned, HIGH-confidence catalyst is present
    sector_conviction_weight: float = 1.0  # weight of the sector-momentum check
    news_idea_candidates: bool = True      # pin LLM news-idea names into the next scan for a technical read

    # --- Refinements (tune / A-B test with backtest_compare.py) ---
    adx_min: float = 20.0           # require ADX >= this for a NEW long (trend-strength gate; 0 = off)
    trail_atr_mult: float = 0.0     # >0 enables a trailing ATR stop in the backtest (e.g. 3.0; 0 = fixed stop)
    regime_block_buys: bool = True  # demote fresh BUYs to HOLD when the market regime is Risk-off
    # Regime-weighted confluence (adapted from Scientia's REGIME_WEIGHTS): instead of counting
    # agreeing strategies flat, weight each by how well its KIND fits the tape — breakouts earn
    # full credit in a trending Risk-on tape but little in a chop/Neutral tape where they whipsaw;
    # mean-reversion is the mirror. Config-driven so the weights can be re-fit on our own results.
    regime_confluence_enabled: bool = True
    regime_kind_weights: dict = field(default_factory=lambda: {
        "Risk-on": {"trend": 1.0, "breakout": 1.0, "breakdown": 1.0, "momentum": 1.0, "mean-reversion": 0.4, "event": 0.8},
        "Neutral": {"trend": 0.6, "breakout": 0.5, "breakdown": 0.5, "momentum": 0.6, "mean-reversion": 1.0, "event": 0.8},
        "Risk-off": {"trend": 0.5, "breakout": 0.3, "breakdown": 0.3, "momentum": 0.5, "mean-reversion": 0.8, "event": 0.7},
    })

    # --- Post-earnings drift (PEAD) setup ---
    pead_enabled: bool = True       # include the post-earnings-drift strategy in confluence/backtests
    pead_window: int = 5            # bars after the reaction during which a drift entry may trigger
    pead_gap_min: float = 0.04      # reaction = close-to-close move >= this (4%) ...
    pead_vol_mult: float = 1.5      # ... AND volume >= this multiple of the median (earnings-like surge)
    pead_vol_window: int = 20       # median-volume lookback for the surge test

    # --- Position sizing by conviction & volatility (live paper path) ---
    size_by_conviction: bool = True   # scale risk by conviction tier
    conv_mult_high: float = 1.0       # High-conviction risk multiplier (of paper_risk_pct)
    conv_mult_medium: float = 0.6     # Medium
    conv_mult_low: float = 0.3        # Low
    vol_target_atr_pct: float = 4.0   # ATR% at which the vol multiplier == 1.0; higher ATR% -> smaller
    min_size_mult: float = 0.25       # floor on the combined multiplier (never size below this x base)
    # Momentum/conviction sizing tilt: size UP with the conviction SCORE so strong-momentum,
    # high-conviction names carry the most weight; a vol guardrail only throttles names that are
    # clearly hyper-volatile (ATR% above vol_guard_mult x vol_target_atr_pct).
    tilt_min: float = 0.40            # size mult at a just-actionable (~50%) conviction score
    tilt_max: float = 1.60            # size mult at a top (~100%) conviction score
    max_size_mult: float = 1.60       # hard cap on the combined size multiplier
    vol_guard_mult: float = 1.5       # only throttle when ATR% exceeds this x vol_target_atr_pct
    kelly_sizing_enabled: bool = True # scale sizing by the book's own half-Kelly fraction (clamped 0.5-1.5x)
    kelly_min_n: int = 30             # resolved trades required before Kelly influences sizing

    # --- Exit management ---
    partial_take_r: float = 1.0       # scale out half the position at this R multiple (0 = off)
    max_hold_days: int = 0            # time-stop: close after this many bars if < 1R reached (0 = off)
    manage_exits: bool = False        # LIVE only: actively amend stops / take partials (default OFF)

    # --- Optional auto paper-trading (build a REAL fills-based track record) ---
    # OFF by default. Set PAPER_TRADE=true (repo var/secret) to let the runner submit bracket
    # orders to your PAPER account for fresh High-conviction signals. Refuses on a live account.
    paper_trade: bool = field(default_factory=lambda: _as_bool(os.getenv("PAPER_TRADE"), False))
    paper_max_new_per_run: int = 10  # cap new positions opened in a single run
    paper_max_open: int = 30         # cap total simultaneous open paper positions
    #   NB: the per-order sizer also caps qty at ~90% of available BUYING POWER, so the book
    #   self-limits — it stops opening when buying power runs low, even below these caps.
    paper_risk_pct: float = 0.005    # risk per position (fraction of equity to the stop) — small so many fit
    paper_allow_shorts: bool = False  # shorts off in the current risk-on tape (−4.47% over 133 trades); re-enable in a bear regime
    # Realized-record cutoff: after a manual cleanup/reset, set PAPER_RESET_DATE=YYYY-MM-DD so the
    # realized stats only count round-trips closed on/after that date (keeps manual closes + stale
    # history out of the strategy's record). Empty = count everything (default).
    paper_reset_date: str = field(default_factory=lambda: os.getenv("PAPER_RESET_DATE", "").strip())

    # --- Portfolio risk engine + kill switch (book-level safety overlay) ---
    # The "non-negotiable overlay": per-trade stops protect a single position; THIS protects the
    # whole book. Evaluated every run before any new order. All gated by risk_engine_enabled.
    risk_engine_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("RISK_ENGINE_ENABLED"), True))
    daily_loss_limit_pct: float = 3.0   # halt NEW entries once today's P&L <= -this% of equity
    dd_derisk_pct: float = 8.0          # at this peak-to-now drawdown, halve new-position sizing
    dd_halt_pct: float = 10.0           # at this drawdown, stop opening new positions entirely
    dd_recovery_ladder_enabled: bool = True  # after a drawdown, ease size back 25->50->75->100% on clean profitable runs
    max_position_pct: float = 12.0      # no single position worth more than this % of equity
    portfolio_heat_cap_pct: float = 6.0     # flag/throttle when TOTAL open risk-to-stop exceeds this % of equity
    book_var_daily_sigma: float = 0.02      # assumed per-name 1-day volatility for the parametric open-book VaR
    book_var_diversification: float = 0.8   # correlation haircut on undiversified book VaR (1.0 = no benefit)
    corr_cluster_threshold: float = 0.75    # held names whose returns correlate above this = one bet, not many
    kill_switch_trips: int = 3          # consecutive failed/errored runs before the kill switch flips
    kill_switch_cooldown_runs: int = 3  # clean runs needed to auto-reset the kill switch
    # Losing-trade cooldown (ported from drawdown-circuit-breaker): after a run of consecutive
    # losing closed theses, throttle new risk — the book is clearly out of sync with the tape.
    # Stateless: recomputed from the track record each run, so it self-clears on the next win.
    loss_streak_derisk: int = 3         # consecutive losing trades -> halve new-position sizing
    loss_streak_halt: int = 4           # consecutive losing trades -> stop opening new positions

    # --- Pairs / mean-reversion diversifier (stat-arb on cointegrated spreads) ---
    # Trades the SPREAD between two economically-related liquid names, not direction. Best when
    # the broad tape is trendless. Gated by pairs_enabled; powers the Pairs tab. Fail-silent.
    pairs_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("PAIRS_ENABLED"), True))
    pairs_lookback: int = 90       # bars used for hedge ratio + spread z-score
    pairs_entry_z: float = 2.0     # |z| at/above which the spread is tradable
    pairs_exit_z: float = 0.5      # |z| at/below which to take the pair off (reverted)
    pairs_stop_z: float = 3.0      # |z| beyond which the relationship is treated as broken (stop)
    pairs_min_corr: float = 0.5    # min return correlation for the legs to count as related
    pairs_min_halflife: float = 2.0   # reject pairs that revert faster than this (noise)
    pairs_max_halflife: float = 40.0  # reject pairs that revert slower than this (too slow / drifting)

    # --- Walk-forward / out-of-sample validation (overfitting controls) ---
    # Optimizes the MA pair on each in-sample window, tests on the next unseen window, and reports
    # OOS-vs-IS so the backtest's headline edge can be trusted (or not). Gated; read-only.
    walkforward_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("WALKFORWARD_ENABLED"), True))
    walkforward_folds: int = 4

    # --- Macro regime → exposure (macro CONTROLS exposure; it never directly buys/sells) ---
    # A composite of VIX, yield curve, credit spreads, USD and equity breadth → risk-on/neutral/
    # risk-off, mapped to an exposure multiplier that scales position sizing. Gated; fail-silent.
    macro_regime_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("MACRO_REGIME_ENABLED"), True))
    macro_exposure_min: float = 0.50    # most defensive sizing scalar (deep risk-off)
    macro_exposure_max: float = 1.20    # most aggressive sizing scalar (strong risk-on)
    macro_exposure_base: float = 0.95   # scalar at a neutral composite (score 0)
    macro_exposure_slope: float = 0.30  # how hard the scalar swings with the composite score

    # --- Execution / liquidity gate (microstructure controls EXECUTION) ---
    # Vets whether a trade is practical: enough dollar volume to fill without moving the price.
    # Used as a soft conviction flag + a hard cap on paper sizing vs ADV. Gated; fail-silent.
    liquidity_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("LIQUIDITY_ENABLED"), True))
    min_dollar_volume: float = 5_000_000.0  # skip paper entries below ~$5M/day average turnover
    max_pct_of_adv: float = 0.02            # a single position may be at most ~2% of avg daily $ volume

    # --- Adaptive asset ranking (allocate capital to the best setups, not every signal) ---
    rank_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("RANK_ENABLED"), True))

    # --- Regime-specific weighting (raise the entry bar when the backdrop is hostile) ---
    regime_weighting_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("REGIME_WEIGHTING_ENABLED"), True))

    # --- Meta-signal model (second-opinion accept/reduce/delay/reject on every candidate) ---
    meta_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("META_ENABLED"), True))

    # --- Structured signal output + uncertainty scoring ---
    structured_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("STRUCTURED_ENABLED"), True))

    # --- Adaptive learning: reweight conviction checks by their realised win/loss edge ---
    adaptive_weights_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("ADAPTIVE_WEIGHTS_ENABLED"), True))
    adaptive_min_n: int = 12            # min decided trades per side before a check's weight adapts
    adaptive_retire_edge: float = -15.0  # a check whose pass-vs-fail edge is <= this (pp) and is well-sampled
                                         # is RETIRED (weight->0): it has proven anti-predictive, so it should
                                         # stop inflating conviction. Set to a large negative to disable retirement.

    # --- Adaptive learning: gate a whole DIRECTION the book has proven it can't trade ---
    # If, say, the short book's realised win rate is provably poor, stop surfacing NEW shorts as
    # actionable (demote to Watch) until they earn their way back. Strictly gated + kill-switch.
    adaptive_direction_gate_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("ADAPTIVE_DIRECTION_GATE"), True))
    adaptive_direction_min_n: int = 25       # min decided trades in a direction before it can be gated
    adaptive_direction_winrate: float = 35.0  # realised win% at/below which that direction is suppressed

    # --- No-trade intelligence layer (sit on your hands when conditions are poor) ---
    notrade_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("NOTRADE_ENABLED"), True))
    notrade_vix_block: float = 36.0     # VIX above this halts NEW entries (panic tape)
    notrade_perf_min_n: int = 25        # min resolved trades before model-performance can veto
    notrade_perf_winrate: float = 35.0  # below this recent win% -> stand down (deteriorating edge)
    timing_gate_enabled: bool = True    # O'Neil timing: a confirmed correction blocks new longs (FTD needed to re-arm)
    dir_gate_probation_new_setups: bool = True  # let a validated new setup (parabolic short) trade despite the blanket dir gate

    # --- Backtest realism (applied to every backtest so edges are net of costs) ---
    slippage_bps: float = 5.0          # modeled slippage per fill (5 bps = 0.05%); ~0.1% round trip
    commission_per_trade: float = 0.0  # per-fill commission (Alpaca = $0; set for other brokers)

    # --- Risk management ---
    starting_cash: float = 100_000.0
    risk_per_trade: float = 0.02     # fraction of equity risked per position
    # Evidence-based (diagnostic Jul 2026): momentum longs win 67% when NOT extended vs 40% when
    # chasing — demote extended fresh BUYs to Watch (buy bases/pullbacks, not rips).
    extension_gate_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("EXTENSION_GATE"), True))
    # A tight stop paired with a far (3-4R) target gets shaken out by noise before the target is
    # reached (our data: 19% hit / 70% stopped). When R:R exceeds the cap, WIDEN the stop to give
    # room (bounded by ATR) and hold $-risk constant by trimming size — rather than cutting targets.
    rr_stop_widen_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("RR_STOP_WIDEN"), True))
    rr_stop_widen_cap: float = 3.0        # widen the stop so R:R lands at ~this when it would exceed it
    rr_stop_widen_max_atr: float = 3.5    # never widen the stop beyond this * ATR
    stop_loss_pct: float = 0.05      # 5% below entry
    take_profit_pct: float = 0.15    # absolute CEILING on the base target (never target more than this)
    target_swing_lookback: int = 30  # bars used to find the nearest structural resistance/support
    target_atr_reach: float = 8.0    # volatility cap: a base target should be within ~8x ATR (a realistic swing move)
    max_positions: int = 5
    atr_period: int = 14             # ATR lookback for volatility-based stop
    atr_stop_mult: float = 2.0       # ATR-based stop = entry - mult * ATR

    # --- Optional AI analyst (Anthropic). Leave key blank to disable. ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    # --- Optional IBKR data (Client Portal Web API via an IBeam gateway). OFF by default. ---
    # Read-only data enrichment; never logs in / trades. See docs/IBKR_INTEGRATION.md.
    ibkr_enabled: bool = field(default_factory=lambda: os.getenv("IBKR_ENABLED", "").lower() == "true")
    ibkr_gateway_url: str = field(default_factory=lambda: os.getenv("IBKR_GATEWAY_URL", "").rstrip("/"))
    ibkr_account_id: str = field(default_factory=lambda: os.getenv("IBKR_ACCOUNT_ID", ""))
    ibkr_timeout: int = field(default_factory=lambda: int(os.getenv("IBKR_TIMEOUT", "12")))

    # --- Optional live-quote proxy (Cloudflare Worker URL). Blank = disabled. ---
    live_quotes_url: str = field(default_factory=lambda: os.getenv("LIVE_QUOTES_URL", "").rstrip("/"))

    # --- Optional research feeds (free keys). Blank = that section is skipped. ---
    finnhub_api_key: str = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY", "").strip())
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", "").strip())
    research_top: int = 25   # fetch analyst/fundamentals for the top N shown (rate-limit budget)

    def validate_for_live(self) -> None:
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
                "Copy .env.example to .env and fill in your PAPER keys."
            )


CONFIG = Config()
