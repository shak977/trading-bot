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
    if val is None:
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
    max_candidates: int = 90         # cap symbols actually analysed per run
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
    wide_universe: bool = False      # include the expanded liquid pool (S&P-500-ish) in live scans

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
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # --- Refinements (tune / A-B test with backtest_compare.py) ---
    adx_min: float = 20.0           # require ADX >= this for a NEW long (trend-strength gate; 0 = off)
    trail_atr_mult: float = 0.0     # >0 enables a trailing ATR stop in the backtest (e.g. 3.0; 0 = fixed stop)
    regime_block_buys: bool = True  # demote fresh BUYs to HOLD when the market regime is Risk-off

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

    # --- Exit management ---
    partial_take_r: float = 1.0       # scale out half the position at this R multiple (0 = off)
    max_hold_days: int = 0            # time-stop: close after this many bars if < 1R reached (0 = off)
    manage_exits: bool = False        # LIVE only: actively amend stops / take partials (default OFF)

    # --- Optional auto paper-trading (build a REAL fills-based track record) ---
    # OFF by default. Set PAPER_TRADE=true (repo var/secret) to let the runner submit bracket
    # orders to your PAPER account for fresh High-conviction signals. Refuses on a live account.
    paper_trade: bool = field(default_factory=lambda: _as_bool(os.getenv("PAPER_TRADE"), False))
    paper_max_new_per_run: int = 4   # cap new positions opened in a single run
    paper_max_open: int = 15         # cap total simultaneous open paper positions
    paper_risk_pct: float = 0.02     # risk per position (fraction of equity to the stop)
    paper_allow_shorts: bool = True  # also open shorts (set False for longs-only)

    # --- Backtest realism (applied to every backtest so edges are net of costs) ---
    slippage_bps: float = 5.0          # modeled slippage per fill (5 bps = 0.05%); ~0.1% round trip
    commission_per_trade: float = 0.0  # per-fill commission (Alpaca = $0; set for other brokers)

    # --- Risk management ---
    starting_cash: float = 100_000.0
    risk_per_trade: float = 0.02     # fraction of equity risked per position
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
