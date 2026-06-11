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
    scan_top: int = 20               # how many names to pull from each screener
    max_candidates: int = 40         # cap symbols actually analysed per run
    min_price: float = 5.0           # ignore sub-$5 names
    rel_volume_window: int = 20      # days for the relative-volume (flow proxy) average
    news_per_symbol: int = 4         # headlines pulled per flagged symbol
    show_top: int = 12               # cards shown on the dashboard

    # --- Strategy params (MA crossover + RSI filter) ---
    fast_ma: int = 20
    slow_ma: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # --- Risk management ---
    starting_cash: float = 100_000.0
    risk_per_trade: float = 0.02     # fraction of equity risked per position
    stop_loss_pct: float = 0.05      # 5% below entry
    take_profit_pct: float = 0.15    # 15% above entry
    max_positions: int = 5
    atr_period: int = 14             # ATR lookback for volatility-based stop
    atr_stop_mult: float = 2.0       # ATR-based stop = entry - mult * ATR

    # --- Optional AI analyst (Anthropic). Leave key blank to disable. ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-3-5-haiku-latest"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    def validate_for_live(self) -> None:
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
                "Copy .env.example to .env and fill in your PAPER keys."
            )


CONFIG = Config()
