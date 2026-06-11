"""Alpaca broker wrapper. Defaults to PAPER trading.

Safety: this module submits orders ONLY to the account tied to the keys in
your .env. Keep ALPACA_PAPER=true unless you fully understand the risk of
live trading. The author of this code does not place live orders for you.
"""
from __future__ import annotations

from config import Config


class Broker:
    def __init__(self, cfg: Config):
        cfg.validate_for_live()
        from alpaca.trading.client import TradingClient
        self.cfg = cfg
        self.client = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)

    # --- account / positions ---
    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions(self) -> dict[str, int]:
        return {p.symbol: int(float(p.qty)) for p in self.client.get_all_positions()}

    def position_qty(self, symbol: str) -> int:
        return self.positions().get(symbol, 0)

    # --- orders ---
    def submit_market(self, symbol: str, qty: int, side: str):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order)
