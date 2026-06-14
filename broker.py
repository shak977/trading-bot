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

    def account(self):
        """Raw account object (equity, last_equity, buying_power, cash, ...)."""
        return self.client.get_account()

    def is_open(self) -> bool:
        try:
            return bool(self.client.get_clock().is_open)
        except Exception:  # noqa: BLE001
            return False

    def positions_detail(self) -> list[dict]:
        """Open positions with live unrealized P&L (straight from Alpaca)."""
        out = []
        for p in self.client.get_all_positions():
            out.append({
                "symbol": p.symbol,
                "qty": int(float(p.qty)),
                "side": getattr(p, "side", None).value if getattr(p, "side", None) else None,
                "avg_entry": round(float(p.avg_entry_price), 2),
                "price": round(float(p.current_price), 2) if p.current_price else None,
                "market_value": round(float(p.market_value), 2) if p.market_value else None,
                "unrealized_pl": round(float(p.unrealized_pl), 2) if p.unrealized_pl else 0.0,
                "unrealized_plpc": round(float(p.unrealized_plpc) * 100, 2) if p.unrealized_plpc else 0.0,
            })
        return out

    def portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> dict | None:
        """Equity curve for the paper account (for the dashboard chart)."""
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            h = self.client.get_portfolio_history(
                GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
            ts = list(getattr(h, "timestamp", []) or [])
            eq = list(getattr(h, "equity", []) or [])
            pts = [{"t": int(t) * 1000, "v": round(float(v), 2)}
                   for t, v in zip(ts, eq) if v is not None]
            return {"points": pts, "base_value": float(getattr(h, "base_value", 0) or 0)}
        except Exception:  # noqa: BLE001
            return None

    def open_client_ids(self) -> set[str]:
        """client_order_ids of all currently open/recent orders — for idempotency."""
        ids = set()
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200)):
                if o.client_order_id:
                    ids.add(o.client_order_id)
        except Exception:  # noqa: BLE001
            pass
        return ids

    def fills(self, after: str | None = None, max_pages: int = 30) -> list[dict]:
        """Every real execution (FILL activity) on the account, oldest-first.

        Each row: symbol, side ('buy'/'sell'/'sell_short'), qty, price, time (ISO),
        order_id. This is the ground truth for closed round-trips and realized P&L.
        Paginates /account/activities/FILL (100/page). Never raises -> partial/[] on error.
        """
        out: list[dict] = []
        try:
            page_token = None
            for _ in range(max_pages):  # safety cap (~3k fills)
                params: dict = {"direction": "asc", "page_size": 100}
                if after:
                    params["after"] = after
                if page_token:
                    params["page_token"] = page_token
                batch = self.client.get("/account/activities/FILL", params)
                if not batch:
                    break
                for a in batch:
                    out.append({
                        "symbol": a.get("symbol"),
                        "side": (a.get("side") or "").lower(),
                        "qty": float(a.get("qty") or 0),
                        "price": float(a.get("price") or 0),
                        "time": a.get("transaction_time"),
                        "order_id": a.get("order_id"),
                    })
                if len(batch) < 100:
                    break
                page_token = batch[-1].get("id")
                if not page_token:
                    break
        except Exception:  # noqa: BLE001
            pass
        return out

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

    def submit_bracket(self, symbol: str, qty: int, side: str,
                       stop: float, target: float, client_order_id: str):
        """Market entry with attached stop-loss + take-profit (one OCO bracket).
        side 'buy' opens a long (stop below, target above); 'sell' opens a short (inverted)."""
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        from alpaca.trading.requests import (MarketOrderRequest, TakeProfitRequest, StopLossRequest)
        order = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(float(target), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop), 2)),
            client_order_id=client_order_id,
        )
        return self.client.submit_order(order)
