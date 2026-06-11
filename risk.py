"""Position sizing and exit levels."""
from __future__ import annotations

from config import Config


def position_size(equity: float, price: float, cfg: Config) -> int:
    """Shares to buy so that a stop-loss hit risks ~risk_per_trade of equity.

    risk_dollars = equity * risk_per_trade
    per_share_risk = price * stop_loss_pct
    shares = risk_dollars / per_share_risk  (also capped by available equity)
    """
    if price <= 0:
        return 0
    risk_dollars = equity * cfg.risk_per_trade
    per_share_risk = price * cfg.stop_loss_pct
    if per_share_risk <= 0:
        return 0
    shares = int(risk_dollars / per_share_risk)
    # never spend more than equity on a single name
    max_affordable = int(equity / price)
    return max(0, min(shares, max_affordable))


def stop_loss_price(entry: float, cfg: Config) -> float:
    return entry * (1 - cfg.stop_loss_pct)


def take_profit_price(entry: float, cfg: Config) -> float:
    return entry * (1 + cfg.take_profit_pct)
