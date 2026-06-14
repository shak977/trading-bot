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


def risk_multiplier(conviction_label: str | None, atr_pct: float | None, cfg: Config) -> float:
    """Combined conviction x volatility multiplier on the base risk budget.

    Conviction tier scales the bet up for stronger setups; the vol term targets a constant
    dollar-volatility by shrinking size as ATR% rises above ``vol_target_atr_pct``. The result
    is clamped to ``[min_size_mult, 1.0]`` — it only ever throttles risk below the base ceiling,
    never above it. Returns 1.0 when ``size_by_conviction`` is off.
    """
    if not getattr(cfg, "size_by_conviction", True):
        return 1.0
    conv = {"High": cfg.conv_mult_high, "Medium": cfg.conv_mult_medium,
            "Low": cfg.conv_mult_low}.get(conviction_label or "", cfg.conv_mult_medium)
    if atr_pct and atr_pct > 0 and cfg.vol_target_atr_pct > 0:
        vol = cfg.vol_target_atr_pct / atr_pct
    else:
        vol = 1.0
    mult = conv * vol
    return max(cfg.min_size_mult, min(mult, 1.0))


def stop_loss_price(entry: float, cfg: Config) -> float:
    return entry * (1 - cfg.stop_loss_pct)


def take_profit_price(entry: float, cfg: Config) -> float:
    return entry * (1 + cfg.take_profit_pct)
