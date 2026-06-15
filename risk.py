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


def honest_target(entry: float, df, atr_val, working_stop: float, cfg: Config, short: bool = False):
    """An evidence-based base target rather than a flat % or an arbitrary risk multiple.

    Anchored to real STRUCTURE — the nearest level where price has actually turned (recent
    swing high for a long / swing low for a short) — then bounded by VOLATILITY (kept within
    a realistic ~Nx-ATR swing move) and capped at take_profit_pct. ``df`` must be the price
    history UP TO the entry bar (no future bars) so it is safe to use inside a backtest. The
    FUNDAMENTAL bound (analyst mean target) is layered on later by scanner._apply_fundamental_cap
    where research is available. Returns (price, basis_text).
    """
    atrm = atr_val if (atr_val and atr_val > 0) else 0.02 * entry
    lb = getattr(cfg, "target_swing_lookback", 30)
    reach = getattr(cfg, "target_atr_reach", 8.0)
    try:
        if short:
            swing, ext = float(df["low"].tail(lb).min()), float(df["low"].min())
        else:
            swing, ext = float(df["high"].tail(lb).max()), float(df["high"].max())
    except Exception:  # noqa: BLE001
        swing = ext = None

    if not short:
        cands = [x for x in (swing, ext) if x is not None and x > entry + 0.5 * atrm]
        if cands:
            tgt, basis = min(cands), "nearest resistance (recent swing high)"
        else:
            tgt, basis = entry + reach * atrm, "measured move — at new highs, no overhead resistance"
        vmax = entry + reach * atrm
        if tgt > vmax:
            tgt, basis = vmax, basis + " · trimmed to a volatility-reachable distance"
        cap = entry * (1 + cfg.take_profit_pct)
        if tgt > cap:
            tgt, basis = cap, basis + f" · capped at {round(cfg.take_profit_pct * 100)}%"
        tgt = max(tgt, entry + 0.5 * atrm)
    else:
        cands = [x for x in (swing, ext) if x is not None and x < entry - 0.5 * atrm]
        if cands:
            tgt, basis = max(cands), "nearest support (recent swing low)"
        else:
            tgt, basis = entry - reach * atrm, "measured move — at new lows, no support below"
        vmin = entry - reach * atrm
        if tgt < vmin:
            tgt, basis = vmin, basis + " · trimmed to a volatility-reachable distance"
        cap = entry * (1 - cfg.take_profit_pct)
        if tgt < cap:
            tgt, basis = cap, basis + f" · capped at {round(cfg.take_profit_pct * 100)}%"
        tgt = min(tgt, entry - 0.5 * atrm)
    return tgt, basis
