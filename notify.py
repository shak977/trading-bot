"""Alerts — ping you the moment a FRESH high-conviction signal appears.

Opt-in and multi-channel; nothing fires unless you set at least one of these env vars:
  ALERT_WEBHOOK_URL   a Discord / Slack / ntfy webhook (payload shape auto-detected)
  ALERT_NTFY_TOPIC    an ntfy.sh topic name (free phone push: install the ntfy app, subscribe)
  ALERT_EMAIL_TO      + SMTP_HOST, SMTP_USER, SMTP_PASS [, SMTP_PORT]  for email
  TELEGRAM_BOT_TOKEN  + TELEGRAM_CHAT_ID  — Telegram push (create a bot with @BotFather, then get
                      your chat id from https://api.telegram.org/bot<token>/getUpdates after messaging it)
Optional: SITE_URL    link included in the alert (e.g. https://shak977.github.io/trading-bot)

Only NEW High-conviction BUY/SHORT calls are alerted (the same bar the dashboard/digest use),
deduped via alerts_sent.json so a symbol pings once per day, not every 30-minute run. Never raises.
"""
from __future__ import annotations

import json
import os

LOG = os.getenv("ALERTS_FILE", "alerts_sent.json")


def _load() -> set:
    try:
        with open(LOG) as f:
            return set(json.load(f))
    except Exception:  # noqa: BLE001
        return set()


def _save(keys: set) -> None:
    try:
        with open(LOG, "w") as f:
            json.dump(sorted(keys), f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def alerted_today(today: str) -> set:
    """Symbols that already fired an alert today, so the dashboard can PIN them and stay
    in line with what you were notified about (keys are 'SYMBOL:ACTION:YYYY-MM-DD')."""
    out = set()
    for k in _load():
        parts = k.split(":")
        if len(parts) == 3 and parts[2] == today:
            out.add(parts[0])
    return out


def _channels() -> dict:
    return {
        "webhook": os.getenv("ALERT_WEBHOOK_URL", "").strip(),
        "ntfy": os.getenv("ALERT_NTFY_TOPIC", "").strip(),
        "email_to": os.getenv("ALERT_EMAIL_TO", "").strip(),
        "tg_token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        "tg_chat": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    }


def _any(ch: dict) -> bool:
    """True if at least one alert channel is configured."""
    return bool(ch["webhook"] or ch["ntfy"] or ch["email_to"] or (ch["tg_token"] and ch["tg_chat"]))


def _deliver(ch: dict, title: str, body: str) -> bool:
    """Send one alert to every configured channel; True if any delivered. Never raises."""
    delivered = False
    if ch["webhook"]:
        delivered = _post_webhook(ch["webhook"], title, body) or delivered
    if ch["ntfy"]:
        delivered = _post_ntfy(ch["ntfy"], title, body) or delivered
    if ch["email_to"]:
        delivered = _send_email(ch["email_to"], title, body) or delivered
    if ch["tg_token"] and ch["tg_chat"]:
        delivered = _post_telegram(ch["tg_token"], ch["tg_chat"], title, body) or delivered
    return delivered


def _post_webhook(url: str, title: str, body: str) -> bool:
    import requests
    try:
        if "discord" in url:
            payload = {"content": f"**{title}**\n{body}"}
        elif "slack" in url:
            payload = {"text": f"*{title}*\n{body}"}
        elif "ntfy" in url:  # ntfy webhook URL -> plain text body
            r = requests.post(url, data=body.encode("utf-8"),
                              headers={"Title": title, "Tags": "chart_with_upwards_trend"}, timeout=12)
            return r.ok
        else:
            payload = {"text": f"{title}\n{body}", "content": f"{title}\n{body}"}
        r = requests.post(url, json=payload, timeout=12)
        return r.ok
    except Exception:  # noqa: BLE001
        return False


def _post_ntfy(topic: str, title: str, body: str) -> bool:
    import requests
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
                          headers={"Title": title, "Tags": "chart_with_upwards_trend", "Priority": "high"},
                          timeout=12)
        return r.ok
    except Exception:  # noqa: BLE001
        return False


def _post_telegram(token: str, chat_id: str, title: str, body: str) -> bool:
    """Send via the Telegram Bot API. Set TELEGRAM_BOT_TOKEN (from @BotFather) + TELEGRAM_CHAT_ID
    (your chat/channel id). Plain text (no Markdown) so tickers, $ and / never break formatting."""
    import requests
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": f"{title}\n\n{body}", "disable_web_page_preview": True},
            timeout=12,
        )
        return r.ok
    except Exception:  # noqa: BLE001
        return False


def _send_email(to: str, title: str, body: str) -> bool:
    host, user, pw = os.getenv("SMTP_HOST", ""), os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", "")
    if not (host and user and pw):
        return False
    import smtplib
    from email.mime.text import MIMEText
    try:
        msg = MIMEText(body)
        msg["Subject"], msg["From"], msg["To"] = title, user, to
        port = int(os.getenv("SMTP_PORT", "465"))
        if port == 587:
            s = smtplib.SMTP(host, port, timeout=20); s.starttls()
        else:
            s = smtplib.SMTP_SSL(host, port, timeout=20)
        s.login(user, pw); s.sendmail(user, [to], msg.as_string()); s.quit()
        return True
    except Exception:  # noqa: BLE001
        return False


def run(signals: list[dict], today: str) -> dict | None:
    """Fire alerts for the NEW fresh BUYs the MODEL rates highest. Returns a small status dict, or
    None when no channels are configured. Never raises.

    Aug 2026 fix: alerts used to fire on "High conviction", which the record showed was inverted —
    alerted longs won 50.3% (+0.46%) while the ones that DIDN'T alert won 66.8% (+1.22%). In other
    words the ping was reliably picking the worse half. Alerts now gate on the validated meta-label
    P(win) (OOS AUC ~0.75) and fall back to conviction only when no model score is present.
    """
    ch = _channels()
    if not _any(ch):
        return None

    try:
        from config import CONFIG as _C
        floor = float(getattr(_C, "meta_pwin_floor", 0.52))
        allow_shorts = bool(getattr(_C, "allow_shorts", False))
    except Exception:  # noqa: BLE001
        floor, allow_shorts = 0.52, False

    acts = ("BUY", "SHORT") if allow_shorts else ("BUY",)

    def _ok(s):
        pw = s.get("p_win")
        if isinstance(pw, (int, float)):
            return pw >= floor                      # model-gated (preferred)
        return (s.get("conviction") or {}).get("label") == "High"   # fallback pre-model

    picks = [s for s in signals if s.get("action") in acts and _ok(s)]
    sent = _load()
    fresh = [s for s in picks if f"{s['symbol']}:{s['action']}:{today}" not in sent]
    if not fresh:
        return {"configured": True, "new": 0, "delivered": False}

    fresh.sort(key=lambda s: -(s.get("p_win") if isinstance(s.get("p_win"), (int, float))
                               else ((s.get("conviction") or {}).get("score_pct") or 0) / 100.0))
    lines = []
    for s in fresh[:10]:
        pw = s.get("p_win")
        cp = (s.get("conviction") or {}).get("score_pct")
        grade = (f"{pw*100:.0f}% win-probability" if isinstance(pw, (int, float))
                 else f"{cp}% conviction")
        entry = (s.get("plan") or {}).get("entry")
        arrow = "🟢" if s["action"] == "BUY" else "🔴"
        lines.append(f"{arrow} {s['symbol']} {s['action']} — {grade}"
                     + (f", entry ${entry:,.2f}" if entry else ""))
    site = os.getenv("SITE_URL", "").strip()
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
    title = f"{len(fresh)} new top-rated signal{'s' if len(fresh) != 1 else ''}"
    foot = f"\n\nas of {stamp} — these names are pinned on the dashboard"
    body = "\n".join(lines) + foot + (f"\n{site}" if site else "")

    delivered = _deliver(ch, title, body)

    if delivered:
        for s in fresh:
            sent.add(f"{s['symbol']}:{s['action']}:{today}")
            s["alerted"] = True      # so the tracker can grade the ALERTS themselves, not just signals
        _save(sent)
    return {"configured": True, "new": len(fresh), "delivered": delivered,
            "symbols": [s["symbol"] for s in fresh[:10]]}


def run_pairs(pairs_data: dict | None, today: str) -> dict | None:
    """Fire alerts for FRESH actionable pairs (a spread stretched to its entry band, |z| ≥ 2σ).
    Deduped via the same log with a 'PAIR:' key so each pair pings once per day. Never raises."""
    ch = _channels()
    if not _any(ch):
        return None
    pairs = (pairs_data or {}).get("pairs") or []
    actionable = [p for p in pairs if p.get("actionable")]
    if not actionable:
        return {"configured": True, "new": 0, "delivered": False}

    sent = _load()
    fresh = [p for p in actionable
             if f"PAIR:{p['a']}/{p['b']}:{p['signal']}:{today}" not in sent]
    if not fresh:
        return {"configured": True, "new": 0, "delivered": False}

    fresh.sort(key=lambda p: -abs(p.get("z") or 0))
    lines = []
    for p in fresh[:8]:
        # spread rich (z>0) => short the spread; cheap (z<0) => long the spread
        arrow = "🔴" if p["signal"] == "SHORT_SPREAD" else "🟢"
        legs = (f"short {p['a']} / long {p['b']}" if p["signal"] == "SHORT_SPREAD"
                else f"long {p['a']} / short {p['b']}")
        lines.append(f"{arrow} {p['a']}/{p['b']} {p['z']:+.1f}σ — {legs} (β {p.get('beta')})")
    site = os.getenv("SITE_URL", "").strip()
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
    title = f"{len(fresh)} pair spread{'s' if len(fresh) != 1 else ''} at entry band"
    foot = (f"\n\nas of {stamp} — mean-reversion: enter at ±2σ, exit toward 0, stop past ±3σ. "
            "Market-neutral diversifier; paper/educational.")
    body = "\n".join(lines) + foot + (f"\n{site}" if site else "")

    delivered = _deliver(ch, title, body)

    if delivered:
        for p in fresh:
            sent.add(f"PAIR:{p['a']}/{p['b']}:{p['signal']}:{today}")
        _save(sent)
    return {"configured": True, "new": len(fresh), "delivered": delivered,
            "pairs": [f"{p['a']}/{p['b']}" for p in fresh[:8]]}


def run_orb(orb_signals: list[dict] | None, today: str) -> dict | None:
    """Fire alerts for FRESH ELIGIBLE ORB breakouts (score ≥ threshold, not risk-blocked). Deduped
    via the same log with an 'ORB:' key so each name pings once per session. Never raises."""
    ch = _channels()
    if not _any(ch):
        return None
    picks = [s for s in (orb_signals or [])
             if s.get("recommended_action") == "paper_trade" and not s.get("risk_blocked")]
    if not picks:
        return {"configured": True, "new": 0, "delivered": False}

    sent = _load()
    fresh = [s for s in picks if f"ORB:{s.get('symbol')}:{s.get('session', today)}" not in sent]
    if not fresh:
        return {"configured": True, "new": 0, "delivered": False}

    fresh.sort(key=lambda s: -(s.get("score") or 0))
    lines = []
    for s in fresh[:8]:
        arrow = "🟢" if s.get("direction") == "LONG" else "🔴"
        lines.append(f"{arrow} {s.get('symbol')} {s.get('direction')} ORB — score {s.get('score')}, "
                     f"entry ${s.get('entry'):,.2f}, stop ${s.get('stop'):,.2f}, "
                     f"target ${s.get('target'):,.2f} (RR {s.get('rr')})")
    site = os.getenv("SITE_URL", "").strip()
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
    title = f"{len(fresh)} ORB breakout{'s' if len(fresh) != 1 else ''} (stocks in play)"
    foot = (f"\n\nas of {stamp} — opening-range breakout + VWAP, day-trade (flatten by 15:45 ET). "
            "Paper/shadow signal, not advice.")
    body = "\n".join(lines) + foot + (f"\n{site}" if site else "")

    delivered = _deliver(ch, title, body)

    if delivered:
        for s in fresh:
            sent.add(f"ORB:{s.get('symbol')}:{s.get('session', today)}")
        _save(sent)
    return {"configured": True, "new": len(fresh), "delivered": delivered,
            "symbols": [s.get("symbol") for s in fresh[:8]]}


def send_test() -> bool:
    """Fire a one-off dummy alert to every configured channel, to confirm the wiring end-to-end.
    Returns True if at least one channel delivered. Run via: python notify.py --test"""
    ch = _channels()
    on = [k for k in ("webhook", "ntfy", "email_to") if ch[k]] + (["telegram"] if ch["tg_token"] and ch["tg_chat"] else [])
    if not on:
        print("No alert channels configured. Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID "
              "(or ALERT_WEBHOOK_URL / ALERT_NTFY_TOPIC / ALERT_EMAIL_TO) first.")
        return False
    site = os.getenv("SITE_URL", "").strip()
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (f"If you're reading this, your trading-bot alerts are wired up correctly. "
            f"This is a manual test — not a real signal.\n\n{stamp}" + (f"\n{site}" if site else ""))
    ok = _deliver(ch, "✅ Trading-bot test alert", body)
    print(f"channels tried: {', '.join(on)} | delivered: {ok}")
    # Verbose Telegram diagnostic — surface the API's own words so failures are obvious.
    if ch["tg_token"] and ch["tg_chat"]:
        import requests
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{ch['tg_token']}/sendMessage",
                json={"chat_id": ch["tg_chat"], "text": "Trading-bot test ✅"}, timeout=12)
            data = r.json()
            if data.get("ok"):
                print("TELEGRAM: OK — message sent. Check your Telegram.")
                ok = True
            else:
                code = data.get("error_code")
                desc = data.get("description", "")
                print(f"TELEGRAM: FAILED — {code}: {desc}")
                if code == 400 and "chat not found" in desc.lower():
                    print("  -> TELEGRAM_CHAT_ID is wrong. Re-grab the number from RawDataBot (the 'id' under 'from').")
                elif code == 403:
                    print("  -> Open your bot @Shak97_bot in Telegram and press Start / send it a message first.")
                elif code == 401:
                    print("  -> TELEGRAM_BOT_TOKEN is wrong. Re-copy it from BotFather (/mybots -> API Token).")
        except Exception as e:  # noqa: BLE001
            print(f"TELEGRAM: request error — {e}")
    return ok


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        ok = send_test()
        print("\nRESULT:", "DELIVERED ✅" if ok else "NOT DELIVERED ❌ (see the message above for why)")
        raise SystemExit(0)          # always exit clean — this is a diagnostic, read the output above
    print("usage: python notify.py --test   (sends a dummy alert to your configured channels)")
