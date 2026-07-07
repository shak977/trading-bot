# Signal Desk — locked design spec (v1)

The agreed visual system to port into `dashboard.py`. Reference mockup: `signal-desk-drift-faint-3.html` (variant **F2 / Softer**).

## Direction
Warm-gold **glassmorphism** on a dark ambient ground. Elevated frosted panels floating over a slowly **drifting** warm glow (F2 = softer/faint). Dense but premium; a real desk, not a template. Dark-first.

## Tokens (dark)
```
--bg        #050506   (page)         --acc    #eaa62b  (warm gold)
--amb-ground#0b0a08   (behind glass) --acc2   #ffcf72
--txt       #e9ecf2                  --up     #22c98a  (long / +)
--txt2      #c2c7d2                  --dn     #f0596b  (short / −)
--mut       #868c9a
--glass-bg  rgba(255,255,255,.045)   --glass-blur 18px
--glass-bd  rgba(255,255,255,.09)    --glass-2  rgba(255,255,255,.03)
--shadow    0 12px 32px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06)
```
Fonts: **Manrope** (UI/headings), **JetBrains Mono** (all numbers/tickers/tables), tabular-nums.
Radius: 14px panels, 10px controls. Green/red reserved for direction/PnL only; gold = brand/charts/highlights.

## Ambient glow (F2)
Two warm radial blobs behind the glass: `b1` opacity **.34**, `b2` **.22**; a diagonal light **sheen** sweeps across panels at opacity **.055**. Blobs `drift` (translate) on 24s / 30s ease-in-out loops. Respect `prefers-reduced-motion` (hold still).

## Components (reusable)
- **App shell**: left **side-tab** rail (icon + label, active = gold left-glow), brand mark (gold gradient), "Active signals" holdings list (logo + conviction), pinned stat footer.
- **Glass panel** (`.glass`): frosted card, lit top edge, soft shadow, rounded.
- **Top bar**: page title, ticker **tape**, live clock + regime, search.
- **KPI row**: 6 glass stat tiles.
- **Data table** (`.dt`): mono, right-aligned nums, logo+ticker first col, conviction bar, direction chip, per-row sparkline.
- **Sector heat**: colored tiles (green→red).
- **Lead chart panel**: houses the **real** `chart_engine.js` candlestick chart + all toggles (range / type / benchmark / zoom-pan) — NOT the mockup sparkline.
- Logos: `assets.parqet.com/logos/symbol/{TICKER}` with monogram fallback.

## Multi-page — design each page in this system
Primary: Signals · Markets · Portfolio · Intel · News · System · About.
Sub-pages: Momentum · Intraday · ORB · Pairs · Paper · Track record · Analyst · All-Weather · IPOs · Heatmap.
Each keeps its content but adopts the shell + glass panels + tokens. Charts everywhere keep their interactivity.

## Port plan (phased, into dashboard.py)
1. **Foundation** — new token set + ambient glow + `.glass` panel base + fonts. (dark-first)
2. **Shell** — convert top-nav → side-tab rail; brand; holdings; top bar/tape/KPI.
3. **Signals page** — lead chart (real engine) + ranked table + rail (internals, sector heat).
4. Roll the system across remaining pages one at a time.
5. Keep the light theme as a secondary (optional) after dark is locked.
Verify each phase (ast.parse, offline render, screenshots) before the next.
