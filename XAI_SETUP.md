# xAI (Grok) integration

Grok is the one model in the stack with **live access to X (Twitter) and the real-time web**. This
adds it three ways, all opt-in and fail-silent (nothing runs, and nothing breaks, without a key):

1. **Committee voter** — Grok joins the multi-model committee (via OpenRouter) as one more
   accept / reduce / reject vote. Already wired into `committee_models`; only active when the swarm
   committee is on (`COMMITTEE_SWARM=true` + an OpenRouter key).
2. **Live-X sentiment check** — for the top actionable names each build, Grok does a real-time
   `x_search` and returns a structured social/news read. It enters conviction as the check
   **"Live X sentiment on side?"**, so the attribution loop grades it and self-weights it from real
   outcomes — it earns its influence like every other check.
3. **Pre-market catalyst sweep** — an overnight X/news scan for fresh movers; those symbols get
   pinned into the next scan (via `news_candidates.json`) for a full technical + episodic read.

## Setup (about 3 minutes)

### 1. Get an xAI API key
Create a key at the xAI console (console.x.ai → API keys). The live-search tools need the **direct
xAI key** — OpenRouter carries Grok for the committee vote, but not the `x_search` tool.

### 2. Add it as a GitHub **Secret**
Repo → Settings → Secrets and variables → Actions → **Secrets** → New repository secret:
- Name: `XAI_API_KEY`
- Value: your xAI key

### 3. Turn features on with **Variables** (same page → Variables tab)
- `XAI_LIVE_SENTIMENT` = `true` — enables the live-X conviction check
- `XAI_PREMARKET_SCAN` = `true` — enables the overnight catalyst sweep

That's it — push nothing else; the workflow already passes these through. The committee voter needs
no new flag (it rides the existing `COMMITTEE_SWARM`).

## Cost control
- `xai_model` defaults to **grok-4.1-fast** (cheapest real-time tier, ~$0.20/$0.50 per 1M tokens).
- `xai_daily_call_cap` (default **40**) hard-caps Grok calls per build.
- `xai_max_names` (default **6**) limits live-sentiment to the top few actionable names.
- Server-side searches bill per call, so keep the caps conservative until you've seen a bill.
- xAI offers free API credits via their data-sharing program — worth enabling while testing.

## How to tell it's working
- Signal cards on the top names show a **"Live X sentiment on side?"** check with a live read.
- After a couple of weeks, the **Attribution** view shows whether that check is earning weight (i.e.
  whether Grok's live-social read actually predicts winners) — if it doesn't, it self-retires.

## Turn it off
Set `XAI_LIVE_SENTIMENT` / `XAI_PREMARKET_SCAN` to `false` (or delete the secret). Everything falls
back cleanly to the pre-Grok behaviour.
