# IBKR gateway — step-by-step runbook (beginner-friendly)

Goal: a small always-on server that stays logged into IBKR and serves data, which your
dashboard build reaches over a secure SSH tunnel. No domain name or TLS certificates needed.

You'll do this once. When anything errors, copy the message to me and I'll tell you the next step.

Time: ~1–2 hours. Cost: ~$5–6/month for the server + IBKR market-data subscriptions.

---

## Before you start — accounts you need

1. **IBKR account** with API turned on:
   - Log in at interactivebrokers.com → User Settings → **API** → enable.
   - Subscribe to market data: **Settings → Market Data Subscriptions** (US equities real-time;
     OPRA for options; CME/ICE for futures). Delayed data is cheaper if you want to start light.
   - Recommended: create a **second username** under your account just for the gateway (so it
     doesn't fight with you logging into the app). IBKR → Settings → Users.
2. **A VPS provider** — pick one with a simple dashboard:
   - Hetzner Cloud (cheapest, ~€4/mo) or DigitalOcean ($6/mo). Either is fine.
3. **Your IBKR phone app** (IB Key) for approving the login 2FA.

---

## Step 1 — Create the server

1. In your VPS provider, create a new server (a "droplet"/"instance"):
   - Image: **Ubuntu 24.04**
   - Size: smallest with **2 GB RAM** (1 vCPU is fine)
   - Add your **SSH key** if you have one, or set a password (we'll make a key in Step 2).
2. Note the server's **public IP address** (e.g. `203.0.113.10`).

## Step 2 — Get an SSH key (so the build can connect securely)

On your **Mac**, in Terminal:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ibkr_vps -N ""      # makes ~/.ssh/ibkr_vps (private) + .pub
cat ~/.ssh/ibkr_vps.pub                              # copy this whole line
```

Add that **.pub** line to the server: in the VPS dashboard paste it into the server's SSH keys,
or run `ssh-copy-id -i ~/.ssh/ibkr_vps.pub root@YOUR_SERVER_IP`.

Test you can log in:

```bash
ssh -i ~/.ssh/ibkr_vps root@YOUR_SERVER_IP
```

## Step 3 — Install Docker on the server

Once logged into the server (the prompt changes to `root@...`):

```bash
curl -fsSL https://get.docker.com | sh
docker --version          # should print a version
```

## Step 4 — Run IBeam (keeps IBKR logged in)

Still on the server:

```bash
mkdir -p ~/ibkr && cd ~/ibkr
nano docker-compose.yml        # paste the file I generated (ibkr/docker-compose.yml), save with Ctrl+O, Enter, Ctrl+X
nano .env                      # put your gateway-login username/password here (see below)
docker compose up -d           # starts IBeam in the background
docker compose logs -f         # watch it log in — approve the 2FA on your phone when prompted
```

Your `.env` on the server (NEVER put this in the GitHub repo or paste it to me):

```
IBEAM_ACCOUNT=your_gateway_username
IBEAM_PASSWORD=your_gateway_password
```

When the logs say it's **authenticated**, test the gateway is answering (on the server):

```bash
curl -k https://localhost:5000/v1/api/iserver/auth/status
```

You should get JSON with `"authenticated": true`. If `false`, re-check the 2FA approval.

## Step 5 — Let the dashboard build reach it (SSH tunnel)

Your GitHub build will open a private SSH tunnel to the server and talk to it on `localhost:5000`.
You give GitHub three secrets (repo → **Settings → Secrets and variables → Actions → New secret**):

| Secret name        | Value                                             |
|--------------------|---------------------------------------------------|
| `IBKR_SSH_KEY`     | contents of `~/.ssh/ibkr_vps` (the **private** key)|
| `IBKR_SSH_HOST`    | your server IP                                    |
| `IBKR_GATEWAY_URL` | `https://localhost:5000/v1/api`                   |

And one repo **variable**: `IBKR_ENABLED` = `true` (set this only when you're ready to go live).

I'll add the workflow step that opens the tunnel before the build runs — that part is code, so I
do it, you just add the secrets above.

## Step 6 — Turn it on, check the light

Once the secrets are set and `IBKR_ENABLED=true`, the next build will try the gateway. Open the
dashboard → **System tab** → look for the **IBKR** health row (I'll wire this in Phase 1). Green =
connected. Red = it tells you what's wrong (gateway down, not authenticated, etc.).

---

## Keeping it running

- IBKR forces a **re-login** periodically (often daily, plus a weekly hard reset). IBeam handles
  most of it automatically; occasionally you'll approve 2FA on your phone again.
- If the System tab goes red: SSH into the server, `cd ~/ibkr`, `docker compose logs -f` to see why,
  `docker compose restart` to bounce it.

## Phase 0 alternative — try it on your Mac first (no VPS, no cost)

Want to see IBKR data flow before committing to a server? Run IBeam on your Mac the same way
(Docker Desktop + the same compose file), then `python ibkr.py` after we point it at
`https://localhost:5000/v1/api`. It won't feed the *cloud* build (the cloud can't reach your Mac),
but it proves the data works and teaches you the moving parts. Tell me and I'll set it up.

---

## When you get stuck

Copy me the exact command you ran and the error/output. The usual snags: 2FA not approved, market
data not subscribed (data calls return empty), or the SSH key pasted with a missing line. All
quick to fix once I see the message.
