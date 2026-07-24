#!/bin/zsh
# Re-sync the Obsidian journal → journal_overrides.json whenever ticker notes change.
# Triggered by the launchd agent (WatchPaths). Fail-soft; logs to sync.log.
set -e
REPO="${JOURNAL_REPO:-$HOME/Desktop/trading_bot}"
VAULT="${JOURNAL_VAULT:-$HOME/Desktop/Trading Brain}"
cd "$REPO"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] change detected — syncing…" >> automation/journal/sync.log
/usr/bin/python3 journal_sync.py "$VAULT" >> automation/journal/sync.log 2>&1 || true

# --- OPTIONAL auto commit + push (leave commented until you're comfortable) ---
# git add journal_overrides.json 2>/dev/null && \
#   git commit -m "journal sync [skip ci]" 2>/dev/null && \
#   git pull --rebase --autostash origin main 2>/dev/null && \
#   git push origin main 2>/dev/null && \
#   echo "  pushed." >> automation/journal/sync.log
