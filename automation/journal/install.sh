#!/bin/zsh
# Install the launchd agent that re-syncs your journal on every ticker-note change.
# Run once:  zsh automation/journal/install.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VAULT="${1:-$HOME/Desktop/Trading Brain}"
LABEL="com.tradingbrain.journalsync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>$REPO/automation/journal/journal_watch.sh</string></array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>JOURNAL_REPO</key><string>$REPO</string>
    <key>JOURNAL_VAULT</key><string>$VAULT</string>
  </dict>
  <key>WatchPaths</key>
  <array><string>$VAULT/Journal/Tickers</string><string>$VAULT/Journal/Trades</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
PL
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✅ Installed. Watching: $VAULT/Journal/Tickers"
echo "   Any edit to a ticker note now re-runs journal_sync automatically."
echo "   Log:       $REPO/automation/journal/sync.log"
echo "   Uninstall: launchctl unload \"$PLIST\" && rm \"$PLIST\""
