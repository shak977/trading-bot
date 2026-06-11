#!/usr/bin/env bash
# Weekly signal run. Generates dashboard.html + signals.json and logs the run.
# Schedule this (cron / launchd / Task Scheduler) to run once a week.
set -euo pipefail
cd "$(dirname "$0")"

# Use a local venv if present, else system python3.
if [ -d ".venv" ]; then source .venv/bin/activate; fi

mkdir -p logs
ts="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$ts] running weekly signals..." >> logs/weekly.log
python3 dashboard.py >> logs/weekly.log 2>&1
echo "[$ts] done -> $(pwd)/dashboard.html" >> logs/weekly.log
