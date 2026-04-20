#!/usr/bin/env bash
# run_alerts.sh — Taoist CIO daily alert launcher
# Called by cron at 8:00 AM. Logs all output to logs/cron.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="$SCRIPT_DIR/logs/cron.log"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Ensure logs dir exists
mkdir -p "$SCRIPT_DIR/logs"

{
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "[$TIMESTAMP] Taoist CIO alert check starting"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Use venv python directly (avoids sourcing issues in cron's minimal shell env)
  "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/src/alerts.py"

  echo "[$TIMESTAMP] Alert check complete"
  echo ""
} >> "$LOG" 2>&1
