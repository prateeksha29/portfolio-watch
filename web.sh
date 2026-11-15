#!/usr/bin/env bash
# Keeps the web app running. Cron calls this every 5 minutes and at boot; the lock makes extra calls exit at once.
#   */5 * * * *  /path/to/portfolio-watch/web.sh
set -euo pipefail
cd "$(dirname "$0")"
exec flock -n data/web.lock .venv/bin/python -m fin_news.web >> logs/web.log 2>&1
