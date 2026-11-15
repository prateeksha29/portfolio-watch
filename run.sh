#!/usr/bin/env bash
# Entry point for cron (6 am, 2 pm, 6 pm):  0 6,14,18 * * *  /path/to/portfolio-watch/run.sh
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m fin_news run 2> logs/cron.err >/dev/null  # cron.err: last run only; full history in logs/run.log
