#!/usr/bin/env bash
# restarts the e21 exploration queue at cap 9 after R7's last jobs have had time to start
cd "$(dirname "${BASH_SOURCE[0]}")/.."
sleep 600
exec bash scripts/master_queue.sh logs/jobs_e21.txt 9 logs/e21.out
