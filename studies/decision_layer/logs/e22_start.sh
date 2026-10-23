#!/usr/bin/env bash
# waits for the R8 router cache, then runs the e22 exploration queue at cap 9
cd "$(dirname "${BASH_SOURCE[0]}")/.."
until [ -f results/r8_router_seed1.pt ]; do sleep 20; done
sleep 5
exec bash scripts/master_queue.sh logs/jobs_e22.txt 9 logs/e22.out
