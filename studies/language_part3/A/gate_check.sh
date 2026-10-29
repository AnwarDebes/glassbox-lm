#!/bin/bash
# Reads one line of decision_layer scripts/cpu_load.py output on stdin, e.g.
#   2026-10-21T03:00:00Z cores 3.45 over 10.0 s; throttled 0 of 100 periods
# and exits 0 only when at most 4.0 cores are in use and 0 periods were throttled (the Part 3 CPU gate).
read -r line
cores=$(echo "$line" | awk '{for (i = 1; i < NF; i++) if ($i == "cores") print $(i + 1)}')
thr=$(echo "$line" | awk '{for (i = 1; i < NF; i++) if ($i == "throttled") print $(i + 1)}')
awk -v c="$cores" -v t="$thr" 'BEGIN { exit !(c ~ /^[0-9.]+$/ && c + 0 <= 4.0 && t == "0") }'
