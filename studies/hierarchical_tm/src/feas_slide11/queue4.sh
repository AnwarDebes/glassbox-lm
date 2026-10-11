#!/usr/bin/env bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while pgrep -f "queue[23].sh" >/dev/null; do sleep 5; done
$D/launch.sh q_h_off_C2000_long --epochs 15 --n-train 40000 --n-held 4000 --n-train-eval 2000 --show 12 --max-minutes 12 --F 1024 --model hier --clauses 2000 --A 8 --K 2 --T 4000 --s 40
echo QUEUE4_DONE
