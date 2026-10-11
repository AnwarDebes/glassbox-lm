#!/usr/bin/env bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while pgrep -f "queue2.sh" >/dev/null; do sleep 5; done
C="--epochs 5 --n-train 40000 --n-held 4000 --n-train-eval 2000 --show 10 --max-minutes 9"
$D/launch.sh q_inter_hier_C200 $C --F 1024 --input-mode inter --model hier --clauses 200 --A 8 --K 1 --T 100 --s 10 --binary-inference 1
$D/launch.sh q_inter_flat_C200 $C --F 1024 --input-mode inter --model flat --clauses 200 --T 100 --s 10
echo QUEUE3_DONE
