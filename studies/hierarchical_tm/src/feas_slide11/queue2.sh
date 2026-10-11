#!/usr/bin/env bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# wait for the running control chain to finish
while pgrep -f "tag ctl_ov2" >/dev/null; do sleep 5; done
C="--epochs 5 --n-train 40000 --n-held 4000 --n-train-eval 2000 --show 10 --max-minutes 9"
$D/launch.sh q_h_off_C1000      $C --F 1024 --model hier --clauses 1000 --A 8 --K 2 --T 2000 --s 40
$D/launch.sh q_h_norm_C1000     $C --F 1024 --model hier --clauses 1000 --A 8 --K 2 --T 500 --s 40 --and-group-normalization 1
$D/launch.sh q_h_allon_C1000    $C --F 1024 --model hier --clauses 1000 --A 8 --K 2 --T 500 --s 40 --binary-inference 1 --and-group-normalization 1 --constant-update-p 1 --no-clipping 1
$D/launch.sh q_h_bin_C1000_A4K4 $C --F 1024 --model hier --clauses 1000 --A 4 --K 4 --T 500 --s 40 --binary-inference 1
$D/launch.sh q_h_bin_F3072_C1000 $C --F 3072 --model hier --clauses 1000 --A 8 --K 2 --T 500 --s 40 --binary-inference 1
$D/launch.sh q_f_C4000_s100     $C --F 1024 --model flat --clauses 4000 --T 2000 --s 100
$D/launch.sh q_h_slide11_multiclass $C --F 1024 --model hier --clauses 1000 --A 8 --K 2 --T 500 --s 40 --multiclass 1
echo QUEUE2_DONE
