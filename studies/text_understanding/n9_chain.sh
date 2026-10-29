#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# N9 (EXPLORATORY, carved val only; plan LOG.md 12:49:05Z, accepted 12:49Z; rules in LOG.md before
# any run): the combined view's own setting search, the fair re-test of "the meaning bits add over bag of words".
# One lane. Waits until (1) W2 part 2 is over (w2/runs_w2p2.log holds 'part 2 done' or a STOP line, and no
# w2_part2_chain.sh runs) and (2) the decision_layer e74 is over (the file .n9/e74_over exists; it is made when
# e74 is over (its text names the source)). Then, in order: the check job (TREC bow seed 1 at
# N7's setting with --check-n8: the val curve must equal N8's row, else STOP), n9s0, n9s1, n9s2, n9s3 (each job list
# written by n9_select.py from the rows at that point), then n9_select.py words > n9_words.txt (a new name if that file
# exists). Before every job: a wait while .n9/pause exists (W2 part 3 goes first once its rules are checked), then the
# hash checks (code, caches, and results_n8_sweep.jsonl as it was when N9 started). A failed job (exit not 0, not 3)
# is retried once after 10 minutes; a job that fails twice is left without a row (n9_select.py names it).
# Log runs_n9.log; rows results_n9_sweep.jsonl.
emb=${REPO_ROOT}/studies/text_understanding
dl=${REPO_ROOT}/studies/decision_layer/src/dl
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$emb/runs_n9.log
cl=$emb/.n9
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $emb/n9_sweep.py 7e4a840e81b0553d; check $emb/n9_select.py 52d0900ee8808b8e
  check $dl/tm.py f649043f5801605b; check $dl/tm_init.py 557ff69706529738
  check $emb/emb_n7/trec__explore.npz b230a81527db270d; check $emb/emb_n7/mr__explore.npz 4bb86bf70498b6c7
  check $emb/w1/feats_n7/ppw_S1000_d500_e0.5/trec.npz 78f1c5ac7b3a8016
  check $emb/w1/feats_n7/ppw_S1000_d500_e0.5/mr.npz 7570b1d89fb8e63e
  check $emb/results_n8_sweep.jsonl "$n8h"; }
w2_over() { grep -q 'part 2 done\|STOP:' $emb/w2/runs_w2p2.log 2>/dev/null; }
w2_running() { pgrep -f "glassbox-lm/studies/text_understanding/w2/w2_part2_chain.sh" > /dev/null; }
runjob() { # $1 = job line
  local ds stage view seed clauses T s negation batch neg maxlit extra="" try e
  IFS='|' read -r ds stage view seed clauses T s negation batch neg maxlit <<< "$1"
  [ "$stage" = n9check ] && extra="--check-n8"
  for try in 1 2; do
    while [ -e "$cl/pause" ]; do sleep 60; done
    checks
    echo "$(ts) start (try $try) $1 $extra" >> "$log"
    "$gpu" shared -- "$py" -I -B $emb/n9_sweep.py --dataset "$ds" --stage "$stage" --view "$view" --seed "$seed" \
      --clauses "$clauses" --T "$T" --s "$s" --negation "$negation" --batch "$batch" --neg "$neg" --max-literals "$maxlit" \
      $extra < /dev/null >> "$log" 2>&1
    e=$?
    echo "$(ts) end exit $e (try $try)" >> "$log"
    [ $e -eq 3 ] && stop "the check run does not reproduce N8's val curve ($1)"
    [ $e -eq 0 ] && return 0
    [ $try = 1 ] && sleep 600
  done
  [ "$stage" = n9check ] && stop "the check run failed twice ($1)"
  echo "$(ts) failed twice, no row: $1" >> "$log"
  return 0
}
stage() { # $1 = stage name
  "$py" -I -B $emb/n9_select.py "$1" < /dev/null > "$cl/jobs_$1.txt" 2>> "$log" || stop "n9_select.py $1 failed"
  echo "$(ts) wrote $(wc -l < "$cl/jobs_$1.txt") jobs for $1" >> "$log"
  local job
  while IFS= read -r job; do runjob "$job"; done < "$cl/jobs_$1.txt"
}
mkdir -p "$cl"
if [ "$1" = --dry ]; then
  w2_over && echo "dry: W2 part 2 over" || echo "dry: would wait for W2 part 2"
  w2_running && echo "dry: w2_part2_chain.sh running" || echo "dry: no w2_part2_chain.sh"
  [ -e "$cl/e74_over" ] && echo "dry: e74 over ($(cat "$cl/e74_over"))" || echo "dry: would wait for e74"
  [ -e "$cl/pause" ] && echo "dry: paused" || echo "dry: not paused"
  exit 0
fi
until w2_over; do sleep 60; done
until ! w2_running; do sleep 30; done
echo "$(ts) W2 part 2 is over: $(grep -m1 'part 2 done\|STOP:' $emb/w2/runs_w2p2.log)" >> "$log"
until [ -e "$cl/e74_over" ]; do sleep 60; done
echo "$(ts) e74 is over: $(head -c 300 "$cl/e74_over")" >> "$log"
n8h=$(sha256sum $emb/results_n8_sweep.jsonl | cut -c1-16)
echo "$(ts) results_n8_sweep.jsonl at the start: $n8h" >> "$log"
for st in check n9s0 n9s1 n9s2 n9s3; do stage $st; done
out=$emb/n9_words.txt
[ -e "$out" ] && out=$emb/n9_words_$(date -u +%Y%m%dT%H%M%SZ).txt
"$py" -I -B $emb/n9_select.py words < /dev/null > "$out" 2>> "$log" || stop "n9_select.py words failed"
cat "$out" >> "$log"
echo "$(ts) N9 done" >> "$log"
