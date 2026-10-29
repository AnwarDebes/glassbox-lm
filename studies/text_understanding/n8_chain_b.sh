#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# N8 (EXPLORATORY, carved val only; rules in LOG.md before any run): TM setting search on N7's features. Lane A writes
# each stage's job list with n8_select.py (s1, then s2 and s3 from the rows), lanes A and B take the jobs by mkdir
# claims; the N7-setting jobs run first with --check-n7 (the val curve must equal N7's registered run, else STOP).
# Code and cache hashes are checked before every run. Shared GPU lock. Log runs_n8b.log (this copy of n8_chain.sh, used
# after the pause of 07:35Z, differs only in its log file; runs_n8.log holds the pause's STOP line); rows results_n8_sweep.jsonl.
# Lane A runs alone unless lane B was started (decided 06:49Z: one lane while the decision_layer CPU queue runs); at the end of a
# stage lane A waits for lane B only when lane B has touched started_B, which it does before its first claim.
lane=$1
emb=${REPO_ROOT}/studies/text_understanding
dl=${REPO_ROOT}/studies/decision_layer/src/dl
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$emb/runs_n8b.log
cl=$emb/.n8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP lane $lane: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $emb/n8_sweep.py ae15f29b21e9b687; check $emb/n8_select.py 83c72196fec0ae2a
  check $dl/tm.py f649043f5801605b; check $dl/tm_init.py 557ff69706529738
  check $emb/emb_n7/trec__explore.npz b230a81527db270d; check $emb/emb_n7/mr__explore.npz 4bb86bf70498b6c7; }
stopped() { grep -q 'STOP lane' "$log" 2>/dev/null; }
n7set='4000|80|9.0|1|128|hardest|0'
runjob() { # $1 = job line, $2 = claim name
  local ds stage view seed clauses T s negation batch neg maxlit extra=""
  IFS='|' read -r ds stage view seed clauses T s negation batch neg maxlit <<< "$1"
  [ "$clauses|$T|$s|$negation|$batch|$neg|$maxlit" = "$n7set" ] && [ "$stage" = s1 ] && extra="--check-n7"
  stopped && exit 1
  checks
  echo "$(ts) lane $lane start $1 $extra" >> "$log"
  "$gpu" shared -- "$py" -I -B $emb/n8_sweep.py --dataset "$ds" --stage "$stage" --view "$view" --seed "$seed" \
    --clauses "$clauses" --T "$T" --s "$s" --negation "$negation" --batch "$batch" --neg "$neg" --max-literals "$maxlit" $extra < /dev/null >> "$log" 2>&1
  local e=$?
  echo "$(ts) lane $lane end $2 exit $e" >> "$log"
  [ $e -eq 3 ] && stop "the N7-setting run does not reproduce N7's val curve ($1)"
  return 0
}
stage() { # $1 = stage name
  local st=$1 i=0
  if [ "$lane" = A ]; then
    "$py" -I -B $emb/n8_select.py "$st" < /dev/null > "$cl/jobs_$st.txt" 2>> "$log" || stop "n8_select.py $st failed"
    echo "$(ts) lane A wrote $(wc -l < "$cl/jobs_$st.txt") jobs for $st" >> "$log"
    touch "$cl/ready_$st"
  else
    until [ -e "$cl/ready_$st" ]; do stopped && exit 1; sleep 20; done
  fi
  # the N7-setting jobs first, then the rest in file order
  while IFS= read -r job; do i=$((i + 1)); case "$job" in *"|$n7set") mkdir "$cl/claim_${st}_$i" 2>/dev/null && runjob "$job" "claim_${st}_$i";; esac; done < "$cl/jobs_$st.txt"
  i=0
  while IFS= read -r job; do i=$((i + 1)); mkdir "$cl/claim_${st}_$i" 2>/dev/null && runjob "$job" "claim_${st}_$i"; done < "$cl/jobs_$st.txt"
  touch "$cl/done_${st}_$lane"
  if [ "$lane" = A ] && [ -e "$cl/started_B" ]; then  # lane B touches started_B before its first claim
    until [ -e "$cl/done_${st}_A" ] && [ -e "$cl/done_${st}_B" ]; do stopped && exit 1; sleep 20; done
  fi
}
mkdir -p "$cl"
[ "$lane" = B ] && touch "$cl/started_B"
for st in s1 s2 s3; do stage $st; done
if [ "$lane" = A ]; then
  "$py" -I -B $emb/n8_select.py words > "$emb/n8_words.txt" 2>> "$log" || stop "n8_select.py words failed"
  cat "$emb/n8_words.txt" >> "$log"
fi
echo "$(ts) lane $lane done" >> "$log"
