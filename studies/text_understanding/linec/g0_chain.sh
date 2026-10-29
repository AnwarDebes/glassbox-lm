#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Line C G0 (EXPLORATORY, validation only; plan LINE_C_NOTE.md cf36f27a30d5353a, approved 13:38Z; rules in LOG.md
# before any run). Order set at 13:38Z: after the decision_layer e74 queue is done, CPU only, gate 4.5 cores;
# N9 on the GPU may overlap. Waits until (1) glassbox-lm/studies/text_understanding/.n9/e74_over exists (made on the queue's go that
# e74 is over, the same marker N9 waits for), (2) memory: this cgroup's anonymous memory (memory.stat anon; the page
# cache is reclaimable, decided 13:51Z) + 40 GiB is under 0.8 of memory.max, (3) the CPU load without it is at most 4.5 cores (cpu_load.py over 10 s), each re-checked every 5 min.
# Then the hash checks and g0.py on one thread. Log linec/runs_g0.log.
emb=${REPO_ROOT}/studies/text_understanding
lc=$emb/linec
tm=${REPO_ROOT}/studies/tm_language_model
py=$tm/.venv-tm/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
lpy=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$lc/runs_g0.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $lc/g0.py bae7e372a655d4b8
  check $tm/src/tmlm/cvtm/kn_gpu.py 582507f75a8a792f; check $tm/src/tmlm/eval/baselines/ngram.py 67449348153c7deb
  check $tm/src/tmlm/cvtm/protocol.py ef0cc0c0d4f86a8c; check $tm/src/tmlm/data/corpus_store.py 4660db695c812332
  check $tm/data/tokenizers/bpe_8000/tokenizer.json a2a13a4e2888d170; check $tm/data/wikitext103/bpe8000/train/kind_map.npy 79193b85fe36afb5; }
anon() { awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat; }
mem_ok() { local cur max; cur=$(anon); max=$(cat /sys/fs/cgroup/memory.max)
  awk -v c="$cur" -v m="$max" 'BEGIN{exit !(c + 40 * 2^30 < 0.8 * m)}'; }
if [ "$1" = --dry ]; then
  [ -e $emb/.n9/e74_over ] && echo "dry: e74 over" || echo "dry: would wait for e74"
  mem_ok && echo "dry: memory ok" || echo "dry: would wait for memory"; exit 0
fi
until [ -e $emb/.n9/e74_over ]; do sleep 60; done
echo "$(ts) e74 is over: $(head -c 300 $emb/.n9/e74_over)" >> "$log"
while true; do
  if mem_ok; then
    line=$("$lpy" -I "$load" 10)
    c=$(echo "$line" | awk '{print $3}')
    echo "$(ts) load check: $line" >> "$log"
    awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}' && break
  else
    echo "$(ts) memory check: anon $(anon) of $(cat /sys/fs/cgroup/memory.max)" >> "$log"
  fi
  sleep 300
done
checks
echo "$(ts) memory at the start: anon $(anon), file $(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat), max $(cat /sys/fs/cgroup/memory.max)" >> "$log"
echo "$(ts) start g0.py" >> "$log"
cd /tmp && "$py" -I -B $lc/g0.py < /dev/null >> "$log" 2>&1
e=$?
echo "$(ts) end g0.py exit $e" >> "$log"
[ $e -eq 0 ] || stop "g0.py exit $e"
echo "$(ts) G0 done" >> "$log"
