#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Step 3c chain (tu/PLAN_3c.md; VAL only; the test split is never read). Waits for tu/.c3/gpu_go (made on the
# or the decision_layer word that it is 3c's turn on the GPU; one big job at a time). Reads tu/c3/base.json (fixed at
# the registration, after 3a's verdict): the base table and cache, the reader's s per set, the reference rows, -fa.
# Then, each step skipped when its output exists (restart-safe):
#   1. llama-server on the GPU (127.0.0.1:18081 only, -np 16, -c 16384, -ngl 99, thinking off, at most 4 state
#      checkpoints per slot) and c3_generate.py (16 workers) on all
#      32,954 training texts (--max-hours 4: the plan's stop rule); the server is stopped before anything else runs;
#   2. c3_filter.py (F1, F2 and F3; bge on the GPU), then c3_nearcopy.py (descriptive);
#   3. c3_feats.py, arms F and C per set (the equivalence check inside); arm orig (the rebuilt originals) only for a
#      set whose check failed;
#   4. c3_exists_tm.py: the code check first (CLINC150, the base cache, seed 4 at the base's s; it must equal the
#      reference row), then 3c-F on seeds 4-6 per set (and the rebuilt originals where the check failed);
#   5. c3_lr.py: the reference on the base cache (it must equal base.json's ref_lr row: the LR code check), then F
#      and C (and the rebuilt originals where needed);
#   6. c3_verdict.py.
# Before every step: the hashes and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the last
# check is over 5 min old). Log tu/c3/runs_c3.log.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
c3=$tu/c3
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
G=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742
M=$G/Qwen3.5-9B-Q8_0.gguf
log=$c3/runs_c3.log
go=$tu/.c3/gpu_go
SETS="clinc150 banking77 hwu64"
N_TRAIN=32954
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false  # env-only CPU change, outcome-neutral, before the run (decided 00:42Z; NOTE_2026-10-21_env.md)
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $tu/c3_common.py b858a6fdb8ea9832; check $tu/c3_generate.py a32e8310fb4c89d4; check $tu/c3_filter.py 409688dad01609f4
  check $tu/c3_nearcopy.py c0a64800afdc2c15; check $tu/c3_feats.py 975d51a946c39a7c; check $tu/c3_exists_tm.py 6474ebb07393f433
  check $tu/c3_lr.py db03297a64b0aa6a; check $tu/c3_verdict.py cd03796e6fc15008; check $tu/a3_teacher.py 7539e26231b77a17
  check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/w1/wordtab_feats.py 001a93b259a1b34f; check $emb/w1/build_corpus_c4.py 21fe0e6e7dad47b7; }
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 120
  done; }
step() { local name=$1 e; shift
  [ "$DRY" = 1 ] && { echo "dry: would run $name: $*"; return 0; }
  gate; checks
  echo "$(ts) start $name: $*" >> "$log"
  (cd /tmp && "$py" -I -B "$@" < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end $name exit $e" >> "$log"
  [ $e -eq 0 ] || stop "$name exit $e"; }
bj() { "$py" -I -c "import json, sys; v = json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(v)" $c3/base.json "$@"; }
have() { local f=$1 tag=$2 arm=$3 d=$4 sd=$5
  [ -e "$f" ] || return 1
  "$py" -I -c "
import json, sys
f, tag, arm, d, sd = sys.argv[1:6]
rows = [json.loads(l) for l in open(f)]
ok = any((tag == '-' or r.get('tag') == tag) and (arm == '-' or r.get('arm') == arm or r.get('name') == arm) and r['dataset'] == d
         and (sd == '-' or r.get('seed') == int(sd)) for r in rows)
sys.exit(0 if ok else 1)
" "$f" "$tag" "$arm" "$d" "$sd"; }
same() { "$py" -I -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['equivalence']['identical'] else 1)" "$1"; }
DRY=0; [ "$1" = --dry ] && DRY=1
if [ $DRY = 1 ]; then [ -e $c3/base.json ] || { echo "no base.json"; exit 1; }; checks; echo "dry: hashes ok"; else
  until [ -e $go ]; do sleep 60; done
  echo "$(ts) GPU go: $(head -c 300 $go)" >> "$log"
  [ -e $c3/base.json ] || stop "no base.json"
  echo "$(ts) base.json $(sha256sum $c3/base.json | cut -c1-16): $(tr '\n' ' ' < $c3/base.json | head -c 900)" >> "$log"
  echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
base=$(bj base); table=$(bj table_npz); ref=$(bj ref_dir); fa=$(bj fa)
# 1. generation
n_gen=0; [ -e $c3/gen.jsonl ] && n_gen=$(grep -c "" $c3/gen.jsonl)
if [ "$n_gen" -lt $N_TRAIN ]; then
  if [ $DRY = 1 ]; then echo "dry: would start llama-server (-fa $fa) and c3_generate.py ($n_gen of $N_TRAIN done)"; else
    gate; checks
    (cd $G && sha256sum -c --quiet FILES.sha256) >> "$log" 2>&1 || stop "GGUF files do not match"
    echo "$(ts) start llama-server -fa $fa" >> "$log"
    (cd /tmp && exec $B/llama-server -m $M --host 127.0.0.1 --port 18081 --no-webui -np 16 -c 16384 -ngl 99 -fa $fa --jinja \
      --reasoning-budget 0 --ctx-checkpoints 4 > $c3/server_gpu.txt 2>&1) &
    spid=$!
    trap 'kill $spid 2>/dev/null' EXIT
    for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:18081/health 2>/dev/null | grep -q '"ok"' && break; sleep 5; done
    echo "$(ts) server health: $(curl -s -m 2 http://127.0.0.1:18081/health)" >> "$log"
    (cd /tmp && "$py" -I -B $tu/c3_generate.py --port 18081 --out $c3/gen.jsonl --workers 16 --max-hours 5 < /dev/null >> "$log" 2>&1)
    e=$?
    kill $spid; wait $spid 2>/dev/null; trap - EXIT
    echo "$(ts) generate exit $e; server stopped; GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
    n_gen=$(grep -c "" $c3/gen.jsonl)
    [ $e -eq 0 ] && [ "$n_gen" -eq $N_TRAIN ] || stop "generation incomplete: $n_gen of $N_TRAIN rows (exit $e)"
  fi
fi
# 2. filters
[ -e $c3/filter_meta.json ] || step filter $tu/c3_cuda_first.py $tu/c3_filter.py --gen $c3/gen.jsonl --out-dir $c3 --device cuda
[ -e $c3/nearcopy_F.json ] || step nearcopy $tu/c3_cuda_first.py $tu/c3_nearcopy.py --kept $c3/kept_F.jsonl --out $c3/nearcopy_F.json --device cuda
# 3. features
for arm in F C; do for d in $SETS; do
  [ -e $c3/feats/$arm/${d}__p32tok_K512.npz ] || step feats $tu/c3_feats.py --dataset $d --table wordtab:$table --kept $c3/kept_$arm.jsonl --arm $arm --out-dir $c3/feats/$arm --ref-dir $ref
done; done
for d in $SETS; do
  [ $DRY = 1 ] && break
  same $c3/feats/F/${d}__p32tok_K512.meta.json && continue
  [ -e $c3/feats/orig/${d}__p32tok_K512.npz ] || step feats $tu/c3_feats.py --dataset $d --table wordtab:$table --kept none --arm orig --out-dir $c3/feats/orig --ref-dir $ref
done
# 4. TM: the code check, then 3c-F (and the rebuilt originals where the check failed)
s1=$(bj s clinc150)
have $c3/results_c3_val.jsonl 3c-check - clinc150 4 || step tm $tu/c3_exists_tm.py --dataset clinc150 --view existsT --seed 4 --s $s1 --emb-dir $ref --tag 3c-check
if [ $DRY = 0 ]; then
  "$py" -I -c "
import json, sys
b = json.load(open(sys.argv[1])); s = float(b['s']['clinc150'])
mine = [r for r in map(json.loads, open(sys.argv[2])) if r.get('tag') == '3c-check' and r['dataset'] == 'clinc150' and r['seed'] == 4]
ref = [r for r in map(json.loads, open(b['ref_tm']['file'])) if r.get('tag') == b['ref_tm']['tag'] and r['table'] == b['base']
       and r['dataset'] == 'clinc150' and float(r['s']) == s and r['seed'] == 4]
assert len(mine) == 1 and len(ref) == 1, (len(mine), len(ref))
print(mine[0]['val_acc'], ref[0]['val_acc']); sys.exit(0 if mine[0]['val_acc'] == ref[0]['val_acc'] else 1)
" $c3/base.json $c3/results_c3_val.jsonl >> "$log" 2>&1 || stop "code check: c3_exists_tm.py does not give the reference row"
  echo "$(ts) code check passed" >> "$log"
fi
for d in $SETS; do
  s=$(bj s $d)
  for sd in 4 5 6; do
    have $c3/results_c3_val.jsonl 3c-tm F $d $sd || step tm $tu/c3_exists_tm.py --dataset $d --view existsT --seed $sd --s $s --emb-dir $c3/feats/F --tag 3c-tm
  done
  if [ $DRY = 0 ] && ! same $c3/feats/F/${d}__p32tok_K512.meta.json; then
    for sd in 4 5 6; do
      have $c3/results_c3_val.jsonl 3c-ref orig $d $sd || step tm $tu/c3_exists_tm.py --dataset $d --view existsT --seed $sd --s $s --emb-dir $c3/feats/orig --tag 3c-ref
    done
  fi
done
# 5. LR
for d in $SETS; do
  have $c3/results_c3_lr.jsonl - ref $d - || step lr $tu/c3_lr.py --name ref --dataset $d --emb-dir $ref
  if [ $DRY = 0 ]; then
    "$py" -I -c "
import json, sys
b, d = json.load(open(sys.argv[1])), sys.argv[3]
mine = [r for r in map(json.loads, open(sys.argv[2])) if r.get('name') == 'ref' and r['dataset'] == d]
ref = [r for r in map(json.loads, open(b['ref_lr']['file'])) if r.get('table') == b['base'] and r['dataset'] == d]
assert len(mine) == 1 and len(ref) == 1, (len(mine), len(ref))
print(d, mine[0]['lr_val'], ref[0]['lr_val']); sys.exit(0 if mine[0]['lr_val'] == ref[0]['lr_val'] else 1)
" $c3/base.json $c3/results_c3_lr.jsonl $d >> "$log" 2>&1 || stop "LR code check: c3_lr.py does not give the reference row ($d)"
  fi
  for arm in F C; do
    have $c3/results_c3_lr.jsonl - $arm $d - || step lr $tu/c3_lr.py --name $arm --dataset $d --emb-dir $c3/feats/$arm
  done
  if [ $DRY = 0 ] && ! same $c3/feats/F/${d}__p32tok_K512.meta.json; then
    have $c3/results_c3_lr.jsonl - ref_rebuilt $d - || step lr $tu/c3_lr.py --name ref_rebuilt --dataset $d --emb-dir $c3/feats/orig
  fi
done
# 6. verdict
[ -e $c3/verdict_c3.json ] || step verdict $tu/c3_verdict.py
[ $DRY = 1 ] || echo "$(ts) 3c done" >> "$log"
