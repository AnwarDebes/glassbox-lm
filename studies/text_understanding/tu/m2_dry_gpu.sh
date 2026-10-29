#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# M2 dry run, BEFORE the freeze (tu/PLAN_M2.md "Dry run"): the M2 scripts on an MTOP stand-in, every python step under
# tu/m2_audit_run.py. No MTOP label is used and en/test.txt is never opened (the audit raises if it is): the stand-in
# gives the real en/train and en/eval texts with labels = sha256(text) mod 20, every 40th en/eval label set to -1 (an
# intent not in train, so the -1 path runs through every step), and test := val. Writes only under
# tu/m2_dry/<UTC>[_cpu]/ (never tu/m2). Not part of the registered chain.
# GPU (a queue request; alone on the GPU): nice -n 19 bash tu/m2_dry_gpu.sh
#   0. the full sha256 of every file of `m2_write_pins.sh --list` is read first (code_sha256.txt, and in the summary),
#      so the summary names the exact code it ran (the freeze check compares it with the pinned files);
#   1. the code check as in m2_chain.sh step 1, into <dry>/check, compared by m2_codecheck.py with 3c's rows;
#   2. m2_feats.py for C4, a3B and potion on m2_standin.py:mtop_unseen (all 15,667 en/train texts), on CUDA;
#   3. m2_tm.py --smoke (200 clauses, T 200, 2 epochs), seed 7, per arm; then ONE full-size m2_tm.py run (4,000
#      clauses, T 4,000, 40 epochs, s 8, seed 7) on the C4 cache: a lower bound on the time of one registered run at
#      MTOP's size (20 stand-in classes, not MTOP's intent count);
#   4. m2_lr.py per arm;
#   5. the dry check (below), then <dry>/dry_summary.json.
# CPU (the tests' smoke of the data path; no GPU): M2_DRY_DEVICE=cpu nice -n 19 bash tu/m2_dry_gpu.sh
#   the same steps on the CPU with m2_standin.py:mtop_small_unseen (the first 1,500 en/train texts), without the code
#   check (CLINC150 at full size) and without the full-size TM run.
# The dry check: every TM row's "test" accuracy equals its val accuracy and every LR row's lr_test equals lr_val
# (test := val); each MTOP row's accuracy equals the one recomputed from its prediction file and the stand-in's labels,
# with the -1 items counted as errors (no prediction is -1) and the row's unseen counts equal to the stand-in's; every
# audit line read only mtop/en/train.txt and mtop/en/eval.txt (or no MTOP member); the C4 and a3B steps imported no
# sentence_transformers, transformers or model2vec; no file under tu/m2 was created or changed.
# The CPU gate (cpu_load.py over 10 s at most 4.5 cores) runs before every step, re-checked when over 5 min old.
# Changes after the review of draft 1 (2026-10-25): the code hashes (data MINOR 4), the -1 stand-in (design MINOR 6,
# data MINOR 3), m2_codecheck.py in place of the inline comparison (design MINOR 10).
# Hooks: M2_DRY_DEVICE (cuda|cpu), M2_DRY_ROOT (the parent folder; default tu/m2_dry), M2_DRY_LOAD (the gate command).
set -u
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
c3=$tu/c3
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
loadpy=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
dev=${M2_DRY_DEVICE:-cuda}
root=${M2_DRY_ROOT:-$tu/m2_dry}
read -r -a LOADCMD <<< "${M2_DRY_LOAD:-$py -I $loadpy 10}"
SET=mtop_en
A3B=$tu/a3/tables/a3B_d768.npz
C4=$emb/w1/tables/ppw_S1000_d500_e0.5.npz
case "$dev" in
  cuda) suffix=""; standin=mtop_unseen; AUD=("$py" -I -B "$tu/m2_audit_run.py") ;;
  cpu) suffix=_cpu; standin=mtop_small_unseen; AUD=("$py" -I -B "$tu/m2_audit_run.py" --cpu) ;;
  *) echo "M2_DRY_DEVICE must be cuda or cpu" >&2; exit 2 ;;
esac
data=$tu/m2_standin.py:$standin
d=$root/$(date -u +%Y%m%dT%H%M%SZ)$suffix
[ -e "$d" ] && { echo "$d exists" >&2; exit 2; }
mkdir -p "$d" || exit 2
log=$d/dry.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 M2_AUDIT_LOG=$d/audit.jsonl
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
bash "$tu/m2_write_pins.sh" --list > "$d/code_files.txt" || { echo "cannot list the code files" >&2; exit 2; }
xargs -d '\n' sha256sum < "$d/code_files.txt" > "$d/code_sha256.txt" || { echo "a listed file is missing" >&2; exit 2; }
m2_before=$(find "$tu/m2" -type f -printf '%p %s %T@\n' 2>/dev/null | sort | sha256sum | cut -c1-64)
last_gate=-1000
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("${LOADCMD[@]}"); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 60 & wait $!
  done; }
step() { local name=$1 t0 e; shift
  gate
  t0=$(date +%s.%N)
  echo "$(ts) start $name: $*" >> "$log"
  (cd /tmp && nice -n 19 "$@" < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end $name exit $e seconds $(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b - a}')" >> "$log"
  [ $e -eq 0 ] || { echo "$(ts) DRY STOP: $name exit $e" >> "$log"; echo "dry run failed at $name (log $log)" >&2; exit 1; }; }
echo "$(ts) m2 dry run on $dev, data $data, folder $d; code_sha256.txt $(sha256sum "$d/code_sha256.txt" | cut -c1-64)" >> "$log"
if [ "$dev" = cuda ]; then
  ck=$d/check
  step check-cache "${AUD[@]}" "$tu/m2_check_cache.py" --src "$c3/feats/orig" --out "$ck"
  step code-check-tm "${AUD[@]}" "$tu/m2_tm.py" --dataset clinc150 --view existsT --seed 4 --s 8 --emb-dir "$ck" \
    --tag m2-check --out-dir "$ck" --load "$tu/m2_standin.py:load"
  step code-check-lr "${AUD[@]}" "$tu/m2_lr.py" --name check --dataset clinc150 --emb-dir "$ck" --out-dir "$ck" \
    --load "$tu/m2_standin.py:load"
  nice -n 19 "$py" -I -B "$tu/m2_codecheck.py" --c3 "$c3" --ck "$ck" >> "$log" 2>&1 \
    || { echo "$(ts) DRY STOP: code check failed" >> "$log"; echo "code check failed (log $log)" >&2; exit 1; }
  echo "$(ts) code check passed" >> "$log"
fi
step feats-C4 "${AUD[@]}" "$tu/m2_feats.py" --dataset "$SET" --table "wordtab:$C4" --arms C4:none --out-root "$d/feats" \
  --device "$dev" --load "$data"
step feats-a3B "${AUD[@]}" "$tu/m2_feats.py" --dataset "$SET" --table "wordtab:$A3B" --arms a3B:none --out-root "$d/feats" \
  --device "$dev" --load "$data"
step feats-potion "${AUD[@]}" "$tu/m2_feats.py" --dataset "$SET" --table potion --arms potion:none --out-root "$d/feats" \
  --device "$dev" --load "$data"
for arm in C4 a3B potion; do
  step "tm-smoke-$arm" "${AUD[@]}" "$tu/m2_tm.py" --dataset "$SET" --view existsT --seed 7 --s 8 --emb-dir "$d/feats/$arm" \
    --tag dry --out-dir "$d/run" --device "$dev" --smoke --load "$data"
done
if [ "$dev" = cuda ]; then
  step tm-full-C4 "${AUD[@]}" "$tu/m2_tm.py" --dataset "$SET" --view existsT --seed 7 --s 8 --emb-dir "$d/feats/C4" \
    --tag dryfull --out-dir "$d/run" --device "$dev" --load "$data"
fi
for arm in C4 a3B potion; do
  step "lr-$arm" "${AUD[@]}" "$tu/m2_lr.py" --name "$arm" --dataset "$SET" --emb-dir "$d/feats/$arm" --out-dir "$d/run" \
    --device "$dev" --load "$data"
done
m2_after=$(find "$tu/m2" -type f -printf '%p %s %T@\n' 2>/dev/null | sort | sha256sum | cut -c1-64)
nice -n 19 "$py" -I -B -c "
import hashlib, json, sys, zipfile
import numpy as np
d, dev, before, after, tu, standin = sys.argv[1:7]
_open = zipfile.ZipFile.open
def _guard(self, name, *a, **k):
    if (name if isinstance(name, str) else name.filename) == 'mtop/en/test.txt':
        raise RuntimeError('AUDIT: mtop/en/test.txt was opened')
    return _open(self, name, *a, **k)
zipfile.ZipFile.open = _guard
sys.path.insert(0, tu)
import m2_standin
ds = getattr(m2_standin, standin)('mtop_en')
y = np.asarray(ds.labels['test'])
tm = [json.loads(x) for x in open(d + '/run/results_m2_tm.jsonl')]
lr = [json.loads(x) for x in open(d + '/run/results_m2_lr.jsonl')]
au = [json.loads(x) for x in open(d + '/audit.jsonl')]
err = []
for r in tm:
    if not (r['test_acc'] == r['val_acc_check'] == r['val_acc']):
        err.append(f\"TM {r['tag']} {r['arm']}: test {r['test_acc']} val {r['val_acc']} check {r['val_acc_check']}\")
for r in lr:
    if r['lr_test'] != r['lr_val']:
        err.append(f\"LR {r['name']}: test {r['lr_test']} val {r['lr_val']}\")
for r in tm + lr:
    p = np.load(r['preds'])
    acc = r.get('test_acc', r.get('lr_test'))
    if len(p) != len(y) or round(float((p == y).mean()), 4) != acc or (p < 0).any():
        err.append(f\"-1 path: {r.get('tag', 'lr')} {r['arm']}: recomputed {float((p == y).mean()):.4f} vs row {acc}, {int((p < 0).sum())} negative predictions\")
    if r.get('unseen_intents') != ds.sources['unseen_intents']:
        err.append(f\"-1 path: {r.get('tag', 'lr')} {r['arm']}: unseen_intents {r.get('unseen_intents')} vs {ds.sources['unseen_intents']}\")
n_unseen = int((y < 0).sum())
if n_unseen == 0:
    err.append('the stand-in has no -1 label')
ok_members = {'mtop/en/train.txt', 'mtop/en/eval.txt'}
for a in au:
    if not set(a['mtop_members']) <= ok_members:
        err.append(f\"audit: {a['script']} read {a['mtop_members']}\")
    wordtab = any('wordtab:' in x for x in a['args']) or any(x.endswith('/C4') or x.endswith('/a3B') for x in a['args'])
    if wordtab and any(a['nn_imported'].values()):
        err.append(f\"network-free: {a['script']} {a['args'][:4]} imported {a['nn_imported']}\")
mtop_steps = [a for a in au if '--dataset' in a['args'] and a['args'][a['args'].index('--dataset') + 1] == 'mtop_en']
if not mtop_steps or any(not a['mtop_members'] for a in mtop_steps):
    err.append('an MTOP step read no MTOP member (the stand-in was not loaded from the zip)')
if before != after:
    err.append('a file under tu/m2 was created or changed')
code = {}
for ln in open(d + '/code_sha256.txt'):
    h, f = ln.rstrip('\n').split('  ', 1)
    code[f] = h
full = [r for r in tm if r['tag'] == 'dryfull']
summ = {'device': dev, 'folder': d, 'stand_in': standin, 'tm_rows': len(tm), 'lr_rows': len(lr), 'audit_lines': len(au),
        'unseen_items': n_unseen, 'unseen_intents': ds.sources['unseen_intents'],
        'mtop_members_read': sorted({m for a in au for m in a['mtop_members']}),
        'nn_imported_by_step': [[a['script'], a['args'][a['args'].index('--table') + 1] if '--table' in a['args'] else
                                 (a['args'][a['args'].index('--emb-dir') + 1].rsplit('/', 1)[-1] if '--emb-dir' in a['args'] else ''),
                                 [m for m, v in a['nn_imported'].items() if v]] for a in au],
        'tm_seconds': {r['tag'] + '_' + r['arm']: [r['train_s'], r['total_s']] for r in tm},
        'lr_seconds': {r['name']: r['seconds'] for r in lr},
        'n_train': sorted({r['n_train'] for r in tm if r['dataset'] == 'mtop_en'}),
        'full_size_tm_train_s': full[0]['train_s'] if full else None,
        'code_sha256': code, 'code_sha256_file': hashlib.sha256(open(d + '/code_sha256.txt', 'rb').read()).hexdigest(),
        'tu_m2_unchanged': before == after, 'errors': err, 'pass': not err}
json.dump(summ, open(d + '/dry_summary.json', 'x'), indent=1)
print(json.dumps({k: summ[k] for k in ('device', 'stand_in', 'tm_rows', 'lr_rows', 'audit_lines', 'unseen_items', 'mtop_members_read',
                                       'n_train', 'full_size_tm_train_s', 'tu_m2_unchanged', 'pass')}))
for e in err:
    print('DRY CHECK FAIL:', e)
sys.exit(0 if not err else 1)
" "$d" "$dev" "$m2_before" "$m2_after" "$tu" "$standin" >> "$log" 2>&1
e=$?
echo "$(ts) dry check exit $e" >> "$log"
echo "m2 dry run on $dev: exit $e, folder $d"
exit $e
