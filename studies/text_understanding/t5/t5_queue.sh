#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# t5_queue.sh controls|tmae [--dry-run | --dry-run-bad-hash] [--done=FILE (dry runs only)]
# The registered val runs of REGISTRATION_T5.md (with Amendment 1), one after another in ONE job slot (one process
# at a time), each under the shared GPU lock (gpu_run.sh shared), with the decision_layer .venv-dl python -I.
#   controls: views in this order (Amendment 1): wordsV, potionV_K512, ppmiV_K512 (PPMI-SVD), ppmiV_K86, potionV_K86
#             last; within each view clinc150 banking77 hwu64 x seeds 1 2 (30 runs)
#   tmae:     R1 R3a x clinc150 banking77 hwu64 x seeds 1 2 (12 runs); needs t5/tmae_done.json (finished TM-AE run)
# Nothing starts this script; start it by hand, e.g.:  bash studies/text_understanding/t5/t5_queue.sh controls
# Before every run (and before building code tables) the sha256 of the frozen scripts is checked; on a mismatch the
# queue writes "Error: hash mismatch" and stops. Missing code tables are built first on the CPU (controls: codes.py
# --controls-only; tmae: codes.py --ckpt <the checkpoint named in tmae_done.json> --tables R1,R3a, also rebuilt when
# the tables came from another checkpoint); t5_tm.py checks every table against t5/codes_manifest.json.
# Log: t5/runs_t5.log (date -u timestamps); rows: t5/results_t5.jsonl (by t5_tm.py).
# --dry-run: the same checks and dispatch, but every command is printed instead of run, nothing is written, no lock;
# --dry-run-bad-hash: a dry run with one expected hash set wrong on purpose (tests the mismatch stop);
# --done=FILE: a dry run of part tmae against another done file (e.g. tmae_done_tiny.json).
set -u
usage() { echo "usage: t5_queue.sh controls|tmae [--dry-run | --dry-run-bad-hash] [--done=FILE]" >&2; exit 2; }
part="${1:-}"
case "$part" in
    controls | tmae) shift ;;
    *) usage ;;
esac
dry=0
badhash=0
done_file=tmae_done.json
for a in "$@"; do
    case "$a" in
        --dry-run) dry=1 ;;
        --dry-run-bad-hash) dry=1; badhash=1 ;;
        --done=*) done_file="${a#--done=}" ;;
        *) usage ;;
    esac
done
if [ "$dry" = 0 ] && [ "$done_file" != tmae_done.json ]; then
    echo "--done is for dry runs only" >&2
    exit 2
fi
T5=${REPO_ROOT}/studies/text_understanding/t5
DL=${REPO_ROOT}/studies/decision_layer/src/dl
GPU=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
LOG="$T5/runs_t5.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
declare -A want=(
    ["$T5/t5_tm.py"]=45f9e733575e6273 ["$T5/codes.py"]=a414e651ced287b4
    ["$DL/tm.py"]=f649043f5801605b ["$DL/tm_init.py"]=557ff69706529738 ["$DL/boolean.py"]=b5f5917f390da8a6
    ["$DL/data.py"]=0c4f8d22576e817c
)
[ "$badhash" = 1 ] && want["$T5/t5_tm.py"]=0000000000000000

log() {
    if [ "$dry" = 1 ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$part dry] $*"
    else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$part] $*" >> "$LOG"
    fi
}
run() {  # run a command with its output appended to the log; a dry run only prints it
    if [ "$dry" = 1 ]; then
        echo "    would run: $*"
        return 0
    fi
    "$@" >> "$LOG" 2>&1
}
check() {
    local f got
    for f in "${!want[@]}"; do
        got=$(sha256sum "$f" | cut -c1-16)
        if [ "$got" != "${want[$f]}" ]; then
            log "Error: hash mismatch $f $got (want ${want[$f]}); queue stopped before: $*"
            exit 1
        fi
    done
}
one() {  # one registered run
    check "$1 $2 seed $3"
    log "start $1 $2 seed $3"
    run "$GPU" shared -- "$PY" -I "$T5/t5_tm.py" --dataset "$1" --view "$2" --seed "$3"
    log "end $1 $2 seed $3 exit $?"
}

if [ "$dry" = 0 ]; then
    exec 9>"$T5/.t5_queue_${part}.lock"
    if ! flock -n 9; then
        log "another t5_queue.sh $part holds the lock; not starting (pid $$)"
        exit 1
    fi
fi
log "queue start (pid $$, host $(hostname))"

if [ "$part" = controls ]; then
    if [ ! -s "$T5/codes_potionV.npz" ] || [ ! -s "$T5/codes_ppmiV.npz" ] || [ ! -s "$T5/codes_wordsV.npz" ] ||
        [ ! -s "$T5/codes_manifest.json" ]; then
        check "codes.py --controls-only"
        log "codes.py --controls-only start"
        run env CUDA_VISIBLE_DEVICES= nice -n 10 "$PY" -I "$T5/codes.py" --controls-only
        rc=$?
        log "codes.py --controls-only exit $rc"
        [ "$rc" -eq 0 ] || exit 1
    else
        log "control tables present: codes_potionV.npz codes_ppmiV.npz codes_wordsV.npz"
    fi
    for v in wordsV potionV_K512 ppmiV_K512 ppmiV_K86 potionV_K86; do
        for ds in clinc150 banking77 hwu64; do
            for seed in 1 2; do
                one "$ds" "$v" "$seed"
            done
        done
    done
else
    if [ ! -s "$T5/$done_file" ]; then
        log "Error: t5/$done_file missing (the TM-AE training has not finished); queue stopped"
        exit 1
    fi
    ckpt=$("$PY" -I -c "import json; print(json.load(open('$T5/$done_file'))['last_checkpoint'] or '')")
    if [ -z "$ckpt" ] || [ ! -s "$T5/$ckpt" ]; then
        log "Error: no checkpoint named in $done_file ($ckpt); queue stopped"
        exit 1
    fi
    need=$("$PY" -I -c "
import hashlib, json, pathlib
t5 = pathlib.Path('$T5'); p = t5 / 'codes_manifest.json'
want = hashlib.sha256((t5 / '$ckpt').read_bytes()).hexdigest()
tabs = json.loads(p.read_text())['tables'] if p.exists() else {}
ok = all(t in tabs and tabs[t].get('source', {}).get('ckpt_sha256') == want and (t5 / tabs[t]['file']).exists()
         for t in ('R1', 'R3a'))
print('no' if ok else 'yes')")
    if [ "$need" = yes ]; then
        check "codes.py --ckpt $ckpt --tables R1,R3a"
        log "codes.py --ckpt $ckpt --tables R1,R3a start"
        run env CUDA_VISIBLE_DEVICES= nice -n 10 "$PY" -I "$T5/codes.py" --ckpt "$T5/$ckpt" --tables R1,R3a
        rc=$?
        log "codes.py exit $rc"
        [ "$rc" -eq 0 ] || exit 1
    fi
    log "R1 / R3a tables from $ckpt"
    for seed in 1 2; do
        for ds in clinc150 banking77 hwu64; do
            for v in R1 R3a; do
                one "$ds" "$v" "$seed"
            done
        done
    done
fi
log "queue end"
