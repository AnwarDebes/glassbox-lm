#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Test 5 timing pilot (DESIGN_NOTES.md Sec. 4.4): every cell one after another in ONE process slot.
# Nothing starts this script; start it by hand when a job slot is free:
#     bash studies/text_understanding/t5/pilot_queue.sh
# CPU only (CUDA_VISIBLE_DEVICES=""), one thread, each cell under `nice -n 10 timeout 900`, .venv-tm python -I.
# Builds corpus (a) first if t5/corpus_a.npz is missing. Rows: t5/pilot_results.jsonl; log: t5/runs_pilot.log.
# "A5 --masked" is an extra row (paper-faithful masked class sum, tmae_fixed.py); R needs A5, A6 and O1 first.
set -u
T5=${REPO_ROOT}/studies/text_understanding/t5
PY=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
LOG="$T5/runs_pilot.log"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

exec 9>"$T5/.pilot_queue.lock"
if ! flock -n 9; then
    log "another pilot_queue.sh holds $T5/.pilot_queue.lock; not starting (pid $$)"
    exit 1
fi

log "queue start (pid $$, host $(hostname))"
if [ ! -s "$T5/corpus_a.npz" ] || [ ! -s "$T5/corpus_a.manifest.json" ]; then
    log "corpus (a) missing: build_corpus_a.py start"
    nice -n 10 timeout 900 "$PY" -I "$T5/build_corpus_a.py" >> "$LOG" 2>&1
    log "build_corpus_a.py exit $?"
fi

for args in "A1" "A2" "A3" "A4" "A5" "A5 --masked" "A6" "O1" "R"; do
    log "cell $args start"
    # shellcheck disable=SC2086
    nice -n 10 timeout 900 "$PY" -I "$T5/pilot_tmae.py" --cell $args >> "$LOG" 2>&1
    log "cell $args exit $?"
done
log "queue end"
