#!/usr/bin/env bash
# R24 after its freeze (registrations/R24_multiwoz_random_negatives.md; this file is pinned by the registration):
#   nohup bash logs/R24_chain.sh >> logs/R24_chain.log 2>&1
# Refuses to start ("R24 end: refused (...)") unless the freeze record holds the registration's text hash, logs/e85_end.log's
# last line is "e85 end: done ...", logs/R24_dry_check.out ends "all dry checks passed", and the MultiWOZ source files match
# their pinned MANIFEST (data/untrusted/multiwoz22/MANIFEST.sha256).
# 0. the gate at s* (dev; scripts/r24_check.py --gate > logs/R24_gate.out): both agents missed -> stop, test unread (exit 3)
# 0b. the network-free path test (tests/test_r24_network_free.py --run; GPU, dry size, dev only) -> logs/R24_network_free.out
# 1. multiwoz_prepare.py --with-test (CPU)  2. multiwoz_embed.py --add-test (GPU, minutes)
# 3. logs/jobs_R24.txt (written by scripts/e85_pick.py --stage final: each agent at its e85 s*) through scripts/seq_queue.sh  4. scripts/r24_check.py > logs/r24_check.out (stops on a failure)
# 5. the registered report (scripts/report_abcd.py)  6. the two-level intervals, no rule (scripts/report_twolevel.py).
# Every way out appends one line "R24 end: <branch> <UTC> exit <code>" to logs/R24_end.log. On TERM or INT the stage's
# process group is stopped first. The *_CMD, MWOZ_MANIFEST_SHA and R24_ROOT overrides exist only for logs/test_R24_chain.sh.
set -uo pipefail
ROOT="${R24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY=".venv-dl/bin/python"
GATE="${GATE_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PY scripts/r24_check.py --gate}"
NF="${NF_CMD:-env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 $PY tests/test_r24_network_free.py --run}"
MWOZ_MANIFEST="${MWOZ_MANIFEST_SHA:-87d8b2cd220b8f2cd2f9527f2a223b658151a5cb3f3bd810bbcd4c50209e58cc}"
REG=registrations/R24_multiwoz_random_negatives
PREP="${PREP_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 $PY -I scripts/multiwoz_prepare.py --with-test}"
EMBED="${EMBED_CMD:-env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false $PY -I scripts/multiwoz_embed.py --add-test}"
QUEUE="${QUEUE_CMD:-bash scripts/seq_queue.sh logs/jobs_R24.txt logs/R24_queue.log}"
CHECK="${CHECK_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PY scripts/r24_check.py}"
REPORT="${REPORT_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PY scripts/report_abcd.py --registration R24mwoz --arms tm_mem_ref_nrand tm_mem_c4_nrand tm_mem_bge_nrand lr_mem_ref lr_mem_c4 lr_mem_bge --pairs tm_mem_ref_nrand:tm_mem_bge_nrand tm_mem_c4_nrand:tm_mem_bge_nrand tm_mem_ref_nrand:lr_mem_ref tm_mem_c4_nrand:lr_mem_c4 tm_mem_bge_nrand:lr_mem_bge --out report_R24mwoz.json}"
TWOLEVEL="${TWOLEVEL_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $PY scripts/report_twolevel.py --registration R24mwoz --pairs tm_mem_ref_nrand:tm_mem_bge_nrand tm_mem_c4_nrand:tm_mem_bge_nrand tm_mem_ref_nrand:lr_mem_ref tm_mem_c4_nrand:lr_mem_c4 tm_mem_bge_nrand:lr_mem_bge --out $ROOT/results/confirm/R24mwoz/twolevel_R24mwoz.json}"
cd "$ROOT" || exit 1
BRANCH="stopped before the start"; CHILD=""
log() { echo "$(date -u +%FT%TZ) $*"; }
trap 'rc=$?; echo "R24 end: $BRANCH $(date -u +%FT%TZ) exit $rc" >> logs/R24_end.log' EXIT
on_signal() { BRANCH="signal $1"; [ -n "$CHILD" ] && kill -TERM -- "-$CHILD" 2>/dev/null && wait "$CHILD" 2>/dev/null; exit "$2"; }
trap 'on_signal TERM 143' TERM; trap 'on_signal INT 130' INT
stage() { setsid -w "$@" & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return $rc; }
refuse() { BRANCH="refused ($1)"; log "$1: R24 stops"; exit 1; }
[ -e "$REG.frozen.json" ] && [ "$(grep -m1 '"sha256"' "$REG.frozen.json" | grep -o '[0-9a-f]\{64\}')" = "$(sha256sum "$REG.md" | cut -d' ' -f1)" ] \
  || refuse "no freeze record of the registration's text"
tail -n 1 logs/e85_end.log 2>/dev/null | grep -q '^e85 end: done ' || refuse "e85 has not ended done"
[ "$(tail -n 1 logs/R24_dry_check.out 2>/dev/null)" = "all dry checks passed" ] || refuse "the dry checks have not passed"
[ "$(sha256sum data/untrusted/multiwoz22/MANIFEST.sha256 2>/dev/null | cut -d' ' -f1)" = "$MWOZ_MANIFEST" ] \
  && (cd data/untrusted/multiwoz22 && sha256sum -c --quiet MANIFEST.sha256) > /dev/null 2>&1 || refuse "the MultiWOZ source files do not match the pinned MANIFEST"
for f in logs/R24_end.log logs/R24_gate.out logs/R24_network_free.out logs/R24_queue.log logs/r24_check.out data/multiwoz/examples_test.json.gz \
         data/multiwoz/labels_test.json data/emb/multiwoz__bge-large.devstage.npz; do
  [ -e "$f" ] && { BRANCH="refused ($f exists)"; log "$f exists: R24 stops"; exit 1; }
done
log "R24 start (pid $$): the gate at s* (dev)"
$GATE > logs/R24_gate.out 2>&1; rc=$?; cat logs/R24_gate.out
[ $rc -eq 3 ] && { BRANCH="gate missed at s* for both agents (test unread)"; exit 3; }
[ $rc -eq 0 ] || { BRANCH="gate check failed (test unread)"; exit 1; }
log "the network-free path test (dry size, dev only)"
BRANCH="network-free test failed (test unread)"; stage $NF > logs/R24_network_free.out 2>&1 || exit 1
tail -n 1 logs/R24_network_free.out
log "test examples"
BRANCH="prepare failed"; stage $PREP >> logs/R24_prep.out 2>&1 || exit 1
log "test vectors (bge-large)"
BRANCH="embed failed"; stage $EMBED >> logs/R24_prep.out 2>&1 || exit 1
log "the 18 test jobs"
BRANCH="jobs running"; stage $QUEUE
log "the checker"
$CHECK > logs/r24_check.out 2>&1 || { BRANCH="check failed"; tail -n 2 logs/r24_check.out; exit 1; }
tail -n 1 logs/r24_check.out
log "the registered report"
$REPORT > logs/R24_report.out 2>&1 || { BRANCH="report failed"; exit 1; }
log "the two-level intervals (no rule)"
$TWOLEVEL > logs/R24_twolevel.out 2>&1 || { BRANCH="two-level report failed (the registered report is written)"; exit 1; }
BRANCH="done"; log "R24 done; logs/R24_report.out $(sha256sum logs/R24_report.out | cut -c1-8), logs/R24_twolevel.out $(sha256sum logs/R24_twolevel.out | cut -c1-8)"
