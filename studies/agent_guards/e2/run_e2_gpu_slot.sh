#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# E2 (AgentDojo v1.2.2, Qwen acting, TM tool guard) -- FOR THE GPU SLOT ONLY. NOT RUN YET (written 2026-10-21, rev3;
# rev4 2026-10-22: the D6 probe's exit code goes to the E2-0 gate; the error flood is checked at every later stage end;
# rev5 2026-10-22: the budget rule's cut (iv), E2-3's no-rule replay arms D2, D3 and W-only, is applied at --stage arms).
# It does NOT start or stop any server. It uses the llama-server that is already running on
# 127.0.0.1:<port>, and it refuses to do anything unless scripts/e2_check.py passes: every pinned file and both venvs
# unchanged, the server's identity (build, GGUF sha256, one slot, flags; after E2-0 equal to E2-0's record), the
# replay store byte-identical to the last stage's end record, and (every stage after E2-0) the write-once freeze record.
#
#   bash run_e2_gpu_slot.sh --go --port <port> --stage smoke     # E2-0 (blind; fixes only the budget and stop/go)
#   # the freeze (CPU, no GPU, no server): complete sections 11-13 and 15 from E2-0's open files, then
#   #   .venv-dl/bin/python -I -B scripts/freeze_e2.py --registration <the completed registration>
#   # which writes runs/E2_frozen.json (write-once); every stage below refuses to start without it
#   bash run_e2_gpu_slot.sh --go --port <port> --stage d0        # E2-1  D0 benign + A1 (also the training traces)
#   bash run_e2_gpu_slot.sh --go --port <port> --stage d1        # E2-2  D1 tool_filter (unless the budget rule cut it)
#   bash run_e2_gpu_slot.sh --go --port <port> --stage build --d0-run runs/E2-1/<stamp>
#                                                               # D6 writer (GPU) first, then the CPU fits, E2.5, checks
#   bash run_e2_gpu_slot.sh --go --port <port> --stage arms      # E2-3  D3r D4 D5 D6 (+ D2 D3 W-only unless cut) by
#                                                               # paired replay; then the A3 gate
#   bash run_e2_gpu_slot.sh --go --port <port> --stage a2        # E2-4  A2 on 300 pairs (unless cut)
#   bash run_e2_gpu_slot.sh --go --port <port> --stage a3        # E2-5  A3: the gate's per-arm plan (runs/gate_a3.json)
# A stage that stops on a transport error (exit 3; connection refused, timeout, HTTP >= 500) is logged and gets ONE
# rerun, in full, on the same store:  bash run_e2_gpu_slot.sh --go --port <port> --stage <stage> --rerun [--d0-run ...]
# (E2-0 is never rerun on its campaign: move runs/ aside unopened and start a fresh campaign.)
# Options: --model <alias> (default qwen3.5-9b-q8_0); anything after -- goes to the driver.
# Client side: no GPU (CUDA_VISIBLE_DEVICES= ), 2 threads, nice 19; heavy CPU steps wait for the cgroup to be quiet.
set -u
E2=${REPO_ROOT}/studies/agent_guards/e2
ADV=${REPO_ROOT}/studies/agent_guards/agentdojo/.venv/bin/python
DLV=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
GO=0; PORT=""; STAGE=""; MODEL=qwen3.5-9b-q8_0; D0RUN=""; RERUN=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --port) PORT=$2; shift 2 ;;
    --stage) STAGE=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --d0-run) D0RUN=$2; shift 2 ;;
    --rerun) RERUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
[ "$GO" = 1 ] || { echo "usage: bash $0 --go --port <port> --stage smoke|d0|d1|build|arms|a2|a3 [--rerun] [--model alias] [-- driver args]  (GPU slot only; nothing done)"; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "STOP: --port <number> is required (the shared llama-server port)"; exit 2; }
case "$STAGE" in smoke|d0|d1|build|arms|a2|a3) ;; *) echo "STOP: unknown --stage '$STAGE'"; exit 2 ;; esac
declare -A SID=([smoke]=E2-0 [d0]=E2-1 [d1]=E2-2 [build]=E2-build [arms]=E2-3 [a2]=E2-4 [a3]=E2-5)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$E2/runs/${SID[$STAGE]}/$STAMP
mkdir -p "$RUN" || exit 1
log=$RUN/slot.log
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$log"; }
ENVC=(env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false)
wait_cpu() { "$E2/scripts/wait_cpu.sh" 3.5 360 | tee -a "$log"; }
drv() { "${ENVC[@]}" nice -n 19 "$ADV" -I -B "$E2/scripts/run_e2_gpu.py" --port "$PORT" --model "$MODEL" "$@" "${EXTRA[@]}" >> "$log" 2>&1; }
check() { "${ENVC[@]}" nice -n 19 "$ADV" -I -B "$E2/scripts/e2_check.py" "$@" >> "$log" 2>&1; }
rep() { "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/report_e2.py" "$@" >> "$log" 2>&1; }
plan_has() { "$DLV" -I -c "import json,sys; d=json.load(open('$E2/runs/budget.json')); sys.exit(0 if d['plan'].get('$1', True) else 1)"; }
# E2.5's runtime audit at the stage end (exit 1 on any ALLOW with a W1-W2 premise or any bit mismatch: a bug)
audit() { rep audit --dirs "$1" --out "$2" || { say "STOP: E2.5's runtime audit failed ($2): a bug; E2 stops and the failure is logged"; return 1; }; }
# rev4 (round-3 M-2): the error flood at every stage end after E2-0 (exit 1 when an arm, an A3 arm and variant, or a D6
# fold has errors in more than 1% of its episodes or rounds: a bug)
errs() { rep errors --dirs "$1" --out "$2" || { say "STOP: error flood ($2): a bug; E2 stops and the failure is logged"; return 1; }; }
SAMPLER=""
start_sampler() {  # read-only GPU samples every 30 s while the stage runs (another compute process = contaminated timing)
  ( while true; do echo "# $(date -u +%H:%M:%SZ)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -c 400; sleep 30; done ) >> "$RUN/gpu_samples.log" 2>&1 &
  SAMPLER=$!
}
stop_sampler() { [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; SAMPLER=""; }
trap stop_sampler EXIT

say "start stage $STAGE (${SID[$STAGE]}) port $PORT model $MODEL rerun $RERUN extra: ${EXTRA[*]:-}"
say "GPU before (read-only): $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -c 200) | apps: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -c 300)"
# the budget rule (runs/budget.json, written by E2-0, final at the freeze) can cut d1, a2 and (rev5, cut iv) E2-3's
# no-rule replay arms D2, D3 and W-only; a3 runs the gate's plan (the gate reads the plan's A3 cuts)
E23=()
case "$STAGE" in
  d1) plan_has d1 || { say "SKIPPED: the budget rule cut D1 (runs/budget.json); nothing done"; exit 0; } ;;
  a2) plan_has a2 || { say "SKIPPED: the budget rule cut A2 (runs/budget.json); nothing done"; exit 0; } ;;
  arms) plan_has e2_3_no_rule_arms || { E23=(--arms d3r d4_tm d5_lr d5_tree d6_rules)
    say "the budget rule cut E2-3's no-rule replay arms D2, D3 and W-only (runs/budget.json, cut iv): E2-3 replays ${E23[*]:1}"; } ;;
  a3) [ -f "$E2/runs/gate_a3.json" ] || { say "STOP: no runs/gate_a3.json (E2-3 writes it)"; exit 1; } ;;
esac
# 1. pins, venvs, server identity, store chain, the freeze record (and models/final for arms, a2, a3); no server start
RR=(); [ "$RERUN" = 1 ] && RR=(--rerun)
check --stage "$STAGE" --begin "${RR[@]}" --port "$PORT" || { say "STOP: e2_check failed (see above in $log); nothing done"; exit 1; }
say "e2_check passed"
start_sampler
e=0
d6e=-1
case "$STAGE" in
  smoke)
    wait_cpu
    drv --stage smoke --out "$RUN/driver"; e=$?
    if [ $e = 0 ]; then  # the D6 rule writer on one full fold (<= 10 rounds) of ground-truth rows, through the store
      "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/build_d6.py" --proposer qwen --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
        --rows "$E2/data/rows_gt_v1.2.2.jsonl" --qbits "$E2/data/query_bits" --out "$RUN/d6_probe" --folds banking >> "$log" 2>&1
      d6e=$?
      say "D6 probe exit $d6e"
      if [ $d6e = 3 ]; then e=3; say "E2-0 STOPPED: a transport error in the D6 probe ($RUN/d6_probe/STOPPED.json)";
      elif [ $d6e != 0 ]; then say "D6 probe failed (exit $d6e): E2-0's gate is not go (rev4: a bug; registration section 12)"; fi
    fi
    stop_sampler
    if [ $e = 0 ]; then audit "$RUN/driver" "$RUN/driver/sealed/audit_E2.5.json" || e=1; fi
    if [ $e = 0 ] && [ ! -e "$E2/runs/budget.json" ]; then
      cp "$RUN/driver/summary.json" "$E2/runs/E2-0_summary.json" || e=1
      rep smoke-gate --smoke "$RUN/driver/summary.json" --d6-probe "$RUN/d6_probe/d6_rules_banking.json" \
        --d6-probe-exit "$d6e" --out "$E2/runs/E2-0_gate.json" || e=1
      "$DLV" -I -c "import json,sys; d=json.load(open('$E2/runs/E2-0_gate.json')); sys.exit(0 if d['go'] else 1)" \
        || say "STOP/GO: E2-0's gate is NOT go (runs/E2-0_gate.json lists the failed checks); no later stage will start"
      rep budget --smoke "$RUN/driver/summary.json" --sealed "$RUN/driver/sealed/summary_sealed.json" \
        --d6-probe "$RUN/d6_probe/d6_rules_banking.json" --gpu-samples "$RUN/gpu_samples.log" \
        --out "$E2/runs/budget.json" --detail-out "$E2/runs/sealed/budget_detail.json" || e=1
    fi ;;
  d0|d1|arms|a2)
    wait_cpu
    drv --stage "$STAGE" --out "$RUN/driver" "${E23[@]}"; e=$?
    if [ $e = 0 ]; then audit "$RUN/driver" "$RUN/audit_E2.5.json" || e=1; fi
    if [ $e = 0 ]; then errs "$RUN/driver" "$RUN/errors.json" || e=1; fi
    if [ "$STAGE" = arms ] && [ $e = 0 ]; then  # the E2-5 gate from E2-3's (b) rules, D6's testability and the budget plan
      rep gate-a3 --runs "$RUN/driver" --d6 "$E2/models/final" --smoke "$E2/runs/E2-0_summary.json" \
        --budget "$E2/runs/budget.json" --out "$E2/runs/gate_a3.json" || e=1
    fi ;;
  a3)
    ARMS=$("$DLV" -I -c "import json; d=json.load(open('$E2/runs/gate_a3.json')); print(' '.join(d['arms']))")
    if [ -z "$ARMS" ]; then say "SKIPPED: the gate runs no A3 (runs/gate_a3.json); nothing done"; e=0
    else
      wait_cpu
      drv --stage a3 --gate "$E2/runs/gate_a3.json" --out "$RUN/driver"; e=$?
      if [ $e = 0 ]; then audit "$RUN/driver" "$RUN/audit_E2.5.json" || e=1; fi
      if [ $e = 0 ]; then errs "$RUN/driver" "$RUN/errors.json" || e=1; fi
    fi ;;
  build)
    [ -n "$D0RUN" ] && [ -f "$D0RUN/driver/rows_d0.jsonl" ] || { say "STOP: --d0-run <E2-1 run dir with driver/rows_d0.jsonl> is required"; exit 1; }
    R0="$D0RUN/driver/rows_d0.jsonl"; MF="$E2/models/final"
    # GPU part first: the D6 rules (4 folds x <= 10 rounds, through the store); then the GPU can go back
    mkdir -p "$MF.d6tmp" || exit 1
    "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/build_d6.py" --proposer qwen --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
      --rows "$E2/data/rows_gt_v1.2.2.jsonl" "$R0" --qbits "$E2/data/query_bits" --out "$MF.d6tmp" >> "$log" 2>&1
    d6e=$?
    if [ $d6e = 3 ]; then  # a transport stop: keep the partial rules in the run folder (never deleted), log the stop
      mv "$MF.d6tmp" "$RUN/d6tmp_stopped" || exit 1
      e=3
    elif [ $d6e != 0 ]; then say "STOP: build_d6 failed (a bug; E2 stops and the failure is logged)"; exit 1
    fi
    stop_sampler
    if [ $e = 0 ]; then errs "$MF.d6tmp" "$RUN/errors_d6.json" || exit 1; fi  # rev4: request errors in the D6 rounds
    if [ $e = 0 ]; then
      say "GPU part of the build done (D6 rules in $MF.d6tmp); the CPU part follows"
      wait_cpu
      "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/train_loso.py" --rows "$E2/data/rows_gt_v1.2.2.jsonl" "$R0" \
        --qbits "$E2/data/query_bits" --out "$MF" --tag final >> "$log" 2>&1 || { say "STOP: train_loso failed"; exit 1; }
      if ! mv "$MF.d6tmp"/d6_rules_*.json "$MF"/; then say "STOP: moving the D6 rules failed"; exit 1; fi
      rmdir "$MF.d6tmp" || exit 1
      wait_cpu
      "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/milp_check.py" --models "$MF" --rows "$E2/data/rows_gt_v1.2.2.jsonl" "$R0" \
        --qbits "$E2/data/query_bits" --out "$MF/milp_E2.5.jsonl" >> "$log" 2>&1 || { say "STOP: milp_check failed"; exit 1; }
      # E2.5 at the build stage's end: 11 premises proved per fold, every row in the domain, 500/500 encoding votes,
      # train_loso's exactness checks; a failure is a bug: no chain record and no models_final_pins, so E2-3 cannot start
      rep e25-build --models "$MF" --out "$RUN/e25_build_check.json" \
        || { say "STOP: E2.5 failed at the build stage ($RUN/e25_build_check.json): a bug; E2 stops before E2-3 and the failure is logged"; exit 1; }
      wait_cpu  # the readings with no rule: the training-seed check and DESIGN's literal label rule
      "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/seed_check.py" --models "$MF" --rows "$E2/data/rows_gt_v1.2.2.jsonl" "$R0" \
        --eval-rows "$R0" --qbits "$E2/data/query_bits" --out "$MF/seed_check.json" >> "$log" 2>&1 || say "seed check failed (a reading with no rule; recorded)"
      wait_cpu
      "${ENVC[@]}" nice -n 19 "$DLV" -I -B "$E2/scripts/train_loso.py" --rows "$E2/data/rows_gt_v1.2.2.jsonl" "$R0" \
        --qbits "$E2/data/query_bits" --out "$MF/diag_literal_labels" --tag diag_literal --label-rule design_literal --inner-only >> "$log" 2>&1 || say "literal-label diagnostic failed (no rule; recorded)"
    fi ;;
esac
stop_sampler
say "stage exit $e"
if [ $e = 0 ]; then
  check --stage "$STAGE" --end --run-dir "$RUN" || { say "STOP: e2_check --end failed"; e=1; }
elif [ $e = 3 ]; then  # the transport rule (registration section 13): no chain record; the stop is logged
  check --stage "$STAGE" --stopped --run-dir "$RUN" --exit-code 3 || say "STOP: logging the stop failed"
  if [ "$STAGE" = smoke ]; then
    say "E2-0 STOPPED (transport error): E2-0 reruns on a fresh campaign; move runs/ aside unopened (mv runs runs.aside.<stamp>)"
  else
    say "STOPPED (transport error): rerun this stage ONCE, in full, on the same store: bash $0 --go --port <port> --stage $STAGE --rerun"
  fi
else
  say "STOP: the stage failed (not a transport error): a bug or a failed check; E2 stops and the failure is logged"
fi
say "GPU after (read-only): $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -c 200)"
say "SLOT DONE (the server was not touched)"
exit "$e"
# ---------------------------------------------------------------------------------------------------------------------
# The report step (registration rev4, section 16). After E2-5 (or its SKIPPED line), CPU only, decision-layer venv;
# <s0>..<s5> are the stamps of the run folders runs/E2-0/<s0>, runs/E2-1/<s1>, runs/E2-2/<s2>, runs/E2-3/<s3>,
# runs/E2-4/<s4>, runs/E2-5/<s5> (leave out a stage the budget rule cut or the gate skipped; for a stage that was
# rerun, give the rerun's folder, never the stopped one: report_e2.py refuses a folder with STOPPED.json):
#   cd studies/agent_guards/e2
#   env -i PATH=/usr/bin:/bin HOME=~ CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
#     studies/decision_layer/.venv-dl/bin/python -I -B scripts/report_e2.py final \
#     --runs runs/E2-1/<s1>/driver runs/E2-2/<s2>/driver runs/E2-3/<s3>/driver runs/E2-4/<s4>/driver \
#     --a3 runs/E2-5/<s5>/driver --models models/final --smoke runs/E2-0_summary.json \
#     --audit-dirs runs/E2-0/<s0>/driver/sealed \
#     --budget runs/budget.json --seed-check models/final/seed_check.json \
#     --literal-log models/final/diag_literal_labels/train_log.json --registered-log models/final/train_log.json \
#     --rows-d0 runs/E2-1/<s1>/driver/rows_d0.jsonl --out runs/report_E2.json --md runs/report_E2.md
#   then: sha256sum runs/report_E2.json runs/report_E2.md (into LOG.md), and the verify-result skill before any verdict.
# The A3 gate (written by the arms stage above), the budget rule and the stop/go gate (written by the smoke stage), by hand:
#   ... scripts/report_e2.py gate-a3 --runs runs/E2-3/<s3>/driver --d6 models/final --smoke runs/E2-0_summary.json \
#       --budget runs/budget.json --out runs/gate_a3.json
#   ... scripts/report_e2.py budget --smoke runs/E2-0/<s0>/driver/summary.json \
#       --sealed runs/E2-0/<s0>/driver/sealed/summary_sealed.json --d6-probe runs/E2-0/<s0>/d6_probe/d6_rules_banking.json \
#       --gpu-samples runs/E2-0/<s0>/gpu_samples.log --out runs/budget.json --detail-out runs/sealed/budget_detail.json
#   ... scripts/report_e2.py smoke-gate --smoke runs/E2-0/<s0>/driver/summary.json \
#       --d6-probe runs/E2-0/<s0>/d6_probe/d6_rules_banking.json --d6-probe-exit <slot.log's "D6 probe exit" value> \
#       --out runs/E2-0_gate.json
#   ... scripts/report_e2.py errors --dirs runs/<stage>/<stamp>/driver --out <json>   # the error flood (rev4)
# runs/E2-0_summary.json is the smoke stage's copy of E2-0's open summary, hashed in the freeze record (e2_check checks
# it at every stage start): the A3 gate and the report read the optimizer refusal rule from it.
# The E2-0 outcomes in runs/E2-0/<s0>/driver/sealed/ and the store runs/sealed/ stay unopened until the registration is
# frozen (runs/E2_frozen.json); only the driver, build_d6.py, e2_check.py and report_e2.py's mechanical budget read them.
