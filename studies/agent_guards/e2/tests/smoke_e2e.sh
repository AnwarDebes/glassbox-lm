#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# End-to-end smoke test on tiny data (no LLM, no server, no GPU): the real entry points with tiny configurations,
# then tests/check_smoke.py asserts the outputs. Usage: bash tests/smoke_e2e.sh <empty output dir inside e2/>
# rev2: also the blind E2-0 driver stage, A2 and A3 with the fake client, report_e2.py (stand-in, fake logs, gate,
# budget), the seed check, the DESIGN-literal label diagnostic, the decision-layer tests and e2_check.py.
# rev3: the transport rule against the stub (HTTP 500 and refused connections stop a stage with exit 3; HTTP 400 is
# the error rule), the A3 gate file (black-box on D6 refused), the budget with its sealed inputs, E2.5's audit and
# build checks, and the driver's dry run of every stage.
# rev4: the error flood (report_e2.py errors) on the fake stages, the HTTP 400 run and the D6 rule file; the E2-0 gate
# with a D6 probe file and its exit code.
set -u
E2=${REPO_ROOT}/studies/agent_guards/e2
T=${1:?output dir}
[ -e "$T" ] && { echo "STOP: $T exists"; exit 2; }
mkdir -p "$T" || exit 2
cd "$E2" || exit 2
# rev3: every temporary file of the steps goes to $T/tmp (inside this folder), never the shared /tmp
mkdir -p "$T/tmp" || exit 2
export TMPDIR=$T/tmp
ENVC=(env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false TMPDIR="$T/tmp" nice -n 19)
AD=$E2/../agentdojo/.venv/bin/python
DL=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
BAD=0
# run EXPECTED_EXIT cmd...: logs the command and its exit code; any other exit code marks the smoke test failed
run() { local want=$1; shift; echo "== $(date -u +%H:%M:%SZ) $*"; "$@"; local e=$?; echo "== exit $e (expected $want)"; [ "$e" = "$want" ] || BAD=1; }
run 0 "${ENVC[@]}" "$AD" -I -B scripts/make_rows.py --suites banking --out "$T/rows_banking.jsonl"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/train_loso.py --rows data/rows_gt_v1.2.2.jsonl --qbits data/query_bits --out "$T/models" --folds banking --quick --tag smoke
run 0 "${ENVC[@]}" "$DL" -I -B scripts/train_loso.py --rows data/rows_gt_v1.2.2.jsonl --qbits data/query_bits --out "$T/diag_literal" --folds banking --quick --tag smoke --label-rule design_literal --inner-only
run 1 "${ENVC[@]}" "$DL" -I -B scripts/train_loso.py --rows data/rows_gt_v1.2.2.jsonl --qbits data/query_bits --out "$T/refused" --folds banking --quick --label-rule design_literal
run 0 "${ENVC[@]}" "$DL" -I -B scripts/milp_check.py --models "$T/models" --rows data/rows_gt_v1.2.2.jsonl --qbits data/query_bits --out "$T/milp.jsonl" --folds banking --n-rows 50 --time-limit 60
run 0 "${ENVC[@]}" "$DL" -I -B scripts/build_d6.py --proposer scripted --rows data/rows_gt_v1.2.2.jsonl --qbits data/query_bits --out "$T/models" --folds banking --rounds 4
run 0 "${ENVC[@]}" "$DL" -I -B scripts/seed_check.py --models "$T/models" --rows data/rows_gt_v1.2.2.jsonl --eval-rows data/rows_standin_traces.jsonl --qbits data/query_bits --out "$T/seed_check.json" --folds banking --seeds 2
run 0 "${ENVC[@]}" "$AD" -I -B scripts/offline_e2e.py --models "$T/models" --qbits data/query_bits --out "$T/e2e.json" --suites banking --arms d0_none d0_guard_allow w_only d2 d3 d3r d4_tm d5_lr d5_tree d6_rules d1_tool_filter
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage smoke --fake --out "$T/drv_smoke" --store "$T/store.jsonl"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage smoke --fake --fake-follow --out "$T/drv_smoke_follow" --store "$T/store_follow.jsonl"
# the real-server code path against an in-process stub (tests/stub_server.py: not an LLM): the driver's non-fake
# branches (HTTP identity, optimizer and rule-writer calls through the store, the determinism re-sends), and
# e2_check.py's refusal of a server that is not the pinned one (it writes nothing: the identity check fails first)
# stub_up NAME [stub flags]: start tests/stub_server.py in the background; sets STUB (pid) and SP (port)
stub_up() { local n=$1; shift; "$AD" -I -B tests/stub_server.py --port-file "$T/port_$n" --seconds 900 "$@" & STUB=$!
  for _ in $(seq 1 100); do [ -s "$T/port_$n" ] && break; sleep 0.2; done; SP=$(cat "$T/port_$n"); }
# the stub's identity as a reference record (the driver's later stages compare the server with E2-0's record)
ident() { "$AD" -I -B -c "import json,sys; sys.path.insert(0,'$E2/scripts'); import e2_check as EC; print(json.dumps(EC.server_identity($SP, hash_gguf=False)))" > "$1"; }
stub_up ok
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage smoke --port "$SP" --out "$T/drv_smoke_stub" --store "$T/store_stub.jsonl"
run 1 bash -c "env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= TMPDIR=$T/tmp $AD -I -B scripts/e2_check.py --stage smoke --begin --port $SP > $T/e2_check_stub.log 2>&1"
kill "$STUB" 2>/dev/null
# rev3, the transport rule: HTTP 500 after 3 answered requests stops E2-0 (exit 3, STOPPED.json, no summary)
stub_up s500 --fail-after 3 --fail-status 500
run 3 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage smoke --port "$SP" --out "$T/drv_stop500" --store "$T/store_stop500.jsonl"
kill "$STUB" 2>/dev/null
# refused connections after 2 answered requests stop E2-1 (exit 3)
stub_up refuse --refuse-after 2
ident "$T/identity_refuse.json"
run 3 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage d0 --port "$SP" --identity-ref "$T/identity_refuse.json" --suites banking --max-jobs-per-suite 4 --out "$T/drv_stop_refused" --store "$T/store_refused.jsonl"
kill "$STUB" 2>/dev/null
# HTTP 400 is the error rule: the stage goes on (exit 0) and every episode carries the error
stub_up s400 --fail-after 0 --fail-status 400
ident "$T/identity_400.json"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage d0 --port "$SP" --identity-ref "$T/identity_400.json" --suites banking --max-jobs-per-suite 3 --out "$T/drv_400" --store "$T/store_400.jsonl"
kill "$STUB" 2>/dev/null
# HTTP 503 on the A3 optimizer's first request stops E2-5 (exit 3); on the D6 rule writer it stops the build (exit 3)
stub_up s503 --fail-after 0 --fail-status 503
ident "$T/identity_503.json"
run 3 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage a3 --arms d4_tm --models "$T/models" --suites banking --limit-targets 1 --port "$SP" --identity-ref "$T/identity_503.json" --out "$T/drv_a3_stop" --store "$T/store_a3stop.jsonl"
run 3 "${ENVC[@]}" "$DL" -I -B scripts/build_d6.py --proposer qwen --base-url "http://127.0.0.1:$SP/v1" --rows data/rows_gt_v1.2.2.jsonl --qbits data/query_bits --out "$T/d6_stop" --folds banking --rounds 2 --store "$T/store_d6stop.jsonl"
kill "$STUB" 2>/dev/null
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py smoke-gate --smoke "$T/drv_smoke_stub/summary.json" --out "$T/smoke_gate_stub.json"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage d0 --fake --fake-follow --suites banking --max-jobs-per-suite 20 --out "$T/drv_d0" --store "$T/pair_store.jsonl"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/traces_to_rows.py --source d0_trace --out "$T/rows_from_d0_traces.jsonl" "$T/drv_d0/traces"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage arms --arms d0_allow d3r d4_tm d6_rules --models "$T/models" --fake --fake-follow --suites banking --max-jobs-per-suite 20 --out "$T/drv_arms" --store "$T/pair_store.jsonl"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage a2 --arms d0 d3r d4_tm --models "$T/models" --fake --fake-follow --suites banking --max-jobs-per-suite 6 --out "$T/drv_a2" --store "$T/pair_store.jsonl"
printf '%s\n' '{"plan": [{"arm": "d4_tm", "variants": ["guard_aware"]}, {"arm": "d3r", "variants": ["guard_aware"]}]}' > "$T/gate_ok.json"
printf '%s\n' '{"plan": [{"arm": "d4_tm", "variants": ["guard_aware"]}, {"arm": "d6_rules", "variants": ["guard_aware", "black_box"]}]}' > "$T/gate_bad.json"
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage a3 --gate "$T/gate_ok.json" --models "$T/models" --fake --fake-follow --suites banking --out "$T/drv_a3" --store "$T/pair_store.jsonl"
run 1 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage a3 --gate "$T/gate_bad.json" --models "$T/models" --fake --suites banking --out "$T/drv_a3_badgate" --store "$T/pair_store.jsonl"
run 1 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage a3 --models "$T/models" --fake --suites banking --out "$T/drv_a3_noarms" --store "$T/pair_store.jsonl"
for st in smoke d0 d1 a2; do run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage "$st" --dry-run --arms d0; done
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage arms --dry-run --models "$T/models" --suites banking
run 0 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage a3 --dry-run --gate "$T/gate_ok.json" --models "$T/models" --suites banking
run 1 "${ENVC[@]}" "$AD" -I -B scripts/run_e2_gpu.py --stage d0 --out "$T/drv_noport"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py standin --standin results/offline_e2e_standin.json --out "$T/report_standin.json" --md "$T/report_standin.md"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py final --runs "$T/drv_d0" "$T/drv_arms" "$T/drv_a2" --a3 "$T/drv_a3" --models "$T/models" --smoke "$T/drv_smoke/summary.json" --seed-check "$T/seed_check.json" --literal-log "$T/diag_literal/train_log.json" --registered-log "$T/models/train_log.json" --rows-d0 "$T/drv_d0/rows_d0.jsonl" --out "$T/report_fake.json" --md "$T/report_fake.md"
run 1 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py final --runs "$T/drv_smoke" --out "$T/report_sealed.json"
run 1 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py gate-a3 --runs "$T/drv_arms" --d6 "$T/models" --smoke "$T/drv_smoke/summary.json" --out "$T/gate.json"
# rev3: an incomplete E2-3 gets no gate (above); the gate's command path on a complete E2-3 (the stand-in episodes of
# D4, D3r and D6 written as runs.jsonl), a testable D6 folder and a budget plan that keeps black-box
run 0 "$DL" -I -B -c "
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import report_e2 as R
out = Path('$T/standin_e23'); out.mkdir()
eps = [e for e in R.standin_episodes('results/offline_e2e_standin.json') if e['arm'] in ('d4_tm', 'd3r', 'd6_rules')]
(out / 'runs.jsonl').write_text(''.join(json.dumps(e) + chr(10) for e in eps))
d6 = Path('$T/d6_testable'); d6.mkdir()
for s in R.SUITES:
    (d6 / f'd6_rules_{s}.json').write_text(json.dumps({'rules': [['kind=read']], 'nontrivial': {'allows_state_changing': True}}))
Path('$T/budget_keep.json').write_text(json.dumps({'plan': {'a3_black_box': True, 'a3_d6': True}}))
"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py gate-a3 --runs "$T/standin_e23" --d6 "$T/d6_testable" --smoke "$T/drv_smoke/summary.json" --budget "$T/budget_keep.json" --out "$T/gate_standin.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py budget --smoke "$T/drv_smoke/summary.json" --sealed "$T/drv_smoke/sealed/summary_sealed.json" --out "$T/budget.json" --detail-out "$T/budget_detail.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py budget --smoke "$T/drv_smoke_follow/summary.json" --sealed "$T/drv_smoke_follow/sealed/summary_sealed.json" --out "$T/budget_follow.json" --detail-out "$T/budget_follow_detail.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py audit --dirs "$T/drv_arms" "$T/drv_a2" "$T/drv_a3" --out "$T/audit_stages.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py audit --dirs "$T/drv_smoke_follow/sealed" --out "$T/audit_smoke.json"
# rev4 (round-3 M-2): the error flood at the stage ends, and the gate's D6-probe check on a stand-in rule file
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py errors --dirs "$T/drv_arms" "$T/drv_a2" "$T/drv_a3" --out "$T/errors_stages.json"
run 1 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py errors --dirs "$T/drv_400" --out "$T/errors_400.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py errors --dirs "$T/models" --out "$T/errors_d6.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py smoke-gate --smoke "$T/drv_smoke_stub/summary.json" --d6-probe "$T/models/d6_rules_banking.json" --d6-probe-exit 0 --out "$T/smoke_gate_stub_d6.json"
run 0 bash -c "mkdir -p $T/e25 && cp $T/models/train_log.json $T/e25/ && cp $T/milp.jsonl $T/e25/milp_E2.5.jsonl"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py e25-build --models "$T/e25" --folds banking --n-rows 50 --out "$T/e25_build.json"
run 1 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py e25-build --models "$T/e25" --out "$T/e25_build_registered.json"
run 0 "${ENVC[@]}" "$DL" -I -B scripts/report_e2.py smoke-gate --smoke "$T/drv_smoke/summary.json" --out "$T/smoke_gate.json"
run 0 "${ENVC[@]}" "$DL" -I -B tests/dl_side_tests.py
run 0 "${ENVC[@]}" "$AD" -I -B scripts/e2_check.py --pins --venvs
run 2 bash run_e2_gpu_slot.sh --go --stage smoke
run 0 "$DL" -I -B tests/check_smoke.py "$T"
echo "smoke_e2e: steps $([ $BAD = 0 ] && echo ok || echo FAILED)"
exit $BAD
