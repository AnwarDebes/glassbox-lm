#!/usr/bin/env bash
# Dry test of logs/R24_chain.sh with stub commands in a temporary folder (no data, no GPU).
set -u
H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d); mkdir -p "$T/logs" "$T/data/multiwoz" "$T/data/emb" "$T/registrations" "$T/data/untrusted/multiwoz22/test"
printf 'echo both missed\nexit 3\n' > "$T/gate3.sh"
echo "made-up dialogue" > "$T/data/untrusted/multiwoz22/test/dialogues_001.json"
(cd "$T/data/untrusted/multiwoz22" && sha256sum test/dialogues_001.json > MANIFEST.sha256)
M=$(sha256sum "$T/data/untrusted/multiwoz22/MANIFEST.sha256" | cut -d' ' -f1)
echo "R24 stub" > "$T/registrations/R24_multiwoz_random_negatives.md"
setup() {
  printf '{\n "registration": "registrations/R24_multiwoz_random_negatives.md",\n "sha256": "%s",\n "frozen_utc": "2026-10-22T20:00:00Z"\n}\n' \
    "$(sha256sum "$T/registrations/R24_multiwoz_random_negatives.md" | cut -d' ' -f1)" > "$T/registrations/R24_multiwoz_random_negatives.frozen.json"
  echo "e85 end: done 2026-10-22T23:00:00Z exit 0" > "$T/logs/e85_end.log"; printf 'ok  x\nall dry checks passed\n' > "$T/logs/R24_dry_check.out"
}
run() { R24_ROOT="$T" MWOZ_MANIFEST_SHA="$M" GATE_CMD="${G:-echo gate-ok}" NF_CMD="${N:-echo network-free ok}" PREP_CMD="touch $T/data/multiwoz/examples_test.json.gz" \
        EMBED_CMD="true" QUEUE_CMD="touch $T/logs/R24_queue.log" CHECK_CMD="$1" REPORT_CMD="echo report" TWOLEVEL_CMD="${W:-echo twolevel}" \
        bash "$H/R24_chain.sh" > "$T/out.txt" 2>&1; echo $?; }
clean() { rm -f "$T/logs/R24_end.log" "$T/logs/R24_gate.out" "$T/logs/R24_network_free.out" "$T/logs/R24_queue.log" "$T/logs/r24_check.out" \
                "$T/logs/R24_twolevel.out" "$T/data/multiwoz/examples_test.json.gz"; setup; }
unread() { [ ! -e "$T/data/multiwoz/examples_test.json.gz" ] && [ ! -e "$T/logs/R24_queue.log" ]; }
fail() { echo "FAIL $1"; cat "$T/out.txt"; exit 1; }
setup
[ "$(run 'echo all checks passed')" = 0 ] && grep -q "^R24 end: done .* exit 0$" "$T/logs/R24_end.log" && grep -q gate-ok "$T/logs/R24_gate.out" \
  && grep -q "network-free ok" "$T/logs/R24_network_free.out" && grep -q twolevel "$T/logs/R24_twolevel.out" || fail "done path"
[ "$(run 'echo again')" = 1 ] && tail -n 1 "$T/logs/R24_end.log" | grep -q "^R24 end: refused (logs/R24_end.log exists)" || fail "refusal"
clean
[ "$(run 'false')" = 1 ] && grep -q "^R24 end: check failed .* exit 1$" "$T/logs/R24_end.log" || fail "check path"
clean
[ "$(G="bash $T/gate3.sh" run 'echo all checks passed')" = 3 ] && grep -q "^R24 end: gate missed at s\* for both agents (test unread) .* exit 3$" "$T/logs/R24_end.log" \
  && unread || fail "gate-missed path"
clean
[ "$(G=false run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: gate check failed (test unread) .* exit 1$" "$T/logs/R24_end.log" && unread || fail "gate-error path"
clean
[ "$(N=false run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: network-free test failed (test unread) .* exit 1$" "$T/logs/R24_end.log" && unread || fail "network-free path"
clean
[ "$(W=false run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: two-level report failed (the registered report is written) .* exit 1$" "$T/logs/R24_end.log" \
  || fail "two-level path"
clean; echo "changed" >> "$T/registrations/R24_multiwoz_random_negatives.md"
[ "$(run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: refused (no freeze record of the registration's text) .* exit 1$" "$T/logs/R24_end.log" && unread || fail "freeze guard"
clean; echo "e85 end: s pick failed 2026-10-22T23:00:00Z exit 1" >> "$T/logs/e85_end.log"
[ "$(run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: refused (e85 has not ended done) .* exit 1$" "$T/logs/R24_end.log" && unread || fail "e85 guard"
clean; echo "FAIL dry" >> "$T/logs/R24_dry_check.out"
[ "$(run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: refused (the dry checks have not passed) .* exit 1$" "$T/logs/R24_end.log" && unread || fail "dry guard"
clean; echo "changed" >> "$T/data/untrusted/multiwoz22/test/dialogues_001.json"
[ "$(run 'echo all checks passed')" = 1 ] && grep -q "^R24 end: refused (the MultiWOZ source files do not match the pinned MANIFEST) .* exit 1$" "$T/logs/R24_end.log" \
  && unread || fail "MANIFEST guard"
echo "ok: done path (gate, network-free test, report, two-level); refusal on an existing end log; a failing check stops with 'check failed';" \
     "before any test file: both agents missing the gate (exit 3), a failing gate check, a failing network-free test, a changed registration," \
     "e85 not ended done, dry checks not passed, a source file off its MANIFEST; a failing two-level report"
echo PASSED; rm -rf "$T"
