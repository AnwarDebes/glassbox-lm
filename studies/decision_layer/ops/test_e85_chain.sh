#!/usr/bin/env bash
# Dry test of e85_chain.sh with stubs: it waits for the dry check file, runs both stages, skips e85b without jobs.
set -u; H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d); mkdir -p "$T/logs"
printf '# h\n' > "$T/logs/jobs_e85b.txt"
cat > "$T/pick.sh" <<'X'
[ "$2" = screen ] && echo "pick s 8"; [ "$2" = final ] && { echo "s*: refined 8"; touch "$T0/logs/jobs_R24.txt"; }; true
X
E85_ROOT="$T" NVSMI_CMD="true" QUEUE_CMD="bash -c touch\ $T/logs/queue_ran;:" PICK_CMD="env T0=$T bash $T/pick.sh" POLL_S=1 bash "$H/e85_chain.sh" > "$T/out.txt" 2>&1 &
pid=$!; sleep 2; [ ! -e "$T/logs/e85_pick_screen.out" ] || { echo "FAIL: did not wait"; exit 1; }
echo "all dry checks passed" > "$T/logs/R24_dry_check.out"; wait $pid; rc=$?
[ $rc -eq 0 ] && grep -q "^e85 end: done .* exit 0$" "$T/logs/e85_end.log" && [ -e "$T/logs/jobs_R24.txt" ] && grep -q "s\*: refined 8" "$T/logs/e85_pick_final.out" \
  || { echo "FAIL rc $rc"; cat "$T/out.txt"; exit 1; }
E85_ROOT="$T" bash "$H/e85_chain.sh" > /dev/null 2>&1; [ $? -eq 1 ] || { echo "FAIL: no refusal"; exit 1; }
echo "ok: waited for the dry check, ran screen and final, skipped e85b without jobs, end line written, refuses a second start"; echo PASSED; rm -rf "$T"
