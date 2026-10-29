"""The end check AT THE STOP of a call (revision 6; registration 4.1 and 11.6; round-5 review Major 1 (a)).

run_e4_gpu.sh runs it right after a client has ended without its own end check, while the server that served the call
is still up (the operator stops the client only, never the server; section 11.3):
  - the operator's stop at a slot's end (slot_end: TERM or INT to the script; or a client ended by TERM or INT);
  - a failed client (exit 3; also every failed attempt, so a stage's third failed attempt has its end record before
    its next start logs SKIP [stage_failed]).
It gives the stopped call's OWN start record (the run's manifest.json, or the judge's <out>.start.json) the light end
check of e4_run.identity_end_check (the pid, /props, /v1/models, the argv and the executable path, against the pins and
that start record) and writes it there as server_identity_end with "at_stop": true (e4_run.at_stop_end_check). Only a
record written by the stopped call (its "utc" at or after --since, the client's start) and without its end is checked;
any other record is left untouched for the deferred check at the run's next start, which runs only after the full start
check has found the pinned server (e4_run.start_checks).
Prints the kind on the first line ("same", "restart_same_identity", "identity_changed", or "none" when nothing was
checked) and the differences on the second. Exit 0 whenever it ran (the kind is the outcome; run_e4_gpu.sh tags
"identity_changed" [identity_changed], an integrity stop, section 9.1); 2 on a usage error.
Run: .../taubench/.venv/bin/python -I -B end_check_at_stop.py --record runs/<c>/<run>/manifest.json \
       --since <UTC of the client's start> --base http://127.0.0.1:18082/v1 --expect-server pins/server_expected.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

from e4_run import at_stop_end_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="the stopped call's start record (manifest.json or <out>.start.json)")
    ap.add_argument("--since", required=True, help="the client's start, UTC %%Y-%%m-%%dT%%H:%%M:%%SZ")
    ap.add_argument("--base", required=True)
    ap.add_argument("--expect-server", required=True, help="the frozen server pins (pins/server_expected.json)")
    a = ap.parse_args()
    pins = json.loads(Path(a.expect_server).read_text())
    kind, diffs = at_stop_end_check(a.base, pins, Path(a.record), a.since)
    print(kind or "none")
    print("; ".join(diffs)[:2000])


if __name__ == "__main__":
    main()
