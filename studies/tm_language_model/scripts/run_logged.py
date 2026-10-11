#!/usr/bin/env python3
"""Run any command as a logged run (protocol rule 5).

    run_logged.py --name NAME --config cfg.json [options] -- <command...>

Creates results/<UTC timestamp>_<NAME>/ and fills it as described in
src/tmlm/common/runlog.py: config.json, env.json, run.json (git commit, source tree
hash, wall time, peak GPU memory, dataset hashes, exit status), metrics.json, log.txt
(the command's stdout and stderr), MANIFEST.json.

The command sees TMLM_RUN_DIR and TMLM_RUN_NAME. To report results it can call
`tmlm.common.runlog.write_metrics({...})` and `write_dataset_hashes({...})`, or write
$TMLM_RUN_DIR/metrics.json itself in the format
{"name": {"value": ..., "label": "measured" | "derived" | "unverified"}}.
The wrapper validates that file when the command ends. An invalid file is renamed to
metrics_rejected.json and the wrapper exits 65 if the command itself succeeded.

Options:
    --results-dir DIR      parent of the run dir (default $TMLM_RESULTS_DIR or results/)
    --dataset NAME=PATH    hash this file with sha256 before the run (repeatable)
    --no-gpu-sampler       do not poll nvidia-smi for peak GPU memory

Exit code: the command's exit code (128 + N when it died from signal N), 2 for usage
errors, 127 or 126 when the command cannot be started, 65 when the command succeeded
but its metrics file was rejected, 70 when the command succeeded but the run dir could
not be finalized. SIGTERM and SIGHUP are forwarded to the command.

The command gets the thread caps and PYTHONHASHSEED of tmlm.common.threadcaps even when
this script is started from a shell that never sourced env/env.sh, and it inherits the
lock descriptors of gpu_run.sh and cpu_slot.sh, so it can yield the GPU
(tmlm.common.locks) and nested lock calls work. Typical composition:

    scripts/gpu_run.sh shared -- scripts/mem_guard.py -- \\
        scripts/run_logged.py --name e1 --config cfg.json -- python experiments/e1.py

A terminal that goes away (`run_logged.py ... | head`, a closed pipe) does not
touch the run: the command keeps running and log.txt stays complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402


def say(text: str, stream=None) -> None:
    """print() that survives a broken terminal stream."""
    try:
        print(text, file=stream if stream is not None else sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


def detach_broken_streams() -> None:
    """Point a broken stdout or stderr at /dev/null, so that the interpreter's flush at
    exit cannot turn the command's exit code into 120."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stream.fileno())
                os.close(devnull)
            except (OSError, ValueError):
                pass


def main(argv: list[str]) -> int:
    if "--" in argv:
        cut = argv.index("--")
        own, command = argv[:cut], argv[cut + 1 :]
    else:
        own, command = argv, []
    parser = argparse.ArgumentParser(
        prog="run_logged.py",
        usage="run_logged.py --name NAME --config cfg.json [options] -- <command...>",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True, type=Path, help="JSON file: the full run config")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--no-gpu-sampler", action="store_true")
    args = parser.parse_args(own)
    if not command:
        parser.error("no command given after --")

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read --config {args.config}: {exc}")

    dataset_hashes = {}
    for item in args.dataset:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--dataset needs NAME=PATH, got {item!r}")
        try:
            dataset_hashes[name] = hashing.sha256_file(path)
        except OSError as exc:
            parser.error(f"cannot hash dataset {name!r}: {exc}")

    try:
        run = runlog.RunLog(
            args.name,
            config,
            results_dir=args.results_dir,
            root=ROOT,
            dataset_hashes=dataset_hashes,
            command=command,
            sample_gpu=not args.no_gpu_sampler,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        with run:
            runlog.install_signal_forwarding(run)
            print(f"[run_logged] run dir: {run.run_dir}")
            print(f"[run_logged] command: {command}")
            try:
                code = run.run_command(command)
            except runlog.CommandStartError as exc:
                # Only a failed START lands here. Whatever happens after the start
                # (a broken terminal included) is never mistaken for it.
                print(f"[run_logged] {exc}", file=sys.stderr)
                code = exc.exit_status
                run.set_exit_status(code)
            print(f"[run_logged] exit status: {code}")
    except runlog.RunLogFinalizeError:
        pass  # already reported on stderr and in run.json, exit_status is set
    if run.metrics_error:
        say(f"[run_logged] {run.metrics_error}", sys.stderr)
    for name, problem in run.stream_errors.items():
        say(f"[run_logged] {name} of the wrapper broke during the run ({problem}), "
            "log.txt is complete", sys.stderr)
    detach_broken_streams()
    return run.exit_status if run.exit_status is not None else 70


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
