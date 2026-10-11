#!/usr/bin/env python3
"""M0 environment check (`make env-check`): what machine is this, and what do the
installed Tsetlin Machine packages really look like (protocol rule 10).

    env_check.py [--no-gpu] [--strict] [--extra-import MODULE ...] [--results-dir DIR]
                 [--timeout-s 600]

Prints, and saves through tmlm.common.runlog into results/<timestamp>_envcheck/ :
  - cgroup limits (cpu.max, memory.max, memory.current) next to os.cpu_count(), which
    reports the host and must not be trusted here
  - GPU name and driver (nvidia-smi), nvcc version, RAPL energy counter readability
  - importability of numpy, pycuda, cupy, tmu, GraphTsetlinMachine: each import runs
    in its OWN SUBPROCESS, so a broken package is reported and the check goes on
  - inspect.signature printouts of the classes of GraphTsetlinMachine.tm, of
    GraphTsetlinMachine.graphs.Graphs, of tmu's TMAutoEncoder, and of the classes of
    PyHierarchicalTsetlinMachineCUDA.tm

The full report is also written to envcheck.json in the run dir.

GPU use: importing GraphTsetlinMachine.tm or PyHierarchicalTsetlinMachineCUDA.tm
creates a CUDA context (both import pycuda.autoinit or autoprimaryctx). `make
env-check` therefore runs this script under the shared GPU lock. With --no-gpu those
two modules are skipped and every probe runs with CUDA_VISIBLE_DEVICES set to empty.

Probes run one after the other: never more than one child process at a time.
Exit code: 0, or with --strict 1 when any probe failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402

MARKER = "TMLM_PROBE_JSON:"

# (module, needs a CUDA context at import)
IMPORT_PROBES = (
    ("numpy", False),
    ("pycuda", False),
    ("cupy", False),
    ("tmu", False),
    ("GraphTsetlinMachine", False),
    ("GraphTsetlinMachine.tm", True),
    ("PyHierarchicalTsetlinMachineCUDA", False),
    ("PyHierarchicalTsetlinMachineCUDA.tm", True),
)

# (module, class names or () for every class defined in the module, needs a CUDA context)
SIGNATURE_PROBES = (
    ("GraphTsetlinMachine.graphs", ("Graphs",), False),
    ("GraphTsetlinMachine.tm", (), True),
    ("tmu.models.autoencoder.autoencoder", ("TMAutoEncoder",), False),
    ("PyHierarchicalTsetlinMachineCUDA.tm", (), True),
)

IMPORT_CODE = f"""
import importlib, importlib.metadata, json, sys
module = importlib.import_module(sys.argv[1])
version = getattr(module, "__version__", None) or getattr(module, "VERSION_TEXT", None)
if version is None:
    top = sys.argv[1].split(".")[0]
    for dist in importlib.metadata.packages_distributions().get(top, []):
        version = importlib.metadata.version(dist)
        break
print({MARKER!r} + json.dumps({{
    "version": None if version is None else str(version),
    "file": getattr(module, "__file__", None),
}}))
"""

SIGNATURE_CODE = f"""
import importlib, inspect, json, sys
module = importlib.import_module(sys.argv[1])
wanted = sys.argv[2:]

def sig(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError) as exc:
        return "<no signature: %s>" % exc

classes = {{}}
for name, cls in inspect.getmembers(module, inspect.isclass):
    if wanted and name not in wanted:
        continue
    if not wanted and cls.__module__ != module.__name__:
        continue
    methods = {{}}
    for method_name, member in inspect.getmembers(cls, callable):
        if not method_name.startswith("_"):
            methods[method_name] = sig(member)
    classes[name] = {{"init": sig(cls), "methods": methods}}
print({MARKER!r} + json.dumps({{"file": getattr(module, "__file__", None), "classes": classes}}))
"""


def probe(code: str, argv: list[str], env: dict[str, str], timeout_s: float) -> dict:
    """Run one probe in a subprocess. Never raises: failures come back as a dict."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, *argv],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s} s"}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER) and proc.returncode == 0:
            return {"ok": True, **json.loads(line[len(MARKER) :])}
    tail = [line for line in proc.stderr.strip().splitlines() if line.strip()]
    return {
        "ok": False,
        "error": tail[-1] if tail else f"exit {proc.returncode}",
        "exit_code": proc.returncode,
        "stderr_tail": tail[-15:],
    }


def cgroup_report() -> dict:
    cgroup = {}
    for name in ("cpu.max", "memory.max", "memory.current", "memory.swap.max"):
        try:
            cgroup[name] = (runlog.CGROUP_DIR / name).read_text().strip()
        except OSError as exc:
            cgroup[name] = None
            cgroup[name + ".error"] = str(exc)
    cpus = None
    parts = (cgroup.get("cpu.max") or "").split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) > 0:
        cpus = int(parts[0]) / int(parts[1])
    cgroup["cpu_quota_cpus"] = cpus
    memory_max = cgroup.get("memory.max") or ""
    cgroup["memory_max_gib"] = int(memory_max) / 1024**3 if memory_max.isdigit() else None
    cgroup["os_cpu_count"] = os.cpu_count()
    return cgroup


def rapl_report() -> dict:
    """Which RAPL energy counters exist and whether this user can read them."""
    base = Path("/sys/class/powercap")
    counters = {}
    if base.is_dir():
        for zone in sorted(base.glob("intel-rapl*")):
            counter = zone / "energy_uj"
            try:
                counter.read_text()
                counters[str(counter)] = "readable"
            except OSError as exc:
                counters[str(counter)] = f"not readable: {type(exc).__name__}"
    return counters


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Environment check, saved under results/.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="skip modules that create a CUDA context")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any probe fails")
    parser.add_argument("--extra-import", action="append", default=[], metavar="MODULE",
                        help="also probe this import (repeatable), treated as GPU free")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args(argv)
    import_probes = (*IMPORT_PROBES, *((module, False) for module in args.extra_import))

    probe_env = dict(os.environ)
    if args.no_gpu:
        probe_env["CUDA_VISIBLE_DEVICES"] = ""

    config = {
        "script": "scripts/env_check.py",
        "no_gpu": args.no_gpu,
        "strict": args.strict,
        "timeout_s": args.timeout_s,
        "import_probes": [name for name, _ in import_probes],
        "signature_probes": [name for name, _, _ in SIGNATURE_PROBES],
    }
    failures = 0
    with runlog.RunLog(
        "envcheck", config, results_dir=args.results_dir, root=ROOT, sample_gpu=False
    ) as run:
        report: dict = {"no_gpu": args.no_gpu}

        print("== cgroup limits ==")
        report["cgroup"] = cgroup_report()
        for key, value in report["cgroup"].items():
            print(f"  {key}: {value}")

        print("== GPU and toolchain ==")
        report["nvidia_smi"] = runlog.nvidia_smi_info()
        report["nvcc"] = runlog.nvcc_version()
        report["rapl"] = rapl_report()
        print(f"  nvidia-smi: {report['nvidia_smi']['status']}")
        for gpu in report["nvidia_smi"]["gpus"]:
            print(f"    {gpu['name']}, driver {gpu['driver_version']}, {gpu['memory_total']}")
        print(f"  nvcc: {report['nvcc']}")
        for counter, state in report["rapl"].items():
            print(f"  rapl {counter}: {state}")

        print("== imports (one subprocess each) ==")
        report["imports"] = {}
        for module, needs_gpu in import_probes:
            if needs_gpu and args.no_gpu:
                result = {"ok": None, "skipped": "--no-gpu: import creates a CUDA context"}
            else:
                result = probe(IMPORT_CODE, [module], probe_env, args.timeout_s)
            report["imports"][module] = result
            if result["ok"] is None:
                print(f"  SKIP {module}: {result['skipped']}")
            elif result["ok"]:
                print(f"  OK   {module} version={result['version']} file={result['file']}")
            else:
                failures += 1
                print(f"  FAIL {module}: {result['error']}")

        print("== signatures (inspect.signature, one subprocess each) ==")
        report["signatures"] = {}
        for module, class_names, needs_gpu in SIGNATURE_PROBES:
            if needs_gpu and args.no_gpu:
                result = {"ok": None, "skipped": "--no-gpu: import creates a CUDA context"}
            else:
                result = probe(SIGNATURE_CODE, [module, *class_names], probe_env, args.timeout_s)
            report["signatures"][module] = result
            if result["ok"] is None:
                print(f"  SKIP {module}: {result['skipped']}")
                continue
            if not result["ok"]:
                failures += 1
                print(f"  FAIL {module}: {result['error']}")
                continue
            print(f"  {module} ({result['file']})")
            for class_name, entry in result["classes"].items():
                print(f"    class {class_name}{entry['init']}")
                for method_name, signature in entry["methods"].items():
                    print(f"      .{method_name}{signature}")

        imports = report["imports"].values()
        run.write_json("envcheck.json", report)
        run.add_metric("probes_failed", failures, "measured")
        run.add_metric("imports_ok", sum(1 for r in imports if r["ok"]), "measured")
        run.add_metric("imports_skipped", sum(1 for r in imports if r["ok"] is None), "measured")
        print(f"== summary: {failures} probe(s) failed, run dir {run.run_dir} ==")
        if args.strict and failures:
            run.set_exit_status(1)
    return run.exit_status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
