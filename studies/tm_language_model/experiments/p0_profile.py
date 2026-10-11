"""T0.2: where does a training step of the STOCK GraphTsetlinMachine spend its time?

For ONE configuration at outputs=1000 and ONE at outputs=16000 (clauses 2000, depth 2,
hypervector 256, message_size 256, the two-node task) this script measures:

  1. The split of GPU time over the stock kernels, grouped the way the initial design asks:
     evaluate, select_clause_updates, update (+ update_message), calculate_messages*
     (calculate_messages + calculate_messages_conditional), other (everything else:
     select_clause_node, prepare/exchange/encode_messages, memcpy).
     Source: nvprof (`--print-gpu-summary --print-api-summary --csv`), because nsys is
     not installed here. Profiling starts AFTER the warm-up fit (`--profile-from-start
     off` plus cudaProfilerStart in the worker), so kernel builds and uploads are not in
     the numbers.
  2. The fraction of WALL time in which the GPU is busy, versus host overhead (Python,
     kernel launch, synchronize). The stock package launches its kernels one example at
     a time and synchronizes after each one. GPU time comes from nvprof (GPU timestamps,
     not inflated by profiler overhead on the host); wall time comes from a separate
     UN-INSTRUMENTED worker with the same seed, N and repetitions, so the profiler's host
     overhead does not enter the ratio.
  3. A cross-check with pycuda.driver.Event pairs around every kernel launch (the kernel
     attributes of the model INSTANCE are wrapped; the package is not touched). If nvprof
     fails, this is the fallback source of the split, and the output says so.

Kill criterion of the initial design (docs/08): "class loop (evaluate + select_clause_updates +
update) below about 40 percent of step time: drop the sparse-draw headline". The verdict is
computed for both readings: share of GPU time, and share of step (wall) time.

Outputs in the run dir: profile_split.csv, profile_kernels.csv, profile_summary.json,
workers/* (specs, results, logs, raw nvprof CSV), metrics.json.

Run (profiling, so the GPU lock is exclusive):
    python experiments/p0_profile.py --write-config /path/cfg.json
    scripts/gpu_run.sh exclusive -- scripts/run_logged.py --name p0_profile \
        --config /path/cfg.json -- python experiments/p0_profile.py --config /path/cfg.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as common  # noqa: E402

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "p0_profile",
    "brief_tasks": ["T0.2"],
    "package": "stock GraphTsetlinMachine (vendored master, unmodified)",
    "task": "two-node graphs, symbols A/B, label = hash(symbol pair) mod K plus label noise",
    "base": {
        "model": "multiclass",
        "clauses": 2000, "T": 1000, "s": 10.0, "q": 1.0, "max_included_literals": None,
        "depth": 2,
        "hypervector_size": 256, "hypervector_bits": 2,
        "message_size": 256, "message_bits": 2,
        "noise": 0.25, "seed": 42, "reps": 3,
        "grid": list(common.SAFE_GRID), "block": list(common.SAFE_BLOCK),
    },
    "outputs": [1000, 16000],
    "pilot_n": 200,
    "pilot_reps": 2,
    "target_rep_s": 8.0,
    "n_min": 200,
    "n_max": 20000,
    "max_label_bytes": 2**30,
    "nvprof": "/usr/local/cuda/bin/nvprof",
    "kill_threshold": 0.40,
}

UNIT_TO_MS = {"s": 1000.0, "ms": 1.0, "us": 1e-3, "ns": 1e-6}
_PREFIX_RE = re.compile(r"^==\d+==\s?")
GROUP_ORDER = ("evaluate", "select_clause_updates", "update", "calculate_messages", "other")


# --------------------------------------------------------------------------- nvprof CSV


def parse_nvprof_csv(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse the summary tables of `nvprof --csv` into rows with times in milliseconds.

    Returns {"gpu": [...], "api": [...]}; each row has name, time_ms, calls, avg_ms,
    min_ms, max_ms, percent. The file may hold one table with a "Type" column (GPU
    activities and API calls together) or separate tables; every table has a header row
    followed by a units row, and every table may use different units.
    """
    out: dict[str, list[dict[str, Any]]] = {"gpu": [], "api": []}
    lines = [_PREFIX_RE.sub("", line) for line in text.splitlines()]
    header: list[str] | None = None
    units: list[str] | None = None
    section = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            header, units = None, None  # a blank line ends a table
            continue
        cells = next(csv.reader(io.StringIO(stripped)))
        if "Name" in cells and "Calls" in cells:
            header, units = cells, None
            continue
        if header is None or len(cells) != len(header):
            # A status or title line: it names the table that follows.
            lowered = stripped.lower()
            if "api calls" in lowered:
                section = "api"
            elif "profiling result" in lowered or "gpu activities" in lowered:
                section = "gpu"
            header, units = None, None
            continue
        if units is None:
            units = cells
            continue
        row = dict(zip(header, cells))
        unit_of = dict(zip(header, units))

        def to_ms(column: str) -> float:
            return float(row[column]) * UNIT_TO_MS[unit_of[column].strip()]

        kind = row.get("Type", "").strip().lower()
        if kind.startswith("gpu"):
            target = "gpu"
        elif kind.startswith("api"):
            target = "api"
        elif section:
            target = section
        else:
            continue
        out[target].append({
            "name": row["Name"],
            "time_ms": to_ms("Time"),
            "calls": int(row["Calls"]),
            "avg_ms": to_ms("Avg"),
            "min_ms": to_ms("Min"),
            "max_ms": to_ms("Max"),
            "percent": float(row["Time(%)"]),
        })
    return out


def bare_kernel_name(name: str) -> str:
    """'evaluate(unsigned int*, int*)' -> 'evaluate'; '[CUDA memcpy HtoD]' unchanged."""
    return name.split("(")[0].strip()


def split_from_kernel_ms(per_kernel_ms: dict[str, float]) -> dict[str, Any]:
    """Group per-kernel GPU milliseconds; return groups, fractions and the class loop."""
    groups = common.group_kernel_times(per_kernel_ms)
    frac = common.fractions(groups)
    return {
        "group_ms": groups,
        "group_fraction": frac,
        "total_ms": sum(groups.values()),
        "class_loop_ms": sum(groups[name] for name in common.CLASS_LOOP_GROUPS),
        "class_loop_fraction": sum(frac[name] for name in common.CLASS_LOOP_GROUPS),
    }


# --------------------------------------------------------------------------- one config


def nvprof_prefix(nvprof: str, log_pattern: Path) -> list[str]:
    return [nvprof, "--profile-from-start", "off", "--unified-memory-profiling", "off",
            "--print-gpu-summary", "--print-api-summary", "--csv",
            "--normalized-time-unit", "ms", "--log-file", str(log_pattern)]


def point_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The configurations to profile. Either `outputs` (a list of class counts on the base
    spec, the initial design's two-node task) or `points` (a list of dicts with `outputs` plus any
    worker spec overrides, e.g. task="masked_windows" for the E1 configuration)."""
    if "points" in config:
        return [dict(point) for point in config["points"]]
    return [{"outputs": int(k)} for k in config["outputs"]]


def profile_point(point_spec: int | dict[str, Any], config: dict[str, Any],
                  work_dir: Path) -> dict[str, Any]:
    base = dict(config["base"])
    overrides = {"outputs": int(point_spec)} if isinstance(point_spec, int) else dict(point_spec)
    outputs = int(overrides["outputs"])
    tag = overrides.pop("tag", None) or (
        f"profile_k{outputs}" if overrides.get("task", base.get("task", "two_node")) == "two_node"
        else f"profile_{overrides.get('task')}_k{outputs}")
    spec = {**base, **overrides, "outputs": outputs}
    point: dict[str, Any] = {"outputs": outputs, "tag": tag, "notes": [],
                             "task": spec.get("task", "two_node"), "spec": spec}

    pilot = common.spawn_worker({**spec, "tag": f"{tag}_pilot", "mode": "time",
                                 "n_graphs": config["pilot_n"],
                                 "reps": config["pilot_reps"]}, work_dir)
    if "error" in pilot:
        point["status"] = "pilot failed"
        point["error"] = pilot["error"]
        return point
    per_example_s = pilot["fit_s"]["min"] / config["pilot_n"]
    n_graphs = common.choose_n(per_example_s, config["target_rep_s"], config["n_min"],
                               config["n_max"], outputs, config["max_label_bytes"])
    spec["n_graphs"] = n_graphs
    point["n_graphs"] = n_graphs
    print(f"[{tag}] pilot: {1000 * per_example_s:.3f} ms/example, N={n_graphs}", flush=True)

    # (a) un-instrumented wall time: the denominator of "step time".
    plain = common.spawn_worker({**spec, "tag": f"{tag}_plain", "mode": "time"}, work_dir)
    point["plain"] = plain
    if "error" in plain:
        point["status"] = "plain timing failed"
        point["error"] = plain["error"]
        return point
    n_examples = n_graphs * plain["fit_s"]["n"]
    wall_ms = plain["ms_per_example"]
    point["wall_ms_per_example_uninstrumented"] = wall_ms
    point["wall_ms_per_example_uninstrumented_min"] = 1000 * plain["fit_s"]["min"] / n_graphs
    point["wall_ms_per_example_uninstrumented_max"] = 1000 * plain["fit_s"]["max"] / n_graphs

    # (b) nvprof: GPU time per kernel from GPU timestamps, plus the API summary.
    nvprof_ok = False
    nvprof = config["nvprof"]
    if not (Path(nvprof).is_file() or shutil.which(nvprof)):
        point["notes"].append(f"nvprof not found at {nvprof}")
    else:
        pattern = work_dir / f"nvprof_{tag}_%p.csv"
        prof = common.spawn_worker({**spec, "tag": f"{tag}_nvprof", "mode": "nvprof"},
                                   work_dir, prefix=nvprof_prefix(nvprof, pattern))
        point["nvprof_worker"] = {k: v for k, v in prof.items() if k != "log_tail"}
        logs = sorted(work_dir.glob(f"nvprof_{tag}_*.csv"))
        point["nvprof_csv_files"] = [p.name for p in logs]
        parsed: dict[str, list[dict[str, Any]]] = {"gpu": [], "api": []}
        for log in logs:
            one = parse_nvprof_csv(log.read_text(encoding="utf-8", errors="replace"))
            parsed["gpu"].extend(one["gpu"])
            parsed["api"].extend(one["api"])
        if "error" in prof:
            point["notes"].append(f"nvprof worker failed: {prof['error']}")
        elif not parsed["gpu"]:
            point["notes"].append("nvprof ran but its CSV holds no GPU activities")
        else:
            nvprof_ok = True
            kernel_ms: dict[str, float] = {}
            kernel_calls: dict[str, int] = {}
            for row in parsed["gpu"]:
                name = bare_kernel_name(row["name"])
                kernel_ms[name] = kernel_ms.get(name, 0.0) + row["time_ms"]
                kernel_calls[name] = kernel_calls.get(name, 0) + row["calls"]
            n_prof = prof["profiled_examples_total"]
            point["nvprof"] = {
                "profiled_examples": n_prof,
                "wall_ms_per_example_under_nvprof": 1000 * prof["profiled_wall_s_total"] / n_prof,
                "kernel_ms_total": kernel_ms,
                "kernel_calls_total": kernel_calls,
                "kernel_ms_per_example": {k: v / n_prof for k, v in kernel_ms.items()},
                "gpu_launches_per_example": sum(
                    calls for name, calls in kernel_calls.items()
                    if not name.startswith("[")) / n_prof,
                "memcpy_per_example": sum(
                    calls for name, calls in kernel_calls.items()
                    if name.startswith("[")) / n_prof,
                "api_ms_per_example": {row["name"]: row["time_ms"] / n_prof
                                       for row in parsed["api"]},
                "api_calls_per_example": {row["name"]: row["calls"] / n_prof
                                          for row in parsed["api"]},
                **split_from_kernel_ms(kernel_ms),
            }
            point["nvprof"]["gpu_ms_per_example"] = point["nvprof"]["total_ms"] / n_prof

    # (c) CUDA events around every kernel launch: cross-check, and fallback for the split.
    events = common.spawn_worker({**spec, "tag": f"{tag}_events", "mode": "events"}, work_dir)
    if "error" in events:
        point["notes"].append(f"event worker failed: {events['error']}")
    else:
        n_ev = events["profiled_examples_total"]
        ev_ms = {name: entry["ms"] for name, entry in events["event_kernel_ms"].items()}
        point["events"] = {
            "profiled_examples": n_ev,
            "wall_ms_per_example_under_events": 1000 * events["profiled_wall_s_total"] / n_ev,
            "kernel_ms_total": ev_ms,
            "kernel_calls_total": {name: entry["calls"]
                                   for name, entry in events["event_kernel_ms"].items()},
            "kernel_ms_per_example": {k: v / n_ev for k, v in ev_ms.items()},
            "note": "event pairs bracket the launch call, so every interval also holds the "
                    "host launch latency; memcpy is not covered",
            **split_from_kernel_ms(ev_ms),
        }
        point["events"]["gpu_ms_per_example"] = point["events"]["total_ms"] / n_ev

    if nvprof_ok:
        source = "nvprof"
    elif "events" in point:
        source = "cuda_events_fallback"
        point["notes"].append("FALLBACK: the split below comes from pycuda.driver.Event "
                              "timing, not from nvprof")
    else:
        point["status"] = "no profile source worked"
        return point
    chosen = point["nvprof"] if nvprof_ok else point["events"]
    point["split_source"] = source
    gpu_ms = chosen["gpu_ms_per_example"]
    class_loop_ms = chosen["class_loop_ms"] / chosen["profiled_examples"]
    point["gpu_ms_per_example"] = gpu_ms
    point["class_loop_gpu_ms_per_example"] = class_loop_ms
    point["class_loop_fraction_of_gpu_time"] = chosen["class_loop_fraction"]
    point["gpu_busy_fraction_of_wall"] = gpu_ms / wall_ms
    point["host_overhead_fraction_of_wall"] = 1.0 - gpu_ms / wall_ms
    point["class_loop_fraction_of_step_time"] = class_loop_ms / wall_ms
    point["step_time_fraction"] = {
        name: (chosen["group_ms"][name] / chosen["profiled_examples"]) / wall_ms
        for name in GROUP_ORDER}
    point["step_time_fraction"]["host_overhead"] = point["host_overhead_fraction_of_wall"]
    point["status"] = "ok"
    return point


# --------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="JSON config (default: DEFAULT_CONFIG)")
    parser.add_argument("--write-config", default=None,
                        help="write DEFAULT_CONFIG to this path and exit")
    args = parser.parse_args(argv)
    if args.write_config:
        Path(args.write_config).write_text(
            json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    config = common.load_config(args.config, DEFAULT_CONFIG)
    threshold = float(config["kill_threshold"])

    with common.open_run("p0_profile", config) as run:
        work_dir = run.run_dir / "workers"
        points = [profile_point(ps, config, work_dir) for ps in point_specs(config)]

        split_rows: list[dict[str, Any]] = []
        kernel_rows: list[dict[str, Any]] = []
        for point in points:
            k = point["outputs"]
            key = point["tag"].replace("profile_", "")
            if point["status"] != "ok":
                run.add_metric(f"status.{key}", point["status"], "measured",
                               error=str(point.get("error")))
                print(f"[result] outputs={k}: FAILED: {point['status']}: "
                      f"{point.get('error')}", flush=True)
                continue
            for source_name in ("nvprof", "events"):
                src = point.get(source_name)
                if not src:
                    continue
                for group in GROUP_ORDER:
                    split_rows.append({
                        "outputs": k, "task": point["task"], "source": source_name, "group": group,
                        "gpu_ms_per_example": src["group_ms"][group] / src["profiled_examples"],
                        "fraction_of_gpu_time": src["group_fraction"][group],
                        "fraction_of_step_time": (src["group_ms"][group]
                                                  / src["profiled_examples"])
                        / point["wall_ms_per_example_uninstrumented"],
                    })
                for name, ms in sorted(src["kernel_ms_total"].items()):
                    kernel_rows.append({
                        "outputs": k, "task": point["task"], "source": source_name, "kernel": name,
                        "calls_total": src["kernel_calls_total"].get(name, ""),
                        "gpu_ms_total": ms,
                        "gpu_ms_per_example": ms / src["profiled_examples"],
                        "fraction_of_gpu_time": ms / src["total_ms"],
                    })

            src_label = point["split_source"]
            run.add_metric(f"split_source.{key}", src_label, "measured")
            run.add_metric(f"n_graphs.{key}", point["n_graphs"], "measured")
            run.add_metric(f"wall_ms_per_example_uninstrumented.{key}",
                           point["wall_ms_per_example_uninstrumented"], "measured", unit="ms",
                           stat="median of timed fits, no profiler",
                           min=point["wall_ms_per_example_uninstrumented_min"],
                           max=point["wall_ms_per_example_uninstrumented_max"])
            chosen = point["nvprof"] if src_label == "nvprof" else point["events"]
            for group in GROUP_ORDER:
                run.add_metric(f"gpu_time_fraction.{group}.{key}",
                               chosen["group_fraction"][group], "measured",
                               source=src_label, note="share of summed GPU activity time")
            run.add_metric(f"gpu_ms_per_example.{key}", point["gpu_ms_per_example"], "measured",
                           unit="ms", source=src_label)
            run.add_metric(f"class_loop_fraction_of_gpu_time.{key}",
                           point["class_loop_fraction_of_gpu_time"], "measured",
                           source=src_label,
                           note="evaluate + select_clause_updates + update + update_message")
            run.add_metric(f"gpu_busy_fraction_of_wall.{key}",
                           point["gpu_busy_fraction_of_wall"], "derived",
                           note="GPU ms per example (profiled run) divided by wall ms per "
                                "example (un-instrumented run, same seed, N and reps)")
            run.add_metric(f"host_overhead_fraction_of_wall.{key}",
                           point["host_overhead_fraction_of_wall"], "derived",
                           note="1 - gpu_busy_fraction_of_wall: Python, kernel launch, "
                                "synchronize latency")
            run.add_metric(f"class_loop_fraction_of_step_time.{key}",
                           point["class_loop_fraction_of_step_time"], "derived",
                           note="class-loop GPU ms per example divided by un-instrumented "
                                "wall ms per example")
            if "nvprof" in point:
                nv = point["nvprof"]
                run.add_metric(f"wall_ms_per_example_under_nvprof.{key}",
                               nv["wall_ms_per_example_under_nvprof"], "measured", unit="ms")
                run.add_metric(f"gpu_kernel_launches_per_example.{key}",
                               nv["gpu_launches_per_example"], "measured")
                run.add_metric(f"memcpy_per_example.{key}", nv["memcpy_per_example"], "measured")
                for api_name in ("cuLaunchKernel", "cuCtxSynchronize", "cuMemcpyHtoD"):
                    if api_name in nv["api_ms_per_example"]:
                        run.add_metric(f"api_ms_per_example.{api_name}.{key}",
                                       nv["api_ms_per_example"][api_name], "measured",
                                       unit="ms", note="host time inside the API call, "
                                                       "measured UNDER nvprof")
            if "events" in point:
                ev = point["events"]
                run.add_metric(f"class_loop_fraction_of_gpu_time_events.{key}",
                               ev["class_loop_fraction"], "measured",
                               note="cross-check with CUDA events, intervals include the "
                                    "host launch latency")
                run.add_metric(f"wall_ms_per_example_under_events.{key}",
                               ev["wall_ms_per_example_under_events"], "measured", unit="ms")
            gpu_verdict = point["class_loop_fraction_of_gpu_time"] < threshold
            step_verdict = point["class_loop_fraction_of_step_time"] < threshold
            run.add_metric(f"kill_criterion_triggers_on_gpu_time.{key}", bool(gpu_verdict),
                           "derived", threshold=threshold)
            run.add_metric(f"kill_criterion_triggers_on_step_time.{key}", bool(step_verdict),
                           "derived", threshold=threshold)
            print(f"[result] {key} ({src_label}): class loop = "
                  f"{100 * point['class_loop_fraction_of_gpu_time']:.1f}% of GPU time, "
                  f"{100 * point['class_loop_fraction_of_step_time']:.1f}% of step time; "
                  f"GPU busy {100 * point['gpu_busy_fraction_of_wall']:.1f}% of wall "
                  f"({point['gpu_ms_per_example']:.3f} of "
                  f"{point['wall_ms_per_example_uninstrumented']:.3f} ms per example)",
                  flush=True)
            for note in point["notes"]:
                print(f"[note] outputs={k}: {note}", flush=True)

        for name, rows in (("profile_split.csv", split_rows),
                           ("profile_kernels.csv", kernel_rows)):
            if rows:
                with open(run.run_dir / name, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        slim = [{key: value for key, value in point.items() if key != "plain"}
                for point in points]
        run.write_json("profile_summary.json", {"kill_threshold": threshold, "points": slim})
        failed = [point for point in points if point["status"] != "ok"]
        print(f"[done] {len(points) - len(failed)} configurations ok, {len(failed)} failed",
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
