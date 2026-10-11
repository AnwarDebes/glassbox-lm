"""T0.1 / E0: training throughput of the STOCK GraphTsetlinMachine versus the number of
outputs, on the two-node graph task (E0 of the initial design: clauses 2000, depth 1, hypervector 256).

Series (see DEFAULT_CONFIG):
  - MultiClassGraphTsetlinMachine, depth 1, outputs {10, 100, 1000, 5000, 16000}
  - MultiClassGraphTsetlinMachine, depth 2 (message_size 256), same outputs
  - MultiOutputGraphTsetlinMachine with one-hot Y, depth 1, outputs {100, 16000}

Every point runs in two fresh subprocesses (experiments/p0_common.py): a short pilot that
estimates the time per example, then the timed worker with N chosen so that one timed
repetition takes about `target_rep_s` seconds. The timed worker does one warm-up fit and
`reps` timed fit(epochs=1, incremental=True) calls with a synchronize before and after.

Outputs in the run dir: scaling.csv, scaling.png, scaling_summary.json, workers/*.json,
metrics.json (every metric labelled).

Run (timing, so the GPU lock is exclusive):
    python experiments/p0_scaling_harness.py --write-config /path/cfg.json
    scripts/gpu_run.sh exclusive -- scripts/run_logged.py --name p0_scaling \
        --config /path/cfg.json -- python experiments/p0_scaling_harness.py --config /path/cfg.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as common  # noqa: E402

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "p0_scaling_harness",
    "brief_tasks": ["T0.1", "E0"],
    "package": "stock GraphTsetlinMachine (vendored master, unmodified)",
    "task": "two-node graphs, symbols A/B, label = hash(symbol pair) mod K plus label noise",
    "base": {
        "clauses": 2000, "T": 1000, "s": 10.0, "q": 1.0, "max_included_literals": None,
        "hypervector_size": 256, "hypervector_bits": 2,
        "message_size": 256, "message_bits": 2,
        "noise": 0.25, "seed": 42, "reps": 5,
        "grid": list(common.SAFE_GRID), "block": list(common.SAFE_BLOCK),
    },
    "series": [
        {"id": "multiclass_depth1", "model": "multiclass", "depth": 1,
         "outputs": [10, 100, 1000, 5000, 16000]},
        {"id": "multiclass_depth2", "model": "multiclass", "depth": 2,
         "outputs": [10, 100, 1000, 5000, 16000]},
        {"id": "multioutput_depth1", "model": "multioutput", "depth": 1,
         "outputs": [100, 16000]},
    ],
    "pilot_n": 200,
    "pilot_reps": 2,
    "target_rep_s": 10.0,
    "n_min": 200,
    "n_max": 60000,
    "max_label_bytes": 2 * 2**30,
}

CSV_FIELDS = [
    "series", "model", "depth", "outputs", "n_graphs", "reps",
    "examples_per_s_median", "examples_per_s_at_min_time", "examples_per_s_at_max_time",
    "fit_s_median", "fit_s_min", "fit_s_max", "ms_per_example_median",
    "examples_per_s_inner_median", "host_label_encoding_s_median",
    "label_matrix_bytes", "label_matrix_mib", "gpu_alloc_mib_tracked",
    "gpu_alloc_largest_buffer_mib", "gpu_peak_used_mib", "gpu_context_only_mib",
    "gpu_model_and_data_mib", "init_gpu_kernels_s", "warmup_fit_s", "graphs_per_s_build",
    "n_distinct_labels", "status",
]

SERIES_STYLE = {
    "multiclass_depth1": {"color": "#2a78d6", "marker": "o",
                          "label": "MultiClass, depth 1"},
    "multiclass_depth2": {"color": "#eb6834", "marker": "s",
                          "label": "MultiClass, depth 2 (message 256)"},
    "multioutput_depth1": {"color": "#1baf7a", "marker": "D",
                           "label": "MultiOutput one-hot, depth 1"},
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


def measure_point(series: dict[str, Any], outputs: int, config: dict[str, Any],
                  work_dir: Path) -> dict[str, Any]:
    base = dict(config["base"])
    tag = f"{series['id']}_k{outputs}"
    spec = {**base, "model": series["model"], "depth": series["depth"], "outputs": outputs,
            "mode": "time"}
    pilot = common.spawn_worker(
        {**spec, "tag": f"{tag}_pilot", "n_graphs": config["pilot_n"],
         "reps": config["pilot_reps"]}, work_dir)
    if "error" in pilot:
        return {"series": series["id"], "outputs": outputs, "status": "pilot failed",
                "error": pilot["error"], "pilot": pilot}
    per_example_s = pilot["fit_s"]["min"] / config["pilot_n"]
    n_graphs = common.choose_n(per_example_s, config["target_rep_s"], config["n_min"],
                               config["n_max"], outputs, config["max_label_bytes"])
    print(f"[{tag}] pilot: {1000 * per_example_s:.3f} ms/example, timed run uses N={n_graphs}",
          flush=True)
    timed = common.spawn_worker({**spec, "tag": tag, "n_graphs": n_graphs}, work_dir)
    timed["series"] = series["id"]
    timed["outputs"] = outputs
    timed["pilot_ms_per_example"] = 1000 * per_example_s
    timed["status"] = "failed" if "error" in timed else "ok"
    return timed


def csv_row(point: dict[str, Any], series: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update({"series": series["id"], "model": series["model"], "depth": series["depth"],
                "outputs": point["outputs"], "status": point["status"]})
    if point["status"] != "ok":
        return row
    row.update({
        "n_graphs": point["spec"]["n_graphs"],
        "reps": point["fit_s"]["n"],
        "examples_per_s_median": point["examples_per_s"],
        "examples_per_s_at_min_time": point["examples_per_s_at_min_time"],
        "examples_per_s_at_max_time": point["examples_per_s_at_max_time"],
        "fit_s_median": point["fit_s"]["median"],
        "fit_s_min": point["fit_s"]["min"],
        "fit_s_max": point["fit_s"]["max"],
        "ms_per_example_median": point["ms_per_example"],
        "examples_per_s_inner_median": point["examples_per_s_inner"],
        "host_label_encoding_s_median": point["host_label_encoding_s"],
        "label_matrix_bytes": point["label_matrix_bytes"],
        "label_matrix_mib": point["label_matrix_bytes"] / 2**20,
        "gpu_alloc_mib_tracked": point["gpu_alloc_mib_after_training"],
        "gpu_alloc_largest_buffer_mib": point["gpu_alloc_after_training"]["largest_bytes"]
        / 2**20,
        "gpu_peak_used_mib": point["gpu_used_mib_after_training"],
        "gpu_context_only_mib": point["gpu_used_mib_context_only"],
        "gpu_model_and_data_mib": point["gpu_model_and_data_mib"],
        "init_gpu_kernels_s": point["init_gpu_kernels_s"],
        "warmup_fit_s": point["warmup_fit_s"],
        "graphs_per_s_build": point["graphs_per_s_build"],
        "n_distinct_labels": point["n_distinct_labels"],
    })
    return row


def characterise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Slope of the curve of one series. All values here are DERIVED from measured rows."""
    ok = [row for row in rows if row["status"] == "ok"]
    ok.sort(key=lambda row: row["outputs"])
    out: dict[str, Any] = {"n_points": len(ok)}
    if len(ok) < 2:
        return out
    ks = [float(row["outputs"]) for row in ok]
    eps = [float(row["examples_per_s_median"]) for row in ok]
    ms = [float(row["ms_per_example_median"]) for row in ok]
    out["loglog_slope_all"] = common.loglog_slope(ks, eps)
    out["loglog_slope_segments"] = {
        f"{int(ks[i])}_to_{int(ks[i + 1])}": common.loglog_slope(ks[i:i + 2], eps[i:i + 2])
        for i in range(len(ks) - 1)
    }
    big = [(k, e) for k, e in zip(ks, eps) if k >= 1000]
    if len(big) >= 2:
        out["loglog_slope_outputs_ge_1000"] = common.loglog_slope(*zip(*big))
    slope, intercept = np.polyfit(np.asarray(ks), np.asarray(ms), 1)
    out["linear_fit_ms_per_example"] = {
        "fixed_ms": float(intercept), "per_output_us": float(1000.0 * slope),
        "note": "least squares of ms_per_example = fixed_ms + per_output_us/1000 * outputs",
    }
    out["slowdown_largest_vs_smallest"] = eps[0] / eps[-1]
    out["outputs_range"] = [int(ks[0]), int(ks[-1])]
    return out


def plot(rows: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for series in config["series"]:
        pts = sorted((row for row in rows
                      if row["series"] == series["id"] and row["status"] == "ok"),
                     key=lambda row: row["outputs"])
        if not pts:
            continue
        style = SERIES_STYLE.get(series["id"], {"color": INK_SECONDARY, "marker": "o",
                                                "label": series["id"]})
        ks = np.array([row["outputs"] for row in pts], dtype=float)
        med = np.array([row["examples_per_s_median"] for row in pts], dtype=float)
        low = np.array([row["examples_per_s_at_max_time"] for row in pts], dtype=float)
        high = np.array([row["examples_per_s_at_min_time"] for row in pts], dtype=float)
        ax.errorbar(ks, med, yerr=[med - low, high - med], color=style["color"],
                    marker=style["marker"], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linewidth=2, capsize=3, label=style["label"],
                    linestyle="-" if len(pts) > 2 else "--")
        last = pts[-1]
        ax.annotate(f"{last['examples_per_s_median']:.0f}/s",
                    (last["outputs"], last["examples_per_s_median"]),
                    textcoords="offset points",
                    xytext=(8, {"multiclass_depth1": 8, "multiclass_depth2": -12}.get(
                        series["id"], -2)),
                    fontsize=9, color=INK)
        if series["id"].startswith("multiclass"):
            first = pts[0]
            ax.annotate(f"{first['examples_per_s_median']:.0f}/s",
                        (first["outputs"], first["examples_per_s_median"]),
                        textcoords="offset points", xytext=(-6, 8), ha="right",
                        fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of outputs (CLASSES, compile-time constant)", color=INK_SECONDARY)
    ax.set_ylabel("training examples per second (median of timed fits)", color=INK_SECONDARY)
    base = config["base"]
    ax.set_title("Stock GraphTsetlinMachine: training throughput vs number of outputs",
                 fontsize=12, color=INK, loc="left", pad=22)
    ax.text(0.0, 1.025,
            f"two-node graphs, clauses {base['clauses']}, hypervector "
            f"{base['hypervector_size']}, one Tesla V100, bars show min and max of "
            f"{base['reps']} timed fits",
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY)
    ax.grid(True, which="major", color="#d9d8d3", linewidth=0.8)
    ax.grid(True, which="minor", color="#ecebe7", linewidth=0.5)
    ax.tick_params(colors=INK_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b5b4ae")
    ax.margins(x=0.12)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


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

    with common.open_run("p0_scaling", config) as run:
        work_dir = run.run_dir / "workers"
        rows: list[dict[str, Any]] = []
        points: list[dict[str, Any]] = []
        for series in config["series"]:
            for outputs in series["outputs"]:
                point = measure_point(series, int(outputs), config, work_dir)
                points.append(point)
                rows.append(csv_row(point, series))
                if point["status"] == "ok":
                    print(f"[result] {series['id']} outputs={outputs}: "
                          f"{point['examples_per_s']:.1f} examples/s "
                          f"(N={point['spec']['n_graphs']})", flush=True)
                else:
                    print(f"[result] {series['id']} outputs={outputs}: FAILED: "
                          f"{point.get('error')}", flush=True)

        with open(run.run_dir / "scaling.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary = {series["id"]: characterise([r for r in rows if r["series"] == series["id"]])
                   for series in config["series"]}
        run.write_json("scaling_summary.json", {"slopes": summary, "rows": rows})
        try:
            plot(rows, config, run.run_dir / "scaling.png")
        except Exception as exc:  # noqa: BLE001 - the CSV is the primary artifact
            print(f"[plot] FAILED: {type(exc).__name__}: {exc}", flush=True)

        for row in rows:
            key = f"{row['series']}.k{row['outputs']}"
            if row["status"] != "ok":
                run.add_metric(f"status.{key}", row["status"], "measured")
                continue
            run.add_metric(f"examples_per_s.{key}", row["examples_per_s_median"], "measured",
                           unit="examples/s", stat="median of timed fits",
                           min=row["examples_per_s_at_max_time"],
                           max=row["examples_per_s_at_min_time"],
                           n_graphs=row["n_graphs"], reps=row["reps"])
            run.add_metric(f"examples_per_s_inner.{key}", row["examples_per_s_inner_median"],
                           "measured", unit="examples/s",
                           note="stock _fit only, without the host label encoding of fit()")
            run.add_metric(f"label_matrix_mib.{key}", row["label_matrix_mib"], "measured",
                           unit="MiB", note="nbytes of the stock encoded_Y (N x outputs int32)")
            run.add_metric(f"gpu_alloc_mib_tracked.{key}", row["gpu_alloc_mib_tracked"],
                           "measured", unit="MiB",
                           note="sum of the sizes this process passed to pycuda mem_alloc "
                                "(model, graphs, dense label matrix); nothing is freed "
                                "during incremental fits, so this is the peak as well")
            run.add_metric(f"gpu_peak_used_mib.{key}", row["gpu_peak_used_mib"], "measured",
                           unit="MiB",
                           note="DEVICE-WIDE, pycuda mem_get_info after training: includes "
                                "the CUDA context and any other process on the GPU")
            run.add_metric(f"gpu_model_and_data_mib.{key}", row["gpu_model_and_data_mib"],
                           "measured", unit="MiB",
                           note="device-wide used minus the context-only reading; wrong "
                                "whenever another process came or went in between")
        for series_id, info in summary.items():
            for name in ("loglog_slope_all", "loglog_slope_outputs_ge_1000",
                         "slowdown_largest_vs_smallest"):
                if name in info:
                    run.add_metric(f"{name}.{series_id}", info[name], "derived")
            if "linear_fit_ms_per_example" in info:
                fit = info["linear_fit_ms_per_example"]
                run.add_metric(f"fixed_ms_per_example.{series_id}", fit["fixed_ms"], "derived")
                run.add_metric(f"per_output_us_per_example.{series_id}", fit["per_output_us"],
                               "derived")
        failed = [row for row in rows if row["status"] != "ok"]
        print(f"[done] {len(rows) - len(failed)} points ok, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
