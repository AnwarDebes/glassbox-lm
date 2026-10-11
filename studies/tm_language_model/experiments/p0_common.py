"""Shared helpers for the Phase 0 (milestone M1) measurements on the STOCK CAIR
GraphTsetlinMachine package (tasks T0.1/E0, T0.2, T0.3 of the initial design).

What is here:
  - make_two_node_task: labels for the initial design's "trivial two-node graph task", extended
    to K classes. Pure numpy, no GPU.
  - build_two_node_graphs: the stock `Graphs` object for that task (2 nodes, one edge
    each way, symbols A and B).
  - timed_fit: one warm-up fit (kernel compilation and the first upload are excluded from
    the timing), then `fit(epochs=1, incremental=True)` repeated `reps` times with a
    pycuda synchronize before and after each repetition. Reports median, min and max.
  - run_worker / spawn_worker: ONE model per process. CLASSES is a compile-time constant
    of the stock kernels and curand is seeded from np.random when GraphTsetlinMachine.tm
    is imported, so every measured point runs in a fresh subprocess: seed first, import
    second (docs/02, blocker 5).
  - open_run: use the run dir of scripts/run_logged.py when there is one, else open a
    RunLog, so that every number lands in results/<timestamp>_<name>/.

Facts from docs/02 that this file respects:
  - grid=(16*13,1,1), block=(128,1,1) is always passed (the stock default grid is the
    H100 setting).
  - MultiClassGraphTsetlinMachine.fit sets number_of_outputs = max(Y) + 1, so the task
    always contains the label K-1 and the worker asserts number_of_outputs == K.
  - fit() builds a dense (N, CLASSES) int32 label matrix on host and device, so N is
    kept small at large CLASSES and the size of that matrix is recorded.

Worker usage (normally through spawn_worker):
    python experiments/p0_common.py --worker spec.json --out result.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SAFE_GRID = (16 * 13, 1, 1)
SAFE_BLOCK = (128, 1, 1)
SYMBOLS = ("A", "B")
PAIR_HASH_MULTIPLIER = 2654435761  # Knuth's multiplicative hash constant

# Kernels of the stock package, grouped the way task T0.2 of the initial design asks for.
KERNEL_GROUPS = {
    "evaluate": ("evaluate",),
    "select_clause_updates": ("select_clause_updates",),
    "update": ("update", "update_message"),
    "calculate_messages": ("calculate_messages", "calculate_messages_conditional"),
}
CLASS_LOOP_GROUPS = ("evaluate", "select_clause_updates", "update")
TIMED_KERNEL_ATTRS = (
    "calculate_messages",
    "calculate_messages_conditional",
    "prepare_messages",
    "exchange_messages",
    "encode_messages",
    "evaluate",
    "select_clause_node",
    "select_clause_updates",
    "update",
    "update_message",
)

DEFAULT_SPEC: dict[str, Any] = {
    "tag": "point",
    "model": "multiclass",  # multiclass | multioutput
    "outputs": 10,
    "clauses": 2000,
    "T": 1000,
    "s": 10.0,
    "q": 1.0,
    "max_included_literals": None,
    "depth": 1,
    "hypervector_size": 256,
    "hypervector_bits": 2,
    "message_size": 256,
    "message_bits": 2,
    "n_graphs": 200,
    "noise": 0.25,
    "seed": 42,
    "reps": 5,
    "grid": list(SAFE_GRID),
    "block": list(SAFE_BLOCK),
    "mode": "time",  # time | events | nvprof
    "disable_pycuda_cache": False,
    "score_after": False,
    # two_node: the initial design's two-node task (above). masked_windows: the E1 masked-position
    # windows of experiments/p0_masked_depth2.py (real WikiText-103 tokens, one target per
    # graph, outputs == vocab); task_params are that module's params (vocab, tokens, window,
    # span, mask_rate, hypervector_bits); n_graphs targets of its epoch-0 mask are used.
    "task": "two_node",
    "task_params": {},
}


# --------------------------------------------------------------------------- the task


def make_two_node_task(
    n_graphs: int, n_classes: int, seed: int, noise: float = 0.25
) -> dict[str, np.ndarray]:
    """Symbols and labels of the two-node task, extended to `n_classes` classes.

    Each graph has two nodes, each carrying symbol A (0) or B (1). The clean label is a
    deterministic function of the symbol pair: a multiplicative hash of the pair index
    (2*x1 + x2 + 1) modulo n_classes. With probability `noise` the label is replaced.
    The replacement labels walk through a fixed permutation of all classes that starts
    with class n_classes-1, so that (a) the largest class id ALWAYS occurs, which keeps
    the stock `number_of_outputs = max(Y) + 1` equal to n_classes, and (b) every class
    occurs as soon as there are at least n_classes noisy examples.

    Returns a dict with x1, x2 (int64 in {0,1}), y (uint32), clean (bool mask).
    """
    if n_graphs < 1 or n_classes < 2:
        raise ValueError("need n_graphs >= 1 and n_classes >= 2")
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, size=n_graphs)
    x2 = rng.integers(0, 2, size=n_graphs)
    pair = 2 * x1 + x2
    base = ((pair + 1) * PAIR_HASH_MULTIPLIER) % (2**32) % n_classes
    noisy = rng.random(n_graphs) < noise
    if not noisy.any():
        noisy[int(rng.integers(n_graphs))] = True
    perm = rng.permutation(n_classes)
    perm = np.concatenate(([n_classes - 1], perm[perm != n_classes - 1]))
    y = base.copy()
    idx = np.flatnonzero(noisy)
    y[idx] = perm[np.arange(idx.size) % n_classes]
    return {
        "x1": x1.astype(np.int64),
        "x2": x2.astype(np.int64),
        "y": y.astype(np.uint32),
        "clean": ~noisy,
    }


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, K) uint32 0/1 matrix, the Y format of MultiOutputGraphTsetlinMachine.fit."""
    out = np.zeros((y.shape[0], n_classes), dtype=np.uint32)
    out[np.arange(y.shape[0]), y.astype(np.int64)] = 1
    return out


def build_two_node_graphs(graphs_cls: Any, x1: np.ndarray, x2: np.ndarray,
                          hypervector_size: int, hypervector_bits: int) -> Any:
    """The stock Graphs object: 2 nodes per graph, one 'Plain' edge each way."""
    n_graphs = int(x1.shape[0])
    graphs = graphs_cls(
        n_graphs,
        symbols=list(SYMBOLS),
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits,
    )
    for graph_id in range(n_graphs):
        graphs.set_number_of_graph_nodes(graph_id, 2)
    graphs.prepare_node_configuration()
    for graph_id in range(n_graphs):
        graphs.add_graph_node(graph_id, "Node 1", 1)
        graphs.add_graph_node(graph_id, "Node 2", 1)
    graphs.prepare_edge_configuration()
    for graph_id in range(n_graphs):
        graphs.add_graph_node_edge(graph_id, "Node 1", "Node 2", "Plain")
        graphs.add_graph_node_edge(graph_id, "Node 2", "Node 1", "Plain")
        graphs.add_graph_node_property(graph_id, "Node 1", SYMBOLS[int(x1[graph_id])])
        graphs.add_graph_node_property(graph_id, "Node 2", SYMBOLS[int(x2[graph_id])])
    graphs.encode()
    return graphs


# --------------------------------------------------------------------------- statistics


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """median, min, max, mean and the raw list of a non-empty sequence of floats."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("no values")
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "n": len(vals),
        "values": vals,
    }


def choose_n(per_example_s: float, target_s: float, n_min: int, n_max: int,
             outputs: int, max_label_bytes: int) -> int:
    """Number of graphs so that one timed repetition takes about target_s, within
    [n_min, n_max] and with the dense label matrix (N * outputs * 4 bytes) capped."""
    if per_example_s <= 0:
        raise ValueError("per_example_s must be positive")
    n = int(round(target_s / per_example_s))
    n = min(n, n_max, max(1, max_label_bytes // (4 * outputs)))
    return max(n, n_min)


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of log10(y) against log10(x)."""
    lx = np.log10(np.asarray(xs, dtype=np.float64))
    ly = np.log10(np.asarray(ys, dtype=np.float64))
    if lx.size < 2:
        raise ValueError("need at least two points")
    return float(np.polyfit(lx, ly, 1)[0])


def group_kernel_times(per_kernel: dict[str, float]) -> dict[str, float]:
    """Sum per-kernel GPU times into the groups of KERNEL_GROUPS plus 'other'.

    Keys of `per_kernel` are kernel names as nvprof or the event timer report them. A
    name is matched exactly after stripping a parameter list, so `update` does not
    swallow `update_message` and `evaluate` does not swallow anything else.
    """
    groups = {name: 0.0 for name in KERNEL_GROUPS}
    groups["other"] = 0.0
    lookup = {kernel: group for group, kernels in KERNEL_GROUPS.items() for kernel in kernels}
    for name, value in per_kernel.items():
        bare = name.split("(")[0].strip()
        groups[lookup.get(bare, "other")] += float(value)
    return groups


def fractions(groups: dict[str, float]) -> dict[str, float]:
    total = sum(groups.values())
    if total <= 0:
        raise ValueError("no GPU time recorded")
    return {name: value / total for name, value in groups.items()}


# --------------------------------------------------------------------------- timing


def gpu_used_mib(cuda: Any) -> float:
    """Device-wide used memory from pycuda mem_get_info. It includes the CUDA context of
    this process AND whatever other processes hold on the same GPU (an idle CUDA context
    of another process is about 300 MiB), so it is not a per-model number. The per-model
    number is AllocationTracker.total_bytes."""
    free, total = cuda.mem_get_info()
    return (total - free) / 2**20


class AllocationTracker:
    """Sums the sizes that this process passes to pycuda.driver.mem_alloc.

    The stock package allocates every GPU buffer through `cuda.mem_alloc` looked up on
    the pycuda.driver module at call time, so replacing that module attribute sees all of
    them; the package itself is not touched. Frees are not tracked. With
    fit(incremental=True) on unchanged graphs and labels the stock code allocates once
    and frees nothing, so after training the sum equals the live model and data buffers.
    The driver rounds allocations up, so the device-side footprint is a little larger.
    Not covered: the CUDA context itself and the curand generator state that
    GraphTsetlinMachine.tm creates at import time.
    """

    def __init__(self, cuda: Any) -> None:
        self._cuda = cuda
        self._original = cuda.mem_alloc
        self.total_bytes = 0
        self.largest_bytes = 0
        self.calls = 0

    def install(self) -> "AllocationTracker":
        def mem_alloc(nbytes: Any, *args: Any, **kwargs: Any) -> Any:
            buffer = self._original(nbytes, *args, **kwargs)
            self.total_bytes += int(nbytes)
            self.largest_bytes = max(self.largest_bytes, int(nbytes))
            self.calls += 1
            return buffer

        self._cuda.mem_alloc = mem_alloc
        return self

    def uninstall(self) -> None:
        self._cuda.mem_alloc = self._original

    def snapshot(self) -> dict[str, int]:
        return {"total_bytes": self.total_bytes, "largest_bytes": self.largest_bytes,
                "calls": self.calls}


class _TimedKernel:
    """Proxy around a pycuda Function: brackets every prepared_call with CUDA events."""

    def __init__(self, fn: Any, name: str, sink: list, cuda: Any) -> None:
        self._fn = fn
        self._name = name
        self._sink = sink
        self._cuda = cuda

    def prepared_call(self, grid: Any, block: Any, *args: Any) -> Any:
        start = self._cuda.Event()
        end = self._cuda.Event()
        start.record()
        result = self._fn.prepared_call(grid, block, *args)
        end.record()
        self._sink.append((self._name, start, end))
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._fn, item)


def install_event_timers(tm: Any, cuda: Any) -> list:
    """Replace the kernel attributes of an INITIALIZED model instance by event-timing
    proxies. The package itself is not touched. Returns the list the events go to."""
    sink: list = []
    for attr in TIMED_KERNEL_ATTRS:
        setattr(tm, attr, _TimedKernel(getattr(tm, attr), attr, sink, cuda))
    return sink


def drain_events(sink: list) -> dict[str, dict[str, float]]:
    """Per kernel: summed GPU milliseconds between the two events, and the call count."""
    out: dict[str, dict[str, float]] = {}
    for name, start, end in sink:
        entry = out.setdefault(name, {"ms": 0.0, "calls": 0})
        entry["ms"] += start.time_till(end)
        entry["calls"] += 1
    sink.clear()
    return out


def timed_fit(tm: Any, graphs: Any, y: np.ndarray, cuda: Any, reps: int = 5,
              progress: str = "", fit_fn: Any = None) -> dict[str, Any]:
    """Warm-up fit, then `reps` timed calls of fit(epochs=1, incremental=True) (or of
    `fit_fn()`, which must be one incremental pass over the same graphs).

    The warm-up call pays for kernel compilation, GPU allocation and the upload of X and
    of the dense label matrix, so none of that is inside the timed repetitions. Every
    timed repetition is bracketed by cuda.Context.synchronize(). `fit_s` is the whole
    stock fit() call (it rebuilds the dense label matrix on the host every time);
    `inner_fit_s` is the stock `_fit` alone (the per-graph kernel loop), measured by a
    wrapper on the instance.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    inner: list[float] = []
    kernels_s: list[float] = []
    original_fit = tm._fit
    original_init_kernels = tm._init_gpu_kernels

    def _fit_timed(*args: Any, **kwargs: Any) -> Any:
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        result = original_fit(*args, **kwargs)
        cuda.Context.synchronize()
        inner.append(time.perf_counter() - t0)
        return result

    def _init_kernels_timed(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_init_kernels(*args, **kwargs)
        kernels_s.append(time.perf_counter() - t0)
        return result

    if fit_fn is None:
        def fit_fn() -> None:
            tm.fit(graphs, y, epochs=1, incremental=True)
    tm._fit = _fit_timed
    tm._init_gpu_kernels = _init_kernels_timed
    try:
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        fit_fn()
        cuda.Context.synchronize()
        warmup_s = time.perf_counter() - t0
        warmup_inner_s = inner.pop()
        print(f"{progress}warm-up fit {warmup_s:.3f} s "
              f"(kernel build {sum(kernels_s):.3f} s)", flush=True)
        fit_s: list[float] = []
        for rep in range(reps):
            cuda.Context.synchronize()
            t0 = time.perf_counter()
            fit_fn()
            cuda.Context.synchronize()
            fit_s.append(time.perf_counter() - t0)
            print(f"{progress}rep {rep + 1}/{reps}: fit {fit_s[-1]:.3f} s, "
                  f"inner _fit {inner[-1]:.3f} s", flush=True)
    finally:
        tm._fit = original_fit
        tm._init_gpu_kernels = original_init_kernels
    return {
        "warmup_fit_s": warmup_s,
        "warmup_inner_fit_s": warmup_inner_s,
        "init_gpu_kernels_s": sum(kernels_s),
        "fit_s": summarize(fit_s),
        "inner_fit_s": summarize(inner),
    }


# --------------------------------------------------------------------------- the worker


def run_worker(spec_in: dict[str, Any]) -> dict[str, Any]:
    """Build the task, train one stock model, return everything that was measured."""
    spec = {**DEFAULT_SPEC, **spec_in}
    unknown = set(spec_in) - set(DEFAULT_SPEC)
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    if spec["disable_pycuda_cache"]:
        os.environ["PYCUDA_DISABLE_CACHE"] = "1"
    tag = spec["tag"]
    progress = f"[{tag}] "
    n_graphs, outputs = int(spec["n_graphs"]), int(spec["outputs"])
    grid, block = tuple(spec["grid"]), tuple(spec["block"])

    from tmlm.common.seeds import seed_everything

    seed_everything(int(spec["seed"]))  # BEFORE importing GraphTsetlinMachine.tm
    t0 = time.perf_counter()
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine import tm as gtm
    import pycuda.driver as cuda

    import_s = time.perf_counter() - t0
    assert hasattr(gtm, "MultiClassGraphTsetlinMachine")
    assert hasattr(gtm, "MultiOutputGraphTsetlinMachine")
    mem_context_mib = gpu_used_mib(cuda)
    tracker = AllocationTracker(cuda).install()

    fit_fn = None
    if spec["task"] == "two_node":
        task = make_two_node_task(n_graphs, outputs, int(spec["seed"]), float(spec["noise"]))
        t0 = time.perf_counter()
        graphs = build_two_node_graphs(Graphs, task["x1"], task["x2"],
                                       int(spec["hypervector_size"]), int(spec["hypervector_bits"]))
        graph_build_s = time.perf_counter() - t0
        y = task["y"] if spec["model"] == "multiclass" else one_hot(task["y"], outputs)
    elif spec["task"] == "masked_windows":
        import p0_masked_depth2 as e1

        if spec["model"] != "multiclass":
            raise ValueError("masked_windows needs model == multiclass")
        tp = {**e1.DEFAULT_PARAMS, **spec["task_params"],
              "hypervector_size": int(spec["hypervector_size"]),
              "hypervector_bits": int(spec["hypervector_bits"])}
        if int(tp["vocab"]) != outputs:
            raise ValueError(f"outputs ({outputs}) must equal task_params.vocab ({tp['vocab']})")
        lm = e1.load_task(tp)
        template = e1.WindowTemplate(int(tp["window"]), int(tp["span"]))
        codebook = e1.make_codebook(Graphs, outputs, int(tp["hypervector_size"]),
                                    int(tp["hypervector_bits"]), template)
        rows = e1.symbol_rows(codebook)
        hidden, positions = e1.epoch_targets(int(lm["train_tok"].shape[0]), float(tp["mask_rate"]),
                                             int(spec["seed"]), 0, n_graphs)
        if positions.shape[0] != n_graphs:
            raise ValueError(f"only {positions.shape[0]} masked positions in the slice")
        t0 = time.perf_counter()
        graphs = e1.build_window_graphs(Graphs, codebook, rows, template, lm["train_tok"], hidden,
                                        positions, codebook.symbol_id[e1.MASK_SYMBOL])
        graph_build_s = time.perf_counter() - t0
        y = lm["train_tok"][positions].astype(np.uint32)
        task = {"y": y, "clean": np.ones(n_graphs, dtype=bool)}
    else:
        raise ValueError(f"unknown task {spec['task']!r}")
    print(f"{progress}{n_graphs} graphs built in {graph_build_s:.2f} s", flush=True)

    cls = {"multiclass": gtm.MultiClassGraphTsetlinMachine,
           "multioutput": gtm.MultiOutputGraphTsetlinMachine}[spec["model"]]
    model = cls(
        int(spec["clauses"]), int(spec["T"]), float(spec["s"]), q=float(spec["q"]),
        max_included_literals=spec["max_included_literals"],
        depth=int(spec["depth"]), message_size=int(spec["message_size"]),
        message_bits=int(spec["message_bits"]), grid=grid, block=block,
    )

    if spec["task"] == "masked_windows":
        def fit_fn() -> None:  # noqa: F811 - the class count is fixed at vocab (docs/02 blocker 1)
            e1.fit_graphs(model, graphs, y, outputs)

    result: dict[str, Any] = {
        "spec": spec,
        "import_s": import_s,
        "graph_build_s": graph_build_s,
        "graphs_per_s_build": n_graphs / graph_build_s,
        "n_distinct_labels": int(np.unique(task["y"]).size),
        "gpu_used_mib_context_only": mem_context_mib,
    }
    mode = spec["mode"]
    if mode == "time":
        timing = timed_fit(model, graphs, y, cuda, reps=int(spec["reps"]), progress=progress,
                           fit_fn=fit_fn)
        result.update(timing)
    elif mode in ("events", "nvprof"):
        if fit_fn is None:
            def fit_fn() -> None:  # noqa: F811
                model.fit(graphs, y, epochs=1, incremental=True)
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        fit_fn()  # warm-up, not profiled
        cuda.Context.synchronize()
        result["warmup_fit_s"] = time.perf_counter() - t0
        sink = install_event_timers(model, cuda) if mode == "events" else None
        walls = []
        per_rep_events = []
        if mode == "nvprof":
            cuda.start_profiler()
        for rep in range(int(spec["reps"])):
            cuda.Context.synchronize()
            t0 = time.perf_counter()
            fit_fn()
            cuda.Context.synchronize()
            walls.append(time.perf_counter() - t0)
            if sink is not None:
                per_rep_events.append(drain_events(sink))
            print(f"{progress}{mode} rep {rep + 1}: fit {walls[-1]:.3f} s", flush=True)
        if mode == "nvprof":
            cuda.stop_profiler()
        result["fit_s"] = summarize(walls)
        result["profiled_wall_s_total"] = float(sum(walls))
        result["profiled_examples_total"] = n_graphs * len(walls)
        if per_rep_events:
            totals: dict[str, dict[str, float]] = {}
            for rep_events in per_rep_events:
                for name, entry in rep_events.items():
                    agg = totals.setdefault(name, {"ms": 0.0, "calls": 0})
                    agg["ms"] += entry["ms"]
                    agg["calls"] += entry["calls"]
            result["event_kernel_ms"] = totals
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if model.number_of_outputs != outputs:
        raise AssertionError(
            f"stock fit() set number_of_outputs={model.number_of_outputs}, wanted {outputs}")
    result["number_of_outputs"] = int(model.number_of_outputs)
    result["label_matrix_shape"] = [int(v) for v in model.encoded_Y.shape]
    result["label_matrix_dtype"] = str(model.encoded_Y.dtype)
    result["label_matrix_bytes"] = int(model.encoded_Y.nbytes)
    result["gpu_used_mib_after_training"] = gpu_used_mib(cuda)
    result["gpu_model_and_data_mib"] = result["gpu_used_mib_after_training"] - mem_context_mib
    result["gpu_alloc_after_training"] = tracker.snapshot()
    result["gpu_alloc_mib_after_training"] = tracker.total_bytes / 2**20
    if spec["score_after"]:
        t0 = time.perf_counter()
        pred = model.predict(graphs)
        result["predict_s"] = time.perf_counter() - t0
        if spec["model"] == "multiclass":
            result["train_accuracy"] = float(np.mean(pred == task["y"]))
            result["train_accuracy_clean_subset"] = float(
                np.mean(pred[task["clean"]] == task["y"][task["clean"]]))
        result["gpu_used_mib_after_predict"] = gpu_used_mib(cuda)
        result["gpu_alloc_after_predict"] = tracker.snapshot()
        result["gpu_alloc_mib_after_predict"] = tracker.total_bytes / 2**20
    if "fit_s" in result:
        result["examples_per_s"] = n_graphs / result["fit_s"]["median"]
        result["examples_per_s_at_min_time"] = n_graphs / result["fit_s"]["min"]
        result["examples_per_s_at_max_time"] = n_graphs / result["fit_s"]["max"]
        result["ms_per_example"] = 1000.0 * result["fit_s"]["median"] / n_graphs
    if "inner_fit_s" in result:
        result["examples_per_s_inner"] = n_graphs / result["inner_fit_s"]["median"]
        result["host_label_encoding_s"] = (
            result["fit_s"]["median"] - result["inner_fit_s"]["median"])
    tracker.uninstall()
    del model
    gc.collect()
    return result


def spawn_worker(spec: dict[str, Any], work_dir: Path, prefix: Sequence[str] = (),
                 timeout_s: float = 3600.0) -> dict[str, Any]:
    """Run one worker subprocess. Its output is streamed to our stdout and kept in
    <work_dir>/<tag>.log. Returns the worker's result, or {"error": ...} on failure."""
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = spec["tag"]
    spec_path = work_dir / f"{tag}.spec.json"
    out_path = work_dir / f"{tag}.result.json"
    log_path = work_dir / f"{tag}.log"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [*prefix, sys.executable, str(Path(__file__).resolve()),
               "--worker", str(spec_path), "--out", str(out_path)]
    print(f"[spawn] {' '.join(command)}", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                if time.perf_counter() - t0 > timeout_s:
                    proc.kill()
                    break
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    wall = time.perf_counter() - t0
    if out_path.is_file():
        result = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        result = {"error": "worker wrote no result file"}
    result["worker_exit_code"] = code
    result["worker_wall_s"] = wall
    result["worker_log"] = log_path.name
    if code != 0 and "error" not in result:
        result["error"] = f"worker exit code {code}"
    if "error" in result:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        result["log_tail"] = tail
    return result


# --------------------------------------------------------------------------- run dir


class _Run:
    def __init__(self, run_dir: Path, runlog_obj: Any) -> None:
        self.run_dir = run_dir
        self._runlog = runlog_obj
        self.metrics: dict[str, dict[str, Any]] = {}

    def add_metric(self, name: str, value: Any, label: str, **extra: Any) -> None:
        self.metrics[name] = {"value": value, "label": label, **extra}

    def write_json(self, relname: str, obj: Any) -> Path:
        from tmlm.common import runlog

        target = self.run_dir / relname
        target.parent.mkdir(parents=True, exist_ok=True)
        runlog.write_json(target, obj, allow_nan=True)
        return target


@contextlib.contextmanager
def open_run(name: str, config: Any) -> Iterator[_Run]:
    """Yield a run handle. Under scripts/run_logged.py ($TMLM_RUN_DIR is set) the wrapper
    owns the run dir and metrics.json is written for it to merge. Otherwise a RunLog is
    opened here, so the script is also governed when started directly."""
    from tmlm.common import runlog

    wrapped = runlog.current_run_dir()
    if wrapped is not None:
        run = _Run(wrapped, None)
        try:
            yield run
        finally:
            runlog.write_metrics(run.metrics, wrapped)
        return
    with runlog.RunLog(name, config, root=ROOT) as log:
        assert log.run_dir is not None
        run = _Run(log.run_dir, log)
        try:
            yield run
        finally:
            log.set_metrics(run.metrics)


def load_config(path: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(default))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 worker: one stock model per process")
    parser.add_argument("--worker", required=True, help="spec JSON file")
    parser.add_argument("--out", required=True, help="result JSON file")
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.worker).read_text(encoding="utf-8"))
    out = Path(args.out)
    try:
        result = run_worker(spec)
        code = 0
    except BaseException as exc:  # noqa: BLE001 - the error text IS the result
        traceback.print_exc()
        result = {"spec": spec, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
        code = 1
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
