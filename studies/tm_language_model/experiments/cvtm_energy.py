"""GPU energy of the CVTM-core fit and the verdicts of lane GX (docs/13 amendment A3 H8.5 and H8.6;
amendment A4 H9.7 pilot and gate G1 frame). Runs in .venv-tm. VALIDATION ONLY: the test split is never
loaded.

Mode "fit_energy" (config `experiments/cvtm/configs/H8_5_energy.json`, key "cvtm_fit"; EXCLUSIVE GPU lock):
- the frozen fit `CVTMCore.fit(train, vocab=8000, window=None, min_counts=None, components="full",
  backend="cupy", max_device_bytes=9 GiB)` (the call of `results/20260924T085728Z_cvtm_cache_build/`
  without its cache callbacks), repeated `repeats` times (docs/13 section 5: median of at least 5);
- its four GPU steps (the two `kn_gpu.fit_kn` calls and the two `SuffixClauses.from_kn` calls) are
  wrapped for measurement only: `tmlm.cvtm.energy.EnergyMeter.measure` around the unchanged call, closed
  by a device synchronise; the wrappers are removed after every fit (and the fit's state hash is checked
  against the frozen core's, so the measured work is the frozen fit);
- a second `tmlm.cvtm.energy.NvmlPowerSampler` covers the whole `CVTMCore.fit` call (GPU and host steps);
- idle: 10 s with the CuPy CUDA context created, once at the start (the transformer_small method); a 10 s
  idle sample after every repeat is logged as a diagnostic only;
- settle (the lane's energy protocol, config key "energy_protocol", fixed from the GX idle probe): before the
  idle measurement and before every repeat after the first, the process waits `settle_s` seconds with
  nothing running (sampled and logged), so that no idle level and no measured interval starts in the tail of
  earlier GPU activity (results/20260924T151319Z_cvtm_H8_5_fit_energy/ measured 86.3 W as its idle one
  second after another parallel job ended; settled idle is about 72 W);
- host CPU seconds (process CPU time, all threads) per step and per fit; CPU energy is not readable here;
- after the last repeat (outside every measurement): the refitted core scores validation and must give
  the frozen 1.240880320113308 bits per byte exactly;
- reported: median, min and max over the repeats of the idle-subtracted joules of the GPU steps (the H8.5
  quantity) and of the whole fit; the derived cap J / 259.2 W of the matched transformer;
- by-product: training tokens per second of the whole fit (a timing under the exclusive lock; the H8.8 (c)
  verdict belongs to its own lane).
`--prefix-tokens N` fits on the leading whole documents of the split (for E*(D) of H9.7 at the H8.8
prefixes, later; the frozen state-hash and validation checks then do not apply and are not run).

Mode "verdict" (config `experiments/cvtm/configs/GX_verdicts.json`; CPU only, no GPU): reads the finished
runs given on the command line and the frozen cache, and writes every verdict of lane GX with its number,
threshold and paired document-block bootstrap interval (60 validation documents, 2,000 resamples,
`tmlm.cvtm.protocol.bootstrap_bpb`, labelled derived):
- H8.5: CVTM-core validation bits per byte < the reported matched transformer's (the rerun when the rule
  triggered one); joule ratios to the 600 s transformer (old shared-lock 155,519.8 J and the new
  exclusive-lock re-measurement);
- H8.6: placement against the 600 s L6 W256 transformer, old number and exclusive re-measurement;
- H9.7 pilot and gate G1 frame: the tuned 600 s transformer's best validation bits per byte against 1.2209.

Launch lines: "launch" of the configs. Smoke (not logged): `--smoke` (fit_energy: 3M-token prefix, 2
repeats, output under $TMLM_SCRATCH/cvtm_gx_smoke/; verdict: accepts smoke directories).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.cvtm import energy as E  # noqa: E402
from tmlm.cvtm import kn_gpu, protocol  # noqa: E402
from tmlm.cvtm.suffix import SuffixClauses  # noqa: E402

if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
from cvtm_gx_settle import settle as _settle, settle_protocol  # noqa: E402


def settle(seconds: float, tag: str) -> dict:
    return _settle(seconds, tag, E.NvmlPowerSampler, log)


LN2 = math.log(2.0)
GIB = 1024.0 ** 3
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False
OLD_600S_JOULES = 155_519.81701787355       # results/20260922T172902Z_M2_baselines_transformer/ (shared lock)
OLD_600S_BPB = 1.2340551926476289
REGISTERED_POWER_W = 259.2                   # W, docs/13 A3 H8.5 (155,519.8 J / 600 s, derived there)
G1_THRESHOLD = 1.2209                         # docs/13 A4 gate G1 frame (CVTM-core minus 0.02)
H9_7_PREDICTION_REF = 1.2409                  # docs/13 A4 H9.7 prediction "the tuned 600 s model beats 1.2409"


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[cvtm_energy {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if SMOKE:
        label = "unverified"
    METRICS[name] = {"value": value, "label": label, **extra}
    if runlog.current_run_dir() is not None and not SMOKE:
        runlog.write_metrics({name: METRICS[name]}, on_conflict="replace")
    if isinstance(value, (int, float, str, bool)) or value is None:
        log(f"{name} = {value}")


def check(name: str, ok: bool, **details) -> None:
    M(f"check.{name}", bool(ok), details=details)
    if not ok:
        FAILED.append(name)
        log(f"CHECK FAILED: {name} {details}")


def run_dir(tag: str) -> Path:
    d = runlog.current_run_dir()
    if d is not None and not SMOKE:
        return Path(d)
    p = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "cvtm_gx_smoke" / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ memory rule
CG = Path("/sys/fs/cgroup")


def cgroup_bytes() -> tuple[int, int]:
    cur = int((CG / "memory.current").read_text().strip())
    mx_txt = (CG / "memory.max").read_text().strip()
    return cur, (int(mx_txt) if mx_txt != "max" else 1 << 62)


def wait_for_memory(peak_gib: float, tag: str, poll_s: int = 120, max_wait_s: int = 6 * 3600) -> int:
    """Start only when memory.current + expected peak < 0.80 memory.max (poll every 2 minutes)."""
    waited = 0
    while True:
        cur, mx = cgroup_bytes()
        if cur + peak_gib * GIB < 0.80 * mx:
            log(f"memory before {tag}: current {cur / GIB:.1f} GiB + peak {peak_gib:.0f} GiB < 0.80 x {mx / GIB:.0f} GiB "
                f"(waited {waited} s)")
            return waited
        if waited >= max_wait_s:
            raise RuntimeError(f"memory never allowed {tag}")
        log(f"memory before {tag}: current {cur / GIB:.1f} GiB + {peak_gib:.0f} GiB >= 0.80 x {mx / GIB:.0f} GiB; "
            f"waiting {poll_s} s")
        time.sleep(poll_s)
        waited += poll_s


# ------------------------------------------------------------------ measured GPU steps (tested)
GPU_STEP_NAMES = ("kn_noH.fit", "kn_noH.suffix", "kn_all.fit", "kn_all.suffix")


@contextmanager
def measured_gpu_steps(meter: E.EnergyMeter, prefix: str, sync=None):
    """While active, `kn_gpu.fit_kn` and `SuffixClauses.from_kn` (the GPU steps of `CVTMCore.fit`) run
    unchanged inside `meter.measure(<prefix>.<step>)`, closed by `sync()`. Yields the list of step names in
    call order; both originals are restored on exit (also after an exception)."""
    orig_fit = kn_gpu.fit_kn
    orig_from = SuffixClauses.__dict__["from_kn"]
    names: list[str] = []
    count = {"fit": 0, "suffix": 0}
    tags = ("kn_noH", "kn_all")

    def tag_of(kind: str) -> str:
        i = count[kind]
        count[kind] += 1
        return tags[i] if i < len(tags) else f"extra{i}"

    def fit_kn_measured(*a, **kw):
        name = f"{prefix}.{tag_of('fit')}.fit"
        with meter.measure(name):
            out = orig_fit(*a, **kw)
            if sync is not None:
                sync()
        names.append(name)
        return out

    def from_kn_measured(cls, *a, **kw):
        name = f"{prefix}.{tag_of('suffix')}.suffix"
        with meter.measure(name):
            out = orig_from.__func__(cls, *a, **kw)
            if sync is not None:
                sync()
        names.append(name)
        return out

    kn_gpu.fit_kn = fit_kn_measured
    SuffixClauses.from_kn = classmethod(from_kn_measured)
    try:
        yield names
    finally:
        kn_gpu.fit_kn = orig_fit
        SuffixClauses.from_kn = orig_from


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": int(a.shape[0]),
            "values": [float(x) for x in a]}


# ------------------------------------------------------------------ mode fit_energy
def fit_energy(cfg: dict, args) -> int:
    from tmlm.cvtm.model import CVTMCore

    fc = cfg["cvtm_fit"]
    V = 8000
    repeats = int(fc["repeats"]) if not SMOKE else int(args.smoke_repeats)
    pool_limit = int(float(fc["gpu_pool_limit_gib"]) * GIB)
    prefix = args.prefix_tokens if not SMOKE else args.smoke_train_tokens
    frozen = prefix is None and not SMOKE
    rd = run_dir("fit_energy")
    lock = os.environ.get("TMLM_GPU_LOCK_MODE")
    M("gpu_lock_mode", lock, note="energy is reportable only under the exclusive lock")
    M("config.cvtm_fit", fc)
    M("prefix_tokens", prefix)
    M("dataset_hashes", protocol.write_dataset_hashes(("train", "validation"), protocol.CORPUS, V),
      note="sha256 entries of the stores' meta.json files (also in dataset_hashes.json)")
    t0 = time.time()
    tok, ds = protocol.load_train(protocol.CORPUS, V, check=True)
    tok = np.ascontiguousarray(np.asarray(tok))
    ptok, pds = protocol.prefix_documents(tok, ds, prefix)
    n_train = int(np.asarray(pds)[-1])
    M("data.train_tokens_fitted", n_train)
    M("data.train_docs_fitted", int(np.asarray(pds).shape[0] - 1))
    M("data.load_wall_s", round(time.time() - t0, 1), unit="s")

    import cupy

    cupy.cuda.Device(0).use()
    cupy.zeros(1, dtype=cupy.float32).sum()
    cupy.cuda.Device().synchronize()

    def sync() -> None:
        cupy.cuda.Device().synchronize()

    def free_pools() -> None:
        sync()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    settle_s = settle_protocol(cfg) if not SMOKE else float(args.smoke_settle_s)
    M("energy_protocol.settle_s", settle_s, label="derived", source=cfg.get("energy_protocol", {}).get("source"),
      note="wait with nothing running before the idle measurement and before every repeat")
    M("settle.start", settle(settle_s, "start"))
    meter = E.EnergyMeter(idle_s=10.0)
    M("nvml.available", bool(meter.sampler.available), error=meter.sampler.error)
    M("nvml.idle", meter.idle, note="10 s with the CuPy CUDA context created and nothing running, after the settle "
                                    "(transformer_small method plus the lane's settle rule)")
    M("nvml.idle_w", meter.idle_w)
    whole = E.NvmlPowerSampler()
    nv = meter.sampler._nvml if meter.sampler.available else None

    def temp() -> int | None:
        try:
            return int(nv.nvmlDeviceGetTemperature(meter.sampler._handle, 0)) if nv is not None else None
        except Exception:  # noqa: BLE001
            return None

    per_rep: list[dict] = []
    core = None
    for r in range(1, repeats + 1):
        M(f"repeat{r}.memory_wait_s", wait_for_memory(float(fc.get("peak_gib", 24)), f"repeat{r}"))
        if r > 1:
            M(f"settle.before_repeat{r}", settle(settle_s, f"before repeat{r}"))
        t_start = temp()
        c0 = time.process_time()
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        tw0 = time.perf_counter()
        whole.start()
        with measured_gpu_steps(meter, f"repeat{r}", sync) as names:
            core = CVTMCore.fit(ptok, pds, vocab=V, prefix_tokens=None, window=None, min_counts=None, components="full",
                                backend="cupy", max_device_bytes=pool_limit, expect_h=None if frozen else False)
            sync()
        wf = whole.stop()
        wall = time.perf_counter() - tw0
        cpu_s = time.process_time() - c0
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        t_end = temp()
        check(f"repeat{r}.four_gpu_steps_in_order", [n.split(".", 1)[1] for n in names] == list(GPU_STEP_NAMES), names=names)
        steps = {n.split(".", 1)[1]: meter.results[n] for n in names}
        jw = E.idle_subtracted(wf, meter.idle_w)
        gpu_j = [steps[s]["joules_idle_subtracted"] for s in GPU_STEP_NAMES if s in steps]
        gpu_j_is = float(sum(gpu_j)) if gpu_j and all(v is not None for v in gpu_j) else None
        gpu_raw = [steps[s]["joules_raw"] for s in GPU_STEP_NAMES if s in steps]
        rec = {
            "gpu_steps": {s: steps[s] for s in GPU_STEP_NAMES if s in steps},
            "gpu_steps_joules_idle_subtracted": gpu_j_is,
            "gpu_steps_joules_raw": float(sum(gpu_raw)) if gpu_raw and all(v is not None for v in gpu_raw) else None,
            "gpu_steps_wall_s": float(sum(steps[s]["wall_s"] for s in GPU_STEP_NAMES if s in steps)),
            "gpu_steps_cpu_s": float(sum(steps[s]["cpu_s"] for s in GPU_STEP_NAMES if s in steps)),
            "whole_fit": {"joules_raw": wf.get("joules"), "joules_idle_subtracted": jw, "span_s": wf.get("span_s"),
                          "mean_w": wf.get("mean_w"), "n_samples": wf.get("n_samples"), "wall_s": wall, "cpu_s": cpu_s,
                          "rusage_user_s": ru1.ru_utime - ru0.ru_utime, "rusage_sys_s": ru1.ru_stime - ru0.ru_stime,
                          "max_rss_gib": ru1.ru_maxrss / 1024.0 ** 2},
            "host_steps_joules_idle_subtracted": (jw - gpu_j_is) if (jw is not None and gpu_j_is is not None) else None,
            "phases_s": core.fit_info["phases_s"], "fit_total_s": core.fit_info["total_s"],
            "kn_noH_fit_info_total_s": core.fit_info["kn_noH.fit_info"]["total_s"],
            "kn_all_fit_info_total_s": core.fit_info["kn_all.fit_info"]["total_s"],
            "gpu_temp_c_start_end": [t_start, t_end], "state_hash": core.state_hash(),
            "train_tokens_per_s": n_train / wall,
        }
        per_rep.append(rec)
        M(f"repeat{r}", rec, note=f"lock {lock}; GPU steps measured with tmlm.cvtm.energy.EnergyMeter; whole fit with a "
                                  "second sampler; host steps = whole minus GPU steps (derived)")
        if frozen:
            check(f"repeat{r}.state_hash_equals_frozen", rec["state_hash"] == fc["frozen_state_hash"],
                  got=rec["state_hash"], frozen=fc["frozen_state_hash"])
        elif r > 1:
            check(f"repeat{r}.state_hash_equals_repeat1", rec["state_hash"] == per_rep[0]["state_hash"])
        if r < repeats:
            del core
            core = None
            gc.collect()
            free_pools()
        idle_after = whole.measure_idle(10.0) if whole.available else None
        M(f"repeat{r}.idle_after_w_diagnostic", None if idle_after is None else idle_after.get("mean_w"),
          note="10 s idle after the repeat; diagnostic, not used")

    # ---- summaries
    def s_of(key_fn) -> dict | None:
        vals = [key_fn(x) for x in per_rep]
        return summarize(vals) if all(v is not None for v in vals) else None

    gsum = s_of(lambda x: x["gpu_steps_joules_idle_subtracted"])
    M("cvtm_fit.gpu_steps.joules_idle_subtracted", gsum,
      note=f"per repeat: sum over the four GPU steps; median over {repeats} repeats (docs/13 section 5); lock {lock}")
    M("cvtm_fit.gpu_steps.joules_idle_subtracted.median", None if gsum is None else gsum["median"],
      note="THE H8.5 CVTM joules (docs/13 A3: GPU steps, idle subtracted, exclusive lock)")
    M("cvtm_fit.gpu_steps.joules_raw", s_of(lambda x: x["gpu_steps_joules_raw"]))
    M("cvtm_fit.gpu_steps.wall_s", s_of(lambda x: x["gpu_steps_wall_s"]))
    M("cvtm_fit.gpu_steps.host_cpu_s", s_of(lambda x: x["gpu_steps_cpu_s"]))
    for s in GPU_STEP_NAMES:
        M(f"cvtm_fit.step.{s}.joules_idle_subtracted", s_of(lambda x, s=s: x["gpu_steps"][s]["joules_idle_subtracted"]))
        M(f"cvtm_fit.step.{s}.wall_s", s_of(lambda x, s=s: x["gpu_steps"][s]["wall_s"]))
    M("cvtm_fit.whole.joules_idle_subtracted", s_of(lambda x: x["whole_fit"]["joules_idle_subtracted"]),
      note="the whole CVTMCore.fit call (GPU and host steps), same method")
    M("cvtm_fit.whole.joules_raw", s_of(lambda x: x["whole_fit"]["joules_raw"]))
    M("cvtm_fit.whole.wall_s", s_of(lambda x: x["whole_fit"]["wall_s"]))
    M("cvtm_fit.whole.host_cpu_s", s_of(lambda x: x["whole_fit"]["cpu_s"]),
      note="process CPU seconds of the whole fit (all threads); CPU energy is not readable in this container (no RAPL)")
    M("cvtm_fit.host_steps.joules_idle_subtracted", s_of(lambda x: x["host_steps_joules_idle_subtracted"]), label="derived",
      note="whole fit minus GPU steps: the GPU's above-idle energy while the host steps run")
    M("cvtm_fit.train_tokens_per_s", s_of(lambda x: x["train_tokens_per_s"]),
      note="training tokens / whole-fit wall (end-to-end fit, exclusive GPU lock, 2 CPU slots on a shared host); "
           "a by-product timing, the H8.8 (c) verdict belongs to its own lane")
    if gsum is not None:
        M("h8_5.transformer_time_cap_s", gsum["median"] / REGISTERED_POWER_W, label="derived",
          note="J_cvtm / 259.2 W (docs/13 A3 H8.5)")
        M("h8_5.joule_ratio_vs_600s_old", gsum["median"] / OLD_600S_JOULES, label="derived",
          note="CVTM-core fit GPU-step joules / 155,519.8 J (the old 600 s run, SHARED lock); no threshold")

    # ---- the refitted core scores validation exactly as the frozen one (outside the measurements)
    if frozen and core is not None:
        t0 = time.time()
        ev = protocol.load_eval_split("validation", protocol.CORPUS, V, check=True)
        sc = core.score_stream(ev.tok, ev.doc_start, n_bytes=ev.n_bytes)
        M("check.validation_bits_per_byte_of_refit", sc["bits_per_byte"], wall_s=round(time.time() - t0, 1))
        check("refit_validation_bits_per_byte_equals_frozen", sc["bits_per_byte"] == float(fc["frozen_validation_bits_per_byte"]),
              refit=sc["bits_per_byte"], frozen=fc["frozen_validation_bits_per_byte"])
    del core
    gc.collect()
    M("peak_rss_gib", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2, 2),
      note="VmHWM-equivalent of the driver process")
    M("wall_s", round(time.time() - T0, 1), unit="s")
    M("checks_failed", list(FAILED))
    if SMOKE:
        (rd / "metrics.json").write_text(json.dumps(METRICS, indent=1, default=float))
        (rd / "run.json").write_text(json.dumps({"exit_status": 0 if not FAILED else 1, "smoke": True}))
    return 1 if FAILED else 0


# ------------------------------------------------------------------ mode verdict
def _run(path: str | None) -> tuple[Path, dict, dict] | None:
    if not path:
        return None
    p = Path(path)
    rj = json.loads((p / "run.json").read_text())
    if rj.get("exit_status") != 0:
        raise SystemExit(f"{p}: exit status {rj.get('exit_status')}; not usable")
    mt = json.loads((p / "metrics.json").read_text())
    return p, rj, mt


def _v(mt: dict, name: str):
    if name not in mt:
        raise KeyError(f"metric {name} missing")
    return mt[name]["value"]


def require_reportable(p: Path, mt: dict) -> None:
    """A run feeding a verdict: exclusive GPU lock and no failed pipeline check (smoke directories exempt)."""
    if SMOKE:
        return
    if _v(mt, "gpu_lock_mode") != "exclusive":
        raise SystemExit(f"{p}: not an exclusive-lock measurement")
    if "energy_protocol.settle_s" not in mt:
        raise SystemExit(f"{p}: measured without the settle rule of the lane's energy protocol")
    if _v(mt, "checks_failed"):
        raise SystemExit(f"{p}: failed checks {_v(mt, 'checks_failed')}")


M2_SETTINGS_KEYS = {"layers": "layers", "width": "width", "heads": "heads", "seq_len": "seq_len", "batch_seqs": "batch_seqs",
                    "lr": "lr", "warmup_steps": "warmup_steps", "seed": "seed"}


def check_m2_settings(p: Path, mt: dict, settings: dict, name: str = "transformer_L6_W256") -> None:
    """The matched and 600 s runs used exactly the M2 settings (fp32) of the H8.5 config."""
    got = {k: _v(mt, f"{name}.{k}") for k in M2_SETTINGS_KEYS}
    want = {k: settings[v] for k, v in M2_SETTINGS_KEYS.items()}
    check(f"{p.name}.m2_settings", got == want and _v(mt, f"{name}.amp_fp16") is False, got=got, want=want)


def doc_bits_of(bits_per_token: np.ndarray, doc_start: np.ndarray) -> np.ndarray:
    ds = np.asarray(doc_start).astype(np.int64)
    doc_of = np.repeat(np.arange(ds.shape[0] - 1, dtype=np.int64), np.diff(ds))
    return np.bincount(doc_of, weights=np.asarray(bits_per_token, dtype=np.float64), minlength=ds.shape[0] - 1)


def tf_post_idle_joules(mt: dict, name: str) -> float | None:
    raw = _v(mt, f"{name}.joules_raw")
    idle_after = _v(mt, f"{name}.post_training_idle_w_diagnostic")
    span = _v(mt, f"{name}.energy_span_s")
    return None if raw is None or idle_after is None or span is None else float(raw) - float(idle_after) * float(span)


def verdict(cfg: dict, args) -> int:
    rd = run_dir("verdict")
    h85 = json.loads((ROOT / cfg["configs"]["H8_5"]).read_text())
    h97 = json.loads((ROOT / cfg["configs"]["H9_7"]).read_text())
    M("configs_sha256", {k: hashing.config_hash(json.loads((ROOT / v).read_text())) for k, v in cfg["configs"].items()})
    V = 8000
    protocol.write_dataset_hashes(("validation",), protocol.CORPUS, V)
    ev = protocol.load_eval_split("validation", protocol.CORPUS, V, check=True)
    idx = protocol.bootstrap_index(ev.n_docs)

    # ---- the frozen CVTM-core per document (L0 cache)
    cache = ROOT / cfg["cache_dir"]
    p_core = np.load(cache / "validation" / "core_p_gold.npy")
    ds_cache = np.load(cache / "validation" / "doc_start.npy")
    check("cache.validation_doc_start_equals_split", bool(np.array_equal(ds_cache.astype(np.int64), ev.doc_start)))
    core_bits = -np.log2(p_core)
    core_bpb = float(core_bits.sum() / ev.n_bytes)
    frozen = float(h85["cvtm_fit"]["frozen_validation_bits_per_byte"])
    check("cache.core_bits_per_byte_equals_frozen", abs(core_bpb - frozen) <= 1e-12, cache=core_bpb, frozen=frozen)
    core_doc = doc_bits_of(core_bits, ev.doc_start)
    M("cvtm_core.validation_bits_per_byte", frozen, source=h85["cvtm_fit"]["frozen_run"])
    b = protocol.bootstrap_bpb(core_doc, ev.doc_bytes, idx)
    M("cvtm_core.validation_bits_per_byte_ci95", b["ci95"], label="derived", note="document-block bootstrap")

    def tf_doc_bits(run: Path, mt: dict, name: str) -> tuple[float, np.ndarray]:
        bits = np.load(run / _v(mt, f"{name}.validation_bits_file"))
        if bits.shape[0] != ev.tok.shape[0]:
            raise SystemExit(f"{run}: per-token file has {bits.shape[0]} rows")
        bpb = float(_v(mt, f"{name}.validation_bits_per_byte"))
        rel = abs(float(bits.sum()) / ev.n_bytes - bpb) / bpb
        check(f"{run.name}.{name}.per_token_file_reproduces_bpb", rel <= 1e-6, rel_diff=rel)
        return bpb, doc_bits_of(bits, ev.doc_start)

    def paired(name: str, a_doc: np.ndarray, b_doc: np.ndarray, note: str) -> dict:
        r = protocol.bootstrap_bpb(a_doc, ev.doc_bytes, idx, b_doc)
        M(name, r["point"], label="derived", ci95=r["ci95"], note=note + " (paired document-block bootstrap, 60 documents, "
                                                                          "2,000 resamples)")
        return r

    # ---- H8.5
    fe = _run(args.fit_energy)
    j_cvtm = None
    if fe is not None:
        p, rj, mt = fe
        require_reportable(p, mt)
        if not SMOKE:
            check("fit_energy.full_split_frozen_fit", _v(mt, "prefix_tokens") is None
                  and _v(mt, "data.train_tokens_fitted") == 135246414
                  and _v(mt, "check.refit_validation_bits_per_byte_equals_frozen") is True)
        j_cvtm = float(_v(mt, "cvtm_fit.gpu_steps.joules_idle_subtracted.median"))
        M("h8_5.cvtm_fit_joules", j_cvtm, source=str(p), note="median of the GPU-step idle-subtracted joules over the repeats",
          range=_v(mt, "cvtm_fit.gpu_steps.joules_idle_subtracted"))
        M("h8_5.cvtm_fit_whole_joules", _v(mt, "cvtm_fit.whole.joules_idle_subtracted"), source=str(p))
        # sensitivity to the idle baseline (derived, not the registered quantity): the same raw joules with the
        # idle power sampled right AFTER each repeat (the board stays above its start idle after activity)
        sens = []
        r = 1
        while f"repeat{r}" in mt:
            rec = _v(mt, f"repeat{r}")
            idle_after = _v(mt, f"repeat{r}.idle_after_w_diagnostic")
            steps = rec["gpu_steps"].values()
            if idle_after is not None and all(s["joules_raw"] is not None for s in steps):
                sens.append(float(sum(s["joules_raw"] - idle_after * s["span_s"] for s in steps)))
            r += 1
        if sens:
            M("h8_5.sensitivity.gpu_steps_joules_with_post_repeat_idle", summarize(sens), label="derived",
              note="raw joules of the four GPU steps minus the idle power sampled after each repeat times the span; "
                   "shows how much of the registered quantity is the idle-baseline choice; not used by the verdict")
        raw = _v(mt, "cvtm_fit.gpu_steps.joules_raw")
        M("h8_5.cvtm_fit_gpu_steps_joules_raw", raw, source=str(p), note="no idle subtraction")
        M("h8_5.cvtm_fit_host_cpu_s", _v(mt, "cvtm_fit.whole.host_cpu_s"), source=str(p),
          note="CPU energy is not readable in this container")
    mfirst = _run(args.matched)
    mre = _run(args.matched_rerun)
    t600 = _run(args.t600)
    name = "transformer_L6_W256"
    for run in (mfirst, mre, t600):
        if run is not None:
            require_reportable(run[0], run[2])
            check_m2_settings(run[0], run[2], h85["transformer_m2_settings"])
    if mfirst is not None and j_cvtm is not None:
        p1, _, mt1 = mfirst
        j1 = float(_v(mt1, f"{name}.joules_idle_subtracted"))
        rerun_needed = abs(j1 - j_cvtm) / j_cvtm > float(0.10)
        M("h8_5.first_matched", {"run": str(p1), "time_cap_s": _v(mt1, "time_cap_s"), "joules_idle_subtracted": j1,
                                 "rel_diff_vs_cvtm": (j1 - j_cvtm) / j_cvtm,
                                 "validation_bits_per_byte": _v(mt1, f"{name}.validation_bits_per_byte"),
                                 "steps": _v(mt1, f"{name}.steps")})
        M("h8_5.rerun_rule_triggered", rerun_needed, label="derived", note="|J_tf - J_cvtm| / J_cvtm > 0.10")
        if rerun_needed and mre is None:
            raise SystemExit("the H8.5 rule requires the rerun; pass --matched-rerun")
        if not rerun_needed and mre is not None:
            raise SystemExit("a rerun was passed but the rule did not trigger one")
        rep = mre if rerun_needed else mfirst
        pr, _, mtr = rep
        jr = float(_v(mtr, f"{name}.joules_idle_subtracted"))
        bpb_tf, tf_doc = tf_doc_bits(pr, mtr, name)
        M("h8_5.reported_matched", {"run": str(pr), "time_cap_s": _v(mtr, "time_cap_s"), "joules_idle_subtracted": jr,
                                    "rel_diff_vs_cvtm": (jr - j_cvtm) / j_cvtm, "validation_bits_per_byte": bpb_tf,
                                    "steps": _v(mtr, f"{name}.steps"), "train_wall_s": _v(mtr, f"{name}.train_wall_s"),
                                    "tokens_predicted": _v(mtr, f"{name}.tokens_predicted")})
        M("h8_5.sensitivity.matched_joules_with_post_training_idle", tf_post_idle_joules(mtr, name), label="derived",
          note="raw joules minus the idle sampled after the training times the span; not used by the verdict")
        d = paired("h8_5.cvtm_minus_matched_transformer.bits_per_byte", core_doc, tf_doc, "CVTM-core minus the matched transformer")
        met = frozen < bpb_tf
        M("h8_5.verdict", "MET" if met else "NOT MET", label="derived",
          rule=h85["verdict"]["H8.5"], cvtm_bpb=frozen, transformer_bpb=bpb_tf, difference=d["point"], ci95=d["ci95"])
        if mre is not None and abs(jr - j_cvtm) / j_cvtm > 0.10:
            M("h8_5.rerun_still_outside_10_percent", True, label="derived",
              note="the rule allows one rerun only; the rerun is reported with its mismatch")
    if t600 is not None:
        p6, _, mt6 = t600
        j6 = float(_v(mt6, f"{name}.joules_idle_subtracted"))
        bpb6, doc6 = tf_doc_bits(p6, mt6, name)
        M("t600.exclusive", {"run": str(p6), "joules_idle_subtracted": j6, "joules_raw": _v(mt6, f"{name}.joules_raw"),
                             "idle_w": _v(mt6, f"{name}.idle_w"), "train_wall_s": _v(mt6, f"{name}.train_wall_s"),
                             "steps": _v(mt6, f"{name}.steps"), "validation_bits_per_byte": bpb6,
                             "mean_power_idle_subtracted_w": _v(mt6, f"{name}.mean_power_idle_subtracted_w")})
        M("t600.old_shared_lock", {"run": "results/20260922T172902Z_M2_baselines_transformer/", "joules_idle_subtracted": OLD_600S_JOULES,
                                   "validation_bits_per_byte": OLD_600S_BPB, "steps": 6505, "idle_w": 70.11199999999998},
          note="copied from that run's metrics.json (shared lock, indicative)")
        M("t600.joules_new_over_old", j6 / OLD_600S_JOULES, label="derived")
        M("t600.sensitivity.joules_with_post_training_idle", tf_post_idle_joules(mt6, name), label="derived",
          note="raw joules minus the idle sampled after the training times the span; not used by any verdict")
        if j_cvtm is not None:
            M("h8_5.joule_ratio_vs_600s_old", j_cvtm / OLD_600S_JOULES, label="derived", note="no threshold")
            M("h8_5.joule_ratio_vs_600s_exclusive", j_cvtm / j6, label="derived", note="no threshold")
        # H8.6 placement (no threshold)
        paired("h8_6.cvtm_minus_600s_exclusive.bits_per_byte", core_doc, doc6, "CVTM-core minus the 600 s L6 W256 re-measured")
        r_old = protocol.bootstrap_bpb(core_doc, ev.doc_bytes, idx, const=OLD_600S_BPB)
        M("h8_6.cvtm_minus_600s_old.bits_per_byte", r_old["point"], label="derived", ci95=r_old["ci95"],
          note="old number as a constant: the interval covers only the CVTM-core's document sampling")

    # ---- H9.7 pilot and gate G1 frame
    ls = _run(args.tuned_lrsel)
    t6 = _run(args.tuned_600)
    for run in (ls, t6):
        if run is not None:
            require_reportable(run[0], run[2])
    if ls is not None:
        pls, _, mls = ls
        for f in h97["model"]["families"]:
            rows = _v(mls, f"selection.{f['name']}.rows")
            energy_rows = {}
            for k in mls:
                pre = f"tuned_{f['name']}_lr"
                if k.startswith(pre) and k.endswith(".joules_idle_subtracted"):
                    nm = k[: -len(".joules_idle_subtracted")]
                    energy_rows[nm] = {"joules_idle_subtracted": _v(mls, k), "joules_raw": _v(mls, f"{nm}.joules_raw"),
                                       "train_wall_s": _v(mls, f"{nm}.train_wall_s"), "steps": _v(mls, f"{nm}.steps"),
                                       "validation_bits_per_byte": _v(mls, f"{nm}.validation_bits_per_byte")}
            M(f"h9_7.lrsel.{f['name']}", {"rows": rows, "chosen_lr": _v(mls, f"selection.{f['name']}.lr"), "runs": energy_rows},
              source=str(pls), note="validation bits per byte and NVML joules at the 150 s cap (seed 100, exploratory)")
    if t6 is not None:
        p7, _, m7 = t6
        rows = {}
        for k in m7:
            if k.startswith("tuned_") and k.endswith(".validation_bits_per_byte"):
                nm = k[: -len(".validation_bits_per_byte")]
                bpb_k, doc_k = tf_doc_bits(p7, m7, nm)
                rows[nm] = {"validation_bits_per_byte": bpb_k, "joules_idle_subtracted": _v(m7, f"{nm}.joules_idle_subtracted"),
                            "train_wall_s": _v(m7, f"{nm}.train_wall_s"), "steps": _v(m7, f"{nm}.steps"),
                            "tokens_predicted": _v(m7, f"{nm}.tokens_predicted"), "lr": _v(m7, f"{nm}.lr")}
                paired(f"h9_7.cvtm_minus_{nm}.bits_per_byte", core_doc, doc_k, f"CVTM-core minus {nm} (exploratory)")
                if j_cvtm is not None and rows[nm]["joules_idle_subtracted"] is not None:
                    M(f"h9_7.{nm}.joules_over_cvtm_fit_joules", rows[nm]["joules_idle_subtracted"] / j_cvtm, label="derived")
        M("h9_7.tuned_600s_rows", rows, note="seed 100, exploratory pilot")
        best_name = min(rows, key=lambda k: rows[k]["validation_bits_per_byte"])
        best = rows[best_name]["validation_bits_per_byte"]
        framed = best <= G1_THRESHOLD
        M("g1_frame.input", best, family=best_name, note="lowest validation bits per byte among the tuned 600 s runs")
        M("g1_frame.verdict", "ENERGY FRONTIER ONLY (no parity sentence)" if framed else "PARITY SENTENCE ALLOWED (against the tuned 600 s comparator only)",
          label="derived", threshold=G1_THRESHOLD, rule=h97["gate_G1_frame"]["rule"], caveat="pilot seed 100 (exploratory)")
        M("h9_7.prediction_tuned_600s_beats_1.2409", bool(best < H9_7_PREDICTION_REF), label="derived",
          note="docs/13 A4 H9.7 prediction (0.6); no threshold; exploratory")
    M("wall_s", round(time.time() - T0, 1), unit="s")
    M("checks_failed", list(FAILED))
    if SMOKE:
        (rd / "metrics.json").write_text(json.dumps(METRICS, indent=1, default=float))
    return 1 if FAILED else 0


def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("fit_energy", "verdict"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=None, help="fit_energy: leading whole documents (E*(D), later)")
    ap.add_argument("--fit-energy", default=None)
    ap.add_argument("--matched", default=None)
    ap.add_argument("--matched-rerun", default=None)
    ap.add_argument("--t600", default=None)
    ap.add_argument("--tuned-lrsel", default=None)
    ap.add_argument("--tuned-600", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-train-tokens", type=int, default=3_000_000)
    ap.add_argument("--smoke-repeats", type=int, default=2)
    ap.add_argument("--smoke-settle-s", type=float, default=3.0)
    args = ap.parse_args()
    SMOKE = bool(args.smoke)
    if SMOKE and os.environ.get("TMLM_RUN_DIR"):
        raise SystemExit("--smoke must not run under run_logged.py")
    cfg = json.loads(Path(args.config).read_text())
    if args.mode == "fit_energy":
        if not SMOKE and os.environ.get("TMLM_GPU_LOCK_MODE") != "exclusive":
            raise SystemExit("reportable energy needs scripts/gpu_run.sh exclusive")
        return fit_energy(cfg, args)
    return verdict(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
