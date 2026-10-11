"""CP: the causal anchor run and protocol CP (docs/05 CP; docs/06; docs/13 H1.1, amendment A2; docs/12 F6;
docs/04 M6). Config `experiments/configs/CP_anchor.json`: causal, V = 8000, C = 1,048,576, T = 64, s = 128,
B = 1024, 13 bags, WikiText-103 (train 135,246,414 tokens at V = 8000, docs/12 F16). One logged run per seed:

    scripts/gpu_run.sh shared -- scripts/run_logged.py --name CP_anchor --config experiments/configs/CP_anchor.json \\
        -- .venv-tm/bin/python -u experiments/cp_causal_perplexity.py --seed 1 [--epochs 4] [--resume ckpt.npz]

    # measure first (protocol rule 1): build cost, memory and a 200-batch throughput probe at C = 1M, no test access
    scripts/gpu_run.sh shared -- scripts/run_logged.py --name CP_anchor_probe --config experiments/configs/CP_anchor.json \\
        -- .venv-tm/bin/python -u experiments/cp_causal_perplexity.py --seed 1 --probe-batches 200

Protocol, in order:
1. Train on the CUDA engine (shared GPU lock, yielding to exclusive jobs at batch boundaries) for `--epochs`
   epochs (4, amendment A2) over every train position (causal: every position is a target). A compressed
   checkpoint (state + config + position, tmlm.engine.checkpoint) is written into the run dir after every
   epoch, and an uncompressed one into the scratch dir every `--checkpoint-every` batches, so the run survives
   interruptions and resumes bit-exactly with `--resume`.
2. After every epoch, on VALIDATION: the two calibration scalars of docs/03 (`logit = a * score + b * class_bias
   / 256`, `class_bias` = round(256 ln((n_c + 1) / (N + V))) from TRAIN counts, exactly what TrainedModel
   uses) are fitted on the validation NLL of a deterministic subsample of `--calib-rows` positions (every k-th
   position; the full [282k, 8000] float64 matrix would not fit); the fitted calibration then scores EVERY
   validation position in chunks; every row is asserted to sum to one over the full vocabulary. Reported:
   bits per byte = total negative log2-likelihood of the validation subword stream / UTF-8 bytes of the
   validation text; per-word perplexity = 2^(bits / n_words) [converted, open-vocabulary]; per-token
   perplexity; causal top-1 from the raw class sums and from the calibrated logits; the calibration report.
3. After the LAST epoch, and only after the validation metrics are on disk: the TEST split is loaded ONCE
   through the counted path and scored with the FROZEN calibration of the last epoch; the in-house modified
   Kneser-Ney 5-gram (src/tmlm/eval/baselines/ngram.py, refitted on the same train tokens; its validation bpb
   is compared with the M2 value 1.3501 as a check) scores the same test tokens. H1.1 (TM bpb below KN-5 bpb)
   and KB7 (TM bpb > 1.5 x KN-5) are evaluated for this seed. The 8-gram test overlap of docs/12 F16 is
   written next to the test number.
Also logged: wall time per epoch, examples/s, candidates, fired and events per example, alive clauses, CuPy
pool bytes, device-wide peak memory (run_logged), NVML joules per epoch with the idle power subtracted
(device-wide: other tenants' jobs on the shared GPU are included; labelled so).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

from engine_common import ROOT, EngineConfig, load_json

sys.path.insert(0, str(ROOT / "src"))
from tmlm.common import runlog  # noqa: E402
from tmlm.common.locks import yield_gpu_if_requested  # noqa: E402
from tmlm.data.corpus_store import load_split, raw_dir, split_dir  # noqa: E402
from tmlm.data.evalsets import test_access_count  # noqa: E402
from tmlm.decode.bank import class_bias_from_counts  # noqa: E402
from tmlm.decode.conftm import Calibration, fit_two_scalar  # noqa: E402
from tmlm.engine.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402

CORPUS = "wikitext103"
LN2 = math.log(2.0)
METRICS: dict[str, dict] = {}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[cp {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def metric(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, **extra}
    if runlog.current_run_dir() is not None:
        runlog.write_metrics({name: METRICS[name]})
    log(f"{name} = {value}")


def dataset_hash(name: str, digest: str) -> None:
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({name: digest}, on_conflict="replace")


def out_dir() -> Path:
    d = runlog.current_run_dir()
    if d is None:
        d = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "cp_unlogged"
        d.mkdir(parents=True, exist_ok=True)
    return Path(d)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


# ------------------------------------------------------------------ energy (the sampler of transformer_small.py)
class NvmlPowerSampler:
    """Samples device-wide GPU power (W) in a thread; `summary()` integrates. `available` is False without NVML."""

    def __init__(self, index: int = 0, period_s: float = 0.1) -> None:
        self.period_s = period_s
        self.samples: list[tuple[float, float]] = []
        self.available = False
        self.error = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period_s)

    def start(self) -> None:
        if self.available:
            self._stop.clear()
            self.samples = []
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        return self.summary()

    def summary(self, since: float | None = None) -> dict:
        s = [x for x in self.samples if since is None or x[0] >= since]
        if not self.available or len(s) < 2:
            return {"available": self.available, "n_samples": len(s), "joules": None, "mean_w": None, "span_s": 0.0, "error": self.error}
        t = np.array([x[0] for x in s])
        p = np.array([x[1] for x in s])
        joules = float(np.trapezoid(p, t)) if hasattr(np, "trapezoid") else float(np.trapz(p, t))
        return {"available": True, "n_samples": int(t.shape[0]), "joules": joules, "mean_w": float(p.mean()), "span_s": float(t[-1] - t[0])}


# ------------------------------------------------------------------ data
def split_meta(vocab: int, split: str) -> tuple[dict, dict]:
    meta = json.loads((split_dir(CORPUS, vocab, split) / "meta.json").read_text())
    raw = json.loads((raw_dir(CORPUS, split) / "meta.json").read_text())
    assert int(raw["n_bytes"]) == int(meta["n_bytes_utf8"])
    return meta, raw


def contiguous(corpus):
    from tmlm.engine.api import CorpusArrays
    return CorpusArrays(**{k: np.ascontiguousarray(getattr(corpus, k)) for k in ("tok", "doc_start", "kind_map", "stop", "sym_freq", "targets")})


# ------------------------------------------------------------------ scoring
def calibration_rows(n: int, rows: int) -> np.ndarray:
    k = max(1, int(math.ceil(n / rows)))
    return np.arange(0, n, k, dtype=np.uint32)


def fit_calibration(engine, tok: np.ndarray, class_bias: np.ndarray, rows: int, rounds: int) -> tuple[Calibration, dict]:
    pos = calibration_rows(tok.shape[0], rows)
    scores = engine.predict_scores(0, pos)
    cal, rep = fit_two_scalar(scores, tok[pos].astype(np.int64), class_bias, rounds=rounds)
    rep["rows"] = int(pos.shape[0])
    rep["row_stride"] = int(pos[1] - pos[0]) if pos.shape[0] > 1 else 1
    rep["rounds"] = rounds
    return cal, rep


def score_stream(engine, tok: np.ndarray, class_bias: np.ndarray, cal: Calibration, chunk: int) -> dict:
    """Every position of a split as a causal target: total bits, top-1 (raw and calibrated), normalisation check."""
    n = int(tok.shape[0])
    total_nats = 0.0
    hits_raw = hits_cal = 0
    worst_norm = 0.0
    idx = np.arange(chunk)
    for lo in range(0, n, chunk):
        pos = np.arange(lo, min(n, lo + chunk), dtype=np.uint32)
        gold = tok[pos].astype(np.int64)
        scores = engine.predict_scores(0, pos)
        hits_raw += int((scores.argmax(axis=1) == gold).sum())
        lp = cal.logprobs(scores, class_bias)
        norm = np.abs(np.log(np.exp(lp).sum(axis=1)))
        worst_norm = max(worst_norm, float(norm.max()))
        assert norm.max() < 1e-6, "a row of logprobs does not sum to one"
        total_nats += float(-lp[idx[: pos.shape[0]], gold].sum())
        hits_cal += int((lp.argmax(axis=1) == gold).sum())
    return {"n_positions": n, "total_bits": total_nats / LN2, "top1_raw": hits_raw / n, "top1_calibrated": hits_cal / n,
            "worst_row_normalisation_abs_log": worst_norm}


def report_split(prefix: str, sc: dict, meta: dict, note: str) -> None:
    bits = sc["total_bits"]
    n_bytes, n_words, n_tok = int(meta["n_bytes_utf8"]), int(meta["n_words"]), int(meta["n_tok"])
    assert sc["n_positions"] == n_tok
    metric(f"{prefix}.bits_per_byte", bits / n_bytes, note=f"total negative log2-likelihood of the subword stream / UTF-8 bytes ({n_bytes}); {note}")
    metric(f"{prefix}.bits_per_token", bits / n_tok)
    metric(f"{prefix}.perplexity_per_token", 2.0 ** (bits / n_tok), label="derived", note="2^(bits per token), subword")
    metric(f"{prefix}.perplexity_per_word", 2.0 ** (bits / n_words), label="derived",
           note=f"converted: 2^(total bits / {n_words} words), open-vocabulary, not like-for-like with published closed-vocabulary numbers")
    metric(f"{prefix}.causal_top1_raw_scores", sc["top1_raw"], note="argmax of the class sums, ties toward the lower id")
    metric(f"{prefix}.causal_top1_calibrated", sc["top1_calibrated"], note="argmax of a * score + b * class_bias / 256")
    metric(f"{prefix}.rows_sum_to_one_worst_abs_log", sc["worst_row_normalisation_abs_log"], note="max |log sum exp(logprobs)| over all rows; asserted < 1e-6")


def save_model(path: Path, engine, class_bias: np.ndarray, cal: Calibration) -> None:
    """TrainedModel.save layout (src/tmlm/decode/scorer.py), compressed; TrainedModel.load reads it."""
    st = engine.state()
    c = engine.corpus
    np.savez_compressed(path, cfg=np.array(repr(engine.cfg.to_dict())), class_bias=np.ascontiguousarray(class_bias, dtype=np.int32),
                        a_q16=np.array(cal.a_q16), b_q16=np.array(cal.b_q16), kind_map=np.ascontiguousarray(c.kind_map),
                        stop=np.ascontiguousarray(c.stop), sym_freq=np.ascontiguousarray(c.sym_freq),
                        **{f"state_{k}": v for k, v in st.items()})


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "CP_anchor.json"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=4, help="docs/13 amendment A2")
    ap.add_argument("--resume", default=None, help="checkpoint .npz to continue from")
    ap.add_argument("--probe-batches", type=int, default=0, help="measure only: build, N timed batches, predict speed, calibration fit time; no epochs, no test")
    ap.add_argument("--calib-rows", type=int, default=4096)
    ap.add_argument("--calib-rounds", type=int, default=8)
    ap.add_argument("--eval-chunk", type=int, default=4096)
    ap.add_argument("--checkpoint-every", type=int, default=20000, help="batches between uncompressed scratch checkpoints (0 = epoch ends only)")
    ap.add_argument("--scratch-dir", default=os.environ.get("TMLM_SCRATCH", "/tmp"))
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--yield-every", type=int, default=500)
    ap.add_argument("--no-test", action="store_true", help="stop after the validation curve (no counted test access)")
    ap.add_argument("--kn-order", type=int, default=5)
    args = ap.parse_args()

    cfg_json = load_json(args.config)
    base = dict(cfg_json["engine"])
    vocab = int(base["vocab"])
    cfg = EngineConfig(**{**base, "seed": args.seed})
    if cfg.mode != "causal":
        raise ValueError("protocol CP needs causal mode")
    metric("seed", args.seed)
    metric("engine_config", cfg.to_dict())
    metric("epochs_planned", args.epochs, note="docs/13 amendment A2")
    metric("test_accesses_before", test_access_count())

    # data
    t0 = time.time()
    train = contiguous(load_split(CORPUS, vocab, "train"))
    val = contiguous(load_split(CORPUS, vocab, "validation"))
    train_meta, _ = split_meta(vocab, "train")
    val_meta, val_raw = split_meta(vocab, "validation")
    for split, meta in (("train", train_meta), ("validation", val_meta)):
        dataset_hash(f"{split}_tok_V{vocab}", meta["sha256"]["tok.npy"])
    dataset_hash("validation_text_utf8", val_raw["sha256"]["text.utf8"])
    metric("data_load_s", time.time() - t0)
    metric("n_train_tokens", int(train.tok.shape[0]))
    metric("n_validation_tokens", int(val.tok.shape[0]))
    metric("n_validation_bytes", int(val_meta["n_bytes_utf8"]))
    metric("n_validation_words", int(val_meta["n_words"]))
    counts = np.bincount(train.tok.astype(np.int64), minlength=vocab)
    class_bias = class_bias_from_counts(counts, vocab)
    metric("class_bias_note", "round(256 ln((n_c + 1) / (N + V))) from TRAIN token counts, as TrainedModel.from_engine (src/tmlm/decode/scorer.py)", label="derived")

    # engine
    import cupy as cp
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    pool = cp.get_default_memory_pool()
    power = NvmlPowerSampler()
    t0 = time.time()
    engine = CudaEngine(cfg, train, profile=bool(args.probe_batches))
    engine.synchronize()
    metric("engine_build_s", time.time() - t0, note="corpus upload incl. the occurrence lists of 4 kinds over the train split, state allocation")
    metric("device_pool_bytes_after_build", int(pool.used_bytes()), unit="bytes")
    metric("host_maxrss_gib_after_build", rss_gib(), unit="GiB")
    n_batches = engine.n_batches()
    metric("batches_per_epoch", n_batches)
    if power.available:
        power.start()
        time.sleep(10.0)
        idle = power.summary()
        metric("nvml_idle_w", idle["mean_w"], note="device-wide power over 10 s with the CUDA context created and this process idle; other tenants' jobs on the shared GPU are included")
        idle_w = float(idle["mean_w"])
    else:
        metric("nvml_idle_w", None, label="unverified", note=f"NVML unavailable: {power.error}")
        idle_w = 0.0
    probe_engine_val = None

    def val_engine():
        nonlocal probe_engine_val
        if probe_engine_val is None:
            probe_engine_val = CudaEngine(cfg, val)
        probe_engine_val.load_state(engine.state())
        return probe_engine_val

    # ---------------------------------------------------------- probe mode
    if args.probe_batches:
        nb = args.probe_batches
        for b in range(5):
            engine.train_batch(0, b)
        engine.synchronize()
        engine.reset_profile()
        diag: dict[str, int] = {}
        t_since = time.time()
        t0 = time.time()
        for b in range(5, 5 + nb):
            d = engine.train_batch(0, b)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
        engine.synchronize()
        wall = time.time() - t0
        n_ex = diag["n_examples"]
        metric("probe.batches", nb)
        metric("probe.examples_per_s", n_ex / wall, unit="examples/s")
        metric("probe.ms_per_batch", 1000 * wall / nb)
        metric("probe.epoch_estimate_s", n_batches * wall / nb, label="derived", note="batches per epoch x measured ms per batch on a young state (candidates per example grow with training)")
        metric("probe.candidates_per_example", diag["n_candidates"] / n_ex)
        metric("probe.fired_per_example", diag["n_fired"] / n_ex)
        metric("probe.events_per_example", diag["n_events_applied"] / n_ex)
        metric("probe.phase_ms_per_batch", {k: v / nb for k, v in engine.profile_ms.items()})
        metric("probe.device_pool_bytes_after_training", int(pool.used_bytes()), unit="bytes")
        metric("probe.host_maxrss_gib", rss_gib(), unit="GiB")
        if power.available:
            s = power.summary(since=t_since)
            metric("probe.nvml_joules_device_wide", s["joules"], note="shared GPU, other jobs included")
        pe = val_engine()
        pos = np.arange(0, min(val.tok.shape[0], 65536), dtype=np.uint32)
        t0 = time.time()
        sc = pe.predict_scores(0, pos)
        pe.synchronize()
        metric("probe.predict_positions_per_s", pos.shape[0] / (time.time() - t0), unit="positions/s")
        t0 = time.time()
        cal, rep = fit_calibration(pe, val.tok, class_bias, args.calib_rows, args.calib_rounds)
        metric("probe.calibration_fit_s", time.time() - t0, note=f"{rep['rows']} rows, {rep['rounds']} rounds")
        metric("probe.calibration", {**cal.to_dict(), **{k: v for k, v in rep.items()}})
        t0 = time.time()
        sc = score_stream(pe, val.tok, class_bias, cal, args.eval_chunk)
        metric("probe.validation_score_stream_s", time.time() - t0)
        report_split("probe.validation", sc, val_meta, "after 5 + probe batches only, NOT a result")
        power.stop()
        log("probe done")
        return 0

    # ---------------------------------------------------------- resume
    start_epoch, start_batch = 0, 0
    if args.resume:
        ccfg, state, meta = load_checkpoint(args.resume)
        if ccfg != cfg:
            raise ValueError("checkpoint config differs from the run config")
        engine.load_state(state)
        start_epoch, start_batch = int(meta["position"]["epoch"]), int(meta["position"]["batch_index"])
        metric("resumed_from", {"path": str(args.resume), "state_hash": meta["state_hash"], "epoch": start_epoch, "batch_index": start_batch, "step": meta["step"]})
        log(f"resumed at epoch {start_epoch} batch {start_batch} (step {meta['step']})")

    scratch_ckpt = Path(args.scratch_dir) / f"cp_anchor_seed{args.seed}_scratch_checkpoint.npz"
    final_cal = None
    for e in range(start_epoch, args.epochs):
        b0 = start_batch if e == start_epoch else 0
        diag = {}
        t_epoch = time.time()
        for b in range(b0, n_batches):
            d = engine.train_batch(e, b)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
            done = b + 1
            if done % args.log_every == 0 or done == n_batches:
                engine.synchronize()
                el = time.time() - t_epoch
                n_ex = diag["n_examples"]
                log(f"epoch {e + 1} batch {done}/{n_batches}: {n_ex / el:.0f} ex/s cand/ex {diag['n_candidates'] / n_ex:.1f} fired/ex {diag['n_fired'] / n_ex:.2f} "
                    f"events/ex {diag['n_events_applied'] / n_ex:.2f} seeded {diag['n_seeded']} recycled {diag['n_recycled']} pool {pool.used_bytes() / 1e9:.2f} GB rss {rss_gib():.1f} GiB")
            if args.checkpoint_every and done % args.checkpoint_every == 0 and done < n_batches:
                engine.synchronize()
                save_checkpoint(scratch_ckpt, engine, e, done)
            if args.yield_every and done % args.yield_every == 0:
                engine.synchronize()
                if yield_gpu_if_requested():
                    log("yielded the GPU to an exclusive job and took the shared lock back")
        engine.synchronize()
        wall = time.time() - t_epoch
        n_ex = diag["n_examples"]
        p = f"epoch{e + 1}"
        partial = " (resumed mid-epoch: counters cover the batches run in this process only)" if b0 else ""
        metric(f"{p}.train_wall_s", wall, note="training only" + partial)
        metric(f"{p}.examples_per_s", n_ex / wall, unit="examples/s", note=partial.strip() or "whole epoch")
        metric(f"{p}.n_examples", n_ex)
        for name in ("n_candidates", "n_fired", "n_events_applied", "n_events_dropped_cap", "n_tgt_gate_pass", "n_neg_gate_pass", "n_inserted", "n_neg_inserted", "n_vetoes"):
            metric(f"{p}.{name}_per_example", diag[name] / n_ex)
        metric(f"{p}.n_seeded", diag["n_seeded"])
        metric(f"{p}.n_recycled", diag["n_recycled"])
        metric(f"{p}.alive_clauses", int(engine.state()["alive"].sum()))
        metric(f"{p}.device_pool_bytes", int(pool.used_bytes()), unit="bytes")
        metric(f"{p}.host_maxrss_gib", rss_gib(), unit="GiB")
        if power.available:
            s = power.summary(since=t_epoch)
            metric(f"{p}.nvml_joules_device_wide", s["joules"], note="whole epoch window incl. checkpoint and nothing else; shared GPU, other tenants' jobs included")
            metric(f"{p}.nvml_joules_minus_idle", (s["joules"] - idle_w * s["span_s"]) if s["joules"] is not None else None, label="derived", note="joules minus idle_w x span; shared GPU")
            metric(f"{p}.nvml_joules_per_example_minus_idle", ((s["joules"] - idle_w * s["span_s"]) / n_ex) if s["joules"] is not None else None, label="derived")
        # checkpoint (compressed, in the run dir)
        t0 = time.time()
        ck = out_dir() / f"checkpoint_epoch{e + 1}.npz"
        h = save_checkpoint(ck, engine, e + 1, 0, compress=True)
        metric(f"{p}.checkpoint", {"path": str(ck), "state_hash": h, "bytes": ck.stat().st_size, "write_s": time.time() - t0})
        # validation curve
        pe = val_engine()
        t0 = time.time()
        cal, rep = fit_calibration(pe, val.tok, class_bias, args.calib_rows, args.calib_rounds)
        metric(f"{p}.calibration", {**cal.to_dict(), **rep, "fit_s": time.time() - t0}, note="two scalars fitted on the validation NLL of the row subsample; used for every validation row")
        t0 = time.time()
        sc = score_stream(pe, val.tok, class_bias, cal, args.eval_chunk)
        report_split(f"{p}.validation", sc, val_meta, f"validation, epoch {e + 1}, calibration of this epoch")
        metric(f"{p}.validation_eval_s", time.time() - t0)
        final_cal = cal
        log(f"epoch {e + 1} done: val bpb {METRICS[f'{p}.validation.bits_per_byte']['value']:.4f}, top1 {sc['top1_calibrated']:.4f}, wall {wall:.0f} s")
    if scratch_ckpt.exists():
        scratch_ckpt.unlink()
    if final_cal is None:
        log("nothing trained (resume position at or past the epoch budget)")
        return 0
    # final model file
    t0 = time.time()
    mp = out_dir() / f"model_seed{args.seed}.npz"
    save_model(mp, engine, class_bias, final_cal)
    metric("model_file", {"path": str(mp), "bytes": mp.stat().st_size, "state_hash": engine.state_hash(), "write_s": time.time() - t0},
           note="TrainedModel.save layout, compressed")
    if power.available:
        power.stop()
    if args.no_test:
        return 0

    # ---------------------------------------------------------- TEST, once, with the frozen calibration
    val_bpb = METRICS[f"epoch{args.epochs}.validation.bits_per_byte"]["value"]
    metric("test_access_purpose", f"CP anchor seed {args.seed}: final test bits per byte after {args.epochs} epochs, calibration frozen on validation")
    test = contiguous(load_split(CORPUS, vocab, "test", purpose=METRICS["test_access_purpose"]["value"]))
    test_meta, test_raw = split_meta(vocab, "test")
    dataset_hash(f"test_tok_V{vocab}", test_meta["sha256"]["tok.npy"])
    dataset_hash("test_text_utf8", test_raw["sha256"]["text.utf8"])
    metric("test_accesses_after", test_access_count())
    metric("n_test_tokens", int(test.tok.shape[0]))
    metric("n_test_bytes", int(test_meta["n_bytes_utf8"]))
    metric("n_test_words", int(test_meta["n_words"]))
    te = CudaEngine(cfg, test)
    te.load_state(engine.state())
    t0 = time.time()
    sc = score_stream(te, test.tok, class_bias, final_cal, args.eval_chunk)
    report_split("test", sc, test_meta, "TEST, once, calibration frozen on validation after the last epoch")
    metric("test_eval_s", time.time() - t0)
    metric("test_8gram_overlap_with_train", 0.114, note="docs/12 F16, results/20260922T160257Z_M2_data/: fraction of test 8-grams seen in train; 13-gram 0.070")
    del te
    cp.get_default_memory_pool().free_all_blocks()
    # KN-5 on identical tokens
    from tmlm.eval.baselines.ngram import KNModel, bits_per_byte
    t0 = time.time()
    kn = KNModel(args.kn_order, vocab, discount_mode="modified").fit(train.tok, train.doc_start)
    metric("kn5.fit_wall_s", time.time() - t0, note="forward model only, full train split, modified KN")
    lp_val = kn.logprob(val.tok, val.doc_start)
    metric("kn5.validation.bits_per_byte", bits_per_byte(lp_val, int(val_meta["n_bytes_utf8"])), note="check against results/20260922T180823Z_M2_baselines_ngram_v8000/ (1.3501)")
    lp_test = kn.logprob(test.tok, test.doc_start)
    kn_test_bpb = bits_per_byte(lp_test, int(test_meta["n_bytes_utf8"]))
    metric("kn5.test.bits_per_byte", kn_test_bpb, note="modified KN-5, in-document, same test tokens and bytes")
    metric("kn5.test.perplexity_per_word", 2.0 ** (-lp_test.sum() / int(test_meta["n_words"])), label="derived", note="converted, open-vocabulary")
    pred = kn.predict(test.tok, test.doc_start, np.arange(test.tok.shape[0]))
    metric("kn5.test.causal_top1", float((pred == test.tok.astype(np.int64)).mean()))
    tm_bpb = METRICS["test.bits_per_byte"]["value"]
    metric("H1_1_seed_tm_below_kn5", bool(tm_bpb < kn_test_bpb), note=f"TM test bpb {tm_bpb:.4f} vs KN-5 {kn_test_bpb:.4f}; H1.1 needs every seed")
    metric("KB7_tm_above_1p5x_kn5", bool(tm_bpb > 1.5 * kn_test_bpb), note="docs/08 KB7: sparse-only bpb > 1.5 x KN-5")
    metric("tm_over_kn5_ratio_test", tm_bpb / kn_test_bpb, label="derived")
    metric("validation_bpb_final", val_bpb)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
