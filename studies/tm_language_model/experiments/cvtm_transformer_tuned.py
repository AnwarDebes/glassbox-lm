"""Transformer comparators of lane GX (docs/13 amendment A3 H8.5; amendment A4 H9.7 and gate G1 frame).
Runs in .venv-dl (torch). Every reportable run holds the EXCLUSIVE GPU lock (energy), one job at a time.

Modes:

- "m2": the in-house L6 W256 transformer with EXACTLY the settings of
  `results/20260922T172902Z_M2_baselines_transformer/` (config `transformer_m2_settings` of
  `experiments/cvtm/configs/H8_5_energy.json`: fp32, seq_len 256, 64 windows per step, lr 1e-3, warmup 200,
  4 heads, seed 1, the full train split), trained by the unchanged `transformer_small.train`; only the
  wall-clock cap changes. The cap is `--time-cap-s S`, or `--cap-from <fit_energy run>` (H8.5: cap =
  J_cvtm / 259.2 W), or `--cap-from ... --rerun-of <matched run>` (the one rerun of H8.5: cap = J_cvtm / P
  with P the measured idle-subtracted mean power of the first matched run; refused unless that run missed
  J_cvtm by more than 10 percent).
- "tuned_lrsel": H9.7 pilot stage 1, every family of `experiments/cvtm/configs/H9_7_tuned_pilot.json`
  (L6 W384, L8 W512; fp16 autocast, context 1024) at every learning rate of the grid, 150 s cap, seed 100.
- "tuned_cap": H9.7 pilot stage 2, every family at the learning rate that the rule of the config selects
  from the stage-1 run (`--select-from`), at `--time-cap-s` (600 for the pilot).

Shared by every mode:
- Model, packing, evaluation and the NVML sampler are those of `tmlm.eval.baselines.transformer_small`
  (imported, not copied); the train and validation data and their byte count come from
  `experiments/m2_baselines.py` (`load_train`, `EvalData`), so the tokens and bytes are those of M2.
- Energy: `transformer_small.NvmlPowerSampler` (10 Hz, device-wide board power) over the training loop
  only; idle = 10 s with the CUDA context created and nothing running, once at the start of the process;
  the lane's settle rule (`experiments/cvtm_gx_settle.py`, config key "energy_protocol") waits settle_s
  seconds with nothing running before that idle measurement and again right before every measured training
  (after the warm-up step and the model build), so no idle level and no measured loop starts in the tail of
  earlier GPU activity;
  joules raw and idle-subtracted (joules - idle_w x span). A 10 s idle sample after every training is
  logged as a diagnostic and not used. Host CPU seconds of the training loop (process CPU time, all
  threads) are logged next to the joules.
- Warm-up (outside every measured interval): one forward and backward step of a throwaway model of the
  same shape and precision, so that CUDA library initialisation is not charged to the measured loop
  (the CVTM side reports the median of warm repeats). `seed_everything(seed)` runs AFTER it, right before
  the real model is built, so the real model's initialisation and data order are unaffected.
- Evaluation: `transformer_small.bits_per_byte` (protocol CP) is the reported number. The per-token
  log2 P of every validation token is saved (`<name>.validation_bits.npy`, float64, stream order) for
  the paired document bootstrap of the verdict step; it is checked to reproduce the reported total
  (same float32 batch sums, exactly; float64 re-sum within 1e-6 relative) and to score every
  validation token exactly once.
- Checkpoints `<name>.pt` (state dict and config) are saved in the run directory.
- The test split is never loaded.

Launch lines: `experiments/cvtm/configs/H8_5_energy.json` and `H9_7_tuned_pilot.json` ("launch").

Smoke (not logged, scratch only): `--smoke` (train prefix of 2M tokens, caps of a few seconds, output
under $TMLM_SCRATCH/cvtm_gx_smoke/<mode>), under `scripts/gpu_run.sh shared`.

Later (NOT run now; the H9.7 frontier), each under the exclusive lock, one at a time, after the
pilot: the caps 2400 s and 9600 s for the tuned families at their selected learning rates, e.g.

    scripts/gpu_run.sh exclusive -- scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name cvtm_H9_7_tuned_2400s --config experiments/cvtm/configs/H9_7_tuned_pilot.json -- \\
        .venv-dl/bin/python -u experiments/cvtm_transformer_tuned.py --mode tuned_cap --time-cap-s 2400 \\
        --config experiments/cvtm/configs/H9_7_tuned_pilot.json --select-from results/<lrsel run>
    (and the same with --time-cap-s 9600 and --name cvtm_H9_7_tuned_9600s)

plus, for the registered family {L6 W256, L6 W384, L8 W512, L12 W768} and caps {150, 600, 2400, 9600} with
confirmatory seeds, a config that adds the L6 W256 and L12 W768 families (heads 4 and 12) and `--seed`;
and the CVTM-core fit joules at the H8.8 prefixes for E*(D):
`experiments/cvtm_energy.py --mode fit_energy --prefix-tokens N` under the exclusive lock (see its docstring).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.cvtm import protocol  # noqa: E402
from tmlm.eval.baselines import transformer_small as ts  # noqa: E402
from cvtm_gx_settle import settle as _settle, settle_protocol  # noqa: E402

LN2 = math.log(2.0)
T0 = time.time()
METRICS: dict = {}
FAILED: list[str] = []
SMOKE = False
REGISTERED_POWER_W = 259.2          # W, docs/13 A3 H8.5 (155,519.8 J / 600 s, derived there)
RERUN_TOLERANCE = 0.10


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[cvtm_tf {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


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


def out_dir(mode: str) -> Path:
    d = runlog.current_run_dir()
    if d is not None and not SMOKE:
        return Path(d)
    p = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "cvtm_gx_smoke" / mode
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_smoke_metrics(d: Path) -> None:
    """Smoke only: a metrics.json in the scratch directory, so the verdict step can be smoke-tested."""
    if SMOKE:
        (d / "metrics.json").write_text(json.dumps(METRICS, indent=1, default=float))
        (d / "run.json").write_text(json.dumps({"exit_status": 0 if not FAILED else 1, "smoke": True}))


# ------------------------------------------------------------------ pure rules (tested)
def matched_cap_s(j_cvtm: float, power_w: float = REGISTERED_POWER_W) -> float:
    """H8.5: the wall-clock cap whose joules match the CVTM-core fit at the given idle-subtracted power."""
    if not (j_cvtm > 0 and power_w > 0):
        raise ValueError("joules and power must be positive")
    return float(j_cvtm) / float(power_w)


def needs_rerun(j_transformer: float, j_cvtm: float, tol: float = RERUN_TOLERANCE) -> bool:
    """H8.5: rerun once iff the transformer's joules differ from the CVTM's by more than tol (relative)."""
    return abs(float(j_transformer) - float(j_cvtm)) / float(j_cvtm) > tol


def select_lr(rows: dict[float, float]) -> float:
    """H9.7 pilot rule: the learning rate with the lowest validation bits per byte; ties to the smaller one."""
    if not rows:
        raise ValueError("no rows")
    return min(rows, key=lambda lr: (rows[lr], lr))


def read_metrics(run: Path) -> dict:
    return json.loads((Path(run) / "metrics.json").read_text())


def mval(metrics: dict, name: str):
    if name not in metrics:
        raise KeyError(f"metric {name} not found")
    return metrics[name]["value"]


def require_finished(run: Path) -> dict:
    """run.json of a finished logged run with exit status 0 (smoke dirs carry a minimal run.json)."""
    rj = json.loads((Path(run) / "run.json").read_text())
    if rj.get("exit_status") != 0:
        raise SystemExit(f"{run}: exit status {rj.get('exit_status')}, cannot be used")
    return rj


# ------------------------------------------------------------------ evaluation (per token)
def per_token_bits(model, packed: np.ndarray, doc: np.ndarray, *, seq_len: int, batch_seqs: int, device):
    """The scoring of `transformer_small.bits_per_byte`, per token: (bits float64 [n real tokens] in stream
    order, total nats summed as bits_per_byte sums them (float32 batch sums), n_scored, n_double)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    half = seq_len // 2
    n = packed.shape[0]
    starts = list(range(0, max(1, n - half), half))
    nats = np.full(n, np.nan, dtype=np.float64)
    total_nats, n_scored, n_double = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(starts), batch_seqs):
            s = np.array(starts[i:i + batch_seqs])
            idx = s[:, None] + np.arange(seq_len + 1)[None]
            valid = idx < n
            idx = np.minimum(idx, n - 1)
            x = torch.from_numpy(packed[idx]).to(device)
            d = torch.from_numpy(doc[idx]).to(device)
            logits = model(x[:, :-1], d[:, :-1])
            lp = F.log_softmax(logits.float(), dim=-1).gather(-1, torch.clamp(x[:, 1:], max=model.vocab - 1)[..., None])[..., 0]
            offs = np.arange(1, seq_len + 1)[None]
            score = (x[:, 1:] < model.vocab).cpu().numpy() & valid[:, 1:]
            second = (s[:, None] > 0) & (offs <= half)
            score &= ~second
            m = torch.from_numpy(score).to(device)
            total_nats -= float(lp[m].sum())
            n_scored += int(m.sum())
            pp = idx[:, 1:][score]
            n_double += int(np.count_nonzero(~np.isnan(nats[pp])))
            nats[pp] = -lp.double().cpu().numpy()[score]
    real = packed < model.vocab
    bits = nats[real] / LN2
    return bits, total_nats, n_scored, n_double


# ------------------------------------------------------------------ training
def train_amp(model, packed: np.ndarray, doc: np.ndarray, *, seq_len: int, batch_seqs: int, lr: float, warmup: int,
              time_cap_s: float, seed: int, device, log=print, amp: bool = True) -> dict:
    """`transformer_small.train` (causal objective) with fp16 autocast and a GradScaler: same window
    sampling (numpy default_rng(seed), shuffled epochs of windows at multiples of seq_len), AdamW
    (0.9, 0.95, wd 0.01), `lr_at` schedule, clipping at 1.0 after unscaling. The loss is the mean over the
    real targets (ignore_index for BOS targets, the same value as indexing the kept positions)."""
    import torch
    import torch.nn.functional as F

    use_amp = bool(amp) and device.type == "cuda"
    rng = np.random.default_rng(seed)
    n_win = packed.shape[0] // seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    t0 = time.time()
    step, epoch = 0, 0
    tokens = torch.zeros((), dtype=torch.int64, device=device)
    losses: list[tuple[int, float, float]] = []
    order = rng.permutation(n_win)
    cursor = 0
    while True:
        if cursor + batch_seqs > order.shape[0]:
            epoch += 1
            order = rng.permutation(n_win)
            cursor = 0
        starts = order[cursor:cursor + batch_seqs] * seq_len
        cursor += batch_seqs
        idx = starts[:, None] + np.arange(seq_len)[None]
        x = torch.from_numpy(packed[idx]).to(device)
        d = torch.from_numpy(doc[idx]).to(device)
        target = x[:, 1:].masked_fill(x[:, 1:] >= model.vocab, -100)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x[:, :-1], d[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
        elapsed = time.time() - t0
        cur_lr = ts.lr_at(step, elapsed, time_cap_s, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens += (target >= 0).sum()
        if step % 100 == 0 or step == 1:
            lv = loss.item()
            sc = float(scaler.get_scale()) if use_amp else 1.0
            losses.append((step, lv, sc))
            log(f"step {step} epoch {epoch} loss {lv:.4f} lr {cur_lr:.2e} scale {sc:.0f} "
                f"{int(tokens.item()) / max(1e-9, time.time() - t0):.0f} tok/s")
        if time.time() - t0 >= time_cap_s:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    return {"steps": step, "tokens_predicted": int(tokens.item()), "wall_s": wall, "epochs_completed": epoch,
            "windows_per_epoch": int(n_win), "final_loss": loss.item(), "loss_curve": losses,
            "final_grad_scale": float(scaler.get_scale()) if use_amp else None}


def warm_up(vocab: int, layers: int, width: int, heads: int, seq_len: int, batch_seqs: int, device, amp: bool) -> float:
    """One forward and backward step of a throwaway model of the same shape (outside every measured interval)."""
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    m = ts.SmallTransformer(vocab, layers, width, heads, seq_len).to(device)
    x = torch.randint(0, vocab, (batch_seqs, seq_len), device=device)
    d = torch.zeros((batch_seqs, seq_len), dtype=torch.long, device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=bool(amp) and device.type == "cuda"):
        logits = m(x[:, :-1], d[:, :-1])
    F.cross_entropy(logits.float().reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    del m, x, d, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return time.time() - t0


# ------------------------------------------------------------------ one measured training
class Ctx:
    """Data, device, sampler and idle power shared by the trainings of one process."""

    def __init__(self, vocab: int, train_tokens: int | None, settle_s: float) -> None:
        import m2_baselines as m2
        import torch

        self.torch = torch
        self.vocab = vocab
        self.settle_s = float(settle_s)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        protocol.write_dataset_hashes(("train", "validation"), protocol.CORPUS, vocab)
        self.ev = m2.EvalData(vocab)
        tok, ds = m2.load_train(vocab, train_tokens)
        self.packed, self.doc = ts.pack_stream(tok, ds, vocab)
        self.vpacked, self.vdoc = ts.pack_stream(self.ev.tok, self.ev.doc_start, vocab)
        self.n_val_tokens = int(self.ev.tok.shape[0])
        M("device", str(self.device))
        M("gpu_lock_mode", os.environ.get("TMLM_GPU_LOCK_MODE"), note="energy is reportable only under the exclusive lock")
        if self.device.type == "cuda":
            torch.zeros(1, device=self.device)
            torch.cuda.synchronize()
            M("gpu_name", torch.cuda.get_device_name(0))
        self.sampler = ts.NvmlPowerSampler()
        M("energy_protocol.settle_s", self.settle_s, label="derived",
          note="wait with nothing running before the idle measurement and before every measured training")
        M("settle.start", self.settle("start"))
        if self.sampler.available:
            self.idle = self.sampler.measure_idle(10.0)
        else:
            self.idle = {"available": False, "error": getattr(self.sampler, "error", None)}
        self.idle_w = self.idle.get("mean_w")
        M("nvml_idle", self.idle, note="10 s with the CUDA context created and nothing running (transformer_small method)")
        self.nvml = None
        try:
            import pynvml

            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            pass

    def settle(self, tag: str) -> dict:
        return _settle(self.settle_s, tag, ts.NvmlPowerSampler, log)

    def temperature_c(self) -> int | None:
        if self.nvml is None:
            return None
        try:
            return int(self.nvml.nvmlDeviceGetTemperature(self.handle, 0))
        except Exception:  # noqa: BLE001
            return None


def train_and_measure(ctx: Ctx, name: str, *, layers: int, width: int, heads: int, seq_len: int, batch_seqs: int,
                      lr: float, warmup: int, time_cap_s: float, seed: int, amp: bool, eval_batch: int,
                      run_dir: Path) -> dict:
    torch = ctx.torch
    dev = ctx.device
    wu = warm_up(ctx.vocab, layers, width, heads, seq_len, batch_seqs, dev, amp)
    det = ts.seed_everything(seed)
    model = ts.SmallTransformer(ctx.vocab, layers, width, heads, seq_len).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    M(f"{name}.settle_before_training", ctx.settle(f"before {name}"))
    temp0 = ctx.temperature_c()
    c0 = time.process_time()
    ctx.sampler.start()
    tlog = lambda m, n=name: print(f"[{n}] {m}", flush=True)  # noqa: E731
    if amp:
        stats = train_amp(model, ctx.packed, ctx.doc, seq_len=seq_len, batch_seqs=batch_seqs, lr=lr, warmup=warmup,
                          time_cap_s=time_cap_s, seed=seed, device=dev, log=tlog, amp=True)
    else:
        stats = ts.train(model, ctx.packed, ctx.doc, seq_len=seq_len, batch_seqs=batch_seqs, lr=lr, warmup=warmup,
                         time_cap_s=time_cap_s, seed=seed, device=dev, log=tlog)
    energy = ctx.sampler.stop()
    cpu_s = time.process_time() - c0
    temp1 = ctx.temperature_c()
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
    t0 = time.time()
    bpb = ts.bits_per_byte(model, ctx.vpacked, ctx.vdoc, ctx.ev.n_bytes, seq_len=seq_len, batch_seqs=eval_batch, device=dev)
    eval_s = time.time() - t0
    bits, tot_nats, n_scored, n_double = per_token_bits(model, ctx.vpacked, ctx.vdoc, seq_len=seq_len, batch_seqs=eval_batch,
                                                        device=dev)
    ref_nats = bpb["bits_per_byte"] * LN2 * ctx.ev.n_bytes
    check(f"{name}.per_token_scores_every_token_once", n_scored == ctx.n_val_tokens and n_double == 0
          and bool(np.all(np.isfinite(bits))) and bits.shape[0] == ctx.n_val_tokens,
          n_scored=n_scored, n_double=n_double, n_tokens=ctx.n_val_tokens)
    check(f"{name}.per_token_total_equals_bits_per_byte", tot_nats / LN2 / ctx.ev.n_bytes == bpb["bits_per_byte"],
          per_token_path=tot_nats / LN2 / ctx.ev.n_bytes, reported=bpb["bits_per_byte"])
    rel = abs(float(bits.sum()) * LN2 - ref_nats) / ref_nats
    check(f"{name}.per_token_float64_resum_within_1e-6", rel <= 1e-6, rel_diff=rel)
    np.save(run_dir / f"{name}.validation_bits.npy", bits.astype(np.float64))
    joules = energy.get("joules")
    j_is = (joules - ctx.idle_w * energy["span_s"]) if (joules is not None and ctx.idle_w is not None) else None
    post_idle = ctx.sampler.measure_idle(10.0) if ctx.sampler.available else None
    lock = os.environ.get("TMLM_GPU_LOCK_MODE")
    enote = ("exclusive GPU lock" if lock == "exclusive" else f"lock {lock!r}: NOT reportable, indicative only")
    r = {"layers": layers, "width": width, "heads": heads, "seq_len": seq_len, "batch_seqs": batch_seqs, "lr": lr,
         "warmup_steps": warmup, "seed": seed, "amp_fp16": bool(amp), "time_cap_s": float(time_cap_s),
         "n_params": model.n_params(), "steps": stats["steps"], "tokens_predicted": stats["tokens_predicted"],
         "epochs_completed": stats["epochs_completed"], "train_wall_s": stats["wall_s"], "final_loss": stats["final_loss"],
         "validation_bits_per_byte": bpb["bits_per_byte"], "validation_bits_per_token": bpb["bits_per_token"],
         "n_scored": bpb["n_scored"], "eval_wall_s": eval_s, "joules_raw": joules, "joules_idle_subtracted": j_is,
         "idle_w": ctx.idle_w, "mean_power_w": energy.get("mean_w"), "energy_span_s": energy.get("span_s"),
         "n_power_samples": energy.get("n_samples"),
         "mean_power_idle_subtracted_w": (j_is / energy["span_s"]) if (j_is is not None and energy.get("span_s")) else None,
         "host_cpu_s": cpu_s, "peak_gpu_mem_MiB": peak, "warm_up_s": wu, "gpu_temp_c_start_end": [temp0, temp1],
         "post_training_idle_w_diagnostic": None if post_idle is None else post_idle.get("mean_w"),
         "tokens_per_s": stats["tokens_predicted"] / max(1e-9, stats["wall_s"]),
         "final_grad_scale": stats.get("final_grad_scale"), "determinism": det}
    energy_keys = ("train_wall_s", "joules_raw", "joules_idle_subtracted", "mean_power_w", "energy_span_s",
                   "mean_power_idle_subtracted_w", "host_cpu_s", "tokens_per_s")
    for k, v in r.items():
        if k in ("post_training_idle_w_diagnostic",):
            M(f"{name}.{k}", v, note="10 s idle after the training; diagnostic, not used")
        elif k in energy_keys:
            M(f"{name}.{k}", v, note=enote)
        else:
            M(f"{name}.{k}", v)
    M(f"{name}.loss_curve", stats["loss_curve"])
    M(f"{name}.validation_bits_file", f"{name}.validation_bits.npy",
      note="per-token log2 loss of every validation token (float64, stream order)")
    ck = run_dir / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": {k: v for k, v in r.items() if k != "determinism"}}, ck)
    M(f"{name}.checkpoint_sha256", hashing.sha256_file(ck))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return r


# ------------------------------------------------------------------ modes
def mode_m2(cfg: dict, args, rd: Path) -> None:
    s = cfg["transformer_m2_settings"]
    if args.time_cap_s is not None:
        if args.cap_from or args.rerun_of:
            raise SystemExit("--time-cap-s excludes --cap-from and --rerun-of")
        cap = float(args.time_cap_s)
        M("time_cap.rule", "given on the command line")
    else:
        if not args.cap_from:
            raise SystemExit("mode m2 needs --time-cap-s or --cap-from")
        require_finished(Path(args.cap_from))
        fm = read_metrics(Path(args.cap_from))
        if not SMOKE:
            if mval(fm, "checks_failed"):
                raise SystemExit(f"{args.cap_from}: failed checks {mval(fm, 'checks_failed')}")
            if mval(fm, "gpu_lock_mode") != "exclusive":
                raise SystemExit(f"{args.cap_from}: joules not measured under the exclusive lock")
        if "energy_protocol.settle_s" not in fm:
            raise SystemExit(f"{args.cap_from}: measured without the settle rule of the lane's energy protocol")
        j_cvtm = float(mval(fm, "cvtm_fit.gpu_steps.joules_idle_subtracted.median"))
        M("time_cap.j_cvtm", j_cvtm, label="measured", source=str(args.cap_from),
          note="median over the repeats of the idle-subtracted joules of the CVTM-core fit's GPU steps")
        if args.rerun_of:
            require_finished(Path(args.rerun_of))
            mm = read_metrics(Path(args.rerun_of))
            if "energy_protocol.settle_s" not in mm:
                raise SystemExit(f"{args.rerun_of}: measured without the settle rule of the lane's energy protocol")
            j1 = float(mval(mm, "transformer_L6_W256.joules_idle_subtracted"))
            span1 = float(mval(mm, "transformer_L6_W256.energy_span_s"))
            if not needs_rerun(j1, j_cvtm):
                raise SystemExit(f"no rerun: the first matched run's joules {j1:.1f} are within 10 percent of {j_cvtm:.1f}")
            power = j1 / span1
            cap = matched_cap_s(j_cvtm, power)
            M("time_cap.rule", "H8.5 rerun: J_cvtm / P, P = idle-subtracted joules / span of the first matched run",
              label="derived")
            M("time_cap.first_run", {"run": str(args.rerun_of), "joules_idle_subtracted": j1, "span_s": span1,
                                     "power_w": power, "rel_diff_vs_cvtm": (j1 - j_cvtm) / j_cvtm}, label="derived")
        else:
            cap = matched_cap_s(j_cvtm)
            M("time_cap.rule", "H8.5: J_cvtm / 259.2 W (155,519.8 J / 600 s of results/20260922T172902Z_M2_baselines_transformer/)",
              label="derived")
    M("time_cap_s", cap, label="derived")
    ctx = Ctx(int(s["vocab"]), args.smoke_train_tokens if SMOKE else s["train_tokens"],
              float(args.smoke_settle_s) if SMOKE else settle_protocol(cfg))
    name = f"transformer_L{s['layers']}_W{s['width']}"
    r = train_and_measure(ctx, name, layers=int(s["layers"]), width=int(s["width"]), heads=int(s["heads"]),
                          seq_len=int(s["seq_len"]), batch_seqs=int(s["batch_seqs"]), lr=float(s["lr"]),
                          warmup=int(s["warmup_steps"]), time_cap_s=cap, seed=int(s["seed"]), amp=False,
                          eval_batch=int(s["batch_seqs"]), run_dir=rd)
    if args.cap_from and r["joules_idle_subtracted"] is not None:
        jc = float(METRICS["time_cap.j_cvtm"]["value"])
        rel = (r["joules_idle_subtracted"] - jc) / jc
        M("matched.rel_diff_joules_vs_cvtm", rel, label="derived", note="(J_transformer - J_cvtm) / J_cvtm")
        M("matched.needs_rerun", needs_rerun(r["joules_idle_subtracted"], jc), label="derived",
          note="docs/13 A3 H8.5: rerun once if the joules differ by more than 10 percent")


def mode_tuned(cfg: dict, args, rd: Path, stage: str) -> None:
    fams = cfg["model"]["families"]
    tr = cfg["training"]
    seed = int(args.seed if args.seed is not None else tr["seed"])
    if not SMOKE and seed < 100:
        raise SystemExit("pilot seeds are 100 or more (docs/13 A4)")
    if stage == "lrsel":
        grid = [float(x) for x in cfg["lr_selection"]["grid"]]
        cap = float(cfg["lr_selection"]["cap_s"]) if not SMOKE else float(args.smoke_cap_s)
        plan = [(f, lr) for f in fams for lr in grid]
    else:
        if not args.select_from:
            raise SystemExit("tuned_cap needs --select-from <lrsel run>")
        require_finished(Path(args.select_from))
        sm = read_metrics(Path(args.select_from))
        if not SMOKE:
            if mval(sm, "checks_failed"):
                raise SystemExit(f"{args.select_from}: failed checks")
            rj = json.loads((Path(args.select_from) / "run.json").read_text())
            if rj.get("config_sha256") != hashing.config_hash(cfg):
                raise SystemExit("the selection run used another config")
        plan = []
        for f in fams:
            chosen = float(mval(sm, f"selection.{f['name']}.lr"))
            rows = {float(k): float(v) for k, v in mval(sm, f"selection.{f['name']}.rows").items()}
            if select_lr(rows) != chosen:
                raise SystemExit(f"{f['name']}: the logged selection does not follow the rule")
            plan.append((f, chosen))
        cap = float(args.time_cap_s) if args.time_cap_s is not None else float(cfg["cap_600"]["cap_s"])
        M("selection.source", str(args.select_from))
    M("time_cap_s", cap)
    M("seed", seed)
    ctx = Ctx(int(cfg["model"]["vocab"]), args.smoke_train_tokens if SMOKE else None,
              float(args.smoke_settle_s) if SMOKE else settle_protocol(cfg))
    ctxlen = int(cfg["model"]["context"])
    results: dict = {}
    for f, lr in plan:
        name = f"tuned_{f['name']}_lr{lr:g}_cap{int(cap)}"
        r = train_and_measure(ctx, name, layers=int(f["layers"]), width=int(f["width"]), heads=int(f["heads"]),
                              seq_len=ctxlen, batch_seqs=int(tr["batch_seqs"]), lr=lr, warmup=200, time_cap_s=cap,
                              seed=seed, amp=True, eval_batch=int(tr["batch_seqs"]), run_dir=rd)
        results.setdefault(f["name"], {})[lr] = r
    if stage == "lrsel":
        for fname, rows in results.items():
            table = {lr: r["validation_bits_per_byte"] for lr, r in rows.items()}
            chosen = select_lr(table)
            M(f"selection.{fname}.rows", {f"{lr:g}": v for lr, v in table.items()}, label="measured",
              note="validation bits per byte at the 150 s cap, per learning rate")
            M(f"selection.{fname}.lr", chosen, label="derived", note=cfg["lr_selection"]["rule"])
    else:
        best = min(((r["validation_bits_per_byte"], fname) for fname, rows in results.items() for r in rows.values()))
        M("tuned.best.validation_bits_per_byte", best[0], label="measured", family=best[1],
          note=f"lowest validation bits per byte among the tuned families at the {int(cap)} s cap (seed {seed}, exploratory)")


def main() -> int:
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("m2", "tuned_lrsel", "tuned_cap"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--time-cap-s", type=float, default=None)
    ap.add_argument("--cap-from", default=None, help="m2: the fit_energy run whose joules set the cap (H8.5)")
    ap.add_argument("--rerun-of", default=None, help="m2: the first matched run (the one H8.5 rerun)")
    ap.add_argument("--select-from", default=None, help="tuned_cap: the tuned_lrsel run")
    ap.add_argument("--seed", type=int, default=None, help="tuned modes: override the config seed (100 or more)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-train-tokens", type=int, default=2_000_000)
    ap.add_argument("--smoke-cap-s", type=float, default=4.0)
    ap.add_argument("--smoke-settle-s", type=float, default=3.0)
    args = ap.parse_args()
    SMOKE = bool(args.smoke)
    if SMOKE and os.environ.get("TMLM_RUN_DIR"):
        raise SystemExit("--smoke must not run under run_logged.py")
    if not SMOKE and os.environ.get("TMLM_GPU_LOCK_MODE") != "exclusive":
        raise SystemExit("reportable energy needs scripts/gpu_run.sh exclusive")
    cfg = json.loads(Path(args.config).read_text())
    rd = out_dir(args.mode)
    M("mode", args.mode)
    M("config_path", str(args.config))
    try:
        if args.mode == "m2":
            mode_m2(cfg, args, rd)
        elif args.mode == "tuned_lrsel":
            mode_tuned(cfg, args, rd, "lrsel")
        else:
            mode_tuned(cfg, args, rd, "cap")
    finally:
        M("wall_s", round(time.time() - T0, 1), unit="s")
        M("checks_failed", list(FAILED))
        dump_smoke_metrics(rd)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
