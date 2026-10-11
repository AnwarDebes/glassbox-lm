"""From-scratch small transformers on the same BPE tokens (docs/05 TB, docs/06 "Baselines").
Runs in .venv-dl (torch); nothing here is imported by the .venv-tm code.

Causal decoder-only model (`objective="causal"`): pre-LN blocks, learned positions, tied output
head, GELU MLP of width 4d. Document boundaries are respected: a BOS token (id V) starts every
document in the packed stream and the attention mask is causal AND same-document, so no token
attends across a boundary. Training samples random `seq_len` windows of the packed train stream
(each window starts at a multiple of seq_len; one epoch = every window once, shuffled), AdamW,
linear warmup then a linear decay driven by the elapsed fraction of the wall-clock cap
(lr = peak * min(step / warmup, 1) * (1 - 0.9 * elapsed / cap)), gradient clipping at 1.0,
fp32. The size sweep of TB trains every (layers, width) for the SAME wall-clock cap and picks the
lowest validation bits per byte.

Bits per byte (protocol CP): windows of seq_len with stride seq_len / 2 over the packed
validation stream, scoring every real token once (window 0 scores all its positions, later
windows their second half, so every token but the first seq_len / 2 of the stream sees at least
seq_len / 2 tokens of left context, cut at its document start); total nats / ln 2 over the UTF-8
byte count of the split. BOS is never scored.

Masked variant (`objective="masked"`, the 24 GPU-h comparator of docs/06): same blocks with a
same-document bidirectional mask; 15% of the real tokens of every window are replaced by MASK
(id V + 1) and predicted. Its evaluation is top-1 on the frozen eval sets with every hidden
position masked at once. In M2 it is only smoke-tested for two minutes.

Energy: an NVML sampler thread reads the GPU power (pynvml, 10 Hz) during training; joules are
the trapezoid integral, reported raw and with the idle power (sampled for 10 s before training,
context created) subtracted. Timing and energy under the SHARED lock are indicative only; TB's
reported numbers are re-measured under the exclusive lock (docs/06).

Determinism: torch and numpy are seeded, `torch.use_deterministic_algorithms(True, warn_only=True)`
and CUBLAS_WORKSPACE_CONFIG=:4096:8 are set before the CUDA context exists; the settings are
logged. The test split is scored only through `score_test`, which loads it with a purpose
(the access is counted, docs/06).
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

LN2 = math.log(2.0)


# ------------------------------------------------------------------ energy
class NvmlPowerSampler:
    """Samples GPU power (W) in a thread; `joules()` integrates. `available` is False without NVML."""

    def __init__(self, index: int = 0, period_s: float = 0.1) -> None:
        self.period_s = period_s
        self.samples: list[tuple[float, float]] = []
        self.available = False
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

    def _read(self) -> float:
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), self._read()))
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

    def summary(self) -> dict:
        if not self.available or len(self.samples) < 2:
            return {"available": self.available, "n_samples": len(self.samples), "joules": None, "mean_w": None, "span_s": 0.0}
        t = np.array([s[0] for s in self.samples])
        p = np.array([s[1] for s in self.samples])
        joules = float(np.trapezoid(p, t)) if hasattr(np, "trapezoid") else float(np.trapz(p, t))
        return {"available": True, "n_samples": int(t.shape[0]), "joules": joules, "mean_w": float(p.mean()),
                "span_s": float(t[-1] - t[0])}

    def measure_idle(self, seconds: float = 10.0) -> dict:
        self.start()
        time.sleep(seconds)
        return self.stop()


# ------------------------------------------------------------------ data
def pack_stream(tok: np.ndarray, doc_start: np.ndarray, bos: int) -> tuple[np.ndarray, np.ndarray]:
    """(packed int64 stream with BOS before every document, doc id per packed position)."""
    tok = np.asarray(tok).astype(np.int64)
    ds = np.asarray(doc_start).astype(np.int64)
    n_doc = ds.shape[0] - 1
    lengths = np.diff(ds)
    out = np.full(tok.shape[0] + n_doc, bos, dtype=np.int64)
    pos = np.arange(tok.shape[0]) + np.repeat(np.arange(1, n_doc + 1), lengths)
    out[pos] = tok
    doc = np.repeat(np.arange(n_doc), lengths + 1)
    return out, doc


def attention_mask(doc: torch.Tensor, causal: bool) -> torch.Tensor:
    """bool [B, 1, L, L]: True where attention is allowed (same document, and j <= i if causal)."""
    same = doc[:, :, None] == doc[:, None, :]
    if causal:
        L = doc.shape[1]
        same = same & torch.tril(torch.ones(L, L, dtype=torch.bool, device=doc.device))[None]
    return same[:, None]


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class SmallTransformer(nn.Module):
    def __init__(self, vocab: int, layers: int, width: int, heads: int, seq_len: int, objective: str = "causal") -> None:
        super().__init__()
        self.vocab, self.objective, self.seq_len = vocab, objective, seq_len
        self.n_symbols = vocab + 2                      # BOS = vocab, MASK = vocab + 1
        self.emb = nn.Embedding(self.n_symbols, width)
        self.pos = nn.Embedding(seq_len, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(width)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        """Logits [B, L, vocab] over the real tokens (BOS and MASK are never predicted)."""
        mask = attention_mask(doc, causal=(self.objective == "causal"))
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for b in self.blocks:
            h = b(h, mask)
        h = self.ln(h)
        return h @ self.emb.weight[: self.vocab].T


# ------------------------------------------------------------------ training
def lr_at(step: int, elapsed: float, cap: float, peak: float, warmup: int) -> float:
    return peak * min(1.0, (step + 1) / max(1, warmup)) * (1.0 - 0.9 * min(1.0, elapsed / cap))


def seed_everything(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    return {"torch_seed": seed, "numpy_seed": seed, "deterministic_algorithms": "True (warn_only)",
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "torch": torch.__version__}


def train(model: SmallTransformer, packed: np.ndarray, doc: np.ndarray, *, seq_len: int, batch_seqs: int, lr: float,
          warmup: int, time_cap_s: float, seed: int, device: torch.device, log=print, mask_rate: float = 0.15) -> dict:
    """Train under a wall-clock cap; returns statistics (steps, tokens, wall, loss curve)."""
    rng = np.random.default_rng(seed)
    n_win = packed.shape[0] // seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    t0 = time.time()
    step, tokens, epoch = 0, 0, 0
    losses: list[tuple[int, float]] = []
    order = rng.permutation(n_win)
    cursor = 0
    stop = False
    mask_id = model.vocab + 1
    while not stop:
        if cursor + batch_seqs > order.shape[0]:
            epoch += 1
            order = rng.permutation(n_win)
            cursor = 0
        starts = order[cursor:cursor + batch_seqs] * seq_len
        cursor += batch_seqs
        idx = starts[:, None] + np.arange(seq_len)[None]
        x = torch.from_numpy(packed[idx]).to(device)
        d = torch.from_numpy(doc[idx]).to(device)
        real = x < model.vocab
        if model.objective == "causal":
            logits = model(x[:, :-1], d[:, :-1])
            target = x[:, 1:]
            keep = real[:, 1:]
            loss = F.cross_entropy(logits[keep], target[keep])
            n_pred = int(keep.sum())
        else:
            sel = real & (torch.from_numpy(rng.random(x.shape) < mask_rate).to(device))
            xin = torch.where(sel, torch.full_like(x, mask_id), x)
            logits = model(xin, d)
            loss = F.cross_entropy(logits[sel], x[sel]) if bool(sel.any()) else logits.sum() * 0.0
            n_pred = int(sel.sum())
        elapsed = time.time() - t0
        cur_lr = lr_at(step, elapsed, time_cap_s, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens += n_pred
        if step % 100 == 0 or step == 1:
            losses.append((step, loss.item()))
            log(f"step {step} epoch {epoch} loss {loss.item():.4f} lr {cur_lr:.2e} {tokens / max(1e-9, time.time() - t0):.0f} tok/s")
        if time.time() - t0 >= time_cap_s:
            stop = True
    torch.cuda.synchronize() if device.type == "cuda" else None
    wall = time.time() - t0
    return {"steps": step, "tokens_predicted": tokens, "wall_s": wall, "epochs_completed": epoch,
            "windows_per_epoch": int(n_win), "final_loss": loss.item(), "loss_curve": losses}


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def bits_per_byte(model: SmallTransformer, packed: np.ndarray, doc: np.ndarray, n_bytes: int, *, seq_len: int,
                  batch_seqs: int, device: torch.device) -> dict:
    """Protocol CP on a packed stream: every real token scored once (see the module docstring)."""
    model.eval()
    half = seq_len // 2
    n = packed.shape[0]
    starts = list(range(0, max(1, n - half), half))
    total_nats, n_scored = 0.0, 0
    for i in range(0, len(starts), batch_seqs):
        s = np.array(starts[i:i + batch_seqs])
        idx = s[:, None] + np.arange(seq_len + 1)[None]
        valid = idx < n
        idx = np.minimum(idx, n - 1)
        x = torch.from_numpy(packed[idx]).to(device)
        d = torch.from_numpy(doc[idx]).to(device)
        logits = model(x[:, :-1], d[:, :-1])
        lp = F.log_softmax(logits.float(), dim=-1).gather(-1, torch.clamp(x[:, 1:], max=model.vocab - 1)[..., None])[..., 0]
        # the token at window offset j+1 is scored by this window iff it is real, valid, and
        # (window 0) or (j + 1 >= half + 1, the second half)
        offs = np.arange(1, seq_len + 1)[None]
        score = (x[:, 1:] < model.vocab).cpu().numpy() & valid[:, 1:]
        second = (s[:, None] > 0) & (offs <= half)
        score &= ~second
        m = torch.from_numpy(score).to(device)
        total_nats -= float(lp[m].sum())
        n_scored += int(m.sum())
    return {"bits_per_byte": total_nats / LN2 / n_bytes, "bits_per_token": total_nats / LN2 / max(1, n_scored),
            "n_scored": n_scored}


@torch.no_grad()
def masked_predict(model: SmallTransformer, tok: np.ndarray, doc_start: np.ndarray, hidden: np.ndarray, pos: np.ndarray, *,
                   seq_len: int, batch_seqs: int, device: torch.device) -> np.ndarray:
    """Argmax at the positions `pos` of the stream `tok` with every hidden position replaced by
    MASK, windows of seq_len (stride seq_len / 2, each position taken from the window where it
    is farthest from an edge)."""
    model.eval()
    packed, doc = pack_stream(tok, doc_start, model.vocab)
    ph, _ = pack_stream(np.asarray(hidden).astype(np.int64), doc_start, 0)
    packed = np.where(ph == 1, model.vocab + 1, packed)
    n_doc = doc_start.shape[0] - 1
    pos = np.asarray(pos).astype(np.int64)
    ds = np.asarray(doc_start).astype(np.int64)
    d_of = np.searchsorted(ds, pos, side="right") - 1
    ppos = pos + d_of + 1                         # packed position of every target
    n = packed.shape[0]
    half = seq_len // 2
    # window start for every target: the multiple of half that puts the target nearest the centre
    wstart = np.clip(((ppos - half) // half) * half, 0, max(0, n - seq_len))
    out = np.zeros(pos.shape[0], dtype=np.int64)
    order = np.argsort(wstart, kind="stable")
    uniq, first = np.unique(wstart[order], return_index=True)
    groups = np.split(order, first[1:])
    for i in range(0, len(groups), batch_seqs):
        g = groups[i:i + batch_seqs]
        s = np.array([wstart[x[0]] for x in g])
        idx = np.minimum(s[:, None] + np.arange(seq_len)[None], n - 1)
        x = torch.from_numpy(packed[idx]).to(device)
        d = torch.from_numpy(doc[idx]).to(device)
        pred = model(x, d).argmax(-1).cpu().numpy()
        for row, members in enumerate(g):
            out[members] = pred[row, ppos[members] - s[row]]
    del n_doc
    return out


# ------------------------------------------------------------------ drivers
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_sweep(tok: np.ndarray, doc_start: np.ndarray, ev, metric, out_dir: Path, *, layers: list[int], widths: list[int],
              heads: int, seq_len: int, batch_seqs: int, lr: float, warmup: int, time_cap_s: float, seed: int,
              test: bool = False, objective: str = "causal") -> dict:
    """TB's size sweep: every (layers, width) under the same wall-clock cap; best = lowest validation bpb."""
    device = _device()
    metric("transformer.device", str(device))
    metric("transformer.determinism", seed_everything(seed))
    sampler = NvmlPowerSampler()
    idle = sampler.measure_idle(10.0) if sampler.available else {"available": False, "error": getattr(sampler, "error", None)}
    metric("transformer.nvml_idle", idle, note="GPU power sampled for 10 s with the CUDA context created, nothing running")
    idle_w = idle.get("mean_w") or 0.0
    vocab = ev.vocab
    packed, doc = pack_stream(tok, doc_start, vocab)
    vpacked, vdoc = pack_stream(ev.tok, ev.doc_start, vocab)
    metric("transformer.train_windows", int(packed.shape[0] // seq_len))
    results = {}
    for L in layers:
        for W in widths:
            name = f"transformer_L{L}_W{W}"
            seed_everything(seed)
            model = SmallTransformer(vocab, L, W, heads, seq_len, objective).to(device)
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            sampler.start()
            stats = train(model, packed, doc, seq_len=seq_len, batch_seqs=batch_seqs, lr=lr, warmup=warmup,
                          time_cap_s=time_cap_s, seed=seed, device=device, log=lambda m, n=name: print(f"[{n}] {m}", flush=True))
            energy = sampler.stop()
            t0 = time.time()
            bpb = bits_per_byte(model, vpacked, vdoc, ev.n_bytes, seq_len=seq_len, batch_seqs=batch_seqs, device=device)
            eval_s = time.time() - t0
            joules = energy.get("joules")
            r = {"layers": L, "width": W, "heads": heads, "seq_len": seq_len, "batch_seqs": batch_seqs, "lr": lr,
                 "n_params": model.n_params(), "steps": stats["steps"], "tokens_predicted": stats["tokens_predicted"],
                 "epochs_completed": stats["epochs_completed"], "train_wall_s": stats["wall_s"], "final_loss": stats["final_loss"],
                 "validation_bits_per_byte": bpb["bits_per_byte"], "validation_bits_per_token": bpb["bits_per_token"],
                 "n_scored": bpb["n_scored"], "eval_wall_s": eval_s, "joules_raw": joules,
                 "joules_idle_subtracted": (joules - idle_w * energy["span_s"]) if joules is not None else None,
                 "mean_power_w": energy.get("mean_w"),
                 "peak_gpu_mem_MiB": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
            results[name] = r
            for k, v in r.items():
                metric(f"{name}.{k}", v, note="shared GPU lock: timing and joules indicative only" if k in ("train_wall_s", "joules_raw", "joules_idle_subtracted", "mean_power_w") else "")
            metric(f"{name}.loss_curve", stats["loss_curve"])
            torch.save({"state_dict": model.state_dict(), "config": r}, Path(out_dir) / f"{name}.pt")
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
    best = min(results, key=lambda k: results[k]["validation_bits_per_byte"])
    metric("transformer.best_config", best, note="lowest validation bits per byte at equal wall-clock")
    metric("transformer.best_validation_bits_per_byte", results[best]["validation_bits_per_byte"])
    if test:
        score_test(best, results[best], out_dir, ev.vocab, metric, seq_len=seq_len, batch_seqs=batch_seqs)
    return results


def score_test(name: str, cfg: dict, out_dir: Path, vocab: int, metric, *, seq_len: int, batch_seqs: int) -> None:
    """Score the test split ONCE (counted access, docs/06). Not called in M2."""
    from tmlm.data.corpus_store import load_raw_split, load_split
    from tmlm.data.evalsets import test_access_count

    purpose = f"TB transformer {name}: bits per byte on test (docs/05 TB)"
    te = load_split("wikitext103", vocab, "test", purpose=purpose)
    raw = load_raw_split("wikitext103", "test", purpose=purpose)
    device = _device()
    ck = torch.load(Path(out_dir) / f"{name}.pt", map_location=device)
    model = SmallTransformer(vocab, cfg["layers"], cfg["width"], cfg["heads"], seq_len).to(device)
    model.load_state_dict(ck["state_dict"])
    packed, doc = pack_stream(te.tok, te.doc_start, vocab)
    n_bytes = int(raw.doc_bytes[-1])
    r = bits_per_byte(model, packed, doc, n_bytes, seq_len=seq_len, batch_seqs=batch_seqs, device=device)
    metric(f"{name}.test_bits_per_byte", r["bits_per_byte"], note=f"test-set accesses recorded so far: {test_access_count()}")


def run_masked_smoke(tok: np.ndarray, doc_start: np.ndarray, ev, metric, out_dir: Path, *, layers: int, width: int, heads: int,
                     seq_len: int, batch_seqs: int, lr: float, warmup: int, time_cap_s: float, seed: int) -> dict:
    """The BERT-style variant for `time_cap_s` seconds, then top-1 on the frozen sets. A smoke run
    only: the 24 GPU-h training of docs/06 is NOT started here."""
    device = _device()
    metric("transformer_mlm.device", str(device))
    metric("transformer_mlm.determinism", seed_everything(seed))
    metric("transformer_mlm.note", f"SMOKE run capped at {time_cap_s} s; the 24 GPU-h masked transformer is not run in M2", label="unverified")
    sampler = NvmlPowerSampler()
    vocab = ev.vocab
    packed, doc = pack_stream(tok, doc_start, vocab)
    model = SmallTransformer(vocab, layers, width, heads, seq_len, "masked").to(device)
    sampler.start()
    stats = train(model, packed, doc, seq_len=seq_len, batch_seqs=batch_seqs, lr=lr, warmup=warmup, time_cap_s=time_cap_s,
                  seed=seed, device=device, log=lambda m: print(f"[mlm_smoke] {m}", flush=True))
    energy = sampler.stop()
    name = f"transformer_mlm_L{layers}_W{width}"
    for k in ("steps", "tokens_predicted", "wall_s", "final_loss"):
        metric(f"{name}.{k}", stats[k])
    metric(f"{name}.n_params", model.n_params())
    metric(f"{name}.joules_raw", energy.get("joules"), note="shared lock, indicative only")
    ev.report_masked(name, lambda pos, hidden: masked_predict(model, ev.tok, ev.doc_start, hidden, pos, seq_len=seq_len,
                                                               batch_seqs=batch_seqs, device=device), truectx_too=False)
    torch.save({"state_dict": model.state_dict()}, Path(out_dir) / f"{name}.pt")
    return stats
