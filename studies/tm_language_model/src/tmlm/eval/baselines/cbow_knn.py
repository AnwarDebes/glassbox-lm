"""In-house CBOW embeddings plus k-NN masked-token prediction (docs/06 "Baselines": the clean
stand-in for the initial design's kNN-over-GloVe, docs/12 D5).

Model: word2vec CBOW with negative sampling (Mikolov et al. 2013), numpy only, trained on the
TRAIN split tokens; a context never crosses a document boundary. Per step, for a batch of centre
positions p: h = mean of the input embeddings of the up-to-2w context tokens inside the document,
scores s = h . W_out[o] for o = the centre token and K negatives drawn from the unigram
distribution raised to 0.75, loss = logistic (label 1 for the centre, 0 for the negatives), plain
SGD with a linearly decaying learning rate. Frequent-token subsampling as in word2vec
(keep probability min(1, sqrt(t / f) + t / f)). Hyperparameters are constructor arguments and are
logged by the experiment; the defaults are dim 128, window 5, 10 negatives, batch 4096,
lr 0.025 -> 0.0001, t = 1e-4, one epoch.

Prediction at a hidden position: the context vector h is the mean of the input embeddings of the
VISIBLE tokens in the window (hidden ones are skipped, as the TM sees them masked). Two readouts:
- "softmax": argmax_t h . W_out[t], the model's own CBOW readout (a 1-NN over output vectors);
- "knn": cosine k-NN over a bank of N_bank training context vectors (random positions of the
  train split, their true centre tokens as labels); the prediction is the label with the largest
  sum of similarities among the k neighbours (k = 32, N_bank = 1,000,000 by default).
The experiment reports both; "knn" is the baseline docs/06 names.

The embedding matrices are exchanged as .npy files (`save` / `load`) so that any environment can
read them.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _doc_bounds(doc_start: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = np.asarray(doc_start).astype(np.int64)
    d = np.searchsorted(ds, pos, side="right") - 1
    return ds[d], ds[d + 1]


def context_index(doc_start: np.ndarray, pos: np.ndarray, window: int, n_tok: int,
                  hidden: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(idx [Q, 2w] int64 clipped into the stream, mask [Q, 2w] bool): the window positions of
    every centre that lie inside the centre's document and are not hidden."""
    pos = np.asarray(pos).astype(np.int64)
    lo, hi = _doc_bounds(doc_start, pos)
    offsets = np.array([o for o in range(-window, window + 1) if o != 0], dtype=np.int64)
    idx = pos[:, None] + offsets[None, :]
    mask = (idx >= lo[:, None]) & (idx < hi[:, None])
    idx = np.clip(idx, 0, n_tok - 1)
    if hidden is not None:
        mask &= np.asarray(hidden)[idx] == 0
    return idx, mask


def context_vectors(tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray, w_in: np.ndarray, window: int,
                    hidden: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(h [Q, d] float32 mean input embedding of the visible window, n_ctx [Q] its size)."""
    tok = np.asarray(tok)
    idx, mask = context_index(doc_start, pos, window, tok.shape[0], hidden)
    e = w_in[tok[idx].astype(np.int64)] * mask[:, :, None]
    n_ctx = mask.sum(axis=1)
    h = e.sum(axis=1) / np.maximum(n_ctx, 1)[:, None]
    return h.astype(np.float32), n_ctx


class CbowModel:
    def __init__(self, vocab: int, dim: int = 128, window: int = 5, negatives: int = 10, batch: int = 4096,
                 lr: float = 0.025, lr_min: float = 1e-4, subsample_t: float = 1e-4, epochs: int = 1,
                 seed: int = 1) -> None:
        self.vocab, self.dim, self.window, self.negatives = vocab, dim, window, negatives
        self.batch, self.lr, self.lr_min, self.subsample_t, self.epochs, self.seed = batch, lr, lr_min, subsample_t, epochs, seed
        rng = np.random.default_rng(seed)
        self.w_in = ((rng.random((vocab, dim), dtype=np.float32) - 0.5) / dim).astype(np.float32)
        self.w_out = np.zeros((vocab, dim), dtype=np.float32)
        self.steps_done = 0
        self.tokens_seen = 0
        self.train_wall_s = 0.0

    def hyperparameters(self) -> dict:
        return {"dim": self.dim, "window": self.window, "negatives": self.negatives, "batch": self.batch,
                "lr": self.lr, "lr_min": self.lr_min, "subsample_t": self.subsample_t, "epochs": self.epochs,
                "seed": self.seed, "init": "w_in uniform(-0.5, 0.5)/dim, w_out zero", "optimizer": "SGD, linear decay",
                "negative_distribution": "unigram^0.75", "loss": "logistic negative sampling"}

    # -------------------------------------------------------------- training
    def _scatter_sub(self, w: np.ndarray, idx: np.ndarray, grad: np.ndarray) -> None:
        """w[idx] -= grad with repeated indices accumulated (sort-based, no np.add.at)."""
        order = np.argsort(idx, kind="stable")
        idx_s = idx[order]
        starts = np.flatnonzero(np.concatenate([[True], idx_s[1:] != idx_s[:-1]]))
        summed = np.add.reduceat(grad[order], starts, axis=0)
        w[idx_s[starts]] -= summed

    def train(self, tok: np.ndarray, doc_start: np.ndarray, log_every: int = 0, time_cap_s: float | None = None) -> dict:
        """One or more epochs of SGD over the stream; returns training statistics."""
        tok = np.asarray(tok).astype(np.int64)
        n = tok.shape[0]
        rng = np.random.default_rng(self.seed + 1000)
        freq = np.bincount(tok, minlength=self.vocab).astype(np.float64)
        rel = freq / max(1.0, freq.sum())
        keep = np.minimum(1.0, np.sqrt(self.subsample_t / np.maximum(rel, 1e-12)) + self.subsample_t / np.maximum(rel, 1e-12))
        neg_p = freq ** 0.75
        neg_p /= neg_p.sum()
        neg_cdf = np.cumsum(neg_p)
        t_start = time.time()
        stopped_early = False
        # centre positions kept by subsampling, one pass per epoch
        total_steps_est = None
        for epoch in range(self.epochs):
            centres = np.flatnonzero(rng.random(n) < keep[tok])
            rng.shuffle(centres)
            n_steps = (centres.shape[0] + self.batch - 1) // self.batch
            if total_steps_est is None:
                total_steps_est = n_steps * self.epochs
            for s in range(n_steps):
                frac = self.steps_done / max(1, total_steps_est)
                lr = max(self.lr_min, self.lr * (1.0 - frac))
                p = centres[s * self.batch:(s + 1) * self.batch]
                b = p.shape[0]
                idx, mask = context_index(doc_start, p, self.window, n)
                ctx = tok[idx]
                n_ctx = mask.sum(axis=1)
                ok = n_ctx > 0
                if not ok.all():
                    p, idx, mask, ctx, n_ctx = p[ok], idx[ok], mask[ok], ctx[ok], n_ctx[ok]
                    b = p.shape[0]
                    if b == 0:
                        continue
                maskf = mask.astype(np.float32)
                h = np.einsum("bcd,bc->bd", self.w_in[ctx], maskf) / n_ctx[:, None].astype(np.float32)
                neg = np.searchsorted(neg_cdf, rng.random((b, self.negatives)))
                neg = np.minimum(neg, self.vocab - 1)
                o = np.concatenate([tok[p][:, None], neg], axis=1)                      # [b, 1+K]
                wo = self.w_out[o]                                                       # [b, 1+K, d]
                sc = np.einsum("bd,bkd->bk", h, wo)
                g = 1.0 / (1.0 + np.exp(-sc))
                g[:, 0] -= 1.0                                                           # sigmoid - label
                g = (g * lr).astype(np.float32)
                grad_h = np.einsum("bk,bkd->bd", g, wo)                                  # [b, d]
                grad_out = g[:, :, None] * h[:, None, :]                                 # [b, 1+K, d]
                self._scatter_sub(self.w_out, o.reshape(-1), grad_out.reshape(-1, self.dim))
                grad_in = (grad_h / n_ctx[:, None].astype(np.float32))[:, None, :] * maskf[:, :, None]
                self._scatter_sub(self.w_in, ctx.reshape(-1), grad_in.reshape(-1, self.dim))
                self.steps_done += 1
                self.tokens_seen += b
                if log_every and self.steps_done % log_every == 0:
                    loss = float(-np.log(np.maximum(1.0 / (1.0 + np.exp(-sc[:, 0])), 1e-9)).mean()
                                 - np.log(np.maximum(1.0 - 1.0 / (1.0 + np.exp(-sc[:, 1:])), 1e-9)).mean())
                    print(f"[cbow] epoch {epoch} step {self.steps_done}/{total_steps_est} lr {lr:.5f} loss {loss:.4f} "
                          f"{self.tokens_seen / (time.time() - t_start):.0f} centres/s", flush=True)
                if time_cap_s is not None and time.time() - t_start > time_cap_s:
                    stopped_early = True
                    break
            if stopped_early:
                break
        self.train_wall_s += time.time() - t_start
        return {"steps": self.steps_done, "centres_seen": self.tokens_seen, "wall_s": self.train_wall_s,
                "stopped_early": stopped_early, "planned_steps": total_steps_est}

    # -------------------------------------------------------------- persistence
    def save(self, directory: Path) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"w_in": directory / "cbow_w_in.npy", "w_out": directory / "cbow_w_out.npy"}
        np.save(paths["w_in"], self.w_in)
        np.save(paths["w_out"], self.w_out)
        return paths

    @classmethod
    def load(cls, directory: Path, **kw) -> "CbowModel":
        directory = Path(directory)
        w_in = np.load(directory / "cbow_w_in.npy")
        m = cls(vocab=w_in.shape[0], dim=w_in.shape[1], **kw)
        m.w_in = w_in
        m.w_out = np.load(directory / "cbow_w_out.npy")
        return m

    # -------------------------------------------------------------- prediction
    def predict_softmax(self, tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray,
                        hidden: np.ndarray | None = None, chunk: int = 8192) -> np.ndarray:
        pos = np.asarray(pos)
        out = np.zeros(pos.shape[0], dtype=np.int64)
        for i in range(0, pos.shape[0], chunk):
            h, _ = context_vectors(tok, doc_start, pos[i:i + chunk], self.w_in, self.window, hidden)
            out[i:i + chunk] = (h @ self.w_out.T).argmax(axis=1)
        return out

    def build_bank(self, tok: np.ndarray, doc_start: np.ndarray, n_bank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """(unit context vectors [n_bank, d] float32, centre tokens [n_bank]) at random train positions."""
        tok = np.asarray(tok)
        rng = np.random.default_rng(seed)
        n_bank = min(n_bank, tok.shape[0])
        pos = np.sort(rng.choice(tok.shape[0], size=n_bank, replace=False))
        vecs = np.zeros((n_bank, self.dim), dtype=np.float32)
        for i in range(0, n_bank, 65536):
            h, _ = context_vectors(tok, doc_start, pos[i:i + 65536], self.w_in, self.window)
            vecs[i:i + 65536] = h
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
        return vecs, tok[pos].astype(np.int64)

    def predict_knn(self, tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray, bank: tuple[np.ndarray, np.ndarray],
                    k: int = 32, hidden: np.ndarray | None = None, chunk: int = 1024) -> np.ndarray:
        """Cosine k-NN over `bank`; the label with the largest summed similarity wins."""
        vecs, labels = bank
        pos = np.asarray(pos)
        out = np.zeros(pos.shape[0], dtype=np.int64)
        k = min(k, vecs.shape[0])
        for i in range(0, pos.shape[0], chunk):
            h, _ = context_vectors(tok, doc_start, pos[i:i + chunk], self.w_in, self.window, hidden)
            h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
            sim = h @ vecs.T                                                    # [q, n_bank]
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            rows = np.arange(top.shape[0])[:, None]
            s = np.maximum(sim[rows, top], 0.0)
            lab = labels[top]
            votes = np.zeros((top.shape[0], self.vocab), dtype=np.float64)
            np.add.at(votes, (np.repeat(np.arange(top.shape[0]), k), lab.reshape(-1)), s.reshape(-1))
            out[i:i + chunk] = votes.argmax(axis=1)
        return out
