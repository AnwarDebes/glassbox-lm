"""Boolean views of a frozen embedding for the Tsetlin Machine, each with readable bit names.

- ``Thermo``: per-dimension quantile thermometer (the Clause-Driven-LLM booleanizer):
  bit = emb[d] >= t. Not readable in words: a dimension has no meaning of its own.
- ``Proto``: similarity to class prototypes. For each class, k spherical k-means centroids of
  its training embeddings; each centroid is named by its nearest training sentence. Bits:
  * level bits  sim(q, p) >= t, with thresholds at quantiles of the similarities of the
    prototype's own cluster members (so every prototype gets thresholds on its own scale);
  * rank bits   "class c owns the r nearest prototype(s)" for r in ``ranks``: bit (c, r) is 1
    when one of the r most similar prototypes of q belongs to class c.
  A literal then reads, for example, ``sim(q, "how do i activate my card") >= 0.71`` or
  ``nearest prototype is activate_my_card``.
- ``Concat``: several views side by side.
"""

from __future__ import annotations

import numpy as np
import torch


class Thermo:
    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y=None, texts=None, class_names=None) -> "Thermo":
        qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.th = [np.unique(np.percentile(X[:, d], qs)).astype(np.float32) for d in range(X.shape[1])]
        self.names = [f"emb[{d}] >= {t:.3f}" for d, th in enumerate(self.th) for t in th]
        return self

    def transform(self, X: np.ndarray, texts=None) -> np.ndarray:
        return np.concatenate([(X[:, d : d + 1] >= th[None, :]) for d, th in enumerate(self.th)], 1).astype(np.uint8)


def _spherical_kmeans(Z: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> torch.Tensor:
    n = Z.shape[0]
    if k == 1 or n <= k:
        c = Z.mean(0, keepdim=True) if k == 1 else Z[:k].clone()
        return torch.nn.functional.normalize(c, dim=1)
    # k-means++ seeding on cosine distance
    idx = [int(torch.randint(0, n, (1,), generator=gen, device=Z.device))]
    d2 = 1 - Z @ Z[idx[0]]
    for _ in range(1, k):
        p = d2.clamp(min=0) + 1e-12
        idx.append(int(torch.multinomial(p / p.sum(), 1, generator=gen)))
        d2 = torch.minimum(d2, 1 - Z @ Z[idx[-1]])
    c = Z[idx].clone()
    for _ in range(iters):
        a = (Z @ c.T).argmax(1)
        new = torch.zeros_like(c).index_add_(0, a, Z)
        cnt = torch.bincount(a, minlength=k)
        empty = cnt == 0
        new[empty] = c[empty]
        c = torch.nn.functional.normalize(new, dim=1)
    return c


class Proto:
    def __init__(self, k: int = 4, levels: tuple = (5, 25, 50, 75), ranks: tuple = (1, 3), seed: int = 0,
                 iters: int = 25, device: str = "cuda", whiten: float = 0.0):
        self.k, self.levels, self.ranks, self.seed, self.iters, self.dev = k, tuple(levels), tuple(ranks), seed, iters, device
        # whiten > 0: similarities are taken after whitening by the within-class covariance of the
        # training embeddings (shrinkage whiten * trace / dim), i.e. a Mahalanobis-like closeness
        self.whiten = float(whiten)
        self.W = None

    def _z(self, X) -> torch.Tensor:
        Z = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        if self.W is not None:
            Z = Z @ self.W
        return torch.nn.functional.normalize(Z, dim=1)

    def fit(self, X: np.ndarray, y: np.ndarray, texts=None, class_names=None) -> "Proto":
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(self.seed)
        yt = torch.as_tensor(y, device=self.dev)
        C = int(y.max()) + 1
        if self.whiten > 0:
            X64 = torch.as_tensor(X, dtype=torch.float64, device=self.dev)
            mu = torch.stack([X64[yt == c].mean(0) for c in range(C)])
            R = X64 - mu[yt]
            Sw = R.T @ R / X64.shape[0]
            D = Sw.shape[0]
            A = Sw + self.whiten * torch.trace(Sw) / D * torch.eye(D, device=self.dev, dtype=torch.float64)
            e, U = torch.linalg.eigh(A)
            self.W = (U @ torch.diag(e.rsqrt()) @ U.T).float()
        Z = self._z(X)
        protos, owner, th = [], [], []
        for c in range(C):
            Zc = Z[yt == c]
            cent = _spherical_kmeans(Zc, self.k, self.iters, gen)
            a = (Zc @ cent.T).argmax(1)
            for j in range(cent.shape[0]):
                members = Zc[a == j] if (a == j).any() else Zc
                sims = (members @ cent[j]).float().cpu().numpy()
                t = np.unique(np.percentile(sims, self.levels)).astype(np.float32)
                protos.append(cent[j])
                owner.append(c)
                th.append(t)
        self.P = torch.stack(protos)  # [n_proto, d]
        self.owner = torch.as_tensor(owner, device=self.dev)
        self.th = th
        self.C = C
        # names: nearest training sentence of each prototype
        sims_all = Z @ self.P.T
        near = sims_all.argmax(0).cpu().numpy()
        cn = class_names or [str(c) for c in range(C)]
        self.proto_names = [f'"{texts[i]}" [{cn[o]}]' if texts is not None else f"proto{j}[{cn[o]}]"
                            for j, (i, o) in enumerate(zip(near, owner))]
        tag = "sim_w" if self.W is not None else "sim"
        self.names = [f"{tag}(q, {self.proto_names[j]}) >= {t:.3f}" for j, ts in enumerate(th) for t in ts]
        self.names += [f"one of the {r} nearest prototypes is {cn[c]}" for r in self.ranks for c in range(C)]
        return self

    def sims(self, X: np.ndarray) -> torch.Tensor:
        return self._z(X) @ self.P.T

    def transform(self, X: np.ndarray, texts=None) -> np.ndarray:
        S = self.sims(X)  # [n, n_proto]
        bits = [torch.cat([(S[:, j : j + 1] >= torch.as_tensor(t, device=self.dev)[None, :]) for j, t in enumerate(self.th)], 1)]
        if self.ranks:
            order = S.argsort(1, descending=True)
            own = self.owner[order]  # [n, n_proto] class of the r-th nearest prototype
            for r in self.ranks:
                b = torch.zeros((S.shape[0], self.C), dtype=torch.bool, device=self.dev)
                b.scatter_(1, own[:, :r], True)
                bits.append(b)
        return torch.cat(bits, 1).to(torch.uint8).cpu().numpy()


class Lex:
    """Presence of the K most frequent word 1- and 2-grams of the training texts (the
    Clause-Driven-LLM lexical backend: binary 1-2-gram presence, 2,000 features)."""

    def __init__(self, k: int = 2000, ngram_max: int = 2):
        self.k, self.ngram_max = k, ngram_max

    def fit(self, X, y=None, texts=None, class_names=None) -> "Lex":
        from sklearn.feature_extraction.text import CountVectorizer

        self.vec = CountVectorizer(ngram_range=(1, self.ngram_max), max_features=self.k, binary=True,
                                   token_pattern=r"(?u)\b\w+\b")
        self.vec.fit(texts)
        self.names = [f'contains "{w}"' for w in self.vec.get_feature_names_out()]
        return self

    def transform(self, X, texts=None) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.uint8)


class Concat:
    def __init__(self, views: list):
        self.views = views

    def fit(self, X, y=None, texts=None, class_names=None) -> "Concat":
        for v in self.views:
            v.fit(X, y, texts=texts, class_names=class_names)
        self.names = [n for v in self.views for n in v.names]
        return self

    def transform(self, X, texts=None):
        return np.concatenate([v.transform(X, texts) for v in self.views], 1)


def make_view(spec: str, seed: int = 0):
    """``thermo:8``, ``proto:k=4:levels=5,25,50,75:ranks=1,3``, or ``a+b``."""
    if "+" in spec:
        return Concat([make_view(s, seed) for s in spec.split("+")])
    kind, *opts = spec.split(":")
    kw = dict(o.split("=") for o in opts if "=" in o)
    if kind == "thermo":
        return Thermo(int(opts[0]) if opts and "=" not in opts[0] else int(kw.get("bins", 8)))
    if kind == "lex":
        return Lex(int(opts[0]) if opts and "=" not in opts[0] else int(kw.get("k", 2000)), int(kw.get("n", 2)))
    if kind in ("lda", "white"):  # thermometer after a fitted linear map (src/dl/boolean_rot.py, exploration e24)
        from dl.boolean_rot import Rotated

        n_bins = int(opts[0]) if opts and "=" not in opts[0] else int(kw.get("bins", 8))
        return Rotated(kind, n_bins, shrink=float(kw.get("shrink", 0.001)), max_axes=int(kw.get("axes", 0)))
    if kind == "proto":
        levels = tuple(float(x) for x in kw.get("levels", "5,25,50,75").split(",") if x)
        ranks = tuple(int(x) for x in kw.get("ranks", "1,3").split(",") if x)
        return Proto(k=int(kw.get("k", 4)), levels=levels, ranks=ranks, seed=seed, whiten=float(kw.get("whiten", 0.0)))
    raise ValueError(spec)
