"""The CVTM-core end to end (docs/11 sections 11.2 to 11.4 and the protocol of 11.6).

`CVTMCore.fit(train_tok, train_doc_start, ...)`, in this fixed order (docs/11 11.6):

1. Training documents: the whole split, or with `prefix_tokens` its leading whole documents
   (`protocol.prefix_documents`); stable document IDs are the indices in the full split.
2. Fit documents H: residue 0 (`protocol`, docs/12 F20 b). On the full V 8000 split the selection is
   asserted (304 documents, 1,397,702 tokens, index sha256 of the exploratory run).
3. KN-5 WITHOUT H: `tmlm.cvtm.kn_gpu.fit_kn` (backend "cupy" by default; array-identical to
   `KNModel.fit`), then its suffix clauses (`tmlm.cvtm.suffix.SuffixClauses.from_kn`). On every
   position of H: ln P_S(gold) from the float64 vote sum (`SuffixClauses.logprob`, the reference form
   of docs/11 11.2) and k*, the longest fired suffix context (0 when only the bias fires).
4. Document clauses on H (`doc_clauses.doc_features`, orders 0..4, window N, pointer 5..64).
5. Buckets and MAP-EM (`mixture.bucket_ids`, `mixture.fit_map_em` with the library defaults: 50
   iterations from uniform, float64, Dirichlet prior of 16 pseudo-counts toward the plain-EM pooled
   weights, docs/12 F20 c). The weights are frozen here.
6. KN-5 on ALL training documents (the model's suffix clauses). With `min_counts` both KN fits use
   the F20 d pruning (`prune_cont="full"`, `prune_mass="backoff"`), so the EM sees the same estimator
   that is scored.

Options: `prefix_tokens` (None: the whole split), `window` N (None: the whole document so far),
`min_counts` ({order: minimum raw count} for orders 3..5, non-decreasing), `components`
("full" = S, D0..D4, P; "S+D0" = the H8.4 comparator; "S" = KN-5 alone; or an explicit tuple).
`CVTMCore.variant(components=..., window=...)` refits only the EM on the stored H inputs (the KN
fits are unchanged), which is what a window or component sweep needs.

Scoring. For a stream (whole documents): ln P_S(gold) and k* from the suffix clauses of the KN on all
training documents, document features with the model's window, the bucket, and
P(gold) = `mixture.gold_prob` (the combination of 11.4). `score` returns the per-token log2 P, bits per
byte (total bits / UTF-8 bytes of the split, protocol CP) and bits per document.

Full distribution (`dense_pass`, `topk`). Per position: ln P_S(w) for every w from the votes
(bias + sum over fired clauses of the clause constant, plus the follower votes, added in the same order
as `SuffixClauses.logprob`, so the gold entry is bit-identical), P_S = exp, then the mixture of 11.4 with
the exact document successor counts (`doc_clauses.doc_candidates`) and the pointer candidate, added in
the order of `mixture.gold_prob` (so the core's gold entry is bit-identical to the scoring pass; both
are checked). Top-k: descending probability, ties to the lower token id.

Save format (`save`, `load`): one uncompressed .npz. The KN counts and statistics are stored as
integers ("kn.<name>" with the names of `kn_gpu.model_arrays`; the float64 totals and pruned masses
are integer-valued and stored as int64, checked exactly), the EM weights (float64 [n_buckets, 7]),
the H inputs of the EM ("fit.*": tokens, document boundaries, stable IDs, ln P_S under the KN without
H, k*) and a JSON header ("meta_json", uint8 bytes: format, config, KN scalars with the discounts as
float.hex, the sha256 of every KN array, the state hash, fit information). `load` rebuilds the
KNModel, checks every KN array hash, rebuilds the suffix clauses (deterministic) and checks the
state hash.

State hash (`state_hash`): sha256 over b"tmlm.cvtm.model.v1\\0", the canonical JSON of the identity
config (`identity`) and b"\\0", then "suffix=<SuffixClauses.state_hash()>\\n" (the integer arrays of the
suffix clauses) and "weights=<tmlm.common.hashing.sha256_array(weights)>\\n" (the float64 weights: the
EM runs in a fixed order and is deterministic on one platform, docs/11 11.6).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..common import hashing
from ..eval.baselines.ngram import KNModel, _Level
from . import kn_gpu, protocol
from .doc_clauses import DocFeatures, doc_candidates, doc_features
from .mixture import (CACHE_COMPONENTS, COMPONENTS, D_ORDERS, N_COMP, BucketLayout, bucket_ids,
                      component_mask, counts_matrix, fit_map_em, gold_components, gold_prob)
from .suffix import SuffixClauses

FORMAT = "tmlm.cvtm.model.v1"
ORDER = 5
PTR_MIN, PTR_MAX = 5, 64                 # docs/11 11.3 (fixed by the spec, not options)
EM_ITERS, PRIOR = 50, 16.0               # docs/11 11.4 (library defaults)
COMPONENT_SETS = {"full": COMPONENTS, "S+D0": CACHE_COMPONENTS, "S": ("S",)}
LN2 = math.log(2.0)
_SHIFT = np.uint64(16)
_MASK16 = np.uint64(0xFFFF)


def resolve_components(components: str | Sequence[str] | None) -> tuple[str, ...]:
    """"full" / None, "S+D0", "S", or an explicit tuple of component names (S required)."""
    if components is None:
        return COMPONENTS
    if isinstance(components, str):
        if components not in COMPONENT_SETS:
            raise ValueError(f"components must be one of {sorted(COMPONENT_SETS)} or a tuple of {COMPONENTS}")
        return COMPONENT_SETS[components]
    return BucketLayout.of(tuple(components)).components


def _norm_min_counts(min_counts: Mapping[int, int] | None) -> dict[int, int] | None:
    if min_counts is None:
        return None
    return {int(k): int(v) for k, v in sorted(dict(min_counts).items(), key=lambda kv: int(kv[0]))}


# ------------------------------------------------------------------ KN serialisation
def kn_to_arrays(kn: KNModel) -> tuple[dict[str, np.ndarray], dict]:
    """(arrays, header) of a fitted KNModel: every array of `kn_gpu.model_arrays` under "kn.<name>",
    float64 arrays converted to int64 (they hold integer sums; checked exactly), and the header
    {"scalars": model_scalars, "hashes": sha256 of every array in its ORIGINAL dtype, "float_as_int": [...]}."""
    arrays: dict[str, np.ndarray] = {}
    as_int: list[str] = []
    hashes: dict[str, str] = {}
    for name, a in kn_gpu.model_arrays(kn).items():
        a = np.asarray(a)
        hashes[name] = hashing.sha256_array(a)
        if a.dtype == np.float64:
            if a.size and (not np.all(np.isfinite(a)) or float(np.max(np.abs(a))) >= 2.0 ** 53):
                raise ValueError(f"{name}: not an exact integer-valued float64 array")
            ai = a.astype(np.int64)
            if not np.array_equal(ai.astype(np.float64), a):
                raise ValueError(f"{name}: float64 values are not integers; cannot store exactly as int64")
            arrays[f"kn.{name}"] = ai
            as_int.append(name)
        else:
            arrays[f"kn.{name}"] = a
    return arrays, {"scalars": kn_gpu.model_scalars(kn), "hashes": hashes, "float_as_int": sorted(as_int)}


def kn_from_arrays(arrays: Mapping[str, np.ndarray], header: dict, verify: bool = True) -> KNModel:
    """Inverse of `kn_to_arrays`; with `verify` every array hash is checked (ValueError on mismatch)."""
    sc = header["scalars"]
    kn = KNModel(int(sc["order"]), int(sc["vocab"]), discount_mode=sc["discount_mode"],
                 discount=float.fromhex(sc["discount"]), unk_interp=bool(sc["unk_interp"]), pad=bool(sc["pad"]))
    as_int = set(header["float_as_int"])

    def get(name: str) -> np.ndarray | None:
        key = f"kn.{name}"
        if key not in arrays:
            return None
        a = np.asarray(arrays[key])
        return a.astype(np.float64) if name in as_int else a

    for k in range(1, kn.order + 1):
        kn.keys[k] = get(f"keys.{k}")
        for variant, levels in (("raw", kn.raw), ("cont", kn.cont)):
            cnt = get(f"{variant}.{k}.cnt")
            if cnt is None:
                levels[k] = None
                continue
            d = tuple(float.fromhex(x) for x in sc[f"{variant}.{k}.d"])
            fields = dict(cnt=cnt, total=get(f"{variant}.{k}.total"), n1=get(f"{variant}.{k}.n1"),
                          n2=get(f"{variant}.{k}.n2"), n3=get(f"{variant}.{k}.n3"), n_any=get(f"{variant}.{k}.n_any"), d=d)
            extra = get(f"{variant}.{k}.extra")
            levels[k] = kn_gpu._PrunedLevel(**fields, extra=extra) if extra is not None else _Level(**fields)
    kn.n_types = [int(x) for x in sc["n_types"]]
    kn.n_train_tokens = int(sc["n_train_tokens"])
    if verify:
        got = {name: hashing.sha256_array(np.asarray(a)) for name, a in kn_gpu.model_arrays(kn).items()}
        if got != header["hashes"]:
            bad = sorted(set(got.items()) ^ set(header["hashes"].items()))
            raise ValueError(f"KN arrays do not match their stored hashes: {bad[:6]}")
        if kn_gpu.model_scalars(kn) != sc:
            raise ValueError("KN scalars do not round-trip")
    return kn


def save_kn(kn: KNModel, path: str | Path, extra_meta: dict | None = None) -> dict:
    """Write one KN-5 as an uncompressed .npz ("kn.*" arrays + "meta_json"); returns
    {"path", "sha256", "bytes", "model_hashes"}."""
    arrays, header = kn_to_arrays(kn)
    meta = {"format": "tmlm.cvtm.kn.v1", "kn": header, **(extra_meta or {})}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    path = Path(path)
    np.savez(path, **arrays)
    return {"path": str(path), "sha256": hashing.sha256_file(path), "bytes": int(path.stat().st_size),
            "arrays_sha256": header["hashes"]}


def load_kn(path: str | Path, verify: bool = True) -> tuple[KNModel, dict]:
    """(KNModel, meta) of a file written by `save_kn` (or the KN inside a model file)."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
        arrays = {k: z[k] for k in z.files if k.startswith("kn.")}
    return kn_from_arrays(arrays, meta["kn"], verify=verify), meta


# ------------------------------------------------------------------ suffix read-out
def suffix_gold(sc: SuffixClauses, tok: np.ndarray, doc_start: np.ndarray,
                chunk: int = 1 << 16) -> tuple[np.ndarray, np.ndarray]:
    """(ln P_S(tok[p] | left in-document context) float64 [n], k* int8 [n]) for every position:
    the float64 vote sum of `SuffixClauses.logprob` and the longest fired suffix context."""
    tok64 = np.asarray(tok).astype(np.int64)
    ds = np.asarray(doc_start).astype(np.int64)
    n = tok64.shape[0]
    lnp = np.empty(n, dtype=np.float64)
    ks = np.zeros(n, dtype=np.int8)
    for i in range(0, n, chunk):
        pos = np.arange(i, min(i + chunk, n), dtype=np.int64)
        ctx, valid = sc.contexts(tok64, ds, pos)
        fired = sc.fired(ctx, valid)
        lnp[pos] = sc.logprob(ctx, tok64[pos], valid)
        k = np.zeros(pos.shape[0], dtype=np.int8)
        for kk in range(1, sc.order):
            k = np.where(fired[:, kk - 1] >= 0, np.int8(kk), k).astype(np.int8)
        ks[pos] = k
    return lnp, ks


def suffix_ln_rows(sc: SuffixClauses, ctx: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """float64 [Q, V + 1]: ln P_S(w | ctx) for every class (column V = BOS, -inf) from the votes, with
    the additions in the order of `SuffixClauses.logprob` (bias, then per level the clause constant,
    then the follower vote), so every entry is bit-identical to `logprob` of that class."""
    ctx = np.asarray(ctx).astype(np.int64)
    fired = sc.fired(ctx, valid)
    q = ctx.shape[0]
    rows = np.empty((q, sc.b.shape[0]), dtype=np.float64)
    rows[:] = sc.b[None, :]
    for j in range(2, sc.order + 1):
        cid = fired[:, j - 2]
        r = np.flatnonzero(cid >= 0)
        if r.shape[0] == 0:
            continue
        c = cid[r]
        # the clause constant of every firing row; + 0.0 on the other rows is exact (x + 0.0 == x), so a
        # dense in-place add performs the same float operations as logprob's masked add
        dcol = np.zeros(q, dtype=np.float64)
        dcol[r] = sc.d[j][c]
        rows += dcol[:, None]
        keys = sc.keys[j]
        cu = c.astype(np.uint64)
        lo = np.searchsorted(keys, cu << _SHIFT)
        hi = np.searchsorted(keys, (cu + np.uint64(1)) << _SHIFT)
        lengths = (hi - lo).astype(np.int64)
        if int(lengths.sum()) == 0:
            continue
        row_rep = np.repeat(r, lengths)
        offs = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        tid = np.repeat(lo.astype(np.int64), lengths) + offs
        wcol = (keys[tid] & _MASK16).astype(np.int64)
        rows[row_rep, wcol] = rows[row_rep, wcol] + sc.v[j][tid]      # (row, class) pairs unique per level
    return rows


def topk_rows(p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(ids int64 [Q, k], values [Q, k]): the k largest entries of every row, in descending order,
    ties broken toward the lower column index (canonical, independent of the partition algorithm)."""
    q, v = p.shape
    if not 1 <= k <= v:
        raise ValueError("need 1 <= k <= number of columns")
    if np.isnan(p).any():
        raise ValueError("NaN in the input of topk_rows")
    ids = np.argpartition(p, v - k, axis=1)[:, v - k:]
    vals = np.take_along_axis(p, ids, axis=1)
    thr = vals.min(axis=1)
    n_ge = (p >= thr[:, None]).sum(axis=1)
    tie = np.flatnonzero(n_ge > k)                  # entries equal to the k-th value lie on both sides of the cut
    if tie.shape[0]:
        pt = p[tie]
        gt = pt > thr[tie, None]
        eq = pt == thr[tie, None]
        keep = gt | (eq & (np.cumsum(eq, axis=1) <= (k - gt.sum(axis=1))[:, None]))
        ids[tie] = np.nonzero(keep)[1].reshape(tie.shape[0], k)   # the lowest ids among the tied entries
        vals[tie] = np.take_along_axis(pt, ids[tie], axis=1)
    order = np.lexsort((ids, -vals), axis=1)        # descending value, then ascending id
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(vals, order, axis=1)


# ------------------------------------------------------------------ records
@dataclass
class FitInputs:
    """The H inputs of the EM: tokens, document boundaries, stable IDs, ln P_S(gold) under the KN
    fitted without H (vote form) and k*."""

    tok: np.ndarray
    doc_start: np.ndarray
    doc_index: np.ndarray
    lnp_s: np.ndarray
    kstar: np.ndarray


@dataclass
class GoldRecord:
    """Per-position gold read-out of one stream under one core (see `CVTMCore.gold`)."""

    tok: np.ndarray             # int64 [n]
    doc_start: np.ndarray       # int64 [d + 1]
    lnp_s: np.ndarray           # float64 [n] ln P_S(gold)
    kstar: np.ndarray           # int8 [n]
    feats: DocFeatures
    bucket: np.ndarray          # int64 [n] (layout of the core's components)
    p_comp: np.ndarray          # float64 [n, 7]
    act: np.ndarray             # bool [n, 7]
    p_gold: np.ndarray          # float64 [n] P(gold) of the core
    timings: dict = field(default_factory=dict)

    @property
    def log2p(self) -> np.ndarray:
        return np.log2(self.p_gold)


def fit_weights(fi: FitInputs, components: Sequence[str], window: int | None, em_iters: int = EM_ITERS,
                prior: float = PRIOR) -> tuple[np.ndarray, dict]:
    """MAP-EM weights of docs/11 11.4 on the H inputs (steps 4 and 5 of the module docstring)."""
    comps = resolve_components(components)
    feats = doc_features(fi.tok, fi.doc_start, orders=D_ORDERS, window=window, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
    tok64 = np.asarray(fi.tok).astype(np.int64)
    p_comp, act = gold_components(np.exp(fi.lnp_s), feats, tok64)
    bucket, nb = bucket_ids(fi.kstar.astype(np.int64), counts_matrix(feats), feats.ptr_len, comps, PTR_MIN)
    trace: dict = {}
    w = fit_map_em(p_comp, act, bucket, iters=int(em_iters), prior=float(prior), n_buckets=nb, components=comps,
                   trace=trace)
    trace["n_fit_rows"] = int(tok64.shape[0])
    trace["n_fallback"] = int(feats.n_fallback)
    return w, trace


# ------------------------------------------------------------------ the model
class CVTMCore:
    """S + D (+ P) with frozen MAP-EM weights over the suffix clauses of the KN-5 on all training docs."""

    def __init__(self, *, suffix: SuffixClauses, weights: np.ndarray, config: dict, fit_inputs: FitInputs | None,
                 kn: KNModel | None = None, fit_info: dict | None = None) -> None:
        self.suffix = suffix
        self.weights = np.asarray(weights, dtype=np.float64)
        self.config = dict(config)
        self.components = resolve_components(self.config["components"])
        self.window = self.config["window"]
        self.layout = BucketLayout.of(self.components)
        if self.weights.shape != (self.layout.n_buckets, N_COMP):
            raise ValueError(f"weights must be [{self.layout.n_buckets}, {N_COMP}] for components {self.components}")
        self.fit_inputs = fit_inputs
        self.kn = kn
        self.fit_info = fit_info or {}
        self.vocab = int(suffix.vocab)

    # -------------------------------------------------------------- identity
    def identity(self) -> dict:
        c = self.config
        return {"format": FORMAT, "order": int(self.suffix.order), "vocab": self.vocab, "window": c["window"],
                "ptr_min": PTR_MIN, "ptr_max": PTR_MAX, "components": list(self.components),
                "min_counts": None if c.get("min_counts") is None else {str(k): v for k, v in c["min_counts"].items()},
                "prune_cont": "full", "prune_mass": "backoff", "prefix_tokens": c.get("prefix_tokens"),
                "em_iters": int(c["em_iters"]), "prior": float(c["prior"]),
                "fit_protocol": {"salt": protocol.SALT, "modulus": protocol.MODULUS, "residues": list(protocol.H_RESIDUES)}}

    def state_hash(self) -> str:
        h = hashlib.sha256()
        h.update(FORMAT.encode("ascii") + b"\x00")
        h.update(hashing.canonical_json(self.identity()).encode("utf-8") + b"\x00")
        h.update(f"suffix={self.suffix.state_hash()}\n".encode("ascii"))
        h.update(f"weights={hashing.sha256_array(self.weights)}\n".encode("ascii"))
        return h.hexdigest()

    # -------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, tok: np.ndarray, doc_start: np.ndarray, *, vocab: int = protocol.VOCAB,
            prefix_tokens: int | None = None, window: int | None = None,
            min_counts: Mapping[int, int] | None = None, components: str | Sequence[str] = "full",
            em_iters: int = EM_ITERS, prior: float = PRIOR, backend: str = "cupy",
            max_device_bytes: int | None = None, expect_h: bool | None = None,
            callback: Callable[..., None] | None = None, log: Callable[[str], None] | None = None) -> "CVTMCore":
        """docs/11 11.6 end to end (module docstring). `callback(stage, **kw)` is called with
        stage "kn_noH" (kn, suffix, fit), "em" (weights, fit_inputs, trace) and "kn_all" (kn, suffix), so a
        caller can save or use the intermediate models before they are released. expect_h: assert the
        F20 b selection (default: when the split is the full 28,472-document V 8000 train split)."""
        say = log or (lambda m: None)
        info: dict = {"phases_s": {}}
        t_all = time.perf_counter()

        def phase(name: str, t0: float) -> None:
            info["phases_s"][name] = round(time.perf_counter() - t0, 3)

        comps = resolve_components(components)
        mc = _norm_min_counts(min_counts)
        tok_all, ds_all = protocol.prefix_documents(tok, doc_start, prefix_tokens)
        ds_all = np.asarray(ds_all).astype(np.int64)
        n_docs = int(ds_all.shape[0] - 1)
        held, rest = protocol.split_by_residue(tok_all, ds_all, protocol.H_RESIDUES)
        if expect_h is None:
            expect_h = prefix_tokens is None and int(vocab) == protocol.VOCAB and n_docs == 28472
        if expect_h:
            protocol.check_h(held.n_docs, held.n_tokens, held.doc_index)
        info.update({"train_docs": n_docs, "train_tokens": int(ds_all[-1]), "H_docs": held.n_docs,
                     "H_tokens": held.n_tokens, "H_doc_index_sha256": protocol.index_sha256(held.doc_index),
                     "H_checked_against_F20b": bool(expect_h)})
        say(f"H: {held.n_docs} documents, {held.n_tokens} tokens; KN without H on {rest.n_tokens} tokens")
        kw = dict(order=ORDER, vocab=int(vocab), backend=backend, max_device_bytes=max_device_bytes)
        if mc is not None:
            kw.update(min_counts=mc, prune_cont="full", prune_mass="backoff")
        # ---- KN-5 without H: ln P_S and k* on H
        t0 = time.perf_counter()
        kn = kn_gpu.fit_kn(rest.tok, rest.doc_start, **kw)
        phase("kn_noH.fit", t0)
        info["kn_noH.fit_info"] = kn.fit_info
        del rest
        t0 = time.perf_counter()
        sc = SuffixClauses.from_kn(kn, backend="auto" if backend == "cupy" else "numpy")
        phase("kn_noH.suffix", t0)
        t0 = time.perf_counter()
        lnp_h, ks_h = suffix_gold(sc, held.tok, held.doc_start)
        phase("kn_noH.gold_H", t0)
        say(f"KN without H: {kn.n_types[1:]} types; H bits per token {float(-lnp_h.mean() / LN2):.4f}")
        if callback is not None:
            callback("kn_noH", kn=kn, suffix=sc, held=held)
        del kn, sc
        fi = FitInputs(tok=np.ascontiguousarray(held.tok).astype(np.uint16), doc_start=held.doc_start,
                       doc_index=held.doc_index, lnp_s=lnp_h, kstar=ks_h)
        # ---- document clauses on H, buckets, MAP-EM
        t0 = time.perf_counter()
        weights, trace = fit_weights(fi, comps, window, em_iters, prior)
        phase("em", t0)
        info["em_trace"] = trace
        if callback is not None:
            callback("em", weights=weights, fit_inputs=fi, trace=trace)
        # ---- KN-5 on all training documents
        t0 = time.perf_counter()
        kn_all = kn_gpu.fit_kn(tok_all, ds_all, **kw)
        phase("kn_all.fit", t0)
        info["kn_all.fit_info"] = kn_all.fit_info
        t0 = time.perf_counter()
        sc_all = SuffixClauses.from_kn(kn_all, backend="auto" if backend == "cupy" else "numpy")
        phase("kn_all.suffix", t0)
        say(f"KN on all training documents: {kn_all.n_types[1:]} types")
        if callback is not None:
            callback("kn_all", kn=kn_all, suffix=sc_all)
        info["total_s"] = round(time.perf_counter() - t_all, 3)
        config = {"vocab": int(vocab), "prefix_tokens": prefix_tokens, "window": window, "min_counts": mc,
                  "components": list(comps), "em_iters": int(em_iters), "prior": float(prior)}
        return cls(suffix=sc_all, weights=weights, config=config, fit_inputs=fi, kn=kn_all, fit_info=info)

    def variant(self, components: str | Sequence[str] | None = None, window: Any = "same") -> "CVTMCore":
        """The same suffix clauses with the EM refitted on the stored H inputs for other components
        and/or another window (window="same" keeps the current one)."""
        if self.fit_inputs is None:
            raise ValueError("this core has no stored fit inputs")
        comps = self.components if components is None else resolve_components(components)
        win = self.window if (isinstance(window, str) and window == "same") else window
        weights, trace = fit_weights(self.fit_inputs, comps, win, self.config["em_iters"], self.config["prior"])
        cfg = {**self.config, "components": list(comps), "window": win}
        return CVTMCore(suffix=self.suffix, weights=weights, config=cfg, fit_inputs=self.fit_inputs, kn=self.kn,
                        fit_info={"variant_of": self.state_hash(), "em_trace": trace})

    def with_suffix(self, suffix: SuffixClauses, kn: KNModel | None = None) -> "CVTMCore":
        """The same frozen weights over other suffix clauses (e.g. a KN fitted without H' for the T2
        inputs of docs/13 A4). Not the frozen model: its state hash differs."""
        return CVTMCore(suffix=suffix, weights=self.weights, config=self.config, fit_inputs=self.fit_inputs, kn=kn,
                        fit_info={"weights_of": self.state_hash()})

    # -------------------------------------------------------------- gold read-out and scoring
    def gold(self, tok: np.ndarray, doc_start: np.ndarray, lnp_s: np.ndarray | None = None,
             kstar: np.ndarray | None = None, feats: DocFeatures | None = None) -> GoldRecord:
        """Per-position read-out of a stream of whole documents (precomputed ln P_S, k* or features
        may be passed; they must come from this core's suffix clauses and window)."""
        tm: dict = {}
        tok64 = np.asarray(tok).astype(np.int64)
        ds = np.asarray(doc_start).astype(np.int64)
        t0 = time.perf_counter()
        if lnp_s is None or kstar is None:
            lnp_s, kstar = suffix_gold(self.suffix, tok64, ds)
        tm["suffix_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if feats is None:
            feats = doc_features(tok64, ds, orders=D_ORDERS, window=self.window, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
        elif feats.window != self.window or feats.orders != D_ORDERS or feats.ptr_min != PTR_MIN or feats.ptr_max != PTR_MAX:
            raise ValueError("document features were computed with other settings than this core's")
        tm["doc_features_s"] = time.perf_counter() - t0
        p_comp, act = gold_components(np.exp(lnp_s), feats, tok64)
        bucket, _ = bucket_ids(np.asarray(kstar).astype(np.int64), counts_matrix(feats), feats.ptr_len,
                               self.components, PTR_MIN)
        pg = gold_prob(p_comp, act, bucket, self.weights, self.components)
        return GoldRecord(tok=tok64, doc_start=ds, lnp_s=np.asarray(lnp_s), kstar=np.asarray(kstar).astype(np.int8),
                          feats=feats, bucket=bucket, p_comp=p_comp, act=act, p_gold=pg, timings=tm)

    def score_stream(self, tok: np.ndarray, doc_start: np.ndarray, n_bytes: int | None = None,
                     doc_bytes: np.ndarray | None = None, rec: GoldRecord | None = None) -> dict:
        """{"log2p", "bits", "bits_per_token", "bits_per_byte" (with n_bytes), "doc_bits", "record"}."""
        rec = rec or self.gold(tok, doc_start)
        log2p = rec.log2p
        n_docs = int(rec.doc_start.shape[0] - 1)
        doc_of = np.repeat(np.arange(n_docs, dtype=np.int64), np.diff(rec.doc_start))
        total_bits = float(-log2p.sum())
        out = {"log2p": log2p, "total_bits": total_bits, "n_tokens": int(log2p.shape[0]),
               "bits_per_token": total_bits / max(1, int(log2p.shape[0])),
               "doc_bits": np.bincount(doc_of, weights=-log2p, minlength=n_docs), "record": rec}
        if n_bytes is not None:
            out["n_bytes"] = int(n_bytes)
            out["bits_per_byte"] = total_bits / int(n_bytes)
        if doc_bytes is not None:
            out["doc_bytes"] = np.asarray(doc_bytes).astype(np.int64)
        return out

    def score(self, split: str = "validation", *, purpose: str | None = None,
              ev: protocol.EvalSplit | None = None) -> dict:
        """Score an evaluation split (validation; the test split only with a purpose, counted access):
        per-token log2 P, bits per byte (protocol CP), bits per document."""
        ev = ev or protocol.load_eval_split(split, vocab=self.vocab, purpose=purpose)
        return self.score_stream(ev.tok, ev.doc_start, n_bytes=ev.n_bytes, doc_bytes=ev.doc_bytes)

    # -------------------------------------------------------------- full distributions
    def dense_pass(self, rec: GoldRecord, k: int = 64, positions: np.ndarray | None = None, *, want_base: bool = True,
                   want_core: bool = True, chunk: int = 2048, sink: Callable[..., None] | None = None,
                   progress: Callable[[int, int], None] | None = None) -> dict:
        """Full next-token distributions of the positions of `rec` (default all), in chunks: top-k of
        P_S (want_base) and of the core (want_core). Results go to `sink(lo, hi, pos, base_ids,
        base_p, core_ids, core_p)` per chunk when given (lo, hi index into `positions`), else they are
        returned as arrays. Checks: every row sums to one, and the gold entries equal the gold pass."""
        V = self.vocab
        pos_all = np.arange(rec.tok.shape[0], dtype=np.int64) if positions is None else np.asarray(positions).astype(np.int64)
        if pos_all.shape[0] > 1 and np.any(np.diff(pos_all) <= 0):
            raise ValueError("positions must be strictly increasing")
        if pos_all.shape[0] and (int(pos_all[0]) < 0 or int(pos_all[-1]) >= rec.tok.shape[0]):
            raise ValueError("positions out of range")
        nq = int(pos_all.shape[0])
        keep = sink is None
        out: dict = {}
        if keep:
            if want_base:
                out["base_ids"] = np.empty((nq, k), dtype=np.uint16)
                out["base_p"] = np.empty((nq, k), dtype=np.float64)
            if want_core:
                out["core_ids"] = np.empty((nq, k), dtype=np.uint16)
                out["core_p"] = np.empty((nq, k), dtype=np.float64)
        chk = {"base_row_sum_max_abs_err": 0.0, "core_row_sum_max_abs_err": 0.0,
               "base_gold_max_abs_diff": 0.0, "core_gold_max_abs_diff": 0.0, "positions": nq}
        cm = component_mask(self.components)
        wmask = self.weights * cm[None, :]
        d_inc = [m for m in D_ORDERS if cm[1 + m]]
        ds = rec.doc_start
        doc_of_pos = np.searchsorted(ds, pos_all, side="right") - 1
        tok = rec.tok
        t0 = time.perf_counter()
        for lo in range(0, nq, chunk):
            hi = min(nq, lo + chunk)
            pos = pos_all[lo:hi]
            q = hi - lo
            ctx, valid = self.suffix.contexts(tok, ds, pos)
            lnrows = suffix_ln_rows(self.suffix, ctx, valid)
            P = np.exp(lnrows[:, :V])
            del lnrows
            g = tok[pos]
            r = np.arange(q)
            chk["base_gold_max_abs_diff"] = max(chk["base_gold_max_abs_diff"],
                                                float(np.max(np.abs(P[r, g] - np.exp(rec.lnp_s[pos])))))
            bi = bp = ci = cp = None
            if want_base:
                chk["base_row_sum_max_abs_err"] = max(chk["base_row_sum_max_abs_err"], float(np.max(np.abs(P.sum(axis=1) - 1.0))))
                bi, bp = topk_rows(P, k)
            if want_core:
                wr = wmask[rec.bucket[pos]]
                act = rec.act[pos] & cm[None, :]
                core = wr[:, 0:1] * P
                den = 0.0 + wr[:, 0]
                if d_inc:
                    d0, d1 = int(doc_of_pos[lo]), int(doc_of_pos[hi - 1])
                    a, b = int(ds[d0]), int(ds[d1 + 1])
                    cands = doc_candidates(tok[a:b], ds[d0:d1 + 2] - a, pos - a, orders=tuple(d_inc), window=self.window)
                for m in d_inc:
                    csr = cands[m]
                    lens = np.diff(csr.indptr)
                    rowrep = np.repeat(r, lens)
                    nm = rec.feats.count(m).astype(np.int64)[pos]
                    if not np.array_equal(lens > 0, act[:, 1 + m]):
                        raise RuntimeError(f"D{m}: candidate rows disagree with the firing mask")
                    pm = csr.cnt / nm[rowrep]
                    core[rowrep, csr.tok] = core[rowrep, csr.tok] + wr[rowrep, 1 + m] * pm
                    den = den + np.where(act[:, 1 + m], wr[:, 1 + m], 0.0)
                if cm[6]:
                    fp = np.flatnonzero(act[:, 6])
                    cand = rec.feats.ptr_cand[pos[fp]].astype(np.int64)
                    core[fp, cand] = core[fp, cand] + wr[fp, 6] * 1.0
                    den = den + np.where(act[:, 6], wr[:, 6], 0.0)
                core = core / den[:, None]
                chk["core_row_sum_max_abs_err"] = max(chk["core_row_sum_max_abs_err"], float(np.max(np.abs(core.sum(axis=1) - 1.0))))
                chk["core_gold_max_abs_diff"] = max(chk["core_gold_max_abs_diff"],
                                                    float(np.max(np.abs(core[r, g] - rec.p_gold[pos]))))
                ci, cp = topk_rows(core, k)
                del core
            del P
            if keep:
                if want_base:
                    out["base_ids"][lo:hi], out["base_p"][lo:hi] = bi, bp
                if want_core:
                    out["core_ids"][lo:hi], out["core_p"][lo:hi] = ci, cp
            else:
                sink(lo, hi, pos, bi, bp, ci, cp)
            if progress is not None:
                progress(hi, nq)
        chk["wall_s"] = time.perf_counter() - t0
        out["checks"] = chk
        return out

    def topk(self, tok: np.ndarray, doc_start: np.ndarray, positions: np.ndarray, k: int = 64,
             rec: GoldRecord | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(ids [Q, k], probabilities [Q, k]) of the core's full next-token distribution at `positions`
        (strictly increasing) of a stream of whole documents; descending, ties to the lower id."""
        rec = rec or self.gold(tok, doc_start)
        res = self.dense_pass(rec, k, positions, want_base=False, want_core=True)
        return res["core_ids"].astype(np.int64), res["core_p"]

    # -------------------------------------------------------------- explanation
    def explain(self, tok: np.ndarray, doc_start: np.ndarray, position: int,
                token_text: Callable[[int], str] | Sequence[str] | None = None,
                rec: GoldRecord | None = None) -> tuple[list[dict], str]:
        """The clauses behind P(tok[position]): the suffix clauses (`SuffixClauses.explain`), the
        firing document clauses with their exact counts, the pointer, the bucket and its weights."""
        tok64 = np.asarray(tok).astype(np.int64)
        ds = np.asarray(doc_start).astype(np.int64)
        p = int(position)
        if rec is None:
            d = int(np.searchsorted(ds, p, side="right")) - 1
            a, b = int(ds[d]), int(ds[d + 1])
            sub = self.gold(tok64[a:b], np.array([0, b - a], dtype=np.int64))
            i = p - a
        else:
            sub, i = rec, p
        w = int(sub.tok[i])

        def txt(t: int) -> str:
            if token_text is None:
                return str(t)
            return str(token_text(t)) if callable(token_text) else str(token_text[t])

        ctx, valid = self.suffix.contexts(sub.tok, sub.doc_start, np.array([i]))
        s_items, s_text = self.suffix.explain(ctx[0], w, token_text, int(valid[0]))
        items: list[dict] = [{"family": "S", "items": s_items, "p": float(np.exp(sub.lnp_s[i]))}]
        lines = ["S (suffix clauses, Kneser-Ney in vote form):", *("  " + ln for ln in s_text.splitlines())]
        wts = self.weights[int(sub.bucket[i])]
        k_star, m_star, cb, lb = self.layout.decode(int(sub.bucket[i]))
        lines.append(f"bucket {int(sub.bucket[i])}: k* = {k_star}, m* = {m_star}, count bucket = {cb}, pointer bucket = {lb}")
        num = den = 0.0
        for j, name in enumerate(COMPONENTS):
            if not sub.act[i, j] or name not in self.components:
                continue
            pj = float(sub.p_comp[i, j])
            num += float(wts[j]) * pj
            den += float(wts[j])
            if name.startswith("D"):
                m = int(name[1:])
                n_m = int(sub.feats.count(m)[i])
                g_m = int(sub.feats.gold_count(m)[i])
                ctx_m = [txt(int(t)) for t in sub.tok[max(0, i - m):i]] if m else []
                desc = (f"IF the {m}-token left context ({', '.join(ctx_m)}) occurred {n_m} times earlier in the document "
                        f"window THEN P_{name}({txt(w)}) = {g_m}/{n_m} = {pj:.4f}")
                items.append({"family": "D", "order": m, "n": n_m, "g": g_m, "p": pj, "weight": float(wts[j]), "text": desc})
            elif name == "P":
                L = int(sub.feats.ptr_len[i])
                cand = int(sub.feats.ptr_cand[i])
                desc = (f"IF the last {L} tokens repeat an earlier span of the document THEN copy its successor "
                        f"{txt(cand)} (P_P({txt(w)}) = {pj:.0f})")
                items.append({"family": "P", "length": L, "candidate": cand, "p": pj, "weight": float(wts[j]), "text": desc})
            else:
                desc = f"S: P_KN({txt(w)} | context) = {pj:.6g}"
            lines.append(f"  weight {float(wts[j]):.4f} x {desc}")
        total = float(sub.p_gold[i])
        lines.append(f"P({txt(w)}) = sum(weight x P_component) / sum(weight) = {num:.6g} / {den:.6g} = {total:.6g} "
                     f"({-math.log2(total):.4f} bits)")
        items.append({"family": "total", "p": total, "bits": -math.log2(total), "bucket": int(sub.bucket[i]),
                      "k_star": int(k_star), "m_star": m_star})
        return items, "\n".join(lines)

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> dict:
        """Write the model (module docstring); returns {"path", "sha256", "bytes", "state_hash"}."""
        if self.kn is None:
            raise ValueError("saving needs the KNModel of the suffix clauses (kn)")
        arrays, kn_header = kn_to_arrays(self.kn)
        arrays["weights"] = self.weights
        if self.fit_inputs is not None:
            fi = self.fit_inputs
            arrays.update({"fit.tok": np.asarray(fi.tok).astype(np.uint16), "fit.doc_start": np.asarray(fi.doc_start).astype(np.int64),
                           "fit.doc_index": np.asarray(fi.doc_index).astype(np.int64), "fit.lnp_s": np.asarray(fi.lnp_s, dtype=np.float64),
                           "fit.kstar": np.asarray(fi.kstar).astype(np.int8)})
        meta = {"format": FORMAT, "config": self.config, "identity": self.identity(), "kn": kn_header,
                "state_hash": self.state_hash(), "fit_info": _jsonable(self.fit_info)}
        arrays["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
        path = Path(path)
        np.savez(path, **arrays)
        return {"path": str(path), "sha256": hashing.sha256_file(path), "bytes": int(path.stat().st_size),
                "state_hash": meta["state_hash"]}

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto", keep_kn: bool = True) -> "CVTMCore":
        """Rebuild a saved model; raises ValueError when any KN array hash or the state hash differs."""
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
            if meta.get("format") != FORMAT:
                raise ValueError(f"not a {FORMAT} file")
            kn = kn_from_arrays({k: z[k] for k in z.files if k.startswith("kn.")}, meta["kn"], verify=True)
            weights = z["weights"]
            fi = None
            if "fit.tok" in z.files:
                fi = FitInputs(tok=z["fit.tok"], doc_start=z["fit.doc_start"], doc_index=z["fit.doc_index"],
                               lnp_s=z["fit.lnp_s"], kstar=z["fit.kstar"])
        cfg = dict(meta["config"])
        if cfg.get("min_counts") is not None:
            cfg["min_counts"] = {int(k): int(v) for k, v in cfg["min_counts"].items()}
        sc = SuffixClauses.from_kn(kn, backend=backend)
        core = cls(suffix=sc, weights=weights, config=cfg, fit_inputs=fi, kn=kn if keep_kn else None,
                   fit_info=meta.get("fit_info", {}))
        if core.state_hash() != meta["state_hash"]:
            raise ValueError(f"state hash mismatch: stored {meta['state_hash']}, rebuilt {core.state_hash()}")
        return core


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
