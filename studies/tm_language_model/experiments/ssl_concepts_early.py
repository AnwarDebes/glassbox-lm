"""EARLY CONCEPT TEST (EXPLORATORY, docs/12 F23; not pre-registered): do the self-supervised,
Tsetlin-automata-learned clauses of the causal V 8000 bank encode word CONCEPTS (challenge C1)?

Word representations are derived from frozen banks without any label (src/tmlm/ssl/concepts.py has the
definitions): the count-vote profile (how often each clause fires when the token is the target, the
counts of the T1 read-out over the counting positions of the first 20M training tokens), the literal
profiles (in which clauses, and at which segment, the token is an included literal), each with the
weightings raw / ppmi / vote and the reductions full / svdK_pP. The same for the YOUNG bank (0.079 epoch
of TA feedback, the untrained control) and optionally the epoch-1 checkpoint. Baselines on the same
tokens: PPMI-SVD (and the same grid) of co-occurrence counts, the in-house CBOW embeddings, the exchange
cluster codes, a frequency control.

Evaluation: SimLex-999, WordSim-353 and MEN (dev split only) restricted to pairs of single BPE-8k tokens
(coverage reported), Spearman with pair-bootstrap intervals, the early verdict rule,
WordNet purity (supersense agreement of nearest neighbours, synonym hits, k-means purity), and
printouts: nearest neighbours, clusters, 20 clauses as readable rules with the tokens they group.

Phases (each its own logged run; see experiments/cvtm/configs/ssl_concepts_early.json):
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1
    scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py -- \\
        .venv-tm/bin/python scripts/run_logged.py --name ssl_concepts_bank_trained_exploratory \\
        --config experiments/cvtm/configs/ssl_concepts_early.json -- \\
        .venv-tm/bin/python -u experiments/ssl_concepts_early.py --phase bank --bank trained
    ... --phase bank --bank young; --phase baselines; --phase evaluate --inputs <run dirs>
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from engine_common import ROOT  # noqa: E402  (puts src/ on the path)
from tmlm.ssl import concepts as K  # noqa: E402

GIB = float(1 << 30)
EPOCH = 4  # forward-pass epoch argument, as the T1 read-out (causal mode: nothing is hidden, the value is inert)


# ------------------------------------------------------------------ bookkeeping
class Rec:
    """Labelled metrics (measured / derived / unverified), every entry marked exploratory."""

    def __init__(self, out: Path, smoke: bool) -> None:
        self.out, self.smoke = out, smoke
        self.metrics: dict = {}
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        print(f"[{time.time() - self.t0:8.1f} s] {msg}", flush=True)

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        e = {"value": _plain(value), "label": label, "exploratory": True, **{k: _plain(v) for k, v in extra.items()}}
        if self.smoke:
            e["smoke"] = True
        self.metrics[name] = e

    def flush(self) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            from tmlm.common import runlog
            runlog.write_metrics(self.metrics, on_conflict="replace")
        else:
            (self.out / "metrics_unlogged.json").write_text(json.dumps(self.metrics, indent=1, default=str) + "\n")

    def dataset(self, name: str, digest: str) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            from tmlm.common import runlog
            runlog.write_dataset_hashes({name: digest}, on_conflict="replace")

    def rss(self) -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB


def _plain(v):
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def out_dir(args) -> Path:
    d = os.environ.get("TMLM_RUN_DIR")
    if d:
        return Path(d)
    if not args.out_dir:
        raise SystemExit("not under run_logged.py: give --out-dir (a scratch directory)")
    p = Path(args.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ shared evaluation setup (no scores are read here except the pair lists)
class Setup:
    def __init__(self, cfg: dict, rec: Rec, need_wordnet: bool = True) -> None:
        from tmlm.data.bpe import load_tokenizer, token_bytes, tokenizer_sha256
        V = int(cfg["vocab"])
        self.V = V
        self.tok = load_tokenizer(V)
        rec.dataset(f"tokenizer_bpe{V}", tokenizer_sha256(V))
        enc = lambda s: self.tok.encode(s).ids  # noqa: E731
        self.texts = K.token_texts(token_bytes(self.tok))
        self.sets: dict[str, dict] = {}
        parsers = {"simlex": K.parse_simlex, "ws353": K.parse_ws353, "men_lemma": K.parse_men_lemma}
        for name, d in cfg["datasets"].items():
            path = ROOT / d["path"]
            if "test" in path.name.lower():
                raise SystemExit("a test split may not be read")
            rec.dataset(f"{name}:{d['path']}", K.sha256_file(path))
            pairs = parsers[d["parser"]](path)
            self.sets[name] = K.map_pairs(pairs, enc)
            self.sets[name]["pairs_text"] = [(p[0], p[1]) for p, c in zip(pairs, self.sets[name]["covered"]) if c]
        ab = np.concatenate([np.stack([np.minimum(s["a"], s["b"]), np.maximum(s["a"], s["b"])], 1) for s in self.sets.values()])
        self.union = np.unique(ab, axis=0)
        key = self.union[:, 0] * V + self.union[:, 1]
        for s in self.sets.values():
            k = np.minimum(s["a"], s["b"]) * V + np.maximum(s["a"], s["b"])
            s["uidx"] = np.searchsorted(key, k)
            assert np.array_equal(key[s["uidx"]], k)
        self.word_like = K.word_like_mask(self.texts)
        self.eval_tokens = np.unique(self.union)
        self.probe_words = [w for w in cfg["probe_words"] if K.single_token_id(enc, w, False) is not None]
        self.probe_ids = np.array([K.single_token_id(enc, w, False) for w in self.probe_words], np.int64)
        self.labels: dict[int, tuple] = {}
        if need_wordnet:
            import nltk
            nltk.data.path.insert(0, str(ROOT / cfg["wordnet"]["nltk_data"]))
            from nltk.corpus import wordnet as wn
            rec.dataset("wordnet.zip", K.sha256_file(ROOT / cfg["wordnet"]["nltk_data"] / "corpora" / "wordnet.zip"))
            for t in np.flatnonzero(self.word_like).tolist():
                ss = wn.synsets(self.texts[t][1:])
                if ss:
                    self.labels[t] = (ss[0].lexname(), ss[0].pos(), frozenset(s.name() for s in ss))
        self.eligible = np.array(sorted(self.labels), np.int64)
        self.word_ids = np.flatnonzero(self.word_like).astype(np.int64)

    def coverage(self, rec: Rec) -> None:
        for name, s in self.sets.items():
            rec.m(f"coverage.{name}", {"covered_pairs": s["n_covered"], "total_pairs": s["n_total"], "fraction": s["n_covered"] / s["n_total"]})
        rec.m("coverage.union_pairs", int(self.union.shape[0]))
        rec.m("coverage.word_like_tokens", int(self.word_like.sum()))
        rec.m("coverage.wordnet_eligible_tokens", int(self.eligible.shape[0]))
        rec.m("coverage.probe_words", self.probe_words)


class Collector:
    """Per variant: similarities of the union pairs, neighbour lists (eligible queries among eligible
    candidates; probe words among word-like tokens); primary embeddings kept separately."""

    def __init__(self, setup: Setup) -> None:
        self.s = setup
        self.names: list[str] = []
        self.sims: list[np.ndarray] = []
        self.nn_elig: list[np.ndarray] = []
        self.nn_probe: list[np.ndarray] = []
        self.emb: dict[str, np.ndarray] = {}

    def add(self, xp, name: str, S) -> None:
        s = self.s
        a = xp.asarray(s.union[:, 0])
        b = xp.asarray(s.union[:, 1])
        sims = S[a, b]
        self.sims.append(np.asarray(sims.get() if hasattr(sims, "get") else sims, np.float64))
        self.nn_elig.append(self._topk(xp, S, s.eligible, s.eligible, 10))
        self.nn_probe.append(self._topk(xp, S, s.probe_ids, s.word_ids, 10))
        self.names.append(name)

    @staticmethod
    def _topk(xp, S, queries, candidates, k):
        q = xp.asarray(queries)
        c = xp.asarray(candidates)
        sub = S[q][:, c].astype(xp.float64)
        sub[q[:, None] == c[None, :]] = -xp.inf
        order = xp.argsort(-sub, axis=1)[:, :k]
        res = c[order]
        return np.asarray(res.get() if hasattr(res, "get") else res, np.int32)

    def save(self, path: Path, extra: dict | None = None) -> None:
        np.savez(path, names=np.array(self.names), pair_a=self.s.union[:, 0], pair_b=self.s.union[:, 1],
                 sims=np.stack(self.sims), nn_elig=np.stack(self.nn_elig), nn_probe=np.stack(self.nn_probe),
                 eligible=self.s.eligible, probe_ids=self.s.probe_ids, word_ids=self.s.word_ids,
                 **{f"emb__{k}": v for k, v in self.emb.items()}, **(extra or {}))


def variants_from_gram(xp, col: Collector, prefix: str, G, cfg: dict, rec: Rec) -> None:
    """All reductions of one weighted Gram matrix (G on the device, float64)."""
    reds = cfg["representations"]["reductions"]
    t0 = time.time()
    S = K.cosine_from_gram(xp, G)
    col.add(xp, f"{prefix}.full", S)
    del S
    kmax = max(K.parse_reduction(r)[0] for r in reds if r != "full")
    w, U = K.eig_desc(xp, G, kmax)
    for r in reds:
        if r == "full":
            continue
        k, p = K.parse_reduction(r)
        E = K.spectral_embedding(xp, w, U, k, p)
        col.add(xp, f"{prefix}.{r}", K.cosine_rows(xp, E))
        prim = cfg["representations"]["primary"]
        if prefix.split(".")[-1] == prim["scheme"] and r == prim["reduction"]:
            col.emb[prefix] = np.asarray(E.get() if hasattr(E, "get") else E, np.float32)
    top = w[:10]
    rec.m(f"spectrum.{prefix}.top10_eigenvalues", np.asarray(top.get() if hasattr(top, "get") else top), label="derived")
    rec.log(f"  variants of {prefix}: {time.time() - t0:.1f} s")


# ------------------------------------------------------------------ corpus prefix
def load_prefix(cfg: dict, rec: Rec, n_tokens: int):
    from tmlm.data.corpus_store import load_split, split_dir
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import contiguous
    from cvtm_t1_readout import fit_document_mask
    V = int(cfg["vocab"])
    full = load_split("wikitext103", V, "train")
    tr = contiguous(slice_train(full, n_tokens))
    del full
    meta = json.loads((split_dir("wikitext103", V, "train") / "meta.json").read_text())
    rec.dataset(f"train_tok_V{V}", meta["sha256"]["tok.npy"])
    tok = np.asarray(tr.tok).astype(np.int64)
    ds = np.asarray(tr.doc_start).astype(np.int64)
    n_doc = ds.shape[0] - 1
    fit_doc = fit_document_mask(n_doc)
    is_fit = fit_doc[np.repeat(np.arange(n_doc), np.diff(ds))]
    rec.m("data.prefix_tokens", int(tok.shape[0]))
    rec.m("data.prefix_docs", int(n_doc))
    rec.m("data.counting_positions", int((~is_fit).sum()))
    return tr, tok, ds, is_fit


# ------------------------------------------------------------------ phase: bank
def count_pass(cfg: dict, ecfg, st, tr, tok, is_fit, rec: Rec, cp):
    """Exact (clause, target) counts over the counting positions (the T1 read-out's counting pass)."""
    from tmlm.common import locks
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    from cvtm_t1_readout import CompactCountAccumulator
    V, C = int(ecfg.vocab), int(ecfg.n_clauses)
    comp = cfg["compute"]
    eng = CudaEngine(ecfg, tr)
    eng.load_state(st)
    n = int(tok.shape[0])
    tok_dev = cp.asarray(tok.astype(np.uint64))
    isfit_dev = cp.asarray(is_fit.astype(np.uint8))
    cap = int(comp["count_buffer_events"])
    buf = cp.empty(cap, dtype=cp.uint64)
    nbuf = 0
    accum = CompactCountAccumulator(C, V, range_bits=int(comp["count_range_bits"]))
    span = np.uint64(accum.span)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 1004, int(cfg["seed"])]))
    chk = np.sort(rng.choice(n, size=min(40, n), replace=False))
    got: dict[int, np.ndarray] = {}
    n_events = 0
    t0 = time.time()
    flush_s = 0.0

    def flush(nb: int) -> float:
        t = time.time()
        k = buf[:nb]
        k.sort()
        brk = cp.flatnonzero(k[1:] != k[:-1]) + 1
        starts = cp.concatenate([cp.zeros(1, dtype=brk.dtype), brk])
        uk = k[starts]
        cnt = cp.diff(cp.concatenate([starts, cp.asarray([nb], dtype=starts.dtype)])).astype(cp.int32)
        bounds = cp.searchsorted(uk, cp.arange(accum.n_ranges + 1, dtype=cp.uint64) * cp.uint64(span)).get()
        for r in range(accum.n_ranges):
            a, b = int(bounds[r]), int(bounds[r + 1])
            if a < b:
                accum.add_range(r, (uk[a:b] - cp.uint64(r * int(span))).astype(cp.uint32).get(), cnt[a:b].get())
        accum.batches += 1
        return time.time() - t

    chunk = int(comp["forward_chunk"])
    for ci, lo in enumerate(range(0, n, chunk)):
        hi = min(n, lo + chunk)
        eng._sync_state_struct()
        fw = eng._forward(EPOCH, cp.arange(lo, hi, dtype=cp.uint32), None, 0)
        eng._marks = []
        fe, fj = fw.fired_ex, fw.fired_j
        del fw
        p = fe.astype(cp.int64) + lo
        keep = isfit_dev[p] == 0
        key = fj[keep].astype(cp.uint64) * cp.uint64(V) + tok_dev[p[keep]]
        m = int(key.size)
        if m > cap:
            raise RuntimeError("count buffer smaller than one forward chunk")
        if nbuf + m > cap:
            flush_s += flush(nbuf)
            nbuf = 0
        buf[nbuf:nbuf + m] = key
        nbuf += m
        n_events += m
        a, b = np.searchsorted(chk, [lo, hi])
        if b > a:
            feh, fjh = fe.get(), fj.get()
            for r in chk[a:b].tolist():
                sel = feh == (r - lo)
                got[r] = np.sort(fjh[sel]).astype(np.uint32)
        del fe, fj, p, keep, key
        if (ci + 1) % 200 == 0:
            locks.yield_gpu_if_requested()
            rec.log(f"  count pass {hi}/{n} positions, {hi / (time.time() - t0):.0f} pos/s, events {n_events}, pairs {accum.n_pairs}, flush {flush_s:.0f} s, rss {rec.rss():.1f} GiB")
    if nbuf:
        flush_s += flush(nbuf)
    eng.synchronize()
    bad = sum(int(not np.array_equal(eng.fired_clauses(EPOCH, r), got[r])) for r in chk.tolist())
    del eng, tok_dev, isfit_dev, buf
    cp.get_default_memory_pool().free_all_blocks()
    rec.m("counts.events", n_events, note="(clause, target) events over the counting positions")
    rec.m("counts.distinct_pairs", accum.n_pairs)
    rec.m("check.fired_list_mismatches", bad, note=f"{chk.shape[0]} random prefix positions vs the engine's public fired_clauses")
    rec.m("throughput.count_pass_s", time.time() - t0)
    rec.m("throughput.count_flush_s", flush_s)
    if bad or accum.events != n_events:
        raise SystemExit("PIPELINE CHECK FAILED: fired lists or event totals differ")
    return accum, n_events


def run_bank(cfg: dict, args, rec: Rec, out: Path) -> int:
    import cupy as cp
    from tmlm.engine.checkpoint import load_checkpoint
    cp.get_default_memory_pool().set_limit(size=int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB))
    bank = cfg["banks"][args.bank]
    setup = Setup(cfg, rec)
    setup.coverage(rec)
    ecfg, st, meta = load_checkpoint(ROOT / bank["checkpoint"])
    hm = json.loads((ROOT / bank["hash_metrics"]).read_text())
    expected = hm[bank["hash_key"]]["value"]["state_hash"]
    if meta["state_hash"] != expected:
        raise SystemExit(f"checkpoint hash {meta['state_hash']} differs from the logged {expected}")
    rec.m("bank.name", args.bank)
    rec.m("bank.checkpoint", bank["checkpoint"])
    rec.m("bank.state_hash", meta["state_hash"], source=bank["hash_metrics"])
    V, C = int(ecfg.vocab), int(ecfg.n_clauses)
    assert V == setup.V and ecfg.mode == "causal"
    alive = st["alive"].astype(bool)
    rec.m("bank.alive_clauses", int(alive.sum()))
    n_tokens = int(args.prefix_tokens or cfg["data"]["train_prefix_tokens"])
    if args.prefix_tokens and not args.smoke:
        raise SystemExit("--prefix-tokens is a smoke-only override")
    tr, tok, ds, is_fit = load_prefix(cfg, rec, n_tokens)
    rec.log(f"bank {args.bank}: {int(alive.sum())} alive clauses; prefix {tok.shape[0]} tokens")
    rec.flush()

    accum, n_events = count_pass(cfg, ecfg, st, tr, tok, is_fit, rec, cp)
    exp = bank.get("expect_counts")
    if exp and not args.smoke:
        ok = n_events == int(exp["events"]) and accum.n_pairs == int(exp["distinct_pairs"])
        rec.m("check.counts_equal_logged_T1", ok, expected={"events": exp["events"], "distinct_pairs": exp["distinct_pairs"]}, source=exp["source"])
        if not ok:
            rec.flush()
            raise SystemExit("PIPELINE CHECK FAILED: counts differ from the logged T1 counts")
    rec.log(f"counts: {n_events} events, {accum.n_pairs} pairs, rss {rec.rss():.1f} GiB")

    # marginals of the count profile
    S_cl = accum.span_clauses
    p_row = K.add_one_unigram(tok[~is_fit], V)
    row_sum = np.zeros(V)
    col_sum = np.zeros(C)
    col_d = np.zeros(C, np.int64)
    for r in range(accum.n_ranges):
        ptr, cls, val = accum._range_arrays(r)
        n_real = ptr.shape[0] - 1
        row_sum += np.bincount(cls.astype(np.int64), weights=val, minlength=V)
        clause = np.repeat(np.arange(n_real, dtype=np.int64), np.diff(ptr))
        col_sum[r * S_cl:r * S_cl + n_real] = np.bincount(clause, weights=val, minlength=n_real)
        col_d[r * S_cl:r * S_cl + n_real] = np.diff(ptr)
    marg = K.Marginals(row_sum, col_sum, col_d, p_row)
    assert int(col_sum.sum()) == n_events
    rec.m("counts.clauses_with_counts", int((col_sum > 0).sum()))
    rec.m("counts.eval_tokens_with_empty_count_profile", int((row_sum[setup.eval_tokens] == 0).sum()), note="tokens of the covered pairs that were never a counted target")

    # clause printouts: choose the clauses before the gram pass (their count rows are read from the ranges)
    clause_sel = select_clauses(cfg, st, col_sum, col_d, accum, setup, rec)

    # Gram matrices of the count profile, block by block on the GPU
    schemes = cfg["representations"]["schemes"]
    md = marg.to(cp)
    grams = {s: cp.zeros((V, V), dtype=cp.float64) for s in schemes}
    blk = int(cfg["compute"]["gram_block_clauses"])
    t0 = time.time()
    for r in range(accum.n_ranges):
        ptr, cls, val = accum._range_arrays(r)
        n_real = ptr.shape[0] - 1
        for c0 in range(0, n_real, blk):
            c1 = min(n_real, c0 + blk)
            a, b = int(ptr[c0]), int(ptr[c1])
            if a == b:
                continue
            d_rows = cp.asarray(cls[a:b].astype(np.int64))
            d_loc = cp.asarray(np.repeat(np.arange(c1 - c0, dtype=np.int64), np.diff(ptr[c0:c1 + 1])))
            d_val = cp.asarray(val[a:b])
            gcols = d_loc + (r * S_cl + c0)
            for s in schemes:
                w = K.weight_values(cp, d_rows, gcols, d_val, md, s)
                K.gram_add_dense(cp, grams[s], d_rows, d_loc, w, V, c1 - c0)
                del w
            del d_rows, d_loc, d_val, gcols
        accum.tab[r] = None  # release the host range
    cp.cuda.Stream.null.synchronize()
    rec.m("throughput.count_gram_s", time.time() - t0)
    rec.log(f"count grams: {time.time() - t0:.1f} s, gpu pool {cp.get_default_memory_pool().total_bytes() / GIB:.2f} GiB")
    del accum

    col = Collector(setup)
    for s in schemes:
        variants_from_gram(cp, col, f"{args.bank}.count.{s}", grams[s], cfg, rec)
        grams[s] = None
    cp.get_default_memory_pool().free_all_blocks()

    # literal profiles (CPU sparse Gram, variants on the GPU)
    for prof, ob in (("lit", False), ("litoff", True)):
        t0 = time.time()
        rows, cols, n_cols = K.literal_profile(st["lit_id"], st["lit_state"], st["alive"], V, ob)
        ones = np.ones(rows.shape[0])
        lm = K.marginals_from_coo(rows, cols, ones, V, n_cols, p_row)
        rec.m(f"literal.{prof}.nonzeros", int(rows.shape[0]))
        rec.m(f"literal.{prof}.tokens_with_any_literal", int((lm.row_sum > 0).sum()))
        rec.m(f"literal.{prof}.eval_tokens_with_empty_profile", int((lm.row_sum[setup.eval_tokens] == 0).sum()))
        mdl = lm.to(np)
        for s in schemes:
            w = K.weight_values(np, rows, cols, ones, mdl, s)
            G = cp.asarray(K.gram_sparse(rows, cols, w, V, n_cols))
            off = G - cp.diag(cp.diag(G))
            rec.m(f"literal.{prof}.{s}.offdiagonal_nonzero_fraction", float((off != 0).sum()) / (V * (V - 1)), label="derived")
            del off
            variants_from_gram(cp, col, f"{args.bank}.{prof}.{s}", G, cfg, rec)
            del G
        del rows, cols, ones, lm, mdl
        rec.log(f"literal profile {prof}: {time.time() - t0:.1f} s")
        cp.get_default_memory_pool().free_all_blocks()

    col.save(out / f"reps_{args.bank}.npz")
    (out / f"clauses_{args.bank}.json").write_text(json.dumps(_plain(clause_sel), indent=1) + "\n")
    rec.m("output.reps", f"reps_{args.bank}.npz", n_variants=len(col.names))
    rec.m("memory.peak_rss_gib", rec.rss())
    rec.m("memory.gpu_pool_total_gib_at_end", cp.get_default_memory_pool().total_bytes() / GIB)
    rec.flush()
    rec.log(f"bank {args.bank} done: {len(col.names)} variants")
    return 0


def select_clauses(cfg: dict, st: dict, N: np.ndarray, D: np.ndarray, accum, setup: Setup, rec: Rec) -> list[dict]:
    """20 clauses as readable rules: 10 drawn at random among clauses with N >= 1000 (representative) and
    10 chosen by a label-free rule (lowest target entropy among clauses with N >= 1000 whose counted
    targets are at least 90 percent word-like tokens and whose top target holds at most half; selected
    for readability, NOT representative)."""
    V = setup.V
    S_cl = accum.span_clauses
    n_print = int(cfg["eval"]["print_clauses"])
    wl = setup.word_like.astype(np.float64)
    H = np.full(N.shape[0], np.inf)
    wshare = np.zeros(N.shape[0])
    top1 = np.zeros(N.shape[0])
    for r in range(accum.n_ranges):
        ptr, cls, val = accum._range_arrays(r)
        n_real = ptr.shape[0] - 1
        clause = np.repeat(np.arange(n_real, dtype=np.int64), np.diff(ptr))
        v = val.astype(np.float64)
        n = np.bincount(clause, weights=v, minlength=n_real)
        vlogv = np.bincount(clause, weights=v * np.log(np.maximum(v, 1.0)), minlength=n_real)
        ok = n > 0
        g = slice(r * S_cl, r * S_cl + n_real)
        Hr = np.full(n_real, np.inf)
        Hr[ok] = np.log(n[ok]) - vlogv[ok] / n[ok]
        H[g] = Hr
        wshare[g] = np.where(ok, np.bincount(clause, weights=v * wl[cls], minlength=n_real) / np.maximum(n, 1), 0)
        nz = np.diff(ptr) > 0
        mx = np.zeros(n_real)
        mx[nz] = np.maximum.reduceat(v, ptr[:-1][nz])
        top1[g] = np.where(ok, mx / np.maximum(n, 1), 0)
    pool = np.flatnonzero(N >= 1000)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 1005, int(cfg["seed"])]))
    rand = np.sort(rng.choice(pool, size=min(n_print // 2, pool.shape[0]), replace=False))
    cand = pool[(wshare[pool] >= 0.9) & (top1[pool] <= 0.5)]
    chosen = cand[np.argsort(H[cand], kind="stable")[: n_print - rand.shape[0]]]
    rec.m("clauses.selection", {"random_pool_N_ge_1000": int(pool.shape[0]), "readable_pool": int(cand.shape[0])},
          rule=select_clauses.__doc__.replace("\n", " "))
    from tmlm.data.corpus_store import split_dir
    freq = np.load(split_dir("wikitext103", V, "train") / "sym_freq.npy")[:V].astype(np.int64)
    km = np.load(split_dir("wikitext103", V, "train") / "kind_map.npy")
    from tmlm.engine.config import kind_of

    def members(sym: int) -> list[str]:
        kind = kind_of(V, sym)
        ids = np.flatnonzero(km[kind] == sym)
        ids = ids[np.argsort(-freq[ids], kind="stable")][:6]
        return [setup.texts[i] for i in ids]

    tt = lambda i: setup.texts[int(i)]  # noqa: E731
    out = []
    for kind_sel, js in (("random", rand), ("readable_low_entropy", chosen)):
        for j in js.tolist():
            r, loc = divmod(j, S_cl)
            ptr, cls, val = accum._range_arrays(r)
            a, b = int(ptr[loc]), int(ptr[loc + 1])
            c, v = cls[a:b].astype(np.int64), val[a:b].astype(np.int64)
            o = np.argsort(-v, kind="stable")[:8]
            vc, vw = st["vote_cls"][j], st["vote_w"][j]
            used = (vc != 0xFFFF) & (vw > 0)
            ov = np.argsort(-vw[used].astype(np.int64), kind="stable")[:8]
            out.append({
                "selection": kind_sel, "clause": int(j),
                "rule": K.clause_rule(st["lit_id"][j], st["lit_state"][j], V, tt, members),
                "N_counted": int(N[j]), "D_counted": int(D[j]), "target_entropy_nats": float(H[j]),
                "top_counted_targets": [(tt(c[i]), int(v[i]), round(float(v[i]) / max(1, int(N[j])), 4)) for i in o],
                "top_positive_votes": [(tt(x), int(w)) for x, w in zip(vc[used][ov], vw[used][ov])],
            })
    return out


# ------------------------------------------------------------------ phase: baselines
def run_baselines(cfg: dict, args, rec: Rec, out: Path) -> int:
    import cupy as cp
    from tmlm.data.corpus_store import load_split, split_dir
    cp.get_default_memory_pool().set_limit(size=int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB))
    setup = Setup(cfg, rec)
    setup.coverage(rec)
    V = setup.V
    col = Collector(setup)
    n_tokens = int(args.prefix_tokens or cfg["data"]["train_prefix_tokens"])
    if args.prefix_tokens and not args.smoke:
        raise SystemExit("--prefix-tokens is a smoke-only override")
    tr, tok, ds, is_fit = load_prefix(cfg, rec, n_tokens)
    p_row = K.add_one_unigram(tok[~is_fit], V)
    schemes = cfg["representations"]["schemes"]
    fam_specs = [(f"cooc{int(cfg['data']['train_prefix_tokens']) // 1_000_000}M_{w}", offs, "prefix") for w, offs in cfg["baselines"]["cooc_prefix_windows"].items()]
    if not args.smoke:
        fam_specs += [(f"cooctrain_{w}", offs, "full") for w, offs in cfg["baselines"]["cooc_full_train_windows"].items()]
    for fam, offs, which in fam_specs:
        t0 = time.time()
        if which == "prefix":
            X = K.cooc_counts(tok, ds, V, offs, target_mask=~is_fit)
            prow = p_row
        else:
            full = load_split("wikitext103", V, "train")
            ftok = np.asarray(full.tok)
            X = K.cooc_counts(ftok, np.asarray(full.doc_start), V, offs)
            prow = K.add_one_unigram(ftok, V)
            rec.m("data.full_train_tokens", int(ftok.shape[0]))
            del full, ftok
        rows, cols = np.nonzero(X)
        vals = X[rows, cols]
        rec.m(f"cooc.{fam}.events", int(X.sum()))
        rec.m(f"cooc.{fam}.nonzeros", int(rows.shape[0]))
        m = K.marginals_from_coo(rows, cols, vals, V, V, prow)
        md = m.to(cp)
        d_rows, d_cols, d_vals = cp.asarray(rows), cp.asarray(cols), cp.asarray(vals)
        for s in schemes:
            G = cp.zeros((V, V), dtype=cp.float64)
            K.gram_add_dense(cp, G, d_rows, d_cols, K.weight_values(cp, d_rows, d_cols, d_vals, md, s), V, V)
            variants_from_gram(cp, col, f"{fam}.cooc.{s}", G, cfg, rec)
            del G
        del X, rows, cols, vals, d_rows, d_cols, d_vals
        cp.get_default_memory_pool().free_all_blocks()
        rec.log(f"{fam}: {time.time() - t0:.1f} s")
    # CBOW
    cb = cfg["baselines"]["cbow"]
    w_in = np.load(ROOT / cb["dir"] / "cbow_w_in.npy")
    w_out = np.load(ROOT / cb["dir"] / "cbow_w_out.npy")
    rec.dataset("cbow_w_in.npy", K.sha256_file(ROOT / cb["dir"] / "cbow_w_in.npy"))
    rec.dataset("cbow_w_out.npy", K.sha256_file(ROOT / cb["dir"] / "cbow_w_out.npy"))
    for name, E in (("cbow.in.-.full", w_in), ("cbow.in+out.-.full", w_in + w_out)):
        col.add(cp, name, K.cosine_rows(cp, cp.asarray(E.astype(np.float32))))
    col.emb["cbow.in"] = w_in.astype(np.float32)
    # exchange codes and frequency control
    km = np.load(split_dir("wikitext103", V, "train") / "kind_map.npy").astype(np.int64)
    dkm = cp.asarray(km)
    S = sum((dkm[k][:, None] == dkm[k][None, :]).astype(cp.float32) for k in (1, 2, 3))
    col.add(cp, "exchange.levels.-.full", S)
    del S
    freq = np.load(split_dir("wikitext103", V, "train") / "sym_freq.npy")[:V].astype(np.float64)
    lf = cp.asarray(np.log(freq + 1.0))
    col.add(cp, "freq.logdiff.-.full", (-cp.abs(lf[:, None] - lf[None, :])).astype(cp.float32))
    col.save(out / "reps_baselines.npz")
    rec.m("output.reps", "reps_baselines.npz", n_variants=len(col.names))
    rec.m("memory.peak_rss_gib", rec.rss())
    rec.flush()
    return 0


# ------------------------------------------------------------------ phase: evaluate
def run_evaluate(cfg: dict, args, rec: Rec, out: Path) -> int:
    setup = Setup(cfg, rec)
    setup.coverage(rec)
    V = setup.V
    names, sims, nn_elig, nn_probe, embs, clauses = [], [], [], [], {}, {}
    for d in args.inputs:
        d = Path(d)
        for f in sorted(d.glob("reps_*.npz")):
            z = np.load(f)
            if not (np.array_equal(z["pair_a"], setup.union[:, 0]) and np.array_equal(z["pair_b"], setup.union[:, 1])
                    and np.array_equal(z["eligible"], setup.eligible) and np.array_equal(z["probe_ids"], setup.probe_ids)):
                raise SystemExit(f"{f}: pair list, eligible set or probe ids differ from this setup")
            names += [str(x) for x in z["names"]]
            sims.append(z["sims"])
            nn_elig.append(z["nn_elig"])
            nn_probe.append(z["nn_probe"])
            for k in z.files:
                if k.startswith("emb__"):
                    embs[k[5:]] = z[k]
            rec.m(f"inputs.{f.name}", str(f), n_variants=int(z["sims"].shape[0]))
        for f in sorted(d.glob("clauses_*.json")):
            clauses[f.stem[len("clauses_"):]] = json.loads(f.read_text())
    if len(set(names)) != len(names):
        raise SystemExit("duplicate variant names across inputs")
    sims = np.concatenate(sims)
    nn_elig = np.concatenate(nn_elig)
    nn_probe = np.concatenate(nn_probe)
    idx_of = {n: i for i, n in enumerate(names)}
    fam = lambda n: n.split(".")[0]  # noqa: E731
    nb = int(cfg["eval"]["bootstrap_resamples"])
    seed = int(cfg["seed"])

    # ---- Spearman per dataset and variant
    table: dict[str, dict[str, dict]] = {}
    boots: dict[str, np.ndarray] = {}
    for dname, s in setup.sets.items():
        gold = s["gold"]
        idx = K.bootstrap_indices(gold.shape[0], nb, seed)
        table[dname] = {}
        for n in names:
            pred = sims[idx_of[n]][s["uidx"]]
            rho = K.spearman(gold, pred)
            bs = K.spearman_boot(gold, pred, idx)
            boots[f"{dname}|{n}"] = bs
            table[dname][n] = {"rho": rho, "ci95": K.ci95(bs)}
        rec.m(f"spearman.{dname}", {n: table[dname][n] for n in names}, label="derived",
              note=f"Spearman over {gold.shape[0]} covered pairs; 95 percent percentile interval of {nb} pair-bootstrap resamples (seed {seed})")

    def best(dname: str, pred) -> str | None:
        cands = [n for n in names if pred(n)]
        if not cands:  # a family absent from this run (the smoke builds only the young bank): the view is NOT_AVAILABLE
            return None
        return max(cands, key=lambda n: (table[dname][n]["rho"], n))

    vd = cfg["verdict"]["dataset"]
    prim = cfg["representations"]["primary"]
    is_ppmi_svd = lambda n: fam(n).startswith(f"cooc{int(cfg['data']['train_prefix_tokens']) // 1_000_000}M_") and n.split(".")[2] == "ppmi" and n.split(".")[3].startswith("svd")  # noqa: E731
    views = {}
    # view 1: best over the grid on SimLex itself (the rule of the design)
    views["best_in_sample"] = {"trained": best(vd, lambda n: fam(n) == "trained"), "young": best(vd, lambda n: fam(n) == "young"),
                               "ppmi_svd": best(vd, is_ppmi_svd)}
    # view 2: the pre-declared primary variant
    views["primary"] = {"trained": f"trained.count.{prim['scheme']}.{prim['reduction']}", "young": f"young.count.{prim['scheme']}.{prim['reduction']}",
                        "ppmi_svd": f"cooc{int(cfg['data']['train_prefix_tokens']) // 1_000_000}M_sym5.cooc.{prim['scheme']}.{prim['reduction']}"}
    # view 3: chosen on WordSim-353 + MEN dev, read on SimLex-999
    other = [d for d in setup.sets if d != vd]
    sel = lambda pred: max([n for n in names if pred(n)], key=lambda n: (np.mean([table[d][n]["rho"] for d in other]), n), default=None)  # noqa: E731
    views["selected_on_other_sets"] = {"trained": sel(lambda n: fam(n) == "trained"), "young": sel(lambda n: fam(n) == "young"), "ppmi_svd": sel(is_ppmi_svd)}
    mg = cfg["verdict"]["margins"]
    verdicts = {}
    for vname, v in views.items():
        if any(x not in idx_of for x in v.values()):
            verdicts[vname] = {"verdict": "NOT_AVAILABLE", "variants": v}
            continue
        r = {k: table[vd][x]["rho"] for k, x in v.items()}
        d_y = boots[f"{vd}|{v['trained']}"] - boots[f"{vd}|{v['young']}"]
        d_p = boots[f"{vd}|{v['trained']}"] - boots[f"{vd}|{v['ppmi_svd']}"]
        gap_y = r["trained"] - r["young"]
        gap_p = r["trained"] - r["ppmi_svd"]
        if gap_y >= mg["vs_young"] and gap_p >= -mg["vs_ppmi_svd"]:
            verdict = "GO"
        elif gap_y <= 0:
            verdict = "NO-GO"
        else:
            verdict = "GREY"
        verdicts[vname] = {"verdict": verdict, "variants": v, "rho": r, "trained_minus_young": gap_y, "trained_minus_ppmi_svd": gap_p,
                           "trained_minus_young_ci95_paired": K.ci95(d_y), "trained_minus_ppmi_svd_ci95_paired": K.ci95(d_p),
                           "p_boot_trained_minus_young_le_0": float((d_y <= 0).mean())}
    rec.m("verdict.early", verdicts["best_in_sample"]["verdict"], label="derived", rule=cfg["verdict"], detail=verdicts["best_in_sample"],
          note="the rule of the design on the in-sample best of each family over the same variant grid; exploratory")
    rec.m("verdict.views", verdicts, label="derived",
          note="robustness views; paired intervals resample the same pairs for both variants; in-sample selection bias is not removed by the bootstrap")

    # ---- family summary table (best and primary on every set)
    fams = sorted(set(fam(n) for n in names))
    summary = {}
    for f in fams:
        summary[f] = {}
        for dname in setup.sets:
            bn = best(dname, lambda n, f=f: fam(n) == f)
            summary[f][dname] = {"best_variant": bn, "best_rho": table[dname][bn]["rho"], "best_ci95": table[dname][bn]["ci95"]}
            pn = f"{f}.count.{prim['scheme']}.{prim['reduction']}" if f in cfg["banks"] else f"{f}.cooc.{prim['scheme']}.{prim['reduction']}"
            if pn in idx_of:
                summary[f][dname].update({"primary_variant": pn, "primary_rho": table[dname][pn]["rho"], "primary_ci95": table[dname][pn]["ci95"]})
    rec.m("summary.by_family", summary, label="derived")

    # ---- WordNet purity
    lab = setup.labels
    elig = setup.eligible
    lex = [lab[int(t)][0] for t in elig]
    pos = [lab[int(t)][1] for t in elig]
    lex_of = {int(t): lab[int(t)][0] for t in elig}
    pos_of = {int(t): lab[int(t)][1] for t in elig}
    syn_of = {int(t): lab[int(t)][2] for t in elig}
    k_nn = int(cfg["eval"]["neighbours"])
    wn = {}
    for n in names:
        nl = nn_elig[idx_of[n]][:, :k_nn]
        wn[n] = {"supersense_agreement@10": K.neighbour_agreement(nl, lex, lex_of), "pos_agreement@10": K.neighbour_agreement(nl, pos, pos_of),
                 "synonym_hit@10": K.synonym_hit_rate(nl, elig, syn_of)}
    chance = {"supersense": K.label_chance(lex), "pos": K.label_chance(pos)}
    rec.m("wordnet.neighbours", wn, label="derived", chance=chance,
          note=f"queries and candidates: the {elig.shape[0]} WordNet-eligible word-like tokens; label = lexicographer file / POS of the first synset")
    km = {}
    kk = int(cfg["eval"]["kmeans_k"])
    elig_lex = np.array(lex, dtype=object)
    for ename, E in sorted(embs.items()):
        a = K.kmeans_cosine(E[elig], kk, seed, int(cfg["eval"]["kmeans_iters"]))
        km[ename] = {"purity": K.purity(a, elig_lex), "chance_permuted": K.permuted_purity(a, elig_lex, int(cfg["eval"]["purity_permutations"]), seed),
                     "assign": a}
    rec.m("wordnet.kmeans_purity", {k: {x: y for x, y in v.items() if x != "assign"} for k, v in km.items()}, label="derived",
          note=f"spherical k-means, k {kk}, on the primary embeddings (ppmi svd300 p0.5; CBOW: w_in) of the eligible tokens; label = supersense")

    # ---- printouts
    rec.log("=" * 100)
    for dname, s in setup.sets.items():
        rec.log(f"{dname}: {s['n_covered']} of {s['n_total']} pairs covered")
    rec.log(f"EARLY VERDICT ({vd}): {verdicts['best_in_sample']['verdict']}")
    for vname, v in verdicts.items():
        rec.log(f"  view {vname}: {json.dumps(_plain(v))}")
    rec.log("-" * 100)
    rec.log("family summary (best variant per set; primary variant = ppmi svd300 p0.5):")
    for f in fams:
        cells = []
        for dname in setup.sets:
            c = summary[f][dname]
            cells.append(f"{dname} best {c['best_rho']:+.3f} [{c['best_ci95'][0]:+.3f},{c['best_ci95'][1]:+.3f}]"
                         + (f" prim {c['primary_rho']:+.3f}" if "primary_rho" in c else ""))
        rec.log(f"  {f:22s} " + " | ".join(cells))
    rec.log("-" * 100)
    rec.log("all variants (SimLex-999, WordSim-353, MEN dev; WordNet supersense@10, POS@10, synonym@10):")
    for n in names:
        w = wn[n]
        rec.log(f"  {n:40s} " + " ".join(f"{table[d][n]['rho']:+.3f}" for d in setup.sets)
                + f"   wn {w['supersense_agreement@10']:.3f} {w['pos_agreement@10']:.3f} {w['synonym_hit@10']:.3f}")
    rec.log(f"  chance: supersense {chance['supersense']:.3f}, pos {chance['pos']:.3f}")
    rec.log("-" * 100)
    for ename, v in km.items():
        rec.log(f"k-means purity {ename}: {v['purity']:.3f} (permuted {v['chance_permuted']:.3f})")
    show = []
    for f in ("trained", "young", "epoch1"):
        for n in (views["best_in_sample"].get(f), f"{f}.count.{prim['scheme']}.{prim['reduction']}", f"{f}.lit.{prim['scheme']}.{prim['reduction']}"):
            if n and n in idx_of and n not in show:
                show.append(n)
    for n in (views["best_in_sample"]["ppmi_svd"], "cbow.in.-.full", "exchange.levels.-.full"):
        if n in idx_of and n not in show:
            show.append(n)
    nbrs = {}
    rec.log("-" * 100)
    for n in show:
        rec.log(f"nearest word-like neighbours, {n}:")
        nbrs[n] = {}
        for w, row in zip(setup.probe_words, nn_probe[idx_of[n]]):
            nbrs[n][w] = [setup.texts[int(t)].strip() for t in row]
            rec.log(f"  {w:10s} -> {', '.join(nbrs[n][w])}")
    rec.m("print.neighbours", nbrs, label="measured")
    freq = np.load(ROOT / "data/wikitext103/bpe8000/train/sym_freq.npy")[:V]
    clus = {}
    for ename in [e for e in ("trained.count.ppmi", "young.count.ppmi", "trained.lit.ppmi") if e in km]:
        a = km[ename]["assign"]
        sizes = np.bincount(a, minlength=kk)
        order = np.argsort(-sizes, kind="stable")[: int(cfg["eval"]["print_clusters"])]
        clus[ename] = []
        rec.log(f"clusters of {ename} (the {len(order)} largest of {kk}; members by train frequency):")
        for c in order.tolist():
            mem = elig[a == c]
            mem = mem[np.argsort(-freq[mem], kind="stable")][:14]
            words = [setup.texts[int(t)].strip() for t in mem]
            top_lex = K.purity(np.zeros(int((a == c).sum()), np.int64), elig_lex[a == c])
            clus[ename].append({"cluster": c, "size": int(sizes[c]), "members": words, "supersense_purity": top_lex})
            rec.log(f"  [{sizes[c]:4d}, purity {top_lex:.2f}] {', '.join(words)}")
    rec.m("print.clusters", clus, label="measured")
    for bname, cl in clauses.items():
        rec.log("-" * 100)
        rec.log(f"clauses of the {bname} bank as rules (counted targets over the prefix; positive votes of the bank):")
        for c in cl:
            rec.log(f"  [{c['selection']}] clause {c['clause']}  N {c['N_counted']}  D {c['D_counted']}  H {c['target_entropy_nats']:.2f}")
            rec.log(f"     {c['rule']}")
            rec.log(f"     THEN targets counted: {', '.join(f'{t!r} {s:.3f}' for t, _, s in c['top_counted_targets'])}")
            rec.log(f"     votes: {', '.join(f'{t!r} {w}' for t, w in c['top_positive_votes'])}")
    rec.m("print.clauses", clauses, label="measured")
    rec.m("dataset_accesses", {d: 1 for d in setup.sets}, label="measured",
          note="this evaluate run read the scores of each set once (SimLex-999 and WordSim-353 have no split; MEN: dev only)")
    rec.flush()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/cvtm/configs/ssl_concepts_early.json"))
    ap.add_argument("--phase", required=True, choices=["bank", "baselines", "evaluate"])
    ap.add_argument("--bank", default=None)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prefix-tokens", type=int, default=None, help="smoke only")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = out_dir(args)
    rec = Rec(out, args.smoke)
    rec.m("argv", sys.argv[1:])
    rec.m("mode_smoke", bool(args.smoke))
    t0 = time.time()
    if args.phase == "bank":
        if args.bank not in cfg["banks"]:
            raise SystemExit("--bank must name a bank of the config")
        rc = run_bank(cfg, args, rec, out)
    elif args.phase == "baselines":
        rc = run_baselines(cfg, args, rec, out)
    else:
        if not args.inputs:
            raise SystemExit("--inputs: the run dirs of the bank and baseline phases")
        rc = run_evaluate(cfg, args, rec, out)
    rec.m("wall_s_total", time.time() - t0)
    rec.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
