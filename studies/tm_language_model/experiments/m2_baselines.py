"""M2 Track C baselines (docs/06 "Baselines", docs/04 M2 ACCEPT "every baseline has a log dir").

One logged run per baseline family, every metric labelled, on the FROZEN validation eval sets
(subword and whole-word variants, docs/13 section 4) and, for the likelihood models, bits per
byte on the validation split (protocol CP). The test split is never loaded here.

    # CPU families (.venv-tm)
    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name M2_baselines_ngram_v8000 \\
        --config experiments/configs/M2_baselines.json -- .venv-tm/bin/python experiments/m2_baselines.py \\
        --family ngram --vocab 8000
    ... --family count_probe --vocab 2000 | 8000
    ... --family cbow_knn --vocab 8000
    # GPU families (.venv-dl, shared lock)
    scripts/gpu_run.sh shared -- scripts/mem_guard.py -- scripts/run_logged.py --name M2_baselines_transformer \\
        --config experiments/configs/M2_baselines.json -- .venv-dl/bin/python experiments/m2_baselines.py \\
        --family transformer --vocab 8000
    ... --family transformer_mlm_smoke --vocab 8000
    ... --family bert

Metric names: <model>.<set>.<metric>. Sets: validation_subword (frozen subword mask set),
validation_wholeword (frozen whole-word set), validation (the whole split, causal positions).
"top1_visible": hidden context tokens are not used (the setting the TM faces); "top1_truectx":
the true tokens around the target are used, hidden or not (the M3 gate's bigram convention).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402
from tmlm.data.corpus_store import NO_WORD, load_split, load_word_id, raw_dir, split_dir  # noqa: E402
from tmlm.data.evalsets import load_evalset  # noqa: E402

CORPUS = "wikitext103"
METRICS: dict[str, dict] = {}


def log(msg: str) -> None:
    print(f"[m2_baselines {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        d = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "m2_baselines_unlogged"
        d.mkdir(parents=True, exist_ok=True)
    return d


class EvalData:
    """Validation split, the two frozen eval sets, word ids and the byte count."""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.val = load_split(CORPUS, vocab, "validation", check=True)
        self.tok = np.ascontiguousarray(self.val.tok)
        self.doc_start = np.ascontiguousarray(self.val.doc_start)
        self.sub = load_evalset(CORPUS, vocab, "validation", "subword")
        self.ww = load_evalset(CORPUS, vocab, "validation", "wholeword")
        self.word_id = load_word_id(CORPUS, vocab, "validation")
        meta = json.loads((split_dir(CORPUS, vocab, "validation") / "meta.json").read_text())
        raw_meta = json.loads((raw_dir(CORPUS, "validation") / "meta.json").read_text())
        self.n_bytes = int(raw_meta["n_bytes"])
        assert self.n_bytes == int(meta["n_bytes_utf8"])
        for name, arr in (("subword", self.sub), ("wholeword", self.ww)):
            dataset_hash(f"evalset_validation_{name}_V{vocab}", arr["meta"]["sha256_combined"])
        dataset_hash(f"validation_tok_V{vocab}", meta["sha256"]["tok.npy"])
        dataset_hash("validation_text_utf8", raw_meta["sha256"]["text.utf8"])
        metric("n_validation_tokens", int(self.tok.shape[0]))
        metric("n_validation_bytes", self.n_bytes, note="UTF-8 bytes of raw/validation/text.utf8 (protocol CP)")
        metric("n_subword_targets", int(self.sub["targets"].shape[0]))
        metric("n_wholeword_targets", int(self.ww["targets"].shape[0]))
        metric("n_wholeword_words", int(self.ww["hidden_words"].shape[0]))

    @property
    def sub_targets(self) -> np.ndarray:
        return self.sub["targets"].astype(np.int64)

    @property
    def ww_targets(self) -> np.ndarray:
        return self.ww["targets"].astype(np.int64)

    def word_of(self, targets: np.ndarray) -> np.ndarray:
        w = self.word_id[targets]
        assert not np.any(w == NO_WORD)
        return w

    def report_masked(self, prefix: str, predict, truectx_too: bool = True) -> None:
        """predict(pos, hidden) -> int64 predictions; writes the four masked-set metrics."""
        gold_s = self.tok[self.sub_targets].astype(np.int64)
        gold_w = self.tok[self.ww_targets].astype(np.int64)
        t0 = time.time()
        p = predict(self.sub_targets, self.sub["hidden"])
        metric(f"{prefix}.validation_subword.top1_visible", float((p == gold_s).mean()), wall_s=round(time.time() - t0, 1))
        if truectx_too:
            t0 = time.time()
            p = predict(self.sub_targets, None)
            metric(f"{prefix}.validation_subword.top1_truectx", float((p == gold_s).mean()), wall_s=round(time.time() - t0, 1))
        t0 = time.time()
        p = predict(self.ww_targets, self.ww["hidden"])
        from tmlm.eval.baselines.ngram import word_accuracy
        metric(f"{prefix}.validation_wholeword.piece_top1_visible", float((p == gold_w).mean()), wall_s=round(time.time() - t0, 1))
        metric(f"{prefix}.validation_wholeword.word_acc_visible", word_accuracy(p, gold_w, self.word_of(self.ww_targets)),
               note="a word counts when every one of its pieces is predicted right, pieces predicted independently")


def load_train(vocab: int, n_tokens: int | None = None):
    tr = load_split(CORPUS, vocab, "train")
    meta = json.loads((split_dir(CORPUS, vocab, "train") / "meta.json").read_text())
    dataset_hash(f"train_tok_V{vocab}", meta["sha256"]["tok.npy"])
    tok, ds = tr.tok, tr.doc_start
    if n_tokens is not None and n_tokens < tok.shape[0]:
        d = int(np.searchsorted(ds, n_tokens, side="right")) - 1
        end = int(ds[max(1, d)])
        tok, ds = tok[:end], ds[:d + 1] if ds[d] == end else ds[:d + 2]
    tok = np.ascontiguousarray(tok)
    ds = np.ascontiguousarray(ds)
    metric(f"train_tokens_V{vocab}", int(tok.shape[0]))
    metric(f"train_docs_V{vocab}", int(ds.shape[0] - 1))
    return tok, ds


# ------------------------------------------------------------------ families
def family_ngram(args) -> None:
    from tmlm.eval.baselines.ngram import (
        BidirectionalKN, bigram_table, bits_per_byte, left_bigram_top1, right_bigram_top1, unigram_top1,
    )

    V = args.vocab
    order = args.order
    ev = EvalData(V)
    tok, ds = load_train(V, args.train_tokens)
    # raw-count predictors
    t0 = time.time()
    big = bigram_table(tok, V)
    for name, pos in (("validation_subword", ev.sub_targets), ("validation_wholeword", ev.ww_targets)):
        metric(f"unigram_argmax.{name}.top1", unigram_top1(tok, V, pos, ev.tok))
        metric(f"left_bigram.{name}.top1_truectx", left_bigram_top1(tok, V, pos, ev.tok, big), note="true left token, hidden or not")
        metric(f"right_bigram.{name}.top1_truectx", right_bigram_top1(tok, V, pos, ev.tok, big), note="true right token, hidden or not")
    metric("raw_count_predictors.wall_s", round(time.time() - t0, 1), unit="s")
    del big
    # Kneser-Ney, forward and backward
    t0 = time.time()
    bi = BidirectionalKN.fit(tok, ds, order, V, discount_mode="modified")
    fit_s = time.time() - t0
    metric(f"kn{order}.fit_wall_s", round(fit_s, 1), unit="s", note="forward and backward models, full train split")
    metric(f"kn{order}.n_types_per_level", bi.fwd.n_types[1:], note="unique k-grams (forward model), k = 1..order")
    for k in range(1, order + 1):
        metric(f"kn{k}.discounts_modified", {"fwd": {f"level{j}": list(d) for j, d in bi.fwd.discounts(k).items()}},
               note="(D1, D2, D3+) per level, Chen and Goodman 1999")
    # bits per byte and causal top-1 for every order (forward model)
    all_pos = np.arange(ev.tok.shape[0])
    for k in range(1, order + 1):
        t0 = time.time()
        lp = bi.fwd.logprob(ev.tok, ev.doc_start, order=k)
        metric(f"kn{k}.validation.bits_per_byte", bits_per_byte(lp, ev.n_bytes), wall_s=round(time.time() - t0, 1),
               note="modified KN, interpolated, in-document, total bits / UTF-8 bytes of the validation text")
        metric(f"kn{k}.validation.bits_per_token", float(-lp.mean()))
        t0 = time.time()
        pred = bi.fwd.predict(ev.tok, ev.doc_start, all_pos, order=k)
        metric(f"kn{k}.validation.causal_top1", float((pred == ev.tok.astype(np.int64)).mean()), wall_s=round(time.time() - t0, 1),
               note="argmax next token from the true left context, every validation position")
    # single-discount variant (NLTK's, D = 0.1) for the record
    bi.fwd.set_discount_mode("single", args.single_discount)
    lp = bi.fwd.logprob(ev.tok, ev.doc_start, order=order)
    metric(f"kn{order}_single_D{args.single_discount}.validation.bits_per_byte", bits_per_byte(lp, ev.n_bytes),
           note="single-discount interpolated KN (NLTK's variant), same counts")
    bi.fwd.set_discount_mode("modified")
    # masked fill-in: left only, right only, both (product of experts)
    for k in sorted({2, order}):
        for side in ("left", "right", "both"):
            ev.report_masked(f"kn{k}_{side}", lambda pos, hidden, k=k, side=side: bi.predict(ev.tok, ev.doc_start, pos, hidden, order=k, side=side))


def family_count_probe(args) -> None:
    from tmlm.eval.baselines.count_probe import count_probe_predict, count_probe_tables

    V = args.vocab
    ev = EvalData(V)
    tok, ds = load_train(V, args.train_tokens)
    for name, offsets in (("masked", (-3, -2, -1, 1, 2, 3)), ("causal", (-3, -2, -1))):
        t0 = time.time()
        tables = count_probe_tables(tok, V, offsets)
        metric(f"count_probe_{name}.tables_wall_s", round(time.time() - t0, 1), unit="s", note=f"offsets {offsets}")
        ev.report_masked(f"count_probe_{name}", lambda pos, hidden, tables=tables: count_probe_predict(ev.tok, pos, tables, hidden))
        if name == "causal":
            t0 = time.time()
            pred = count_probe_predict(ev.tok, np.arange(ev.tok.shape[0]), tables, None)
            metric("count_probe_causal.validation.causal_top1", float((pred == ev.tok.astype(np.int64)).mean()), wall_s=round(time.time() - t0, 1))
        del tables


def family_cbow_knn(args) -> None:
    from tmlm.eval.baselines.cbow_knn import CbowModel

    V = args.vocab
    ev = EvalData(V)
    tok, ds = load_train(V, args.train_tokens)
    model = CbowModel(V, dim=args.dim, window=args.window, negatives=args.negatives, batch=args.batch, lr=args.lr,
                      lr_min=args.lr_min, subsample_t=args.subsample_t, epochs=args.epochs, seed=args.seed)
    metric("cbow.hyperparameters", model.hyperparameters())
    stats = model.train(tok, ds, log_every=500, time_cap_s=args.time_cap_s)
    for k, v in stats.items():
        metric(f"cbow.train.{k}", v)
    paths = model.save(out_dir())
    metric("cbow.embedding_files", {k: str(p) for k, p in paths.items()}, note="exchanged as .npy")
    ev.report_masked("cbow_softmax", lambda pos, hidden: model.predict_softmax(ev.tok, ev.doc_start, pos, hidden))
    t0 = time.time()
    bank = model.build_bank(tok, ds, args.bank_size, seed=args.seed + 7)
    metric("cbow_knn.bank", {"size": int(bank[0].shape[0]), "k": args.knn_k, "wall_s": round(time.time() - t0, 1)})
    ev.report_masked("cbow_knn", lambda pos, hidden: model.predict_knn(ev.tok, ev.doc_start, pos, bank, k=args.knn_k, hidden=hidden))


def family_transformer(args) -> None:
    from tmlm.eval.baselines import transformer_small

    ev = EvalData(args.vocab)
    tok, ds = load_train(args.vocab, args.train_tokens)
    transformer_small.run_sweep(tok, ds, ev, metric, out_dir(), layers=args.layers, widths=args.widths, heads=args.heads,
                                seq_len=args.seq_len, batch_seqs=args.batch_seqs, lr=args.lr, warmup=args.warmup,
                                time_cap_s=args.time_cap_s, seed=args.seed, test=args.test)


def family_transformer_mlm_smoke(args) -> None:
    from tmlm.eval.baselines import transformer_small

    ev = EvalData(args.vocab)
    tok, ds = load_train(args.vocab, args.train_tokens)
    transformer_small.run_masked_smoke(tok, ds, ev, metric, out_dir(), layers=args.layers[0], width=args.widths[0], heads=args.heads,
                                       seq_len=args.seq_len, batch_seqs=args.batch_seqs, lr=args.lr, warmup=args.warmup,
                                       time_cap_s=args.time_cap_s, seed=args.seed)


def family_bert(args) -> None:
    from tmlm.eval.baselines import bert_reference

    ev = EvalData(args.vocab)
    bert_reference.run(ev, metric, out_dir(), model_name=args.bert_model, window=args.bert_window, stride=args.bert_stride,
                       batch=args.bert_batch, limit=args.limit)


FAMILIES = {
    "ngram": family_ngram, "count_probe": family_count_probe, "cbow_knn": family_cbow_knn,
    "transformer": family_transformer, "transformer_mlm_smoke": family_transformer_mlm_smoke, "bert": family_bert,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-tokens", type=int, default=None, help="slice of the train split (tests, smoke runs)")
    ap.add_argument("--test", action="store_true", help="transformer only: also score the test split (COUNTED access); not used in M2")
    # ngram
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--single-discount", type=float, default=0.1)
    # cbow
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--subsample-t", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--knn-k", type=int, default=32)
    ap.add_argument("--bank-size", type=int, default=1_000_000)
    ap.add_argument("--time-cap-s", type=float, default=None)
    # transformer
    ap.add_argument("--layers", default="2,4,6")
    ap.add_argument("--widths", default="128,256")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-seqs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=200)
    # bert
    ap.add_argument("--bert-model", default="bert-base-uncased")
    ap.add_argument("--bert-window", type=int, default=512)
    ap.add_argument("--bert-stride", type=int, default=256)
    ap.add_argument("--bert-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="bert: score only the first N hidden words (smoke)")
    args = ap.parse_args()
    args.layers = [int(x) for x in str(args.layers).split(",")]
    args.widths = [int(x) for x in str(args.widths).split(",")]
    if args.lr is None:
        args.lr = 0.025 if args.family == "cbow_knn" else 1e-3
    if args.family in ("transformer", "transformer_mlm_smoke") and args.time_cap_s is None:
        args.time_cap_s = 600.0 if args.family == "transformer" else 120.0
    metric("family", args.family)
    metric("args", {k: v for k, v in vars(args).items()})
    metric("seed", args.seed)
    t0 = time.time()
    FAMILIES[args.family](args)
    metric("family_wall_s", round(time.time() - t0, 1), unit="s")
    if runlog.current_run_dir() is None:
        print(json.dumps(METRICS, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
