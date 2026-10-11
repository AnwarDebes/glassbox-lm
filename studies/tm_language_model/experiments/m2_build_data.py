"""M2 data pipeline (docs/04 M2): raw article store, BPE tokenizers, token corpus
stores, cluster codes, frozen evaluation mask sets, the paragraph-pair task, contamination
report, PTB last. Every artifact is hashed; the hashes are registered as dataset hashes of this
run and written ONCE to docs/notes/preregistration-data-hashes.json (docs/13 section 4).

Run (CPU only, through the run logger, protocol rules 5 and 13):
    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name M2_data \\
        --config experiments/configs/M2_data.json -- .venv-tm/bin/python experiments/m2_build_data.py

Files under data/ are only created, never overwritten (protocol rule 9). When an artifact already
exists its recorded sha256 is checked and the artifact is reused, so an interrupted run can
be resumed by running the script again. Every number reported to metrics.json is measured
by this script (label "measured"), except where stated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.data import bpe, clusters, contamination, corpus_store, evalsets, pairs  # noqa: E402
from tmlm.data.synthetic import stop_of, sym_freq_of  # noqa: E402

CORPUS = "wikitext103"
HF_ID = ("Salesforce/wikitext", "wikitext-103-raw-v1")
SPLITS = ("train", "validation", "test")
ACCESS_CALLER = "experiments/m2_build_data.py"

METRICS: dict[str, dict] = {}
HASHES: dict[str, str] = {}


def log(msg: str) -> None:
    print(f"[m2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metric(name: str, value, label: str = "measured", **extra) -> None:
    METRICS[name] = {"value": value, "label": label, **extra}
    if runlog.current_run_dir() is not None:
        runlog.write_metrics({name: METRICS[name]})


def dataset_hash(name: str, digest: str) -> None:
    HASHES[name] = digest
    if runlog.current_run_dir() is not None:
        runlog.write_dataset_hashes({name: digest}, on_conflict="error")


def write_run_json(relname: str, obj) -> None:
    d = runlog.current_run_dir()
    if d is not None:
        runlog.write_json(d / relname, obj)


def load_meta(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------ step 1: raw articles
def step_raw(root: Path, config: dict, synthetic_rows: Path | None = None) -> dict[str, corpus_store.RawSplit]:
    out = {}
    need = [s for s in SPLITS if load_meta(corpus_store.raw_dir(CORPUS, s, root) / "meta.json") is None]
    if need:
        if synthetic_rows is not None:
            # tests only: {split: [rows]} in the WikiText row format instead of the Hugging Face dataset
            rows_by_split = json.loads(synthetic_rows.read_text(encoding="utf-8"))
            source = {"synthetic_rows": str(synthetic_rows)}
        else:
            import datasets

            t0 = time.time()
            ds = datasets.load_dataset(*HF_ID)
            metric("raw_hf_load_wall_s", time.time() - t0, unit="s")
            rows_by_split = {s: list(ds[s]["text"]) for s in need}  # datasets 5 Column: materialize
            source = {"hf_id": HF_ID}
        for split in need:
            t0 = time.time()
            rows = rows_by_split[split]
            texts, titles = corpus_store.rows_to_articles(rows)
            meta = corpus_store.write_raw_split(CORPUS, split, texts, titles, {**source, "n_rows": len(rows)}, root)
            metric(f"raw_{split}_wall_s", time.time() - t0, unit="s")
            log(f"raw {split}: {len(rows)} rows -> {meta['n_doc']} articles, {meta['n_bytes']} bytes")
    for split in SPLITS:
        meta = load_meta(corpus_store.raw_dir(CORPUS, split, root) / "meta.json")
        purpose = "M2 build: raw article store, tokenization, eval-set freeze, pairs, contamination" if split == "test" else None
        raw = corpus_store.load_raw_split(CORPUS, split, root, purpose=purpose, check=True)
        out[split] = raw
        metric(f"raw_{split}_rows", meta["source"]["n_rows"])
        metric(f"raw_{split}_articles", raw.n_doc)
        metric(f"raw_{split}_bytes_utf8", len(raw.text), unit="bytes")
        for fname, digest in meta["sha256"].items():
            dataset_hash(f"{CORPUS}/raw/{split}/{fname}", digest)
    return out


# ------------------------------------------------------------------ step 2: BPE
def step_bpe(root: Path, raw: dict[str, corpus_store.RawSplit], vocab_sizes: list[int], limit_32k_s: float) -> list[int]:
    done = []
    train_texts = None
    last_wall = None
    for V in vocab_sizes:
        meta = load_meta(bpe.tokenizer_dir(V, root) / "meta.json")
        if meta is None:
            if V == 32000 and last_wall is not None and last_wall > limit_32k_s:
                log(f"skip V=32000: the V=16000 training took {last_wall:.0f} s > {limit_32k_s} s")
                metric("bpe_32000_skipped", 1, note=f"16000 training wall {last_wall:.0f} s exceeded the limit")
                continue
            if train_texts is None:
                train_texts = raw["train"].articles()
            t0 = time.time()
            tok = bpe.train_bpe(train_texts, V)
            wall = time.time() - t0
            if tok.get_vocab_size() != V:
                raise RuntimeError(f"BPE V={V}: got {tok.get_vocab_size()} symbols")
            meta = bpe.save_tokenizer(tok, V, {"corpus": CORPUS, "split": "train",
                                                "text_sha256": hashing.sha256_bytes(raw["train"].text),
                                                "n_articles": raw["train"].n_doc}, root)
            metric(f"bpe_{V}_train_wall_s", wall, unit="s")
            last_wall = wall
            log(f"bpe {V}: trained in {wall:.1f} s")
        metric(f"bpe_{V}_vocab_size", meta["vocab_size"])
        dataset_hash(f"tokenizers/bpe_{V}/tokenizer.json", meta["sha256"]["tokenizer.json"])
        done.append(V)
    return done


# ------------------------------------------------------------------ step 3: encode splits
def step_encode(root: Path, raw: dict[str, corpus_store.RawSplit], V: int) -> None:
    tok = None
    for split in SPLITS:
        d = corpus_store.split_dir(CORPUS, V, split, root)
        if (d / "tok.npy").exists():
            continue
        if tok is None:
            tok = bpe.load_tokenizer(V, root)
            ws_lead, all_ws = bpe.token_whitespace_tables(tok)
        if split == "test":
            evalsets.record_test_access(ACCESS_CALLER, f"M2 build: encode test split at V={V}", root=root)
        t0 = time.time()
        ids, doc_start = bpe.encode_docs(tok, raw[split].articles())
        wid = bpe.word_ids(ids, doc_start, ws_lead, all_ws)
        hashes = corpus_store.write_split_arrays(CORPUS, V, split, {"tok": ids, "doc_start": doc_start, "word_id": wid}, root)
        with open(d / "encode.json", "x", encoding="utf-8") as handle:
            handle.write(json.dumps({"n_tok": int(ids.shape[0]), "n_doc": int(doc_start.shape[0] - 1),
                                                   "n_words": int((wid != corpus_store.NO_WORD).sum()),
                                                   "n_bytes_utf8": len(raw[split].text), "sha256": hashes,
                                                   "wall_s": time.time() - t0}, indent=1, sort_keys=True) + "\n")
        log(f"encode V={V} {split}: {ids.shape[0]} tokens in {time.time() - t0:.1f} s")


# ------------------------------------------------------------------ step 4: train statistics
def step_train_stats(root: Path, V: int, seed: int, max_iter: int) -> None:
    train_dir = corpus_store.split_dir(CORPUS, V, "train", root)
    if not (train_dir / "kind_map.npy").exists():
        ids = np.load(train_dir / "tok.npy")
        doc_start = np.load(train_dir / "doc_start.npy")
        t0 = time.time()
        B = clusters.bigram_counts(ids, doc_start, V)
        metric(f"clusters_{V}_bigram_types", int(B.nnz))
        metric(f"clusters_{V}_bigram_count_wall_s", time.time() - t0, unit="s")
        t0 = time.time()
        levels, logs = clusters.hierarchical_exchange(B, seed=seed, max_iter=max_iter)
        wall = time.time() - t0
        if not clusters.check_refinement(levels):
            raise RuntimeError("cluster levels do not refine")
        km = clusters.kind_map_from_levels(V, levels)
        stop = stop_of(ids, V, corpus_store.N_STOP)
        sym_freq = sym_freq_of(ids, km, V)
        stats = {f"level{lg.level}": {**lg.as_dict(), **clusters.cluster_stats(levels[lg.level - 1], lg.n_clusters)} for lg in logs}
        write_run_json(f"cluster_objective_V{V}.json", {"vocab": V, "seed": seed, "max_iter": max_iter, "wall_s": wall, **stats})
        metric(f"clusters_{V}_exchange_wall_s", wall, unit="s")
        for lg in logs:
            st = clusters.cluster_stats(levels[lg.level - 1], lg.n_clusters)
            metric(f"clusters_{V}_level{lg.level}_objective_final", lg.objective[-1],
                   note="F = sum f(N(g,h)) - sum f(N_left) - sum f(N_right), f(x) = x log x; higher is better")
            metric(f"clusters_{V}_level{lg.level}_objective_initial", lg.initial_objective)
            metric(f"clusters_{V}_level{lg.level}_iterations", len(lg.objective))
            metric(f"clusters_{V}_level{lg.level}_converged", int(lg.converged))
            metric(f"clusters_{V}_level{lg.level}_empty_clusters", st["empty"])
            metric(f"clusters_{V}_level{lg.level}_singleton_clusters", st["singletons"])
        log(f"clusters V={V}: {wall:.0f} s, final objectives {[lg.objective[-1] for lg in logs]}")
        for split in SPLITS:
            corpus_store.write_split_arrays(CORPUS, V, split, {"kind_map": km, "stop": stop, "sym_freq": sym_freq}, root)
    for split in SPLITS:
        d = corpus_store.split_dir(CORPUS, V, split, root)
        if not (d / "targets.npy").exists():
            n_tok = int(np.load(d / "doc_start.npy")[-1])
            corpus_store.write_split_arrays(CORPUS, V, split, {"targets": np.arange(n_tok, dtype=np.uint32)}, root)
        if not (d / "meta.json").exists():
            enc = json.loads((d / "encode.json").read_text(encoding="utf-8"))
            hashes = {f"{name}.npy": hashing.sha256_array(np.load(d / f"{name}.npy")) for name in corpus_store.STORE_ARRAYS}
            for k, v in enc["sha256"].items():
                if hashes[k] != v:
                    raise RuntimeError(f"{d}/{k}: hash changed since encoding")
            corpus_store.write_split_meta(CORPUS, V, split, {
                "corpus": CORPUS, "vocab": V, "split": split, "n_tok": enc["n_tok"], "n_doc": enc["n_doc"],
                "n_words": enc["n_words"], "n_bytes_utf8": enc["n_bytes_utf8"], "n_stop": corpus_store.N_STOP,
                "clusters": {"seed": seed, "max_iter": max_iter, "method": "3-level 16-way exchange, train bigrams"},
                "sha256": hashes}, root)
        meta = load_meta(d / "meta.json")
        metric(f"store_{V}_{split}_n_tok", meta["n_tok"])
        metric(f"store_{V}_{split}_n_doc", meta["n_doc"])
        metric(f"store_{V}_{split}_n_words", meta["n_words"])
        metric(f"store_{V}_{split}_bytes_utf8", meta["n_bytes_utf8"], unit="bytes")
        for fname, digest in meta["sha256"].items():
            dataset_hash(f"{CORPUS}/bpe{V}/{split}/{fname}", digest)
    # the split loads must validate as CorpusArrays
    from tmlm.engine.config import EngineConfig

    cfg = EngineConfig(mode="masked", vocab=V, n_clauses=64, seed=1)
    corpus_store.load_split(CORPUS, V, "validation", root).validate(cfg)


# ------------------------------------------------------------------ step 5: frozen eval sets
def step_evalsets(root: Path, V: int) -> None:
    for split in ("validation", "test"):
        for variant in ("subword", "wholeword"):
            path = evalsets.evalset_path(CORPUS, V, split, variant, root)
            if not path.exists():
                if split == "test":
                    evalsets.record_test_access(ACCESS_CALLER, f"M2 build: freeze {variant} eval mask set at V={V}", root=root)
                d = corpus_store.split_dir(CORPUS, V, split, root)
                if variant == "subword":
                    n_tok = int(np.load(d / "doc_start.npy")[-1])
                    built = evalsets.build_subword(n_tok, V, split,
                                                   max_targets=evalsets.MAX_VALID_TARGETS if split == "validation" else None)
                else:
                    built = evalsets.build_wholeword(np.load(d / "word_id.npy"), V, split)
                evalsets.save_evalset(CORPUS, V, split, variant, built, root)
            meta = load_meta(path.with_suffix(".meta.json"))
            metric(f"evalset_{V}_{split}_{variant}_n_hidden", meta["n_hidden"])
            metric(f"evalset_{V}_{split}_{variant}_n_targets", meta["n_targets"], rule=meta["rule"])
            dataset_hash(f"{CORPUS}/bpe{V}/evalsets/{split}_{variant}.npz", meta["sha256_combined"])


# ------------------------------------------------------------------ step 6: paragraph pairs
def step_pairs(root: Path, raw: dict[str, corpus_store.RawSplit], V: int, sizes: dict[str, int]) -> None:
    tok = None
    for split in SPLITS:
        path = pairs.pairs_dir(CORPUS, V, root) / f"{split}.npz"
        if not path.exists():
            if tok is None:
                tok = bpe.load_tokenizer(V, root)
            if split == "test":
                evalsets.record_test_access(ACCESS_CALLER, f"M2 build: paragraph pairs of the test split at V={V}", root=root)
            t0 = time.time()
            paras = pairs.paragraphs_of_split(raw[split].articles(), tok)
            built = pairs.build_pairs(paras, sizes[split], split)
            pairs.save_pairs(CORPUS, V, split, built, root)
            metric(f"pairs_{V}_{split}_wall_s", time.time() - t0, unit="s")
        meta = load_meta(path.with_suffix(".meta.json"))
        metric(f"pairs_{V}_{split}_n_pairs", meta["n_pairs"], rule=meta["rule"])
        metric(f"pairs_{V}_{split}_n_pos", meta["n_pos"])
        metric(f"pairs_{V}_{split}_n_neg", meta["n_neg"])
        metric(f"pairs_{V}_{split}_n_paragraphs", meta["n_paragraphs"])
        metric(f"pairs_{V}_{split}_n_pos_possible", meta["n_pos_possible"])
        dataset_hash(f"{CORPUS}/pairs_bpe{V}/{split}.npz", meta["sha256_combined"])


# ------------------------------------------------------------------ step 7: contamination
def step_contamination(root: Path, V: int) -> None:
    train = None
    for n in contamination.NGRAM_ORDERS:
        train_set = None
        for split in ("validation", "test"):
            path = contamination.contamination_dir(CORPUS, V, root) / f"{split}_{n}gram.npz"
            if not path.exists():
                if train is None:
                    d = corpus_store.split_dir(CORPUS, V, "train", root)
                    train = (np.load(d / "tok.npy"), np.load(d / "doc_start.npy"))
                if train_set is None:
                    t0 = time.time()
                    train_set = contamination.train_ngram_set(train[0], train[1], n)
                    metric(f"contamination_{V}_train_{n}gram_distinct", int(train_set.shape[0]))
                    metric(f"contamination_{V}_train_{n}gram_wall_s", time.time() - t0, unit="s")
                if split == "test":
                    evalsets.record_test_access(ACCESS_CALLER, f"M2 build: {n}-gram overlap of the test split at V={V}", root=root)
                d = corpus_store.split_dir(CORPUS, V, split, root)
                built = contamination.overlap(train_set, np.load(d / "tok.npy"), np.load(d / "doc_start.npy"), n)
                contamination.save_overlap(CORPUS, V, split, built, root)
            meta = load_meta(path.with_suffix(".meta.json"))
            for key in ("n_ngrams", "n_seen", "ngram_seen_fraction", "token_covered_fraction", "n_doc", "n_doc_keep", "tokens_in_kept_docs"):
                metric(f"contamination_{V}_{split}_{n}gram_{key}", meta[key],
                       **({"rule": meta["deoverlap_rule"]} if key == "n_doc_keep" else {}))
            dataset_hash(f"{CORPUS}/contamination_bpe{V}/{split}_{n}gram.npz", meta["sha256_combined"])


# ------------------------------------------------------------------ step 8: pre-registration hashes
def step_prereg_hashes(path: Path, run_dir: Path | None) -> None:
    if path.exists():
        log(f"{path} exists: NOT overwritten (written once, docs/13 section 4)")
        metric("prereg_hashes_written", 0, note="file existed; left untouched")
        return
    obj = {"written_by": ACCESS_CALLER, "run_dir": run_dir.name if run_dir else None,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule": "written once by milestone M2 and never edited (docs/13 section 4)",
           "dataset_hashes": dict(sorted(HASHES.items()))}
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    metric("prereg_hashes_written", 1)
    metric("prereg_hashes_n_artifacts", len(HASHES))


# ------------------------------------------------------------------ step 9: PTB (last)
def step_ptb(root: Path, vocab_sizes: list[int]) -> None:
    """Penn Treebank by the parquet route of docs/02 V5. Word-level text with <unk>; one document
    per sentence row; tokenized with the WikiText-103 BPE tokenizers (no PTB-trained BPE)."""
    corpus = "ptb"
    try:
        import datasets

        need = [s for s in SPLITS if load_meta(corpus_store.raw_dir(corpus, s, root) / "meta.json") is None]
        if need:
            t0 = time.time()
            ds = datasets.load_dataset("ptb-text-only/ptb_text_only", revision="refs/convert/parquet")
            metric("ptb_hf_load_wall_s", time.time() - t0, unit="s")
            for split in need:
                col = "sentence" if "sentence" in ds[split].column_names else ds[split].column_names[0]
                rows = ds[split][col]
                texts = [r.rstrip("\n") + "\n" for r in rows]
                meta = corpus_store.write_raw_split(corpus, split, texts, [""] * len(texts),
                                                   {"hf_id": ("ptb-text-only/ptb_text_only", "refs/convert/parquet"),
                                                    "column": col, "n_rows": len(rows)}, root)
                log(f"ptb raw {split}: {len(rows)} rows, {meta['n_bytes']} bytes")
        raws = {}
        for split in SPLITS:
            purpose = "M2 build: PTB raw store and tokenization" if split == "test" else None
            raws[split] = corpus_store.load_raw_split(corpus, split, root, purpose=purpose)
            meta = load_meta(corpus_store.raw_dir(corpus, split, root) / "meta.json")
            metric(f"ptb_raw_{split}_rows", meta["n_doc"])
            metric(f"ptb_raw_{split}_bytes_utf8", meta["n_bytes"], unit="bytes")
            for fname, digest in meta["sha256"].items():
                dataset_hash(f"{corpus}/raw/{split}/{fname}", digest)
        for V in vocab_sizes:
            tok = bpe.load_tokenizer(V, root)
            ws_lead, all_ws = bpe.token_whitespace_tables(tok)
            train_dir = corpus_store.split_dir(corpus, V, "train", root)
            for split in SPLITS:
                d = corpus_store.split_dir(corpus, V, split, root)
                if (d / "tok.npy").exists():
                    continue
                ids, doc_start = bpe.encode_docs(tok, raws[split].articles())
                wid = bpe.word_ids(ids, doc_start, ws_lead, all_ws)
                corpus_store.write_split_arrays(corpus, V, split, {"tok": ids, "doc_start": doc_start, "word_id": wid}, root)
            if not (train_dir / "kind_map.npy").exists():
                # PTB reuses the WikiText-103 train cluster codes (kind_map) and computes its own stop and sym_freq
                km = np.load(corpus_store.split_dir(CORPUS, V, "train", root) / "kind_map.npy")
                ids = np.load(train_dir / "tok.npy")
                stop = stop_of(ids, V, corpus_store.N_STOP)
                sym_freq = sym_freq_of(ids, km, V)
                for split in SPLITS:
                    corpus_store.write_split_arrays(corpus, V, split, {"kind_map": km, "stop": stop, "sym_freq": sym_freq}, root)
            for split in SPLITS:
                d = corpus_store.split_dir(corpus, V, split, root)
                if not (d / "targets.npy").exists():
                    n_tok = int(np.load(d / "doc_start.npy")[-1])
                    corpus_store.write_split_arrays(corpus, V, split, {"targets": np.arange(n_tok, dtype=np.uint32)}, root)
                if not (d / "meta.json").exists():
                    hashes = {f"{name}.npy": hashing.sha256_array(np.load(d / f"{name}.npy")) for name in corpus_store.STORE_ARRAYS}
                    n_tok = int(np.load(d / "doc_start.npy")[-1])
                    corpus_store.write_split_meta(corpus, V, split, {
                        "corpus": corpus, "vocab": V, "split": split, "n_tok": n_tok,
                        "n_doc": int(np.load(d / "doc_start.npy").shape[0] - 1),
                        "n_words": int((np.load(d / "word_id.npy") != corpus_store.NO_WORD).sum()),
                        "n_bytes_utf8": len(raws[split].text), "n_stop": corpus_store.N_STOP,
                        "clusters": {"note": "kind_map copied from wikitext103 train clusters"}, "sha256": hashes}, root)
                meta = load_meta(d / "meta.json")
                metric(f"ptb_store_{V}_{split}_n_tok", meta["n_tok"])
                for fname, digest in meta["sha256"].items():
                    dataset_hash(f"{corpus}/bpe{V}/{split}/{fname}", digest)
        metric("ptb_done", 1)
    except Exception as exc:  # noqa: BLE001 - the failure is recorded, not hidden
        err = "".join(traceback.format_exception(exc))
        log("PTB FAILED:\n" + err)
        write_run_json("ptb_error.json", {"error": repr(exc), "traceback": err})
        metric("ptb_done", 0, note=f"failed: {exc!r}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "M2_data.json"))
    ap.add_argument("--data-root", default=None, help="override the data root (tests, dry runs)")
    ap.add_argument("--prereg-file", default=str(ROOT / "docs" / "notes" / "preregistration-data-hashes.json"))
    ap.add_argument("--skip-ptb", action="store_true")
    ap.add_argument("--synthetic-rows", default=None, help="tests only: JSON {split: [rows]} instead of Hugging Face")
    ap.add_argument("--ptb-only", action="store_true", help="internal: run only the PTB step (child process, online)")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    params = config["params"]
    root = Path(args.data_root) if args.data_root else corpus_store.data_root()
    if args.data_root:
        os.environ["TMLM_DATA_DIR"] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = runlog.current_run_dir()
    t_all = time.time()
    if args.ptb_only:
        built = [V for V in params["vocab_sizes"] if (bpe.tokenizer_dir(V, root) / "meta.json").exists()]
        step_ptb(root, built)
        return 0

    raw = step_raw(root, config, Path(args.synthetic_rows) if args.synthetic_rows else None)
    vocab_sizes = step_bpe(root, raw, list(params["vocab_sizes"]), float(params["bpe_32000_time_limit_s"]))
    metric("vocab_sizes_built", vocab_sizes)
    for V in vocab_sizes:
        t0 = time.time()
        step_encode(root, raw, V)
        metric(f"encode_{V}_wall_s", time.time() - t0, unit="s")
        step_train_stats(root, V, int(params["cluster_seed"]), int(params["cluster_max_iter"]))
        step_evalsets(root, V)
    pair_vocab = int(params["pairs_vocab"])
    if pair_vocab in vocab_sizes:
        t0 = time.time()
        step_pairs(root, raw, pair_vocab, {k: int(v) for k, v in params["pair_sizes"].items()})
        metric("pairs_wall_s", time.time() - t0, unit="s")
    cont_vocab = int(params["contamination_vocab"])
    if cont_vocab in vocab_sizes:
        t0 = time.time()
        step_contamination(root, cont_vocab)
        metric("contamination_wall_s", time.time() - t0, unit="s")
    metric("test_access_count_after_build", evalsets.test_access_count(root),
           note="lines in data/test_access_log.jsonl, all written by this build")
    step_prereg_hashes(Path(args.prereg_file), run_dir)
    if not args.skip_ptb:
        # PTB is not in the local cache: the child runs WITHOUT the offline flags and with a timeout.
        # set to "0" explicitly: the child's own setdefault at import would otherwise re-enable offline mode
        env = {**os.environ, "HF_DATASETS_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}
        cmd = [sys.executable, __file__, "--config", args.config, "--data-root", str(root), "--ptb-only"]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, env=env, timeout=float(params.get("ptb_timeout_s", 1200)))
            metric("ptb_child_exit_code", proc.returncode)
        except subprocess.TimeoutExpired as exc:
            log(f"PTB child timed out: {exc!r}")
            write_run_json("ptb_error.json", {"error": repr(exc)})
            metric("ptb_done", 0, note=f"child timed out after {params.get('ptb_timeout_s', 1200)} s")
        metric("ptb_wall_s", time.time() - t0, unit="s")
    metric("total_wall_s", time.time() - t_all, unit="s")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
