#!/usr/bin/env python3
"""T0.5 (milestone M1): reproduce the tmu TMAutoEncoder on a WikiText-103 subset, on CPU,
and measure masked-word recovery (top-1 / top-5) against count baselines.

Modes (each one creates its own run dir under results/ through tmlm.common.runlog):

    p0_tmu_ae_repro.py run  [--seed S] [--name NAME] [--set key=value ...]
        Train one TMAutoEncoder and evaluate it and the baselines on held-out documents.
    p0_tmu_ae_repro.py cost [--set key=value ...]
        Measure the training cost per example round for several numbers of target words
        (the "cost linear in the number of target words" premise of docs/02 V2).
    p0_tmu_ae_repro.py determinism [--set key=value ...]
        Train the same small config twice with the same seed in two fresh processes and
        compare the state hashes (is the CPU path bit-reproducible or not).

Evaluation protocol (identical for the TM and for every baseline):
    For each held-out document and each target word present in it, the word is removed
    from the binary bag of words. The model scores all target words from the remaining
    bag. The removed word is "recovered at k" when it ranks in the top k.
    Two candidate sets are reported:
      all      rank among ALL target words (the protocol as specified for T0.5);
      absent   rank among the target words that are NOT in the remaining bag. The bag is
               binary, so a word still in the bag cannot be the removed one.
    Ties (frequent: TM class sums are small integers) are resolved in expectation under
    uniform random tie-breaking: with g candidates strictly above the true word and t
    candidates tied with it, the top-k credit is clip((k - g) / (t + 1), 0, 1).

TM scores are class sums computed from public tmu methods only: the clause outputs come
from `clause_bank.prepare_X` + `clause_bank.calculate_clause_outputs_predict`, the
weights from `TMAutoEncoder.get_weights(i)` (tmu's own `predict` only exposes the sign).
A numpy re-implementation from `clause_bank.get_literals()` cross-checks the clause
outputs on a sample.

The heavy imports (tmu, datasets, sklearn) happen inside functions, so the pure helpers
can be unit-tested quickly (tests/test_p0tmu_helpers.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG: dict[str, Any] = {
    # data
    "dataset_id": "Salesforce/wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "train_split": "train",
    "heldout_split": "validation",
    "n_train_docs": 30000,
    "n_heldout_docs": 2000,
    "min_doc_tokens": 20,
    "vocab_size": 2000,
    "n_targets": 200,
    "stopwords": "sklearn ENGLISH_STOP_WORDS",
    "exclude_nonalpha_targets": True,
    # tmu TMAutoEncoder (IMDb demo: clauses 80, T 160, s 5.0, accumulation 25,
    # max_included_literals 3, feature_negation False, for 24 target words)
    "number_of_clauses": 800,
    "T": 1600,
    "s": 5.0,
    "accumulation": 25,
    "max_included_literals": 3,
    "output_balancing": 0.5,
    "feature_negation": False,
    "platform": "CPU",
    # One round = one fit() "example" = one update for EVERY target word. Shrunk from 2000
    # to 1500 to keep three repetitions plus the cost and determinism runs inside the
    # 40 minute compute box (about 0.39 s per round at 200 targets and 800 clauses).
    "rounds": 1500,
    "rounds_per_chunk": 250,
    "seed": 1,
    # evaluation (the curve uses the first 400 held-out documents only)
    "eval_every_chunks": 1,
    "nb_alpha": 1.0,
    "crosscheck_instances": 300,
    # cost mode
    "cost_target_counts": [25, 50, 100, 200],
    "cost_rounds": 40,
    "cost_warmup_rounds": 5,
    "cost_repeats": 3,
    # determinism mode
    "det_rounds": 30,
    "det_n_targets": 50,
}

TOP_KS = (1, 5)


# --------------------------------------------------------------------------- pure helpers


def is_heading(line: str) -> bool:
    """WikiText headings look like ' = Title = ' or ' = = Section = = '."""
    stripped = line.strip()
    return stripped.startswith("=") and stripped.endswith("=") and len(stripped) >= 2


def clean_text(text: str) -> str:
    """Drop the <unk> placeholder so that it does not become the vocabulary word 'unk'."""
    return text.replace("<unk>", " ")


def paragraphs_from_lines(
    lines: Iterable[str], min_tokens: int, max_docs: int
) -> list[str]:
    """The first `max_docs` paragraphs: non-empty, non-heading lines with at least
    `min_tokens` whitespace tokens. One paragraph is one document."""
    docs: list[str] = []
    for line in lines:
        if len(docs) >= max_docs:
            break
        if not line.strip() or is_heading(line):
            continue
        if len(line.split()) < min_tokens:
            continue
        docs.append(clean_text(line))
    return docs


def select_target_columns(
    terms: Sequence[str],
    doc_freq: Sequence[int],
    stopwords: Iterable[str],
    n_targets: int,
    exclude_nonalpha: bool = True,
) -> np.ndarray:
    """Column indices of the `n_targets` most frequent non-stopword terms.

    Order: document frequency descending, ties by term ascending. With
    `exclude_nonalpha`, terms without any alphabetic character (numbers) are skipped.
    """
    stop = set(stopwords)
    candidates = []
    for col, term in enumerate(terms):
        if term in stop:
            continue
        if exclude_nonalpha and not any(ch.isalpha() for ch in term):
            continue
        candidates.append((-int(doc_freq[col]), term, col))
    candidates.sort()
    if len(candidates) < n_targets:
        raise ValueError(f"only {len(candidates)} candidate target words, need {n_targets}")
    return np.array([col for _, _, col in candidates[:n_targets]], dtype=np.uint32)


def topk_credit(
    scores: np.ndarray, true_idx: int, ks: Sequence[int], mask: np.ndarray | None = None
) -> tuple[list[float], float]:
    """Expected top-k credit and expected rank of `true_idx` under random tie-breaking.

    `mask` (bool, same length) restricts the candidates; the true index must be inside.
    Returns ([credit for k in ks], expected rank starting at 1).
    """
    scores = np.asarray(scores)
    true_score = scores[true_idx]
    if mask is None:
        greater = int(np.count_nonzero(scores > true_score))
        tied = int(np.count_nonzero(scores == true_score)) - 1
    else:
        if not mask[true_idx]:
            raise ValueError("the true index is not a candidate")
        greater = int(np.count_nonzero((scores > true_score) & mask))
        tied = int(np.count_nonzero((scores == true_score) & mask)) - 1
    credits = [float(min(1.0, max(0.0, (k - greater) / (tied + 1)))) for k in ks]
    return credits, greater + 1 + tied / 2.0


def cooccurrence_counts(X_csr: Any, target_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C[t, j] = number of documents that contain target t and word j; N[t] = C[t, t]."""
    Xt = X_csr[:, target_cols]
    C = np.asarray((Xt.T @ X_csr).todense(), dtype=np.float64)
    N = np.asarray(Xt.sum(axis=0), dtype=np.float64).reshape(-1)
    return C, N


def nb_bernoulli_params(
    C: np.ndarray, N: np.ndarray, target_cols: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    """Bernoulli naive Bayes for the task "which target word was removed from this bag".

    A training instance of class t is a document that contains t, with t itself removed.
    So p[t, j] = P(word j in the bag | class t) = (C[t, j] + alpha) / (N[t] + 2 alpha)
    for j != t, and the own-word count is 0. The prior is proportional to N[t].
    """
    C = C.copy()
    C[np.arange(len(target_cols)), target_cols] = 0.0
    p = (C + alpha) / (N[:, None] + 2.0 * alpha)
    log_p = np.log(p)
    log_q = np.log1p(-p)
    return {
        "log_prior": np.log(N / N.sum()),
        "delta": log_p - log_q,  # added for every word in the bag
        "base": log_q.sum(axis=1),  # all words absent
    }


def nb_scores(params: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """rows: (n, V) 0/1. Returns (n, n_targets) Bernoulli NB log scores."""
    return params["log_prior"][None, :] + params["base"][None, :] + rows @ params["delta"].T


def pmi_params(
    C: np.ndarray, N: np.ndarray, doc_freq: np.ndarray, n_docs: int, target_cols: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Present-words-only co-occurrence score: log P(t) + sum over bag words of PMI(t, j),
    PMI(t, j) = log P(j | t) - log P(j), add-alpha smoothed. The own word gets PMI 0."""
    p_j_given_t = (C + alpha) / (N[:, None] + 2.0 * alpha)
    p_j = (np.asarray(doc_freq, dtype=np.float64) + alpha) / (n_docs + 2.0 * alpha)
    pmi = np.log(p_j_given_t) - np.log(p_j)[None, :]
    pmi[np.arange(len(target_cols)), target_cols] = 0.0
    return {"log_prior": np.log(N / N.sum()), "pmi": pmi}


def pmi_scores(params: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return params["log_prior"][None, :] + rows @ params["pmi"].T


def masked_rows(doc_row: np.ndarray, target_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For one dense 0/1 document row: the target indices present in it, and one copy of
    the row per present target with that target's column cleared."""
    present = np.flatnonzero(doc_row[target_cols] > 0)
    rows = np.repeat(doc_row[None, :], len(present), axis=0)
    rows[np.arange(len(present)), target_cols[present]] = 0
    return present, rows


def numpy_clause_outputs(include: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Reference clause evaluation. include: (clauses, 2V) 0/1 from get_literals();
    rows: (n, V) 0/1. A clause fires when every included literal is true, and a clause
    with no included literal outputs 0 at prediction time (tmu convention)."""
    literals = np.concatenate([rows, 1 - rows], axis=1).astype(np.int64)
    inc = include.astype(np.int64)
    violated = (1 - literals) @ inc.T  # number of included literals that are false
    nonempty = inc.sum(axis=1) > 0
    return ((violated == 0) & nonempty[None, :]).astype(np.uint32)


def linear_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Least squares y = a + b x, with R^2."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    b, a = np.polyfit(xa, ya, 1)
    pred = a + b * xa
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    return {"intercept": float(a), "slope": float(b), "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0}


# --------------------------------------------------------------------------- data


def load_documents(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset_id"], cfg["dataset_config"])
    # Over-read lines: roughly half of the lines are blank or headings.
    n_lines = min(len(ds[cfg["train_split"]]), cfg["n_train_docs"] * 6)
    train_lines = ds[cfg["train_split"]][:n_lines]["text"]
    train_docs = paragraphs_from_lines(train_lines, cfg["min_doc_tokens"], cfg["n_train_docs"])
    held_lines = ds[cfg["heldout_split"]][:]["text"]
    held_docs = paragraphs_from_lines(held_lines, cfg["min_doc_tokens"], cfg["n_heldout_docs"])
    return train_docs, held_docs


def build_data(cfg: dict[str, Any]) -> dict[str, Any]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    from tmlm.common import hashing

    train_docs, held_docs = load_documents(cfg)
    vectorizer = CountVectorizer(binary=True, max_features=cfg["vocab_size"])
    X_train = vectorizer.fit_transform(train_docs).astype(np.uint32).tocsr()
    X_held = vectorizer.transform(held_docs).astype(np.uint32).tocsr()
    X_train.sort_indices()
    X_held.sort_indices()
    terms = list(vectorizer.get_feature_names_out())
    doc_freq = np.asarray(X_train.sum(axis=0)).reshape(-1)
    target_cols = select_target_columns(
        terms, doc_freq, ENGLISH_STOP_WORDS, cfg["n_targets"], cfg["exclude_nonalpha_targets"]
    )
    digest = hashlib.sha256()
    for doc in train_docs:
        digest.update(doc.encode("utf-8"))
        digest.update(b"\n")
    held_digest = hashlib.sha256()
    for doc in held_docs:
        held_digest.update(doc.encode("utf-8"))
        held_digest.update(b"\n")
    return {
        "X_train": X_train,
        "X_held": X_held,
        "terms": terms,
        "doc_freq": doc_freq,
        "target_cols": target_cols,
        "hashes": {
            "train_docs_text": digest.hexdigest(),
            "heldout_docs_text": held_digest.hexdigest(),
            "X_train_indices": hashing.sha256_array(X_train.indices),
            "X_train_indptr": hashing.sha256_array(X_train.indptr),
            "X_heldout_indices": hashing.sha256_array(X_held.indices),
            "X_heldout_indptr": hashing.sha256_array(X_held.indptr),
        },
        "n_train_docs": len(train_docs),
        "n_heldout_docs": len(held_docs),
    }


# --------------------------------------------------------------------------- numpy 2 shim

# tmu 0.8.3 writes the all-ones uint32 word as `~0` (the Python int -1). numpy 2 (NEP 50)
# refuses to convert -1 to uint32 and raises OverflowError, so the stock TMAutoEncoder
# cannot even be constructed under the installed numpy 2.4.6. The installed package must
# not be edited and nothing may be installed, so the two affected functions are re-compiled
# IN MEMORY from their own upstream source with exactly these expressions replaced by the
# same value written as 0xFFFFFFFF. Nothing else changes and no file is touched.
NUMPY2_SHIM_PATCHES = (
    # (module, class, function, ((old text, new text, expected occurrences), ...))
    ("tmu.clause_bank.clause_bank", "ClauseBank", "initialize_clauses", (
        ("np.uint32(~0)", "np.uint32(0xFFFFFFFF)", 2),
    )),
    ("tmu.models.autoencoder.autoencoder", "TMAutoEncoder", "update", (
        ("dtype=np.uint32) | ~0)", "dtype=np.uint32) | np.uint32(0xFFFFFFFF))", 1),
    )),
    ("tmu.models.autoencoder.autoencoder", "TMAutoEncoder", "activate_literals", (
        ("&= (~(1 << chunk_pos))", "&= np.uint32((~(1 << chunk_pos)) & 0xFFFFFFFF)", 1),
    )),
    ("tmu.models.autoencoder.autoencoder", "TMAutoEncoder", "fit", (
        ("&= ~(1 << chunk_pos_negated)", "&= np.uint32((~(1 << chunk_pos_negated)) & 0xFFFFFFFF)", 1),
        ("&= ~(1 << chunk_pos)\n", "&= np.uint32((~(1 << chunk_pos)) & 0xFFFFFFFF)\n", 1),
    )),
)


def patch_source(source: str, old: str, new: str, expected: int) -> str:
    """Replace `old` by `new` in `source`, insisting on the expected occurrence count."""
    found = source.count(old)
    if found != expected:
        raise RuntimeError(f"shim: expected {expected} occurrence(s) of {old!r}, found {found}")
    return source.replace(old, new)


def apply_numpy2_shim() -> list[dict[str, Any]]:
    """Apply NUMPY2_SHIM_PATCHES when numpy >= 2. Idempotent. Returns what was done."""
    import importlib
    import inspect
    import textwrap

    applied: list[dict[str, Any]] = []
    if int(np.__version__.split(".")[0]) < 2:
        return applied
    for module_name, class_name, func_name, replacements in NUMPY2_SHIM_PATCHES:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        func = getattr(cls, func_name)
        if getattr(func, "_tmlm_numpy2_shim", False):
            continue
        source = textwrap.dedent(inspect.getsource(func))
        code_only = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
        if not any(code_only.count(old) for old, _, _ in replacements):
            # tmu at or after upstream commit 62c0e6a6 (2024-11-22) already carries the NumPy 2 fix in
            # its CODE (its comments still quote the old expression); the project pins commit 5605ff07
            # since 2026-09-22 (CHANGELOG "tmu pin"). Nothing to patch: mark the function and move on.
            func._tmlm_numpy2_shim = True
            continue
        patched = source
        for old, new, expected in replacements:
            patched = patch_source(patched, old, new, expected)
        scope: dict[str, Any] = {}
        exec(compile(patched, f"<numpy2 shim of {module_name}.{class_name}.{func_name}>", "exec"),
             module.__dict__, scope)
        new_func = scope[func_name]
        new_func._tmlm_numpy2_shim = True
        setattr(cls, func_name, new_func)
        applied.append({
            "function": f"{module_name}.{class_name}.{func_name}",
            "replacements": [{"old": o.strip(), "new": n.strip(), "occurrences": c}
                             for o, n, c in replacements],
            "upstream_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        })
    return applied


# --------------------------------------------------------------------------- TM wrappers


def make_autoencoder(cfg: dict[str, Any], target_cols: np.ndarray, seed: int | None) -> Any:
    from tmu.models.autoencoder.autoencoder import TMAutoEncoder

    for patch in apply_numpy2_shim():
        print(f"[shim] numpy {np.__version__}: in-memory patch {json.dumps(patch)}", flush=True)
    return TMAutoEncoder(
        cfg["number_of_clauses"],
        cfg["T"],
        cfg["s"],
        np.ascontiguousarray(target_cols, dtype=np.uint32),
        accumulation=cfg["accumulation"],
        max_included_literals=cfg["max_included_literals"],
        output_balancing=cfg["output_balancing"],
        feature_negation=cfg["feature_negation"],
        platform=cfg["platform"],
        seed=seed,
    )


def weight_matrix(model: Any, n_targets: int) -> np.ndarray:
    return np.stack([np.asarray(model.get_weights(i), dtype=np.int64) for i in range(n_targets)])


def state_hash(model: Any, n_targets: int) -> str:
    """sha256 over the TA state bits of the clause bank and the weight matrix."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(model.clause_bank.clause_bank).tobytes())
    digest.update(np.ascontiguousarray(weight_matrix(model, n_targets)).tobytes())
    return digest.hexdigest()


def tm_clause_outputs(model: Any, rows: np.ndarray) -> np.ndarray:
    """(n, clauses) 0/1 clause outputs for dense 0/1 rows, through tmu's C code."""
    rows = np.ascontiguousarray(rows, dtype=np.uint32)
    encoded = model.clause_bank.prepare_X(rows)
    out = np.empty((rows.shape[0], model.number_of_clauses), dtype=np.int64)
    for e in range(rows.shape[0]):
        out[e, :] = model.clause_bank.calculate_clause_outputs_predict(encoded, e)
    return out


# --------------------------------------------------------------------------- evaluation


class Tally:
    """Accumulates expected top-k credit and rank for one (method, candidate set)."""

    def __init__(self) -> None:
        self.n = 0
        self.credit = np.zeros(len(TOP_KS), dtype=np.float64)
        self.rank_sum = 0.0

    def add(self, credits: Sequence[float], rank: float) -> None:
        self.n += 1
        self.credit += np.asarray(credits)
        self.rank_sum += rank

    def result(self) -> dict[str, float]:
        out = {f"top{k}": float(self.credit[i] / self.n) for i, k in enumerate(TOP_KS)}
        out["mean_rank"] = float(self.rank_sum / self.n)
        out["n_instances"] = self.n
        return out


def evaluate(
    data: dict[str, Any],
    cfg: dict[str, Any],
    model: Any | None,
    max_docs: int | None = None,
) -> dict[str, Any]:
    """Masked-word recovery of the TM (when `model` is given) and the baselines."""
    target_cols = data["target_cols"]
    n_targets = len(target_cols)
    X_train, X_held = data["X_train"], data["X_held"]
    C, N = cooccurrence_counts(X_train, target_cols)
    nb = nb_bernoulli_params(C, N, target_cols, cfg["nb_alpha"])
    pmi = pmi_params(C, N, data["doc_freq"], X_train.shape[0], target_cols, cfg["nb_alpha"])
    unigram = np.log(N / N.sum())

    methods = ["unigram", "pmi_sum", "nb_bernoulli"] + (["tm_ae"] if model is not None else [])
    tallies = {(m, c): Tally() for m in methods for c in ("all", "absent")}
    W = weight_matrix(model, n_targets) if model is not None else None
    include = model.clause_bank.get_literals() if model is not None else None

    fired_sum = 0.0
    zero_sum_instances = 0
    distinct_sum_values = 0.0
    crosscheck_left = cfg["crosscheck_instances"] if model is not None else 0
    crosscheck_checked = 0
    crosscheck_mismatch = 0
    n_docs_used = 0
    t0 = time.time()
    n_docs = X_held.shape[0] if max_docs is None else min(max_docs, X_held.shape[0])
    for d in range(n_docs):
        doc_row = np.asarray(X_held[d].todense(), dtype=np.uint32).reshape(-1)
        present, rows = masked_rows(doc_row, target_cols)
        if len(present) == 0:
            continue
        n_docs_used += 1
        rows_f = rows.astype(np.float64)
        scores = {
            "unigram": np.repeat(unigram[None, :], len(present), axis=0),
            "pmi_sum": pmi_scores(pmi, rows_f),
            "nb_bernoulli": nb_scores(nb, rows_f),
        }
        if model is not None:
            co = tm_clause_outputs(model, rows)
            scores["tm_ae"] = co @ W.T
            fired_sum += float(co.sum())
            zero_sum_instances += int(np.count_nonzero(~scores["tm_ae"].any(axis=1)))
            distinct_sum_values += float(sum(len(np.unique(r)) for r in scores["tm_ae"]))
            if crosscheck_left > 0:
                take = min(crosscheck_left, len(present))
                ref = numpy_clause_outputs(include, rows[:take])
                crosscheck_mismatch += int(np.count_nonzero(ref != co[:take]))
                crosscheck_checked += take
                crosscheck_left -= take
        still_present = rows[:, target_cols] > 0  # (n_present, n_targets)
        for i, t in enumerate(present):
            absent_mask = ~still_present[i]
            for m in methods:
                credits, rank = topk_credit(scores[m][i], int(t), TOP_KS)
                tallies[(m, "all")].add(credits, rank)
                credits, rank = topk_credit(scores[m][i], int(t), TOP_KS, absent_mask)
                tallies[(m, "absent")].add(credits, rank)
        if (d + 1) % 500 == 0:
            print(f"[eval] {d + 1}/{n_docs} docs, {time.time() - t0:.1f} s", flush=True)

    result: dict[str, Any] = {
        "n_heldout_docs_with_target": n_docs_used,
        "eval_wall_s": time.time() - t0,
        "methods": {m: {c: tallies[(m, c)].result() for c in ("all", "absent")} for m in methods},
    }
    n_inst = tallies[("unigram", "all")].n
    result["n_instances"] = n_inst
    result["mean_targets_present_per_doc"] = n_inst / max(1, n_docs_used)
    if model is not None:
        result["tm_diagnostics"] = {
            "mean_clauses_fired_per_instance": fired_sum / max(1, n_inst),
            "fraction_instances_all_class_sums_zero": zero_sum_instances / max(1, n_inst),
            "mean_distinct_class_sum_values_per_instance": distinct_sum_values / max(1, n_inst),
            "crosscheck_instances": crosscheck_checked,
            "crosscheck_clause_output_mismatches": crosscheck_mismatch,
            "mean_included_literals_per_clause": float(include.sum(axis=1).mean()),
            "max_included_literals_per_clause": int(include.sum(axis=1).max()),
            "clauses_with_no_included_literal": int(np.count_nonzero(include.sum(axis=1) == 0)),
        }
    return result


# --------------------------------------------------------------------------- modes


def parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in DEFAULT_CONFIG:
            raise SystemExit(f"bad --set {pair!r}: use key=value with a key of DEFAULT_CONFIG")
        out[key] = json.loads(value)
    return out


def describe_nondeterminism() -> dict[str, Any]:
    return {
        "numpy_rng": "tmu seeds np.random.RandomState(seed) and its pcg32/xorshift C streams "
        "from the constructor's seed",
        "c_rand": "tmu_produce_autoencoder_example draws the documents with C rand(); the "
        "library never calls srand(), so the stream is whatever the process-wide "
        "glibc state is (default seed 1 in a fresh process)",
    }


def mode_run(cfg: dict[str, Any], name: str) -> int:
    from tmlm.common.runlog import RunLog

    cfg = dict(cfg, mode="run", rng_notes=describe_nondeterminism())
    with RunLog(name, cfg, sample_gpu=False) as run:
        print(f"[run] config: {json.dumps(cfg, sort_keys=True)}", flush=True)
        t0 = time.time()
        data = build_data(cfg)
        for key, digest in data["hashes"].items():
            run.add_dataset_hash(key, digest)
        target_terms = [data["terms"][c] for c in data["target_cols"]]
        print(f"[run] docs: train {data['n_train_docs']}, held-out {data['n_heldout_docs']}; "
              f"vocab {len(data['terms'])}; data build {time.time() - t0:.1f} s", flush=True)
        print(f"[run] first 40 target words: {target_terms[:40]}", flush=True)
        run.write_json("target_words.json", {
            "target_words": target_terms,
            "target_doc_freq": [int(data["doc_freq"][c]) for c in data["target_cols"]],
        })

        model = make_autoencoder(cfg, data["target_cols"], cfg["seed"])
        n_targets = len(data["target_cols"])
        chunks = []
        rounds_done = 0
        curve = []
        train_t0 = time.time()
        chunk_index = 0
        while rounds_done < cfg["rounds"]:
            n = min(cfg["rounds_per_chunk"], cfg["rounds"] - rounds_done)
            c0 = time.perf_counter()
            model.fit(data["X_train"], number_of_examples=n)
            dt = time.perf_counter() - c0
            rounds_done += n
            chunk_index += 1
            chunks.append({"rounds": n, "wall_s": dt})
            W = weight_matrix(model, n_targets)
            print(f"[train] rounds {rounds_done}/{cfg['rounds']}  chunk {dt:.1f} s  "
                  f"{1e3 * dt / (n * n_targets):.3f} ms per target update  "
                  f"mean|w| {np.abs(W).mean():.2f}  max|w| {np.abs(W).max()}", flush=True)
            if cfg["eval_every_chunks"] and chunk_index % cfg["eval_every_chunks"] == 0 \
                    and rounds_done < cfg["rounds"]:
                mid = evaluate(data, cfg, model, max_docs=400)
                curve.append({"rounds": rounds_done, "tm_ae": mid["methods"]["tm_ae"]})
                print(f"[curve] rounds {rounds_done}: {json.dumps(mid['methods']['tm_ae'])}",
                      flush=True)
        train_wall = time.time() - train_t0
        fit_s = sum(c["wall_s"] for c in chunks)

        result = evaluate(data, cfg, model)
        result["curve_first_400_docs"] = curve
        result["train_chunks"] = chunks
        result["state_hash"] = state_hash(model, n_targets)
        run.write_json("eval.json", result)
        print(f"[run] result: {json.dumps(result['methods'], indent=1)}", flush=True)
        print(f"[run] tm diagnostics: {json.dumps(result['tm_diagnostics'])}", flush=True)
        print(f"[run] state hash: {result['state_hash']}", flush=True)

        note = "expected credit under uniform random tie-breaking"
        for method, per_set in result["methods"].items():
            for cand, vals in per_set.items():
                for k in TOP_KS:
                    run.add_metric(f"{method}.{cand}.top{k}", vals[f"top{k}"], "measured", note=note)
                run.add_metric(f"{method}.{cand}.mean_rank", vals["mean_rank"], "measured")
        run.add_metric("n_instances", result["n_instances"], "measured")
        run.add_metric("n_heldout_docs_with_target", result["n_heldout_docs_with_target"], "measured")
        for key, value in result["tm_diagnostics"].items():
            run.add_metric(f"tm_diag.{key}", value, "measured")
        run.add_metric("train.fit_wall_s", fit_s, "measured", unit="s",
                       note="sum of the fit() calls, single thread, other parallel jobs active on the host")
        run.add_metric("train.wall_s_incl_progress", train_wall, "measured", unit="s")
        run.add_metric("train.ms_per_target_update", 1e3 * fit_s / (cfg["rounds"] * n_targets),
                       "derived", unit="ms", note="fit_wall_s / (rounds * n_targets)")
        run.add_metric("train.s_per_round", fit_s / cfg["rounds"], "derived", unit="s")
        run.add_metric("eval.wall_s", result["eval_wall_s"], "measured", unit="s")
        run.add_metric("state_hash", result["state_hash"], "measured")
        if result["tm_diagnostics"]["crosscheck_clause_output_mismatches"] != 0:
            print("[run] ERROR: numpy cross-check of the clause outputs found mismatches",
                  flush=True)
            run.set_exit_status(3)
            return 3
    return 0


def mode_cost(cfg: dict[str, Any], name: str) -> int:
    from tmlm.common.runlog import RunLog

    cfg = dict(cfg, mode="cost")
    with RunLog(name, cfg, sample_gpu=False) as run:
        print(f"[cost] config: {json.dumps(cfg, sort_keys=True)}", flush=True)
        data = build_data(dict(cfg, n_targets=max(cfg["cost_target_counts"])))
        for key, digest in data["hashes"].items():
            run.add_dataset_hash(key, digest)
        rows = []
        for n_targets in cfg["cost_target_counts"]:
            cols = data["target_cols"][:n_targets]
            for rep in range(cfg["cost_repeats"]):
                model = make_autoencoder(cfg, cols, cfg["seed"] + rep)
                model.fit(data["X_train"], number_of_examples=cfg["cost_warmup_rounds"])
                o0 = time.perf_counter()
                model.fit(data["X_train"], number_of_examples=0)
                overhead = time.perf_counter() - o0
                c0 = time.perf_counter()
                model.fit(data["X_train"], number_of_examples=cfg["cost_rounds"])
                dt = time.perf_counter() - c0
                per_round = (dt - overhead) / cfg["cost_rounds"]
                rows.append({
                    "n_targets": n_targets, "repeat": rep, "fit_wall_s": dt,
                    "fit_call_overhead_s": overhead, "s_per_round": per_round,
                    "ms_per_target_update": 1e3 * per_round / n_targets,
                })
                print(f"[cost] targets {n_targets:4d} rep {rep}: {per_round * 1e3:9.2f} ms per round, "
                      f"{1e3 * per_round / n_targets:.4f} ms per target update "
                      f"(fit call overhead {overhead * 1e3:.1f} ms)", flush=True)
        run.write_json("cost_rows.json", rows)
        xs, ys = [], []
        for n_targets in cfg["cost_target_counts"]:
            vals = [r["s_per_round"] for r in rows if r["n_targets"] == n_targets]
            med = float(np.median(vals))
            xs.append(n_targets)
            ys.append(med)
            run.add_metric(f"s_per_round.targets_{n_targets}.median", med, "measured", unit="s",
                           note=f"median of {len(vals)} repeats, min {min(vals):.5f}, max {max(vals):.5f}")
            run.add_metric(f"ms_per_target_update.targets_{n_targets}", 1e3 * med / n_targets,
                           "derived", unit="ms")
        fit = linear_fit(xs, ys)
        run.add_metric("linear_fit.slope_s_per_round_per_target", fit["slope"], "derived", unit="s")
        run.add_metric("linear_fit.intercept_s_per_round", fit["intercept"], "derived", unit="s")
        run.add_metric("linear_fit.r2", fit["r2"], "derived")
        ratio = (ys[-1] / ys[0]) / (xs[-1] / xs[0])
        run.add_metric("cost_ratio_over_target_ratio", ratio, "derived",
                       note="(time at most targets / time at fewest) / (most / fewest); 1.0 is exactly linear")
        print(f"[cost] linear fit: {json.dumps(fit)}; cost ratio over target ratio {ratio:.3f}", flush=True)
    return 0


def _child_state_hash(cfg: dict[str, Any]) -> None:
    """Runs in a fresh process: train a small model, print its state hash as JSON."""
    data = build_data(cfg)
    model = make_autoencoder(cfg, data["target_cols"], cfg["seed"])
    model.fit(data["X_train"], number_of_examples=cfg["det_rounds"])
    print("CHILD_RESULT " + json.dumps({"state_hash": state_hash(model, len(data["target_cols"]))}),
          flush=True)


def mode_determinism(cfg: dict[str, Any], name: str) -> int:
    from tmlm.common.runlog import RunLog

    cfg = dict(cfg, mode="determinism", n_targets=cfg["det_n_targets"],
               rng_notes=describe_nondeterminism())
    with RunLog(name, cfg, sample_gpu=False) as run:
        print(f"[det] config: {json.dumps(cfg, sort_keys=True)}", flush=True)
        hashes: dict[str, str] = {}
        for label, seed in (("seed_a_run1", cfg["seed"]), ("seed_a_run2", cfg["seed"]),
                            ("seed_b_run1", cfg["seed"] + 1), ("seed_none_run1", None),
                            ("seed_none_run2", None)):
            child_cfg = dict(cfg, seed=seed)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "_child", json.dumps(child_cfg)],
                capture_output=True, text=True, check=False,
            )
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("CHILD_RESULT ")]
            if proc.returncode != 0 or not line:
                print(f"[det] child {label} failed: exit {proc.returncode}\n{proc.stderr[-2000:]}",
                      flush=True)
                run.set_exit_status(4)
                return 4
            hashes[label] = json.loads(line[-1][len("CHILD_RESULT "):])["state_hash"]
            print(f"[det] {label} (seed {seed}): {hashes[label]}", flush=True)
        run.write_json("state_hashes.json", hashes)
        same_seed_equal = hashes["seed_a_run1"] == hashes["seed_a_run2"]
        run.add_metric("same_seed_two_processes_identical_state", same_seed_equal, "measured")
        run.add_metric("different_seed_changes_state", hashes["seed_a_run1"] != hashes["seed_b_run1"],
                       "measured")
        run.add_metric("seed_none_two_processes_identical_state",
                       hashes["seed_none_run1"] == hashes["seed_none_run2"], "measured")
        for label, digest in hashes.items():
            run.add_metric(f"state_hash.{label}", digest, "measured")
        print(f"[det] same seed, two fresh processes, identical state: {same_seed_equal}", flush=True)
    return 0


def main(argv: Sequence[str]) -> int:
    if len(argv) >= 2 and argv[0] == "_child":
        _child_state_hash(json.loads(argv[1]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("run", "cost", "determinism"))
    parser.add_argument("--name", default=None, help="run name (default p0_tmu_ae_<mode>)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args(argv)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(parse_overrides(args.set))
    if args.seed is not None:
        cfg["seed"] = args.seed
    name = args.name or f"p0_tmu_ae_{args.mode}"
    return {"run": mode_run, "cost": mode_cost, "determinism": mode_determinism}[args.mode](cfg, name)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
