"""CPU-only tests for experiments/p0_tmu_ae_repro.py (task T0.5 of the initial design).

The pure helpers (document splitting, target selection, tie-aware top-k credit, count
baselines, reference clause evaluation) are tested without importing tmu. One integration
test at the end builds a very small TMAutoEncoder through the in-memory numpy 2 shim and
checks that the C clause outputs agree with the numpy reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import p0_tmu_ae_repro as ae  # noqa: E402

SCRIPT = EXPERIMENTS / "p0_tmu_ae_repro.py"


# --------------------------------------------------------------------------- file sanity


def test_script_is_small_and_compiles():
    # Regression guard: a bad text replacement once blew this file up to 38 MB.
    text = SCRIPT.read_text(encoding="utf-8")
    assert len(text) < 200_000
    compile(text, str(SCRIPT), "exec")


# --------------------------------------------------------------------------- documents


def test_is_heading():
    assert ae.is_heading(" = Valkyria Chronicles III = \n")
    assert ae.is_heading(" = = Gameplay = = \n")
    assert not ae.is_heading(" The game began development in 2010 . \n")
    assert not ae.is_heading("")


def test_paragraphs_from_lines_filters_and_caps():
    long_a = " ".join(["alpha"] * 25)
    long_b = " ".join(["beta"] * 30)
    long_c = " ".join(["gamma"] * 40)
    lines = ["", " = Title = ", "too short", long_a, " = = Section = = ", long_b, long_c]
    assert ae.paragraphs_from_lines(lines, min_tokens=20, max_docs=10) == [long_a, long_b, long_c]
    assert ae.paragraphs_from_lines(lines, min_tokens=20, max_docs=2) == [long_a, long_b]
    assert ae.paragraphs_from_lines(lines, min_tokens=35, max_docs=10) == [long_c]


def test_clean_text_drops_unk():
    assert "unk" not in ae.clean_text("the <unk> of <unk>")


# --------------------------------------------------------------------------- target words


def test_select_target_columns_order_and_filters():
    terms = ["1999", "apple", "the", "banana", "cherry", "2nd"]
    doc_freq = [90, 10, 99, 30, 30, 50]
    cols = ae.select_target_columns(terms, doc_freq, {"the"}, 3)
    # "the" is a stopword, "1999" has no letter; "2nd" has one; ties go by term.
    assert cols.dtype == np.uint32
    assert [terms[c] for c in cols] == ["2nd", "banana", "cherry"]
    cols = ae.select_target_columns(terms, doc_freq, {"the"}, 2, exclude_nonalpha=False)
    assert [terms[c] for c in cols] == ["1999", "2nd"]


def test_select_target_columns_raises_when_too_few():
    with pytest.raises(ValueError):
        ae.select_target_columns(["a", "b"], [1, 2], {"a"}, 2)


# --------------------------------------------------------------------------- top-k credit


def test_topk_credit_without_ties():
    scores = np.array([5.0, 9.0, 1.0, 7.0])
    credits, rank = ae.topk_credit(scores, 1, (1, 5))
    assert credits == [1.0, 1.0] and rank == 1.0
    credits, rank = ae.topk_credit(scores, 3, (1, 2))
    assert credits == [0.0, 1.0] and rank == 2.0
    credits, rank = ae.topk_credit(scores, 2, (1, 3))
    assert credits == [0.0, 0.0] and rank == 4.0


def test_topk_credit_with_ties_is_the_expectation():
    # One strictly greater, the true word tied with two others: ranks 2, 3, 4 equally likely.
    scores = np.array([9, 3, 3, 3, 0])
    credits, rank = ae.topk_credit(scores, 2, (1, 2, 3, 4))
    assert credits == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert rank == pytest.approx(3.0)


def test_topk_credit_all_tied_equals_chance():
    scores = np.zeros(200)
    credits, rank = ae.topk_credit(scores, 17, (1, 5))
    assert credits == pytest.approx([1 / 200, 5 / 200])
    assert rank == pytest.approx(100.5)


def test_topk_credit_matches_monte_carlo_tie_breaking():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=30)
    true_idx = 11
    ks = (1, 5, 10)
    expected, _ = ae.topk_credit(scores, true_idx, ks)
    hits = np.zeros(len(ks))
    trials = 20000
    for _ in range(trials):
        order = np.lexsort((rng.random(30), -scores))  # score descending, random ties
        position = int(np.flatnonzero(order == true_idx)[0])
        hits += [position < k for k in ks]
    assert hits / trials == pytest.approx(expected, abs=0.015)


def test_topk_credit_mask():
    scores = np.array([9, 8, 7, 6])
    mask = np.array([False, False, True, True])
    credits, rank = ae.topk_credit(scores, 2, (1,), mask)
    assert credits == [1.0] and rank == 1.0
    with pytest.raises(ValueError):
        ae.topk_credit(scores, 0, (1,), mask)


# --------------------------------------------------------------------------- baselines


def _toy() -> tuple[csr_matrix, np.ndarray]:
    dense = np.array([
        [1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ], dtype=np.uint32)
    return csr_matrix(dense), np.array([0, 2], dtype=np.uint32)


def test_cooccurrence_counts_match_brute_force():
    X, targets = _toy()
    dense = X.toarray()
    C, N = ae.cooccurrence_counts(X, targets)
    assert C.shape == (2, 5)
    for ti, t in enumerate(targets):
        assert N[ti] == dense[:, t].sum()
        for j in range(5):
            assert C[ti, j] == np.count_nonzero((dense[:, t] > 0) & (dense[:, j] > 0))


def test_nb_scores_match_brute_force():
    X, targets = _toy()
    dense = X.toarray().astype(np.float64)
    C, N = ae.cooccurrence_counts(X, targets)
    alpha = 1.0
    params = ae.nb_bernoulli_params(C, N, targets, alpha)
    rows = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]], dtype=np.float64)
    got = ae.nb_scores(params, rows)
    for r, row in enumerate(rows):
        for ti, t in enumerate(targets):
            docs = dense[dense[:, t] > 0].copy()
            docs[:, t] = 0  # a training instance of class t has t removed
            p = (docs.sum(axis=0) + alpha) / (len(docs) + 2 * alpha)
            want = np.log(N[ti] / N.sum()) + np.sum(row * np.log(p) + (1 - row) * np.log(1 - p))
            assert got[r, ti] == pytest.approx(want)


def test_pmi_own_word_is_neutral_and_prior_is_used():
    X, targets = _toy()
    doc_freq = np.asarray(X.sum(axis=0)).reshape(-1)
    C, N = ae.cooccurrence_counts(X, targets)
    params = ae.pmi_params(C, N, doc_freq, X.shape[0], targets, 1.0)
    assert params["pmi"][0, targets[0]] == 0.0 and params["pmi"][1, targets[1]] == 0.0
    empty = np.zeros((1, 5))
    assert ae.pmi_scores(params, empty)[0] == pytest.approx(np.log(N / N.sum()))


def test_masked_rows():
    targets = np.array([0, 2, 3], dtype=np.uint32)
    row = np.array([1, 1, 0, 1, 1], dtype=np.uint32)
    present, rows = ae.masked_rows(row, targets)
    assert list(present) == [0, 2]  # indices into `targets`: words 0 and 3 are present
    assert rows.tolist() == [[0, 1, 0, 1, 1], [1, 1, 0, 0, 1]]
    assert row.tolist() == [1, 1, 0, 1, 1]  # the input row is not modified


# --------------------------------------------------------------------------- reference TM


def test_numpy_clause_outputs():
    # 3 features, literals = [x0, x1, x2, not x0, not x1, not x2]
    include = np.array([
        [1, 1, 0, 0, 0, 0],  # x0 and x1
        [0, 0, 0, 0, 0, 1],  # not x2
        [0, 0, 0, 0, 0, 0],  # empty: outputs 0 at prediction time
        [1, 0, 0, 1, 0, 0],  # x0 and not x0: never fires
    ], dtype=np.uint32)
    rows = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint32)
    assert ae.numpy_clause_outputs(include, rows).tolist() == [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ]


def test_linear_fit():
    fit = ae.linear_fit([1, 2, 3, 4], [3, 5, 7, 9])
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- shim


def test_patch_source_insists_on_the_occurrence_count():
    assert ae.patch_source("a ~0 b ~0", "~0", "X", 2) == "a X b X"
    with pytest.raises(RuntimeError):
        ae.patch_source("a ~0 b", "~0", "X", 2)
    with pytest.raises(RuntimeError):
        ae.patch_source("nothing here", "~0", "X", 1)


def test_shim_table_has_no_empty_pattern():
    # str.replace("", x) inserts x between all characters: never allow an empty pattern.
    for _module, _cls, _func, replacements in ae.NUMPY2_SHIM_PATCHES:
        assert replacements
        for old, new, expected in replacements:
            assert old and new and old != new and expected >= 1


def test_parse_overrides():
    assert ae.parse_overrides(["rounds=5", "s=2.5"]) == {"rounds": 5, "s": 2.5}
    with pytest.raises(SystemExit):
        ae.parse_overrides(["not_a_key=1"])
    with pytest.raises(SystemExit):
        ae.parse_overrides(["rounds"])


def test_tiny_autoencoder_trains_and_matches_the_numpy_reference():
    pytest.importorskip("tmu")
    rng = np.random.default_rng(5)
    n_docs, vocab = 300, 64
    dense = (rng.random((n_docs, vocab)) < 0.15).astype(np.uint32)
    dense[:, 1] = dense[:, 0]  # word 1 co-occurs perfectly with target word 0
    X = csr_matrix(dense)
    targets = np.array([0, 5, 9], dtype=np.uint32)
    cfg = dict(ae.DEFAULT_CONFIG, number_of_clauses=20, T=40, accumulation=1)

    model = ae.make_autoencoder(cfg, targets, seed=3)
    model.fit(X, number_of_examples=30)
    assert ae.apply_numpy2_shim() == []  # idempotent: a second call patches nothing

    W = ae.weight_matrix(model, len(targets))
    assert W.shape == (3, 20)
    include = model.clause_bank.get_literals()
    assert include.shape == (20, 2 * vocab)
    # feature_negation False: no negated literal may be included
    assert int(include[:, vocab:].sum()) == 0
    # No assertion on the clause size: max_included_literals is NOT a hard cap in tmu 0.8.3.
    # It is only handed to the compiled cb_type_i_feedback, and clauses with 4 and 5
    # included literals were observed here at max_included_literals=3. The measured runs
    # record the real maximum as tm_diag.max_included_literals_per_clause.

    rows = dense[:40].copy()
    got = ae.tm_clause_outputs(model, rows)
    want = ae.numpy_clause_outputs(include, rows)
    assert np.array_equal(got, want)
    assert len(ae.state_hash(model, len(targets))) == 64
