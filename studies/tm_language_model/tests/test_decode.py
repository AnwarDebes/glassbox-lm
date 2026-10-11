"""Decoders (docs/04 M8; docs/03 "Track C: decoders"): incremental re-evaluation identical to
the C engine's full re-evaluation (T2.3), calibrated logprobs that sum to one (docs/05 CP),
deterministic samplers that give the same tokens on the engine and on the incremental
evaluator, the elimination decoder's three-valued sets, and the ELBO / any-order NLL."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import toy_corpus
from tmlm.decode.anyorder import anyorder_generate, sequential_nll
from tmlm.decode.bank import Bank, WindowEval, bank_sha256
from tmlm.decode.conftm import Calibration, conformal_qhat, conformal_report, ece, fit_two_scalar, mean_nll
from tmlm.decode.diffusion import diffusion_generate, elbo_nll
from tmlm.decode.eliminate import TRUE, UNKNOWN, clause_truth3, elimination_generate
from tmlm.decode.incremental import IncrementalEvaluator
from tmlm.decode.scorer import TrainedModel
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine


def _train(mode: str, seed: int = 1, epochs: int = 4):
    corpus = toy_corpus(vocab=40, n_docs=6, doc_len=80, seed=7)
    cfg = EngineConfig(mode=mode, vocab=40, n_clauses=256, seed=seed, batch=32, sweep_period=8, n_bag_k=2)
    eng = CpuEngine(cfg, corpus=corpus)
    for e in range(epochs):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    model = TrainedModel.from_engine(eng, corpus.tok)
    pos = np.arange(0, corpus.tok.shape[0], 3, dtype=np.uint32)
    cal, _ = fit_two_scalar(eng.predict_scores(0, pos), corpus.tok[pos], model.class_bias)
    return corpus, eng, model.with_calibration(cal)


@pytest.fixture(scope="module")
def masked():
    return _train("masked")


@pytest.fixture(scope="module")
def causal():
    return _train("causal")


def _check_equal(wm, inc, bank):
    op = wm.open_positions()
    assert np.array_equal(wm.scores(op), inc.scores(op))
    for p in op:
        f_engine, f_inc = wm.fired(int(p)), inc.fired(int(p))
        assert np.array_equal(f_engine, f_inc), (int(p), f_engine, f_inc)
        we = WindowEval(bank, wm.tokens.copy(), wm.hidden.copy())
        assert np.array_equal(we.fired_all(int(p)), f_engine)


@pytest.mark.parametrize("which", ["masked", "causal"])
def test_incremental_reevaluation_is_identical_to_full_reevaluation(which, masked, causal):
    corpus, eng, model = masked if which == "masked" else causal
    L = 40
    wm, inc = model.bind(L), IncrementalEvaluator(model, L)
    rng = np.random.default_rng(0)  # test data only; nothing here reaches a model output
    tokens = rng.integers(0, 40, L).astype(np.uint16)
    hidden = np.ones(L, dtype=np.uint8)
    wm.set_window(tokens, hidden)
    inc.set_window(tokens, hidden)
    bank = model.bank()
    _check_equal(wm, inc, bank)
    order = np.arange(L) if which == "causal" else rng.permutation(L)
    for i, q in enumerate(order):
        wm.set_token(int(q), int(tokens[q]))
        inc.set_token(int(q), int(tokens[q]))
        _check_equal(wm, inc, bank)
        if which == "masked" and i % 7 == 3:
            r = int(order[rng.integers(0, i + 1)])
            wm.clear(r)
            inc.clear(r)
            _check_equal(wm, inc, bank)
    assert inc.n_verified > 0


def test_incremental_scores_visible_positions_like_the_engine(masked):
    corpus, eng, model = masked
    L = 24
    wm, inc = model.bind(L), IncrementalEvaluator(model, L)
    hidden = np.zeros(L, dtype=np.uint8)
    hidden[[3, 9, 17]] = 1
    wm.set_window(corpus.tok[:L], hidden)
    inc.set_window(corpus.tok[:L], hidden)
    assert np.array_equal(wm.scores(np.arange(L)), inc.scores(np.arange(L)))


def test_window_model_matches_the_training_engine_on_a_corpus_document(masked):
    """The bound window scores a document exactly as the training engine scores the same
    positions with the same hidden bitmap (docs/11 section 4 is one function of the window)."""
    corpus, eng, model = masked
    d0, d1 = int(corpus.doc_start[1]), int(corpus.doc_start[2])
    L = d1 - d0
    hidden_full = np.zeros(corpus.tok.shape[0], dtype=np.uint8)
    hidden_full[d0 + 5] = hidden_full[d0 + 20] = 1
    wm = model.bind(L)
    wm.set_window(corpus.tok[d0:d1], hidden_full[d0:d1])
    pos = np.array([5, 20], dtype=np.int64)
    assert np.array_equal(wm.scores(pos), eng.predict_scores(0, (pos + d0).astype(np.uint32), hidden_full))


def test_export_bank_has_the_interface_keys_and_a_stable_hash(masked, tmp_path):
    corpus, eng, model = masked
    d = model.export_bank()
    for k in ("format", "vocab_size", "state_hash", "lit_ptr", "lit_id", "vote_ptr", "vote_cls", "vote_w"):
        assert k in d
    assert d["state_hash"] == eng.state_hash()
    h = model.bank().save(tmp_path / "bank.npz")
    loaded = Bank.load(tmp_path / "bank.npz")
    assert bank_sha256(loaded.export()) == h == bank_sha256(d)


def test_calibrated_logprobs_sum_to_one_and_the_fit_beats_identity(masked):
    corpus, eng, model = masked
    pos = np.arange(1, corpus.tok.shape[0], 5, dtype=np.uint32)
    scores = eng.predict_scores(0, pos)
    gold = corpus.tok[pos].astype(np.int64)
    lp = model.calibration.logprobs(scores, model.class_bias)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(lp))
    assert mean_nll(scores, gold, model.class_bias, model.calibration) < mean_nll(scores, gold, model.class_bias, Calibration.identity())
    assert model.calibration.a_q16 >= 0
    rep = ece(np.exp(lp), gold)
    assert 0.0 <= rep["ece"] <= 1.0
    q = conformal_qhat(np.exp(lp[: len(lp) // 2]), gold[: len(lp) // 2], alpha=0.2)
    cr = conformal_report(np.exp(lp[len(lp) // 2:]), gold[len(lp) // 2:], q)
    assert 1.0 <= cr["mean_set_size"] <= 40.0 and 0.0 <= cr["coverage"] <= 1.0


def test_greedy_decision_is_exact_integer_argmax(masked):
    corpus, eng, model = masked
    scores = eng.predict_scores(0, np.arange(0, 60, 2, dtype=np.uint32))
    am = model.calibration.argmax(scores, model.class_bias)
    lp = model.calibration.logprobs(scores, model.class_bias)
    assert np.array_equal(am, lp.argmax(axis=1))


@pytest.mark.parametrize("decoder", ["anyorder_l2r", "anyorder_random", "anyorder_confidence", "greedy", "diffusion", "diffusion_remask", "eliminate"])
def test_samplers_are_deterministic_and_agree_between_engine_and_incremental(masked, decoder):
    corpus, eng, model = masked
    prompt = corpus.tok[:8]

    def run(scorer):
        if decoder.startswith("anyorder"):
            return anyorder_generate(scorer, prompt, 12, seed=3, order=decoder.split("_", 1)[1].replace("l2r", "left_to_right"))
        if decoder == "greedy":
            return anyorder_generate(scorer, prompt, 12, seed=3, greedy=True)
        if decoder == "diffusion":
            return diffusion_generate(scorer, prompt, 12, seed=3, steps=4)
        if decoder == "diffusion_remask":
            return diffusion_generate(scorer, prompt, 12, seed=3, steps=5, remask_frac=0.25)
        return elimination_generate(scorer, prompt, 12, seed=3)

    g1, g2, g3 = run(model.bind(20)), run(model.bind(20)), run(IncrementalEvaluator(model, 20))
    assert np.array_equal(g1.tokens, g2.tokens)
    assert np.array_equal(g1.tokens, g3.tokens)
    assert np.array_equal(g1.tokens[:8], prompt) and g1.generated.shape == (12,)
    assert len(g1.records) >= 12
    for r in g1.records:
        assert r.hidden[r.position] == 1
        assert np.array_equal(model.bank().sums(r.fired), r.sums)


def test_causal_model_decodes_left_to_right_only(causal):
    corpus, eng, model = causal
    g = anyorder_generate(model.bind(20), corpus.tok[:8], 12, seed=1)
    assert g.generated.shape == (12,)
    with pytest.raises(ValueError):
        anyorder_generate(model.bind(20), corpus.tok[:8], 12, seed=1, order="random")
    with pytest.raises(ValueError):
        diffusion_generate(model.bind(20), corpus.tok[:8], 12, seed=1)


def test_three_valued_truth_equals_two_valued_firing_on_complete_windows(masked):
    corpus, eng, model = masked
    bank = model.bank()
    L = 30
    for p in range(L):
        we = WindowEval(bank, corpus.tok[:L].copy(), np.zeros(L, dtype=np.uint8))
        we.hidden[p] = 1
        t3 = clause_truth3(bank, we, p)
        assert not np.any(t3 == UNKNOWN)
        assert np.array_equal(np.flatnonzero(t3 == TRUE), we.fired_all(p).astype(np.int64))


def test_elimination_claims_survivors_and_records_sets(masked):
    corpus, eng, model = masked
    g = elimination_generate(model.bind(20), corpus.tok[:8], 12, seed=5)
    for r in g.records:
        assert r.survivors[r.claimed]
        assert r.licensing is not None and r.eliminating is not None
        for j in r.eliminating:
            c, w = model.bank().votes(int(j))
            assert not r.survivors[c[w < 0]].any() or r.conflict


def test_elbo_and_sequential_nll_are_finite_and_deterministic(masked):
    corpus, eng, model = masked
    w = model.bind(30)
    e1 = elbo_nll(w, corpus.tok[:30], n_samples=6, seed=1)
    e2 = elbo_nll(model.bind(30), corpus.tok[:30], n_samples=6, seed=1)
    assert e1 == e2 and np.isfinite(e1["nelbo_nats"]) and e1["nelbo_nats"] > 0
    s = sequential_nll(model.bind(30), corpus.tok[:30])
    assert np.isfinite(s["nll_nats"]) and s["nll_nats"] > 0
    assert s["perplexity_per_token"] <= 40.0 * 10  # a finite distribution over 40 classes
