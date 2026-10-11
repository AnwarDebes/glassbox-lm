"""tmlm.common.interface: the contract is checkable, and it contains no implementation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tmlm.common import interface
from tmlm.common.interface import Explanation, FiredClause, MaskedBatch, MaskedTokenModel


def make_batch(**overrides) -> MaskedBatch:
    fields = dict(
        tokens=np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint16),
        doc_start=np.array([0, 3, 8], dtype=np.int64),
        masked_positions=np.array([1, 4, 6], dtype=np.int64),
        positions=np.array([4, 1, 4], dtype=np.int64),
        causal=False,
    )
    fields.update(overrides)
    return MaskedBatch(**fields)


def test_the_five_methods_are_the_interface():
    methods = {name for name, member in vars(MaskedTokenModel).items()
               if inspect.isfunction(member) and not name.startswith("_")}
    assert methods == {"predict_scores", "logprobs", "explain", "state_hash", "export_bank"}
    for name in methods:
        doc = inspect.getdoc(getattr(MaskedTokenModel, name))
        assert doc and len(doc) > 80, f"{name} needs a precise docstring"
    for name in ("predict_scores", "logprobs"):
        doc = inspect.getdoc(getattr(MaskedTokenModel, name))
        assert "[B, V]" in doc and ("int32" in doc or "float32" in doc)


def test_the_interface_has_no_implementation():
    for name in ("predict_scores", "logprobs", "explain", "state_hash", "export_bank"):
        body = inspect.getsource(getattr(MaskedTokenModel, name)).split('"""')[-1].strip()
        assert body == "...", f"{name} must stay abstract"
    assert not any(name.startswith(("pycuda", "cupy")) for name in vars(interface))


def test_runtime_check_accepts_a_complete_model_and_rejects_a_partial_one():
    class Complete:
        vocab_size = 10

        def predict_scores(self, batch):
            return np.zeros((len(batch.positions), self.vocab_size), dtype=np.int32)

        def logprobs(self, batch):
            return np.full((len(batch.positions), self.vocab_size), -np.log(10), dtype=np.float32)

        def explain(self, batch, index):
            return Explanation(int(batch.positions[index]), 0, 0.0, ())

        def state_hash(self):
            return "0" * 64

        def export_bank(self):
            raise NotImplementedError

    class Partial:
        vocab_size = 10

        def predict_scores(self, batch):
            return None

    assert isinstance(Complete(), MaskedTokenModel)
    assert not isinstance(Partial(), MaskedTokenModel)


def test_valid_batches_pass():
    make_batch().validate()
    make_batch().validate(vocab_size=13)
    make_batch(masked_positions=np.array([], dtype=np.int64), causal=True).validate()
    make_batch(positions=np.array([], dtype=np.int64)).validate()


@pytest.mark.parametrize("overrides", [
    {"tokens": np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)},
    {"tokens": np.zeros((2, 4), dtype=np.uint16)},
    {"doc_start": np.array([0, 3, 8], dtype=np.int32)},
    {"doc_start": np.array([1, 3, 8], dtype=np.int64)},
    {"doc_start": np.array([0, 3, 7], dtype=np.int64)},
    {"doc_start": np.array([0, 3, 3, 8], dtype=np.int64)},
    {"masked_positions": np.array([4, 1, 6], dtype=np.int64)},
    {"masked_positions": np.array([1, 4, 4], dtype=np.int64)},
    {"masked_positions": np.array([1, 4, 8], dtype=np.int64)},
    {"masked_positions": np.array([-1, 4], dtype=np.int64), "positions": np.array([4], dtype=np.int64)},
    {"positions": np.array([4, 2], dtype=np.int64)},  # 2 is not masked: its token would leak
    {"positions": np.array([4, 9], dtype=np.int64)},
    {"positions": [4, 1]},
    {"causal": True},  # causal batches carry no mask set
])
def test_invalid_batches_are_rejected(overrides):
    with pytest.raises(ValueError):
        make_batch(**overrides).validate()


def test_vocab_size_is_checked_when_given():
    with pytest.raises(ValueError):
        make_batch().validate(vocab_size=12)


def test_containers_are_immutable():
    clause = FiredClause(3, (131074,), ("t[-1]=New",), ((42, 57), (43, 3)))
    explanation = Explanation(position=4, predicted=42, score=57.0, clauses=(clause,))
    with pytest.raises(AttributeError):
        explanation.predicted = 1
    with pytest.raises(AttributeError):
        clause.clause_id = 1
    with pytest.raises(AttributeError):
        make_batch().causal = True
