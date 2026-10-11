"""docs/11 section 3: the reference randomness. Golden values are fixed here and reused by the
C++/CUDA probe (tests/test_tm_core_constants.py) and, later, by the engines' golden-vector test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmlm.engine import rng
from tmlm.engine.config import Stream

GOLDEN = Path(__file__).parent / "golden" / "rng_vectors.json"


def test_mix64_reference_value():
    assert rng.mix64(0) == 0xE220A8397B1DCDAF  # SplitMix64 first output for seed 0


def test_rng32_is_upper_half_and_deterministic():
    assert rng.rng32(1, 2, 3, 4, 5) == rng.rng64(1, 2, 3, 4, 5) >> 32
    assert rng.rng64(1, 2, 3, 4, 5) == rng.rng64(1, 2, 3, 4, 5)
    assert rng.rng64(1, 2, 3, 4, 5) != rng.rng64(2, 2, 3, 4, 5)
    assert rng.rng64(1, 2, 3, 4, 5) != rng.rng64(1, 3, 3, 4, 5)
    assert rng.rng64(1, 2, 3, 4, 5) != rng.rng64(1, 2, 3, 4, 6)


def test_arguments_are_taken_mod_2_64():
    assert rng.rng64(1 << 64, 0, 0, 0, 0) == rng.rng64(0, 0, 0, 0, 0)
    assert rng.rng64(0, 0, (1 << 64) + 7, 0, 0) == rng.rng64(0, 0, 7, 0, 0)


def test_q32_conversions():
    assert rng.q32(0.0) == 0
    assert rng.q32(1.0) == (1 << 32) - 1
    assert rng.q32(0.5) == 1 << 31
    assert rng.inv_q32(16) == (1 << 32) // 16
    assert rng.inv_q32(1) == (1 << 32) - 1
    with pytest.raises(ValueError):
        rng.q32(1.5)


def test_bern_and_pick_ranges():
    for i in range(200):
        assert rng.pick(7, 1, 1, i, 0, 0) in range(7)
    assert rng.bern((1 << 32) - 1, 1, 1, 1, 1, 1)  # p = 1 - 2^-32 passes for almost all draws
    assert not rng.bern(0, 1, 1, 1, 1, 1)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 1000, 4097])
def test_perm_is_a_bijection(n):
    for epoch in (0, 1):
        image = sorted(rng.perm(1, int(Stream.PERM), epoch, x, n) for x in range(n))
        assert image == list(range(n))


def test_perm_changes_with_epoch_and_seed():
    n = 1000
    a = [rng.perm(1, int(Stream.PERM), 0, x, n) for x in range(n)]
    b = [rng.perm(1, int(Stream.PERM), 1, x, n) for x in range(n)]
    c = [rng.perm(2, int(Stream.PERM), 0, x, n) for x in range(n)]
    assert a != b and a != c


def test_golden_vectors_match_file():
    """The file is the contract for the C implementation. Regenerate ONLY when the spec changes."""
    vectors = json.loads(GOLDEN.read_text())
    for v in vectors["rng64"]:
        assert rng.rng64(*v["args"]) == int(v["value"], 16)
    for v in vectors["perm"]:
        assert rng.perm(v["seed"], int(Stream.PERM), v["epoch"], v["x"], v["n"]) == v["value"]
