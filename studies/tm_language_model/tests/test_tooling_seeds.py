"""tmlm.common.seeds: determinism, and the seed-before-import rule for GraphTM."""

from __future__ import annotations

import random
import subprocess
import sys
import types

import numpy as np
import pytest

from tmlm.common import seeds
from tmlm.common.seeds import seed_everything


@pytest.fixture(autouse=True)
def _pretend_graphtm_is_not_imported(monkeypatch):
    """Other test modules of the same session may have imported the real GraphTM."""
    for name in seeds.IMPORT_TIME_SEEDED_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


def draw(seed):
    rng = seed_everything(seed)
    return (
        [random.random() for _ in range(5)],
        random.getrandbits(64),
        np.random.rand(5).tolist(),
        np.random.randint(0, (1 << 31) - 1, 8).tolist(),  # the call pycuda's curand seeding makes
        rng.integers(0, 2**63 - 1, 5).tolist(),
        rng.random(5).tolist(),
    )


def test_same_seed_gives_identical_streams():
    first = draw(1234)
    random.random(), np.random.rand(3)  # disturb the global state in between
    assert draw(1234) == first


def test_different_seeds_give_different_streams():
    a, b = draw(1), draw(2)
    assert all(x != y for x, y in zip(a, b))


def test_returned_generator_is_independent_of_the_global_numpy_state():
    rng = seed_everything(5)
    np.random.rand(100)
    expected = np.random.default_rng(5).random(4)
    assert np.array_equal(rng.random(4), expected)
    assert isinstance(rng, np.random.Generator)


def test_known_values_pin_the_algorithms():
    """Python's Mersenne Twister and numpy's legacy seeding are version stable."""
    seed_everything(0)
    assert random.random() == 0.8444218515250481
    assert np.random.rand() == 0.5488135039273248


def test_seed_is_stable_across_processes(repo_root):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "import random, numpy as np\n"
        "from tmlm.common.seeds import seed_everything\n"
        "rng = seed_everything(99)\n"
        "print(random.random(), np.random.rand(), rng.random())\n"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code, str(repo_root / "src")], capture_output=True,
                       text=True, check=True).stdout
        for _ in range(2)
    }
    assert len(outs) == 1
    rng = seed_everything(99)
    assert outs.pop().split() == [repr(random.random()), repr(np.random.rand()), repr(rng.random())]


@pytest.mark.parametrize("bad", [-1, 2**32, 10**20])
def test_out_of_range_seeds_are_rejected(bad):
    with pytest.raises(ValueError):
        seed_everything(bad)


@pytest.mark.parametrize("bad", [1.5, "7", None, True])
def test_non_integer_seeds_are_rejected(bad):
    with pytest.raises(TypeError):
        seed_everything(bad)


def test_boundary_seeds_and_numpy_integers_are_accepted():
    seed_everything(0)
    seed_everything(2**32 - 1)
    assert draw(np.int64(42)) == draw(42)


def test_seeding_after_graphtm_import_is_refused(monkeypatch):
    """GraphTsetlinMachine.tm seeds curand from np.random when it is imported."""
    assert "GraphTsetlinMachine.tm" in seeds.IMPORT_TIME_SEEDED_MODULES
    monkeypatch.setitem(sys.modules, "GraphTsetlinMachine.tm", types.ModuleType("fake_tm"))
    assert seeds.late_seeded_modules() == ["GraphTsetlinMachine.tm"]
    state = random.getstate()
    with pytest.raises(RuntimeError, match="Seed first, import second"):
        seed_everything(3)
    assert random.getstate() == state  # a refused call seeds nothing
    late = seed_everything(3, allow_late=True)
    assert np.array_equal(late.random(3), np.random.default_rng(3).random(3))
