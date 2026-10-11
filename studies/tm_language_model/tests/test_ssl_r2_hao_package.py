"""GPU tests of the hierarchical-package read-outs of experiments/ssl_r2_hao.py (docs/13 A7, H17.5):
the one-copy include decoding equals the package's own get_literals / ta_action, `render_hierarchy` reproduces the
package's print_hierarchy output character for character, the OR-groups parsed from that text equal the ones read
from the array, and a model trained twice with the same seed, grid and block has the same state (protocol rule 4).

The package imports pycuda.autoprimaryctx (a CUDA context at import), so the module is skipped when no context can
be created. Run it through the locks, CPU slot first (docs/12 F25):
    scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest -q tests/test_ssl_r2_hao_package.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import pycuda.driver as _cuda

    _cuda.init()
    if _cuda.Device.count() < 1:
        raise RuntimeError("no CUDA device visible")
    import PyHierarchicalTsetlinMachineCUDA.tm as htm  # creates the primary context
except Exception as exc:  # noqa: BLE001 - any failure here means: no usable GPU context
    pytest.skip(f"pycuda cannot create a CUDA context: {type(exc).__name__}: {exc}", allow_module_level=True)

from tmlm.ssl import hao_world as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
E = pytest.importorskip("ssl_r2_hao")

pytestmark = pytest.mark.gpu

F = 60
P = {"clauses": 3, "alternatives": 3, "n_concepts": 4, "T": 20, "s": 4.0, "state_bits": 8, "boost": 0}


def _tiny_data(seed=1, n=200):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, F)) < 0.15).astype(np.uint32)
    y = (X[:, :4].sum(axis=1) > 0).astype(np.uint32)
    return X, y


@pytest.mark.parametrize("neg", [True, False])
def test_decode_render_parse_against_the_package(neg):
    X, y = _tiny_data()
    tm, _, _, _ = E.pkg_fit_predict(htm, "hier", {**P, "append_negated": neg}, 3, X, y, [], 2)
    inc = E.pkg_included(tm)
    assert np.array_equal(inc, tm.get_literals().astype(bool))
    for (i, j, k) in [(0, 0, 0), (1, 5, 7), (2, 11, inc.shape[2] - 1)]:
        assert bool(inc[i, j, k]) == bool(tm.ta_action(i, j, k))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tm.print_hierarchy()
    hier = [tuple(h) for h in E.pkg_hierarchy(htm, "hier", F, P)]
    assert H.render_hierarchy(inc, hier, neg) == buf.getvalue()
    assert H.parse_printed_concepts(buf.getvalue(), 4) == H.concept_groups(inc, 3, 4, neg)


def test_flat_render_matches_the_package():
    X, y = _tiny_data(2)
    tm, _, _, _ = E.pkg_fit_predict(htm, "flat", {**P, "append_negated": True}, 4, X, y, [], 1)
    inc = E.pkg_included(tm)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tm.print_hierarchy()
    assert H.render_hierarchy(inc, [tuple(h) for h in E.pkg_hierarchy(htm, "flat", F, P)], True) == buf.getvalue()
    assert inc.shape[0] == P["clauses"] * P["alternatives"] * P["n_concepts"]


def test_same_seed_same_state():
    X, y = _tiny_data(3)
    a, (pa,), _, _ = E.pkg_fit_predict(htm, "hier", {**P, "append_negated": True}, 9, X, y, [X], 2)
    b, (pb,), _, _ = E.pkg_fit_predict(htm, "hier", {**P, "append_negated": True}, 9, X, y, [X], 2)
    assert E.pkg_state_sha(a) == E.pkg_state_sha(b)
    assert np.array_equal(pa, pb)
