"""T0.7: the THREE defects of PyHierarchicalTsetlinMachineCUDA.tm.MultiOutputTsetlinMachine
(installed 0.3.1, git c85ca4e7; docs/02 V3).

This file documents a known limitation of an upstream package. Every test here PASSES
while the defects exist. A failure means upstream changed, and docs/02 V3 plus the
"Specialist A" plan (docs/12) must be re-read.

  1. `MultiOutputTsetlinMachine.__init__` neither accepts nor forwards
     `hierarchy_structure`. The parent constructor defaults it to None and runs
     `self.depth = len(hierarchy_structure)`, so EVERY construction raises TypeError:
     with the keyword ("unexpected keyword argument") and without it (len of None).
  2. The class never sets `flip_polarity`, which the parent reads when it formats the
     kernel parameters (`cuda_modules`).
  3. The class never sets `tm_type`, which the parent reads in `_fit` and in the state
     export.

Defect 1 fires before any CUDA call. Defects 2 and 3 are shown by source inspection, and
defect 2 also behaviourally with a subclass that repairs defect 1 only.

The module under test imports `pycuda.autoprimaryctx`, which creates a CUDA context at
import time, so the whole module is skipped when no context can be created. Run it
through the GPU lock: scripts/gpu_run.sh shared -- python -m pytest -q <this file>
"""

from __future__ import annotations

import inspect

import pytest

try:
    import pycuda.driver as _cuda

    _cuda.init()
    if _cuda.Device.count() < 1:
        raise RuntimeError("no CUDA device visible")
    import PyHierarchicalTsetlinMachineCUDA.tm as htm  # creates the primary context
except Exception as exc:  # noqa: BLE001 - any failure here means: no usable GPU context
    pytest.skip(f"pycuda cannot create a CUDA context: {type(exc).__name__}: {exc}",
                allow_module_level=True)

pytestmark = pytest.mark.gpu

HIERARCHY = ((htm.AND_GROUP, 4), (htm.OR_ALTERNATIVES, 2), (htm.AND_GROUP, 2))


def test_defect_1_hierarchy_structure_keyword_is_rejected():
    with pytest.raises(TypeError, match="hierarchy_structure"):
        htm.MultiOutputTsetlinMachine(10, 10, 3.0, hierarchy_structure=HIERARCHY)


def test_defect_1_default_construction_calls_len_of_none():
    with pytest.raises(TypeError, match="NoneType"):
        htm.MultiOutputTsetlinMachine(10, 10, 3.0)


def test_defect_1_is_in_the_signatures():
    own = inspect.signature(htm.MultiOutputTsetlinMachine.__init__).parameters
    parent = inspect.signature(htm.CommonTsetlinMachine.__init__).parameters
    assert "hierarchy_structure" not in own
    assert parent["hierarchy_structure"].default is None
    assert "self.depth = len(hierarchy_structure)" in inspect.getsource(
        htm.CommonTsetlinMachine.__init__)
    # The sibling classes do take the keyword, so the omission is specific to this class.
    assert "hierarchy_structure" in inspect.signature(
        htm.MultiClassCoalescedTsetlinMachine.__init__).parameters


def test_defects_2_and_3_attributes_are_never_set():
    source = inspect.getsource(htm.MultiOutputTsetlinMachine)
    assert "flip_polarity" not in source
    assert "tm_type" not in source
    # The parent READS both attributes and never assigns them ...
    parent_source = inspect.getsource(htm.CommonTsetlinMachine)
    assert "self.flip_polarity" in parent_source
    assert "self.tm_type" in parent_source
    assert "self.flip_polarity =" not in parent_source
    assert "self.tm_type =" not in parent_source
    # ... and a working sibling sets both itself.
    sibling = inspect.getsource(htm.MultiClassCoalescedTsetlinMachine)
    assert "self.flip_polarity =" in sibling
    assert "self.tm_type =" in sibling


def test_defect_2_shows_up_once_defect_1_is_repaired():
    class Repaired1(htm.MultiOutputTsetlinMachine):
        """Repairs defect 1 only: forwards hierarchy_structure to the parent."""

        def __init__(self, number_of_clauses, T, s, hierarchy_structure):
            self.negative_clauses = 1
            htm.CommonTsetlinMachine.__init__(
                self, number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, seed=1)

    with pytest.raises(AttributeError, match="flip_polarity"):
        Repaired1(10, 10, 3.0, HIERARCHY)
