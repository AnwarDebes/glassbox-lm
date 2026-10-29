"""Subclasses of tmu 0.8.3's TMAutoEncoder for the test-5 pilot (the venv is never edited).

TMAEFixed: numpy-2-safe activate_literals. The stock method (autoencoder.py lines 229-246) clears the negated
    half with `literal_active[ta_chunk] &= (~(1 << chunk_pos))` on a Python int; numpy 2 rejects the negative
    int (OverflowError, found by the tiny dry run of 2026-10-17 23:54Z), so feature_negation=False cannot run.
    This version draws the same single self.rng.rand(number_of_literals) vector, applies the same masks, and
    packs the bits in the same layout (literal k -> chunk k // 32, bit k % 32), vectorized.

MaskedClassSumAE(TMAEFixed): the class sum masks the target word, as in Bhattarai et al. (arXiv 2301.00709,
    Eq. 4: x_k is replaced by 1 when x_k is predicted). Stock TMAutoEncoder.update (lines 121-124) evaluates the
    clauses for the class sum and the weight updates with ALL literals active, so a clause that holds the
    target's own literal (learned through another target's feedback) sees the target; the TA feedback already
    treats the masked literal as true (ClauseBank.c: Xi | ~literal_active). This subclass passes the masked
    literal_active to that one clause evaluation (DESIGN_NOTES.md Sec. 3.8 item 4).
"""

from __future__ import annotations

import inspect

import numpy as np
from tmu.models.autoencoder.autoencoder import TMAutoEncoder

_UPDATE_SRC = inspect.getsource(TMAutoEncoder.update)
_ACT_SRC = inspect.getsource(TMAutoEncoder.activate_literals)
if ("all_literal_active" not in _UPDATE_SRC or _UPDATE_SRC.count("calculate_clause_outputs_update") != 1
        or "self.rng.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p" not in _ACT_SRC):
    raise RuntimeError("tmu TMAutoEncoder changed; tmae_fixed.py must be re-checked")


class TMAEFixed(TMAutoEncoder):
    def activate_literals(self):
        cb = self.clause_bank
        L, C = int(cb.number_of_literals), int(cb.number_of_ta_chunks)
        active = self.rng.rand(L) >= self.literal_drop_p  # the stock method's one draw, same stream
        if not self.feature_negation:
            active[L // 2:] = False
        padded = np.zeros(C * 32, dtype=np.uint64)
        padded[:L] = active
        words = (padded.reshape(C, 32) << np.arange(32, dtype=np.uint64)).sum(axis=1)
        return np.ascontiguousarray(words.astype(np.uint32))


class MaskedClassSumAE(TMAEFixed):
    def update(self, target_output, Y, encoded_X, clause_active, literal_active):
        cb = self.clause_bank
        stock = cb.calculate_clause_outputs_update

        def masked(_all_active, X, e):
            return stock(literal_active, X, e)

        cb.calculate_clause_outputs_update = masked  # instance attribute shadows the method for this one call
        try:
            super().update(target_output, Y, encoded_X, clause_active, literal_active)
        finally:
            del cb.calculate_clause_outputs_update
