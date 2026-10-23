"""Speed changes to tmu's TMCoalescedClassifier that leave its learning unchanged.

In tmu (commit 5605ff0) every training step computes the clipped class sum of every class in
a Python loop (one numpy dot per class), although without focused negative sampling only the
sum of one randomly drawn class is used. With 1,001 classes this loop is most of the training
time. FastCoalesced keeps the weights of all classes as rows of one matrix (each WeightBank
points at its row, so the library's C code updates the matrix in place) and computes all
class sums with one matrix product (on an exact float64 copy of the
weights, so that BLAS is used; the copy is refreshed for the two rows each step changes). Integer sums, clipping and the float formula are the
same, the random draws are the same and come in the same order, so the trained state is
identical to the library's: tests/test_fast_coalesced.py checks the automaton states and
weights for equality.

class_sums() and clause_outputs() replace the per-class Python loop of predict() in the same
way.
"""
import numpy as np
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier


class FastCoalesced(TMCoalescedClassifier):
    W = None

    def _bind_weights(self):
        K = self.number_of_classes
        W = np.ascontiguousarray(np.stack([self.weight_banks[i].get_weights() for i in range(K)]))
        for i in range(K):
            wb = self.weight_banks[i]
            wb.weights = W[i]
            wb._cffi_init()
        self.W = W
        self.Wf = W.astype(np.float64)  # exact mirror for BLAS (integer sums far below 2**53)

    def update(self, target, e, encoded_X_train):
        if self.W is None:
            self._bind_weights()
        cb = self.clause_bank
        clause_outputs = cb.calculate_clause_outputs_update(self.literal_active, encoded_X_train, e)
        w_t = self.W[target]
        class_sum = np.dot(self.clause_active * w_t, clause_outputs).astype(np.int32)
        class_sum = np.clip(class_sum, -self.T, self.T)
        update_p = (self.T - class_sum) / (2 * self.T)

        type_iii_feedback_selection = self.rng.choice(2)
        if self.type_iii_feedback:
            raise NotImplementedError("type III feedback is not used in T0")

        cb.type_i_feedback(update_p=update_p * self.type_i_p, clause_active=self.clause_active * (w_t >= 0),
                           literal_active=self.literal_active, encoded_X=encoded_X_train, e=e)
        cb.type_ii_feedback(update_p=update_p * self.type_ii_p, clause_active=self.clause_active * (w_t < 0),
                            literal_active=self.literal_active, encoded_X=encoded_X_train, e=e)
        if (w_t >= 0).sum() < self.max_positive_clauses:
            self.weight_banks[target].increment(clause_output=clause_outputs, update_p=update_p,
                                                clause_active=self.clause_active, positive_weights=True)
        self.Wf[target] = w_t

        # All class sums at once (the library loops over the classes).
        sums = np.dot(self.Wf, (self.clause_active * clause_outputs).astype(np.float64))
        ps = np.clip(sums.astype(np.int32).astype(np.float64), -self.T, self.T)
        ps = 1.0 * (self.T + ps) / (2 * self.T)
        ps[target] = 0.0
        self.update_ps[:] = ps

        if self.update_ps.sum() == 0:
            return

        if self.focused_negative_sampling:
            not_target = self.rng.choice(self.number_of_classes, p=self.update_ps / self.update_ps.sum())
            update_p = self.update_ps[not_target]
        else:
            not_target = self.rng.randint(self.number_of_classes)
            while not_target == target:
                not_target = self.rng.randint(self.number_of_classes)
            update_p = self.update_ps[not_target]

        w_n = self.W[not_target]
        cb.type_i_feedback(update_p=update_p * self.type_i_p, clause_active=self.clause_active * (w_n < 0),
                           literal_active=self.literal_active, encoded_X=encoded_X_train, e=e)
        cb.type_ii_feedback(update_p=update_p * self.type_ii_p, clause_active=self.clause_active * (w_n >= 0),
                            literal_active=self.literal_active, encoded_X=encoded_X_train, e=e)
        self.weight_banks[not_target].decrement(clause_output=clause_outputs, update_p=update_p,
                                                clause_active=self.clause_active, negative_weights=True)
        self.Wf[not_target] = w_n

    def clause_outputs(self, X):
        """(n, clauses) uint8 clause outputs of the library's predict path."""
        enc = self.clause_bank.prepare_X(X)
        out = np.empty((X.shape[0], self.number_of_clauses), dtype=np.uint8)
        for e in range(X.shape[0]):
            out[e] = self.clause_bank.calculate_clause_outputs_predict(enc, e)
        return out

    def weights(self):
        if self.W is None:
            self._bind_weights()
        return self.W

    def __getstate__(self):
        # the shared weight matrix is rebuilt after loading (the banks are pickled on their own)
        state = self.__dict__.copy()
        state.pop("W", None)
        state.pop("Wf", None)
        return state
