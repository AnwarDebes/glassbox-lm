"""Thin layer over tmu's TMClassifier (CPU backend, single thread).

The training step is tmu's own: fit_indices() repeats the body of TMClassifier.fit (one draw of
clause and literal masks per call, a shuffled pass, _fit_sample per example) over a given list
of example indices of a data set encoded once. Nothing inside the library is changed. Rule T
and its controls only choose which examples a member sees; the coupled team only replaces the
update probability by one computed from the joint vote.
"""
import pickle

import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

from tally.b6 import config, data

ENCODE_CHUNK = 4000


class Member:
    def __init__(self, spec, n_classes, X_like, seed, clauses=config.CLAUSES, T=None):
        self.spec = dict(spec)
        self.name = spec["name"]
        self.T = config.T_of(spec, clauses) if T is None else T
        self.tm = TMClassifier(
            number_of_clauses=clauses, T=self.T, s=spec["s"], platform="CPU",
            patch_dim=spec["patch"], weighted_clauses=spec["weighted"],
            max_included_literals=spec["max_lit"], clause_drop_p=spec["drop"], seed=seed)
        # tmu sizes its banks from the first data it sees: shape from X_like, classes 0..K-1
        self.tm.init(data.dense(X_like, np.arange(min(2, X_like.shape[0]))),
                     np.arange(n_classes, dtype=np.uint32))
        self.n_classes = n_classes

    def encode(self, X):
        """tmu encoding of all rows of X, built in chunks to bound memory."""
        bank = self.tm.clause_banks[0]
        parts = [bank.prepare_X(data.dense(X, np.arange(i, min(i + ENCODE_CHUNK, X.shape[0]))))
                 for i in range(0, X.shape[0], ENCODE_CHUNK)]
        return np.ascontiguousarray(np.concatenate(parts, axis=0))

    def fit_indices(self, enc, y, idx):
        """One tmu training pass over the examples idx (shuffled by the machine's own rng)."""
        tm = self.tm
        clause_active = tm.mechanism_clause_active()
        literal_active = tm.mechanism_literal_active()
        order = np.array(idx, dtype=np.int64, copy=True)
        tm.rng.shuffle(order)
        for e in order:
            target = int(y[e])
            not_target = tm.weight_banks.sample(exclude=[target])
            tm._fit_sample(target=target, not_target=not_target, sample_idx=int(e),
                           clause_active=clause_active, literal_active=literal_active,
                           encoded_X_train=enc)
        return len(order)

    def class_sums(self, enc, rows=None):
        rows = range(enc.shape[0]) if rows is None else rows
        return np.array([self.tm.predict_compute_class_sums(enc, int(i), False) for i in rows],
                        dtype=np.int32).reshape(-1, self.n_classes)

    def clause_outputs(self, enc):
        """Clause outputs (n, classes * clauses) as uint8, like TMClassifier.transform."""
        tm = self.tm
        n, K, C = enc.shape[0], self.n_classes, tm.number_of_clauses
        out = np.empty((n, K * C), dtype=np.uint8)
        for e in range(n):
            for k in range(K):
                out[e, k * C:(k + 1) * C] = tm.clause_banks[k].calculate_clause_outputs_predict(enc, e)
        return out

    def save(self, p):
        with open(p, "wb") as f:
            pickle.dump(self.tm, f)


def load_tm(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def coupled_epoch(members, encs, y, idx, rng):
    """Fully coupled team: every member votes into one sum from the start.

    For the target class and one sampled other class, each member's clipped class sum is
    divided by its own T (so every member's vote lies in [-1, 1]); the team sum V lies in
    [-M, M], and every member gets tmu's feedback with the update probability of a single
    machine whose margin is M: (M - V) / 2M for the target, (M + V) / 2M for the other class.
    """
    M = len(members)
    masks = [(m.tm.mechanism_clause_active(), m.tm.mechanism_literal_active()) for m in members]
    order = np.array(idx, dtype=np.int64, copy=True)
    rng.shuffle(order)
    K = members[0].n_classes
    for e in order:
        e = int(e)
        target = int(y[e])
        other = int(rng.choice([k for k in range(K) if k != target]))
        for cls, is_target in ((target, True), (other, False)):
            sums, outs = [], []
            for m, enc, (ca, la) in zip(members, encs, masks):
                s, o = m.tm.mechanism_clause_sum(target=cls, clause_active=ca, literal_active=la,
                                                 encoded_X_train=enc, sample_idx=e)
                sums.append(s / m.T)
                outs.append(o)
            V = float(np.sum(sums))
            p = (M - V) / (2 * M) if is_target else (M + V) / (2 * M)
            for m, enc, (ca, la), o in zip(members, encs, masks, outs):
                m.tm.mechanism_feedback(is_target=is_target, target=cls, clause_outputs=o, update_p=p,
                                        clause_active=ca, literal_active=la, encoded_X_train=enc,
                                        sample_idx=e)
    return len(order)
