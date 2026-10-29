"""The count twin (keys J1-J5, Dirichlet back-off) and the tree twin (a counted decision tree, path smoothing).

Count twin: P(true | key) = (positives + w * P(parent)) / (rows + w), counted on rows; an unseen cell takes its
parent's value. Keys: J1 = rank (9) x floor-probability bucket (8) x type (5) x placeholders due (0 / 1 / 2+);
J2 = J1 x already said; J3 = J2 x candidate top-32 identity (33); J4 = J2 x domain-act set; J5 = J2 x identity x
domain-act set. Back-off: J5 -> J3, J4 -> J2, J3 -> J2, J2 -> J1 -> rank x bucket x type -> rank x bucket -> the
overall rate. J1 is a back-off level only. Grid: {J2, J3, J4, J5} x w {3, 30}.

Tree twin: sklearn DecisionTreeClassifier (criterion entropy, min_samples_leaf m, random_state 0) on the arm's
bits; each node's P(true) = (positives + alpha * P(parent)) / (rows + alpha), the root's = its rate; a row's score
is its leaf's value. Grid: m {20, 100, 500, 2000} x alpha {1, 10}.
"""

from __future__ import annotations

import numpy as np

TWIN_KEYS = ("J2", "J3", "J4", "J5")
TWIN_W = (3.0, 30.0)
TREE_LEAF = (20, 100, 500, 2000)
TREE_ALPHA = (1.0, 10.0)
PARENT = {"J5": "J3", "J4": "J2", "J3": "J2", "J2": "J1", "J1": "RBT", "RBT": "RB", "RB": None}
COLS = {"RB": ["r", "pb"], "RBT": ["r", "pb", "t5"], "J1": ["r", "pb", "t5", "d3"], "J2": ["r", "pb", "t5", "d3", "s"],
        "J3": ["r", "pb", "t5", "d3", "s", "i"], "J4": ["r", "pb", "t5", "d3", "s", "a"],
        "J5": ["r", "pb", "t5", "d3", "s", "i", "a"]}
FIELD = {"r": "rank", "pb": "pb", "t5": "type5", "d3": "due3", "s": "said", "i": "ident", "a": "dset"}


def chain_of(level: str) -> list:
    out = []
    while level is not None:
        out.append(level)
        level = PARENT[level]
    return out


RADIX = {"r": 9, "pb": 8, "t5": 5, "d3": 3, "s": 2, "i": 33}


def key_columns(rows: dict, level: str) -> np.ndarray:
    return np.stack([np.asarray(rows[FIELD[f]], np.int64) for f in COLS[level]], 1)


def pack(rows: dict, level: str, dv: np.ndarray) -> np.ndarray:
    """One exact int64 key per row: the level's columns in mixed radix; the domain-act set as its index in dv (the
    sorted sets of the counting rows), len(dv) for a set never counted (its cells are then unseen)."""
    key = np.zeros(len(rows["rank"]), np.int64)
    for f in COLS[level]:
        x = np.asarray(rows[FIELD[f]], np.int64)
        if f == "a":
            i = np.minimum(np.searchsorted(dv, x), max(len(dv) - 1, 0))
            x = np.where((len(dv) > 0) & (dv[i] == x), i, len(dv)) if len(dv) else np.zeros_like(x)
            radix = len(dv) + 1
        else:
            radix = RADIX[f]
            if len(x) and (x.min() < 0 or x.max() >= radix):
                raise ValueError(f"key column {f} out of range")
        key = key * radix + x
    return key


def twin_tables(rows: dict, y: np.ndarray, levels) -> dict:
    """Rows and positives per cell for every level in the back-off chains of `levels` (shared by every w)."""
    y = np.asarray(y, np.float64)
    dv = np.unique(np.asarray(rows["dset"], np.int64))
    tabs = {"p0": float(y.mean()) if len(y) else 0.0, "dv": dv}
    for k in sorted({k for lv in levels for k in chain_of(lv)}):
        u, inv, n = np.unique(pack(rows, k, dv), return_inverse=True, return_counts=True)
        tabs[k] = {"u": u, "n": n.astype(np.float64), "t": np.bincount(inv.reshape(-1), weights=y, minlength=len(u))}
    return tabs


class CountTwin:
    """P(true | key) with Dirichlet back-off of weight w. Row by row, coarsest level first: v = p0, then at each level
    v = (positives + w * v) / (rows + w) of the row's cell (0 and 0 for a cell never counted, which keeps v). Every cell's
    parent is the row's own cell one level coarser, so this equals the cell-wise definition."""

    def __init__(self, level: str, w: float, tables: dict):
        assert level in TWIN_KEYS
        self.level, self.w = level, float(w)
        self.tab = {"p0": tables["p0"], "dv": tables["dv"], **{k: tables[k] for k in chain_of(level)}}
        self.p0 = tables["p0"]

    def predict(self, rows: dict) -> np.ndarray:
        v = np.full(len(rows["rank"]), self.p0)
        for k in chain_of(self.level)[::-1]:
            t = self.tab[k]
            key = pack(rows, k, self.tab["dv"])
            if len(t["u"]):
                j = np.minimum(np.searchsorted(t["u"], key), len(t["u"]) - 1)
                hit = t["u"][j] == key
                n = np.where(hit, t["n"][j], 0.0)
                tp = np.where(hit, t["t"][j], 0.0)
            else:
                n = tp = np.zeros(len(key))
            v = (tp + self.w * v) / (n + self.w)
        return v

    def cells(self) -> dict:
        return {k: len(self.tab[k]["u"]) for k in chain_of(self.level)}


class TreeTwin:
    def __init__(self, min_leaf: int, alpha: float, seed: int = 0):
        self.min_leaf, self.alpha, self.seed = int(min_leaf), float(alpha), seed

    def fit(self, X: np.ndarray, y: np.ndarray, tree=None) -> TreeTwin:
        """Fit the tree (or reuse a fitted one: alpha only changes the read-out), then the smoothed node values."""
        from sklearn.tree import DecisionTreeClassifier
        y = np.asarray(y, np.int64)
        self.tree = tree if tree is not None else DecisionTreeClassifier(
            criterion="entropy", min_samples_leaf=self.min_leaf, random_state=self.seed).fit(X, y)
        tr = self.tree.tree_
        nn = tr.node_count
        leaf = self.tree.apply(X)
        n = np.bincount(leaf, minlength=nn).astype(np.float64)
        t = np.bincount(leaf, weights=y, minlength=nn)
        parent = np.full(nn, -1, np.int64)
        internal = np.flatnonzero(tr.children_left >= 0)
        parent[tr.children_left[internal]] = internal
        parent[tr.children_right[internal]] = internal
        # sklearn numbers a parent before its children, so decreasing ids visit children first
        for node in range(nn - 1, 0, -1):
            n[parent[node]] += n[node]
            t[parent[node]] += t[node]
        val = np.empty(nn)
        val[0] = t[0] / max(n[0], 1.0)
        for node in range(1, nn):
            val[node] = (t[node] + self.alpha * val[parent[node]]) / (n[node] + self.alpha)
        self.n_node, self.t_node, self.val, self.parent = n, t, val, parent
        self.n_leaves = int((tr.children_left < 0).sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.val[self.tree.apply(X)]
