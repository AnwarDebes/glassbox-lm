"""Pure helpers of the H-AO hierarchy diagnostics (docs/13 amendment A8, item 3; runner
experiments/ssl_r2_hao_diagnostics.py). Numpy and scipy only: no GPU, no package import, so everything here is
unit-tested on the CPU (tests/test_ssl_r2_hao_diagnostics.py).

Contents
  TA state bit planes   encode / decode the state layout of PyHierarchicalTsetlinMachineCUDA (both pinned commits):
                        uint32 words [clause][ta team][literal chunk][state bit]; bit (k mod 32) of word
                        (chunk k // 32, plane b) is bit b of the state of literal k; a TA includes its literal iff
                        its state is at least 2^(bits - 1), i.e. iff the top plane bit is set (tm.ta_action).
  oracle hierarchy      the TRUE AND-of-OR concept of an H-AO world written as an include array for the package
                        structure ((AND_GROUP, V), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, K)), with the package's
                        vanilla clause weights (1 - 2 * (clause mod 2), kernels.py prepare_weights), and a numpy
                        reference of the package's inference on it (leaf = AND of included literals, empty leaf
                        true; OR_ALTERNATIVES node = number of true leaves; AND node = product; class sum = sum of
                        weight x clause output, or of weight x [output > 0] with HEAD's binary_inference)
  node recovery         per-node Jaccard of a (trained) include array against the oracle's node identities
  Valiant elimination   Valiant (1984, Theorem A) for monotone k-CNF over bag-of-words positives: a frequency
                        prefilter, all ORs of at most k kept words, delete every OR some positive bag misses,
                        predict positive iff every surviving OR is hit
  set_state audit       parse `CommonTsetlinMachine.set_state` from the package source and evaluate the size of
                        its TA-state allocation (the package's own code is read, never edited)
  MultiConceptDemo data the generator of the package's examples/MultiConceptDemo.py, statement for statement, with
                        the unseeded global numpy generator replaced by one seeded RandomState

Oracle layout (fixed here, before any diagnostic output; docs/13 A8 3 (iii)):
  leaf j of a clause belongs to OR node c = (j // A) mod K as alternative a = j mod A (the package's evaluate_leaves
  index order; hao_world.concept_groups reads it the same way).
  POSITIVE clauses (even index, weight +1): node c is OR-group c of the world; alternative a holds the single
      positive literal of word groups[c][a mod G] (G = group size; A >= G is required, alternatives beyond G repeat
      the group's words, which leaves the node true exactly when a word of group c is present).
  NEGATIVE clauses (odd index, weight -1), design "de_morgan" (needs negated literals and at least K negative
      clauses): clause 2i + 1 says "group g = i mod K is missing": every leaf of every node holds the negated literals
      of all G words of group g.
  NEGATIVE clauses, design "bias" (positive literals only, where "group missing" cannot be written): clause 1 is a
      constant: in every node alternative 0 is EMPTY (true) and every other alternative holds all G words of the
      node's group, a conjunction that no bag of the world satisfies (a bag holds at most 1 + (1 + max k) words of
      one group, 4 < 6 with the H-AO defaults; the runner checks it on its data), so the clause outputs exactly 1 on
      every bag; the other odd clauses are "never" clauses (every leaf holds all G words of its node's group).
  With vanilla weights the class sum is then >= 0 exactly on positive bags (the package predicts positive at class
  sum >= 0), for both designs and with or without binary_inference.
State depths: "deep" puts included TAs at 2^bits - 1 and excluded ones at 0 (maximal margin); "boundary" puts them at
2^(bits-1) and 2^(bits-1) - 1 (the package initialises every TA at 2^(bits-1) - 1, kernels.py prepare_hierarchy).
Padding TAs of the last chunk get the package's initial value 2^(bits-1) - 1 in both.
"""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# ------------------------------------------------------------------ TA state bit planes


def n_chunks(n_literals: int) -> int:
    return (int(n_literals) - 1) // 32 + 1


def encode_ta_state(states: np.ndarray, state_bits: int, pad_value: int | None = None) -> np.ndarray:
    """states: integer array [clauses, teams, literals] with values in [0, 2^bits). Returns the package's flat uint32
    state buffer (clauses * teams * chunks * bits words). Padding TAs get `pad_value` (default 2^(bits-1) - 1)."""
    st = np.asarray(states)
    if st.ndim != 3:
        raise ValueError("states must be [clauses, teams, literals]")
    bits = int(state_bits)
    if st.size and (int(st.min()) < 0 or int(st.max()) >= (1 << bits)):
        raise ValueError(f"states outside [0, 2^{bits})")
    C, T, L = st.shape
    ch = n_chunks(L)
    pad = (1 << (bits - 1)) - 1 if pad_value is None else int(pad_value)
    full = np.full((C, T, ch * 32), pad, dtype=np.uint64)
    full[:, :, :L] = st
    full = full.reshape(C, T, ch, 32)
    shifts = np.arange(32, dtype=np.uint64)
    raw = np.empty((C, T, ch, bits), dtype=np.uint32)
    for b in range(bits):
        plane = (full >> np.uint64(b)) & np.uint64(1)
        raw[..., b] = (plane << shifts).sum(axis=-1).astype(np.uint32)
    return raw.reshape(-1)


def decode_ta_state(raw: np.ndarray, clauses: int, teams: int, literals: int, state_bits: int) -> np.ndarray:
    """Inverse of encode_ta_state (padding dropped): uint32 states [clauses, teams, literals]."""
    bits = int(state_bits)
    ch = n_chunks(literals)
    r = np.asarray(raw, dtype=np.uint32).reshape(int(clauses), int(teams), ch, bits)
    shifts = np.arange(32, dtype=np.uint32)
    out = np.zeros((int(clauses), int(teams), ch, 32), dtype=np.uint32)
    for b in range(bits):
        out |= ((r[..., b][..., None] >> shifts) & np.uint32(1)) << np.uint32(b)
    return out.reshape(int(clauses), int(teams), ch * 32)[:, :, : int(literals)]


def include_from_raw(raw: np.ndarray, clauses: int, teams: int, literals: int, state_bits: int) -> np.ndarray:
    """bool [clauses, teams, literals] from the top plane only (the decoding of ssl_r2_hao.pkg_included)."""
    bits = int(state_bits)
    ch = n_chunks(literals)
    top = np.ascontiguousarray(np.asarray(raw, dtype=np.uint32).reshape(int(clauses), int(teams), ch, bits)[:, :, :, bits - 1])
    b = np.unpackbits(top.view(np.uint8).reshape(int(clauses), int(teams), ch * 4), axis=2, bitorder="little")
    return b[:, :, : int(literals)].astype(bool)


# ------------------------------------------------------------------ oracle hierarchy
@dataclass
class OracleHierarchy:
    """The true concept as a package state. inc: bool [clauses, A*K, L]; weights: int32 [clauses];
    roles[i] = ("positive", None) | ("de_morgan", g) | ("bias", None) | ("never", None)."""

    inc: np.ndarray
    weights: np.ndarray
    roles: list
    alternatives: int
    n_concepts: int
    n_features: int
    append_negated: bool
    design: str
    groups: np.ndarray = field(repr=False)

    @property
    def clauses(self) -> int:
        return int(self.inc.shape[0])

    @property
    def leaves(self) -> int:
        return int(self.inc.shape[1])

    @property
    def literals(self) -> int:
        return int(self.inc.shape[2])


def vanilla_weights(clauses: int) -> np.ndarray:
    """kernels.py prepare_weights for a non-coalesced machine with negative clauses: 1 - 2 * (clause mod 2)."""
    c = np.arange(int(clauses))
    return (1 - 2 * (c % 2)).astype(np.int32)


def oracle_hierarchy(groups: np.ndarray, n_features: int, clauses: int, alternatives: int, append_negated: bool,
                     design: str = "auto") -> OracleHierarchy:
    """The oracle of the module docstring for true OR-groups `groups` (int [K, G] word ids < n_features)."""
    groups = np.asarray(groups, dtype=np.int64)
    K, G = groups.shape
    A = int(alternatives)
    V = int(n_features)
    if A < G:
        raise ValueError(f"an OR node of {A} alternatives cannot hold a group of {G} words one word per leaf")
    if int(clauses) < 2:
        raise ValueError("need at least one positive and one negative clause")
    if design == "auto":
        design = "de_morgan" if append_negated else "bias"
    if design == "de_morgan" and not append_negated:
        raise ValueError("the de_morgan design needs negated literals (append_negated=True)")
    if design not in ("de_morgan", "bias"):
        raise ValueError(f"unknown negative-clause design {design!r}")
    if design == "de_morgan" and int(clauses) // 2 < K:
        raise ValueError(f"the de_morgan design needs one negative clause per group: {int(clauses) // 2} < {K}")
    L = 2 * V if append_negated else V
    leaves = A * K
    inc = np.zeros((int(clauses), leaves, L), dtype=bool)
    roles: list = []
    node_of = [(j // A) % K for j in range(leaves)]
    alt_of = [j % A for j in range(leaves)]
    for i in range(int(clauses)):
        if i % 2 == 0:
            roles.append(("positive", None))
            for j in range(leaves):
                inc[i, j, int(groups[node_of[j], alt_of[j] % G])] = True
        elif design == "de_morgan":
            g = (i // 2) % K
            roles.append(("de_morgan", int(g)))
            inc[i, :, V + groups[g]] = True
        else:
            roles.append(("bias", None) if i == 1 else ("never", None))
            for j in range(leaves):
                if i == 1 and alt_of[j] == 0:
                    continue  # empty leaf: true on every bag
                inc[i, j, groups[node_of[j]]] = True
    return OracleHierarchy(inc=inc, weights=vanilla_weights(int(clauses)), roles=roles, alternatives=A,
                           n_concepts=int(K), n_features=V, append_negated=bool(append_negated), design=design,
                           groups=groups)


def oracle_states(inc: np.ndarray, state_bits: int, depth: str) -> np.ndarray:
    """uint32 states [clauses, teams, literals] for an include array (see 'State depths' above)."""
    bits = int(state_bits)
    if depth == "deep":
        hi, lo = (1 << bits) - 1, 0
    elif depth == "boundary":
        hi, lo = 1 << (bits - 1), (1 << (bits - 1)) - 1
    else:
        raise ValueError(f"unknown depth {depth!r}")
    return np.where(np.asarray(inc, dtype=bool), hi, lo).astype(np.uint32)


def bags_csr(X: np.ndarray):
    """scipy CSR float32 of a binary bag matrix."""
    import scipy.sparse as sp

    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def leaf_outputs(inc: np.ndarray, X, n_features: int, append_negated: bool, chunk: int = 4096) -> np.ndarray:
    """bool [n, clauses * leaves]: leaf (clause i, leaf j) is true on bag x iff every included literal is true
    (positive literal k: x_k = 1; negated literal: x_k = 0); an empty leaf is true. Exact integer arithmetic on
    sparse products (literal counts <= 2 * n_features)."""
    import scipy.sparse as sp

    inc = np.asarray(inc, dtype=bool)
    C, J, L = inc.shape
    V = int(n_features)
    flat = inc.reshape(C * J, L)
    Ap = sp.csr_matrix(flat[:, :V].astype(np.float32))
    cnt = flat[:, :V].sum(axis=1).astype(np.int64)
    An = None
    if append_negated:
        An = sp.csr_matrix(flat[:, V:].astype(np.float32))
        cnt_n = flat[:, V:].sum(axis=1).astype(np.int64)
    Xs = X if sp.issparse(X) else bags_csr(X)
    Xs = sp.csr_matrix(Xs)
    out = np.empty((Xs.shape[0], C * J), dtype=bool)
    for a in range(0, Xs.shape[0], chunk):
        xb = Xs[a:a + chunk]
        hit = np.rint((xb @ Ap.T).toarray()).astype(np.int64)
        if append_negated:
            # negated literal of word k is true iff x_k = 0: true count = |neg| - (included negated words present)
            hit = hit + cnt_n[None, :] - np.rint((xb @ An.T).toarray()).astype(np.int64)
            out[a:a + chunk] = hit == (cnt + cnt_n)[None, :]
        else:
            out[a:a + chunk] = hit == cnt[None, :]
    return out


def reference_scores(inc: np.ndarray, weights: np.ndarray, X, alternatives: int, n_concepts: int, n_features: int,
                     append_negated: bool, binary_inference: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(class_sum int64 [n], clause_output int64 [n, clauses]) of the package's evaluation for the structure
    ((AND_GROUP, V), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, K)) (unclipped; the package clips score() at +-T
    unless no_clipping). HEAD's and_group_normalization without binary_inference is not modelled."""
    inc = np.asarray(inc, dtype=bool)
    C, J, _ = inc.shape
    A, K = int(alternatives), int(n_concepts)
    if J != A * K:
        raise ValueError(f"{J} leaves is not A x K = {A * K}")
    leaf = leaf_outputs(inc, X, n_features, append_negated)
    node = leaf.reshape(-1, C, K, A).sum(axis=3, dtype=np.int64)
    out = np.prod(node, axis=2, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64).reshape(-1)
    vote = (out > 0).astype(np.int64) if binary_inference else out
    return vote @ w, out


# ------------------------------------------------------------------ node recovery against the oracle
def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def node_recovery(inc: np.ndarray, oracle: OracleHierarchy) -> dict:
    """Per-node Jaccard with FIXED node identities (the oracle wrote group c into node c of every positive clause
    and group g into every node of the de_morgan clause for g). Positive clauses: positive-literal words of node c
    vs groups[c]. de_morgan clauses: negated-literal words of node c vs groups[g]. bias/never clauses: whether the
    clause still equals the oracle exactly. Also literal additions/removals against the oracle, per role."""
    inc = np.asarray(inc, dtype=bool)
    if inc.shape != oracle.inc.shape:
        raise ValueError(f"shape {inc.shape} != oracle {oracle.inc.shape}")
    A, K, V = oracle.alternatives, oracle.n_concepts, oracle.n_features
    groups = [set(int(w) for w in g) for g in oracle.groups]
    pos_j, dm_j, intact = [], [], {"bias": [], "never": []}
    added = {}
    removed = {}
    exact = []
    for i, (role, g) in enumerate(oracle.roles):
        now, ref = inc[i], oracle.inc[i]
        added[role] = added.get(role, 0) + int((now & ~ref).sum())
        removed[role] = removed.get(role, 0) + int((ref & ~now).sum())
        exact.append(bool(np.array_equal(now, ref)))
        if role in ("bias", "never"):
            intact[role].append(exact[-1])
            continue
        row = []
        for c in range(K):
            leaves = [j for j in range(A * K) if (j // A) % K == c]
            if role == "positive":
                words = set(np.flatnonzero(now[leaves, :V].any(axis=0)).tolist())
                row.append(jaccard(words, groups[c]))
            else:
                words = set(np.flatnonzero(now[leaves, V:].any(axis=0)).tolist()) if oracle.append_negated else set()
                row.append(jaccard(words, groups[int(g)]))
        (pos_j if role == "positive" else dm_j).append(row)
    out: dict = {"clauses_exact_share": float(np.mean(exact)), "literals_added": added, "literals_removed": removed}
    for tag, arr in (("pos", pos_j), ("dm", dm_j)):
        if arr:
            a = np.asarray(arr, dtype=np.float64)
            out[f"{tag}_node_jaccard_mean"] = float(a.mean())
            out[f"{tag}_node_jaccard_min"] = float(a.min())
            out[f"{tag}_node_share_exact"] = float(np.mean(a == 1.0))
            out[f"{tag}_node_jaccard_per_node"] = a.mean(axis=0).tolist()
    for role, v in intact.items():
        if v:
            out[f"{role}_clauses_intact_share"] = float(np.mean(v))
    return out


# ------------------------------------------------------------------ Valiant elimination (monotone k-CNF)
def _bag_masks(indptr: np.ndarray, indices: np.ndarray, rows: np.ndarray, bit_of: Mapping[int, int]) -> np.ndarray:
    out = np.zeros(len(rows), dtype=np.uint64)
    for r, i in enumerate(rows):
        m = 0
        for w in indices[indptr[i]:indptr[i + 1]]:
            b = bit_of.get(int(w))
            if b is not None:
                m |= 1 << b
        out[r] = m
    return out


def n_candidates(n_kept: int, max_or: int) -> int:
    return int(sum(math.comb(int(n_kept), r) for r in range(1, int(max_or) + 1)))


def valiant_fit(indptr: np.ndarray, indices: np.ndarray, y: np.ndarray, vocab: int, min_freq: float = 0.1,
                max_or: int = 6, max_candidates: int = 20_000_000, n_pos: int | None = None,
                chunk_cells: int = 20_000_000) -> dict:
    """Valiant's elimination for a monotone k-CNF (k = max_or) on binary bags (CSR indptr/indices, labels y).
    Uses the POSITIVE bags only (the first n_pos of them in row order, all when None): (1) word frequency over
    those bags, keep words with frequency >= min_freq (at most 63, bit masks are uint64); (2) candidates = every OR
    of 1 to max_or kept words; (3) delete every candidate that some positive bag misses (no word in common). The
    survivors are the learned CNF; `minimal` are the survivors with no surviving proper subset (the survivors are
    closed under supersets, so checking the subsets one word smaller suffices; predicting with the minimal ones is
    the same as predicting with all). Raises when the candidate count exceeds max_candidates."""
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    pos_rows = np.flatnonzero(np.asarray(y).astype(bool))
    if n_pos is not None:
        pos_rows = pos_rows[: int(n_pos)]
    npos = int(pos_rows.size)
    if npos == 0:
        raise ValueError("no positive bag")
    counts = np.zeros(int(vocab), dtype=np.int64)
    for i in pos_rows:
        counts[np.unique(indices[indptr[i]:indptr[i + 1]])] += 1
    freq = counts / npos
    kept = np.flatnonzero(freq >= float(min_freq)).astype(np.int64)
    if kept.size > 63:
        raise ValueError(f"{kept.size} words pass the prefilter; the uint64 masks hold at most 63")
    n_cand = n_candidates(kept.size, max_or)
    if n_cand > int(max_candidates):
        raise ValueError(f"{n_cand} candidate ORs exceed max_candidates = {max_candidates}")
    bit_of = {int(w): b for b, w in enumerate(kept)}
    masks = np.unique(_bag_masks(indptr, indices, pos_rows, bit_of))
    survivors: list[int] = []
    per_size = {}
    for r in range(1, int(max_or) + 1):
        if r > kept.size:
            break
        cand = np.fromiter((sum(1 << b for b in comb) for comb in itertools.combinations(range(kept.size), r)),
                           dtype=np.uint64, count=math.comb(int(kept.size), r))
        step = max(1, int(chunk_cells) // max(1, masks.size))
        alive_r = 0
        for a in range(0, cand.size, step):
            c = cand[a:a + step]
            ok = np.all((c[:, None] & masks[None, :]) != 0, axis=1)
            survivors.extend(int(v) for v in c[ok])
            alive_r += int(ok.sum())
        per_size[r] = {"candidates": int(cand.size), "survivors": alive_r}
    alive = set(survivors)
    minimal = []
    for m in survivors:
        bits = [b for b in range(kept.size) if (m >> b) & 1]
        if all((m & ~(1 << b)) not in alive for b in bits):
            minimal.append(m)
    to_words = lambda m: tuple(int(kept[b]) for b in range(kept.size) if (m >> b) & 1)  # noqa: E731
    return {"n_pos": npos, "min_freq": float(min_freq), "max_or": int(max_or), "kept": kept.tolist(),
            "kept_freq": freq[kept].tolist(), "n_candidates": n_cand, "n_distinct_positive_masks": int(masks.size),
            "per_size": per_size, "n_survivors": len(survivors), "survivor_masks": survivors,
            "minimal_masks": minimal, "minimal": [to_words(m) for m in minimal]}


def valiant_predict(model: Mapping, indptr: np.ndarray, indices: np.ndarray, use: str = "minimal") -> np.ndarray:
    """bool [n]: positive iff every OR of the learned CNF shares a word with the bag (vacuously positive when
    nothing survived). use = "minimal" or "all" (the two agree; the test checks it)."""
    kept = [int(w) for w in model["kept"]]
    bit_of = {w: b for b, w in enumerate(kept)}
    indptr = np.asarray(indptr, dtype=np.int64)
    n = indptr.size - 1
    bm = _bag_masks(indptr, np.asarray(indices, dtype=np.int64), np.arange(n), bit_of)
    ors = np.asarray(model["minimal_masks"] if use == "minimal" else model["survivor_masks"], dtype=np.uint64)
    if ors.size == 0:
        return np.ones(n, dtype=bool)
    out = np.ones(n, dtype=bool)
    for a in range(0, ors.size, 4096):
        out &= np.all((bm[:, None] & ors[None, a:a + 4096]) != 0, axis=1)
    return out


# ------------------------------------------------------------------ set_state audit (package source, read only)
def _set_state_alloc_expr(source: str) -> ast.expr:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CommonTsetlinMachine":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "set_state":
                    for st in ast.walk(fn):
                        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                                and isinstance(st.targets[0], ast.Attribute) and st.targets[0].attr == "ta_state_hierarchy_gpu"
                                and isinstance(st.value, ast.Call) and getattr(st.value.func, "attr", None) == "mem_alloc"):
                            return st.value.args[0]
    raise ValueError("CommonTsetlinMachine.set_state with a ta_state_hierarchy_gpu mem_alloc not found")


def _eval_product(expr: ast.expr, attrs: Mapping[str, int]) -> int:
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
        return _eval_product(expr.left, attrs) * _eval_product(expr.right, attrs)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
        return int(attrs[expr.attr])
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
        return int(expr.value)
    raise ValueError(f"unsupported allocation expression node {ast.dump(expr)}")


def set_state_allocation(source: str, attrs: Mapping[str, int]) -> dict:
    """Bytes allocated for the TA state by CommonTsetlinMachine.set_state (parsed from the package source) against
    the bytes the state needs (clauses x ta teams x literal chunks per leaf x state bits x 4, the size used by
    allocate_gpu_memory), for the machine attributes `attrs`. Also whether set_state sets `first` (if it does not,
    the next fit() calls allocate_gpu_memory and initialize_weights_and_ta_states and discards the state)."""
    expr = _set_state_alloc_expr(source)
    alloc = _eval_product(expr, attrs)
    need = int(attrs["number_of_clauses"]) * int(attrs["number_of_ta_teams"]) * int(attrs["number_of_literal_chunks_per_leaf"]) \
        * int(attrs["number_of_state_bits"]) * 4
    tree = ast.parse(source)
    sets_first = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_state":
            sets_first = any(isinstance(t, ast.Attribute) and t.attr == "first"
                             for st in ast.walk(node) if isinstance(st, ast.Assign) for t in st.targets)
    return {"alloc_expr": ast.unparse(expr), "alloc_bytes": int(alloc), "needed_bytes": int(need),
            "needed_over_alloc": need / alloc if alloc else float("inf"),
            "chunks_per_leaf": int(attrs["number_of_literal_chunks_per_leaf"]),
            "under_allocates": bool(alloc < need), "sets_first": bool(sets_first)}


# ------------------------------------------------------------------ Granmo's MultiConceptDemo data
def multiconcept_data(seed: int, n_elements: int, n_examples: int, noise: float):
    """examples/MultiConceptDemo.py's generator (identical at c85ca4e7 and 5b4a223), statement for statement, with
    np.random replaced by RandomState(seed). Returns (Xtr, ytr_noisy, ytr_clean, Xte, yte), uint32; an example has
    one active element in each half of the 2 x n_elements features, label = parity(e0) XOR parity(e1)."""
    rs = np.random.RandomState(int(seed))
    E = int(n_elements)

    def block(n):
        X = np.zeros((n, 2 * E), dtype=np.uint32)
        Y = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            x = rs.randint(E, size=(2))
            X[i, x[0]] = 1
            X[i, E + x[1]] = 1
            Y[i] = np.logical_xor(x[0] % 2, x[1] % 2)
        return X, Y

    Xtr, Ytr = block(int(n_examples))
    Ytr_noisy = np.where(rs.rand(int(n_examples)) <= float(noise), 1 - Ytr, Ytr).astype(np.uint32)
    Xte, Yte = block(int(n_examples))
    return Xtr, Ytr_noisy, Ytr, Xte, Yte


def multiconcept_rule(X: np.ndarray, n_elements: int) -> np.ndarray:
    E = int(n_elements)
    e0 = np.argmax(X[:, :E], axis=1)
    e1 = np.argmax(X[:, E:], axis=1)
    return ((e0 % 2) ^ (e1 % 2)).astype(np.uint32)


def parse_argparse_defaults(source: str) -> dict:
    """{"number_of_clauses": 2, ...}: the `parser.add_argument("--x", default=..., type=int|float)` defaults of a
    demo script (store_true flags are skipped: their default is False)."""
    out = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if "default" not in kw:
            continue
        name = str(node.args[0].value).lstrip("-").replace("-", "_")
        val = ast.literal_eval(kw["default"])
        typ = getattr(kw.get("type"), "id", None)
        out[name] = int(val) if typ == "int" else float(val) if typ == "float" else val
    return out


def summarize_runs(finals: Sequence[float]) -> dict:
    a = np.asarray(list(finals), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}
