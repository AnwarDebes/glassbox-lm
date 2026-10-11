"""H-AO: the synthetic AND-OR mechanism world of docs/13 A7, hypothesis H17.5 (phase R2).

The world (all of it fixed by one integer seed):

    vocabulary       V = 2,000 word ids
    OR-groups        4 disjoint groups of 6 words each (24 group words); every other word is a distractor
    concept          a bag is POSITIVE iff it contains at least one word of EVERY group
                     (an AND over groups of an OR over the words of a group)
    combinations     the 6^4 = 1,296 one-word-per-group combinations, id = m0*216 + m1*36 + m2*6 + m3 with
                     m_g in 0..5 the position of the word inside group g
    split            a random 30 percent of the combinations (round(0.3 * 1296) = 389) are TRAIN combinations,
                     the other 907 are TEST combinations; training bags use train combinations only and test
                     bags use test combinations only (below), so every test positive is a NOVEL combination

Bags (one draw per example; examples come in (positive, negative) pairs that share the number of extra group
words k, so the two classes have exactly the same multiset of group-word counts; the pairs are then shuffled):

    k                number of extra group words, P(k = 0, 1, 2) = EXTRA_PROBS = (0.5, 0.3, 0.2)
    positive         a core combination c drawn uniformly from the allowed combinations, its 4 words, plus k
                     extra group words (not in the core) chosen uniformly among the k-subsets for which EVERY
                     one-word-per-group combination contained in the bag is an allowed combination (so a test
                     positive contains test combinations only, a training positive train combinations only);
                     when no k-subset qualifies the core is redrawn
    negative         a core combination c from the allowed combinations and a missing group m uniform in 0..3:
                     the 3 core words of the other groups plus 1 + k further words drawn uniformly from the
                     other three groups (never from group m); it therefore has the SAME number of group words
                     as its paired positive (4 + k) but misses group m entirely
    distractors      24 words drawn uniformly without replacement from the 1,976 non-group words

A linear model on the bag of words cannot separate the classes by counting group words (both classes have 4 + k);
the rule itself separates them perfectly. `check_bags` verifies all of the above on generated data.

Randomness: numpy Generator(PCG64) from SeedSequence([20260924, 175, seed, stream]) with stream 0 for the world
(groups, split), and a caller-given stream for every bag set (the experiment uses 1 train, 2 test, 3 validation
fit, 4 validation). Everything is deterministic given (seed, stream, n).

Read-outs of a trained hierarchical package (pure functions of the decoded include array, no GPU here):
`render_hierarchy` reproduces the text of `PyHierarchicalTsetlinMachineCUDA.tm.CommonTsetlinMachine.
print_hierarchy` (installed commit c85ca4e7) character for character (tests/test_ssl_r2_hao_package.py checks it
against the package on the GPU); `parse_printed_concepts` reads the OR-groups back from that text and
`concept_groups` reads them from the array (the two agree, tests check it); `or_group_recovery` scores
recovered groups against the true ones by the Jaccard index under the best one-to-one matching.

Also here: the noisy-parity data of Granmo's package demo (the package's positive control), as the package's
own `examples/NoisyParityData.py` generates it (12 features, 4 variables of 2 bits, label = parity of the 8
relevant bits, 20 percent label noise on the training labels only), seeded.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

VOCAB = 2000
N_GROUPS = 4
GROUP_SIZE = 6
N_DISTRACTORS = 24
TRAIN_FRACTION = 0.30
EXTRA_PROBS = (0.5, 0.3, 0.2)
SEED_TAG = (20260924, 175)

# the constants of PyHierarchicalTsetlinMachineCUDA.tm (c85ca4e7, tm.py:36-39)
AND_GROUP, AND_ALTERNATIVES, OR_GROUP, OR_ALTERNATIVES = 0, 1, 2, 3


def rng_for(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*SEED_TAG, int(seed), int(stream)]))


# ------------------------------------------------------------------ world
@dataclass(frozen=True)
class HaoWorld:
    seed: int
    vocab: int
    groups: np.ndarray        # int64 [n_groups, group_size], word ids, ascending inside a group
    train_combo: np.ndarray   # bool [group_size ** n_groups]
    word_group: np.ndarray    # int64 [vocab]: group of a word, -1 for a distractor
    word_pos: np.ndarray      # int64 [vocab]: position of the word inside its group, -1 for a distractor

    @property
    def n_groups(self) -> int:
        return int(self.groups.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.groups.shape[1])

    @property
    def n_combos(self) -> int:
        return int(self.group_size ** self.n_groups)

    @property
    def distractors(self) -> np.ndarray:
        return np.flatnonzero(self.word_group < 0).astype(np.int64)

    def combo_id(self, members: Sequence[int]) -> int:
        out = 0
        for m in members:
            out = out * self.group_size + int(m)
        return out

    def combo_members(self, cid: int) -> list[int]:
        out = []
        for _ in range(self.n_groups):
            out.append(int(cid % self.group_size))
            cid //= self.group_size
        return out[::-1]

    def combo_words(self, cid: int) -> list[int]:
        return [int(self.groups[g, m]) for g, m in enumerate(self.combo_members(cid))]


def make_world(seed: int, vocab: int = VOCAB, n_groups: int = N_GROUPS, group_size: int = GROUP_SIZE,
               train_fraction: float = TRAIN_FRACTION) -> HaoWorld:
    rng = rng_for(seed, 0)
    words = rng.choice(vocab, size=n_groups * group_size, replace=False).astype(np.int64)
    groups = np.sort(words.reshape(n_groups, group_size), axis=1)
    n_combos = group_size ** n_groups
    n_train = int(round(train_fraction * n_combos))
    train = np.zeros(n_combos, dtype=bool)
    train[rng.permutation(n_combos)[:n_train]] = True
    wg = np.full(vocab, -1, dtype=np.int64)
    wp = np.full(vocab, -1, dtype=np.int64)
    for g in range(n_groups):
        wg[groups[g]] = g
        wp[groups[g]] = np.arange(group_size)
    return HaoWorld(seed=int(seed), vocab=int(vocab), groups=groups, train_combo=train, word_group=wg, word_pos=wp)


def carve_combos(world: HaoWorld, allowed: np.ndarray, fraction: float, stream: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the allowed combinations into (fit, held) masks, `held` = round(fraction * n_allowed) of them drawn
    uniformly (validation combinations for hyperparameter selection, carved from the training combinations)."""
    idx = np.flatnonzero(allowed)
    rng = rng_for(world.seed, stream)
    n_held = int(round(fraction * idx.shape[0]))
    held_idx = idx[rng.permutation(idx.shape[0])[:n_held]]
    held = np.zeros_like(allowed)
    held[held_idx] = True
    return allowed & ~held, held


def contained_combos(world: HaoWorld, bag_words: Sequence[int]) -> list[int]:
    """Every one-word-per-group combination contained in the bag (empty if a group is missing)."""
    per_group: list[list[int]] = [[] for _ in range(world.n_groups)]
    for w in bag_words:
        g = int(world.word_group[int(w)])
        if g >= 0:
            per_group[g].append(int(world.word_pos[int(w)]))
    if any(len(p) == 0 for p in per_group):
        return []
    return [world.combo_id(m) for m in itertools.product(*[sorted(set(p)) for p in per_group])]


def is_positive(world: HaoWorld, bag_words: Sequence[int]) -> bool:
    seen = {int(world.word_group[int(w)]) for w in bag_words}
    return all(g in seen for g in range(world.n_groups))


@dataclass
class HaoBags:
    """n bags. X: u8 [n, vocab] binary bag of words; y: u8 [n] (1 = positive); words: CSR (indptr int64 [n+1],
    indices int64, ascending per row); core: int64 [n] core combination; missing: int64 [n] missing group of a
    negative (-1 for a positive); n_group_words: int64 [n]; k_extra: int64 [n]."""

    X: np.ndarray
    y: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    core: np.ndarray
    missing: np.ndarray
    n_group_words: np.ndarray
    k_extra: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    def bag(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i]:self.indptr[i + 1]]


def _positive(world: HaoWorld, rng: np.random.Generator, allowed: np.ndarray, allowed_idx: np.ndarray, k: int,
              max_tries: int = 10000) -> tuple[list[int], int]:
    all_group_words = [int(w) for w in world.groups.reshape(-1)]
    for _ in range(max_tries):
        cid = int(allowed_idx[rng.integers(allowed_idx.shape[0])])
        core = world.combo_words(cid)
        if k == 0:
            return core, cid
        cands = sorted(set(all_group_words) - set(core))
        valid = []
        for sub in itertools.combinations(cands, k):
            combos = contained_combos(world, core + list(sub))
            if combos and all(allowed[c] for c in combos):
                valid.append(sub)
        if valid:
            return core + list(valid[int(rng.integers(len(valid)))]), cid
    raise RuntimeError("no valid positive bag found (split too small for the extra-word rule)")


def _negative(world: HaoWorld, rng: np.random.Generator, allowed_idx: np.ndarray, k: int) -> tuple[list[int], int, int]:
    cid = int(allowed_idx[rng.integers(allowed_idx.shape[0])])
    core = world.combo_words(cid)
    m = int(rng.integers(world.n_groups))
    base = [w for g, w in enumerate(core) if g != m]
    cands = sorted(int(w) for g in range(world.n_groups) if g != m for w in world.groups[g] if int(w) not in base)
    extra = rng.choice(np.asarray(cands, np.int64), size=1 + k, replace=False)
    return base + [int(w) for w in extra], cid, m


def sample_bags(world: HaoWorld, n: int, allowed: np.ndarray, stream: int,
                n_distractors: int = N_DISTRACTORS, extra_probs: Sequence[float] = EXTRA_PROBS) -> HaoBags:
    """n bags (n even): n/2 positives and n/2 negatives from the `allowed` combinations, in pairs that share k."""
    if n % 2:
        raise ValueError("n must be even (bags come in positive/negative pairs)")
    allowed = np.asarray(allowed, dtype=bool)
    allowed_idx = np.flatnonzero(allowed)
    if allowed_idx.size == 0:
        raise ValueError("no allowed combination")
    rng = rng_for(world.seed, stream)
    probs = np.asarray(extra_probs, np.float64)
    probs = probs / probs.sum()
    distract = world.distractors
    rows: list[list[int]] = []
    y, core, missing, ngw, kex = [], [], [], [], []
    for _ in range(n // 2):
        k = int(rng.choice(probs.shape[0], p=probs))
        pw, pc = _positive(world, rng, allowed, allowed_idx, k)
        nw, nc, m = _negative(world, rng, allowed_idx, k)
        for words, lab, c, mm in ((pw, 1, pc, -1), (nw, 0, nc, m)):
            d = rng.choice(distract, size=n_distractors, replace=False)
            rows.append(sorted(set(words) | {int(x) for x in d}))
            y.append(lab)
            core.append(c)
            missing.append(mm)
            ngw.append(len(set(words)))
            kex.append(k)
    order = rng.permutation(n)
    rows = [rows[i] for i in order]
    lens = np.asarray([len(r) for r in rows], np.int64)
    indptr = np.zeros(n + 1, np.int64)
    indptr[1:] = np.cumsum(lens)
    indices = np.asarray([w for r in rows for w in r], np.int64)
    X = np.zeros((n, world.vocab), dtype=np.uint8)
    X[np.repeat(np.arange(n), lens), indices] = 1
    pick = lambda a: np.asarray(a, np.int64)[order]  # noqa: E731
    return HaoBags(X=X, y=pick(y).astype(np.uint8), indptr=indptr, indices=indices, core=pick(core),
                   missing=pick(missing), n_group_words=pick(ngw), k_extra=pick(kex))


def check_bags(world: HaoWorld, bags: HaoBags, allowed: np.ndarray, n_distractors: int = N_DISTRACTORS) -> dict:
    """Verify the construction rules; returns a dict of named booleans (all must be True)."""
    allowed = np.asarray(allowed, dtype=bool)
    ok_rule = ok_split = ok_neg_missing = ok_distract = ok_core = True
    for i in range(bags.n):
        w = bags.bag(i)
        pos = is_positive(world, w)
        ok_rule &= pos == bool(bags.y[i])
        n_d = int((world.word_group[w] < 0).sum())
        ok_distract &= n_d == n_distractors
        ok_core &= bool(allowed[int(bags.core[i])])
        if pos:
            ok_split &= all(allowed[c] for c in contained_combos(world, w))
        else:
            present = {int(world.word_group[x]) for x in w if world.word_group[x] >= 0}
            ok_neg_missing &= present == set(range(world.n_groups)) - {int(bags.missing[i])}
    cnt_pos = np.sort(bags.n_group_words[bags.y == 1])
    cnt_neg = np.sort(bags.n_group_words[bags.y == 0])
    x_ok = bool(np.array_equal(np.flatnonzero(bags.X.reshape(-1)), (np.repeat(np.arange(bags.n), np.diff(bags.indptr)) * world.vocab + bags.indices)))
    return {"rule_matches_label": bool(ok_rule), "positives_only_allowed_combos": bool(ok_split),
            "negatives_miss_exactly_one_group": bool(ok_neg_missing), "distractor_count": bool(ok_distract),
            "cores_allowed": bool(ok_core), "balanced": int(bags.y.sum()) * 2 == bags.n,
            "same_group_word_counts_by_class": bool(np.array_equal(cnt_pos, cnt_neg)), "X_matches_words": x_ok}


# ------------------------------------------------------------------ recovery of OR-groups
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def or_group_recovery(recovered: Sequence[set], true_groups: np.ndarray) -> dict:
    """Best one-to-one matching of recovered word sets to the true groups (maximum total Jaccard, scipy's
    linear_sum_assignment; a true group left unmatched scores 0). Returns the mean over the TRUE groups and the
    per-group Jaccard in true-group order."""
    from scipy.optimize import linear_sum_assignment

    truth = [set(int(w) for w in g) for g in np.asarray(true_groups)]
    rec = [set(int(w) for w in r) for r in recovered]
    per = [0.0] * len(truth)
    if rec:
        J = np.array([[jaccard(t, r) for r in rec] for t in truth], dtype=np.float64)
        ri, ci = linear_sum_assignment(-J)
        for a, b in zip(ri, ci):
            per[int(a)] = float(J[a, b])
    return {"mean_jaccard": float(np.mean(per)), "per_group": per}


# ------------------------------------------------------------------ the package's printed hierarchy
def _structure_sizes(hierarchy: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """(number of leaves = hierarchy_size[1], number of features spanned) as tm.py computes them."""
    depth = len(hierarchy)
    size = [0] * (depth + 1)
    size[depth] = 1
    for d in range(depth - 1):
        size[depth - d - 1] = hierarchy[depth - d - 1][1] * size[depth - d]
    n_feat = 1
    for d in range(depth - 1, -1, -1):
        if hierarchy[d][0] in (OR_GROUP, AND_GROUP):
            n_feat *= hierarchy[d][1]
    return int(size[1]), int(n_feat)


def render_hierarchy(inc: np.ndarray, hierarchy: Sequence[tuple[int, int]], append_negated: bool) -> str:
    """The exact stdout of print_hierarchy(print_ta_state=False) for include array inc: bool [clauses, leaves,
    literals_per_leaf] (leaf axis = the package's TA-team axis; equal to the leaf count for structures without
    OR_GROUP, the only ones used here)."""
    inc = np.asarray(inc, dtype=bool)
    depth = len(hierarchy)
    n_leaves, _ = _structure_sizes(hierarchy)
    L = inc.shape[2]
    out: list[str] = []
    for i in range(inc.shape[0]):
        out.append("\nCLAUSE #%d: " % i)
        previous = [-1] * (depth - 1)
        for j in range(n_leaves):
            rem = j
            left: list[str] = []
            right: list[str] = []
            inside: list[str] = []
            feature_base = 0
            size = hierarchy[0][1]
            for d in range(1, depth):
                idx = rem % hierarchy[d][1]
                rem = rem // hierarchy[d][1]
                if hierarchy[d][0] not in (OR_ALTERNATIVES, AND_ALTERNATIVES):
                    feature_base += size * idx
                    size *= hierarchy[d][1]
                if previous[d - 1] == -1:
                    left.append("(")
                elif idx == 0 and previous[d - 1] != idx:
                    right.append(")")
                    left.insert(0, "(")
                elif previous[d - 1] != idx:
                    if hierarchy[d][0] in (AND_GROUP, AND_ALTERNATIVES):
                        inside.append(" ∧ ")
                    elif hierarchy[d][0] in (OR_GROUP, OR_ALTERNATIVES):
                        inside.append(" ∨ ")
                previous[d - 1] = idx
            out.extend(right)
            out.extend(inside)
            out.extend(left)
            lits = []
            for k in np.flatnonzero(inc[i, j]):
                k = int(k)
                if (not append_negated) or k < L // 2:
                    lits.append("x%d" % (feature_base + k))
                else:
                    lits.append("¬x%d" % (feature_base + k - L // 2))
            if len(lits) > 1:
                out.append("(" + " ∧ ".join(lits) + ")")
            elif len(lits) == 1:
                out.append(lits[0])
        out.append(")" * (depth - 1) + "\n")
    return "".join(out)


def concept_groups(inc: np.ndarray, alternatives: int, n_concepts: int, append_negated: bool) -> list[list[set]]:
    """For the structure ((AND_GROUP, F), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, n_concepts)): per clause, per
    concept node, the set of POSITIVE feature literals included in any of its A leaves (leaf j belongs to concept
    (j // A) % n_concepts, as print_hierarchy orders them)."""
    inc = np.asarray(inc, dtype=bool)
    L = inc.shape[2]
    n_pos = L // 2 if append_negated else L
    out = []
    for i in range(inc.shape[0]):
        groups: list[set] = [set() for _ in range(n_concepts)]
        for j in range(alternatives * n_concepts):
            c = (j // alternatives) % n_concepts
            groups[c].update(int(k) for k in np.flatnonzero(inc[i, j, :n_pos]))
        out.append(groups)
    return out


_POS_LIT = re.compile("(?<!¬)x(\\d+)")


def parse_printed_concepts(text: str, n_concepts: int) -> list[list[set]]:
    """Read the OR-groups back from print_hierarchy text of the 3-level structure of `concept_groups`: per clause
    line, the positive literals x<k> inside each of the n_concepts parenthesised OR nodes."""
    out = []
    for line in text.split("\n"):
        if not line.startswith("CLAUSE #"):
            continue
        body = line.split(": ", 1)[1]
        if not (body.startswith("((") and body.endswith("))")):
            raise ValueError(f"unexpected clause text: {line[:80]}")
        parts = body[2:-2].split(") ∧ (")
        if len(parts) != n_concepts:
            raise ValueError(f"expected {n_concepts} concept nodes, found {len(parts)}")
        out.append([{int(m) for m in _POS_LIT.findall(p)} for p in parts])
    return out


# ------------------------------------------------------------------ noisy parity (the package's positive control)
def noisy_parity(n: int, seed: int, stream: int, noise: float, n_features: int = 12, n_variables: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X uint32 [n, 12], y_noisy uint32 [n], y_clean uint32 [n]) as examples/NoisyParityData.py of the
    package builds them: every bit uniform, label = parity of the bits j*F/V and j*F/V + 1 of the V variables,
    labels flipped with probability `noise` (the demo uses 0.2 for training and 0 for testing)."""
    rng = rng_for(seed, stream)
    X = rng.integers(0, 2, size=(n, n_features), dtype=np.uint32)
    rel = [j * n_features // n_variables + b for j in range(n_variables) for b in (0, 1)]
    y = (X[:, rel].sum(axis=1) % 2).astype(np.uint32)
    flip = rng.random(n) <= noise if noise > 0 else np.zeros(n, dtype=bool)
    return X, np.where(flip, 1 - y, y).astype(np.uint32), y
