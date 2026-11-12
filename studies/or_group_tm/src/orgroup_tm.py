"""orgroup_tm: a Tsetlin machine whose clauses are ANDs of OR-groups of words (clauses in conjunctive normal form).

Model
  C clauses. A clause has G = sides * K groups: K groups for each text side (sides = 2 for a pair of texts,
  1 for one text). state[c, g, w] is the Tsetlin automaton of word w in group g of clause c (int16).
  Word w is a MEMBER of the group iff state[c, g, w] >= 0.
  A group is true on a text iff at least one member is present in the text of its side.
  A group without members is false (the OR over nothing). A clause fires iff all its groups are true.
  Vote sum v = sum of pol[c] over the firing clauses (pol = +1 or -1, fixed).

Feedback for one example (texts, y), clause by clause, each clause picked with the usual vote gate:
  p = (T - clip(v)) / 2T  when y = 1,   p = (T + clip(v)) / 2T  when y = 0.
  The clause SHOULD FIRE when its polarity agrees with y.
    should fire, every false group is empty (this includes the blank clause)
        SEED       the present words whose state is still at least the initial state are dealt at random to the
                   empty groups of their side and become members (state := seed_state); with max_seed > 0 at
                   most that many clauses are seeded by one example; with seed_all every empty group of the
                   side receives every such word
    should fire, fires
        REINFORCE  present members: state + 1 (reinforce = all); or only a member that is the one present
                   member of its group (critical); or one present member per group, chosen at random (share)
        DECAY      (p_decay > 0, same event) a present member weaker than the strongest present member of
                   its group: state - 1 with probability p_decay. Which member REINFORCE picks is decided on
                   the states before the decay, so a member is never reinforced because another one just left
    should fire, does not fire, some false group has members
        RECRUIT    present non-members of the designated false groups: state + 1 (a word turns member at 0)
    should not fire, fires
        REJECT     present members of one group: state - r. The group is chosen at random (reject = random),
                   or is the weakest firing group, the one whose strongest present member has the lowest state
                   (weakest; ties at random); or every group is hit (all)
    should not fire, does not fire
        REPRESS    present non-members of the designated false groups: state - 1
  Designated false groups (mode):
    near  the one false group, when exactly one group is false
    side  the false groups of one side, when every group of the other side is true (pair tasks; one text: as near)
    all   every false group
    atleast  every false group, when at least min_true groups of the clause are true (texts of several sides)
  When several groups of one side are designated, each present word goes to one of them, chosen at random.
  tied = True (pair tasks): group k of the first text and group k of the second text are ONE group; every change
  is made in both copies, and a blank clause is seeded with the words of both texts.
  wmax > 0: every clause also keeps a weight in 0..wmax that follows the share of its firings that were right;
  the weights give two further readings of the firing clauses (scores()), and do not change the learning.

No word becomes a member unless it was present in an example on which the clause should fire.
"""
import numpy as np
from numba import njit

MODES = {"near": 0, "side": 1, "all": 2, "atleast": 3}
REJECT = {"random": 0, "all": 1, "weakest": 2}
REINFORCE = {"all": 0, "critical": 1, "share": 2}
EVENTS = ("seed", "reinforce", "recruit", "reject", "repress", "gated")


def pack_pairs(Xp, P):
    """Flat word ids and bounds for pairs of rows of a 0/1 matrix. Xp [rows, F], P [n, sides] row indices.
    Returns ids int32 and bounds int64 [n, sides + 1]: side s of example e is ids[bounds[e, s]:bounds[e, s + 1]]."""
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[:, None]
    rows = [np.flatnonzero(Xp[i]).astype(np.int32) for i in range(Xp.shape[0])]
    n, sides = P.shape
    bounds = np.zeros((n, sides + 1), np.int64)
    chunks, pos = [], 0
    for e in range(n):
        bounds[e, 0] = pos
        for s in range(sides):
            r = rows[P[e, s]]
            chunks.append(r)
            pos += len(r)
            bounds[e, s + 1] = pos
    ids = np.concatenate(chunks) if chunks else np.zeros(0, np.int32)
    return ids.astype(np.int32), bounds


def pack_csr(sides_ids_offs):
    """Flat ids and bounds from per-side (ids, offsets) arrays of the pair files (a_ids, a_off), (b_ids, b_off)."""
    n = len(sides_ids_offs[0][1]) - 1
    sides = len(sides_ids_offs)
    bounds = np.zeros((n, sides + 1), np.int64)
    lens = [np.diff(off) for _, off in sides_ids_offs]
    total = np.zeros(n, np.int64)
    for ln in lens:
        total += ln
    start = np.concatenate([[0], np.cumsum(total)[:-1]])
    bounds[:, 0] = start
    for s in range(sides):
        bounds[:, s + 1] = bounds[:, s] + lens[s]
    ids = np.zeros(int(total.sum()), np.int32)
    for s, (sid, off) in enumerate(sides_ids_offs):
        for e in range(n):
            ids[bounds[e, s]:bounds[e, s + 1]] = sid[off[e]:off[e + 1]]
    return ids, bounds


@njit(cache=True)
def _evaluate(state, nmem, pol, ids, bounds, e, K, gt, nfalse):
    """Fill gt[c, g] and nfalse[c] for example e; return the vote sum."""
    C, G = nmem.shape
    v = 0
    for c in range(C):
        nf = 0
        for g in range(G):
            s = g // K
            t = False
            if nmem[c, g] > 0:
                for j in range(bounds[e, s], bounds[e, s + 1]):
                    if state[c, g, ids[j]] >= 0:
                        t = True
                        break
            gt[c, g] = t
            if not t:
                nf += 1
        nfalse[c] = nf
        if nf == 0:
            v += pol[c]
    return v


@njit(cache=True)
def _designate(gt_c, nf, sides, K, mode, min_true, desig):
    """Mark the designated false groups of one clause in desig; return how many."""
    G = gt_c.shape[0]
    n = 0
    for g in range(G):
        desig[g] = False
    if mode == 3 and G - nf < min_true:  # atleast: too few true groups, nothing is designated
        return 0
    if mode == 2 or mode == 3:  # all (and atleast, once enough groups are true)
        for g in range(G):
            if not gt_c[g]:
                desig[g] = True
                n += 1
        return n
    if mode == 0 or sides != 2:  # near (the side mode is defined for two sides only)
        if nf == 1:
            for g in range(G):
                if not gt_c[g]:
                    desig[g] = True
                    n += 1
        return n
    # side: the false groups of side s when every group of the other sides is true
    for s in range(sides):
        other_ok = True
        own_false = 0
        for g in range(G):
            if g // K == s:
                if not gt_c[g]:
                    own_false += 1
            elif not gt_c[g]:
                other_ok = False
        if other_ok and own_false > 0:
            for g in range(s * K, (s + 1) * K):
                if not gt_c[g]:
                    desig[g] = True
                    n += 1
            return n
    return n


@njit(cache=True)
def _bump(state, nmem, c, g, w, delta, lo, hi, twin):
    """Add delta to the automaton of word w in group g of clause c, clipped to [lo, hi], and keep the member count
    in step. With twin >= 0 the same change is made in the twin group (tied sides)."""
    st = state[c, g, w]
    ns = min(max(st + delta, lo), hi)
    if ns != st:
        if st < 0 and ns >= 0:
            nmem[c, g] += 1
        elif st >= 0 and ns < 0:
            nmem[c, g] -= 1
        state[c, g, w] = ns
    if twin >= 0:
        st = state[c, twin, w]
        ns = min(max(st + delta, lo), hi)
        if ns != st:
            if st < 0 and ns >= 0:
                nmem[c, twin] += 1
            elif st >= 0 and ns < 0:
                nmem[c, twin] -= 1
            state[c, twin, w] = ns


@njit(cache=True)
def _raise_to(state, nmem, c, g, w, value, floor, twin):
    """SEED: make word w a member of group g with at least `value`, unless its state is below `floor`."""
    st = state[c, g, w]
    if st >= floor and st < value:
        if st < 0:
            nmem[c, g] += 1
        state[c, g, w] = value
    if twin >= 0:
        st = state[c, twin, w]
        if st >= floor and st < value:
            if st < 0:
                nmem[c, twin] += 1
            state[c, twin, w] = value


@njit(cache=True)
def train_pass(state, nmem, pol, wts, ids, bounds, y, order, sides, K, T, mode, init_state, seed_state, r,
               p_rec, p_rep, p_reinf, q_seed, reject_mode, reinforce_mode, p_decay, seed_all, tied, wmax, smax, smin,
               max_seed, min_true, force_gate, rng_seed, counts):
    """One pass over the examples in `order`, one example at a time. counts[6] is incremented per event kind
    (see EVENTS). max_seed > 0 allows at most that many SEED events per example (0 = no limit).
    reject_mode: 0 one random group, 1 all groups, 2 the weakest firing group. reinforce_mode: 0 every present
    member, 1 the present member of a group only when it is the group's one present member, 2 one present member
    of each group chosen at random. p_decay > 0: when a clause fires as it should, a present member whose
    state is below that of the strongest present member of its group loses 1 with this probability.
    tied: group k of side 0 and group k of side 1 are one group (every change is made in both copies).
    wmax > 0: wts[c] in 0..wmax follows the share of the clause's firings that were right (it rises with
    probability (wmax - w) / wmax on a right firing and falls with probability w / wmax on a wrong one); the
    weights do not enter the vote gate.
    force_gate = True picks every clause (for tests)."""
    np.random.seed(rng_seed)
    C, G = nmem.shape
    gt = np.zeros((C, G), np.bool_)
    nfalse = np.zeros(C, np.int64)
    desig = np.zeros(G, np.bool_)
    pick = np.zeros(K, np.int64)
    for ii in range(order.shape[0]):
        e = order[ii]
        v = _evaluate(state, nmem, pol, ids, bounds, e, K, gt, nfalse)
        vc = min(max(v, -T), T)
        if y[e] == 1:
            p_up = (T - vc) / (2.0 * T)
        else:
            p_up = (T + vc) / (2.0 * T)
        if wmax > 0:
            for c in range(C):
                if nfalse[c] == 0:
                    wc = wts[c]
                    if (pol[c] > 0) == (y[e] == 1):
                        if wc < wmax and np.random.random() * wmax < wmax - wc:
                            wts[c] = wc + 1
                    elif wc > 0 and np.random.random() * wmax < wc:
                        wts[c] = wc - 1
        seeds_left = max_seed if max_seed > 0 else C + 1
        for c in range(C):
            if not force_gate and np.random.random() >= p_up:
                continue
            counts[5] += 1
            want = (pol[c] > 0) == (y[e] == 1)
            nf = nfalse[c]
            if want:
                if nf == 0:
                    for g in range(G):
                        s = g // K
                        twin = (g + K) % G if tied else -1
                        lo = bounds[e, s]
                        hi = bounds[e, s + 1]
                        # the present members and their strongest state, before any change by this event
                        npres = 0
                        last = -1
                        mx = -1
                        for j in range(lo, hi):
                            st = state[c, g, ids[j]]
                            if st >= 0:
                                npres += 1
                                last = ids[j]
                                mx = max(mx, st)
                        target = -1
                        if reinforce_mode == 1:
                            if npres == 1:
                                target = last
                        elif reinforce_mode == 2 and npres >= 1:
                            target = last
                            if npres > 1:
                                k = np.random.randint(npres)
                                idx = 0
                                for j in range(lo, hi):
                                    if state[c, g, ids[j]] >= 0:
                                        if idx == k:
                                            target = ids[j]
                                            break
                                        idx += 1
                        if p_decay > 0.0 and npres > 1:
                            for j in range(lo, hi):
                                w = ids[j]
                                st = state[c, g, w]
                                if st >= 0 and st < mx and np.random.random() < p_decay:
                                    _bump(state, nmem, c, g, w, -1, smin, smax, twin)
                        if reinforce_mode == 0:
                            for j in range(lo, hi):
                                w = ids[j]
                                if state[c, g, w] >= 0 and (p_reinf >= 1.0 or np.random.random() < p_reinf):
                                    _bump(state, nmem, c, g, w, 1, smin, smax, twin)
                        elif target >= 0 and state[c, g, target] >= 0:
                            _bump(state, nmem, c, g, target, 1, smin, smax, twin)
                    counts[1] += 1
                    continue
                fillable = True
                for g in range(G):
                    if (not gt[c, g]) and nmem[c, g] > 0:
                        fillable = False
                        break
                if fillable:
                    if seeds_left == 0:
                        continue
                    seeds_left -= 1
                    ne = 0
                    for s in range(sides):
                        if not (tied and s > 0):
                            ne = 0
                            for k in range(K):
                                if nmem[c, s * K + k] == 0:
                                    pick[ne] = s * K + k
                                    ne += 1
                        if ne == 0:
                            continue
                        for j in range(bounds[e, s], bounds[e, s + 1]):
                            if q_seed < 1.0 and np.random.random() >= q_seed:
                                continue
                            w = ids[j]
                            g1 = pick[0] if (ne == 1 or seed_all) else pick[np.random.randint(ne)]
                            for q in range(ne if seed_all else 1):
                                g = pick[q] if seed_all else g1
                                _raise_to(state, nmem, c, g, w, seed_state, init_state, (g + K) % G if tied else -1)
                    counts[0] += 1
                    continue
                nd = _designate(gt[c], nf, sides, K, mode, min_true, desig)
                if nd == 0:
                    continue
                for s in range(sides):
                    ne = 0
                    for k in range(K):
                        if desig[s * K + k]:
                            pick[ne] = s * K + k
                            ne += 1
                    if ne == 0:
                        continue
                    for j in range(bounds[e, s], bounds[e, s + 1]):
                        if p_rec < 1.0 and np.random.random() >= p_rec:
                            continue
                        g = pick[0] if ne == 1 else pick[np.random.randint(ne)]
                        w = ids[j]
                        if state[c, g, w] < 0:
                            _bump(state, nmem, c, g, w, 1, smin, smax, (g + K) % G if tied else -1)
                counts[2] += 1
            else:
                if nf == 0:
                    g0 = np.random.randint(G)
                    if reject_mode == 2:
                        best = smax + 1
                        nties = 0
                        for g in range(G):
                            s = g // K
                            mx = -1
                            for j in range(bounds[e, s], bounds[e, s + 1]):
                                mx = max(mx, state[c, g, ids[j]])
                            if mx < best:
                                best = mx
                                g0 = g
                                nties = 1
                            elif mx == best:
                                nties += 1
                                if np.random.randint(nties) == 0:
                                    g0 = g
                    for g in range(G):
                        if reject_mode != 1 and g != g0:
                            continue
                        s = g // K
                        for j in range(bounds[e, s], bounds[e, s + 1]):
                            w = ids[j]
                            if state[c, g, w] >= 0:
                                _bump(state, nmem, c, g, w, -r, smin, smax, (g + K) % G if tied else -1)
                    counts[3] += 1
                    continue
                nd = _designate(gt[c], nf, sides, K, mode, min_true, desig)
                if nd == 0:
                    continue
                for s in range(sides):
                    ne = 0
                    for k in range(K):
                        if desig[s * K + k]:
                            pick[ne] = s * K + k
                            ne += 1
                    if ne == 0:
                        continue
                    for j in range(bounds[e, s], bounds[e, s + 1]):
                        if p_rep < 1.0 and np.random.random() >= p_rep:
                            continue
                        g = pick[0] if ne == 1 else pick[np.random.randint(ne)]
                        w = ids[j]
                        if state[c, g, w] < 0:
                            _bump(state, nmem, c, g, w, -1, smin, smax, (g + K) % G if tied else -1)
                counts[4] += 1


@njit(cache=True)
def votes_all(state, nmem, pol, ids, bounds, K, fire_out):
    """Vote sum of every example. fire_out [n, C] (bool) is filled when it has n rows, else ignored."""
    n = bounds.shape[0]
    C, G = nmem.shape
    out = np.zeros(n, np.int32)
    gt = np.zeros((C, G), np.bool_)
    nfalse = np.zeros(C, np.int64)
    keep = fire_out.shape[0] == n
    for e in range(n):
        out[e] = _evaluate(state, nmem, pol, ids, bounds, e, K, gt, nfalse)
        if keep:
            for c in range(C):
                fire_out[e, c] = nfalse[c] == 0
    return out


@njit(cache=True)
def side_counts(state, nmem, pol, ids, bounds, K, sides):
    """For every example: both[e] = number of positive clauses that fire (every group of every side true), and
    any_side[e] = number of positive clauses whose groups of at least one side are all true."""
    n = bounds.shape[0]
    C = nmem.shape[0]
    both = np.zeros(n, np.int64)
    any_side = np.zeros(n, np.int64)
    for e in range(n):
        for c in range(C):
            if pol[c] <= 0:
                continue
            n_true_sides = 0
            for s in range(sides):
                ok = True
                for g in range(s * K, (s + 1) * K):
                    t = False
                    if nmem[c, g] > 0:
                        for j in range(bounds[e, s], bounds[e, s + 1]):
                            if state[c, g, ids[j]] >= 0:
                                t = True
                                break
                    if not t:
                        ok = False
                        break
                if ok:
                    n_true_sides += 1
            if n_true_sides == sides:
                both[e] += 1
            if n_true_sides > 0:
                any_side[e] += 1
    return both, any_side


class OrGroupTM:
    def __init__(self, n_clauses, n_features, sides=2, K=1, T=4, mode="near", init_state=-1, seed_state=0, r=2,
                 p_rec=1.0, p_rep=1.0, p_reinf=1.0, q_seed=1.0, neg_frac=0.0, reject_all=False, smax=31, smin=-32,
                 max_seed=0, reject=None, reinforce="all", p_decay=0.0, seed_all=False, tied=False, wmax=0, min_true=1,
                 seed=1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if reject is None:
            reject = "all" if reject_all else "random"
        if reject not in REJECT or reinforce not in REINFORCE:
            raise ValueError(f"reject must be one of {sorted(REJECT)}, reinforce one of {sorted(REINFORCE)}")
        if not (smin <= init_state < 0 <= seed_state <= smax):
            raise ValueError("need smin <= init_state < 0 <= seed_state <= smax")
        if sides < 1 or K < 1 or n_clauses < 1 or T <= 0 or r < 1 or min_true < 1:
            raise ValueError("bad sides, K, n_clauses, T or r")
        self.C, self.F, self.sides, self.K, self.G = int(n_clauses), int(n_features), int(sides), int(K), sides * K
        self.T, self.mode = float(T), mode
        self.init_state, self.seed_state, self.r = int(init_state), int(seed_state), int(r)
        self.p_rec, self.p_rep, self.p_reinf, self.q_seed = float(p_rec), float(p_rep), float(p_reinf), float(q_seed)
        self.reject, self.reinforce, self.p_decay = reject, reinforce, float(p_decay)
        self.seed_all = bool(seed_all)
        self.tied, self.wmax, self.min_true = bool(tied), int(wmax), int(min_true)
        if self.tied and sides != 2:
            raise ValueError("tied needs two sides")
        if self.wmax < 0:
            raise ValueError("wmax must be 0 (off) or positive")
        self.smax, self.smin, self.seed = int(smax), int(smin), int(seed)
        self.max_seed = int(max_seed)
        self.state = np.full((self.C, self.G, self.F), self.init_state, np.int16)
        self.nmem = np.zeros((self.C, self.G), np.int64)
        n_neg = round(neg_frac * self.C)
        self.pol = np.ones(self.C, np.int64)
        if n_neg:
            self.pol[self.C - n_neg:] = -1
        self.wts = np.full(self.C, self.wmax // 2, np.int64)
        self.counts = np.zeros(len(EVENTS), np.int64)
        self.passes_done = 0

    def fit_pass(self, ids, bounds, y, order=None, force_gate=False):
        """One pass. order defaults to a permutation drawn from (seed, pass number)."""
        if bounds.shape[1] != self.sides + 1:
            raise ValueError("bounds do not match the number of sides")
        if len(ids) and int(ids.max()) >= self.F:
            raise ValueError("word id out of range")
        y = np.ascontiguousarray(y, np.int64)
        if order is None:
            order = np.random.default_rng([self.seed, self.passes_done, 7]).permutation(len(y))
        order = np.ascontiguousarray(order, np.int64)
        rng_seed = (self.seed * 1000003 + self.passes_done * 7919 + 13) % (2 ** 31 - 1)
        train_pass(self.state, self.nmem, self.pol, self.wts, ids, bounds, y, order, self.sides, self.K, self.T,
                   MODES[self.mode], self.init_state, self.seed_state, self.r, self.p_rec, self.p_rep, self.p_reinf,
                   self.q_seed, REJECT[self.reject], REINFORCE[self.reinforce], self.p_decay, self.seed_all, self.tied,
                   self.wmax, self.smax, self.smin,
                   self.max_seed, self.min_true,
                   bool(force_gate), rng_seed, self.counts)
        self.passes_done += 1
        return self

    def votes(self, ids, bounds, return_fire=False):
        fire = np.zeros((bounds.shape[0] if return_fire else 0, self.C), np.bool_)
        v = votes_all(self.state, self.nmem, self.pol, ids, bounds, self.K, fire)
        return (v, fire) if return_fire else v

    def groups(self):
        """members[c][g] = sorted array of member word ids."""
        return [[np.flatnonzero(self.state[c, g] >= 0) for g in range(self.G)] for c in range(self.C)]

    def scores(self, ids, bounds):
        """Readings of the same clauses: 'unit' = the vote sum; for pairs 'ratio' = the number of positive clauses
        that fire divided by the number whose groups are all true on at least one of the two texts (the overlap
        of the two texts counted in clauses instead of words); with wmax > 0 also 'weight' = the sum of the
        weights of the firing clauses and 'logodds' = the sum of log((w + 0.5) / (wmax - w + 0.5))."""
        v, fire = self.votes(ids, bounds, return_fire=True)
        out = {"unit": v.astype(np.float64)}
        if self.sides == 2:  # share of the clauses matched by either text that are matched by both
            both, any_side = side_counts(self.state, self.nmem, self.pol, ids, bounds, self.K, self.sides)
            out["ratio"] = both / np.maximum(any_side, 1)
        if self.wmax > 0:
            w = self.wts.astype(np.float64) * self.pol
            out["weight"] = fire.astype(np.float64) @ w
            lo = np.log((self.wts + 0.5) / (self.wmax - self.wts + 0.5)) * self.pol
            out["logodds"] = fire.astype(np.float64) @ lo
        return out

    def twins_consistent(self):
        """Tied sides: the two copies of every group hold the same states."""
        return (not self.tied) or bool((self.state[:, :self.K] == self.state[:, self.K:]).all())

    def nmem_consistent(self):
        return bool(((self.state >= 0).sum(-1) == self.nmem).all())

    def event_counts(self):
        return {k: int(v) for k, v in zip(EVENTS, self.counts)}
