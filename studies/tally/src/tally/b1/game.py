"""B1 planted team game.

n = 256 uniform random bits. R = M / 4 planted rules, pairwise exclusive:
    s = log2(R) address bits (shared by all rules), one address pattern per rule,
    plus one private "free" literal per rule (its own variable, random sign).
    Rule r fires iff the address bits equal pattern r and its free literal is true.
Every x matches exactly one address pattern, so at most one rule fires and
P(some rule fires) = 1/2 exactly at every M. Rule length k = s + 1 = log2(M) - 1.
P(y = 1 | x) = 0.9 if a rule fires, else 0.1, so Hb(p(x)) = Hb(0.1) for every x and the
per-sample entropy is H = Hb(0.1) bits exactly. Because the rules never fire together, the
planted rule set used as a team with the counted log-odds sum reaches H exactly.

Bits are packed into 4 uint64 words per sample (bit j of word j // 64).
"""
import numpy as np

N_BITS = 256
WORDS = 4
P_HI, P_LO = 0.9, 0.1


def hb(p):
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


ENTROPY = float(hb(P_LO))  # exact per-sample entropy in bits


def n_rules(M):
    assert M >= 8 and M & (M - 1) == 0, "M must be a power of two >= 8"
    return M // 4


def rule_len(M):
    return int(np.log2(n_rules(M))) + 1


class Game:
    """Planted rules for one (M, seed). Rules are stored as include masks over literals."""

    def __init__(self, M, seed):
        self.M, self.seed = M, seed
        self.R = n_rules(M)
        self.s = int(np.log2(self.R))
        self.k = self.s + 1
        rng = np.random.default_rng([seed, M, 1])
        var = rng.permutation(N_BITS)
        self.addr_vars = var[:self.s]
        self.free_vars = var[self.s:self.s + self.R]
        self.free_sign = rng.integers(0, 2, self.R)          # 1: x_j must be 1, 0: x_j must be 0
        pats = rng.permutation(self.R)                        # address pattern of rule r
        self.addr_pat = pats
        self.pos = np.zeros((self.R, WORDS), dtype=np.uint64)   # literals x_j included
        self.neg = np.zeros((self.R, WORDS), dtype=np.uint64)   # literals not x_j included
        for r in range(self.R):
            for b in range(self.s):
                j = self.addr_vars[b]
                self._set(r, j, (pats[r] >> b) & 1)
            self._set(r, self.free_vars[r], self.free_sign[r])

    def _set(self, r, j, val):
        m = self.pos if val else self.neg
        m[r, j // 64] |= np.uint64(1) << np.uint64(j % 64)

    def literal_lists(self):
        """[(var, sign), ...] per rule, for logs."""
        out = []
        for r in range(self.R):
            lits = [(int(self.addr_vars[b]), int((self.addr_pat[r] >> b) & 1)) for b in range(self.s)]
            lits.append((int(self.free_vars[r]), int(self.free_sign[r])))
            out.append(lits)
        return out

    def fires(self, X):
        """(n,) bool: does any planted rule fire. Uses the address directly (exclusive rules)."""
        bits = unpack(X)
        addr = np.zeros(len(X), dtype=np.int64)
        for b in range(self.s):
            addr |= bits[:, self.addr_vars[b]].astype(np.int64) << b
        r = np.argsort(self.addr_pat)[addr]                   # rule whose pattern matches
        return bits[np.arange(len(X)), self.free_vars[r]] == self.free_sign[r]

    def p_true(self, X):
        return np.where(self.fires(X), P_HI, P_LO)

    def sample(self, n, rng):
        X = rng.integers(0, np.iinfo(np.uint64).max, size=(n, WORDS), dtype=np.uint64, endpoint=True)
        p = self.p_true(X)
        y = (rng.random(n) < p).astype(np.uint8)
        return X, y, p


def unpack(X):
    """(n, 4) uint64 -> (n, 256) uint8 bits."""
    b = np.unpackbits(X.view(np.uint8).reshape(len(X), WORDS * 8), axis=1, bitorder="little")
    return b


def streams(seed):
    """Independent generators per purpose; the game itself uses its own key."""
    keys = {"train": 2, "holdout": 3, "calib": 4}
    return {k: np.random.default_rng([seed, v]) for k, v in keys.items()}


TUNE_SEED = 10_000  # flat TM tuning game and streams; never one of the registered seeds


def kl_bits(p, q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


def logloss_bits(y, q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -np.where(y == 1, np.log2(q), np.log2(1 - q))
