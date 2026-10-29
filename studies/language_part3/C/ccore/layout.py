"""The 60 input bits of C-core and their decoding (numpy only, so both environments can import it).

For each previous word t-1 .. t-4 inside the sentence of t, 15 bits: the E path (6 bits, top split
first), the B path (8 bits, top split first) and a pad flag. A position before the sentence start has
pad 1 and all its code bits 0. Bit index of (position k, field) = (k - 1) * 15 + offset."""
import numpy as np

DEPTH_E = 6
DEPTH_B = 8
N_PREV = 4
BITS_PER_WORD = DEPTH_E + DEPTH_B + 1      # 15
N_BITS = N_PREV * BITS_PER_WORD            # 60
K_E = 2 ** DEPTH_E                         # 64 word classes
N_CLASSES = K_E + 1                        # 65 (<eos> class 64)
PAD_E, PAD_B = 2 ** DEPTH_E, 2 ** DEPTH_B  # 64, 256: PAD value of a code component


def prev_words(x, eos, k=N_PREV):
    """(n, k) ids of the words at t-1 .. t-k inside the sentence of t; -1 before the sentence start."""
    x = np.asarray(x, dtype=np.int64)
    n = len(x)
    start_flag = np.zeros(n, dtype=bool)
    start_flag[0] = True
    start_flag[1:] = x[:-1] == eos
    start = np.maximum.accumulate(np.where(start_flag, np.arange(n), 0))
    prev = np.full((n, k), -1, dtype=np.int64)
    for j in range(1, k + 1):
        idx = np.arange(n) - j
        ok = idx >= start
        prev[ok, j - 1] = x[idx[ok]]
    assert not (prev == eos).any(), "a previous word inside a sentence is <eos>"
    return prev


def features(x, eos, path6, path8):
    prev = prev_words(x, eos)
    n = len(prev)
    X = np.zeros((n, N_BITS), dtype=np.uint8)
    for j in range(N_PREV):
        w = prev[:, j]
        pad = w < 0
        e = np.where(pad, 0, path6[np.maximum(w, 0)])
        b = np.where(pad, 0, path8[np.maximum(w, 0)])
        o = j * BITS_PER_WORD
        for i in range(DEPTH_E):
            X[:, o + i] = (e >> (DEPTH_E - 1 - i)) & 1
        for i in range(DEPTH_B):
            X[:, o + DEPTH_E + i] = (b >> (DEPTH_B - 1 - i)) & 1
        X[:, o + BITS_PER_WORD - 1] = pad
    return X


def decode(X):
    """(E6, B8, pad), each (n, 4); E6 = PAD_E and B8 = PAD_B where pad."""
    X = np.asarray(X, dtype=np.int64)
    n = X.shape[0]
    E6 = np.zeros((n, N_PREV), dtype=np.int64)
    B8 = np.zeros((n, N_PREV), dtype=np.int64)
    pad = np.zeros((n, N_PREV), dtype=bool)
    for j in range(N_PREV):
        o = j * BITS_PER_WORD
        for i in range(DEPTH_E):
            E6[:, j] = (E6[:, j] << 1) | X[:, o + i]
        for i in range(DEPTH_B):
            B8[:, j] = (B8[:, j] << 1) | X[:, o + DEPTH_E + i]
        pad[:, j] = X[:, o + BITS_PER_WORD - 1] == 1
    assert not (pad & ((E6 != 0) | (B8 != 0))).any()
    E6[pad] = PAD_E
    B8[pad] = PAD_B
    return E6, B8, pad


# LR inputs (proposal C, section 5): one-hot tree nodes per position: E at depths 2/4/6, B at depths
# 2/4/6/8, and the pad flag; 425 features per position, 1,700 in all, 28 on in a row without pads.
LR_FIELDS = (("E", 2), ("E", 4), ("E", 6), ("B", 2), ("B", 4), ("B", 6), ("B", 8))
LR_PER_POS = sum(2 ** d for _, d in LR_FIELDS) + 1   # 425
LR_N_FEATURES = N_PREV * LR_PER_POS                  # 1,700


def lr_columns(X):
    """Column indices of the 1-features of every row: (n, 4 * 8) with -1 for absent ones."""
    E6, B8, pad = decode(X)
    n = X.shape[0]
    cols = np.full((n, N_PREV, len(LR_FIELDS) + 1), -1, dtype=np.int64)
    for j in range(N_PREV):
        base = j * LR_PER_POS
        off = 0
        for f, (name, d) in enumerate(LR_FIELDS):
            v = (E6[:, j] >> (DEPTH_E - d)) if name == "E" else (B8[:, j] >> (DEPTH_B - d))
            cols[:, j, f] = np.where(pad[:, j], -1, base + off + v)
            off += 2 ** d
        cols[:, j, -1] = np.where(pad[:, j], base + off, -1)
    return cols.reshape(n, -1)


def lr_matrix(X):
    from scipy.sparse import csr_matrix
    cols = lr_columns(X)
    keep = cols >= 0
    indptr = np.concatenate([[0], np.cumsum(keep.sum(1))]).astype(np.int64)
    indices = cols[keep]
    data = np.ones(len(indices), dtype=np.float64)
    M = csr_matrix((data, indices, indptr), shape=(X.shape[0], LR_N_FEATURES))
    M.sort_indices()
    return M
