"""Engine configuration, constants and literal-space layout of docs/11-engine-spec.md.

This module is the shared CONTRACT between the three engine implementations (Python oracle,
C++ host engine, CUDA engine). It contains no learning logic. Everything probabilistic is
converted to integer thresholds here, once, on the host (spec section 3), and the engines
receive only those integers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

from .rng import inv_q32, q32


class Stream(IntEnum):
    MASK = 1
    RATE = 2
    GATE_TGT = 3
    GATE_NEG = 4
    TYPE1_DEC = 5
    INSERT = 6
    INSERT_PICK = 7
    NEG_INSERT = 8
    NEG_PICK = 9
    SEED = 10
    SEED_PICK = 11
    DECAY = 12
    VETO = 13
    PERM = 14
    SAMPLER = 15
    NEG_DRAW = 16


LIT_EMPTY = 0xFFFFFFFF
ANCHOR_NONE = 0xFFFFFFFF
CLS_EMPTY = 0xFFFF
N_EXACT_SEG = 16
MAX_BAG_K = 13
CLUSTER_SIZES = (16, 256, 4096)  # kinds 1, 2, 3
INCLUDE_THRESHOLD = 128
STATE_INSERT = 123
STATE_SEED = 136
STATE_DENSE_INIT = 127
W_MAX = 32767

MODES = ("masked", "diffusion", "causal", "tabular")
NEG_MODES = ("focused", "uniform_binomial", "binomial")
NEG_K_MAX = 32  # binomial mode: at most this many drawn negatives per event (tail mass below 1e-35 at Q <= 1 [derived])
DECAY_MODES = ("lazy_owned", "eager_exact")
SEG_WEIGHTINGS = ("recency", "uniform")


# ---------------------------------------------------------------- symbol space (spec 1.2)
def sym_cluster_base(vocab: int, kind: int) -> int:
    if kind == 0:
        return 0
    return vocab + sum(CLUSTER_SIZES[: kind - 1])


def sym_mask(vocab: int) -> int:
    return vocab + sum(CLUSTER_SIZES)


def sym_edge(vocab: int) -> int:
    return sym_mask(vocab) + 1


def n_sym(vocab: int, n_pol: int = 0) -> int:
    return sym_edge(vocab) + 1 + n_pol


def kind_of(vocab: int, sym: int) -> int:
    """Kind 0..3 for token / cluster symbols, -1 for SYM_MASK, SYM_EDGE and polarity symbols."""
    if sym < vocab:
        return 0
    for kind in (1, 2, 3):
        if sym < sym_cluster_base(vocab, kind) + CLUSTER_SIZES[kind - 1]:
            return kind
    return -1


# ---------------------------------------------------------------- literals (spec 1.4)
def lit_pack(seg: int, sym: int, neg: int) -> int:
    return (seg << 17) | (sym << 1) | neg


def lit_unpack(lit: int) -> tuple[int, int, int]:
    return lit >> 17, (lit >> 1) & 0xFFFF, lit & 1


def seg_offset(seg: int) -> int:
    """Offset of an exact segment: segs 0..7 are -8..-1, segs 8..15 are +1..+8."""
    if not 0 <= seg < N_EXACT_SEG:
        raise ValueError("not an exact segment")
    return seg - 8 if seg < 8 else seg - 7


def offset_seg(o: int) -> int:
    if o == 0 or not -8 <= o <= 8:
        raise ValueError("offset out of range")
    return o + 8 if o < 0 else o + 7


def bag_seg(k: int, side: int) -> int:
    return N_EXACT_SEG + 2 * k + side


def bag_of_seg(seg: int) -> tuple[int, int]:
    """(bucket k, side) of a bag segment; side 0 = left, 1 = right."""
    if seg < N_EXACT_SEG:
        raise ValueError("not a bag segment")
    return (seg - N_EXACT_SEG) // 2, (seg - N_EXACT_SEG) % 2


def bag_range(k: int) -> tuple[int, int]:
    """Inclusive distance range (lo, hi) of bucket k: 8*2^k < d <= 8*2^(k+1)."""
    return 8 * (1 << k) + 1, 8 * (1 << (k + 1))


def is_bag_seg(seg: int) -> bool:
    return seg >= N_EXACT_SEG


def lit_text(lit: int, vocab: int, token_text: Any = None, tabular: bool = False) -> str:
    """Human-readable literal, for example 't[-1]=New' or 'NOT bagL(9..16) has c8:17'.
    In tabular mode (spec section 6) a literal is a feature: 'x5' or 'NOT x5'."""
    seg, sym, neg = lit_unpack(lit)
    if tabular:
        return ("NOT " if neg else "") + f"x{sym}"
    kind = kind_of(vocab, sym)
    if sym == sym_mask(vocab):
        name = "MASK"
    elif sym == sym_edge(vocab):
        name = "EDGE"
    elif kind == 0:
        name = repr(token_text(sym)) if token_text else f"tok:{sym}"
    elif kind > 0:
        bits = (4, 8, 12)[kind - 1]
        name = f"c{bits}:{sym - sym_cluster_base(vocab, kind)}"
    else:
        name = f"pol:{sym}"
    if is_bag_seg(seg):
        k, side = bag_of_seg(seg)
        lo, hi = bag_range(k)
        body = f"bag{'LR'[side]}({lo}..{hi}) has {name}"
    else:
        body = f"t[{seg_offset(seg):+d}]={name}"
    return ("NOT " if neg else "") + body


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class EngineConfig:
    """All hyperparameters of spec section 7 plus the run mode. Logged with every run."""

    mode: str = "masked"
    vocab: int = 2000  # V: number of classes (token ids), or number of classes in tabular mode
    n_features: int = 0  # F, tabular mode only
    n_pol: int = 0
    n_clauses: int = 65536  # C
    seed: int = 1
    T: int = 64
    s: float = 128  # tuned on text 2026-09-22 (docs/12 F17; planted rules used 32, v0.1 16)
    l_slots: int = 32
    v_slots: int = 64  # tuned on text 2026-09-22 (docs/12 F17; was 16)
    budget: int = 12
    k_neg: int = 2
    k_cover: int = 8  # tuned on text 2026-09-22 (docs/12 F17; was 2)
    p_ins: float = 1 / 4  # tuned 2026-09-22 (was 1/8)
    p_neg: float = 1 / 16
    p_veto: float = 1 / 16
    rho_seed: float = 1.0  # tuned on text 2026-09-22 (docs/12 F17; was 1/4)
    seed_cap: int = 32  # tuned on text 2026-09-22 (docs/12 F17; was 4)
    e_cap: int = 256
    sweep_period: int = 256
    dead_sweeps: int = 32  # tuned 2026-09-22 (was 8)
    margin: int = -1  # -1 means T // 4
    n_stop: int = 64
    n_bag_k: int = MAX_BAG_K
    batch: int = 1024  # B
    mask_rate: float = 0.15
    neg_mode: str = "binomial"  # tuned on text 2026-09-22 (docs/12 F17, spec 8c; was focused)
    q: float = 1.0  # uniform_binomial only: expected negatives per clause
    decay_mode: str = "lazy_owned"
    dense_equivalent: bool = False
    seg_weighting: str = "recency"  # candidate-sampling weights (spec 5.2.3, 8b): recency = 2^(8-|o|) exact, 8 per bag (default on text since 2026-09-22, docs/12 F17); uniform = 1 each
    neg_budget: int = 2  # max included NEGATED literals per clause (spec 8b, default since 2026-09-22); -1 = no separate cap
    cand_tries: int = 1  # spec 8d (N15): draw this many candidates, keep the one with the smallest sym_freq (1 = single draw)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.neg_mode not in NEG_MODES or self.decay_mode not in DECAY_MODES:
            raise ValueError("bad neg_mode / decay_mode")
        if self.seg_weighting not in SEG_WEIGHTINGS:
            raise ValueError(f"seg_weighting must be one of {SEG_WEIGHTINGS}")
        if not 0 <= self.n_bag_k <= MAX_BAG_K:
            raise ValueError("n_bag_k out of range")
        if self.mode == "tabular":
            if self.n_features <= 0:
                raise ValueError("tabular mode needs n_features")
        elif not 2 <= self.vocab <= 32768 or n_sym(self.vocab, self.n_pol) > 65535:
            raise ValueError("vocab must be in [2, 32768]")
        if self.dense_equivalent:
            if self.mode != "tabular":
                raise ValueError("dense_equivalent mode is defined for tabular input only")
            if self.l_slots < 2 * self.n_features or self.v_slots < self.vocab:
                raise ValueError("dense_equivalent needs l_slots >= 2F and v_slots >= classes")
        elif self.budget > self.l_slots:
            raise ValueError("budget > l_slots")
        if self.neg_budget < -1:
            raise ValueError("neg_budget must be -1 or >= 0")
        if not 1 <= self.cand_tries <= 8:
            raise ValueError("cand_tries must be in 1..8")
        if self.T < 1 or self.s < 1 or self.batch < 1 or self.n_clauses < 1:
            raise ValueError("T, s, batch, n_clauses must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Derived:
    """Integer thresholds and tables computed once on the host and handed to every engine."""

    inv_s_q32: int
    p_ins_q32: int
    p_neg_q32: int
    p_veto_q32: int
    rho_seed_q32: int
    mask_rate_q32: int
    q_neg_q32: int  # uniform_binomial only: per (clause, class) selection probability Q / (K - 1)
    margin: int
    kind_t70: int
    kind_t80: int
    kind_t90: int
    nlit_t1: int  # n_lit = 1 if u < nlit_t1
    nlit_t2: int  # n_lit = 2 if u < nlit_t2, else 3
    seg_ids: tuple[int, ...]  # segments available for candidate sampling, ascending
    seg_cum: tuple[int, ...]  # cumulative weights, same length
    seg_total: int
    exact_seg_ids: tuple[int, ...]  # the exact subset (negated insertion and seeding use only these)
    exact_cum: tuple[int, ...]
    exact_total: int
    neg_cdf_q32: tuple[int, ...]  # binomial mode: CDF of Binomial(V-1, Q/(V-1)) at k = 0..NEG_K_MAX, Q32, capped at 2^32-1


def binomial_cdf_q32(n: int, p: float, k_max: int) -> tuple[int, ...]:
    """CDF of Binomial(n, p) at k = 0..k_max in Q32 (floor), last entry forced to 2^32 - 1 so the
    inversion never draws more than k_max. Computed in float64 from the log-pmf recursion; the
    absolute error is far below 2^-32 for the (n, p) used here (n = V - 1 up to 32767, p <= 1/n)."""
    import math
    if p <= 0.0 or n <= 0:
        return tuple([(1 << 32) - 1] * (k_max + 1))
    if p >= 1.0:  # every class is selected: k = min(n, k_max) with certainty
        k = min(n, k_max)
        return tuple([0] * k + [(1 << 32) - 1] * (k_max + 1 - k))
    log_p, log_q = math.log(p), math.log1p(-p)
    log_pmf = n * log_q
    cdf = 0.0
    out = []
    for k in range(k_max + 1):
        if k > n:  # pmf is zero beyond n (small V: n = V - 1 < NEG_K_MAX); the CDF is 1 from k = n on
            out.append((1 << 32) - 1)
            continue
        if k > 0:
            log_pmf += math.log(n - k + 1) - math.log(k) + log_p - log_q
        cdf = min(1.0, cdf + math.exp(log_pmf))
        out.append(min((1 << 32) - 1, int(cdf * 4294967296.0)))
    out[-1] = (1 << 32) - 1
    return tuple(out)


def exact_segments(mode: str) -> tuple[int, ...]:
    if mode == "causal":
        return tuple(range(8))
    if mode == "tabular":
        return (0,)
    return tuple(range(N_EXACT_SEG))


def bag_segments(mode: str, n_bag_k: int) -> tuple[int, ...]:
    if mode == "tabular":
        return ()
    sides = (0,) if mode == "causal" else (0, 1)
    return tuple(bag_seg(k, side) for k in range(n_bag_k) for side in sides)


def derive(cfg: EngineConfig) -> Derived:
    cfg.validate()
    exact = exact_segments(cfg.mode)
    if cfg.mode == "tabular" or cfg.seg_weighting == "uniform":
        exact_w = [1] * len(exact)
    else:
        exact_w = [1 << (8 - abs(seg_offset(sg))) for sg in exact]
    bags = sorted(bag_segments(cfg.mode, cfg.n_bag_k))
    segs = list(exact) + bags
    weights = exact_w + [1 if cfg.seg_weighting == "uniform" else 8] * len(bags)

    def cumulative(ws: list[int]) -> tuple[int, ...]:
        out, acc = [], 0
        for w in ws:
            acc += w
            out.append(acc)
        return tuple(out)

    return Derived(
        inv_s_q32=inv_q32(cfg.s),
        p_ins_q32=q32(cfg.p_ins),
        p_neg_q32=q32(cfg.p_neg),
        p_veto_q32=q32(cfg.p_veto),
        rho_seed_q32=q32(cfg.rho_seed),
        mask_rate_q32=q32(cfg.mask_rate),
        q_neg_q32=q32(min(1.0, cfg.q / max(1, cfg.vocab - 1))),
        margin=cfg.T // 4 if cfg.margin < 0 else cfg.margin,
        kind_t70=q32(0.7),
        kind_t80=q32(0.8),
        kind_t90=q32(0.9),
        nlit_t1=q32(0.5),
        nlit_t2=q32(0.85),
        seg_ids=tuple(segs),
        seg_cum=cumulative(weights),
        seg_total=sum(weights),
        exact_seg_ids=tuple(exact),
        exact_cum=cumulative(exact_w),
        exact_total=sum(exact_w),
        neg_cdf_q32=binomial_cdf_q32(cfg.vocab - 1, min(1.0, cfg.q / max(1, cfg.vocab - 1)), NEG_K_MAX),
    )
