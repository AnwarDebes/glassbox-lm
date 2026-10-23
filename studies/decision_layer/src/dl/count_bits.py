"""2d-a: concept counts for the customer's words so far, an additive block beside xcust (added 2026-10-19).

Spec: PLAN_text_understanding.md of the text-understanding line (project notes, not published), "P7 result and the 2d block
definition", with its amendment (f8f552fd, logged 2026-10-20 00:23:53Z, decided before any 2d-a run: count at
level 3, not at the lowest level, which saturates in long texts). Ported here; nothing from the project notes is imported. The
text-stack map's rules R1 (additive) and R5 (guard domain) apply. A new file, so no frozen file changes
(src/dl/abcd.py, static_text.py and count_text.py stay byte for byte). The level-0 version is kept in
logs/archive/count_bits_level0_eceaadca.py.

- For each concept k of the fitted xcust block, let j_k = 3, or k's highest level where k has fewer than 4 levels.
  n_k = the number of the customer's tokens so far whose cosine to k's centre is at or above k's level-j_k threshold.
  The tokens are exactly the ones xcust's OR reads (the table's known tokens of r["cust"], "(nothing yet)" for an
  empty text), with repetitions. The concepts and thresholds are xcust's own, fitted on the training rows; nothing is
  refitted.
- Bits: n_k >= 2, n_k >= 4, n_k >= 8 for every k, concept by concept (1,536 bits for K 512). n_k >= 1 is xcust's
  level-j_k bit for k, so it is not repeated; ``ABCDViewN.counts`` returns n_k, so that equality can be tested.
- ``ABCDViewN``: ABCDView plus the part "xcustn" (needs "xcust"). The block comes after every ABCDView block, so the
  bits of the other parts are byte-identical to ABCDView's, and the memory bits stay first.
- ``count_domain_rows(names)``: the guard-domain rows of the block in verify_policy.domain_rows' format: n >= 8
  implies n >= 4, n >= 4 implies n >= 2, and n >= 2 implies xcust's level-j_k bit for k.
- ``patch()``: for this process, dl.abcd.ABCDView is ABCDViewN, so the unedited run scripts accept "xcustn". Used by
  scripts/stack_run.py.
A table lookup and a sum: no neural network at any stage on the C4 table.
"""

from __future__ import annotations

import re

import numpy as np

from dl import abcd

COUNT_LEVELS = (2, 4, 8)
COUNT_AT = 3  # the thermometer level counted (the spec's amendment); a concept with fewer levels uses its highest
PART = "xcustn"
_NAME = re.compile(r"^at least (\d+) words of the customer's words so far are close to concept (\d+) .* at level (\d+) \(cos >= ([-0-9.]+)\)$")
_XCUST = re.compile(r"^some word of the customer's words so far is close to concept (\d+) .* at level (\d+) \(cos >= ([-0-9.]+)\)$")


def count_level(th) -> int:
    """j_k: COUNT_AT, or the concept's highest level where it has fewer levels."""
    return min(COUNT_AT, len(th) - 1)


class ABCDViewN(abcd.ABCDView):
    """ABCDView with the 2d-a block "xcustn" (concept counts of the customer's words so far) after all other blocks."""

    def __init__(self, parts=("mem", "intent", "step", "words"), **kw):
        parts = tuple(parts)
        self.with_counts = PART in parts
        base = tuple(p for p in parts if p != PART)
        if self.with_counts and "xcust" not in base:
            raise ValueError(f"part {PART} needs xcust (it reuses xcust's concepts and thresholds)")
        super().__init__(parts=base, **kw)
        self.parts_with_counts = parts

    def fit(self, rows: list[dict], emb: dict, buttons_: list[str]) -> "ABCDViewN":
        super().fit(rows, emb, buttons_)
        if not self.with_counts:
            return self
        x = self.xcust
        if any(len(t) == 0 for t in x.th):
            raise ValueError("a concept of xcust has no thermometer level")
        if any(not np.all(np.diff(t) > 0) for t in x.th):
            raise ValueError("xcust's thresholds are not strictly ascending")
        self.count_at = np.array([count_level(t) for t in x.th], dtype=np.int64)  # j_k per concept
        lows = np.array([t[j] for t, j in zip(x.th, self.count_at)])  # the dtype of xcust's thresholds: xcust's own comparison
        self.count_low = lows
        self._hit = x._S_all >= lows[None, :]  # [V, K]: the word reaches concept k at level j_k
        at = [f" at level {j} (cos >= " for j in self.count_at]
        offs = np.concatenate([[0], np.cumsum([len(t) for t in x.th])[:-1]])
        self.count_xcust_cols = offs + self.count_at  # within the xcust block: k's level-j_k bit
        src = [x.names[int(c)] for c in self.count_xcust_cols]
        if not all(a in n and f"concept {k} (" in n for k, (a, n) in enumerate(zip(at, src))):
            raise ValueError("xcust's names do not line up with its thresholds")
        self.count_names = [n.replace("some word of the customer's words so far is close to",
                                      f"at least {m} words of the customer's words so far are close to")
                            for n in src for m in COUNT_LEVELS]
        self.names = list(self.names) + self.count_names
        self.count_info = {"part": PART, "levels": list(COUNT_LEVELS), "count_at": COUNT_AT, "K": int(x.K), "n_bits": len(self.count_names),
                           "n_concepts_below_level": int((self.count_at < COUNT_AT).sum()),
                           "on": "xcust's concepts and level-3 thresholds (the highest level where a concept has no level 3), fitted on train",
                           "spec": "PLAN_text_understanding.md f8f552fd"}
        return self

    def counts(self, texts: list[str]) -> np.ndarray:
        """[n, K] int32: n_k per text, over the tokens xcust's OR reads, with repetitions."""
        ids = self.xcust._table().ids(texts)
        out = np.zeros((len(ids), self._hit.shape[1]), np.int32)
        for i, t in enumerate(ids):
            if t:
                out[i] = self._hit[t].sum(0)
        return out

    def transform(self, rows: list[dict], emb: dict) -> np.ndarray:
        X = super().transform(rows, emb)
        if not self.with_counts:
            return X
        n = self.counts([r["cust"] for r in rows])
        bits = np.stack([n >= m for m in COUNT_LEVELS], 2).reshape(len(rows), -1)  # concept by concept, levels inside
        return np.concatenate([X, bits.astype(np.uint8)], 1)

    def static_info(self) -> dict:
        info = super().static_info()
        if self.with_counts:
            info[PART] = self.count_info
        return info


def count_domain_rows(names: list[str]) -> list[tuple[dict[int, float], float, float]]:
    """Rows (coefficients, lower, upper) over bit indices: x[n>=8] <= x[n>=4] <= x[n>=2] <= x[xcust's level-j_k bit of k]."""
    xbit: dict[tuple[int, int], int] = {}
    chain: dict[int, dict[int, int]] = {}
    level: dict[int, int] = {}
    for i, n in enumerate(names):
        m = _XCUST.match(n)
        if m:
            xbit[(int(m.group(1)), int(m.group(2)))] = i
            continue
        m = _NAME.match(n)
        if m:
            k, j = int(m.group(2)), int(m.group(3))
            if level.setdefault(k, j) != j:
                raise ValueError(f"concept {k}: count bits at two levels")
            chain.setdefault(k, {})[int(m.group(1))] = i
    rows = []
    for k, lv in sorted(chain.items()):
        if sorted(lv) != list(COUNT_LEVELS) or (k, level[k]) not in xbit:
            raise ValueError(f"concept {k}: count bits {sorted(lv)}, xcust bit at level {level[k]} {'found' if (k, level[k]) in xbit else 'missing'}")
        rows.append(({lv[8]: 1, lv[4]: -1}, -np.inf, 0))
        rows.append(({lv[4]: 1, lv[2]: -1}, -np.inf, 0))
        rows.append(({lv[2]: 1, xbit[(k, level[k])]: -1}, -np.inf, 0))
    return rows


def patch() -> None:
    """For this process: dl.abcd.ABCDView is ABCDViewN (it accepts the part "xcustn"; without it, it is ABCDView)."""
    abcd.ABCDView = ABCDViewN
