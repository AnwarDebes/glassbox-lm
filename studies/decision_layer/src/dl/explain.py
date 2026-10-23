"""Print trained clauses as readable rules."""

from __future__ import annotations

import numpy as np


def clause_literals(ta_row: np.ndarray, n_states: int, f: int, names: list[str], negation: bool = True) -> list[str]:
    inc = np.flatnonzero(ta_row > n_states)
    out = []
    for k in inc:
        if k < f:
            out.append(names[k])
        elif negation:
            out.append(f"NOT {names[k - f]}")
    return out


def simplify(lits: list[str]) -> list[str]:
    """Merge thermometer literals of one feature into an interval: "f >= a" literals give the lower
    bound (the largest a), "NOT f >= b" literals the upper bound (the smallest b)."""
    import re

    lo, hi, other = {}, {}, []
    for l in lits:
        m = re.fullmatch(r"(NOT )?(.+) >= (-?[0-9.]+)", l)
        if not m:
            other.append(l)
            continue
        neg, f, t = m.group(1), m.group(2), float(m.group(3))
        if neg:
            hi[f] = min(hi.get(f, float("inf")), t)
        else:
            lo[f] = max(lo.get(f, float("-inf")), t)
    out = []
    for f in sorted(set(lo) | set(hi), key=lambda x: x):
        a, b = lo.get(f), hi.get(f)
        if a is not None and b is not None:
            out.append(f"{a:g} <= {f} < {b:g}" if a < b else f"[never: {f}]")
        elif a is not None:
            out.append(f"{f} >= {a:g}")
        else:
            out.append(f"{f} < {b:g}")
    return other + out


def top_rules(ta: np.ndarray, w: np.ndarray, names: list[str], class_names: list[str], n_states: int = 256,
              per_class: int = 3, classes=None, negation: bool = True, min_literals: int = 1, compact: bool = True) -> list[str]:
    """The ``per_class`` clauses with the largest positive weight for each class, as text."""
    f = len(names)
    lines = []
    for c in (classes if classes is not None else range(w.shape[1])):
        order = np.argsort(-w[:, c])
        shown = 0
        for j in order:
            if w[j, c] <= 0 or shown >= per_class:
                break
            lits = clause_literals(ta[j], n_states, f, names, negation)
            if len(lits) < min_literals:
                continue
            lines.append(f"[{class_names[c]}] +{int(w[j, c])}: IF " + " AND ".join(simplify(lits) if compact else lits))
            shown += 1
    return lines


def decision_receipt(ta: np.ndarray, w: np.ndarray, x_bits: np.ndarray, names: list[str], class_names: list[str],
                     n_states: int = 256, top: int = 3, negation: bool = True) -> str:
    """Why one input got its decision: the firing clauses with the largest vote for the winner."""
    f = len(names)
    lit = np.concatenate([x_bits, 1 - x_bits]) if negation else x_bits
    incl = ta > n_states
    fired = (incl & (lit[None, :] == 0)).sum(1) == 0
    fired &= incl.sum(1) > 0
    S = (fired[:, None] * w).sum(0)
    c = int(np.argmax(S))
    contrib = fired * w[:, c]
    order = np.argsort(-contrib)[:top]
    rows = [f"decision: {class_names[c]} (vote {int(S[c])}; runner-up {class_names[int(np.argsort(-S)[1])]} {int(np.sort(S)[-2])})"]
    for j in order:
        if contrib[j] <= 0:
            break
        rows.append(f"  +{int(contrib[j])}: " + " AND ".join(clause_literals(ta[j], n_states, f, names, negation)))
    return "\n".join(rows)
