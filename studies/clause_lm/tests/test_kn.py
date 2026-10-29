"""src/kn.py: the Kneser-Ney trigram model on a toy corpus, against a brute-force version written from its formulas.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_kn.py

Run from studies/clause_lm. CPU only, a few seconds. Toy sentences from a random bigram chain (tests/toyutil.py) are
written in the layout src/ptb.py reads, in a temporary folder; no Penn Treebank file is read and nothing is written
into results/ (src/runlog.py is pointed at the temporary folder).

Asserts:
- mid(u) sums to 1 for every word u, and so does P(. | t, u) for every context of the validation stream;
- every distribution equals a dictionary-based interpolated Kneser-Ney trigram written from the same formulas
  (one discount per order, D = n1 / (n1 + 2 n2); continuation counts for the middle order; Pcont smoothed by 1e-3);
- walk() visits every validation position exactly once, with the right (t, u), and flags a seen trigram context
  exactly when the context occurs in training;
- src/ngram_baselines.py (trigram_kn) gives the same perplexity as kn.KN3 on the same stream, as the docstring of
  kn.py says.
"""

import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()


def brute_kn3(train, V, eos):
    """Dictionary-based interpolated KN trigram with kn.py's conventions (two leading <eos>)."""
    s = [eos, eos] + list(train)
    tri = Counter(zip(s[:-2], s[1:-1], s[2:]))
    big_types = set(zip(s[1:-1], s[2:]))

    def disc(counts):
        n1 = sum(1 for c in counts if c == 1)
        n2 = sum(1 for c in counts if c == 2)
        return n1 / (n1 + 2 * n2)

    D3 = disc(tri.values())
    cont1 = np.zeros(V)
    for (u, w) in big_types:
        cont1[w] += 1
    pcont = (cont1 + 1e-3) / (cont1.sum() + 1e-3 * V)
    n_mid = Counter((u, w) for (t, u, w) in tri)          # N1+(. u w): distinct t before (u, w)
    Dm = disc(n_mid.values())
    su, mu = defaultdict(float), defaultdict(int)
    for (u, w), c in n_mid.items():
        su[u] += c
        mu[u] += 1
    ctx = defaultdict(dict)
    for (t, u, w), c in tri.items():
        ctx[(t, u)][w] = c

    def mid(u):
        if su[u] == 0:
            return pcont.copy()
        d = Dm * mu[u] / su[u] * pcont
        for (uu, w), c in n_mid.items():
            if uu == u:
                d[w] += max(c - Dm, 0) / su[u]
        return d

    def tri_dist(t, u):
        base = mid(u)
        if (t, u) not in ctx:
            return base, False
        cs = ctx[(t, u)]
        tot = sum(cs.values())
        d = D3 * len(cs) / tot * base
        for w, c in cs.items():
            d[w] += max(c - D3, 0) / tot
        return d, True

    return mid, tri_dist


def main():
    with tempfile.TemporaryDirectory(prefix="clm_kn_") as tmp:
        train, valid = toyutil.toy_splits(seed=3)
        ptb, runlog = toyutil.use_src(Path(tmp), train, valid)
        import kn
        import ngram_baselines

        vocab, w2i, ids = ptb.load(("train", "valid"))
        V, eos = len(vocab), w2i["<eos>"]
        tr, va = ids["train"].astype(np.int64), ids["valid"].astype(np.int64)
        print(f"toy corpus: V {V}, train {len(tr)} tokens, valid {len(va)} tokens")
        m = kn.KN3(tr, V, eos)
        mid_bf, tri_bf = brute_kn3(tr.tolist(), V, eos)

        mids_ok = all(abs(m.mid(u).sum() - 1) < 1e-12 for u in range(V))
        check(mids_ok, "mid(u) sums to 1 for every u")
        check(all(np.allclose(m.mid(u), mid_bf(u), rtol=0, atol=1e-13) for u in range(V)),
              "mid(u) equals the brute-force middle order for every u")

        seen_pos = np.zeros(len(va), int)
        prev2 = np.concatenate([[eos, eos], va])[:-2]
        prev1 = np.concatenate([[eos], va])[:-1]
        sums_ok = eq_ok = ctx_ok = seenflag_ok = True
        n_ctx = n_seen_ctx = 0
        lp = np.zeros(len(va))
        for pos, t, u, d, seen in m.walk(va):
            n_ctx += 1
            n_seen_ctx += bool(seen)
            seen_pos[pos] += 1
            ctx_ok &= bool(np.all(prev2[pos] == t) and np.all(prev1[pos] == u))
            sums_ok &= abs(d.sum() - 1) < 1e-12
            bf, bf_seen = tri_bf(t, u)
            eq_ok &= bool(np.allclose(d, bf, rtol=0, atol=1e-13))
            seenflag_ok &= (bool(seen) == bf_seen)
            lp[pos] = np.log(d[va[pos]])
        print(f"contexts walked: {n_ctx}, of which seen in training: {n_seen_ctx}")
        check(sums_ok, "P(. | t, u) sums to 1 for every validation context")
        check(eq_ok, "P(. | t, u) equals the brute-force trigram for every validation context")
        check(bool(np.all(seen_pos == 1)), "walk() visits every validation position exactly once")
        check(ctx_ok, "walk() yields the right (t, u) for every position")
        check(seenflag_ok and 0 < n_seen_ctx < n_ctx, "seen flag equals 'context occurs in training' (both kinds occur)")

        ngram_baselines.main()
        runs = sorted((Path(tmp) / "results").glob("*_ngram_baselines"))
        out = json.loads((runs[-1] / "metrics.json").read_text())
        ppl_kn = float(np.exp(-lp.mean()))
        print(f"perplexity: kn.KN3 walk {ppl_kn!r}, ngram_baselines trigram_kn {out['trigram_kn']['ppl']!r}")
        check(abs(ppl_kn - out["trigram_kn"]["ppl"]) <= 1e-9 * ppl_kn, "ngram_baselines.py trigram_kn equals kn.KN3")
        check(abs(out["trigram_kn"]["D"] - m.D3) < 1e-15 and abs(out["trigram_kn"]["D_mid"] - m.Dm) < 1e-15,
              "both scripts use the same discounts")
    check.done()


if __name__ == "__main__":
    main()
