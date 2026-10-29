"""src/clause_lm.py on Penn Treebank train and validation, against the recorded k-means run of 2026-10-02.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
        nice -n 19 .venv-cdllm/bin/python tests/test_clause_lm_ptb_valid.py

Run from studies/clause_lm. CPU only, about a minute at 2 threads. Reads the Penn Treebank train and validation text
through src/ptb.py (never the test text: ptb.load refuses it, and a trip wire checks that no path under a test folder
is opened) and data/ptb_pairs_K1000.npz; needs the local run folder results/20261002T174323Z_clause_lm_kmeans200 for
the recorded numbers. src/runlog.py is pointed at a temporary folder, so nothing is written into results/.

Asserts:
- with no group (an empty groups.json), KN3 alone gives exactly the recorded KN3 perplexity, top-1, top-5 and token
  counts of the k-means run (all, unseen and seen trigram contexts), and KN3 with clause votes equals KN3;
- the k-means groups (k = 200) are the same in two calls (the SVD start vector is seeded since 2026-10-25), and the
  version of clause_lm.py that made the recorded run (its copy in the run folder, the version before 2026-10-25)
  makes the same groups as the current one (the shared filter changes nothing for k-means groups).
Skipped when the Penn Treebank text, data/ptb_pairs_K1000.npz or the recorded run folder is missing (as in a clone).
Printed, not asserted: the k-means:200 run with the seeded SVD beside the recorded run. The recorded run used an
unseeded SVD, whose start vector scipy draws from operating-system entropy, so its 157 groups and perplexity 157.24
are one draw that cannot be repeated exactly.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
RECORDED = toyutil.STUDY / "results" / "20261002T174323Z_clause_lm_kmeans200"
ORIG = RECORDED / "clause_lm.py"


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


def run_main(mod, groups, root):
    before = set((root / "results").glob("*")) if (root / "results").exists() else set()
    sys.argv = ["clause_lm.py", "--groups", groups]
    t0 = time.time()
    mod.main()
    new = sorted(set((root / "results").glob("*")) - before)
    return (json.loads((new[-1] / "metrics.json").read_text()), json.loads((new[-1] / "config.json").read_text()),
            time.time() - t0)


def partition(groups):
    return sorted(tuple(sorted(g)) for g in groups)


def main():
    needed = [RECORDED / "metrics.json", ORIG, toyutil.STUDY / "data" / "ptb_pairs_K1000.npz",
              toyutil.STUDY.parent / "tm_language_model" / "data" / "ptb" / "raw" / "train" / "text.utf8",
              toyutil.STUDY.parent / "tm_language_model" / "data" / "ptb" / "raw" / "validation" / "text.utf8"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(f"skip: local files missing: {missing}")
        check.done()
    rec_m = json.loads((RECORDED / "metrics.json").read_text())
    rec_c = json.loads((RECORDED / "config.json").read_text())
    with tempfile.TemporaryDirectory(prefix="clm_ptbvalid_") as tmp:
        root = Path(tmp)
        ptb, _ = toyutil.use_src(root)            # real Penn Treebank; run folders go to the temporary folder
        import clause_lm
        with toyutil.Tripwire("/test/") as tw:
            empty = root / "groups_empty.json"
            empty.write_text("[]")
            met, cfg, dt = run_main(clause_lm, str(empty), root)
            print(f"no group: {dt:.1f} s")
            for k in ("all", "trigram_unseen", "trigram_seen"):
                r, m = rec_m[k]["kn3"], met[k]["kn3"]
                check(r["n"] == m["n"] and all(close(r[x], m[x]) for x in ("ppl", "top1", "top5")),
                      f"KN3 on {k}: ppl {m['ppl']!r} (recorded {r['ppl']!r}), n {m['n']}")
                check(met[k]["kn3+clauses"] == m, f"no group: KN3 with clause votes equals KN3 on {k}")

            vocab, w2i, ids = ptb.load(("train", "valid"))
            tr, V = ids["train"].astype(np.int64), len(vocab)
            top = np.load(toyutil.STUDY / "data" / "ptb_pairs_K1000.npz")["top"].astype(np.int64)
            g1 = clause_lm.kmeans_groups(1000, tr, V, top, 200)
            g2 = clause_lm.kmeans_groups(1000, tr, V, top, 200)
            check(partition(g1) == partition(g2), f"k-means groups are the same in two calls ({len(g1)} groups)")
            old, _ = toyutil.load_module_copy(ORIG, "clause_lm_before")
            g0 = old.kmeans_groups(1000, tr, V, top, 200)
            check(partition(g0) == partition(g1), "the run-folder copy (before 2026-10-25) makes the same k-means groups")

            met, cfg, dt = run_main(clause_lm, "kmeans:200", root)
            print(f"kmeans:200: {dt:.1f} s")
        check(tw.hits == [], "no file under a test folder was opened")
        print(f"INFO kmeans:200 with the seeded SVD: {cfg['n_groups']} groups, lambda {met['lam_chosen_per_half']}, "
              f"fire share {met['fire_share']:.4f}; recorded (unseeded SVD): {rec_c['n_groups']} groups, lambda "
              f"{rec_m['lam_chosen_per_half']}, fire share {rec_m['fire_share']:.4f}")
        for k in ("all", "trigram_unseen", "trigram_seen"):
            print(f"INFO {k}: KN3 with clause votes ppl {met[k]['kn3+clauses']['ppl']:.4f} (recorded "
                  f"{rec_m[k]['kn3+clauses']['ppl']:.4f}), top-1 {met[k]['kn3+clauses']['top1']:.4f} (recorded "
                  f"{rec_m[k]['kn3+clauses']['top1']:.4f})")
    check.done()


if __name__ == "__main__":
    main()
