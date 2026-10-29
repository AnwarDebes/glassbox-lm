"""src/clause_lm.py on a toy corpus: KN3 alone when no group is given, and one group filter for both arms.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_clause_lm.py [CLAUSE_LM]

Run from studies/clause_lm. CPU only, a few seconds. CLAUSE_LM defaults to src/clause_lm.py; give another copy (for
example the version before 2026-10-25) to test it the same way. Toy sentences (tests/toyutil.py) in a temporary
folder; the module's ROOT holds a toy data/ptb_pairs_K{K}.npz with `top`, and src/runlog.py writes its run folders
there. No Penn Treebank file is read and nothing is written into results/.

Asserts:
- with an empty groups.json the model is KN3 alone: no group, no position fires, and KN3 with clause votes equals KN3
  in perplexity, top-1 and top-5, overall and on seen and unseen trigram contexts;
- that KN3 perplexity equals the one of kn.KN3 on the same stream (computed here independently of clause_lm.py);
- BUG-M2FILTER: from a groups.json with a 1-word group, a 2-word group, a 3-word group whose u-side has an empty
  option and an empty group, only the 2-word group is kept, as for k-means groups (at least 2 words);
- with that group the model runs, a group fires, the chosen lambda values are on the grid and the perplexity is finite.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
K = 12


def run_main(mod, groups_path, root):
    """Calls mod.main() with --groups PATH and returns the metrics and config of the run folder it wrote."""
    before = set((root / "results").glob("*")) if (root / "results").exists() else set()
    sys.argv = ["clause_lm.py", "--groups", str(groups_path), "--K", str(K)]
    mod.main()
    new = sorted(set((root / "results").glob("*")) - before)
    d = new[-1]
    return json.loads((d / "metrics.json").read_text()), json.loads((d / "config.json").read_text())


def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else toyutil.SRC / "clause_lm.py"
    print(f"clause_lm under test: {target}")
    with tempfile.TemporaryDirectory(prefix="clm_clauselm_") as tmp:
        root = Path(tmp)
        train, valid = toyutil.toy_splits(seed=7, n_train=400, n_valid=100)
        ptb, runlog = toyutil.use_src(root, train, valid)
        import kn
        mod, _ = toyutil.load_module_copy(target, "clause_lm_under_test")
        mod.ROOT = str(root)
        vocab, w2i, ids = ptb.load(("train", "valid"))
        tr, va = ids["train"].astype(np.int64), ids["valid"].astype(np.int64)
        V, eos = len(vocab), w2i["<eos>"]
        counts = np.bincount(tr, minlength=V)
        top = np.argsort(-counts, kind="stable")[:K]
        (root / "data").mkdir()
        np.savez(root / "data" / f"ptb_pairs_K{K}.npz", top=top.astype(np.int32))

        m = kn.KN3(tr, V, eos)
        lp = np.zeros(len(va))
        for pos, t, u, d, seen in m.walk(va):
            lp[pos] = np.log(d[va[pos]])
        ppl_ref = float(np.exp(-lp.mean()))

        empty = root / "groups_empty.json"
        empty.write_text("[]")
        met, cfg = run_main(mod, empty, root)
        check(cfg["n_groups"] == 0 and met["fire_share"] == 0.0, "no group given: no group, nothing fires")
        same = all(met[k]["kn3"] == met[k]["kn3+clauses"] for k in ("all", "trigram_unseen", "trigram_seen"))
        check(same, "no group given: KN3 with clause votes equals KN3 (ppl, top-1, top-5; all, unseen, seen)")
        check(abs(met["all"]["kn3"]["ppl"] - ppl_ref) <= 1e-9 * ppl_ref,
              f"KN3 perplexity equals kn.KN3 on the same stream ({met['all']['kn3']['ppl']!r} vs {ppl_ref!r})")
        check(met["all"]["kn3"]["n"] == len(va), "every validation token is scored")

        gj = [dict(clause=0, G1_ids=[3], empty_options_G1=0),
              dict(clause=2, G1_ids=[0, 1], empty_options_G1=0),
              dict(clause=4, G1_ids=[4, 5, 6], empty_options_G1=1),
              dict(clause=6, G1_ids=[], empty_options_G1=0)]
        gpath = root / "groups_mixed.json"
        gpath.write_text(json.dumps(gj))
        met, cfg = run_main(mod, gpath, root)
        check(cfg["n_groups"] == 1, f"BUG-M2FILTER: only the 2-word group is kept (n_groups {cfg['n_groups']})")
        if hasattr(mod, "tm_groups") and hasattr(mod, "keep_groups"):
            g = mod.tm_groups(gj, top)
            check(g == [top[[0, 1]].tolist()], "tm_groups() maps G1_ids through top and keeps the 2-word group")
            check(mod.keep_groups([[1], [1, 2], [], [5, 6, 7]]) == [[1, 2], [5, 6, 7]],
                  "keep_groups(): at least 2 words, order kept")
        else:
            check(False, "the module has tm_groups() and keep_groups()")
        check(met["fire_share"] > 0 and all(l in mod.LAMS for l in met["lam_chosen_per_half"])
              and np.isfinite(met["all"]["kn3+clauses"]["ppl"]),
              f"with the group: fires on {met['fire_share']:.3f} of tokens, lambda {met['lam_chosen_per_half']}, "
              f"ppl {met['all']['kn3+clauses']['ppl']:.3f} vs KN3 {met['all']['kn3']['ppl']:.3f}")
    check.done()


if __name__ == "__main__":
    main()
