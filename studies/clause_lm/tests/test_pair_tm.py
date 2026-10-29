"""src/pair_tm.py without a GPU: its pure helpers, and main() against a fake package module.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_pair_tm.py [PAIR_TM]

Run from studies/clause_lm. CPU only, a few seconds. PAIR_TM defaults to src/pair_tm.py; give another copy (for
example the version before 2026-10-25) to test it the same way. Neither pycuda nor PyHierarchicalTsetlinMachineCUDA is
imported: part 2 puts a fake module of that name into sys.modules, whose machine scores pairs from a table set by the
test, with a quality that peaks at pass 5 and falls after it, and whose save() returns states with known include bits.
Toy pair data and the run folders live in a temporary folder; nothing is written into results/ or data/.

Part 1, helpers (need the 2026-10-25 version):
- dense(): one-hot u in the first K bits, w in the last K, uint32;
- auc(): equals a brute-force pairwise AUC with ties counted 1/2; nan without both classes;
- include_bits_from_state(): equals the package's own rule for every clause, team and literal (tm.py:236:
  bit k % 32 of chunk k // 32 in the most significant state plane), with K not a multiple of 32;
- clause_groups(): only single-word options give words; empty and dead (two or more words) options are counted;
  odd clauses are skipped;
- eval_halves(), eval_pick(), pick_best_on_tune().
Part 2, main() end to end (BUG-EVAL), with 50,000 validation pairs (PTB has 93,766; both exceed the old default
subset of 40,000):
- by default every validation pair is scored at every evaluation (the pairs pair_baselines.py scores);
- metrics.json reports best_on_tune (the pass with the highest tune-half unseen AUC: pass 5 here) and last_pass
  (pass 20) separately, each with all, tune and check values;
- groups.json, include_bits.npz and model.pkl come from the best-on-tune pass, model_last.pkl from the last;
- with --n-eval 30000 (more than the unseen pairs, fewer than all), only that subset is scored and
  eval_all_pairs is false.
"""

import json
import pickle
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()


def ref_include(raw4, c, team, k):
    """tm.py:236 of the package (ta_action): (state[clause, leaf, ta // 32, bits-1] & (1 << (ta % 32))) > 0."""
    return (int(raw4[c, team, k // 32, raw4.shape[3] - 1]) & (1 << (k % 32))) > 0


def make_state(C, L, K, bits, includes, rng):
    """Flat uint32 state with the given {(clause, team): [literals]} included and random lower planes."""
    ch = (K - 1) // 32 + 1
    raw4 = rng.integers(0, 2**32, size=(C, L, ch, bits), dtype=np.uint64).astype(np.uint32)
    raw4[..., bits - 1] = 0
    for (c, team), lits in includes.items():
        for k in lits:
            raw4[c, team, k // 32, bits - 1] |= np.uint32(1 << (k % 32))
    return raw4.reshape(-1), raw4


def part1(mod):
    if not all(hasattr(mod, f) for f in ("include_bits_from_state", "clause_groups", "eval_halves", "eval_pick",
                                        "pick_best_on_tune", "eval_metrics")):
        check(False, "part 1: the module has the helpers of 2026-10-25")
        return
    K = 7
    X = mod.dense(np.array([0, 6, 3]), np.array([2, 0, 6]), K)
    want = np.zeros((3, 14), np.uint32)
    want[[0, 1, 2], [0, 6, 3]] = 1
    want[[0, 1, 2], [K + 2, K + 0, K + 6]] = 1
    check(X.dtype == np.uint32 and np.array_equal(X, want), "dense(): one-hot u then one-hot w, uint32")

    rng = np.random.default_rng(0)
    sc = rng.integers(0, 5, 60).astype(float)
    y = rng.integers(0, 2, 60)
    pos, neg = sc[y == 1], sc[y == 0]
    bf = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    check(abs(mod.auc(sc, y) - bf) < 1e-12, f"auc() equals the pairwise AUC with ties as 1/2 ({bf:.4f})")
    check(np.isnan(mod.auc(sc, np.ones(60, int))), "auc() is nan with one class only")

    C, A, K, bits = 6, 3, 45, 8
    L = 2 * A
    includes = {(c, t): sorted(rng.choice(K, int(rng.integers(0, 4)), replace=False).tolist())
                for c in range(C) for t in range(L)}
    flat, raw4 = make_state(C, L, K, bits, includes, rng)
    inc = mod.include_bits_from_state(flat, C, L, (K - 1) // 32 + 1, bits, K)
    ok = inc.shape == (C, L, K) and all(inc[c, t, k] == ref_include(raw4, c, t, k)
                                        for c in range(C) for t in range(L) for k in range(K))
    check(ok, "include_bits_from_state() equals the package rule for every clause, team and literal (K = 45)")
    check(all(sorted(np.flatnonzero(inc[c, t]).tolist()) == includes[(c, t)] for c in range(C) for t in range(L)),
          "include_bits_from_state() recovers the included literals")
    try:
        mod.include_bits_from_state(flat[:-1], C, L, (K - 1) // 32 + 1, bits, K)
        refused = False
    except AssertionError:
        refused = True
    check(refused, "include_bits_from_state() refuses a state of the wrong size")

    vocab = [f"v{i}" for i in range(K)]
    inc = np.zeros((4, 2 * A, K), bool)
    inc[0, 0, 3] = True                    # u option 0: {3}, live
    inc[0, 1, [5, 7]] = True               # u option 1: {5, 7}, dead
    inc[0, 2, 9] = True                    # u option 2: {9}, live
    inc[0, A + 0, 4] = True                # w option 0: {4}; w options 1 and 2 empty
    inc[2, :A, 1] = True                   # clause 2: u options all {1}
    inc[2, A:, [8, 8, 8]] = True           # (each w option holds 8)
    inc[1, :, :] = True                    # odd clause: skipped
    g = mod.clause_groups(inc, A, vocab, 4)
    check(len(g) == 2 and [x["clause"] for x in g] == [0, 2], "clause_groups(): even clauses only")
    g0, g2 = g
    check(g0["G1_ids"] == [3, 9] and g0["G1"] == ["v3", "v9"] and g0["dead_options_G1"] == 1
          and g0["empty_options_G1"] == 0, f"clause 0, u side: words of live options only, 1 dead ({g0['G1_ids']})")
    check(g0["G2_ids"] == [4] and g0["empty_options_G2"] == 2 and g0["dead_options_G2"] == 0,
          "clause 0, w side: 1 live option, 2 empty")
    check(g2["G1_ids"] == [1] and g2["G2_ids"] == [8] and g2["empty_options_G1"] == 0, "clause 2: repeated options")

    tune = mod.eval_halves(10)
    check(tune.tolist() == [True, False, True, False, True] * 2, "eval_halves(): positive i and negative n + i together")
    seen = np.array([True, False] * 10)
    check(mod.eval_pick(seen, 0).tolist() == list(range(20)) and mod.eval_pick(seen, 25).tolist() == list(range(20)),
          "eval_pick(): 0 or a large number means all pairs")
    sub = mod.eval_pick(seen, 14)
    check(len(sub) == 14 and set(np.flatnonzero(~seen)) <= set(sub.tolist()) and np.all(np.diff(sub) > 0)
          and sub.tolist() == mod.eval_pick(seen, 14).tolist(), "eval_pick(): subset = all unseen + fixed seen draw")
    h = [{"pass_": 1, "tune": {"auc_unseen": 0.6}}, {"pass_": 5, "tune": {"auc_unseen": 0.8}},
         {"pass_": 10, "tune": {"auc_unseen": float("nan")}}, {"pass_": 15, "tune": {"auc_unseen": 0.8}}]
    check(mod.pick_best_on_tune(h)["pass_"] == 5, "pick_best_on_tune(): highest, earliest on a tie, nan ignored")
    check(mod.pick_best_on_tune([{"pass_": 1, "tune": {"auc_unseen": float("nan")}}]) is None,
          "pick_best_on_tune(): None when no value is a number")


class FakeMachine:
    """Stands in for PyHierarchicalTsetlinMachineCUDA.tm.MultiClassTsetlinMachine (2 classes)."""
    truth, K, A, C = {}, 0, 0, 0
    scored, passes_seen = [], 0

    def __init__(self, n_clauses, T, s, hierarchy_structure=None, append_negated=True, number_of_state_bits=8,
                 boost_true_positive_feedback=1, seed=None):
        FakeMachine.C = n_clauses
        self.bits = number_of_state_bits
        self.passes = 0
        self.tms = [object(), object()]

    def fit(self, X, Y):
        self.passes += 1                    # the toy data has one batch per pass
        FakeMachine.passes_seen = self.passes

    def quality(self):
        return {1: 0.2, 5: 0.45, 10: 0.3, 15: 0.2, 20: 0.1}.get(self.passes, 0.15)

    def score(self, X):
        K = FakeMachine.K
        u, w = X[:, :K].argmax(1), X[:, K:].argmax(1)
        y = np.array([FakeMachine.truth[(a, b)] for a, b in zip(u.tolist(), w.tolist())], float)
        noise = ((u * 7919 + w * 104729) % 1000) / 1000.0
        FakeMachine.scored.append((self.passes, len(u)))
        s1 = self.quality() * y + (1 - self.quality()) * noise
        return np.vstack([np.zeros(len(u), np.float32), s1.astype(np.float32)])

    def save(self):
        C, A, K = FakeMachine.C, FakeMachine.A, FakeMachine.K
        word = 3 if self.passes == 5 else 6  # clause 0, u option 0 holds word 3 at pass 5 and word 6 otherwise
        flat, _ = make_state(C, 2 * A, K, self.bits, {(0, 0): [word], (0, A): [4], (0, 1): [10, 11]},
                             np.random.default_rng(self.passes))
        return {"pass": self.passes, "tms": [{"ta_state_hierarchy": flat * 0}, {"ta_state_hierarchy": flat}]}


def fake_package():
    tm = types.ModuleType("PyHierarchicalTsetlinMachineCUDA.tm")
    tm.AND_GROUP, tm.OR_GROUP, tm.AND_ALTERNATIVES, tm.OR_ALTERNATIVES = 0, 1, 2, 3
    tm.MultiClassTsetlinMachine = FakeMachine
    pkg = types.ModuleType("PyHierarchicalTsetlinMachineCUDA")
    pkg.tm = tm
    sys.modules["PyHierarchicalTsetlinMachineCUDA"] = pkg
    sys.modules["PyHierarchicalTsetlinMachineCUDA.tm"] = tm
    sys.modules.pop("pycuda", None)


def toy_pairs(root, K):
    rng = np.random.default_rng(11)
    n_tr, n_va = 600, 25000                  # 50,000 validation pairs: more than the old default of 40,000
    Xu, Xw = rng.integers(0, K, 2 * n_tr), rng.integers(0, K, 2 * n_tr)
    Y = np.r_[np.ones(n_tr), np.zeros(n_tr)]
    vu = rng.integers(0, K, n_va)
    Vu, Vw = np.r_[vu, vu], np.r_[rng.integers(0, K, n_va), rng.integers(0, K, n_va)]
    VY = np.r_[np.ones(n_va), np.zeros(n_va)]
    Vseen = rng.random(2 * n_va) < 0.6
    (root / "data").mkdir()
    np.savez_compressed(root / "data" / f"ptb_pairs_K{K}.npz", Xu=Xu.astype(np.int16), Xw=Xw.astype(np.int16),
                        Y=Y.astype(np.uint8), Vu=Vu.astype(np.int16), Vw=Vw.astype(np.int16), VY=VY.astype(np.uint8),
                        Vseen=Vseen, top=np.arange(K, dtype=np.int32), q=np.full(K, 1.0 / K))
    (root / "data" / f"vocab_K{K}.json").write_text(json.dumps([f"word{i}" for i in range(K)]))
    truth = {}
    for a, b, y in zip(Vu.tolist(), Vw.tolist(), VY.tolist()):
        truth[(a, b)] = max(truth.get((a, b), 0), int(y))
    return truth, len(VY)


def run_main(mod, root, extra):
    FakeMachine.scored = []
    before = set((root / "results").glob("*")) if (root / "results").exists() else set()
    sys.argv = ["pair_tm.py", "--K", str(FakeMachine.K), "--A", str(FakeMachine.A), "--clauses", "4", "--passes",
                "20", "--eval-every", "5", "--n-train", "0", "--max-minutes", "0", *extra]
    err = None
    try:
        mod.main()
    except BaseException as e:  # the version before the fix may crash; the test reports it
        err = e
    new = sorted(set((root / "results").glob("*")) - before)
    return (new[-1] if new else None), err


def part2(mod, root):
    K, A = 1000, 3                           # K as in PTB, so that toy pairs rarely repeat
    FakeMachine.K, FakeMachine.A = K, A
    truth, n_valid = toy_pairs(root, K)
    FakeMachine.truth = truth
    mod.ROOT = str(root)
    d, err = run_main(mod, root, [])
    check(err is None, f"main() runs with the default options ({err!r})")
    evals = sorted({p for p, n in FakeMachine.scored})
    per_eval = {p: sum(n for q, n in FakeMachine.scored if q == p) for p in evals}
    check(evals == [1, 5, 10, 15, 20] and all(n == n_valid for n in per_eval.values()),
          f"every validation pair ({n_valid}) is scored at every evaluation ({per_eval})")
    if err is not None or d is None or not (d / "metrics.json").exists():
        check(False, "metrics.json was written")
        return
    m = json.loads((d / "metrics.json").read_text())
    check(m.get("eval_pairs") == n_valid and m.get("eval_all_pairs") is True, "metrics: all pairs were scored")
    b, last = m.get("best_on_tune", {}), m.get("last_pass", {})
    check(b.get("pass_") == 5 and last.get("pass_") == 20 and m.get("picked_on") == "tune half, auc_unseen",
          f"best_on_tune is pass {b.get('pass_')} and last_pass is pass {last.get('pass_')}, reported separately")
    check(all(k in b for k in ("all", "tune", "check")) and b["check"]["auc_unseen"] == b["check"]["auc_unseen"],
          f"best_on_tune has all, tune and check AUCs (check unseen {b.get('check', {}).get('auc_unseen')})")
    groups = json.loads((d / "groups.json").read_text())
    check(groups[0]["G1_ids"] == [3] and groups[0]["G2_ids"] == [4] and groups[0]["dead_options_G1"] == 1,
          f"groups.json is from the best-on-tune pass (clause 0: G1 {groups[0]['G1_ids']})")
    z = np.load(d / "include_bits.npz")
    inc = np.unpackbits(z["class1"], axis=-1)[..., :K].astype(bool)
    check(int(z["pass_"]) == 5 and inc.shape == (4, 2 * A, K) and inc[0, 0, 3] and not inc[0, 0, 6],
          "include_bits.npz is from the best-on-tune pass")
    with open(d / "model.pkl", "rb") as f:
        best_state = pickle.load(f)
    with open(d / "model_last.pkl", "rb") as f:
        last_state = pickle.load(f)
    check(best_state["pass"] == 5 and last_state["pass"] == 20, "model.pkl is the best-on-tune pass, model_last.pkl the last")

    d, err = run_main(mod, root, ["--n-eval", "30000", "--name", "pair_tm_subset"])
    m = json.loads((d / "metrics.json").read_text()) if d and (d / "metrics.json").exists() else {}
    per_eval = {p: sum(n for q, n in FakeMachine.scored if q == p) for p in {p for p, n in FakeMachine.scored}}
    check(err is None and m.get("eval_pairs") == 30000 and m.get("eval_all_pairs") is False
          and all(n == 30000 for n in per_eval.values()),
          f"with --n-eval 30000 only that subset is scored, and it says so ({per_eval}; {err!r})")


def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else toyutil.SRC / "pair_tm.py"
    print(f"pair_tm under test: {target}")
    with tempfile.TemporaryDirectory(prefix="clm_pairtm_") as tmp:
        root = Path(tmp)
        toyutil.use_src(root)
        fake_package()
        mod, _ = toyutil.load_module_copy(target, "pair_tm_under_test")
        check("pycuda" not in sys.modules, "importing the module does not import pycuda")
        part1(mod)
        part2(mod, root)
        check("pycuda" not in sys.modules, "pycuda was never imported")
    check.done()


if __name__ == "__main__":
    main()
