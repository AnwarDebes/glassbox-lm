"""CPU tests for E5a (no GPU): data files and encoding, batching, metrics, the OR-sum recomputation and the rule
printer on fake include bits, the driver's main loop with a stand-in model, and the NPMI scorer on a toy corpus.
Run from the study folder: ../tm_language_model/.venv-tm/bin/python src/test_e5a.py   (needs pyarrow for the encoding test)
"""
import contextlib, io, json, os, sys, tempfile, types
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_agnews as ba, htm_driver as hd, npmi_groups as ng, text_driver as td  # noqa: E402

RESULTS = {}


def test_data():
    vocab = json.load(open(os.path.join(td.DATA, "vocab.json")))
    assert len(vocab) == len(set(vocab)) == 1024
    rows = {}
    for s, n in (("train", 60000), ("dev", 10000), ("npmi_ref", 40000)):
        X, y, r = td.load_split(td.DATA, s)
        assert X.shape == (n, 1024) and len(y) == len(r) == n and set(np.unique(y)) == {0, 1, 2, 3} and X.max() == 1
        rows[s] = set(r.tolist())
    assert not (rows["train"] & rows["dev"] or rows["train"] & rows["npmi_ref"] or rows["dev"] & rows["npmi_ref"])
    try:
        td.load_split(td.DATA, "test")
        raise AssertionError("the driver must refuse the final-test split")
    except SystemExit:
        pass


def test_encoding():
    import pyarrow.parquet as pq
    tab = pq.read_table(ba.PARQUET, columns=["text", "label"])
    texts, labels = tab.column("text").to_pylist(), tab.column("label").to_pylist()
    vocab, stop = json.load(open(os.path.join(td.DATA, "vocab.json"))), ba.stop_words()
    wid = {w: i for i, w in enumerate(vocab)}
    X, y, r = td.load_split(td.DATA, "train")
    for i in np.random.default_rng(0).choice(len(y), 500, replace=False):
        assert X[i].indices.tolist() == sorted(wid[w] for w in ba.words(texts[r[i]], stop) if w in wid) and labels[r[i]] == y[i]
    df = np.asarray(X.sum(0)).ravel().astype(np.int64)  # (uint64 sums would wrap in np.diff)
    assert (np.diff(df) <= 0).all(), "vocabulary must be ordered by train document frequency"
    assert ba.words("He said quot;Hi quot; &lt;b&gt;bold&lt;/b&gt; at www.x.com #39;s a", stop) == {"said", "hi", "bold"}


def test_batches():
    y = np.repeat(np.arange(4), 50)
    bs = td.batches(y, 60, np.random.default_rng(0).permutation(200))
    assert sum(map(len, bs)) == 200 and all(len(np.unique(y[b])) == 4 for b in bs)
    try:
        td.batches(np.r_[np.zeros(10, int), np.arange(4)], 5, np.arange(14))
        raise AssertionError("a batch without all labels must be refused")
    except AssertionError as e:
        assert "all labels" in str(e)


def test_metrics():
    cs = np.array([[3, 0, 0, 1], [1, 5, 0, 1], [0, 0, 2, 1], [0, 0, 0, 0]], float)  # text 3: classes 0-2 tie -> class 0
    m = td.metrics(cs, np.array([0, 1, 2, 3]))
    assert m["acc"] == 0.75 and m["per_class_acc"] == [1, 1, 1, 0] and m["pred_share"] == [0.5, 0.25, 0.25, 0]
    assert m["tie_share"] == 0.25 and abs(m["macro_f1"] - (2 / 3 + 2) / 4) < 1e-12 and m["flag_unused_class"] and not m["flag_constant"]


VOCAB = ["game", "team", "season", "red", "sox", "oil", "stocks", "said"]


def fake_bits():
    """4 topics x 4 clauses (0, 2 positive) x (A=3 options x K=2 groups) x 8 words."""
    inc, w = np.zeros((4, 4, 6, 8), bool), {v: i for i, v in enumerate(VOCAB)}
    for c, j, leaf, words in [(1, 0, 0, ["game"]), (1, 0, 1, ["team"]), (1, 0, 2, ["red", "sox"]), (1, 0, 3, ["season"]),
                              (1, 0, 4, ["season"]),  # Sports clause 0, group 2 option 3 stays EMPTY
                              (2, 0, 0, ["oil"]), (2, 0, 1, ["stocks"]), (2, 0, 2, ["oil"])] + \
                             [(2, 0, k, ["said"]) for k in (3, 4, 5)] + [(1, 2, k, ["said"]) for k in range(6)]:
        inc[c, j, leaf, [w[x] for x in words]] = True
    return inc


def test_or_sums_and_rules():
    inc = fake_bits()
    docs = [["game", "season"], ["red", "sox", "oil"], ["oil", "stocks", "said"], []]
    X = sp.csr_matrix(np.array([[v in d for v in VOCAB] for d in docs], np.uint8))
    for c in range(4):
        got = td.or_sums(inc[c], X, 3, 2)
        want = [[[sum(set(np.flatnonzero(inc[c, j, k * 3 + a]).tolist()) <= {VOCAB.index(v) for v in d} for a in range(3))
                  for k in range(2)] for j in range(4)] for d in docs]
        assert got.tolist() == want, (c, got.tolist(), want)
    assert td.or_sums(inc[1], X, 3, 2)[:, 0].tolist() == [[1, 3], [1, 1], [0, 1], [0, 1]]  # the empty option counts 1
    fires = [(td.or_sums(i, X, 3, 2) > 0).all(axis=2) for i in inc]
    lines, gj = td.rules(list(inc), fires, np.array([1, 1, 2, 0]), VOCAB, 3, 2, top=5)
    text = "\n".join(lines)
    assert "   Sports: game OR team OR (red & sox)   [3 options, 3 distinct]" in lines, text
    assert "   Sports: ALWAYS-TRUE (1 of 3 options empty; others: season)" in lines, text
    assert "   Business: said   [3 options, 1 distinct]" in lines and "   World: ALWAYS-TRUE (all 3 options empty)" in lines
    sports = gj["classes"][1]["clauses"]
    assert [d["clause"] for d in sports] == [0, 2] and sports[0]["fire_topic"] == 1.0 and sports[1]["fire_other"] == 1 / 2
    counted = ng.counted_groups(gj, 10)
    assert counted == [("Sports", ["game", "team", "red", "sox"]), ("Business", ["oil", "stocks"])], counted
    empty, always = td.group_shares(list(inc), 3, 2)
    assert abs(empty - (1 - 17 / 48)) < 1e-12 and always == 11 / 16  # 17 of 48 options filled; 5 of 16 groups have none empty
    assert td.group_text({"options": [["game"]], "n_empty": 0, "always_true": False, "words": ["game"]}, 1) == "game"


class FakeTM:
    """Stand-in for MultiClassTsetlinMachine: scores = naive word counts per topic (or all zero), fake include bits."""

    def __init__(self, a, good):
        self.good, self.W, self.calls = good, np.zeros((4, a.F)), 0
        rng = np.random.default_rng(0)
        self.tms = [types.SimpleNamespace(inc=rng.random((a.clauses, a.A * a.K, a.F)) < 0.002) for _ in range(4)]

    def fit(self, X, y):
        assert X.dtype == np.uint32 and y.dtype == np.uint32 and len(np.unique(y)) == 4
        for c in range(4):
            self.W[c] += X[y == c].sum(0)
        self.calls += 1

    def score(self, X):
        if not self.good:
            return np.zeros((4, len(X)), np.float32)
        return (X @ np.log((self.W + 1) / (self.W.sum(1, keepdims=True) + self.W.shape[1])).T).T.astype(np.float32)

    def save(self):
        return {"fake": True}


def run_main(argv, good):
    out, saved = io.StringIO(), (td.build, hd.include_bits, sys.argv, os.environ.pop("TMLM_RUN_DIR", None))
    try:
        td.build = lambda a: (FakeTM(a, good), ((0, a.F), (3, a.A), (1, a.K)))
        hd.include_bits, sys.argv = (lambda t: t.inc), ["text_driver.py"] + argv
        with contextlib.redirect_stdout(out):
            td.main()
    finally:
        td.build, hd.include_bits, sys.argv = saved[:3]
        if saved[3] is not None:
            os.environ["TMLM_RUN_DIR"] = saved[3]
    return out.getvalue().splitlines()


def test_main_loop():
    with tempfile.TemporaryDirectory() as d:
        log = run_main(["--n-train", "3000", "--batch", "1000", "--clauses", "6", "--A", "4", "--passes", "6", "--eval-every", "5",
                        "--early-stop-margin", "0.05", "--save-model", "--out", d], good=True)
        kinds = [(l.split(" ", 1)[0], json.loads(l.split(" ", 1)[1]).get("pass")) for l in log if l.split(" ", 1)[0] in ("PASS", "EPOCH")]
        assert kinds == [("PASS", 1)] + [("EPOCH", p) for p in range(2, 7)], kinds  # pass 2 = 30% check, 2-6 = last 5
        fin = json.loads(next(l for l in log if l.startswith("FINAL")).split(" ", 1)[1])
        assert fin["stopped"] == "" and fin["pass"] == 6 and fin["dev_acc"] > 0.6 and abs(fin["last5_mean_dev_acc"] - np.mean(
            [json.loads(l.split(" ", 1)[1])["dev_acc"] for l in log if l.startswith("EPOCH")])) < 1e-12
        assert set(os.listdir(d)) >= {"metrics.json", "rules.txt", "groups.json", "include_bits.npz", "model_final.pkl",
                                      "dataset_hashes.json", "text_driver.py", "htm_driver.py"}
        gj = json.load(open(os.path.join(d, "groups.json")))
        assert len(gj["classes"]) == 4 and all(len(c["clauses"]) == 3 for c in gj["classes"]) and gj["model"] == "hier"
        assert "ALWAYS-TRUE" in open(os.path.join(d, "rules.txt")).read()
        RESULTS["smoke_good_dev_acc"] = fin["dev_acc"]
    with tempfile.TemporaryDirectory() as d:
        log = run_main(["--n-train", "3000", "--clauses", "6", "--A", "4", "--passes", "10", "--eval-every", "5",
                        "--early-stop-margin", "0.05", "--out", d], good=False)
        fin = json.loads(next(l for l in log if l.startswith("FINAL")).split(" ", 1)[1])
        assert fin["stopped"].startswith("stopped early by rule at pass 3") and fin["last5_mean_dev_acc"] is None, fin
        assert fin["dev_pred_share"][0] == 1.0 and fin["dev_flag_constant"] and fin["dev_tie_share"] == 1.0
    with tempfile.TemporaryDirectory() as d:  # flat: one option per clause, A = K = 1
        log = run_main(["--model", "flat", "--n-train", "3000", "--clauses", "20", "--passes", "2", "--out", d], good=True)
        assert json.load(open(os.path.join(d, "groups.json")))["A"] == 1 and any(l.startswith("FINAL") for l in log)


def test_npmi_toy():
    RESULTS["npmi_toy"] = ng.toy_check()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    RESULTS["tests_passed"] = len(tests)
    print("RESULTS", json.dumps(RESULTS))
    if os.environ.get("TMLM_RUN_DIR"):
        json.dump({k: {"value": v, "label": "measured"} for k, v in RESULTS.items()},
                  open(os.path.join(os.environ["TMLM_RUN_DIR"], "metrics.json"), "w"), indent=1)
