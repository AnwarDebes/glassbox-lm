"""src/gate0_cause.py on the CPU: the two kernel variants, the literal statistics and the validation-only data path.

    NUMBA_CACHE_DIR=$TMPDIR/clm_numba_cache CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
        nice -n 19 .venv-cdllm-gpu/bin/python tests/test_gate0_cause.py

Run from studies/clause_lm with .venv-cdllm-gpu (torch and numba; torch runs on the CPU because no GPU is visible).
About a minute, most of it numba compiling. Needs the local upstream work copy (work/Clause-Driven-LLM), which it
imports read-only; parts D and E also read the two saved Gate 0 models and the lexical feature file when present.
Run folders go to a temporary folder; numba's cache goes to NUMBA_CACHE_DIR (set here if missing).

Asserts:
A. the torch variant changes one line (the g1nf line); the numba variant adds five lines after the negative-class
   Type I block, a copy of the target-class erosion block, and changes nothing else;
B. torch kernel on a toy problem where no clause fires: with every weight negative, the unmodified train() erodes the
   automata (the negative-class part of g1nf) and the variant leaves them unchanged; with every weight positive, both
   give identical states (the variant touches only the negative-class part);
C. numba kernel on a toy minibatch where no clause fires: with negative weights and negative-class feedback only, the
   unmodified kernel leaves the automata unchanged and the variant erodes every literal once per sample (s = 1);
   with positive weights (Type II side) or target-class feedback only, both give identical states;
D. literal_stats() on the saved Gate 0 models (when present, SHA-256 checked) gives the recorded clause statistics
   (median, mean and maximum literals, non-empty clauses: docs/results.md) and the positive-literal counts 0 and 2,954
   of 4,000;
E. load_train_val() reads only Xtrain, ytrain, Xval and yval of the real lexical feature file (when present);
F. main() end to end on a toy feature file whose test arrays would raise if read, for all four variants with tiny
   settings: run folders with val_acc, literal_stats, the variant diff and its SHA-256, the keys read, the package
   listing of the environment, the versions, and the call count of the variant's attribute (above 0; replaced only
   for torch_noneg and numba_erode); the driver refuses (no run folder) when the cdllm manifest or a kernel file is
   not the pinned one;
G. the pinned literals of src/gate0_cause_report.py equal what they pin: the SHA-256 of src/gate0_cause.py and
   src/runlog.py, the kernel files, the cdllm manifest, the feature file, the package listings of .venv-cdllm-gpu and
   .venv-cdllm, and the SHA-256 of each variant's diff;
H. the upstream kernel files and the upstream __pycache__ folders are unchanged afterwards.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "clm_numba_cache"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import toyutil  # noqa: E402

check = toyutil.Checks()
STUDY = toyutil.STUDY
WORK = STUDY / "work" / "Clause-Driven-LLM"
SAVED = {"numba": (STUDY / "work/m5/runs/sorted/s8_seed42/clinc150__lexical__seed42/tm.npz",
                   "f1183ee36a2a8568564ca4ce60d3dbd48f99fb71dadfc2d63a53027be495191e",
                   dict(clauses_with_positive_literal=0, clauses_with_negated_literal=4000, non_empty_clauses=4000,
                        literals_median=1021.0, literals_max=1411)),
         "torch": (STUDY / "work/m5/runs/sorted_torchgpu/s8_seed42/clinc150__lexical__seed42/tm.npz",
                   "5e2478a4c0a3a984773901d8c0ee371bf4fa74d1b103f20dbccaf9235608d75e",
                   dict(clauses_with_positive_literal=2954, clauses_with_negated_literal=1165, non_empty_clauses=3908,
                        literals_median=1.0, literals_max=834))}
MEANS = {"numba": 1021.9, "torch": 147.8}          # docs/results.md, rounded to one decimal there


def upstream_state():
    files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((WORK / "src" / "cdllm" / "tm").glob("*.py"))}
    caches = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(WORK.rglob("__pycache__/*"))}
    return files, caches


def part_a(g):
    from cdllm.tm import coalesced, torch_backend
    old, new = g.torch_noneg_source(torch_backend)
    d = g.diff(old, new, "train").splitlines()
    minus = [l for l in d if l.startswith("-") and not l.startswith("---")]
    plus = [l for l in d if l.startswith("+") and not l.startswith("+++")]
    check(len(minus) == 1 and len(plus) == 1 and "sel_neg" in minus[0] and "sel_neg" not in plus[0]
          and "(sel_pos & (Wt >= 0)) & (~co)" in plus[0], "torch variant: only the g1nf line changes")
    old, new = g.numba_erode_source(coalesced)
    d = g.diff(old, new, "kernel").splitlines()
    minus = [l for l in d if l.startswith("-") and not l.startswith("---")]
    plus = [l[1:] + "\n" for l in d if l.startswith("+") and not l.startswith("+++")]
    check(minus == [] and "".join(plus) == g.NUMBA_EROSION and new.count(g.NUMBA_EROSION) == 2
          and new.index(g.NUMBA_ANCHOR) + len(g.NUMBA_ANCHOR) == new.index(g.NUMBA_EROSION, new.index(g.NUMBA_ANCHOR)),
          "numba variant: 5 added lines, a copy of the target-class erosion, right after the negative-class block")


def part_b(g):
    import torch
    from cdllm.tm import torch_backend
    torch.set_num_threads(2)
    _, attr, train_noneg, _ = g.build_variant("torch_noneg")
    n_states, f, C, n = 16, 5, 6, 8
    X = np.random.default_rng(0).integers(0, 2, size=(n, f)).astype(np.uint8)
    X[:, 0] = 0
    y = np.arange(n) % 3
    ta = np.full((C, 2 * f), n_states, np.int32)
    ta[:, 0] = n_states + 1                             # every clause includes x0, which is 0 everywhere: none fires
    kw = dict(n_states=n_states, T=10, s=1.0, epochs=1, batch_size=n, seed=3, neg_sampling="hardest", device="cpu")
    for sign, what in ((-1, "negative"), (1, "positive")):
        w = np.full((C, 3), sign, np.int32)
        ta0, w0, _, _ = torch_backend.train(ta.copy(), w.copy(), X, y, **kw)
        ta1, w1, _, _ = train_noneg(ta.copy(), w.copy(), X, y, **kw)
        if sign < 0:
            check(ta0.sum() < ta.sum() and np.array_equal(ta1, ta) and np.array_equal(w1, w),
                  f"torch, no clause fires, {what} weights: unmodified erodes ({ta.sum()} -> {ta0.sum()}), "
                  f"variant leaves the state ({ta1.sum()})")
        else:
            check(np.array_equal(ta0, ta1) and np.array_equal(w0, w1) and ta0.sum() < ta.sum(),
                  f"torch, no clause fires, {what} weights: both erode identically ({ta.sum()} -> {ta0.sum()})")


def part_c(g):
    from cdllm.tm import coalesced
    _, _, kernel_erode, _ = g.build_variant("numba_erode")
    n_states, f, C, n = 16, 5, 6, 8
    X = np.random.default_rng(1).integers(0, 2, size=(n, f)).astype(np.uint8)
    X[:, 0] = 0
    ta = np.full((C, 2 * f), n_states, np.int32)
    ta[:, 0] = n_states + 1
    co = np.zeros((n, C), np.uint8)                       # consistent with ta and X: no clause fires
    yt, yn = np.zeros(n, np.int64), np.ones(n, np.int64)
    rng0 = np.random.default_rng(2).integers(1, 2**63 - 1, size=C, dtype=np.uint64)

    def run(kernel, w, p_pos, p_neg):
        t, ww, st = ta.copy(), w.copy(), rng0.copy()
        kernel(t, ww, X, co, np.full(n, p_pos), np.full(n, p_neg), yt, yn, n_states, 1.0, st)
        return t, ww

    w_neg = np.full((C, 2), -1, np.int32)
    t0, _ = run(coalesced._train_minibatch, w_neg, 0.0, 1.0)
    t1, _ = run(kernel_erode, w_neg, 0.0, 1.0)
    check(np.array_equal(t0, ta) and np.array_equal(t1, np.maximum(ta - n, 1)),
          f"numba, no clause fires, negative weights, negative-class feedback: unmodified leaves the state "
          f"({ta.sum()} -> {t0.sum()}), variant erodes once per sample ({t1.sum()})")
    w_pos = np.full((C, 2), 1, np.int32)
    t0, w0 = run(coalesced._train_minibatch, w_pos, 0.0, 1.0)
    t1, w1 = run(kernel_erode, w_pos, 0.0, 1.0)
    check(np.array_equal(t0, t1) and np.array_equal(w0, w1), "numba, positive weights (Type II side): identical")
    t0, w0 = run(coalesced._train_minibatch, w_pos, 1.0, 0.0)
    t1, w1 = run(kernel_erode, w_pos, 1.0, 0.0)
    check(np.array_equal(t0, t1) and np.array_equal(w0, w1) and t0.sum() < ta.sum(),
          "numba, target-class feedback only: identical, both erode")


def part_d(g):
    for name, (path, sha, want) in SAVED.items():
        if not path.exists():
            print(f"skip: saved {name} model not present ({path})")
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        z = np.load(path)
        st = g.literal_stats(z["ta"], z["weight"], int(z["cfg_n_states"]))
        ok = h == sha and all(st[k] == v for k, v in want.items()) and round(st["literals_mean"], 1) == MEANS[name]
        check(ok, f"saved {name} model: {st['clauses_with_positive_literal']} of {st['n_clauses']} clauses with a "
                  f"positive literal, median {st['literals_median']:g}, mean {st['literals_mean']:.2f}, max "
                  f"{st['literals_max']}, non-empty {st['non_empty_clauses']}, negative weights "
                  f"{st['weight_negative_share']:.4f}, zero {st['weight_zero_share']:.4f}")


class Recorder:
    """np.load stand-in that records which members of an .npz are indexed."""

    def __init__(self, real, seen):
        self.real, self.seen = real, seen

    def __call__(self, *a, **k):
        z = self.real(*a, **k)
        seen = self.seen

        class Proxy:
            def __enter__(self):
                return self

            def __exit__(self, *e):
                z.close()

            def __getitem__(self, key):
                seen.append(key)
                return z[key]
        return Proxy()


def part_e(g):
    if not Path(g.EMB).exists():
        print(f"skip: no lexical feature file at {g.EMB}")
        return
    seen = []
    real = g.np.load
    g.np.load = Recorder(real, seen)
    try:
        d, keys = g.load_train_val(g.EMB)
    finally:
        g.np.load = real
    check(seen == list(g.TRAIN_VAL_KEYS) and keys == list(g.TRAIN_VAL_KEYS) and d["Xval"].shape == (3000, 2000)
          and g.sha256(g.EMB) == g.EMB_SHA256, f"real feature file: only {seen} are read; Xval {d['Xval'].shape}")


def part_f(g, root):
    rng = np.random.default_rng(5)
    n_cls, f = 4, 24
    proto = rng.random((n_cls, f)) < 0.25

    def make(n):
        y = rng.integers(0, n_cls, n)
        X = (proto[y] ^ (rng.random((n, f)) < 0.05)).astype(np.float32)
        return X, y
    Xtr, ytr = make(300)
    Xva, yva = make(80)
    emb = root / "toy__lexical.npz"
    np.savez(emb, Xtrain=Xtr, ytrain=ytr, Xval=Xva, yval=yva, Xtest=np.full((5, f), np.nan), ytest=np.full(5, -1),
             Xoos_test=np.full((5, f), np.nan))
    rep = __import__("gate0_cause_report")
    for variant in ("torch", "torch_noneg", "numba", "numba_erode"):
        seen = []
        real = g.np.load
        g.np.load = Recorder(real, seen)
        try:
            out = g.main(["--variant", variant, "--device", "cpu", "--emb", str(emb), "--clauses", "40", "--T", "40",
                          "--epochs", "4", "--seed", "1"])
        finally:
            g.np.load = real
        m = json.loads((Path(out) / "metrics.json").read_text())
        c = json.loads((Path(out) / "config.json").read_text())
        patched_variant = variant in ("torch_noneg", "numba_erode")
        calls = m["variant_calls"]
        ok = (0 <= m["val_acc"] <= 1 and m["literal_stats"]["n_clauses"] == 40 and seen == list(g.TRAIN_VAL_KEYS)
              and c["emb_keys_read"] == list(g.TRAIN_VAL_KEYS) and c["locked_settings_of_m5_step1"] is False
              and (c["variant_diff"] != "") == patched_variant and (Path(out) / "tm.npz").exists()
              and c["variant_diff_sha256"] == rep.VARIANT_DIFF_SHA256[variant]
              and c["cdllm_manifest_sha256"] == rep.PIN["cdllm_manifest"]
              and c["venv"]["listing_sha256"] == rep.PIN["venv_torch"] and c["versions"]["torch"] is not None
              and c["versions"]["numba"] is not None and isinstance(c["nice"], int))
        check(ok, f"main() {variant}: val_acc {m['val_acc']:.3f}, {m['literal_stats']['clauses_with_positive_literal']}"
                  f" of 40 clauses with a positive literal, keys read {seen}")
        check(calls["calls"] > 0 and calls["replaced"] == patched_variant,
              f"main() {variant}: the variant's attribute was called ({calls})")
    for what, attr, key in (("the cdllm manifest", "CDLLM_MANIFEST_SHA256", None),
                            ("coalesced.py", "UPSTREAM_SHA256", "coalesced.py"),
                            ("torch_backend.py", "UPSTREAM_SHA256", "torch_backend.py")):
        saved = getattr(g, attr)
        setattr(g, attr, "0" * 64 if key is None else {**saved, key: "0" * 64})
        before = set((root / "results").glob("*"))
        try:
            g.main(["--variant", "torch", "--device", "cpu", "--emb", str(emb), "--clauses", "40", "--T", "40",
                    "--epochs", "1", "--seed", "2"])
            msg = "no refusal"
        except SystemExit as e:
            msg = str(e)
        finally:
            setattr(g, attr, saved)
        check(msg.startswith("REFUSED") and (key or "cdllm tree") in msg and set((root / "results").glob("*")) == before,
              f"the driver refuses when {what} is not the pinned one, and writes no run folder")


def part_g(g):
    import gate0_cause_report as rep
    from cdllm.tm import coalesced, torch_backend  # noqa: F401
    for v in g.VARIANTS:
        d = g.build_variant(v)[3]
        check(hashlib.sha256(d.encode()).hexdigest() == rep.VARIANT_DIFF_SHA256[v], f"pinned diff of {v} is its diff")
    check(rep.PIN["gate0_cause.py"] == hashlib.sha256((toyutil.SRC / "gate0_cause.py").read_bytes()).hexdigest(),
          "pinned SHA-256 of src/gate0_cause.py is the file's")
    check(rep.PIN["runlog.py"] == hashlib.sha256((toyutil.SRC / "runlog.py").read_bytes()).hexdigest(),
          "pinned SHA-256 of src/runlog.py is the file's")
    check(rep.PIN["coalesced.py"] == g.UPSTREAM_SHA256["coalesced.py"]
          and rep.PIN["torch_backend.py"] == g.UPSTREAM_SHA256["torch_backend.py"]
          and all(g.upstream_hashes()[n]["work"] == rep.PIN[n] for n in ("coalesced.py", "torch_backend.py")),
          "pinned kernel files are the driver's pins and the work copy's files")
    check(rep.PIN["cdllm_manifest"] == g.CDLLM_MANIFEST_SHA256 == g.cdllm_manifest()[1],
          f"pinned cdllm manifest is the work copy's ({g.cdllm_manifest()[1][:16]})")
    check(rep.PIN["emb"] == g.EMB_SHA256, "pinned feature file is the driver's")
    for key, venv in (("venv_torch", ".venv-cdllm-gpu"), ("venv_numba", ".venv-cdllm")):
        got = g.venv_listing(str(STUDY / venv))[1]
        check(rep.PIN[key] == got, f"pinned package listing of {venv} is the environment's ({got[:16]})")
    check(rep.A0_TARGET == 0.8186666666666667, "A0 target is the recorded validation accuracy")


def main():
    if not (WORK / "src" / "cdllm").exists():
        print(f"skip: no upstream work copy at {WORK}")
        check.done()
    before = upstream_state()
    with tempfile.TemporaryDirectory(prefix="clm_gate0_") as tmp:
        root = Path(tmp)
        toyutil.use_src(root)
        import gate0_cause as g
        g.use_upstream()
        part_a(g)
        part_b(g)
        part_c(g)
        part_d(g)
        part_e(g)
        part_f(g, root)
        part_g(g)
    after = upstream_state()
    check(before[0] == after[0] and all(h in g.UPSTREAM_SHA256.values() for p, h in after[0].items()
                                         if p.name in g.UPSTREAM_SHA256),
          "upstream kernel files unchanged and equal to the recorded SHA-256")
    check(before[1] == after[1], f"upstream __pycache__ folders unchanged ({len(after[1])} files)")
    check.done()


if __name__ == "__main__":
    main()
