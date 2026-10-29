"""Gate 0 cause check of M5 step 1: does the missing Type Ib erosion of the numba kernel explain the failed Gate 0?

Dev plan: docs/plan_gate0_cause_2026-10-25.md (DRAFT until frozen with scripts/freeze.py). Validation split of
CLINC150 only: load_train_val reads four named arrays of the feature file and records them (np.load of an .npz is lazy,
so no test array is loaded; the file itself is hashed whole, test arrays included).

The reading under test (docs/results.md, "Diagnosis notes of 2026-10-02"): for the sampled negative class with a
clause weight <= 0, the numba kernel of Clause-Driven-LLM gives Type I feedback only to clauses that fired
(coalesced.py:173-188, no else branch), while the torch kernel also erodes the clauses that did not fire
(torch_backend.py:130-131 g1nf, 154 cnt_dec). The variants change exactly that and nothing else:

  torch        the unmodified torch backend (arm A0, reference)
  torch_noneg  torch with the negative-class part of g1nf removed: g1nf = (sel_pos & (Wt >= 0)) & (~co) (arm A)
  numba        the unmodified numba backend (the kernel of the failed Gate 0 run; not in the plan)
  numba_erode  numba with an else branch after the negative-class Type I block that erodes every literal with
               probability 1/s, a copy of the target-class block of coalesced.py:141-145 (arm B)

The upstream files are imported read-only from work/Clause-Driven-LLM/src. Pins and their provenance: coalesced.py
must have the SHA-256 that the configs of all 15 M5 training runs record (key coalesced_sha256, bfffb131...);
torch_backend.py must equal the reference copy (4828a9b5...), but no run ever hashed it: the setup run
20261002T210555Z_m5_setup_torchcpu recorded only torch_backend_identical_to_reference: true, and arm A0 is the
empirical check of the torch path. The whole cdllm tree (every .py file under src/cdllm, binarize.py and seeding.py
included) must match one manifest digest pinned here. A variant is built from the source text of the upstream function
with one exact replacement or insertion, compiled in a copy of the module's namespace, and put in place of the module
attribute only while fit() runs; a counting wrapper records how often the attribute was called (for the unmodified
variants it wraps the original), so a run shows that its variant was used. Bytecode writing is off, and numba's cache
goes to work/gate0_cause/numba_cache unless NUMBA_CACHE_DIR is set, so nothing is written into the upstream copy.

Training settings are the locked settings of M5 step 1 (lexical features, 2 bins, 4,000 clauses, T 4000, s 8,
256 states, 40 epochs, batch 128, hardest negative, seed 42), through the same calls as the patched
scripts/train_controller.py: seed_all(seed), Booleanizer(n_bins=2) fitted on train, fit(..., X_val, y_val,
eval_every=2), so the model is the best-validation snapshot and val_acc is its accuracy (the quantity recorded as
val_accuracy by the Gate 0 runs: numba 0.5493, torch seed 42 0.8187).

Writes a run folder results/<UTC>_gate0_cause_<variant>_s<S>_seed<SEED>/ (src/runlog.py) with config.json (settings,
upstream hashes and manifest, the variant's diff, the feature-file keys read, the environment's package list and
versions), metrics.json (val_acc, history, literal_stats, the call count of the variant's attribute, train_s) and
tm.npz (the trained ta and weight). src/gate0_cause_report.py applies the plan's rules to these folders.

Usage (from studies/clause_lm):
  .venv-cdllm-gpu/bin/python src/gate0_cause.py --variant torch_noneg --seed 42            # GPU arms A0 and A
  .venv-cdllm/bin/python src/gate0_cause.py --variant numba_erode --seed 42 --device cpu   # CPU arm B
  .venv-cdllm/bin/python src/gate0_cause.py --describe PATH/tm.npz [...]                   # literal statistics only
  <venv>/bin/python src/gate0_cause.py --manifest | --venv-listing                         # print a pinned digest
"""
import argparse, contextlib, difflib, glob, hashlib, inspect, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from runlog import Run, sha256  # noqa: E402

LL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = f"{LL}/work/Clause-Driven-LLM"
REF = f"{LL}/references/code/Clause-Driven-LLM"
EMB = f"{LL}/work/m5/emb_sorted/clinc150__lexical.npz"
EMB_SHA256 = "15a987add2b590d8498768045d465167c808d8cbf8afd483330e89330444d0fc"     # m5_setup_sorted record
UPSTREAM_SHA256 = {
    # recorded as coalesced_sha256 in the config of all 15 M5 training runs (2026-10-02 and 2026-10-14)
    "coalesced.py": "bfffb131874cbc0b6edff7e1152d25ae612df69a24c66cf66b26ecd46ca3ef2d",
    # the reference copy's hash; no run hashed this file (only torch_backend_identical_to_reference: true, 2026-10-02)
    "torch_backend.py": "4828a9b515908793dbd5655d7eb3eed8b5ad306ab82cc2e96917db1669c69215"}
# cdllm_manifest() of work/Clause-Driven-LLM (22 files), computed 2026-10-25 with --manifest
CDLLM_MANIFEST_SHA256 = "5c648da62b8b760ac4a0edfd3adeb11e555993c3f73a14f78b65d560ee094dab"
VARIANTS = ("torch", "torch_noneg", "numba", "numba_erode")
TRAIN_VAL_KEYS = ("Xtrain", "ytrain", "Xval", "yval")
LOCKED = {"n_bins": 2, "n_clauses": 4000, "T": 4000, "s": 8.0, "n_states": 256, "epochs": 40, "batch_size": 128,
          "neg_sampling": "hardest", "eval_every": 2}

TORCH_LINE = "g1nf = ((sel_pos & (Wt >= 0)) | (sel_neg & (Wn < 0))) & (~co)  # Type I, not fired"
TORCH_NEW = "g1nf = (sel_pos & (Wt >= 0)) & (~co)  # Type I, not fired: target class only (gate0_cause torch_noneg)"
NUMBA_ANCHOR = "                        weight[j, yn] = wj - 1  # more negative\n"
NUMBA_EROSION = ("                    else:\n"
                 "                        for k in range(m):\n"
                 "                            st, r = _u01(st)\n"
                 "                            if r <= s_inv and ta[j, k] > 1:\n"
                 "                                ta[j, k] -= 1\n")


def use_upstream(work=WORK):
    """Puts the upstream src folder on sys.path (read-only use) and keeps numba's cache out of it."""
    os.environ.setdefault("NUMBA_CACHE_DIR", f"{LL}/work/gate0_cause/numba_cache")
    if f"{work}/src" not in sys.path:
        sys.path.insert(0, f"{work}/src")


def upstream_hashes(work=WORK, ref=REF):
    out = {}
    for name in UPSTREAM_SHA256:
        out[name] = {"work": sha256(f"{work}/src/cdllm/tm/{name}"),
                     "reference": sha256(f"{ref}/src/cdllm/tm/{name}") if os.path.exists(f"{ref}/src/cdllm/tm/{name}")
                     else None}
    return out


def cdllm_manifest(work=WORK):
    """([path, sha256] of every .py file under src/cdllm, paths relative to src/, sorted; digest of the lines)."""
    root = os.path.join(work, "src")
    rows = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "cdllm")):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        rows += [[os.path.relpath(os.path.join(dirpath, f), root), sha256(os.path.join(dirpath, f))]
                 for f in filenames if f.endswith(".py")]
    rows.sort()
    return rows, hashlib.sha256("".join(f"{p} {h}\n" for p, h in rows).encode()).hexdigest()


def venv_listing(prefix=sys.prefix):
    """(sorted name==version of every .dist-info folder of the environment, digest of the lines)."""
    pkgs = []
    for sp in sorted(glob.glob(os.path.join(prefix, "lib", "python*", "site-packages"))):
        for d in os.listdir(sp):
            if d.endswith(".dist-info"):
                name, _, version = d[:-len(".dist-info")].rpartition("-")
                pkgs.append(f"{name}=={version}")
    pkgs.sort()
    return pkgs, hashlib.sha256("\n".join(pkgs).encode()).hexdigest()


def versions(names=("numpy", "numba", "scipy", "scikit-learn", "torch")):
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def diff(old, new, name):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"))


def torch_noneg_source(torch_backend):
    """(original source, variant source) of torch_backend.train."""
    old = inspect.getsource(torch_backend.train)
    if old.count(TORCH_LINE) != 1:
        raise RuntimeError("torch_backend.train does not hold the expected g1nf line exactly once")
    return old, old.replace(TORCH_LINE, TORCH_NEW)


def numba_erode_source(coalesced):
    """(original source, variant source) of coalesced._train_minibatch_impl."""
    old = inspect.getsource(coalesced._train_minibatch_impl)
    if old.count(NUMBA_EROSION) != 1 or old.count(NUMBA_ANCHOR) != 1:
        raise RuntimeError("_train_minibatch_impl does not hold the target-class erosion block and the "
                           "negative-class anchor exactly once")
    return old, old.replace(NUMBA_ANCHOR, NUMBA_ANCHOR + NUMBA_EROSION)


def build_variant(variant):
    """Returns (module, attribute, replacement or None for the unmodified function, diff text) for the variant."""
    from cdllm.tm import coalesced, torch_backend
    if variant == "torch":
        return torch_backend, "train", None, ""
    if variant == "numba":
        return coalesced, "_train_minibatch", None, ""
    if variant == "torch_noneg":
        old, new = torch_noneg_source(torch_backend)
        ns = dict(vars(torch_backend))
        exec(compile("from __future__ import annotations\n" + new, "<gate0_cause torch_noneg>", "exec"), ns)
        return torch_backend, "train", ns["train"], diff(old, new, "torch_backend.train")
    if variant == "numba_erode":
        old, new = numba_erode_source(coalesced)
        ns = dict(vars(coalesced))
        exec(compile(new, "<gate0_cause numba_erode>", "exec"), ns)
        impl = ns["_train_minibatch_impl"]
        kernel = coalesced.njit(parallel=True)(impl) if coalesced._HAVE_NUMBA else impl
        return coalesced, "_train_minibatch", kernel, diff(old, new, "coalesced._train_minibatch_impl")
    raise ValueError(variant)


@contextlib.contextmanager
def patched(module, attr, value):
    """Puts a counting wrapper of value (or of the module's own function when value is None) in place of module.attr
    while the block runs, and yields the record {attribute, replaced, calls}."""
    old = getattr(module, attr)
    target = old if value is None else value
    rec = {"attribute": f"{module.__name__}.{attr}", "replaced": value is not None, "calls": 0}

    def counted(*a, **k):
        rec["calls"] += 1
        return target(*a, **k)
    setattr(module, attr, counted)
    try:
        yield rec
    finally:
        setattr(module, attr, old)


def load_train_val(path):
    """The four training and validation arrays of the lexical feature file, and the keys read. The file also holds
    test arrays; they are never accessed (np.load of an .npz reads a member only when it is indexed)."""
    keys, out = [], {}
    with np.load(path, allow_pickle=False) as z:
        for k in TRAIN_VAL_KEYS:
            out[k] = z[k]
            keys.append(k)
    return out, keys


def literal_stats(ta, weight, n_states):
    """Clause literal statistics of a trained CoalescedTsetlinMachine (include iff ta > n_states; literals k < f are
    positive, k >= f negated), the signature that separates the numba and torch Gate 0 models."""
    inc = np.asarray(ta) > n_states
    f = inc.shape[1] // 2
    pos, neg = inc[:, :f].sum(1), inc[:, f:].sum(1)
    tot = pos + neg
    w = np.asarray(weight)
    return {"n_clauses": int(len(inc)), "clauses_with_positive_literal": int((pos > 0).sum()),
            "clauses_with_negated_literal": int((neg > 0).sum()), "non_empty_clauses": int((tot > 0).sum()),
            "literals_median": float(np.median(tot)), "literals_mean": float(tot.mean()), "literals_max": int(tot.max()),
            "positive_median": float(np.median(pos)), "positive_max": int(pos.max()),
            "negated_median": float(np.median(neg)), "negated_max": int(neg.max()),
            "weight_negative_share": float((w < 0).mean()), "weight_zero_share": float((w == 0).mean())}


def describe(paths):
    for p in paths:
        z = np.load(p)
        print(json.dumps({"file": p, "sha256": sha256(p), **literal_stats(z["ta"], z["weight"], int(z["cfg_n_states"]))}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", help="torch device (torch variants); cpu for tests")
    ap.add_argument("--emb", default=EMB, help="lexical feature file (tests give a toy file)")
    ap.add_argument("--work", default=WORK, help="upstream work copy (tests only)")
    ap.add_argument("--clauses", type=int, default=LOCKED["n_clauses"])
    ap.add_argument("--T", type=int, default=LOCKED["T"])
    ap.add_argument("--s", type=float, default=LOCKED["s"])
    ap.add_argument("--epochs", type=int, default=LOCKED["epochs"])
    ap.add_argument("--describe", nargs="+", metavar="TM_NPZ", help="print literal statistics of saved models only")
    ap.add_argument("--manifest", action="store_true", help="print the cdllm manifest of --work and its digest")
    ap.add_argument("--venv-listing", action="store_true", help="print this environment's package list and digest")
    a = ap.parse_args(argv)
    if a.describe:
        describe(a.describe)
        return None
    if a.manifest:
        rows, digest = cdllm_manifest(a.work)
        print("\n".join(f"{h}  {p}" for p, h in rows) + f"\nmanifest {digest}")
        return digest
    if a.venv_listing:
        pkgs, digest = venv_listing()
        print("\n".join(pkgs) + f"\nprefix {sys.prefix}\nlisting {digest}")
        return digest
    if not a.variant:
        ap.error("--variant is required")
    use_upstream(a.work)
    hashes = upstream_hashes(a.work)
    for name, h in hashes.items():
        if h["work"] != UPSTREAM_SHA256[name]:
            sys.exit(f"REFUSED: {name} of the work copy has SHA-256 {h['work']}, not the pinned {UPSTREAM_SHA256[name]} "
                     "(coalesced.py: the coalesced_sha256 of the 15 M5 training runs; torch_backend.py: the reference "
                     "copy, which no run hashed)")
    manifest, manifest_sha = cdllm_manifest(a.work)
    if manifest_sha != CDLLM_MANIFEST_SHA256:
        sys.exit(f"REFUSED: the cdllm tree of the work copy has manifest digest {manifest_sha}, not the pinned "
                 f"{CDLLM_MANIFEST_SHA256} (print it with --manifest)")
    from cdllm.features.binarize import Booleanizer
    from cdllm.tm.coalesced import CoalescedTsetlinMachine
    from cdllm.utils.seeding import seed_all
    module, attr, value, diff_text = build_variant(a.variant)
    settings = {**LOCKED, "n_clauses": a.clauses, "T": a.T, "s": a.s, "epochs": a.epochs}
    locked = all(settings[k] == LOCKED[k] for k in LOCKED)
    backend = "torch" if a.variant.startswith("torch") else "numba"
    pkgs, listing_sha = venv_listing()
    emb_sha = sha256(a.emb)
    config = {"plan": "docs/plan_gate0_cause_2026-10-25.md", "variant": a.variant, "backend": backend,
              "device": a.device if backend == "torch" else "cpu", "seed": a.seed, **settings,
              "locked_settings_of_m5_step1": locked, "emb": a.emb, "emb_sha256": emb_sha,
              "emb_is_registered_file": emb_sha == EMB_SHA256, "upstream_sha256": hashes,
              "cdllm_manifest_sha256": manifest_sha, "cdllm_manifest": manifest,
              "variant_diff": diff_text, "variant_diff_sha256": hashlib.sha256(diff_text.encode()).hexdigest(),
              "split": "train and validation arrays only; the feature file is hashed whole, no test array is loaded",
              "venv": {"prefix": sys.prefix, "listing_sha256": listing_sha, "packages": pkgs}, "versions": versions(),
              "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
              "threads": {k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "nice": os.nice(0)}
    if backend == "torch":
        import torch
        config.update({"torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda})
    d, keys = load_train_val(a.emb)
    config["emb_keys_read"] = keys
    run = Run(f"gate0_cause_{a.variant}_s{a.s:g}_seed{a.seed}", config, data_files=[a.emb], script=__file__)
    if diff_text:
        run.print("variant diff:\n" + diff_text)
    seed_all(a.seed)
    boolz = Booleanizer(n_bins=settings["n_bins"])
    Btr = boolz.fit_transform(d["Xtrain"])
    Bva = boolz.transform(d["Xval"])
    ytr, yva = d["ytrain"], d["yval"]
    tm = CoalescedTsetlinMachine(n_clauses=settings["n_clauses"], T=settings["T"], s=settings["s"],
                                 n_states=settings["n_states"], epochs=settings["epochs"], seed=a.seed,
                                 batch_size=settings["batch_size"], neg_sampling=settings["neg_sampling"])
    t0 = time.time()
    with patched(module, attr, value) as calls:
        tm.fit(Btr, ytr, X_val=Bva, y_val=yva, eval_every=settings["eval_every"], backend=backend, device=a.device)
    train_s = time.time() - t0
    val_acc = float((tm.predict(Bva) == np.asarray(yva)).mean())
    stats = literal_stats(tm.ta, tm.weight, settings["n_states"])
    tm.save(os.path.join(run.dir, "tm.npz"))
    hist = getattr(tm, "history_", [])
    best = max(hist, key=lambda h: h["val_acc"]) if hist else None
    run.print("literal_stats", json.dumps(stats))
    run.print("variant attribute calls", json.dumps(calls))
    return run.finish({"variant": a.variant, "val_acc": val_acc, "best_val_acc": float(getattr(tm, "best_val_acc_",
                       float("nan"))), "best_epoch": best["epoch"] if best else None,
                       "val_history": [round(h["val_acc"], 4) for h in hist], "literal_stats": stats,
                       "variant_calls": calls, "train_s": round(train_s, 1)})


if __name__ == "__main__":
    main()
