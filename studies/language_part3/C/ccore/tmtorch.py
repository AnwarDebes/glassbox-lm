"""Step 9 of C-core, the CoTM side (torch; .venv-dl). Written under the step-9 amendment.

Our CoTM (glassbox-lm/studies/decision_layer/src/dl/tm.py, workspace code, not a CAIR library) is loaded from its
file after a full SHA-256 check of the very bytes that are compiled (no import of the dl package, so nothing is
written next to it; dl/cpu_tm_fast.py the same way). Neither file is ever edited.

Its own training loop (CoTM.fit() is not used: it calls torch.cuda.synchronize() and keeps the best epoch by accuracy),
as G1 and A-Struct run it: every epoch a permutation from the machine's own generator, minibatches of 128 through
CoTM._step (the supervised rule; the negative class drawn by neg_sampling "random"). After every epoch: the class sums
on calib (inference rule: an empty clause is silent), the read-out fitted on calib (tmcore.fit_readout), its calib
class log-loss; the state with the lowest one is kept (ties to the earlier epoch), and the state after epoch 1. No
valid number is computed here: valid is scored later, only at the kept checkpoints."""
import hashlib
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import paths, tmcore

DL_SRC = paths.DL / "src" / "dl"
TM_FILE = DL_SRC / "tm.py"
TM_SHA256 = "f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424"
FAST_FILE = DL_SRC / "cpu_tm_fast.py"
FAST_SHA256 = "bf1c4a9086d76448d9d77cf7775e0bb0e5f4bffc926d777b867f89148279423e"
N_STATES = 256
DEFAULTS = {"n_states": 256, "negation": True, "max_literals": 0, "budget_mode": "exact", "max_negated": 0,
            "budget_select": "threshold", "clause_drop": 0.0, "half": True}
EXACT_MAX = 2 ** 24        # float32 class sums are exact integers below this
_MODS = {}


def load_by_path(path, sha, name):
    """Compile and run the file's bytes as module `name` after checking their full SHA-256."""
    if name in _MODS:
        return _MODS[name]
    src = Path(path).read_bytes()
    h = hashlib.sha256(src).hexdigest()
    if h != sha:
        raise SystemExit(f"{path}: sha256 {h}, not the registered {sha}")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    try:
        exec(compile(src, str(path), "exec"), mod.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    _MODS[name] = mod
    return mod


def cotm():
    return load_by_path(TM_FILE, TM_SHA256, "ccore_dl_tm")


def fast():
    return load_by_path(FAST_FILE, FAST_SHA256, "ccore_dl_cpu_tm_fast")


def setup_torch(device):
    """2 threads; the device must be what the mode says (cuda visible for a GPU stage, hidden for a CPU run)."""
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("a GPU stage found no GPU visible to torch")
    if device == "cpu" and torch.cuda.is_available():
        raise SystemExit("a CPU run must see no GPU (CUDA_VISIBLE_DEVICES empty)")
    return torch


def make_config(T, s, seed, clauses, epochs):
    """TMConfig of a registered run: the named fields, every other field at tm.py's default (checked)."""
    mod = cotm()
    cfg = mod.TMConfig(n_clauses=int(clauses), T=int(T), s=float(s), epochs=int(epochs), batch_size=tmcore.BATCH,
                       neg_sampling=tmcore.NEG_SAMPLING, seed=int(seed))
    d = asdict(cfg)
    bad = {k: (d[k], v) for k, v in DEFAULTS.items() if d[k] != v}
    if bad:
        raise SystemExit(f"TMConfig defaults are not the registered ones: {bad}")
    return cfg


def _sync(torch, tm):
    if tm.dev.type == "cuda":
        torch.cuda.synchronize()


def class_sums(tm, X, bs=4096):
    """Class sums (n, 65) as int64 (inference rule; float32 sums of integers, exact below 2**24)."""
    import torch
    out = []
    for i in range(0, X.shape[0], 65536):
        Xt = torch.as_tensor(np.ascontiguousarray(X[i:i + 65536]), dtype=torch.uint8, device=tm.dev)
        S = tm.class_sums(Xt, bs=bs)
        if bool((S.abs() >= EXACT_MAX).any()):
            raise SystemExit("a class sum reached 2**24: float32 sums are no longer exact")
        out.append(S.round().to(torch.int64).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, tmcore.N_CLASSES), np.int64)


def checkpoint(tm, epoch, ro, stats):
    st = tm.state()
    return {"epoch": int(epoch), "ta": st["ta"], "w": st["w"].astype(np.int32), "readout": ro, "stats": stats,
            "calib_class_bits": ro["calib_class_bits"]}


def train(X, y, Xc, yc, T, s, seed, clauses, epochs, device, log=print, tag=""):
    """One run: `epochs` epochs on (X, y), the read-out on (Xc, yc) after every epoch. Returns the kept checkpoints
    (best by calib class log-loss, and epoch 1), the per-epoch history and the times (training synchronised)."""
    import torch
    mod = cotm()
    cfg = make_config(T, s, seed, clauses, epochs)
    t_run = time.time()
    tm = mod.CoTM(tmcore.N_CLASSES, X.shape[1], cfg, device=device)
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.uint8, device=tm.dev)
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long, device=tm.dev)
    n, bs = Xt.shape[0], cfg.batch_size
    hist, best, ep1 = [], None, None
    train_s = readout_s = 0.0
    batches = 0
    for epoch in range(1, epochs + 1):
        _sync(torch, tm)
        t = time.time()
        perm = torch.randperm(n, generator=tm.gen, device=tm.dev)
        for start in range(0, n, bs):
            bidx = perm[start:start + bs]
            tm._step(Xt[bidx], yt[bidx])
            batches += 1
        _sync(torch, tm)
        dt = time.time() - t
        train_s += dt
        t = time.time()
        S = class_sums(tm, Xc)
        ro = tmcore.fit_readout(S, yc)
        dr = time.time() - t
        readout_s += dr
        st = tm.stats()
        hist.append({"epoch": epoch, "train_s": round(dt, 3), "readout_s": round(dr, 3),
                     "calib_class_bits": ro["calib_class_bits"], "tau": ro["tau"], "readout_nit": ro["nit"],
                     "readout_converged": ro["converged"], "stats": st})
        if epoch == 1:
            ep1 = checkpoint(tm, epoch, ro, st)
        if best is None or ro["calib_class_bits"] < best["calib_class_bits"]:
            best = ep1 if epoch == 1 else checkpoint(tm, epoch, ro, st)
        log(f"{tag} epoch {epoch}/{epochs}: train {dt:.1f} s, read-out {dr:.1f} s ({ro['nit']} it), calib class bits "
            f"{ro['calib_class_bits']:.5f}, tau {ro['tau']:.4g}; clauses empty {st['empty']}, median "
            f"{st['median_literals']}, no positive {st['no_positive_literal']}")
    if tmcore.pick_epoch([h["calib_class_bits"] for h in hist]) != best["epoch"]:
        raise SystemExit("kept epoch differs from the epoch rule")
    return {"best": best, "epoch1": ep1, "history": hist, "cfg": asdict(cfg), "n_train": int(n), "batches": batches,
            "train_s": round(train_s, 3), "readout_s": round(readout_s, 3), "run_s": round(time.time() - t_run, 3),
            "device": str(tm.dev)}


def machine(cfg_dict, ta, w, device):
    """A CoTM holding a stored state (for the class sums of a checkpoint)."""
    import torch
    mod = cotm()
    tm = mod.CoTM(tmcore.N_CLASSES, tmcore.N_FEATURES, mod.TMConfig(**cfg_dict), device=device)
    tm.ta = torch.as_tensor(np.asarray(ta, dtype=np.int32), device=tm.dev)
    tm.w = torch.as_tensor(np.asarray(w, dtype=np.int32), device=tm.dev)
    return tm


def checkpoint_sums(cfg_dict, ck, X, device):
    return class_sums(machine(cfg_dict, ck["ta"], ck["w"], device), X)


def packed_equal(ck, X_rows, S_rows):
    """dl/cpu_tm_fast (integer gather, no float, CPU) must give the same class sums: the check that also opens a
    CPU path to the test scoring of step 10."""
    F = fast()
    model = F.pack(ck["ta"], ck["w"], n_states=N_STATES, negation=True)
    Sp = F.class_sums(model, np.asarray(X_rows, dtype=np.uint8))
    return {"rows": int(len(X_rows)), "equal": bool(np.array_equal(Sp, np.asarray(S_rows, dtype=np.int64))),
            "clauses_kept": int(model["n_clauses_kept"])}


def score_checkpoint(cfg_dict, ck, Z, sp, e, device, name):
    """Score one kept checkpoint (the only place where valid class sums are computed): calib and valid class sums,
    the packed CPU check at the contract positions, the read-out of that epoch, the calib class bits recomputed from
    the stored state (must equal training's), then the system {F, f, s} (tmcore.system_score)."""
    S = {s: checkpoint_sums(cfg_dict, ck, Z[f"X_{s}"], device) for s in ("calib", "valid")}
    pk = {s: packed_equal(ck, Z[f"X_{s}"][Z[f"pos_{s}"]], S[s][Z[f"pos_{s}"]]) for s in S}
    if not all(v["equal"] for v in pk.values()):
        raise SystemExit(f"{name}: packed CPU class sums differ from the CoTM's: {pk}")
    PX = {s: tmcore.readout_proba(S[s], ck["readout"]) for s in S}
    chk = tmcore.class_logloss_bits(PX["calib"], Z["y_calib"])
    if abs(chk - ck["calib_class_bits"]) > 1e-12:
        raise SystemExit(f"{name}: calib class bits of the stored state {chk} differ from training's "
                         f"{ck['calib_class_bits']}")
    lp, lamb, info = tmcore.system_score(PX, sp, Z, e, name)
    return {"lp": lp, "lamb": lamb, "info": info, "packed": pk,
            "pc_valid": tmcore.true_class_prob(PX["valid"], Z["y_valid"])}


def time_cpu_epoch(X, y, T, s, seed, clauses, log=print):
    """C-K3's timed CPU epoch: one epoch of a registered setting on the CPU at 2 threads, no read-out, no score."""
    import torch
    mod = cotm()
    cfg = make_config(T, s, seed, clauses, 1)
    tm = mod.CoTM(tmcore.N_CLASSES, X.shape[1], cfg, device="cpu")
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.uint8)
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)
    n, bs = Xt.shape[0], cfg.batch_size
    t = time.time()
    perm = torch.randperm(n, generator=tm.gen)
    batches = 0
    for start in range(0, n, bs):
        bidx = perm[start:start + bs]
        tm._step(Xt[bidx], yt[bidx])
        batches += 1
    dt = time.time() - t
    log(f"timed CPU epoch: {n} positions, {batches} batches, {dt:.1f} s ({dt / batches * 1000:.1f} ms per batch), "
        f"torch threads {torch.get_num_threads()}")
    return {"positions": int(n), "batches": batches, "seconds": round(dt, 3), "s_per_batch": dt / batches,
            "torch_threads": int(torch.get_num_threads()), "cfg": asdict(cfg)}
