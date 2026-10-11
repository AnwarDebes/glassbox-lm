"""R1 CONTROL STUDY (exploratory; config experiments/cvtm/configs/ssl_r1_controls.json): does the trained R1 bank
beat (1) the strongest early stop of its own run (a young ladder) and (2) a twin in which ONLY Tsetlin-automata
literal learning is switched off, and (3) what does training cost on the natural token distribution (Q_pred)?
The reading rule is in the config and was written before any output.

Arms: A3 and A0 of the R1 tournament (src/tmlm/ssl/pretext.py, unchanged) and their twins A3T and A0T: the same data
and EngineConfig with p_ins = 0, p_neg = 0 and s = 2^62. The config's "literal_paths" block lists every docs/11 path
that changes the literals of an existing clause, with the oracle and tm_core.h line references, and how each one is
off in the twin (Type Ia reinforcement of true literals cannot be switched off and stays: it only raises the states
of included literals, which never changes inclusion, the anchor or firing once every downward path is off).

This file only IMPORTS the running R1 code (experiments/ssl_r1_pretext.py as R, tmlm.ssl.pretext as P and the modules
they import); nothing of it is edited. Q_pred imports experiments/cvtm_t1_readout.py and experiments/ckm_posthoc_probe.py
(the H8.1 read-out helpers and reference kernels) unchanged.

Stages (each its own logged run):
  train     (GPU) --arm {A3,A0,A3T,A0T} --seed S: the R1 training loop (4 epochs, one engine per epoch for F2) with
            checkpoint_epoch{1..4}.npz, the registered checkpoint_c0y.npz (ceil(young_fraction x nb) batches) and the
            young ladder checkpoint_young_b{N}.npz after absolute batch counts {77, 196, 255} and after round(5%) and
            round(10%) of the epoch-1 batches nb (each with batch, examples seen and alive count). Seed 100 of A3 and A0:
            the C0y, per-epoch and final state hashes must equal the logged R1 runs (replay check, fails loudly).
            Twins: derived thresholds p_ins_q32 = p_neg_q32 = inv_s_q32 = 0, no insertion counted, sum(G) < s at the end
            (so owned-class decay provably never fired), and at every checkpoint every clause alive at the previous
            checkpoint with the same birth step has the identical literal slots, all included, states never lower.
  census    (CPU) --train-run R: per checkpoint alive, included literals per clause (mean, quantiles), fraction of clauses
            with a negated literal, with a bag literal, with more than 3 literals, the exact-token literal share, votes,
            and the literal churn between consecutive checkpoints. Metrics only.
  features  (GPU) --train-run R --checkpoints c0y,ladder,epoch1,epoch4: clause-count codes of the R1 downstream document
            set exactly as R1 stage_features builds them (R.doc_set, R.bank_codes, R.pipeline_checks, R.save_sparse,
            R.fire_counts and the dim-matched rule K.top_columns_by over the 20M prefix), features_<name>.npz and
            dim_<name>.npy per checkpoint, and tokens.npz with the R1 layout and labels (checked equal to an R1 features
            run, so label draws and folds pair exactly). "ladder" = every young rung of the run.
  qpred     (GPU) --train-run R --checkpoints c0y,epoch4: A7's Q_pred, the docs/11 11.5 T1 (loglin) masked top-1 on the
            frozen V 8000 validation targets; the H8.1 masked flow of cvtm_t1_readout.run_masked (which is bound to the
            V 2000 tiny-LM task) re-assembled from its helpers: count O_jc over the prefix outside the fit documents,
            fit alpha and beta0 on the fit documents (GPU objective, CPU float64 check), score the validation targets
            (CPU float64 reference kernel, GPU cross-check). Validation only; the test-access counter must not change.
  probe     (CPU) --features-run F --sets auto|<name.all,name.dim,...> --budgets 8,32,128: the R1 LR probe (PR.lr_probe,
            CPU backend, identical draws and folds), cells.jsonl in the R1 row format; AnyStore accepts any
            features_<name>.npz. No TM probe (LR is the registered primary probe).
  report    (CPU) --probe-runs a,b,... --qpred-runs c,d,...: the reading rule of the config, applied mechanically.

Launch: the chain chain_controls.sh (CPU slot first, GPU lock inside each step).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for _q in (ROOT / "src", EXPERIMENTS):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_pretext as R  # noqa: E402  (imported, never edited)
from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.engine.config import INCLUDE_THRESHOLD, LIT_EMPTY, N_EXACT_SEG, EngineConfig, derive  # noqa: E402
from tmlm.ssl import controls as K  # noqa: E402
from tmlm.ssl import downstream as D  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402  (imported, never edited)
from tmlm.ssl import probes as PR  # noqa: E402

T0 = time.time()
CORPUS = R.CORPUS
DEFAULT_CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_controls.json"
# control arm -> (R1 arm whose data and config it uses, literal learning off)
CONTROL_ARMS: dict[str, tuple[str, bool]] = {"A3": ("A3", False), "A0": ("A0", False), "A3T": ("A3", True), "A0T": ("A0", True)}
CODES = ("all", "dim")


# ------------------------------------------------------------------ config
def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_shared_params(cfg_json: dict, rec: R.Rec | None = None) -> None:
    """The params this study shares with R1 must equal the R1 config's (they define data, doc set and probes)."""
    r1 = json.loads((ROOT / cfg_json["r1_config"]).read_text(encoding="utf-8"))["params"]
    p = cfg_json["params"]
    bad = {k: (p.get(k), r1.get(k)) for k in cfg_json["params_shared_with_r1"] if p.get(k) != r1.get(k)}
    if rec is not None:
        rec.check("config.params_shared_with_r1_equal", not bad, differing=bad)
    if bad:
        raise SystemExit(f"params differ from the R1 config: {bad}")


def twin_overrides(p: dict) -> dict:
    t = p["twin_overrides"]
    return {"p_ins": float(t["p_ins"]), "p_neg": float(t["p_neg"]), "s": float(2 ** int(t["s_log2"]))}


def arm_config(arm: str, seed: int, p: dict, **extra) -> EngineConfig:
    """EngineConfig of a control arm: the R1 arm's config (P.engine_config), plus the twin overrides for A3T/A0T,
    plus `extra` (tests: tiny sizes)."""
    base, twin = CONTROL_ARMS[arm]
    kw = dict(twin_overrides(p)) if twin else {}
    kw.update(extra)
    vocab = kw.pop("vocab", int(p["vocab"]))
    n_clauses = kw.pop("n_clauses", int(p["n_clauses"]))
    return P.engine_config(base, int(seed), int(vocab), int(n_clauses), **kw)


def twin_thresholds(cfg: EngineConfig) -> dict:
    """The integers the engines actually receive for the literal paths (derive() and the rational s of settle)."""
    d = derive(cfg)
    s_num, s_den = Fraction(cfg.s).as_integer_ratio()
    return {"p_ins_q32": int(d.p_ins_q32), "p_neg_q32": int(d.p_neg_q32), "inv_s_q32": int(d.inv_s_q32),
            "s_num": int(s_num), "s_den": int(s_den), "p_veto_q32": int(d.p_veto_q32)}


def twin_thresholds_ok(cfg: EngineConfig, base: EngineConfig) -> bool:
    t, b = twin_thresholds(cfg), twin_thresholds(base)
    return (t["p_ins_q32"] == 0 and t["p_neg_q32"] == 0 and t["inv_s_q32"] == 0 and t["s_den"] == 1
            and t["s_num"] >= 1 << 62 and t["p_veto_q32"] == b["p_veto_q32"])


def ladder_points(nb: int, p: dict) -> dict[int, str]:
    """batch count (1-based, after that many batches of epoch 1) -> checkpoint name young_b<N>."""
    lad = p["young_ladder"]
    pts: dict[int, str] = {}
    for b in lad["abs_batches"]:
        if 1 <= int(b) <= int(nb):
            pts[int(b)] = f"young_b{int(b)}"
    for pct in lad["percent_of_epoch1"]:
        b = max(1, (int(nb) * int(pct) + 50) // 100)  # round half up of nb * pct / 100, exact integers
        pts[b] = f"young_b{b}"
    return dict(sorted(pts.items()))


def out_dir(stage: str) -> Path:
    d = runlog.current_run_dir()
    if d is not None:
        return Path(d)
    s = Path(os.environ.get("SSL_SCRATCH", "/tmp")) / f"ssl_r1c_{stage}_{os.getpid()}_{int(time.time())}"
    s.mkdir(parents=True, exist_ok=True)
    return s


# ------------------------------------------------------------------ literal freezing (all arms: diagnostics; twins: checks)
def literal_view(st: dict) -> dict:
    return {"alive": np.asarray(st["alive"]).astype(bool), "birth": np.asarray(st["birth_step"]).copy(),
            "lit_id": np.asarray(st["lit_id"]).copy(), "lit_state": np.asarray(st["lit_state"]).copy()}


def literal_compare(prev: dict | None, cur: dict) -> dict:
    """Counts over clauses alive now (and, with `prev`, over the incarnations alive at both points with the same
    birth step). For a twin every count except n_alive / n_common must be 0."""
    alive, lid, ls = cur["alive"], cur["lit_id"], cur["lit_state"]
    tracked = (lid != LIT_EMPTY) & alive[:, None]
    seg, neg = lid >> 17, lid & 1
    out = {"n_alive": int(alive.sum()),
           "tracked_not_included": int((tracked & (ls < INCLUDE_THRESHOLD)).sum()),
           "tracked_negated_or_bag": int((tracked & ((neg == 1) | (seg >= N_EXACT_SEG))).sum())}
    if prev is None:
        return out
    both = prev["alive"] & alive & (prev["birth"] == cur["birth"])
    same_ids = np.all(prev["lit_id"] == lid, axis=1)
    lowered = np.any(ls < prev["lit_state"], axis=1)
    inc_p = np.sort(np.where(prev["lit_state"] >= INCLUDE_THRESHOLD, prev["lit_id"], LIT_EMPTY), axis=1)
    inc_c = np.sort(np.where(ls >= INCLUDE_THRESHOLD, lid, LIT_EMPTY), axis=1)
    inc_changed = ~np.all(inc_p == inc_c, axis=1)
    out.update({"n_common": int(both.sum()), "lit_slots_changed": int((both & ~same_ids).sum()),
                "included_set_changed": int((both & inc_changed).sum()),
                "state_lowered": int((both & same_ids & lowered).sum())})
    return out


def twin_frozen(stats: dict) -> bool:
    return (stats["tracked_not_included"] == 0 and stats["tracked_negated_or_bag"] == 0
            and stats.get("lit_slots_changed", 0) == 0 and stats.get("included_set_changed", 0) == 0
            and stats.get("state_lowered", 0) == 0)


# ------------------------------------------------------------------ train
def train_ladder(ad: P.ArmData, cfg: EngineConfig, epochs: int, out: Path, engine_kind: str, rec: R.Rec, p: dict,
                 twin: bool, yield_gpu: bool = True) -> dict:
    """The R1 training loop of R.train_arm (same engine rebuilds, same batches, same C0y checkpoint) plus the young
    ladder checkpoints and the literal-freezing observations. Returns {'c0y': h, 'young': {name: h}, 'epochs': {k: h},
    'totals': summed DIAG, 'sum_G': int}."""
    from tmlm.engine.checkpoint import save_checkpoint

    hashes: dict[str, Any] = {"c0y": None, "young": {}, "epochs": {}, "totals": {}}
    eng = None
    prev_view = None
    for e in range(int(epochs)):
        if eng is None or ad.targets_vary_by_epoch:
            corpus = ad.corpus(e)
            t_b = time.time()
            state = None
            if eng is not None:
                state = eng.state()
                del eng
            eng = R.make_engine(engine_kind, cfg, corpus)
            if state is not None:
                eng.load_state(state)
            rec.m(f"train.epoch{e + 1}.engine_build_s", time.time() - t_b)
        n_ex = eng.n_examples
        nb = eng.n_batches()
        rec.m(f"train.epoch{e + 1}.targets", int(n_ex))
        saves: dict[int, list[str]] = {}
        if e == 0:
            young_b = K.c0y_batches(n_ex, cfg.batch, float(p["young_fraction"]))
            saves.setdefault(young_b, []).append("c0y")
            pts = ladder_points(nb, p)
            for b, name in pts.items():
                saves.setdefault(b, []).append(name)
            rec.m("ladder.batches_in_epoch1", int(nb))
            rec.m("ladder.points", {name: int(b) for b, name in pts.items()}, note=p["young_ladder"]["rounding"])
            rec.m("ladder.skipped_abs_batches", [int(b) for b in p["young_ladder"]["abs_batches"] if int(b) > nb])
        diag: dict[str, int] = {}
        t0 = time.time()
        t_ck = 0.0
        for b in range(nb):
            d = eng.train_batch(e, b)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + int(v)
            names = saves.get(b + 1)
            if not names:
                continue
            tc = time.time()
            st = eng.state()
            ex_seen = int(min(n_ex, (b + 1) * cfg.batch))
            alive = int(st["alive"].sum())
            for name in names:
                if name == "c0y":
                    h = save_checkpoint(out / "checkpoint_c0y.npz", eng, epoch=0, batch_index=b + 1, compress=True,
                                        extra={"what": "C0y: the same run stopped at 1 percent of epoch 1 (A7)",
                                               "batches": b + 1, "batches_in_epoch": nb, "examples": ex_seen})
                    hashes["c0y"] = h
                    rec.m("c0y.batches", b + 1, note=f"ceil({p['young_fraction']} x {nb} batches of epoch 1)")
                    rec.m("c0y.examples", ex_seen)
                    rec.m("c0y.state_hash", h)
                    rec.m("c0y.alive", alive)
                else:
                    h = save_checkpoint(out / f"checkpoint_{name}.npz", eng, epoch=0, batch_index=b + 1, compress=True,
                                        extra={"what": "young ladder rung (control study)", "batches": b + 1,
                                               "batches_in_epoch": nb, "examples": ex_seen, "alive": alive})
                    hashes["young"][name] = h
                    rec.m(f"young.{name}.batches", b + 1)
                    rec.m(f"young.{name}.examples", ex_seen)
                    rec.m(f"young.{name}.alive", alive)
                    rec.m(f"young.{name}.state_hash", h)
            view = literal_view(st)
            stats = literal_compare(prev_view, view)
            tag = "+".join(names)
            rec.m(f"literals.{tag}", stats, note="vs the previous checkpoint of this run (same clause, same birth step)")
            if twin:
                rec.check(f"twin.literals_frozen.{tag}", twin_frozen(stats), **stats)
            prev_view = view
            t_ck += time.time() - tc
        if hasattr(eng, "synchronize"):
            eng.synchronize()
        wall = time.time() - t0 - t_ck
        st = eng.state()
        h = save_checkpoint(out / f"checkpoint_epoch{e + 1}.npz", eng, epoch=e + 1, batch_index=0, compress=True)
        hashes["epochs"][e + 1] = h
        for k, v in diag.items():
            hashes["totals"][k] = hashes["totals"].get(k, 0) + v
        pre = f"train.epoch{e + 1}"
        rec.m(f"{pre}.wall_s", wall, note="training only (checkpoint writing excluded)")
        rec.m(f"{pre}.checkpoint_s", t_ck)
        rec.m(f"{pre}.examples_per_s", diag.get("n_examples", 0) / max(1e-9, wall))
        rec.m(f"{pre}.n_examples", int(diag.get("n_examples", 0)))
        for k in ("n_candidates", "n_fired", "n_events_applied", "n_events_dropped_cap", "n_tgt_gate_pass",
                  "n_neg_gate_pass", "n_inserted", "n_neg_inserted"):
            rec.m(f"{pre}.{k}_per_example", diag.get(k, 0) / max(1, diag.get("n_examples", 0)))
        for k in ("n_seeded", "n_recycled", "n_events_dropped_cap", "n_inserted", "n_neg_inserted", "n_type1", "n_type2", "n_vetoes"):
            rec.m(f"{pre}.{k}", int(diag.get(k, 0)))
        rec.m(f"{pre}.alive", int(st["alive"].sum()))
        rec.m(f"{pre}.state_hash", h)
        view = literal_view(st)
        stats = literal_compare(prev_view, view)
        rec.m(f"literals.epoch{e + 1}", stats, note="vs the previous checkpoint of this run (same clause, same birth step)")
        if twin:
            rec.check(f"twin.literals_frozen.epoch{e + 1}", twin_frozen(stats), **stats)
        prev_view = view
        if hasattr(eng, "device_memory_used"):
            rec.m(f"{pre}.device_pool_bytes", int(eng.device_memory_used()), unit="bytes")
        rec.m(f"{pre}.memory", R.mem_info())
        rec.log(f"epoch {e + 1}: {diag.get('n_examples', 0)} examples in {wall:.0f} s, alive {int(st['alive'].sum())}, hash {h[:12]}")
        rec.flush()
        if yield_gpu and engine_kind == "cuda":
            from tmlm.common.locks import yield_gpu_if_requested
            if yield_gpu_if_requested():
                rec.log("yielded the GPU to an exclusive job")
    hashes["sum_G"] = int(sum(int(x) for x in np.asarray(eng.state()["G"]).tolist()))
    return hashes


def replay_check(cfg_json: dict, arm: str, seed: int, cfg: EngineConfig, hashes: dict, epochs: int, rec: R.Rec) -> None:
    ref = cfg_json.get("replay_reference", {}).get(arm, {}).get(str(seed))
    if not ref:
        rec.m("replay.reference", None, note="no logged R1 run for this arm and seed: no replay check")
        return
    f = ROOT / ref / "metrics.json"
    if not f.exists():
        rec.check("replay.reference_exists", False, path=str(ref))
        return
    m = json.loads(f.read_text())
    rec.m("replay.reference", str(ref))
    rec.check("replay.engine_config_equal", m["engine_config"]["value"] == cfg.to_dict())
    pairs = [("c0y.state_hash", hashes["c0y"])] + [(f"train.epoch{k}.state_hash", hashes["epochs"][k]) for k in range(1, epochs + 1)]
    pairs.append(("final.state_hash", hashes["epochs"][epochs]))
    for key, ours in pairs:
        theirs = m.get(key, {}).get("value")
        rec.check(f"replay.{key}_equals_r1_run", ours == theirs, ours=ours, reference=theirs)


def stage_train(args, cj: dict, rec: R.Rec) -> None:
    p = cj["params"]
    out = out_dir("train")
    base, twin = CONTROL_ARMS[args.arm]
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    cfg = arm_config(args.arm, args.seed, p)
    rec.m("control.arm", args.arm)
    rec.m("control.seed", int(args.seed))
    rec.m("control.base_arm", base)
    rec.m("control.twin", bool(twin))
    rec.m("arm", P.ARMS[base].to_dict())
    rec.m("engine_config", cfg.to_dict())
    rec.m("engine_thresholds", twin_thresholds(cfg), note="derive() and the rational s of settle: what the engine receives")
    rec.m("arm_data", ad.info)
    if twin:
        rec.m("twin.overrides", twin_overrides(p))
        rec.check("twin.literal_thresholds_zero", twin_thresholds_ok(cfg, P.engine_config(base, args.seed, int(p["vocab"]), int(p["n_clauses"]))),
                  **twin_thresholds(cfg))
    rec.flush()
    hashes = train_ladder(ad, cfg, int(p["epochs"]), out, args.engine, rec, p, twin)
    E = int(p["epochs"])
    rec.m("final.state_hash", hashes["epochs"][E])
    rec.m("train.totals", hashes["totals"])
    rec.m("train.sum_G_final", str(hashes["sum_G"]), note="Q16 gate mass summed over classes (string: exact integer)")
    if twin:
        t = hashes["totals"]
        rec.check("twin.no_literal_inserted", t.get("n_inserted", 0) == 0 and t.get("n_neg_inserted", 0) == 0,
                  n_inserted=int(t.get("n_inserted", 0)), n_neg_inserted=int(t.get("n_neg_inserted", 0)))
        s_num = twin_thresholds(cfg)["s_num"]
        rec.check("twin.decay_provably_zero", hashes["sum_G"] < s_num, sum_G=str(hashes["sum_G"]), s_num=str(s_num),
                  rule="owed <= M(j) <= sum(G) < s_num implies n_q16 = floor(owed / s) = 0 in every settle of the run")
    replay_check(cj, args.arm, int(args.seed), cfg, hashes, E, rec)
    rec.m("train.wall_total_s", time.time() - T0)
    rec.flush()


# ------------------------------------------------------------------ checkpoints of a train run
def checkpoint_step(path: Path) -> int:
    """Completed batches of a checkpoint (its meta 'step'; only the small meta array is read)."""
    with np.load(path) as z:
        return int(json.loads(bytes(z["__meta__"]).decode("utf-8"))["step"])


def checkpoint_names(train_run: Path) -> list[str]:
    """Every checkpoint of a train run in training order: by step (completed batches), young rungs before c0y at
    an equal step, then by name."""
    names = [q.stem[len("checkpoint_"):] for q in train_run.glob("checkpoint_*.npz")]
    return sorted(names, key=lambda n: (checkpoint_step(train_run / f"checkpoint_{n}.npz"), n == "c0y", n))


def resolve_names(spec: str, train_run: Path) -> list[str]:
    have = checkpoint_names(train_run)
    out: list[str] = []
    for tok in [s.strip() for s in spec.split(",") if s.strip()]:
        if tok == "all":
            add = have
        elif tok == "ladder":
            add = [n for n in have if n.startswith("young_b")]
        else:
            if tok not in have:
                raise SystemExit(f"no checkpoint_{tok}.npz in {train_run}")
            add = [tok]
        out += [n for n in add if n not in out]
    return out


def logged_hash_key(name: str) -> str:
    if name == "c0y":
        return "c0y.state_hash"
    if name.startswith("young_b"):
        return f"young.{name}.state_hash"
    if name.startswith("epoch"):
        return f"train.epoch{int(name[len('epoch'):])}.state_hash"
    raise KeyError(name)


def train_run_identity(train_run: Path) -> tuple[str, int]:
    return str(R.read_metric(train_run, "control.arm")), int(R.read_metric(train_run, "control.seed"))


def load_named_checkpoint(train_run: Path, name: str, arm: str, seed: int, p: dict, rec: R.Rec, **cfg_extra):
    """(cfg, state, meta) of checkpoint `name`, with the config equal to the arm's and the state hash equal to the
    one the train run logged (both checked)."""
    from tmlm.engine.checkpoint import load_checkpoint

    cfg, st, meta = load_checkpoint(train_run / f"checkpoint_{name}.npz")
    exp = arm_config(arm, seed, p, **cfg_extra)
    if cfg != exp:
        raise SystemExit(f"checkpoint_{name}: config differs from arm {arm} seed {seed}")
    logged = R.read_metric(train_run, logged_hash_key(name))
    rec.check(f"{name}.state_hash_equals_train_run", meta["state_hash"] == logged, state_hash=meta["state_hash"])
    return cfg, st, meta


# ------------------------------------------------------------------ census
def census_of(st: dict, vocab: int) -> dict:
    alive = np.asarray(st["alive"]).astype(bool)
    lid = np.asarray(st["lit_id"])[alive]
    ls = np.asarray(st["lit_state"])[alive]
    tracked = lid != LIT_EMPTY
    inc = tracked & (ls >= INCLUDE_THRESHOLD)
    seg, sym, neg = lid >> 17, (lid >> 1) & 0xFFFF, lid & 1
    n_inc = inc.sum(axis=1)
    n_all = int(inc.sum())
    exact = seg < N_EXACT_SEG
    tok = exact & (sym < vocab)
    vc = np.asarray(st["vote_cls"])[alive]
    vw = np.asarray(st["vote_w"])[alive]
    used = vc != 0xFFFF
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    frac = (lambda x: float(x.mean()) if x.shape[0] else None)
    return {
        "alive": int(alive.sum()),
        "included_per_clause_mean": frac(n_inc.astype(np.float64)),
        "included_per_clause_quantiles": dict(zip([str(v) for v in q], np.quantile(n_inc, q).tolist())) if n_inc.shape[0] else None,
        "included_per_clause_max": int(n_inc.max()) if n_inc.shape[0] else None,
        "included_per_clause_hist": np.bincount(n_inc, minlength=13).tolist(),
        "frac_clauses_with_negated": frac(np.any(inc & (neg == 1), axis=1)),
        "frac_clauses_with_bag": frac(np.any(inc & ~exact, axis=1)),
        "frac_clauses_with_more_than_3": frac(n_inc > 3),
        "frac_clauses_with_excluded_tracked": frac(np.any(tracked & ~inc, axis=1)),
        "tracked_per_clause_mean": frac(tracked.sum(axis=1).astype(np.float64)),
        "included_literals": n_all,
        "frac_literals_exact_token": float((inc & tok).sum() / n_all) if n_all else None,
        "frac_literals_exact_token_positive": float((inc & tok & (neg == 0)).sum() / n_all) if n_all else None,
        "frac_literals_exact_cluster": float((inc & exact & (sym >= vocab)).sum() / n_all) if n_all else None,
        "frac_literals_bag": float((inc & ~exact).sum() / n_all) if n_all else None,
        "frac_literals_negated": float((inc & (neg == 1)).sum() / n_all) if n_all else None,
        "owned_classes_per_clause_mean": frac((used & (vw > 0)).sum(axis=1).astype(np.float64)),
        "negative_votes_per_clause_mean": frac((used & (vw < 0)).sum(axis=1).astype(np.float64)),
        "abs_vote_mean": float(np.abs(vw[used]).mean()) if used.any() else None,
    }


def stage_census(args, cj: dict, rec: R.Rec, train_run: Path) -> None:
    from tmlm.engine.checkpoint import load_checkpoint

    p = cj["params"]
    arm, seed = train_run_identity(train_run)
    rec.m("census.train_run", str(train_run))
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    prev, prev_name = None, None
    for name in checkpoint_names(train_run):
        cfg, st, meta = load_checkpoint(train_run / f"checkpoint_{name}.npz")
        if cfg != arm_config(arm, seed, p):
            raise SystemExit(f"checkpoint_{name}: config differs from arm {arm} seed {seed}")
        logged = R.read_metric(train_run, logged_hash_key(name))
        rec.check(f"{name}.state_hash_equals_train_run", meta["state_hash"] == logged)
        c = census_of(st, int(cfg.vocab))
        c["step"] = int(st["step"][0])
        c["examples_seen"] = None if name.startswith("epoch") else int(meta.get("extra", {}).get("examples", -1))
        rec.m(f"census.{name}", c)
        view = literal_view(st)
        churn = literal_compare(prev, view)
        rec.m(f"census.{name}.churn_vs_previous", {**churn, "previous": prev_name,
                                                   "frac_common_included_set_changed": (churn.get("included_set_changed", 0) / max(1, churn.get("n_common", 0))) if prev is not None else None})
        prev, prev_name = view, name
        rec.log(f"census {name}: alive {c['alive']}, included/clause {c['included_per_clause_mean']}")
    rec.flush()


# ------------------------------------------------------------------ features
def featurize_checkpoints(names: list[str], train_run: Path, arm: str, seed: int, docs, corpus_docs, corpus_prefix,
                          n_prefix: int, p: dict, engine_kind: str, out: Path, rec: R.Rec, **cfg_extra) -> dict:
    """features_<name>.npz, fires_<name>.npy and dim_<name>.npy per checkpoint, by the R1 functions. A checkpoint
    whose state hash equals an earlier one's (for example c0y and the ladder rung at the same batch) is copied, not
    recomputed (identical state, deterministic featurisation). Returns {name: state_hash}."""
    chunk = int(p["forward_chunk"])
    done: dict[str, str] = {}
    for name in names:
        cfg, st, meta = load_named_checkpoint(train_run, name, arm, seed, p, rec, **cfg_extra)
        h = meta["state_hash"]
        rec.m(f"{name}.alive", int(np.asarray(st["alive"]).sum()))
        twin_of = next((n for n, hh in done.items() if hh == h), None)
        if twin_of is not None:
            for pat in ("features_{}.npz", "fires_{}.npy", "dim_{}.npy"):
                shutil.copyfile(out / pat.format(twin_of), out / pat.format(name))
            rec.m(f"features.{name}.duplicate_of", twin_of, note="identical state hash: files copied")
            rec.m(f"features.{name}.file_sha256", hashing.sha256_file(out / f"features_{name}.npz"))
            done[name] = h
            continue
        t0 = time.time()
        eng = R.make_engine(engine_kind, cfg, corpus_docs)
        eng.load_state(st)
        sd = R.bank_codes(eng, cfg, docs, chunk, rec, name)
        R.pipeline_checks(eng, cfg, docs, sd, rec, name)
        rec.m(f"features.{name}.nnz", int(sd.ids.shape[0]))
        rec.m(f"features.{name}.distinct_clauses_per_doc_mean", float(np.diff(sd.indptr).mean()) if sd.n_doc else None)
        rec.m(f"features.{name}.file_sha256", R.save_sparse(out / f"features_{name}.npz", sd, state_hash=h))
        del eng
        eng = R.make_engine(engine_kind, cfg, corpus_prefix)
        eng.load_state(st)
        t1 = time.time()
        fc = R.fire_counts(eng, cfg, int(n_prefix), chunk)
        rec.m(f"prefix_fires.{name}.wall_s", time.time() - t1)
        rec.m(f"prefix_fires.{name}.clauses_firing", int((fc > 0).sum()))
        rec.m(f"prefix_fires.{name}.fired_per_position", float(fc.sum()) / max(1, int(n_prefix)))
        dim = K.top_columns_by(fc.astype(np.int64), int(p["dim_matched"]))
        np.save(out / f"fires_{name}.npy", fc)
        np.save(out / f"dim_{name}.npy", dim)
        rec.m(f"features.{name}.dim_matched_columns", int(dim.shape[0]))
        del eng
        if engine_kind == "cuda":
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        rec.m(f"memory.after_{name}", R.mem_info())
        rec.log(f"{name}: features and prefix fire counts in {time.time() - t0:.0f} s")
        rec.flush()
        done[name] = h
    return done


def save_tokens(out: Path, docs, layout: dict, labels: dict, arm: str, seed: int) -> None:
    np.savez(out / "tokens.npz", tok=docs.tok, doc_start=docs.doc_start, end_id=np.int64(docs.end_id),
             layout=json.dumps(layout), labels=json.dumps(labels), arm=arm, seed=np.int64(seed))


def doc_set_equal(ours: Path, ref: Path) -> dict:
    a, b = np.load(ours), np.load(ref)
    return {"tok": bool(np.array_equal(a["tok"], b["tok"])), "doc_start": bool(np.array_equal(a["doc_start"], b["doc_start"])),
            "end_id": int(a["end_id"]) == int(b["end_id"]),
            "layout": json.loads(str(a["layout"])) == json.loads(str(b["layout"])),
            "labels": json.loads(str(a["labels"])) == json.loads(str(b["labels"]))}


def stage_features(args, cj: dict, rec: R.Rec, train_run: Path) -> Path:
    p = cj["params"]
    out = out_dir("features")
    arm, seed = train_run_identity(train_run)
    base, _ = CONTROL_ARMS[arm]
    names = resolve_names(args.checkpoints, train_run)
    rec.m("features.train_run", str(train_run))
    rec.m("features.checkpoints", names)
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    docs, layout, labels = R.doc_set(p, rec, False)
    save_tokens(out, docs, layout, labels, arm, seed)
    ref = cj.get("doc_set_reference", {}).get("run")
    if ref:
        eq = doc_set_equal(out / "tokens.npz", ROOT / ref / "tokens.npz")
        rec.check("docs.equal_to_r1_features_run", all(eq.values()), reference=ref, **eq)
    rec.flush()
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    from tmlm.ssl import features as F
    corpus_docs = F.corpus_arrays(docs, ad.feature_tables())
    corpus_prefix = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    featurize_checkpoints(names, train_run, arm, seed, docs, corpus_docs, corpus_prefix, int(prefix.tok.shape[0]), p,
                          args.engine, out, rec)
    rec.flush()
    return out


# ------------------------------------------------------------------ probe
class AnyStore(R.Store):
    """R.Store that serves any features_<name>.npz of a features run (young rungs, c0y, epochK, ...); dim_<name>.npy
    and the doc layout are read exactly as R.Store reads them."""

    def sparse(self, name: str):
        if name not in self._cache:
            f = self.fr / f"features_{name}.npz"
            if not f.exists():
                return super().sparse(name)
            self._cache[name] = R.load_sparse(f)
        return self._cache[name]

    def feature_names(self) -> list[str]:
        return sorted(q.stem[len("features_"):] for q in self.fr.glob("features_*.npz"))

    def file_sha256(self, name: str, var: str = "all") -> str:
        """sha256 of features_<name>.npz (and of dim_<name>.npy for the dim code): the identity of a probe input."""
        key = ("sha", name, var)
        if key not in self._cache:
            h = hashing.sha256_file(self.fr / f"features_{name}.npz")
            if var == "dim":
                h += hashing.sha256_file(self.fr / f"dim_{name}.npy")
            self._cache[key] = h
        return self._cache[key]


def parse_sets(spec: str, store: AnyStore) -> list[str]:
    if spec.strip() == "auto":
        return [f"{n}.{c}" for n in store.feature_names() for c in CODES]
    sets = [s.strip() for s in spec.split(",") if s.strip()]
    for s in sets:
        name, _, code = s.partition(".")
        if code not in CODES or not (store.fr / f"features_{name}.npz").exists():
            raise SystemExit(f"bad set {s!r}: need <name>.all or <name>.dim with features_<name>.npz in {store.fr}")
    return sets


def stage_probe(args, cj: dict, rec: R.Rec, features_run: Path) -> Path:
    p = cj["params"]
    out = out_dir("probe")
    if (features_run / "metrics.json").exists():
        failed = json.loads((features_run / "metrics.json").read_text()).get("failed_checks", {}).get("value")
        if failed:
            raise SystemExit(f"{features_run} has failed checks {failed}; not probing it")
    store = AnyStore(features_run, None, p)
    sets = parse_sets(args.sets, store)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    if not set(budgets) <= {int(b) for b in p["budgets"]}:
        raise SystemExit(f"budgets {budgets} must be R1 budgets {p['budgets']} (the doc set holds only their labelled rows)")
    rec.m("probe.sets", sets)
    rec.m("probe.budgets", budgets)
    rec.m("probe.features_run", str(features_run))
    rec.m("probe.arm", store.arm)
    rec.m("probe.seed", store.seed)
    backend = str(p.get("lr_backend", "cpu"))
    rec.m("probe.lr_backend", backend)
    cache: dict = {}
    cells = out / "cells.jsonl"
    n_cells = n_reused = 0
    with open(cells, "a", encoding="utf-8") as fh:
        for task in p["tasks"]:
            code = D.TASKS[task]["code"]
            pool_y = np.asarray(store.labels[task]["pool_y"], np.int64)
            dev_y = np.asarray(store.labels[task]["dev_y"], np.int64)
            dev = store.dev_rows(task)
            for budget in budgets:
                for draw in p["draws"]:
                    pr = PR.label_sample(pool_y, budget, code, draw)
                    lab = store.rows(task, pr)
                    yl = pool_y[pr]
                    seed = PR.fold_seed(code, budget, draw)
                    for fs in sets:
                        t0 = time.time()
                        name, _, var = fs.partition(".")
                        key = (store.file_sha256(name, var), var, task, budget, draw)
                        if key in cache:
                            c = cache[key]
                            n_reused += 1
                        else:
                            Xl, Xd, dense = store.lr_matrices(fs, lab, dev)
                            c = PR.lr_probe(Xl, yl, Xd, dev_y, seed, dense=dense, c_grid=p["c_grid"], folds=int(p["folds"]),
                                            max_iter=int(p["lr_max_iter"]), backend=backend)
                            cache[key] = c
                        row = {"arm": store.arm, "seed": store.seed, "set": fs, "clf": "lr", "task": task, "budget": budget,
                               "draw": draw, "acc": c.accuracy, "choice": c.choice, "cv": c.cv, "n_labelled": c.n_labelled,
                               "n_eval": c.n_eval, "class_counts": np.bincount(yl, minlength=D.TASKS[task]["n_classes"]).tolist(),
                               "wall_s": time.time() - t0, "smoke": False}
                        fh.write(json.dumps(row, default=str) + "\n")
                        fh.flush()
                        n_cells += 1
                        rec.m(f"cell.{task}.{fs}.lr.{budget}.d{draw}", c.accuracy, n_labelled=c.n_labelled, n_eval=c.n_eval)
                    rec.log(f"{task} budget {budget} draw {draw}: {len(sets)} sets done")
                    rec.flush()
    rec.m("probe.cells", n_cells)
    rec.m("probe.cells_reused_identical_features", n_reused)
    rec.m("probe.cells_sha256", hashing.sha256_file(cells))
    rec.m("memory", R.mem_info())
    rec.flush()
    return out


# ------------------------------------------------------------------ qpred (docs/11 11.5 T1 read-out, H8.1 flow)
def api_fired_lists(eng, positions: np.ndarray, bitmap: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Fired lists (ascending ids per row) through the engine's public fired_clauses (any engine)."""
    import cvtm_t1_readout as T1
    from tmlm.engine.cpu.cpu_engine import CpuEngine

    if isinstance(eng, CpuEngine):
        ptr, ids = T1.cpu_fired_lists(eng, np.asarray(positions, np.int64), bitmap)
        rowid = np.repeat(np.arange(ptr.shape[0] - 1, dtype=np.int64), np.diff(ptr))
        return ptr, ids[np.lexsort((ids, rowid))].astype(np.uint32)  # ascending ids within each row
    n = int(positions.shape[0])
    ptr = np.zeros(n + 1, np.int64)
    parts = []
    for i, q in enumerate(np.asarray(positions, np.int64).tolist()):
        a = np.sort(np.asarray(eng.fired_clauses(T1.EPOCH, int(q), bitmap), np.uint32))
        parts.append(a)
        ptr[i + 1] = ptr[i] + a.shape[0]
    return ptr, (np.concatenate(parts) if parts else np.zeros(0, np.uint32))


def count_table(eng, engine_kind: str, positions: np.ndarray, targets_of_rows: np.ndarray, C: int, V: int, chunk: int,
                zero: np.ndarray, check_rows: np.ndarray | None = None) -> tuple[np.ndarray, int, dict]:
    """O[j, c] = number of counting positions where clause j fires and the target is c (exact int32), the number of
    counted events, and the fired lists of `check_rows` (row indices into `positions`) for the cross-engine check."""
    import cvtm_t1_readout as T1

    counts = np.zeros((C, V), dtype=np.int32)
    tgt = np.asarray(targets_of_rows, np.int64)
    got: dict[int, np.ndarray] = {}
    chk = np.zeros(0, np.int64) if check_rows is None else np.sort(np.asarray(check_rows, np.int64))
    n_events = 0
    if engine_kind == "cuda":
        cp = eng.cp
        zero_dev = eng._bitmap(zero)
        for lo in range(0, positions.shape[0], chunk):
            hi = min(positions.shape[0], lo + chunk)
            eng._sync_state_struct()
            fw = eng._forward(T1.EPOCH, cp.asarray(positions[lo:hi].astype(np.uint32)), zero_dev, 0)
            eng._marks = []
            fe = fw.fired_ex.get().astype(np.int64)
            fj = fw.fired_j.get().astype(np.int64)
            del fw
            if fe.shape[0] and not bool(np.all(fe[1:] >= fe[:-1])):
                raise RuntimeError("fired_ex is not example-major")
            T1.k_accumulate_events(fe, fj, tgt[lo:hi], counts)
            n_events += int(fe.shape[0])
            a, b = np.searchsorted(chk, [lo, hi])
            for r in chk[a:b].tolist():
                s0, s1 = np.searchsorted(fe, [r - lo, r - lo + 1])
                got[r] = np.sort(fj[s0:s1]).astype(np.uint32)
        eng.synchronize()
    else:
        ptr, ids = api_fired_lists(eng, positions, zero)
        fe = np.repeat(np.arange(positions.shape[0], dtype=np.int64), np.diff(ptr))
        T1.k_accumulate_events(fe, ids.astype(np.int64), tgt, counts)
        n_events = int(ids.shape[0])
        for r in chk.tolist():
            got[r] = ids[ptr[r]:ptr[r + 1]].astype(np.uint32)
    return counts, n_events, got


def fit_rows_fired(eng, engine_kind: str, positions: np.ndarray, zero: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    import cvtm_t1_readout as T1

    if engine_kind == "cuda":
        return T1.cuda_rows(eng, positions, eng._bitmap(zero), chunk)
    return api_fired_lists(eng, positions, zero)


class Tables:
    """The count tables and the read-out constants of docs/11 11.5 (cvtm_t1_readout helpers, unchanged)."""

    def __init__(self, counts: np.ndarray, n_inc: np.ndarray, count_targets: np.ndarray, V: int) -> None:
        import cvtm_t1_readout as T1
        from ckm_posthoc_probe import k_dense_to_csr, k_gval

        self.V = int(V)
        self.indptr, self.idx, self.val = k_dense_to_csr(counts)
        self.N = T1.k_row_sums(self.indptr, self.val)
        self.D = np.diff(self.indptr)
        self.lam, self.inv_n = T1.witten_bell(self.N, self.D)
        self.p_uni = T1.add_one_unigram(np.asarray(count_targets, np.int64), self.V)
        self.logpu = np.log(self.p_uni)
        self.grp = T1.clause_groups(n_inc, self.N)
        self.gval = k_gval(self.indptr, self.idx, self.val, self.lam, self.inv_n, self.p_uni)
        self.has = self.N > 0

    def rows(self, ptr: np.ndarray, ids: np.ndarray, gold: np.ndarray) -> dict:
        import cvtm_t1_readout as T1

        rp, rowid, comp = T1.csr_select(ptr, ids, self.has)
        return dict(n=int(ptr.shape[0] - 1), rp=rp, rowid=rowid, comp=comp, bkt=self.grp[comp].astype(np.int64),
                    gold=np.asarray(gold, np.int64))

    def loglin(self, rows: dict, alpha: np.ndarray, beta0: float, want_grad: bool):
        import cvtm_t1_readout as T1

        return T1.cpu_loglin(rows, alpha, beta0, self.indptr, self.idx, self.gval, self.logpu, want_grad)


def loglin_fit_cpu(tab: Tables, rows: dict, budget: dict) -> tuple[np.ndarray, float, dict]:
    """alpha per group and beta0 by L-BFGS-B on the CPU float64 reference objective (tests and the CPU/oracle path;
    the CUDA path uses cvtm_t1_readout.loglin_fit_gpu, the H8.1 fit, unchanged). Same init and bounds."""
    import cvtm_t1_readout as T1

    bk = rows["bkt"]
    used = np.unique(bk)
    n = rows["n"]

    def fun(theta):
        alpha = np.zeros(T1.N_BUCKETS)
        alpha[used] = theta[:-1]
        ll, grad, gb, _, _, _ = tab.loglin(rows, alpha, float(theta[-1]), True)
        return -ll / n, -np.append(grad[used], gb) / n
    a_init = 1.0 / max(1.0, float(np.diff(rows["rp"]).mean())) if n else 1.0
    lo = np.append(np.full(used.shape[0], float(budget["alpha_bounds"][0])), float(budget["beta0_bounds"][0]))
    hi = np.append(np.full(used.shape[0], float(budget["alpha_bounds"][1])), float(budget["beta0_bounds"][1]))
    x0 = np.append(np.full(used.shape[0], a_init), 1.0)
    res, trace = T1.lbfgsb(fun, x0, lo, hi, budget)
    f, g = fun(res.x)
    alpha = np.zeros(T1.N_BUCKETS)
    alpha[used] = res.x[:-1]
    return alpha, float(res.x[-1]), T1.fit_info(res, trace, res.x, g, lo, hi, n, {"used_groups": used.tolist(), "objective_cpu_float64": float(f)})


def qpred_checkpoint(name: str, cfg: EngineConfig, st: dict, prefix, val, targets: np.ndarray, hidden: np.ndarray,
                     fit_pos: np.ndarray, cnt_pos: np.ndarray, qp: dict, engine_kind: str, chunk: int, rec: R.Rec,
                     out: Path) -> dict:
    """Q_pred of one bank: the H8.1 masked flow (cvtm_t1_readout.run_masked) with the R1 prefix as counting text
    and the frozen V 8000 validation targets. Returns the metric rows."""
    import cvtm_t1_readout as T1
    from ckm_posthoc_probe import k_vote_scores

    V, C = int(cfg.vocab), int(cfg.n_clauses)
    pre = f"qpred.{name}"
    tok_tr = np.asarray(prefix.tok).astype(np.int64)
    gold = np.asarray(val.tok)[targets].astype(np.int64)
    n_inc = T1.included_literals(st)
    chk = qp["checks"]
    # ---- validation fired lists (CPU engine, frozen hidden mask), API and brute-force checks
    t0 = time.time()
    eng_va = R.make_engine("cpu" if engine_kind != "oracle" else "oracle", cfg, val)
    eng_va.load_state(st)
    vptr, vids = api_fired_lists(eng_va, targets, hidden)
    rec.m(f"{pre}.validation_fired_wall_s", time.time() - t0)
    bad = 0
    for i in T1.sample_positions(np.arange(targets.shape[0]), int(chk["validation_api_rows"]), 1).tolist():
        bad += int(not np.array_equal(np.sort(eng_va.fired_clauses(T1.EPOCH, int(targets[i]), hidden)), vids[vptr[i]:vptr[i + 1]]))
    for i in T1.sample_positions(np.arange(targets.shape[0]), int(chk["validation_bruteforce_rows"]), 2).tolist():
        bad += int(not np.array_equal(np.sort(eng_va.fired_clauses_bruteforce(T1.EPOCH, int(targets[i]), hidden)), vids[vptr[i]:vptr[i + 1]]))
    rec.check(f"{pre}.validation_fired_lists_equal_api_and_bruteforce", bad == 0, mismatches=bad)
    # ---- the bank's own signed votes (secondary row) from our fired lists, checked against predict_scores
    signed_correct = np.zeros(targets.shape[0], np.int8)
    vcls, vw = np.asarray(st["vote_cls"]), np.asarray(st["vote_w"]).astype(np.int32)
    for i in range(0, targets.shape[0], 4096):
        s = k_vote_scores(vptr, vids, np.arange(i, min(targets.shape[0], i + 4096), dtype=np.int64), vcls, vw, V)
        signed_correct[i:i + s.shape[0]] = (s.argmax(axis=1) == gold[i:i + s.shape[0]])
    smp = T1.sample_positions(np.arange(targets.shape[0]), int(chk["signed_predict_scores_rows"]), 5)
    ps = eng_va.predict_scores(T1.EPOCH, targets[smp].astype(np.uint32), hidden)
    rebuilt = k_vote_scores(vptr, vids, smp.astype(np.int64), vcls, vw, V)
    rec.check(f"{pre}.rebuilt_vote_sums_equal_predict_scores", bool(np.array_equal(ps, rebuilt)), rows=int(smp.shape[0]))
    signed_top1 = float(signed_correct.mean()) if targets.shape[0] else 0.0
    del eng_va
    # ---- counting pass over the prefix outside the fit documents (only the target hidden)
    zero = np.zeros(int(prefix.tok.shape[0]), np.uint8)
    eng_g = R.make_engine(engine_kind, cfg, prefix)
    eng_g.load_state(st)
    check_cnt = T1.sample_positions(np.arange(cnt_pos.shape[0]), int(chk["cross_engine_count_rows"]), 3)
    t0 = time.time()
    counts, n_events, got = count_table(eng_g, engine_kind, cnt_pos, tok_tr[cnt_pos], C, V, chunk, zero, check_cnt)
    rec.m(f"{pre}.count_pass_wall_s", time.time() - t0)
    rec.m(f"{pre}.counts_events", int(n_events))
    fptr, fids = fit_rows_fired(eng_g, engine_kind, fit_pos, zero, chunk)
    del eng_g
    if engine_kind == "cuda":
        eng_c = R.make_engine("cpu", cfg, prefix)
        eng_c.load_state(st)
        bad = sum(int(not np.array_equal(api_fired_lists(eng_c, cnt_pos[[r]], zero)[1], got[r])) for r in check_cnt.tolist())
        check_fit = T1.sample_positions(np.arange(fit_pos.shape[0]), int(chk["cross_engine_fit_rows"]), 4)
        bad += sum(int(not np.array_equal(api_fired_lists(eng_c, fit_pos[[r]], zero)[1], fids[fptr[r]:fptr[r + 1]])) for r in check_fit.tolist())
        rec.check(f"{pre}.cuda_fired_lists_equal_cpu_engine", bad == 0, mismatches=bad, rows=int(check_cnt.shape[0] + check_fit.shape[0]))
        del eng_c
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    # ---- tables and read-out constants
    tab = Tables(counts, n_inc, tok_tr[cnt_pos], V)
    del counts
    rec.check(f"{pre}.counted_events_equal_fired_events", int(tab.N.sum()) == int(n_events), counted=int(tab.N.sum()), fired=int(n_events))
    rec.m(f"{pre}.clauses_with_counts", int(tab.has.sum()))
    rec.m(f"{pre}.distinct_pairs", int(tab.idx.shape[0]))
    fit_rows = tab.rows(fptr, fids, tok_tr[fit_pos])
    val_rows = tab.rows(vptr, vids, gold)
    rec.m(f"{pre}.fit_rows", int(fit_rows["n"]))
    rec.m(f"{pre}.fit_counted_fired_per_row", float(fit_rows["comp"].shape[0] / max(1, fit_rows["n"])))
    rec.m(f"{pre}.validation_counted_fired_per_row", float(val_rows["comp"].shape[0] / max(1, val_rows["n"])))
    rec.m(f"{pre}.validation_frac_rows_without_counted_clause", float((np.diff(val_rows["rp"]) == 0).mean()) if val_rows["n"] else None)
    # ---- fit (GPU objective as H8.1 on CUDA; CPU reference objective otherwise), CPU float64 check at the solution
    budget = dict(qp["fit"])
    t1 = None
    if engine_kind == "cuda":
        from tmlm.common import locks
        import cupy as cp
        locks.yield_gpu_if_requested()
        cp.get_default_memory_pool().set_limit(size=int(float(qp["gpu_pool_limit_gib"]) * (1 << 30)))
        t1 = T1.T1Gpu(V, tab.indptr, tab.idx, tab.val, tab.lam, tab.inv_n, tab.p_uni)
        alpha, beta0, info = T1.loglin_fit_gpu(t1, fit_rows, budget, rec, f"{name} loglin")
        ll, _, _, _, _, _ = tab.loglin(fit_rows, alpha, beta0, False)
        info["objective_cpu_float64"] = float(-ll / max(1, fit_rows["n"]))
        info["abs_diff_objective_gpu_vs_cpu"] = abs(info["objective_cpu_float64"] - info["objective_gpu_recomputed"])
        rec.check(f"{pre}.fit_objective_gpu_equals_cpu", info["abs_diff_objective_gpu_vs_cpu"] < 1e-6, diff=info["abs_diff_objective_gpu_vs_cpu"])
    else:
        alpha, beta0, info = loglin_fit_cpu(tab, fit_rows, budget)
    rec.m(f"{pre}.fit", info)
    # ---- validation scoring (CPU float64 reference kernel; GPU cross-check on CUDA)
    _, _, _, pg, rank, maxerr = tab.loglin(val_rows, alpha, beta0, False)
    rec.check(f"{pre}.rows_sum_to_one", bool(maxerr < 1e-9 and np.all(pg > 0)), max_row_sum_error=float(maxerr))
    if t1 is not None:
        vch = t1.prepare(val_rows["rp"], val_rows["comp"], val_rows["bkt"], val_rows["gold"])
        og = t1.run(vch, val_rows["n"], int(val_rows["comp"].shape[0]), alpha, beta0, False)
        gdiff = float(np.abs(og["lpg"] - np.log(pg)).max()) if val_rows["n"] else 0.0
        rec.check(f"{pre}.validation_gpu_equals_cpu", gdiff < 1e-6, max_abs_logprob_diff=gdiff, rank_mismatch_rows=int((og["rank"] != rank).sum()))
        del t1, vch
    row = {"loglin_top1": float((rank == 0).mean()), "loglin_top5": float((rank < 5).mean()),
           "loglin_bits_per_token": float(-np.log(pg).mean() / math.log(2.0)), "signed_top1": signed_top1,
           "n_targets": int(targets.shape[0]), "alive": int(np.asarray(st["alive"]).sum())}
    for k, v in row.items():
        rec.m(f"{pre}.{k}", v, note="Q_pred = loglin_top1 (T1, docs/11 11.5); signed_top1 = the bank's own vote sums" if k == "loglin_top1" else "")
    np.savez_compressed(out / f"qpred_{name}.npz", targets=targets, gold=gold, loglin_rank=rank.astype(np.int32),
                        loglin_gold_prob=pg, signed_correct=signed_correct)
    (out / f"qpred_{name}_readout.json").write_text(json.dumps({"alpha_by_group": {int(g): float(alpha[g]) for g in np.flatnonzero(alpha)},
                                                                "beta0": float(beta0)}, indent=1) + "\n")
    rec.log(f"qpred {name}: loglin top-1 {row['loglin_top1']:.4f} top-5 {row['loglin_top5']:.4f}, signed top-1 {signed_top1:.4f}")
    rec.flush()
    return row


def qpred_split(prefix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fit document mask over the prefix documents, fit positions, counting positions) of docs/11 11.6."""
    import cvtm_t1_readout as T1

    ds = np.asarray(prefix.doc_start, np.int64)
    n_doc = ds.shape[0] - 1
    fit_doc = T1.fit_document_mask(n_doc)
    doc_of = np.repeat(np.arange(n_doc), np.diff(ds))
    is_fit = fit_doc[doc_of]
    return fit_doc, np.flatnonzero(is_fit).astype(np.int64), np.flatnonzero(~is_fit).astype(np.int64)


def stage_qpred(args, cj: dict, rec: R.Rec, train_run: Path) -> Path:
    from tmlm.data.corpus_store import load_split, split_dir
    from tmlm.data.evalsets import load_evalset, test_access_count

    p = cj["params"]
    qp = p["qpred"]
    h81 = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "H8_1_masked.json").read_text())["fit"]["loglin"]
    rec.check("qpred.fit_budget_equals_H8_1", qp["fit"] == h81)
    out = out_dir("qpred")
    ta0 = test_access_count()
    rec.m("qpred.test_accesses_before", ta0)
    arm, seed = train_run_identity(train_run)
    base, _ = CONTROL_ARMS[arm]
    names = resolve_names(args.checkpoints, train_run)
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    rec.m("qpred.train_run", str(train_run))
    rec.m("qpred.checkpoints", names)
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    prefix_t = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    V = int(p["vocab"])
    val0 = load_split(CORPUS, V, "validation")
    val = P.with_tables(val0, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    ev = load_evalset(CORPUS, V, "validation", "subword")
    targets = np.asarray(ev["targets"], np.int64)
    hidden = np.asarray(ev["hidden"], np.uint8)
    meta = json.loads((split_dir(CORPUS, V, "validation") / "meta.json").read_text())
    R.dataset_hashes({f"{CORPUS}_bpe{V}_validation_tok": meta["sha256"]["tok.npy"],
                      f"evalset_validation_subword_V{V}": ev["meta"]["sha256_combined"]})
    rec.m("qpred.validation_targets", int(targets.shape[0]))
    rec.m("qpred.evalset_sha256_combined", ev["meta"]["sha256_combined"])
    fit_doc, fit_pos, cnt_pos = qpred_split(prefix)
    rec.m("qpred.fit_documents", np.flatnonzero(fit_doc).tolist(), rule="splitmix64(i XOR 0xC0FFEE) mod 100 == 0 (docs/11 11.6), prefix documents")
    rec.m("qpred.fit_positions", int(fit_pos.shape[0]))
    rec.m("qpred.counting_positions", int(cnt_pos.shape[0]))
    rec.flush()
    results = {}
    for name in names:
        cfg, st, _ = load_named_checkpoint(train_run, name, arm, seed, p, rec)
        results[name] = qpred_checkpoint(name, cfg, st, prefix_t, val, targets, hidden, fit_pos, cnt_pos, qp, args.engine,
                                         int(p["forward_chunk"]), rec, out)
    ta1 = test_access_count()
    rec.m("qpred.test_accesses_after", ta1)
    rec.check("qpred.no_test_access", ta1 == ta0, before=ta0, after=ta1)
    rec.m("qpred.summary", results)
    rec.flush()
    return out


# ------------------------------------------------------------------ report (the reading rule, mechanically)
def read_rows(probe_runs: list[Path]) -> list[dict]:
    rows = []
    for d in probe_runs:
        with open(d / "cells.jsonl", encoding="utf-8") as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    return rows


def ladder_set_names(q: dict, arm: str, seed: int, code: str, budget: str) -> list[str]:
    return sorted({k[2] for k in q if k[0] == arm and k[1] == seed and k[3] == "lr" and k[4] == budget
                   and k[2].endswith("." + code) and (k[2].startswith("young_b") or k[2].startswith("c0y."))})


def components(q: dict, qpred: dict, arm: str, seed: int, code: str, budget: str = "32") -> dict:
    """d_ladder, d_twin (points) and Q_pred of bank and twin for one base arm, seed and code (None where missing)."""
    mean = (lambda d: float(np.mean([d[k] for k in sorted(d)])) if d else None)
    bank = q.get((arm, seed, f"epoch4.{code}", "lr", budget))
    twin = q.get((arm + "T", seed, f"epoch4.{code}", "lr", budget))
    rungs = {s: q[(arm, seed, s, "lr", budget)] for s in ladder_set_names(q, arm, seed, code, budget)}
    best = max(rungs, key=lambda s: mean(rungs[s])) if rungs else None
    out: dict[str, Any] = {"bank": mean(bank), "twin": mean(twin), "max_rung": best, "max_rung_q": mean(rungs[best]) if best else None,
                           "rungs": {s: mean(v) for s, v in rungs.items()}}
    out["d_ladder"] = 100.0 * (out["bank"] - out["max_rung_q"]) if bank and best else None
    out["d_twin"] = 100.0 * (out["bank"] - out["twin"]) if bank and twin else None
    if bank and best:
        common = sorted(set(bank) & set(rungs[best]))
        out["d_ladder_bootstrap"] = PR.paired_bootstrap(100.0 * np.array([bank[d] - rungs[best][d] for d in common]))
    if bank and twin:
        common = sorted(set(bank) & set(twin))
        out["d_twin_bootstrap"] = PR.paired_bootstrap(100.0 * np.array([bank[d] - twin[d] for d in common]))
    out["q_pred_bank"] = qpred.get((arm, seed, "epoch4"))
    out["q_pred_twin"] = qpred.get((arm + "T", seed, "epoch4"))
    return out


def reading(comp: dict[tuple[int, str], dict], seeds: list[int]) -> str:
    """The rows of the config's reading rule (first that holds). comp[(seed, code)] = components(...)."""
    cells = [comp.get((s, c)) for s in seeds for c in CODES]
    if any(x is None or x["d_ladder"] is None or x["d_twin"] is None for x in cells):
        return "INCOMPLETE"
    qp_ok = all(comp[(s, "all")]["q_pred_bank"] is not None and comp[(s, "all")]["q_pred_twin"] is not None for s in seeds)
    if all(x["d_ladder"] >= 1.0 and x["d_twin"] >= 1.0 for x in cells):
        if not qp_ok:
            return "INCOMPLETE"
        if all(comp[(s, "all")]["q_pred_bank"] >= comp[(s, "all")]["q_pred_twin"] for s in seeds):
            return "1 LITERAL LEARNING CREDITED"
    if all(x["d_ladder"] >= 1.0 and abs(x["d_twin"]) < 1.0 for x in cells):
        return "2 GAIN FROM CONTINUED SEEDING, SELECTION OR VOTES, NOT FROM LOGIC"
    if all(x["d_ladder"] < 0.0 for x in cells):
        return "3 TRAINING DOES NOT ADD VALUE"
    return "4 MIXED"


def twin_above_bank(comp: dict[tuple[int, str], dict], seeds: list[int]) -> bool | None:
    """Reported flag, not a reading row: d_twin <= -1.0 on both seeds and both codes."""
    cells = [comp.get((s, c)) for s in seeds for c in CODES]
    if any(x is None or x["d_twin"] is None for x in cells):
        return None
    return all(x["d_twin"] <= -1.0 for x in cells)


def stage_report(args, cj: dict, rec: R.Rec) -> Path:
    p = cj["params"]
    out = out_dir("report")
    rows = read_rows([R.rp(d) for d in args.probe_runs.split(",") if d.strip()])
    q = R.summarise(rows, list(p["tasks"]))
    qpred: dict = {}
    for d in [R.rp(x) for x in (args.qpred_runs or "").split(",") if x.strip()]:
        m = json.loads((d / "metrics.json").read_text())
        arm, seed = m["control.arm"]["value"], int(m["control.seed"]["value"])
        for k, v in m.items():
            if k.startswith("qpred.") and k.endswith(".loglin_top1"):
                qpred[(arm, seed, k[len("qpred."):-len(".loglin_top1")])] = v["value"]
    seeds = [int(s) for s in cj["reading_rule"]["seeds"]]
    verdicts = {}
    for arm in ("A3", "A0"):
        comp = {(s, c): components(q, qpred, arm, s, c) for s in seeds for c in CODES}
        for (s, c), v in comp.items():
            rec.m(f"report.{arm}.seed{s}.{c}", v, label="derived", note="Q_text points at 32 labels per class, LR; Q_pred T1 loglin top-1")
        verdicts[arm] = reading(comp, seeds)
        rec.m(f"report.{arm}.reading", verdicts[arm], label="derived", rule="experiments/cvtm/configs/ssl_r1_controls.json reading_rule")
        rec.m(f"report.{arm}.flag_twin_above_bank", twin_above_bank(comp, seeds), label="derived",
              note="reported flag, not a reading row: d_twin <= -1.0 on both seeds and both codes")
    for budget in ("8", "128"):
        for arm in ("A3", "A0"):
            for s in seeds:
                for c in CODES:
                    rec.m(f"report.{arm}.seed{s}.{c}.budget{budget}", components(q, qpred, arm, s, c, budget), label="derived")
    rec.m("report.q_pred", {f"{a}.{s}.{n}": v for (a, s, n), v in sorted(qpred.items())})
    (out / "reading.json").write_text(json.dumps(verdicts, indent=1) + "\n")
    rec.flush()
    return out


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "census", "features", "qpred", "probe", "report"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--arm", default=None, choices=sorted(CONTROL_ARMS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--engine", default="cuda", choices=["cuda", "cpu", "oracle"])
    ap.add_argument("--train-run", default=None)
    ap.add_argument("--features-run", default=None)
    ap.add_argument("--checkpoints", default="c0y,ladder,epoch1,epoch4")
    ap.add_argument("--sets", default="auto")
    ap.add_argument("--budgets", default="8,32,128")
    ap.add_argument("--probe-runs", default=None)
    ap.add_argument("--qpred-runs", default=None)
    args = ap.parse_args(argv)
    cj = load_config(args.config)
    p = cj["params"]
    if p.get("smoke", False):
        raise SystemExit("this study has no smoke config")
    rec = R.Rec(False)
    rec.m("stage", args.stage)
    rec.m("config_params", p)
    check_shared_params(cj, rec)
    if args.stage == "train":
        if args.arm is None or args.seed is None:
            raise SystemExit("--stage train needs --arm and --seed")
        if int(args.seed) not in [int(s) for s in p["seeds"]]:
            raise SystemExit(f"seed {args.seed} is not a seed of this study {p['seeds']}")
        stage_train(args, cj, rec)
    elif args.stage == "census":
        stage_census(args, cj, rec, R.rp(args.train_run))
    elif args.stage == "features":
        stage_features(args, cj, rec, R.rp(args.train_run))
    elif args.stage == "qpred":
        stage_qpred(args, cj, rec, R.rp(args.train_run))
    elif args.stage == "probe":
        stage_probe(args, cj, rec, R.rp(args.features_run))
    else:
        stage_report(args, cj, rec)
    rec.m("wall_total_s", time.time() - T0)
    rec.m("failed_checks", rec.failed)
    rec.flush()
    rec.log(f"stage {args.stage} done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
