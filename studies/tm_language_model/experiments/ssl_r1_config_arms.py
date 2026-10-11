"""A8 ITEM 2 CONFIGURATION ARMS (exploratory; config experiments/cvtm/configs/ssl_r1_config_arms.json): does a
docs/11 bank trained on R1 arm A3's data represent text better with the settings the literature puts near the text
optimum than with the R1 settings (s = 128, budget 12, negated literals allowed), which were tuned for the tiny-LM gate
KB2? The arms inform the DESIGN only: no A7 threshold, no A8 item 1 label and no reading rule is applied to them.

Arms (all on A3's data: F2, at most 2,000 targets per class per epoch, redrawn per epoch; seeds 100, then 101):
    A3s8    A3 with s = 8
    A3s32   A3 with s = 32
    A3pos   A3 with budget = 3, p_neg = 0, neg_budget = 0 (positive short clauses, the slide-11 setting)
Each gets the young ladder, census, features (c0y, ladder, epoch1, epoch4), Q_pred and the LR probe (budgets 8, 32,
128) of experiments/ssl_r1_controls.py, by ITS stage functions (imported as the module C; nothing of it is edited).

Twin: the literal-frozen twin A3T of the control study serves as the twin of all three arms. A twin is the arm's
EngineConfig with p_ins = 0, p_neg = 0, s = 2^62, so its clauses keep their seed literals whatever the arm's s is.
For A3s8 and A3s32 the twin EngineConfig equals A3T's exactly. For A3pos it differs only in budget and neg_budget,
which the engines read only at a 127 -> 128 literal-state crossing (oracle _state_up, tm_core.h lines 348 and 349)
that a twin never reaches (seeds enter at 136, nothing is inserted, no state decreases); `twin_relation` states and
checks this in every train run, tests/test_ssl_r1_config_arms.py checks it bit for bit on tiny runs.

Registration: the arms are added to tmlm.ssl.pretext.ARMS (an Arm with A3's data flags plus the engine overrides,
exactly how A2 carries F3 and F4) and to ssl_r1_controls.CONTROL_ARMS (not a twin) by `register(cj)`, which `main`
calls before any stage. It is deliberately NOT done at import: tests/test_ssl_r1_pretext.py asserts the R1 arm list,
and pytest imports every test module (and so this one) before running any test. The arm definitions live in the
config (params.config_arms), the file run_logged.py hashes.

Stages (each its own logged run):
  train     (GPU) --arm {A3s8,A3s32,A3pos} --seed S: checks that the overrides reach the EngineConfig and nothing else
            changes, that the data equal A3's, that A3T serves as the twin; then ssl_r1_controls.stage_train (the R1
            loop with the young ladder); then, per checkpoint, that its config equals the arm's EngineConfig
            (C.arm_config) and the literal caps of that intended config (at most `budget` included literals per
            clause, at most `neg_budget` negated ones, no negated literal when p_neg = 0).
  census    (CPU) --train-run R: ssl_r1_controls.stage_census plus the A8 item 6 diagnostics per checkpoint (literal
            histogram, negated and bag shares, owned clauses per class against T, net include-boundary crossings
            between checkpoints, class-sum histograms on a fixed validation sample). Also accepted for the control
            study's A3 and A3T train runs (the same diagnostics for the reference bank and the twin).
  features  (GPU) --train-run R --checkpoints c0y,ladder,epoch1,epoch4: ssl_r1_controls.stage_features.
  qpred     (GPU) --train-run R --checkpoints c0y,epoch4: ssl_r1_controls.stage_qpred.
  probe     (CPU) --features-run F --sets auto --budgets 8,32,128: ssl_r1_controls.stage_probe.
  report    (CPU) --probe-runs a,b,... --qpred-runs c,d,...: the descriptive comparisons of the config (no verdict):
            against the arm's own young ladder, the twin A3T and the R1-settings bank A3 of the control study.

Launch: the chain chain_config_arms.sh (CPU slot first, GPU lock inside each step).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for _q in (ROOT / "src", EXPERIMENTS):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_controls as C  # noqa: E402  (imported, never edited)
import ssl_r1_pretext as R  # noqa: E402  (imported, never edited)
from tmlm.engine.config import CLS_EMPTY, INCLUDE_THRESHOLD, LIT_EMPTY, EngineConfig  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402  (imported, never edited)
from tmlm.ssl import probes as PR  # noqa: E402

T0 = time.time()
DEFAULT_CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_config_arms.json"
DATA_ARM = "A3"     # every configuration arm trains on A3's data
TWIN_ARM = "A3T"    # the control study's literal-frozen twin of A3, the twin of every configuration arm
REF_ARM = "A3"      # the control study's A3 bank (R1 settings), the reference of the report
DIAG_ARMS = (REF_ARM, TWIN_ARM)  # control-study runs the census stage also accepts
MAX_SEED_LITERALS = 3  # docs/11 5.6: a seed clause gets 1 to 3 literals (oracle _seed_literals: n_lit in {1, 2, 3})
CROSSING_ONLY_FIELDS = ("budget", "neg_budget")  # read only at a 127 -> 128 crossing (oracle _state_up, tm_core.h 348-349)
EXTRA_PARAMS = ("config_arms", "diagnostics")  # params of this study that the control study does not have
COPIED_FROM_CONTROLS = ("r1_config", "params_shared_with_r1", "doc_set_reference")
CODES = C.CODES
STAGES = ("train", "census", "features", "qpred", "probe", "report")


# ------------------------------------------------------------------ arms and registration
_FIELD_TYPES = {f.name: str(f.type) for f in dataclasses.fields(EngineConfig)}
_CASTS = {"float": float, "int": int, "bool": bool, "str": str}


def typed_override(key: str, value: Any) -> Any:
    """An engine override from the config, cast to its EngineConfig field type (JSON has no int/float split)."""
    if key not in _FIELD_TYPES:
        raise SystemExit(f"engine override {key!r} is not an EngineConfig field")
    cast = _CASTS[_FIELD_TYPES[key]]
    if cast is int and (isinstance(value, bool) or float(value) != int(value)):
        raise SystemExit(f"engine override {key!r} must be an integer, got {value!r}")
    return cast(value)


def overrides_of(cj: dict, arm: str) -> dict[str, Any]:
    return {k: typed_override(k, v) for k, v in sorted(cj["params"]["config_arms"][arm]["engine_overrides"].items())}


def arm_entries(cj: dict) -> dict[str, P.Arm]:
    """name -> P.Arm: A3's data flags (mode, F1, F2, kind_map) plus the arm's engine overrides."""
    base = P.ARMS[DATA_ARM]
    out: dict[str, P.Arm] = {}
    for name, spec in cj["params"]["config_arms"].items():
        if spec["data_arm"] != DATA_ARM:
            raise SystemExit(f"{name}: data_arm must be {DATA_ARM}")
        if name.endswith("T"):
            raise SystemExit(f"{name}: a name ending in T means a twin in ssl_r1_controls")
        ov = tuple(overrides_of(cj, name).items())
        out[name] = P.Arm(name, base.mode, f1=base.f1, f2=base.f2, random_kind_map=base.random_kind_map,
                          engine=tuple(base.engine) + ov, description=str(spec["description"]))
    return out


def register(cj: dict) -> dict[str, P.Arm]:
    """Adds the configuration arms to P.ARMS and C.CONTROL_ARMS (idempotent; refuses a conflicting entry)."""
    arms = arm_entries(cj)
    for name, arm in arms.items():
        if name in P.ARMS and P.ARMS[name] != arm:
            raise SystemExit(f"P.ARMS already holds a different arm {name!r}")
        if name in C.CONTROL_ARMS and C.CONTROL_ARMS[name] != (name, False):
            raise SystemExit(f"CONTROL_ARMS already holds a different entry {name!r}")
    for name, arm in arms.items():
        P.ARMS[name] = arm
        C.CONTROL_ARMS[name] = (name, False)
    return arms


def check_params_equal_controls(cj: dict, rec: R.Rec | None = None) -> None:
    """params (without this study's extras) and the copied blocks must equal the control study's."""
    ctl = C.load_config(ROOT / cj["controls_config"])
    mine = {k: v for k, v in cj["params"].items() if k not in EXTRA_PARAMS}
    bad = sorted(k for k in set(ctl["params"]) | set(mine) if ctl["params"].get(k) != mine.get(k))
    bad += [k for k in COPIED_FROM_CONTROLS if ctl.get(k) != cj.get(k)]
    if rec is not None:
        rec.check("config.params_equal_control_study", not bad, differing=bad)
    if bad:
        raise SystemExit(f"params differ from the control study's config: {bad}")


def override_diff(arm: str, seed: int, p: dict, **extra) -> dict[str, list]:
    """Fields where the arm's EngineConfig differs from A3's: {field: [arm value, A3 value]}."""
    a, b = C.arm_config(arm, seed, p, **extra).to_dict(), C.arm_config(DATA_ARM, seed, p, **extra).to_dict()
    return {k: [a[k], b[k]] for k in a if a[k] != b[k]}


def twin_config_of(arm: str, seed: int, p: dict, **extra) -> EngineConfig:
    """The arm's EngineConfig with the control study's twin overrides (p_ins = 0, p_neg = 0, s = 2^62)."""
    kw = dict(C.twin_overrides(p))
    kw.update(extra)
    vocab = kw.pop("vocab", int(p["vocab"]))
    n_clauses = kw.pop("n_clauses", int(p["n_clauses"]))
    return P.engine_config(P.ARMS[arm], int(seed), int(vocab), int(n_clauses), **kw)


def twin_relation(arm: str, seed: int, p: dict, **extra) -> dict:
    """Why A3T is the twin of `arm`: exact (identical EngineConfig) or equivalent (only fields that a twin never
    reads differ). `equivalent` is the checked condition."""
    tw = twin_config_of(arm, seed, p, **extra)
    ref = C.arm_config(TWIN_ARM, seed, p, **extra)
    a, b = tw.to_dict(), ref.to_dict()
    diff = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
    zero = C.twin_thresholds_ok(tw, C.arm_config(arm, seed, p, **extra)) and C.twin_thresholds_ok(ref, C.arm_config(DATA_ARM, seed, p, **extra))
    unreached = (set(diff) <= set(CROSSING_ONLY_FIELDS) and not tw.dense_equivalent and not ref.dense_equivalent
                 and min(tw.budget, ref.budget) >= MAX_SEED_LITERALS)
    return {"twin_arm": TWIN_ARM, "exact": not diff, "differing_fields": diff, "literal_thresholds_zero": bool(zero),
            "equivalent": bool(zero and (not diff or unreached)),
            "argument": ("identical EngineConfig" if not diff else
                         "budget and neg_budget are read only at a 127 -> 128 literal-state crossing (oracle _state_up, "
                         "tm_core.h lines 348-349); in a twin seeds enter at 136, nothing is inserted and no state "
                         f"decreases, so no crossing happens; seeds have at most {MAX_SEED_LITERALS} literals <= budget")}


def literal_caps(st: dict, cfg: EngineConfig) -> dict:
    """Included literals per alive clause (max), included negated (max), tracked negated literals, and the caps."""
    alive = np.asarray(st["alive"]).astype(bool)
    lid = np.asarray(st["lit_id"])[alive]
    ls = np.asarray(st["lit_state"])[alive]
    tracked = lid != LIT_EMPTY
    neg = tracked & ((lid & 1) == 1)
    inc = tracked & (ls >= INCLUDE_THRESHOLD)
    n_inc = inc.sum(axis=1)
    n_neg = (inc & neg).sum(axis=1)
    return {"alive": int(alive.sum()), "max_included": int(n_inc.max()) if n_inc.size else 0,
            "clauses_at_budget": int((n_inc == cfg.budget).sum()), "max_included_negated": int(n_neg.max()) if n_neg.size else 0,
            "tracked_negated": int(neg.sum()), "budget": int(cfg.budget), "neg_budget": int(cfg.neg_budget), "p_neg": float(cfg.p_neg)}


def caps_ok(c: dict) -> bool:
    ok = c["max_included"] <= c["budget"]
    if c["neg_budget"] >= 0:
        ok = ok and c["max_included_negated"] <= c["neg_budget"]
    if c["p_neg"] == 0.0:
        ok = ok and c["tracked_negated"] == 0
    return bool(ok)


# ------------------------------------------------------------------ train
def data_check(arm: str, prefix, p: dict, rec: R.Rec) -> bool:
    """The arm's data (flags, tables, epoch-1 example set) equal A3's."""
    a = P.arm_data(arm, prefix, int(p["vocab"]), float(p["f1_t"]))
    b = P.arm_data(DATA_ARM, prefix, int(p["vocab"]), float(p["f1_t"]))
    fa, fb = (a.arm.mode, a.arm.f1, a.arm.f2, a.arm.random_kind_map), (b.arm.mode, b.arm.f1, b.arm.f2, b.arm.random_kind_map)
    ok = (fa == fb and np.array_equal(a.kind_map, b.kind_map) and np.array_equal(a.sym_freq, b.sym_freq)
          and a.info == b.info and np.array_equal(a.targets(0), b.targets(0)))
    rec.check("config_arm.data_equal_A3", ok, flags=list(fa), flags_A3=list(fb), epoch1_targets=int(a.targets(0).shape[0]))
    return bool(ok)


def stage_train(args, cj: dict, rec: R.Rec) -> None:
    from tmlm.engine.checkpoint import load_checkpoint

    p = cj["params"]
    arm, seed = args.arm, int(args.seed)
    ov = overrides_of(cj, arm)
    rec.m("config_arm.name", arm)
    rec.m("config_arm.data_arm", DATA_ARM)
    rec.m("config_arm.engine_overrides", ov)
    diff = override_diff(arm, seed, p)
    rec.check("config_arm.overrides_reach_engine_config", set(diff) == set(ov) and all(diff[k][0] == v for k, v in ov.items()),
              differing_from_A3=diff)
    rel = twin_relation(arm, seed, p)
    rec.m("twin.relation", rel)
    rec.check("twin.A3T_serves_as_twin", rel["equivalent"], exact=rel["exact"])
    prefix = R.load_prefix(p, rec)
    data_check(arm, prefix, p, rec)
    del prefix
    rec.flush()
    C.stage_train(args, cj, rec)
    run = C.out_dir("train")
    exp = C.arm_config(arm, seed, p)  # the arm's intended EngineConfig: the caps are checked against ITS budget
    ed = exp.to_dict()
    for name in C.checkpoint_names(run):
        cfg, st, _ = load_checkpoint(run / f"checkpoint_{name}.npz")
        same = cfg == exp
        rec.check(f"config_arm.checkpoint_config_equals_arm.{name}", same,
                  differing={k: [v, ed[k]] for k, v in cfg.to_dict().items() if v != ed[k]})
        c = literal_caps(st, exp)
        rec.check(f"config_arm.literal_caps.{name}", same and caps_ok(c), **c)
    rec.flush()


# ------------------------------------------------------------------ A8 item 6 diagnostics
def e_bound(T: int) -> int:
    """The smallest m with floor(m / e) >= T (docs/11 section 8 item 5)."""
    m = int(math.ceil(T * math.e))
    while math.floor(m / math.e) < T:
        m += 1
    while m > 0 and math.floor((m - 1) / math.e) >= T:
        m -= 1
    return m


def owned_per_class(st: dict, vocab: int, T: int, class_share: np.ndarray | None = None) -> dict:
    """m_c = number of alive clauses owning class c (vote slot for c with w > 0), against T."""
    alive = np.asarray(st["alive"]).astype(bool)
    vc = np.asarray(st["vote_cls"])[alive]
    vw = np.asarray(st["vote_w"])[alive]
    own = (vc != CLS_EMPTY) & (vw > 0)
    m = np.bincount(vc[own].astype(np.int64), minlength=int(vocab))[: int(vocab)]
    b2, be = 2 * int(T), e_bound(int(T))
    lg = np.where(m == 0, 0, np.floor(np.log2(np.maximum(m, 1))).astype(np.int64) + 1)
    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    out: dict[str, Any] = {
        "T": int(T), "bound_2T": b2, "bound_eT": be, "owned_total": int(m.sum()), "max": int(m.max()) if m.size else 0,
        "quantiles": dict(zip([str(v) for v in q], np.quantile(m, q).tolist())),
        "frac_classes_zero": float((m == 0).mean()), "frac_classes_ge_T": float((m >= T).mean()),
        "frac_classes_ge_2T": float((m >= b2).mean()), "frac_classes_ge_eT": float((m >= be).mean()),
        "hist_log2": np.bincount(lg, minlength=18).tolist(),
        "hist_log2_bins": "bin 0: m = 0; bin k >= 1: 2^(k-1) <= m < 2^k"}
    if class_share is not None:
        w = np.asarray(class_share, np.float64)[: int(vocab)]
        out.update({"target_share_zero": float(w[m == 0].sum()), "target_share_ge_T": float(w[m >= T].sum()),
                    "target_share_ge_2T": float(w[m >= b2].sum()), "target_share_ge_eT": float(w[m >= be].sum())})
    return out


def _lit_keys(ids: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, k = np.nonzero(ids != LIT_EMPTY)
    key = (r.astype(np.uint64) << np.uint64(32)) | ids[r, k].astype(np.uint64)
    inc = states[r, k] >= INCLUDE_THRESHOLD
    o = np.argsort(key, kind="stable")
    return key[o], inc[o]


def include_crossings(prev: dict | None, cur: dict) -> dict:
    """Net include-boundary crossings between two literal views (C.literal_view) over clause incarnations alive at
    both (same birth step), matched by literal id. A lower bound on crossing transitions (the engine counts none)."""
    if prev is None:
        return {"n_common": 0, "note": "no previous checkpoint"}
    both = np.flatnonzero(prev["alive"] & cur["alive"] & (prev["birth"] == cur["birth"]))
    pk, pinc = _lit_keys(prev["lit_id"][both], prev["lit_state"][both])
    ck, cinc = _lit_keys(cur["lit_id"][both], cur["lit_state"][both])
    _, pi, ci = np.intersect1d(pk, ck, assume_unique=True, return_indices=True)
    up = int((~pinc[pi] & cinc[ci]).sum())
    down = int((pinc[pi] & ~cinc[ci]).sum())
    gone = np.ones(pk.shape[0], bool)
    gone[pi] = False
    new = np.ones(ck.shape[0], bool)
    new[ci] = False
    freed_inc = int((gone & pinc).sum())    # an included literal leaves only by decaying to 0: crossed down
    new_inc = int((new & cinc).sum())       # inserted (at 123) after `prev` and included now: crossed up
    n = up + down + freed_inc + new_inc
    return {"n_common": int(both.shape[0]), "tracked_prev": int(pk.shape[0]), "included_prev": int(pinc.sum()),
            "tracked_cur": int(ck.shape[0]), "included_cur": int(cinc.sum()),
            "same_literal_up": up, "same_literal_down": down, "included_then_freed": freed_inc,
            "inserted_then_included": new_inc, "excluded_then_gone": int((gone & ~pinc).sum()),
            "inserted_still_excluded": int((new & ~cinc).sum()), "net_crossings": n,
            "net_crossings_per_tracked_prev": float(n / pk.shape[0]) if pk.shape[0] else None}


def class_sum_stats(eng, cfg: EngineConfig, st: dict, rows: np.ndarray, gold: np.ndarray, hidden: np.ndarray,
                    bin_w: int, chunk: int) -> tuple[dict, bool]:
    """Class-sum histograms of `rows` (positions of the engine's corpus) under `hidden`; the second value says
    whether S_y rebuilt from the fired lists equals predict_scores for every row."""
    import cvtm_t1_readout as T1

    T = int(cfg.T)
    n = int(rows.shape[0])
    S = eng.predict_scores(T1.EPOCH, rows.astype(np.uint32), hidden).astype(np.int64)
    ar = np.arange(n)
    gold = np.asarray(gold, np.int64)
    sy = S[ar, gold].copy()
    n_pos = (S > 0).sum(axis=1)
    edges = np.append(np.arange(-T, T + 1, int(bin_w)), T + 1)
    all_hist = np.histogram(np.clip(S, -T, T), bins=edges)[0]
    S[ar, gold] = np.iinfo(np.int64).min
    top = S.max(axis=1)
    del S
    vc, vw = np.asarray(st["vote_cls"]), np.asarray(st["vote_w"]).astype(np.int64)
    fired = np.zeros(n, np.int64)
    cover = np.zeros(n, np.int64)
    sy_re = np.zeros(n, np.int64)
    for lo in range(0, n, int(chunk)):
        hi = min(n, lo + int(chunk))
        ptr, ids = C.api_fired_lists(eng, rows[lo:hi].astype(np.int64), hidden)
        rid = np.repeat(np.arange(hi - lo), np.diff(ptr))
        match = vc[ids] == gold[lo:hi][rid][:, None]
        w_y = (vw[ids] * match).sum(axis=1)
        owns_y = (match & (vw[ids] > 0)).any(axis=1)
        fired[lo:hi] = np.diff(ptr)
        sy_re[lo:hi] = np.bincount(rid, weights=w_y, minlength=hi - lo).round().astype(np.int64)
        cover[lo:hi] = np.bincount(rid, weights=owns_y, minlength=hi - lo).round().astype(np.int64)
    hist = (lambda v: np.histogram(np.clip(v, -T, T), bins=edges)[0].tolist())
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    out = {"rows": n, "T": T, "bin_edges": edges.tolist(), "bins": "[edge_i, edge_i+1), values clamped to [-T, T]",
           "target_sum_hist": hist(sy), "top_other_sum_hist": hist(top), "all_class_sums_hist": all_hist.tolist(),
           "target_sum_quantiles": dict(zip([str(v) for v in q], np.quantile(sy, q).tolist())) if n else None,
           "frac_target_sum_ge_T": float((sy >= T).mean()) if n else None,
           "frac_target_sum_le_0": float((sy <= 0).mean()) if n else None,
           "frac_target_sum_le_minus_T": float((sy <= -T).mean()) if n else None,
           "frac_target_above_every_other": float((sy > top).mean()) if n else None,
           "frac_rows_top_other_gt_0": float((top > 0).mean()) if n else None,
           "positive_class_sums_per_row_mean": float(n_pos.mean()) if n else None,
           "fired_per_row_mean": float(fired.mean()) if n else None,
           "fired_per_row_quantiles": dict(zip([str(v) for v in q], np.quantile(fired, q).tolist())) if n else None,
           "cover_mean": float(cover.mean()) if n else None, "k_cover": int(cfg.k_cover),
           "frac_cover_ge_k_cover": float((cover >= cfg.k_cover).mean()) if n else None,
           "frac_cover_zero": float((cover == 0).mean()) if n else None}
    return out, bool(np.array_equal(sy_re, sy))


def epoch_feedback_events(train_run: Path) -> dict[int, int]:
    """Type I plus Type II feedback events per epoch, from the train run's metrics (missing epochs left out)."""
    out: dict[int, int] = {}
    for k in range(1, 64):
        try:
            out[k] = int(R.read_metric(train_run, f"train.epoch{k}.n_type1")) + int(R.read_metric(train_run, f"train.epoch{k}.n_type2"))
        except KeyError:
            break
    return out


def diag_checkpoints(train_run: Path, arm: str, seed: int, p: dict, val, targets: np.ndarray, hidden: np.ndarray,
                     sample: np.ndarray, rec: R.Rec, engine_kind: str = "cpu", **cfg_extra) -> dict:
    """A8 item 6 per checkpoint of a train run (training order). `val` is the validation CorpusArrays with the arm's
    tables, `targets`/`hidden` the frozen eval set, `sample` row indices into `targets` for the class sums."""
    dg = p["diagnostics"]
    tok = np.asarray(val.tok).astype(np.int64)
    V = int(p["vocab"]) if "vocab" not in cfg_extra else int(cfg_extra["vocab"])
    share = np.bincount(tok[targets], minlength=V)[:V] / max(1, int(targets.shape[0]))
    rows = np.asarray(targets, np.int64)[sample]
    events = epoch_feedback_events(train_run)
    per_epoch: dict[int, dict] = {}
    prev, prev_name, out = None, None, {}
    for name in C.checkpoint_names(train_run):
        cfg, st, meta = C.load_named_checkpoint(train_run, name, arm, seed, p, rec, **cfg_extra)
        cen = C.census_of(st, int(cfg.vocab))
        lits = {k: cen[k] for k in ("included_per_clause_hist", "included_per_clause_mean", "included_per_clause_max",
                                    "frac_literals_negated", "frac_clauses_with_negated", "frac_literals_bag",
                                    "frac_clauses_with_bag")}
        own = owned_per_class(st, int(cfg.vocab), int(cfg.T), share)
        view = C.literal_view(st)
        cr = include_crossings(prev, view)
        cr["previous"] = prev_name
        epoch_of = int(name[len("epoch"):]) if name.startswith("epoch") else 1
        if prev is not None:
            e = per_epoch.setdefault(epoch_of, {"net_crossings": 0, "tracked_prev": 0, "intervals": []})
            e["net_crossings"] += cr["net_crossings"]
            e["tracked_prev"] += cr["tracked_prev"]
            e["intervals"].append(f"{prev_name}->{name}")
            if name.startswith("epoch") and prev_name == f"epoch{epoch_of - 1}" and epoch_of in events:
                e["feedback_events"] = events[epoch_of]
                e["net_crossings_per_1000_feedback_events"] = 1000.0 * cr["net_crossings"] / max(1, events[epoch_of])
        eng = R.make_engine(engine_kind, cfg, val)
        eng.load_state(st)
        cs, same = class_sum_stats(eng, cfg, st, rows, tok[rows], hidden, int(dg["class_sum_bin_width"]), int(dg["fired_chunk_rows"]))
        del eng
        rec.check(f"diag.{name}.target_sum_rebuilt_equals_predict_scores", same, rows=int(rows.shape[0]))
        rec.m(f"diag.{name}.literals", lits, note="A8 item 6: literal histogram, negated share, bag share (census_of)")
        rec.m(f"diag.{name}.owned_per_class", own, note="A8 item 6: owned clauses per class against T (docs/11 section 8 item 5)")
        rec.m(f"diag.{name}.include_crossings", cr, note="net crossings vs the previous checkpoint: a lower bound on crossing transitions")
        rec.m(f"diag.{name}.class_sums", cs, note="frozen V 8000 validation sample, frozen hidden mask")
        out[name] = {"literals": lits, "owned_per_class": own, "include_crossings": cr, "class_sums": cs}
        prev, prev_name = view, name
        rec.log(f"diag {name}: owned median {own['quantiles']['0.5']}, classes >= 2T {own['frac_classes_ge_2T']:.3f}, "
                f"S_y >= T {cs['frac_target_sum_ge_T']}, net crossings {cr.get('net_crossings')}")
        rec.flush()
    for k, e in sorted(per_epoch.items()):
        e["net_crossings_per_tracked_prev"] = e["net_crossings"] / e["tracked_prev"] if e["tracked_prev"] else None
    rec.m("diag.include_crossings_per_epoch", {str(k): v for k, v in sorted(per_epoch.items())},
          note="sum over the checkpoint intervals ending in that epoch; epoch 1 from the young rungs on; net, a lower bound")
    return out


def stage_census(args, cj: dict, rec: R.Rec, train_run: Path) -> None:
    from tmlm.data.corpus_store import load_split, split_dir
    from tmlm.data.evalsets import load_evalset, test_access_count

    import cvtm_t1_readout as T1

    p = cj["params"]
    dg = p["diagnostics"]
    C.stage_census(args, cj, rec, train_run)
    ta0 = test_access_count()
    rec.m("diag.test_accesses_before", ta0)
    arm, seed = C.train_run_identity(train_run)
    base, _ = C.CONTROL_ARMS[arm]
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    V = int(p["vocab"])
    val = P.with_tables(load_split(R.CORPUS, V, "validation"), kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    del prefix
    ev = load_evalset(R.CORPUS, V, "validation", "subword")
    targets = np.asarray(ev["targets"], np.int64)
    hidden = np.asarray(ev["hidden"], np.uint8)
    meta = json.loads((split_dir(R.CORPUS, V, "validation") / "meta.json").read_text())
    R.dataset_hashes({f"{R.CORPUS}_bpe{V}_validation_tok": meta["sha256"]["tok.npy"],
                      f"evalset_validation_subword_V{V}": ev["meta"]["sha256_combined"]})
    sample = T1.sample_positions(np.arange(targets.shape[0]), int(dg["class_sum_rows"]), int(dg["class_sum_sample_seed"]))
    rec.m("diag.validation_targets", int(targets.shape[0]))
    rec.m("diag.class_sum_sample_rows", int(sample.shape[0]))
    rec.m("diag.evalset_sha256_combined", ev["meta"]["sha256_combined"])
    diag_checkpoints(train_run, arm, seed, p, val, targets, hidden, sample, rec, str(dg["engine"]))
    ta1 = test_access_count()
    rec.m("diag.test_accesses_after", ta1)
    rec.check("diag.no_test_access", ta1 == ta0, before=ta0, after=ta1)
    rec.flush()


# ------------------------------------------------------------------ report (descriptive, no verdict)
def read_qpred(runs: list[Path]) -> dict:
    out: dict = {}
    for d in runs:
        m = json.loads((d / "metrics.json").read_text())
        arm, seed = m["control.arm"]["value"], int(m["control.seed"]["value"])
        for k, v in m.items():
            if k.startswith("qpred.") and k.endswith(".loglin_top1"):
                out[(arm, seed, k[len("qpred."):-len(".loglin_top1")])] = v["value"]
    return out


def _mean(d: dict | None) -> float | None:
    return float(np.mean([d[k] for k in sorted(d)])) if d else None


def _pdiff(a: dict | None, b: dict | None) -> tuple[float | None, dict | None]:
    """Paired difference in Q_text points over the common label draws, with the bootstrap interval."""
    if not a or not b:
        return None, None
    common = sorted(set(a) & set(b))
    if not common:
        return None, None
    d = 100.0 * np.array([a[k] - b[k] for k in common])
    return float(d.mean()), PR.paired_bootstrap(d)


def comparisons(q: dict, qpred: dict, arm: str, seed: int, code: str, budget: str) -> dict:
    get = (lambda a, name: q.get((a, seed, f"{name}.{code}", "lr", budget)))

    def best_rung(a: str) -> tuple[str | None, dict | None, dict]:
        rungs = {s: q[(a, seed, s, "lr", budget)] for s in C.ladder_set_names(q, a, seed, code, budget)}
        b = max(rungs, key=lambda s: _mean(rungs[s])) if rungs else None
        return b, (rungs[b] if b else None), {s: _mean(v) for s, v in rungs.items()}

    bank, e1, twin, ref = get(arm, "epoch4"), get(arm, "epoch1"), get(TWIN_ARM, "epoch4"), get(REF_ARM, "epoch4")
    best, best_q, rungs = best_rung(arm)
    rbest, rbest_q, _ = best_rung(REF_ARM)
    out: dict[str, Any] = {"q_bank": _mean(bank), "q_epoch1": _mean(e1), "q_twin": _mean(twin), "q_ref": _mean(ref),
                           "max_rung": best, "q_max_rung": _mean(best_q), "rungs": rungs,
                           "ref_max_rung": rbest, "q_ref_max_rung": _mean(rbest_q)}
    for key, other in (("d_ladder", best_q), ("d_twin", twin), ("d_vs_A3", ref), ("d_vs_A3_ladder", rbest_q), ("d_epoch4_vs_epoch1", e1)):
        out[key], out[key + "_bootstrap"] = _pdiff(bank, other)
    out.update({"q_pred_bank": qpred.get((arm, seed, "epoch4")), "q_pred_c0y": qpred.get((arm, seed, "c0y")),
                "q_pred_twin": qpred.get((TWIN_ARM, seed, "epoch4")), "q_pred_ref": qpred.get((REF_ARM, seed, "epoch4"))})
    return out


def stage_report(args, cj: dict, rec: R.Rec) -> Path:
    p = cj["params"]
    out = C.out_dir("report")
    rows = C.read_rows([R.rp(d) for d in (args.probe_runs or "").split(",") if d.strip()])
    q = R.summarise(rows, list(p["tasks"]))
    qpred = read_qpred([R.rp(d) for d in (args.qpred_runs or "").split(",") if d.strip()])
    seeds = [int(s) for s in p["seeds"]]
    arms = list(p["config_arms"])
    present = sorted({(str(r["arm"]), int(r["seed"])) for r in rows})
    want = [(a, s) for a in arms + [REF_ARM, TWIN_ARM] for s in seeds]
    rec.m("report.probe_arms_seeds_present", [list(x) for x in present])
    rec.m("report.probe_arms_seeds_missing", [list(x) for x in want if x not in present])
    rec.m("report.qpred_present", sorted(f"{a}.{s}.{n}" for a, s, n in qpred))
    summary: dict[str, dict] = {}
    for arm in arms:
        for s in seeds:
            for code in CODES:
                for b in p["probe_budgets"]:
                    comp = comparisons(q, qpred, arm, s, code, str(b))
                    key = f"{arm}.seed{s}.{code}.budget{b}"
                    summary[key] = comp
                    rec.m(f"report.{key}", comp, label="derived",
                          note="exploratory, A8 item 2, informs design only; Q_text points, LR probe; Q_pred T1 loglin top-1")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    rec.flush()
    return out


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A8 item 2 configuration arms (exploratory)")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--arm", default=None)
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
    cj = C.load_config(args.config)
    p = cj["params"]
    if p.get("smoke", False):
        raise SystemExit("this study has no smoke config")
    check_params_equal_controls(cj)
    arms = register(cj)
    rec = R.Rec(False)
    rec.m("stage", args.stage)
    rec.m("config_params", p)
    rec.m("config_arms", {n: a.to_dict() for n, a in arms.items()})
    C.check_shared_params(cj, rec)
    check_params_equal_controls(cj, rec)
    if args.stage == "train":
        if args.arm not in arms:
            raise SystemExit(f"--stage train needs --arm in {sorted(arms)}")
        if args.seed is None or int(args.seed) not in [int(s) for s in p["seeds"]]:
            raise SystemExit(f"--stage train needs --seed in {p['seeds']}")
        stage_train(args, cj, rec)
    elif args.stage in ("census", "features", "qpred"):
        tr = R.rp(args.train_run)
        if tr is None:
            raise SystemExit(f"--stage {args.stage} needs --train-run")
        try:
            arm, _ = C.train_run_identity(tr)
        except KeyError as exc:
            raise SystemExit(f"{tr} is not a train run of this study or of the control study ({exc})") from None
        allowed = set(arms) | (set(DIAG_ARMS) if args.stage == "census" else set())
        if arm not in allowed:
            raise SystemExit(f"train run arm {arm!r}: --stage {args.stage} accepts {sorted(allowed)}")
        {"census": stage_census, "features": C.stage_features, "qpred": C.stage_qpred}[args.stage](args, cj, rec, tr)
    elif args.stage == "probe":
        C.stage_probe(args, cj, rec, R.rp(args.features_run))
    else:
        stage_report(args, cj, rec)
    rec.m("wall_total_s", time.time() - T0)
    rec.m("failed_checks", rec.failed)
    rec.flush()
    rec.log(f"stage {args.stage} done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
