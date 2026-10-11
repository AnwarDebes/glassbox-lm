"""CONFIRMATORY R1 of docs/13 amendment A7 (family H17) with the additions of amendment A8 and the readings of
amendment A9 (binding; config experiments/cvtm/configs/ssl_r1_confirm.json, library src/tmlm/ssl/confirm.py).
Seeds 1 to 3 of the top two arms of the seed-100 tournament (A3, the fixed winner of A9 item 1, and A2) with their
literal-frozen twins (A3T, A2T), of A0 and A1 (H17.4 is "A1 minus A0"), of A6 (the A7 control "random kind_map"), of
A3R (A3 with the random kind_map of A6, A9 item 9, reported only), every learning-credit control of A7 and A8, the
full-label-pool rows of A9 item 8, and the registered verdicts.

Nothing of the running exploratory code is edited: this file IMPORTS experiments/ssl_r1_pretext.py (R),
experiments/ssl_r1_controls.py (C: the train loop with the young ladder, the features, Q_pred and the census),
experiments/ssl_r1_config_arms.py (CA: the A8 item 6 diagnostics) and tmlm.ssl (unchanged). `register(cj)` adds the
arm A3R to P.ARMS and the arms A1, A2, A6, A3R and the twin A2T to C.CONTROL_ARMS when `main` starts (never at
import).

Frozen configurations and the test discipline (A7 "Splits", A9 item 10; the config's "test_discipline"):
  * a configuration is one trained run (arm, seed) with all its checkpoints, or "common" (the seed-independent
    controls: G+, bag of words, cluster bag of words, PPMI-SVD, CBOW, MNB and NBSVM on bag of words), or one random
    conjunction bank "rand_seed<S>": 28 configurations (eight arms on three seeds, common, three random banks);
  * its DEV stage runs everything on the R1 dev carve and FREEZES the configuration: every probe cell's refit model
    is pickled with its dev predictions (stage probe); for A3, A2 and common the dev stage also fits the full-pool
    probes of A9 item 8 (stages fullfeat and probe on the full doc set) before any test access;
  * the test hash references are written first (stage testhash: the sha256 of each test file from the host's
    published metadata, no content, no counted access);
  * then ONE counted access per (configuration, task) of the confirmatory sets (AG News and TREC-6 official test
    splits, the SST-2 confirm carve), through tmlm.ssl.confirm.load_confirm (tmlm.data.evalsets.record_test_access),
    only after a PREFLIGHT has passed: the dev runs are clean, the frozen model files have their logged hashes, the
    checkpoints load with their logged state hashes, the tokenizer, prefix and every artefact the codes need load,
    the references exist, and the stored dev predictions are reproduced by the loaded models. The frozen models are
    then applied, never refit (stage confirm). The access count is checked per (configuration, task) (exactly one new
    log line each), never run-wide. Q_con reads the MEN test pairs the same way (stage qcon), after the dev Q_con.

Stages (each its own logged run through scripts/run_logged.py):
  testhash (CPU) the A9 item 10 reference hash files of the AG News and TREC test files (metadata only)
  train    (GPU) --arm A --seed S: ssl_r1_controls.stage_train (R1 loop, C0y, young ladder, twin checks)
  census   (CPU) --train-run R: ssl_r1_config_arms.stage_census (the control study's census + A8 item 6)
  features (GPU) --train-run R [--checkpoints]: ssl_r1_controls.stage_features on the R1 dev doc set
  fullfeat (GPU) --train-run R --features-run F | --common-run G: the full-pool doc set and its codes (A9 item 8)
  common   (GPU) G+ selection and dev features, PPMI-SVD, the R1 dev doc set (checked equal to the R1 artefacts)
  randbank (GPU) --seed S: the A8 random conjunction bank, dev features, prefix fire counts (dimension-matched)
  probe    (CPU) --features-run F: the dev probes of the configuration's plan (the full-pool plan for a fullfeat
           run), models frozen
  confirm  (GPU) --probe-run P [--probe-full-run PF]: preflight, counted test access, test codes, frozen models
  qpred    (GPU) --train-run R --checkpoints ...: ssl_r1_controls.stage_qpred (A8 item 1 Q_pred)
  qcon     (GPU) --train-run R --probe-run P --common-run G: Q_con (A7 word code), dev then MEN test (counted)
  qint     (GPU) --train-run R --features-run F --probe-run P: Q_int (NPMI coherence, word intrusion)
  report   (CPU) --probe-runs --confirm-runs --qpred-runs --qcon-runs --qint-runs [--census-runs] [--skipped-file]

Launch: the chain launch/chain_r1_confirm.sh (waits for the exploratory chains, then CPU
slot first, test-access lock, GPU lock inside each step).
"""

from __future__ import annotations

import argparse
import json
import pickle
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

import ssl_r1_config_arms as CA  # noqa: E402  (imported, never edited)
import ssl_r1_controls as C  # noqa: E402  (imported, never edited)
import ssl_r1_pretext as R  # noqa: E402  (imported, never edited)
from tmlm.common import hashing  # noqa: E402
from tmlm.ssl import confirm as CF  # noqa: E402
from tmlm.ssl import controls as K  # noqa: E402
from tmlm.ssl import downstream as D  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402
from tmlm.ssl import probes as PR  # noqa: E402

T0 = time.time()
DEFAULT_CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_confirm.json"
PREREG = ROOT / "docs" / "13-preregistration.md"   # an amendment named by the config must exist here (tests replace it)
CODES = ("all", "dim")
STAGES = ("testhash", "train", "census", "features", "fullfeat", "common", "randbank", "probe", "confirm", "qpred", "qcon",
          "qint", "report")


# ------------------------------------------------------------------ bookkeeping
class Rec(R.Rec):
    """R.Rec with confirmatory labels (every metric of this script is confirmatory, not exploratory)."""

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, np.bool_):
            value = bool(value)
        self.metrics[name] = {"value": value, "label": label, "confirmatory": True, "exploratory": False, **extra}


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extra_arms(cj: dict) -> dict[str, P.Arm]:
    """The confirmatory arms that are not R1 arms (A3R, docs/13 A9 item 9), as tmlm.ssl.pretext Arm objects."""
    out = {}
    for name, s in cj.get("extra_arms", {}).items():
        out[name] = P.Arm(name, str(s["mode"]), f1=bool(s.get("f1", False)), f2=bool(s.get("f2", False)),
                          random_kind_map=bool(s.get("random_kind_map", False)),
                          engine=tuple((str(k), v) for k, v in s.get("engine", {}).items()), description=str(s["description"]))
    return out


def register(cj: dict) -> None:
    """Add the extra arms to P.ARMS (A3R: A3's data lever F2 with A6's random kind_map; P.arm_data and P.engine_config
    are table driven, so this is a data-level arm like A6) and the confirmatory arms to C.CONTROL_ARMS (base arm,
    twin flag); idempotent, refuses a conflict."""
    extra = extra_arms(cj)
    for name, arm in extra.items():
        if name in P.ARMS and P.ARMS[name] != arm:
            raise SystemExit(f"P.ARMS already holds a different arm {name}")
    known = set(P.ARMS) | set(extra)
    for arm, spec in cj["configurations"]["arms"].items():
        entry = (str(spec["base"]), bool(spec["twin"]))
        if entry[0] not in known:
            raise SystemExit(f"{arm}: base arm {entry[0]} is not an R1 arm or a registered extra arm")
        if arm in C.CONTROL_ARMS and C.CONTROL_ARMS[arm] != entry:
            raise SystemExit(f"CONTROL_ARMS already holds a different entry for {arm}")
    for name, arm in extra.items():
        P.ARMS[name] = arm
    for arm, spec in cj["configurations"]["arms"].items():
        C.CONTROL_ARMS[arm] = (str(spec["base"]), bool(spec["twin"]))


def a9_problems(cj: dict) -> list[str]:
    """Where the config disagrees with a reading docs/13 A9 fixes (the config may not decide these)."""
    bad = []
    w = cj["winner"]
    if w.get("rule") != "seed100" or w.get("seed100_winner") != "A3" or w.get("seed100_runner_up") != "A2":
        bad.append("A9 item 1: winner fixed A3 (rule seed100), A2 second")
    if cj["verdicts"].get("code") != "all":
        bad.append("A9 item 2: the all-clauses code decides H17.1 to H17.4")
    h11 = cj["verdicts"]["H17.11"]
    if h11.get("tm_set") != "epoch4.dim" or h11.get("lr_set") != "epoch4.dim" or h11.get("reported_lr_set") != "epoch4.all":
        bad.append("A9 item 3: H17.11 = TM(dim) - LR(dim), TM(dim) - LR(all) reported")
    if cj["verdicts"]["H17.1"].get("stronger_control") != "per_seed_mean_confirmatory_q_text":
        bad.append("A9 item 4: max(C0y, G+) per seed on the mean confirmatory Q_text")
    if cj["statistics"].get("decides") != ["H17.1", "H17.2", "H17.4"] or int(cj["statistics"]["n_boot"]) != 10000:
        bad.append("A9 item 5: the two-level bootstrap (10,000) decides H17.1, H17.2 and H17.4")
    if cj["a8"].get("rows_without_threshold") != ["minus_exposure_matched", "minus_best_young_any_arm"]:
        bad.append("A9 item 5: the exposure-matched and best-young rows have no threshold")
    if cj["qcon"].get("men_confirm") != "dev+test" or cj["verdicts"]["H17.3"].get("over") != "mean_over_seeds":
        bad.append("A9 item 6: MEN in full (dev and test pairs), H17.3 on the mean over seeds")
    qi = cj["qint"]
    if (qi.get("primary_reference") != "train_remainder" or int(qi["top_words"]) != 10 or int(qi["min_count"]) != 5
            or int(qi["window"]) != 10 or qi.get("clauses") != "dim_matched"):
        bad.append("A9 item 7: top-10 PPMI words (>= 5 counts) of the dim-matched clauses, NPMI window 10, primary train remainder")
    fb = cj.get("full_budget", {})
    if sorted(fb.get("arms", [])) != ["A2", "A3"] or fb.get("draw") != 0 or "common" not in fb.get("configurations", []):
        bad.append("A9 item 8: full label pool for A3, A2 and common (G+, bag of words), draw 0")
    a3r = cj["configurations"]["arms"].get("A3R", {})
    ex = cj.get("extra_arms", {}).get("A3R", {})
    if a3r.get("base") != "A3R" or a3r.get("twin") or not (ex.get("f2") and ex.get("random_kind_map") and not ex.get("f1")
                                                            and ex.get("mode") == "masked" and not ex.get("engine")):
        bad.append("A9 item 9: A3R = A3 with A6's random kind_map, trained on seeds 1 to 3")
    td = cj["test_discipline"]
    if td.get("non_lfs_reference", "refuse") not in CF.NON_LFS_MODES or (
            td.get("non_lfs_reference", "refuse") != "refuse" and not td.get("non_lfs_reference_amendment")):
        bad.append("A9 item 10: the reference is the published sha256; the git blob id of a non-LFS file needs a named amendment")
    elif td.get("non_lfs_reference", "refuse") != "refuse":
        import re

        ids = re.findall(r"\bA(\d+)\b", str(td.get("non_lfs_reference_amendment")))
        text = PREREG.read_text(encoding="utf-8") if PREREG.exists() else ""
        if not ids or not any(re.search(rf"^### A{i} \(", text, flags=re.M) for i in ids):
            bad.append("A9 item 10: non_lfs_reference_amendment must name an amendment that exists in docs/13 (a config is not a "
                       "registration, protocol rule 16)")
    n_cfg = len(cj["configurations"]["arms"]) * len(cj["params"]["seeds"]) + 1 + len(cj["random_bank"]["seeds"])
    if n_cfg != int(cj["test_discipline"]["n_configurations"]) or n_cfg != 28:
        bad.append(f"A9 item 10: 28 configurations (config gives {n_cfg})")
    return bad


def check_config(cj: dict, rec: R.Rec | None = None) -> None:
    """The params shared with R1 equal R1's; the ladder, twin overrides and Q_pred equal the control study's; the
    item 6 diagnostics equal the configuration-arms study's; every reading docs/13 A9 fixes is the config's."""
    C.check_shared_params(cj, rec)
    ctl = C.load_config(ROOT / cj["controls_config"])["params"]
    arms = C.load_config(ROOT / cj["config_arms_config"])["params"]
    bad = [k for k in cj["params_equal_controls"] if cj["params"].get(k) != ctl.get(k)]
    if cj["params"].get("diagnostics") != arms.get("diagnostics"):
        bad.append("diagnostics")
    if rec is not None:
        rec.check("config.params_equal_control_and_arms_studies", not bad, differing=bad)
    if bad:
        raise SystemExit(f"params differ from the control / arms studies: {bad}")
    a9 = a9_problems(cj)
    if rec is not None:
        rec.check("config.a9_readings", not a9, differing=a9)
    if a9:
        raise SystemExit(f"the config disagrees with docs/13 A9: {a9}")


def needs_full(cj: dict, kind: str, arm: str | None) -> bool:
    """A9 item 8: the configuration has full-label-pool probes (A3 and A2 on every seed, and common)."""
    fb = cj["full_budget"]
    return (kind == "common" and "common" in fb["configurations"]) or (kind == "arm" and arm in fb["arms"])


def arm_spec(cj: dict, arm: str) -> dict:
    return cj["configurations"]["arms"][arm]


def config_id(kind: str, arm: str | None = None, seed: int | None = None) -> str:
    if kind == "arm":
        return f"{arm}_seed{int(seed)}"
    if kind == "rand":
        return f"rand_seed{int(seed)}"
    return "common"


def out_dir(stage: str) -> Path:
    return C.out_dir(stage)


def run_ok(run: Path) -> tuple[bool, dict]:
    """A finished logged run with exit 0 (run.json, when run_logged wrote one) and no failed check (metrics)."""
    info: dict[str, Any] = {"run": str(run)}
    rj = run / "run.json"
    if rj.exists():
        info["exit_status"] = json.loads(rj.read_text()).get("exit_status")
        if info["exit_status"] != 0:
            return False, info
    mf = run / "metrics.json"
    if not mf.exists():
        info["reason"] = "no metrics.json"
        return False, info
    m = json.loads(mf.read_text())
    info["failed_checks"] = m.get("failed_checks", {}).get("value")
    info["complete"] = m.get("probe.complete", {}).get("value", True)
    return bool(info["failed_checks"] == [] and info["complete"] is not False), info


def metric(run: Path, name: str):
    return R.read_metric(run, name)


def metric_or(run: Path, name: str, default=None):
    try:
        return R.read_metric(run, name)
    except (KeyError, FileNotFoundError):
        return default


# ------------------------------------------------------------------ testhash (A9 item 10: references before any access)
def stage_testhash(args, cj: dict, rec: R.Rec) -> None:
    """The write-once content-hash reference of every confirmatory TEST file from the host's published file metadata at
    the pinned revision (no content download; no counted access, proved by this process's access-log lines). The
    metadata response is logged for every task. A file without a published sha256 (stored in git, not LFS) is a failed
    check unless test_discipline.non_lfs_reference names an amendment that allows its git blob id (A9 item 10 names
    sha256); the other task's reference is still written."""
    p = cj["params"]
    td = cj["test_discipline"]
    mode = str(td.get("non_lfs_reference", "refuse"))
    amendment = td.get("non_lfs_reference_amendment")
    rec.m("confirm.kind", "testhash")
    rec.m("testhash.non_lfs_reference", mode, amendment=amendment)
    own0 = len(CF.own_accesses())
    for task in p["tasks"]:
        if CF.CONFIRM_SOURCES[task]["kind"] != "test_split":
            rec.m(f"testhash.{task}", "no test file (the SST-2 confirm carve is referenced by the A7 train hash file)")
            continue
        meta = CF.test_file_metadata(task)
        rec.m(f"testhash.{task}.metadata", meta, note="the host's response (HfApi.get_paths_info), metadata only")
        try:
            r = CF.write_test_hash_reference(task, metadata_fn=lambda t, meta=meta: meta, non_lfs=mode, amendment=amendment)
        except ValueError as exc:
            rec.check(f"testhash.{task}.reference_written", False, error=str(exc))
            continue
        rec.check(f"testhash.{task}.reference_written", True, reference=r["reference"], digest=r["digest"], bytes=r["bytes"],
                  written=r["written"])
        rec.m(f"testhash.{task}", r, note="written False = the file existed and is equal")
        if len(r["digest"]) == 64:  # run_logged accepts sha256 only; a git blob id (docs/13 A10) stays in the metric above
            R.dataset_hashes({f"{task}_test_reference_{r['reference']}": r["digest"]})
        rec.log(f"{task}: reference {r['reference']} {r['digest']} ({r['bytes']} bytes), written {r['written']}")
    rec.check("testhash.no_counted_access", len(CF.own_accesses()) == own0)
    rec.flush()


# ------------------------------------------------------------------ train, census, features, qpred (imported stages)
def stage_train(args, cj: dict, rec: R.Rec) -> None:
    p = cj["params"]
    if args.arm not in cj["configurations"]["arms"]:
        raise SystemExit(f"--arm {args.arm} is not a confirmatory arm {sorted(cj['configurations']['arms'])}")
    if int(args.seed) not in [int(s) for s in p["seeds"]]:
        raise SystemExit(f"--seed {args.seed} is not a confirmatory seed {p['seeds']}")
    rec.m("confirm.config_id", config_id("arm", args.arm, args.seed))
    rec.m("confirm.kind", "arm")
    rec.m("confirm.role", arm_spec(cj, args.arm)["role"])
    C.stage_train(args, cj, rec)


def stage_features(args, cj: dict, rec: R.Rec, train_run: Path) -> Path:
    arm, seed = C.train_run_identity(train_run)
    if args.checkpoints is None:
        args.checkpoints = arm_spec(cj, arm)["features_checkpoints"]
    rec.m("confirm.config_id", config_id("arm", arm, seed))
    rec.m("confirm.kind", "arm")
    ok, info = run_ok(train_run)
    rec.check("features.train_run_ok", ok, **info)
    if speedup_used(cj, "O10_prefix_counts", rec):
        return features_core(args, cj, rec, train_run)
    out = C.stage_features(args, cj, rec, train_run)
    return out


# ------------------------------------------------------------------ exact speedups (compute-reduction items O4, O10, O12)
# docs/13 A8 item 7: a speedup is used for a registered row only if it reproduces the registered estimator on logged
# cells under a rule written before the check. The rule: experiments/cvtm/configs/ssl_r1_speedup_verify.json; the
# runs: experiments/ssl_r1_speedup_verify.py. The config's "speedups" block names, per item, whether it is enabled and
# the verification run(s); an enabled item is used only when every named verification run is clean (exit 0, no failed
# check), otherwise the stage runs the original computation of ssl_r1_controls.py (logged either way).
SPEEDUPS = ("O4_t1_gval", "O10_prefix_counts", "O12_qpred_dedupe")


def speedup_used(cj: dict, item: str, rec: R.Rec | None = None) -> bool:
    spec = cj.get("speedups", {}).get(item, {})
    enabled = bool(spec.get("enabled", False))
    runs = spec.get("verified_by") or []
    runs = [runs] if isinstance(runs, str) else list(runs)
    clean = {r: run_ok(R.rp(r))[0] for r in runs}
    used = enabled and bool(runs) and all(clean.values())
    if rec is not None:
        rec.m(f"speedup.{item}", {"enabled": enabled, "verified_by": runs, "verification_clean": clean, "used": used})
    return used


def featurize_checkpoints_counts(names: list[str], train_run: Path, arm: str, seed: int, docs, corpus_docs, corpus_prefix,
                                 prefix, p: dict, engine_kind: str, out: Path, rec: R.Rec, **cfg_extra) -> dict:
    """ssl_r1_controls.featurize_checkpoints (the same files, metrics and duplicate rule) with its prefix pass replaced by
    tmlm.ssl.qpred_fast.prefix_pass, which ALSO writes prefix_counts_<name>.npz: the Q_pred count table and the fit
    rows' and check rows' fired lists (by the registered fit_rows_fired, as Q_pred computes them) (O10). A duplicate
    state (equal hash) gets no table: Q_pred finds tables by hash."""
    import shutil

    import cvtm_t1_readout as T1
    from tmlm.ssl import qpred_fast as QF

    chunk = int(p["forward_chunk"])
    n_prefix = int(prefix.tok.shape[0])
    _, fit_pos, cnt_pos = C.qpred_split(prefix)
    check_rows = T1.sample_positions(np.arange(cnt_pos.shape[0]), int(p["qpred"]["checks"]["cross_engine_count_rows"]), 3)
    tok = np.asarray(prefix.tok).astype(np.int64)
    zero = np.zeros(n_prefix, np.uint8)        # Q_pred's bitmap for its fit rows (ssl_r1_controls.qpred_checkpoint)
    done: dict[str, str] = {}
    for name in names:
        cfg, st, meta = C.load_named_checkpoint(train_run, name, arm, seed, p, rec, **cfg_extra)
        h = meta["state_hash"]
        rec.m(f"{name}.alive", int(np.asarray(st["alive"]).sum()))
        twin_of = next((n for n, hh in done.items() if hh == h), None)
        if twin_of is not None:
            for pat in ("features_{}.npz", "fires_{}.npy", "dim_{}.npy"):
                shutil.copyfile(out / pat.format(twin_of), out / pat.format(name))
            rec.m(f"features.{name}.duplicate_of", twin_of, note="identical state hash: files copied")
            rec.m(f"features.{name}.file_sha256", hashing.sha256_file(out / f"features_{name}.npz"))
            rec.m(f"prefix_counts.{name}.duplicate_of", twin_of, note="identical state hash: no table written (Q_pred finds tables by state hash)")
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
        pp = QF.prefix_pass(eng, int(cfg.n_clauses), int(cfg.vocab), n_prefix, chunk, R.hidden_for(cfg, n_prefix), tok,
                            cnt_pos, fit_pos, check_rows,
                            lambda pos, eng=eng: C.fit_rows_fired(eng, engine_kind, pos, zero, chunk))
        fc = pp.fire_counts
        rec.m(f"prefix_fires.{name}.wall_s", time.time() - t1, note="one pass: R1 fire counts and the Q_pred count table (O10)")
        rec.m(f"prefix_fires.{name}.clauses_firing", int((fc > 0).sum()))
        rec.m(f"prefix_fires.{name}.fired_per_position", float(fc.sum()) / max(1, n_prefix))
        dim = K.top_columns_by(fc.astype(np.int64), int(p["dim_matched"]))
        np.save(out / f"fires_{name}.npy", fc)
        np.save(out / f"dim_{name}.npy", dim)
        rec.m(f"features.{name}.dim_matched_columns", int(dim.shape[0]))
        del eng
        t2 = time.time()
        pmeta = QF.pass_meta(h, prefix, corpus_prefix, fit_pos, cnt_pos, check_rows, cfg, engine_kind)
        fname = f"prefix_counts_{name}.npz"
        rec.m(f"prefix_counts.{name}.file", fname)
        rec.m(f"prefix_counts.{name}.state_hash", h)
        rec.m(f"prefix_counts.{name}.file_sha256", QF.save_prefix_counts(out / fname, pp, pmeta))
        rec.m(f"prefix_counts.{name}.counted_events", int(pp.n_events))
        rec.m(f"prefix_counts.{name}.distinct_pairs", int(pp.idx.shape[0]))
        rec.m(f"prefix_counts.{name}.save_wall_s", time.time() - t2)
        del pp
        if engine_kind == "cuda":
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        rec.m(f"memory.after_{name}", R.mem_info())
        rec.log(f"{name}: features, prefix fire counts and the Q_pred count table in {time.time() - t0:.0f} s")
        rec.flush()
        done[name] = h
    return done


def features_core(args, cj: dict, rec: R.Rec, train_run: Path) -> Path:
    """ssl_r1_controls.stage_features, line for line, with featurize_checkpoints_counts (O10)."""
    p = cj["params"]
    out = out_dir("features")
    arm, seed = C.train_run_identity(train_run)
    base, _ = C.CONTROL_ARMS[arm]
    names = C.resolve_names(args.checkpoints, train_run)
    rec.m("features.train_run", str(train_run))
    rec.m("features.checkpoints", names)
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    docs, layout, labels = R.doc_set(p, rec, False)
    C.save_tokens(out, docs, layout, labels, arm, seed)
    ref = cj.get("doc_set_reference", {}).get("run")
    if ref:
        eq = C.doc_set_equal(out / "tokens.npz", ROOT / ref / "tokens.npz")
        rec.check("docs.equal_to_r1_features_run", all(eq.values()), reference=ref, **eq)
    rec.flush()
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    corpus_docs = F.corpus_arrays(docs, ad.feature_tables())
    corpus_prefix = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    featurize_checkpoints_counts(names, train_run, arm, seed, docs, corpus_docs, corpus_prefix, prefix, p, args.engine, out, rec)
    rec.flush()
    return out


def prefix_count_index(features_run: Path | None, train_run: Path, arm: str, seed: int) -> tuple[dict, str | None]:
    """{state hash: (file, logged sha256)} of a features run's count tables, or ({}, reason) when the run cannot serve
    this Q_pred (not clean, another train run, arm or seed)."""
    if features_run is None:
        return {}, "no features run given"
    ok, info = run_ok(features_run)
    if not ok:
        return {}, f"features run not clean: {info}"
    m = json.loads((features_run / "metrics.json").read_text())

    def val(k):
        return m[k]["value"] if k in m else None
    if val("control.arm") != arm or val("control.seed") != seed:
        return {}, f"features run is arm {val('control.arm')} seed {val('control.seed')}"
    if R.rp(str(val("features.train_run"))) != train_run:
        return {}, f"features run was made from {val('features.train_run')}"
    idx = {}
    for k in m:
        if k.startswith("prefix_counts.") and k.endswith(".file"):
            name = k[len("prefix_counts."):-len(".file")]
            idx[str(val(f"prefix_counts.{name}.state_hash"))] = (features_run / str(val(k)), str(val(f"prefix_counts.{name}.file_sha256")))
    return idx, (None if idx else "the features run holds no count table")


def load_matching_counts(index: dict, why_not: str | None, expected: dict):
    """(PrefixPass, None) when the table of expected['state_hash'] exists, has its logged sha256 and was computed from
    exactly the inputs in `expected` (qpred_fast.pass_meta); else (None, reason): Q_pred then recomputes."""
    from tmlm.ssl import qpred_fast as QF

    if expected["state_hash"] not in index:
        return None, why_not or "no table for this state hash"
    path, sha = index[expected["state_hash"]]
    if not path.exists():
        return None, f"missing {path.name}"
    got_sha = hashing.sha256_file(path)
    if got_sha != sha:
        return None, f"{path.name}: sha256 {got_sha} differs from the logged {sha}"
    pp, meta = QF.load_prefix_counts(path)
    diff = sorted(k for k in set(expected) | set(meta) if meta.get(k) != expected.get(k))
    if diff:
        return None, f"{path.name} was computed from other inputs: {diff}"
    return pp, None


def copy_qpred(src: str, name: str, state_hash: str, rec: R.Rec, out: Path, row: dict) -> dict:
    """O12: `name` has the state of the already scored `src`: its Q_pred metrics (timing excluded) and files are copies,
    labelled derived with copied_from; nothing is recomputed."""
    import shutil

    pre_s, pre_n = f"qpred.{src}.", f"qpred.{name}."
    flags = ("value", "label", "exploratory", "confirmatory", "smoke")
    for k, v in list(rec.metrics.items()):
        if not k.startswith(pre_s) or k[len(pre_s):].endswith("wall_s"):
            continue
        value = v["value"]
        if k[len(pre_s):] == "fit" and isinstance(value, dict):
            value = {kk: vv for kk, vv in value.items() if kk not in ("wall_s", "s_per_evaluation")}
        rec.m(pre_n + k[len(pre_s):], value, label="derived", copied_from=src,
              copy_basis=f"state hash {state_hash} equals {src}'s: copied, not recomputed (O12)",
              **{kk: vv for kk, vv in v.items() if kk not in flags})
    for pat in ("qpred_{}.npz", "qpred_{}_readout.json"):
        shutil.copyfile(out / pat.format(src), out / pat.format(name))
    rec.m(f"qpred.{name}.duplicate_of", src, note="identical state hash: Q_pred metrics and files copied (O12)")
    rec.log(f"qpred {name}: state of {src}, metrics and files copied")
    rec.flush()
    return dict(row)


def qpred_checkpoint_fast(name: str, cfg, st: dict, prefix, val, targets: np.ndarray, hidden: np.ndarray, fit_pos: np.ndarray,
                          cnt_pos: np.ndarray, qp: dict, engine_kind: str, chunk: int, rec: R.Rec, out: Path,
                          loaded=None, fast_kernel: bool = True) -> dict:
    """ssl_r1_controls.qpred_checkpoint, line for line, except (O10) `loaded` (a qpred_fast.PrefixPass of the features
    run) replaces the counting pass and the fit-row pass, and (O4) the CUDA fit and the validation cross-check use
    qpred_fast.T1GpuG when fast_kernel."""
    import math

    import cvtm_t1_readout as T1
    from ckm_posthoc_probe import k_vote_scores
    from tmlm.ssl import qpred_fast as QF

    V, Cn = int(cfg.vocab), int(cfg.n_clauses)
    pre = f"qpred.{name}"
    tok_tr = np.asarray(prefix.tok).astype(np.int64)
    gold = np.asarray(val.tok)[targets].astype(np.int64)
    n_inc = T1.included_literals(st)
    chk = qp["checks"]
    # ---- validation fired lists (CPU engine, frozen hidden mask), API and brute-force checks
    t0 = time.time()
    eng_va = R.make_engine("cpu" if engine_kind != "oracle" else "oracle", cfg, val)
    eng_va.load_state(st)
    vptr, vids = C.api_fired_lists(eng_va, targets, hidden)
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
    # ---- counting pass over the prefix outside the fit documents (only the target hidden), or its table (O10)
    zero = np.zeros(int(prefix.tok.shape[0]), np.uint8)
    check_cnt = T1.sample_positions(np.arange(cnt_pos.shape[0]), int(chk["cross_engine_count_rows"]), 3)
    t0 = time.time()
    if loaded is None:
        eng_g = R.make_engine(engine_kind, cfg, prefix)
        eng_g.load_state(st)
        t0 = time.time()
        counts, n_events, got = C.count_table(eng_g, engine_kind, cnt_pos, tok_tr[cnt_pos], Cn, V, chunk, zero, check_cnt)
        rec.m(f"{pre}.count_pass_wall_s", time.time() - t0)
        rec.m(f"{pre}.counts_events", int(n_events))
        fptr, fids = C.fit_rows_fired(eng_g, engine_kind, fit_pos, zero, chunk)
        del eng_g
    else:
        if not np.array_equal(loaded.check_rows, check_cnt):
            raise RuntimeError("loaded table: check rows differ (load_matching_counts compares their hash)")
        counts, n_events, got = loaded.counts_dense(), int(loaded.n_events), loaded.got()
        fptr, fids = loaded.fit_ptr, loaded.fit_ids
        rec.m(f"{pre}.count_load_wall_s", time.time() - t0)
        rec.m(f"{pre}.counts_events", int(n_events))
    if engine_kind == "cuda":
        eng_c = R.make_engine("cpu", cfg, prefix)
        eng_c.load_state(st)
        bad = sum(int(not np.array_equal(C.api_fired_lists(eng_c, cnt_pos[[r]], zero)[1], got[r])) for r in check_cnt.tolist())
        check_fit = T1.sample_positions(np.arange(fit_pos.shape[0]), int(chk["cross_engine_fit_rows"]), 4)
        bad += sum(int(not np.array_equal(C.api_fired_lists(eng_c, fit_pos[[r]], zero)[1], fids[fptr[r]:fptr[r + 1]])) for r in check_fit.tolist())
        rec.check(f"{pre}.cuda_fired_lists_equal_cpu_engine", bad == 0, mismatches=bad, rows=int(check_cnt.shape[0] + check_fit.shape[0]))
        del eng_c
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    # ---- tables and read-out constants
    tab = C.Tables(counts, n_inc, tok_tr[cnt_pos], V)
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
        ts = time.time()
        if fast_kernel:
            t1 = QF.T1GpuG(V, tab.indptr, tab.idx, tab.val, tab.lam, tab.inv_n, tab.p_uni)
        else:
            t1 = T1.T1Gpu(V, tab.indptr, tab.idx, tab.val, tab.lam, tab.inv_n, tab.p_uni)
        rec.m(f"{pre}.t1_setup_wall_s", time.time() - ts,
              kernel="qpred_fast.T1GpuG (log term precomputed once, O4)" if fast_kernel else "cvtm_t1_readout.T1Gpu")
        alpha, beta0, info = T1.loglin_fit_gpu(t1, fit_rows, budget, rec, f"{name} loglin")
        ll, _, _, _, _, _ = tab.loglin(fit_rows, alpha, beta0, False)
        info["objective_cpu_float64"] = float(-ll / max(1, fit_rows["n"]))
        info["abs_diff_objective_gpu_vs_cpu"] = abs(info["objective_cpu_float64"] - info["objective_gpu_recomputed"])
        rec.check(f"{pre}.fit_objective_gpu_equals_cpu", info["abs_diff_objective_gpu_vs_cpu"] < 1e-6, diff=info["abs_diff_objective_gpu_vs_cpu"])
    else:
        alpha, beta0, info = C.loglin_fit_cpu(tab, fit_rows, budget)
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


def qpred_core(args, cj: dict, rec: R.Rec, train_run: Path, features_run: Path | None = None, fast_kernel: bool = True,
               load_counts: bool = True, dedupe: bool = True) -> Path:
    """ssl_r1_controls.stage_qpred, line for line, with the exact speedups O4 (fast_kernel), O10 (load_counts: the
    features run's tables, used only when they were computed from exactly this Q_pred's inputs) and O12 (dedupe)."""
    from tmlm.data.corpus_store import load_split, split_dir
    from tmlm.data.evalsets import load_evalset, test_access_count
    from tmlm.ssl import qpred_fast as QF

    import cvtm_t1_readout as T1

    p = cj["params"]
    qp = p["qpred"]
    h81 = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "H8_1_masked.json").read_text())["fit"]["loglin"]
    rec.check("qpred.fit_budget_equals_H8_1", qp["fit"] == h81)
    out = out_dir("qpred")
    ta0 = test_access_count()
    rec.m("qpred.test_accesses_before", ta0)
    arm, seed = C.train_run_identity(train_run)
    base, _ = C.CONTROL_ARMS[arm]
    names = C.resolve_names(args.checkpoints, train_run)
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    rec.m("qpred.train_run", str(train_run))
    rec.m("qpred.checkpoints", names)
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    prefix_t = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    V = int(p["vocab"])
    val0 = load_split(C.CORPUS, V, "validation")
    val = P.with_tables(val0, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    ev = load_evalset(C.CORPUS, V, "validation", "subword")
    targets = np.asarray(ev["targets"], np.int64)
    hidden = np.asarray(ev["hidden"], np.uint8)
    meta = json.loads((split_dir(C.CORPUS, V, "validation") / "meta.json").read_text())
    R.dataset_hashes({f"{C.CORPUS}_bpe{V}_validation_tok": meta["sha256"]["tok.npy"],
                      f"evalset_validation_subword_V{V}": ev["meta"]["sha256_combined"]})
    rec.m("qpred.validation_targets", int(targets.shape[0]))
    rec.m("qpred.evalset_sha256_combined", ev["meta"]["sha256_combined"])
    fit_doc, fit_pos, cnt_pos = C.qpred_split(prefix)
    rec.m("qpred.fit_documents", np.flatnonzero(fit_doc).tolist(), rule="splitmix64(i XOR 0xC0FFEE) mod 100 == 0 (docs/11 11.6), prefix documents")
    rec.m("qpred.fit_positions", int(fit_pos.shape[0]))
    rec.m("qpred.counting_positions", int(cnt_pos.shape[0]))
    rec.m("qpred.speedups_used", {"O4_t1_gval": bool(fast_kernel), "O10_prefix_counts": bool(load_counts), "O12_qpred_dedupe": bool(dedupe)})
    if load_counts:
        index, why_not = prefix_count_index(features_run, train_run, arm, seed)
        rec.m("qpred.features_run", str(features_run) if features_run else None, tables=len(index), unusable_reason=why_not)
        check_cnt = T1.sample_positions(np.arange(cnt_pos.shape[0]), int(qp["checks"]["cross_engine_count_rows"]), 3)
    rec.flush()
    results: dict[str, dict] = {}
    scored: dict[str, str] = {}
    for name in names:
        cfg, st, cmeta = C.load_named_checkpoint(train_run, name, arm, seed, p, rec)
        h = cmeta["state_hash"]
        src = next((n for n, hh in scored.items() if hh == h), None) if dedupe else None
        if src is not None:
            results[name] = copy_qpred(src, name, h, rec, out, results[src])
            continue
        loaded = None
        if load_counts:
            expected = QF.pass_meta(h, prefix, prefix_t, fit_pos, cnt_pos, check_cnt, cfg, args.engine)
            t0 = time.time()
            loaded, reason = load_matching_counts(index, why_not, expected)
            rec.m(f"qpred.{name}.count_source", "loaded" if loaded is not None else "recomputed",
                  file=str(index[h][0]) if loaded is not None else None, reason=reason, check_wall_s=time.time() - t0)
        results[name] = qpred_checkpoint_fast(name, cfg, st, prefix_t, val, targets, hidden, fit_pos, cnt_pos, qp, args.engine,
                                              int(p["forward_chunk"]), rec, out, loaded=loaded, fast_kernel=fast_kernel)
        del loaded
        scored[name] = h
    ta1 = test_access_count()
    rec.m("qpred.test_accesses_after", ta1)
    rec.check("qpred.no_test_access", ta1 == ta0, before=ta0, after=ta1)
    rec.m("qpred.summary", results)
    rec.flush()
    return out


def stage_qpred(args, cj: dict, rec: R.Rec, train_run: Path, features_run: Path | None) -> Path:
    """The A8 item 1 Q_pred: ssl_r1_controls.stage_qpred, or qpred_core with the speedups the config enables and whose
    verification runs are clean (docs/13 A8 item 7)."""
    use = {k: speedup_used(cj, k, rec) for k in SPEEDUPS}
    if not any(use.values()):
        return C.stage_qpred(args, cj, rec, train_run)
    return qpred_core(args, cj, rec, train_run, features_run, fast_kernel=use["O4_t1_gval"],
                      load_counts=use["O10_prefix_counts"] and features_run is not None, dedupe=use["O12_qpred_dedupe"])


# ------------------------------------------------------------------ common (seed-independent controls)
def stage_common(args, cj: dict, rec: R.Rec) -> Path:
    p = cj["params"]
    out = out_dir("common")
    rec.m("confirm.config_id", config_id("common"))
    rec.m("confirm.kind", "common")
    docs, layout, labels = R.doc_set(p, rec, False)
    C.save_tokens(out, docs, layout, labels, "common", 0)
    ref = cj.get("doc_set_reference", {}).get("run")
    if ref:
        eq = C.doc_set_equal(out / "tokens.npz", ROOT / ref / "tokens.npz")
        rec.check("docs.equal_to_r1_features_run", all(eq.values()), reference=ref, **eq)
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data("A0", prefix, int(p["vocab"]), float(p["f1_t"]))   # every masked arm but A6 has these tables
    xp = np
    if args.engine == "cuda":
        import cupy as xp  # noqa: F811
    t0 = time.time()
    g = K.gplus_select(xp, prefix.tok, prefix.doc_start, ad.kind_map, "masked", int(p["vocab"]), int(p["n_clauses"]),
                       chunk=int(p["gplus_chunk_positions"]), log=rec.log)
    g.save(out / "gplus_select.npz")
    rec.m("gplus.select", g.info)
    rec.m("gplus.select_wall_s", time.time() - t0)
    sdg = K.gplus_features(xp, g, docs.tok, docs.doc_start, ad.kind_map, chunk=int(p["gplus_feature_chunk"]))
    rec.m("features.gplus.file_sha256", R.save_sparse(out / "features_gplus.npz", sdg))
    np.save(out / "dim_gplus.npy", np.arange(min(int(p["dim_matched"]), g.n), dtype=np.int64))
    rs = np.random.default_rng(777)
    ok = True
    key_to_f = {(int(a), int(b)): f for f, (a, b) in enumerate(zip(g.lit1, g.lit2))}
    for d in np.sort(rs.choice(docs.n_doc, size=min(3, docs.n_doc), replace=False)).tolist():
        pos = range(int(docs.doc_start[d]), int(docs.doc_start[d + 1]))
        bf = K.gplus_bruteforce(docs.tok, docs.doc_start, ad.kind_map, "masked", int(p["vocab"]), positions=pos)
        exp = {key_to_f[k]: v for k, v in bf.items() if k in key_to_f}
        row = sdg.rows(np.asarray([d]))
        ok &= dict(zip(row.ids.tolist(), row.cnt.tolist())) == exp
    rec.check("gplus.features_equal_bruteforce", ok, docs=3)
    t0 = time.time()
    E, info = K.ppmi_svd(np.asarray(prefix.tok), np.asarray(prefix.doc_start), int(p["vocab"]), int(p["ppmi"]["window"]),
                         int(p["ppmi"]["dim"]), float(p["ppmi"]["power"]))
    np.save(out / "ppmi_svd.npy", E)
    rec.m("ppmi_svd.info", info)
    rec.m("ppmi_svd.wall_s", time.time() - t0)
    rec.m("ppmi_svd.sha256", hashing.sha256_file(out / "ppmi_svd.npy"))
    W = np.load(ROOT / cj["baselines"]["cbow"]["file"])
    rec.check("cbow.shape", W.shape == (int(p["vocab"]), int(cj["baselines"]["cbow"]["dim"])), shape=list(W.shape))
    rec.m("cbow.sha256", hashing.sha256_file(ROOT / cj["baselines"]["cbow"]["file"]))
    ref = cj["baselines"]["r1_reference"]            # the same deterministic artefacts as R1 (checked / reported)
    g_ref = ROOT / ref["gplus"]
    if (g_ref / "features_gplus.npz").exists():
        a, b = R.load_sparse(out / "features_gplus.npz"), R.load_sparse(g_ref / "features_gplus.npz")
        same = (np.array_equal(a.indptr, b.indptr) and np.array_equal(a.ids, b.ids) and np.array_equal(a.cnt, b.cnt)
                and np.array_equal(np.load(out / "dim_gplus.npy"), np.load(g_ref / "dim_gplus.npy")))
        rec.check("common.gplus_equals_r1", same, reference=ref["gplus"])
    e_ref = ROOT / ref["ppmi_svd"]
    if e_ref.exists():
        rec.m("common.ppmi_svd_max_abs_diff_vs_r1", float(np.abs(np.load(e_ref).astype(np.float64) - E.astype(np.float64)).max()),
              reference=ref["ppmi_svd"], note="reported, not a check (BLAS rounding may differ between runs)")
    rec.flush()
    return out


# ------------------------------------------------------------------ random conjunction bank (A8 item 1)
def stage_randbank(args, cj: dict, rec: R.Rec) -> Path:
    p = cj["params"]
    rb = cj["random_bank"]
    if int(args.seed) not in [int(s) for s in rb["seeds"]]:
        raise SystemExit(f"--seed {args.seed} is not a random-bank seed {rb['seeds']}")
    out = out_dir("randbank")
    rec.m("confirm.config_id", config_id("rand", seed=args.seed))
    rec.m("confirm.kind", "rand")
    docs, layout, labels = R.doc_set(p, rec, False)
    C.save_tokens(out, docs, layout, labels, "rand", int(args.seed))
    ref = cj.get("doc_set_reference", {}).get("run")
    if ref:
        eq = C.doc_set_equal(out / "tokens.npz", ROOT / ref / "tokens.npz")
        rec.check("docs.equal_to_r1_features_run", all(eq.values()), reference=ref, **eq)
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data("A0", prefix, int(p["vocab"]), float(p["f1_t"]))
    xp = np
    if args.engine == "cuda":
        import cupy as xp  # noqa: F811
    t0 = time.time()
    bank = CF.random_bank(np.asarray(prefix.tok), np.asarray(prefix.doc_start), ad.kind_map, "masked", int(p["vocab"]),
                          int(p["n_clauses"]), int(args.seed), int(rb["max_literals"]))
    bank.save(out / "random_bank.npz")
    rec.m("randbank.info", bank.info)
    rec.m("randbank.sha256", hashing.sha256_file(out / "random_bank.npz"))
    single = CF.literal_single_counts(xp, prefix.tok, prefix.doc_start, ad.kind_map, "masked", int(p["vocab"]))
    np.save(out / "single_counts.npy", single)
    firer = CF.RandomBankFirer(xp, bank, single, ad.kind_map)
    sd = CF.random_bank_doc_counts(xp, firer, docs.tok, docs.doc_start, int(rb["chunk_positions"]))
    rec.m("features.rand.file_sha256", R.save_sparse(out / "features_rand.npz", sd, seed=np.int64(args.seed)))
    rec.m("features.rand.nnz", int(sd.ids.shape[0]))
    rec.m("features.rand.distinct_clauses_per_doc_mean", float(np.diff(sd.indptr).mean()))
    fc = CF.random_bank_fire_counts(xp, firer, prefix.tok, prefix.doc_start, int(rb["chunk_positions"]))
    np.save(out / "fires_rand.npy", fc)
    np.save(out / "dim_rand.npy", K.top_columns_by(fc.astype(np.int64), int(p["dim_matched"])))
    rec.m("prefix_fires.rand.fired_per_position", float(fc.sum()) / max(1, int(prefix.tok.shape[0])))
    rec.m("prefix_fires.rand.clauses_firing", int((fc > 0).sum()))
    # checks: a few documents against the pure-Python reference; the drawn literals exist in the prefix
    rs = np.random.default_rng(778)
    ok = True
    sub = np.sort(rs.choice(bank.n, size=min(int(rb["check_clauses"]), bank.n), replace=False))
    small = CF.RandomBank(bank.mode, bank.vocab, bank.col[sub], bank.sym[sub], bank.n_lit[sub], bank.seed, {})
    for d in np.sort(rs.choice(docs.n_doc, size=min(3, docs.n_doc), replace=False)).tolist():
        pos = list(range(int(docs.doc_start[d]), int(docs.doc_start[d + 1])))
        bf = CF.random_bank_bruteforce(small, docs.tok, docs.doc_start, ad.kind_map, pos)
        exp: dict[int, int] = {}
        for lst in bf:
            for j in lst:
                exp[int(sub[j])] = exp.get(int(sub[j]), 0) + 1
        row = sd.rows(np.asarray([d]))
        got = {int(i): int(c) for i, c in zip(row.ids.tolist(), row.cnt.tolist()) if int(i) in set(sub.tolist())}
        ok &= got == exp
    rec.check("randbank.features_equal_bruteforce", ok, docs=3, clauses=int(sub.shape[0]))
    lit_ok = bool(np.all(single[np.maximum(bank.col, 0), np.maximum(bank.sym, 0)][bank.col >= 0] > 0))
    rec.check("randbank.every_literal_true_somewhere_in_the_prefix", lit_ok)
    rec.m("randbank.wall_s", time.time() - t0)
    rec.flush()
    return out


# ------------------------------------------------------------------ fullfeat (A9 item 8: the full label pool)
def dev_blocks_equal(ours: Path, ref: Path) -> dict:
    """Per task: the dev block (tokens, document lengths, dev labels) of the full-pool doc set equals the one of the
    R1 dev doc set, so the full-pool probes are frozen on exactly the dev documents of the configuration."""
    a, b = np.load(ours), np.load(ref)
    la, lb = json.loads(str(a["layout"])), json.loads(str(b["layout"]))
    ya, yb = json.loads(str(a["labels"])), json.loads(str(b["labels"]))
    out = {}
    for task in lb:
        (a0, a1), (b0, b1) = la[task]["dev"], lb[task]["dev"]
        da, db = np.asarray(a["doc_start"], np.int64), np.asarray(b["doc_start"], np.int64)
        same = (a1 - a0 == b1 - b0 and np.array_equal(np.diff(da[a0:a1 + 1]), np.diff(db[b0:b1 + 1]))
                and np.array_equal(a["tok"][da[a0]:da[a1]], b["tok"][db[b0]:db[b1]])
                and ya[task]["dev_y"] == yb[task]["dev_y"] and ya[task]["pool_y"] == yb[task]["pool_y"])
        out[task] = bool(same)
    return out


def full_checkpoints(cj: dict) -> list[str]:
    E = int(cj["params"]["epochs"])
    return [n.replace("epochE", f"epoch{E}") for n in cj["full_budget"]["checkpoints"]]


def stage_fullfeat(args, cj: dict, rec: R.Rec) -> Path:
    """The full-pool doc set of one configuration (every task's dev carve plus its WHOLE label pool, R.doc_set with
    full=True, the R1 path of budget 'full') with the codes the full-pool probes read: an arm configuration (A3, A2)
    its full_budget.checkpoints; common its G+ (the common run's selection). The dev blocks, dimension-matched columns
    and state hashes must equal the configuration's dev features run."""
    p = cj["params"]
    out = out_dir("fullfeat")
    if (args.train_run is None) == (args.common_run is None):
        raise SystemExit("fullfeat needs exactly one of --train-run (arm) or --common-run (common)")
    if args.train_run is not None:
        tr = R.rp(args.train_run)
        arm, seed = C.train_run_identity(tr)
        kind, cid = "arm", config_id("arm", arm, seed)
        if not needs_full(cj, "arm", arm):
            raise SystemExit(f"{arm} has no full-pool row (A9 item 8: {cj['full_budget']['arms']})")
        dev_fr = R.rp(args.features_run) if args.features_run else None
        if dev_fr is None:
            raise SystemExit("fullfeat of an arm needs --features-run (the configuration's dev features run)")
        if C.train_run_identity(dev_fr) != (arm, seed) or R.rp(str(metric(dev_fr, "features.train_run"))) != tr:
            raise SystemExit("the dev features run belongs to another train run")
    else:
        tr, arm, seed = None, None, 0
        kind, cid = "common", config_id("common")
        dev_fr = R.rp(args.common_run)
        if str(metric(dev_fr, "confirm.kind")) != "common":
            raise SystemExit(f"{dev_fr} is not a common run")
    ok, info = run_ok(dev_fr)
    rec.check("fullfeat.dev_features_run_ok", ok, **info)
    if not ok:
        raise SystemExit(f"the dev features run did not finish cleanly: {info}")
    rec.m("confirm.config_id", cid)
    rec.m("confirm.kind", kind)
    rec.m("confirm.doc_set", "full")
    rec.m("fullfeat.dev_features_run", str(dev_fr))
    docs, layout, labels = R.doc_set(p, rec, True)
    C.save_tokens(out, docs, layout, labels, arm or "common", int(seed))
    eq = dev_blocks_equal(out / "tokens.npz", dev_fr / "tokens.npz")
    rec.check("fullfeat.dev_blocks_equal_the_dev_doc_set", all(eq.values()), **eq)
    rec.flush()
    prefix = R.load_prefix(p, rec)
    if kind == "arm":
        base, _ = C.CONTROL_ARMS[arm]
        ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
        names = full_checkpoints(cj)
        rec.m("features.train_run", str(tr))
        rec.m("features.checkpoints", names)
        rec.m("control.arm", arm)
        rec.m("control.seed", int(seed))
        corpus_docs = F.corpus_arrays(docs, ad.feature_tables())
        corpus_prefix = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
        hashes = C.featurize_checkpoints(names, tr, arm, seed, docs, corpus_docs, corpus_prefix, int(prefix.tok.shape[0]), p,
                                         args.engine, out, rec)
        for name in names:
            with np.load(dev_fr / f"features_{name}.npz") as z:
                dev_hash = str(z["state_hash"])
            rec.check(f"fullfeat.{name}.state_hash_equals_dev_features_run", hashes[name] == dev_hash, dev=dev_hash, full=hashes[name])
            rec.check(f"fullfeat.{name}.dim_columns_equal_dev_features_run",
                      bool(np.array_equal(np.load(out / f"dim_{name}.npy"), np.load(dev_fr / f"dim_{name}.npy"))))
    else:
        ad = P.arm_data("A0", prefix, int(p["vocab"]), float(p["f1_t"]))   # the kind_map of the common run's G+
        xp = np
        if args.engine == "cuda":
            import cupy as xp  # noqa: F811
        g = K.GPlus.load(dev_fr / "gplus_select.npz")
        t0 = time.time()
        sdg = K.gplus_features(xp, g, docs.tok, docs.doc_start, ad.kind_map, chunk=int(p["gplus_feature_chunk"]))
        rec.m("fullfeat.gplus.features_wall_s", time.time() - t0)
        rec.m("features.gplus.file_sha256", R.save_sparse(out / "features_gplus.npz", sdg))
        rec.m("features.gplus.nnz", int(sdg.ids.shape[0]))
        np.save(out / "dim_gplus.npy", np.load(dev_fr / "dim_gplus.npy"))
        rs = np.random.default_rng(779)
        ok_bf = True
        key_to_f = {(int(a), int(b)): f for f, (a, b) in enumerate(zip(g.lit1, g.lit2))}
        for d in np.sort(rs.choice(docs.n_doc, size=min(3, docs.n_doc), replace=False)).tolist():
            pos = range(int(docs.doc_start[d]), int(docs.doc_start[d + 1]))
            bf = K.gplus_bruteforce(docs.tok, docs.doc_start, ad.kind_map, "masked", int(p["vocab"]), positions=pos)
            exp = {key_to_f[k]: v for k, v in bf.items() if k in key_to_f}
            row = sdg.rows(np.asarray([d]))
            ok_bf &= dict(zip(row.ids.tolist(), row.cnt.tolist())) == exp
        rec.check("fullfeat.gplus.features_equal_bruteforce", ok_bf, docs=3)
        # the dev rows of the full doc set carry exactly the common run's dev G+ codes
        dev_sd = R.load_sparse(dev_fr / "features_gplus.npz")
        lf, ld = layout, json.loads(str(np.load(dev_fr / "tokens.npz")["layout"]))
        same = True
        for task in ld:
            a = sdg.rows(np.arange(*lf[task]["dev"], dtype=np.int64))
            b = dev_sd.rows(np.arange(*ld[task]["dev"], dtype=np.int64))
            same &= bool(np.array_equal(a.indptr - a.indptr[0], b.indptr - b.indptr[0]) and np.array_equal(a.ids, b.ids)
                         and np.array_equal(a.cnt, b.cnt))
        rec.check("fullfeat.gplus.dev_rows_equal_common_run", same)
        del dev_sd, sdg
    rec.m("memory", R.mem_info())
    rec.flush()
    return out


# ------------------------------------------------------------------ representations (dev store and test docs)
class Reps:
    """Probe inputs of every feature set, for any document rows (the transforms of R1's Store, plus MNB/NBSVM):
    '<name>.all' / '<name>.dim' (clause-like count codes, log1p and L2 rows; dim = the run's dim_<name>.npy),
    'bow', 'cluster' (all columns), 'ppmi', 'cbow' (dense mean vectors)."""

    p: dict
    docs: F.TokenDocs

    def sparse(self, name: str) -> F.SparseDocs:  # pragma: no cover (abstract)
        raise NotImplementedError

    def dim_cols(self, name: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def dense(self, name: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _src(self, fs: str) -> tuple[str, str]:
        src, _, var = fs.partition(".")
        return src, var

    def lr(self, fs: str, rows: np.ndarray):
        if fs in ("ppmi", "cbow"):
            return self.dense(fs)[rows], True
        src, var = self._src(fs)
        cols = self.dim_cols(src) if var == "dim" else None
        return self.sparse(src).rows(rows).to_csr("log1p", cols=cols), False

    def tm(self, fs: str, rows: np.ndarray, edges: np.ndarray | None = None):
        """(binary uint32 matrix, edges); edges (thermometer quantiles) are computed from `rows` when not given."""
        if fs in ("ppmi", "cbow"):
            X = self.dense(fs)
            if edges is None:
                edges = F.quantile_edges(X[rows], int(self.p["tm_ppmi_bits"]))
            return F.thermometer(X[rows], edges), edges
        src, var = self._src(fs)
        sd = self.sparse(src)
        if src in ("bow", "cluster"):
            cols = np.arange(sd.n_cols, dtype=np.int64)
        elif var == "dim":
            cols = self.dim_cols(src)
        else:
            raise KeyError(f"no TM input for {fs} (the TM probe reads dimension-matched columns only)")
        return K.binarise_counts(sd, rows, cols), None

    def counts(self, fs: str, rows: np.ndarray):
        return self.sparse(self._src(fs)[0]).rows(rows).to_csr("count", l2=False)

    def binary(self, fs: str, rows: np.ndarray):
        return self.sparse(self._src(fs)[0]).rows(rows).to_csr("binary", l2=False)

    def matrix(self, fs: str, clf: str, rows: np.ndarray, edges=None):
        if clf == "lr":
            return self.lr(fs, rows)[0]
        if clf == "tm":
            return self.tm(fs, rows, edges)[0]
        if clf == "mnb":
            return self.counts(fs, rows)
        if clf == "nbsvm":
            return self.binary(fs, rows)
        raise KeyError(clf)


class DevStore(C.AnyStore, Reps):
    """The R1 dev doc set of a features run (bank checkpoints, G+, random bank) with the count baselines, PPMI-SVD
    (from the common run) and CBOW."""

    def __init__(self, features_run: Path, common_run: Path | None, p: dict, cbow_file: Path | None) -> None:
        super().__init__(features_run, common_run, p)
        self.cbow_file = cbow_file

    def kind_map(self) -> np.ndarray:
        """The WikiText-103 train-split kind_map (cluster bags, G+ and the random bank of the masked arms but A6)."""
        if "kind_map" not in self._cache:
            from tmlm.data.corpus_store import load_split

            self._cache["kind_map"] = np.asarray(load_split(R.CORPUS, int(self.p["vocab"]), "train").kind_map)
        return self._cache["kind_map"]

    def dense(self, name: str) -> np.ndarray:
        if name == "ppmi":
            return self.ppmi_docs()
        if name == "cbow":
            if "cbow" not in self._cache:
                self._cache["cbow"] = F.mean_embedding(self.docs, np.load(self.cbow_file))
            return self._cache["cbow"]
        raise KeyError(name)


class ConfirmReps(Reps):
    """The confirmatory doc set: sparse codes computed by the confirm stage, dimension-matched columns and dense
    tables of the dev run (the frozen configuration)."""

    def __init__(self, docs: F.TokenDocs, sparse: dict, dev: DevStore, p: dict, vocab: int, kind_map: np.ndarray) -> None:
        self.docs, self._sp, self.dev, self.p, self.vocab, self.kind_map = docs, dict(sparse), dev, p, int(vocab), kind_map
        self._dense: dict = {}

    def sparse(self, name: str) -> F.SparseDocs:
        if name not in self._sp:
            if name == "bow":
                self._sp[name] = K.bag_of_words(self.docs, self.vocab)
            elif name == "cluster":
                self._sp[name] = cluster_bag_like(self.docs, self.kind_map, self.dev.docs)
            else:
                raise KeyError(name)
        return self._sp[name]

    def dim_cols(self, name: str) -> np.ndarray:
        return self.dev.dim_cols(name)

    def dense(self, name: str) -> np.ndarray:
        if name not in self._dense:
            if name == "ppmi":
                self._dense[name] = K.ppmi_svd_docs(self.docs, np.load(self.dev.br / "ppmi_svd.npy"))
            elif name == "cbow":
                self._dense[name] = F.mean_embedding(self.docs, np.load(self.dev.cbow_file))
            else:
                raise KeyError(name)
        return self._dense[name]


def cluster_bag_like(docs: F.TokenDocs, kind_map: np.ndarray, dev_docs: F.TokenDocs) -> F.SparseDocs:
    """Cluster bag of words of `docs` in the SAME compact column space as the dev doc set's (K.cluster_bag_of_words
    renumbers the used symbols of its own documents; a test document's symbol unseen in the dev set is dropped, which
    changes nothing for a frozen linear probe or TM whose training rows never had that column)."""
    _, used_dev = F.cluster_bags(dev_docs, kind_map, levels=(1, 2, 3))
    sd_full = F._aggregate_wide(*_cluster_events(docs, kind_map), docs.n_doc, int(np.asarray(kind_map).max()) + 1)
    remap = np.full(int(np.asarray(kind_map).max()) + 1, -1, np.int64)
    remap[used_dev] = np.arange(used_dev.shape[0])
    new = remap[sd_full.ids.astype(np.int64)]
    keep = new >= 0
    row = np.repeat(np.arange(sd_full.n_doc), np.diff(sd_full.indptr))[keep]
    ip = np.zeros(sd_full.n_doc + 1, np.int64)
    ip[1:] = np.cumsum(np.bincount(row, minlength=sd_full.n_doc))
    return F.SparseDocs(ip, new[keep].astype(np.uint32), sd_full.cnt[keep], int(used_dev.shape[0]))


def _cluster_events(docs: F.TokenDocs, kind_map: np.ndarray, levels=(1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
    km = np.asarray(kind_map, np.int64)
    tm = docs.text_mask()
    dop = docs.doc_of_pos()[tm]
    tok = docs.tok.astype(np.int64)[tm]
    return np.concatenate([dop] * len(levels)), np.concatenate([km[k][tok] for k in levels])


# ------------------------------------------------------------------ probe (dev stage: fit and freeze)
def expand_sets(block_sets, store: DevStore) -> list[str]:
    if block_sets == "ladder":
        names = sorted(n for n in store.feature_names() if n.startswith("young_b"))
        return [f"{n}.{c}" for n in names for c in CODES]
    return list(block_sets)


def plan_of(cj: dict, kind: str, arm: str | None) -> list[dict]:
    if kind == "arm":
        return cj["probe_plans"][arm_spec(cj, arm)["probe_plan"]]
    return cj["probe_plans"][kind]


def plan_cells(plan: list[dict], store: DevStore) -> list[tuple[str, str, int]]:
    """(set, classifier, budget) of every cell of the plan (sets expanded), in plan order, without duplicates."""
    out: list[tuple[str, str, int]] = []
    for blk in plan:
        sets = expand_sets(blk["sets"], store)
        for clf in ("lr", "tm", "mnb", "nbsvm"):
            spec = blk.get(clf)
            if spec is None:
                continue
            budgets = spec["budgets"] if isinstance(spec, dict) else spec
            only = spec.get("sets") if isinstance(spec, dict) else None
            for fs in sets:
                if only is not None and fs not in only:
                    continue
                for b in budgets:
                    key = (fs, clf, int(b))
                    if key not in out:
                        out.append(key)
    return out


def fit_cell(clf: str, Xl, yl, seed: int, p: dict, cj: dict, dense: bool):
    if clf == "lr":
        return CF.fit_lr(Xl, yl, seed, dense, p["c_grid"], int(p["folds"]), int(p["lr_max_iter"]), str(p.get("lr_backend", "cpu")))
    if clf == "tm":
        return CF.fit_tm(Xl, yl, seed, p["tm_grid"], int(p["tm_epochs"]), int(p["folds"]))
    if clf == "mnb":
        return CF.fit_mnb(Xl, yl, seed, cj["baselines"]["mnb"]["alpha_grid"], int(p["folds"]))
    if clf == "nbsvm":
        nb = cj["baselines"]["nbsvm"]
        return CF.fit_nbsvm(Xl, yl, seed, nb["c_grid"], int(p["folds"]), float(nb["alpha"]), float(nb["beta"]), int(nb["max_iter"]))
    raise KeyError(clf)


def dev_matrix(reps: Reps, fs: str, clf: str, rows: np.ndarray, edges, cache: dict):
    """reps.matrix for evaluation rows, cached per (set, classifier) unless the input depends on the cell (the TM
    thermometer edges of a dense set are fitted on the cell's labelled rows)."""
    if clf == "tm" and edges is not None:
        return reps.matrix(fs, clf, rows, edges)
    key = (fs, clf)
    if key not in cache:
        cache[key] = reps.matrix(fs, clf, rows, edges)
    return cache[key]


def model_file(out: Path, task: str) -> Path:
    return out / f"frozen_models_{task}.pkl"


def stage_probe(args, cj: dict, rec: R.Rec, features_run: Path) -> Path:
    p = cj["params"]
    out = out_dir("probe")
    ok, info = run_ok(features_run)
    rec.check("probe.features_run_ok", ok, **info)
    if not ok:
        raise SystemExit(f"{features_run} did not finish cleanly: {info}")
    kind = str(metric(features_run, "confirm.kind"))
    cid = str(metric(features_run, "confirm.config_id"))
    common_run = features_run if kind == "common" else None   # PPMI-SVD of the common configuration is its own
    store = DevStore(features_run, common_run, p, ROOT / cj["baselines"]["cbow"]["file"])
    arm = store.arm if kind == "arm" else None
    if str(metric_or(features_run, "confirm.doc_set", "r1")) == "full":
        return probe_full(cj, rec, store, kind, cid, features_run, out)
    rec.m("probe.doc_set", "r1")
    cells = plan_cells(plan_of(cj, kind, arm), store)
    budgets_ok = {int(b) for b in p["budgets"]}
    if any(b not in budgets_ok for _, _, b in cells):
        raise SystemExit("a probe budget is not an R1 budget (the doc set holds only their labelled rows)")
    rec.m("confirm.config_id", cid)
    rec.m("confirm.kind", kind)
    rec.m("probe.features_run", str(features_run))
    rec.m("probe.common_run", str(common_run) if common_run else None)
    rec.m("probe.arm", store.arm)
    rec.m("probe.seed", store.seed)
    rec.m("probe.cells_planned", [list(c) for c in cells])
    rec.m("probe.complete", False)
    rec.flush()
    rows_out = out / "cells.jsonl"
    n = 0
    hashes = {}
    with open(rows_out, "w", encoding="utf-8") as fh:
        for task in p["tasks"]:
            code = D.TASKS[task]["code"]
            pool_y = np.asarray(store.labels[task]["pool_y"], np.int64)
            dev_y = np.asarray(store.labels[task]["dev_y"], np.int64)
            dev = store.dev_rows(task)
            frozen: dict[str, CF.Frozen] = {}
            dev_cache: dict = {}   # dev matrices of this task (identical inputs for every draw and budget)
            for budget in sorted({b for _, _, b in cells}):
                for draw in p["draws"]:
                    pr = PR.label_sample(pool_y, budget, code, draw)
                    lab = store.rows(task, pr)
                    yl = pool_y[pr]
                    seed = PR.fold_seed(code, budget, draw)
                    for fs, clf, b in cells:
                        if b != budget:
                            continue
                        t0 = time.time()
                        dense = False
                        edges = None
                        if clf == "lr":
                            Xl, dense = store.lr(fs, lab)
                        elif clf == "tm":
                            Xl, edges = store.tm(fs, lab)
                        else:
                            Xl = store.matrix(fs, clf, lab)
                        f = fit_cell(clf, Xl, yl, seed, p, cj, dense)
                        f.extra["edges"] = edges
                        Xd = dev_matrix(store, fs, clf, dev, edges, dev_cache)
                        pred = CF.frozen_predict(f, Xd)
                        f.extra["dev_pred"] = pred.astype(np.int8)
                        acc = float(np.mean(pred == dev_y))
                        key = f"{fs}|{clf}|{budget}|{draw}"
                        frozen[key] = f
                        row = {"config_id": cid, "arm": store.arm, "seed": store.seed, "set": fs, "clf": clf, "task": task,
                               "budget": budget, "draw": draw, "split": "dev", "acc": acc, "choice": f.choice, "cv": f.cv,
                               "n_labelled": int(yl.shape[0]), "n_eval": int(dev_y.shape[0]),
                               "class_counts": np.bincount(yl, minlength=D.TASKS[task]["n_classes"]).tolist(),
                               "model_key": key, "dev_pred_sha256": hashing.sha256_bytes(pred.astype("<i8").tobytes()),
                               "wall_s": time.time() - t0}
                        fh.write(json.dumps(row, default=str) + "\n")
                        fh.flush()
                        n += 1
                    rec.log(f"{task} budget {budget} draw {draw}: done")
            with open(model_file(out, task), "wb") as mf:
                pickle.dump(frozen, mf, protocol=pickle.HIGHEST_PROTOCOL)
            hashes[task] = hashing.sha256_file(model_file(out, task))
            rec.m(f"probe.frozen_models.{task}.sha256", hashes[task], n_models=len(frozen))
            rec.flush()
    rec.m("probe.cells", n)
    rec.m("probe.cells_sha256", hashing.sha256_file(rows_out))
    rec.m("probe.complete", n == len(cells) * len(p["draws"]) * len(p["tasks"]), expected=len(cells) * len(p["draws"]) * len(p["tasks"]))
    rec.m("memory", R.mem_info())
    rec.flush()
    return out


def rows_csr(sd: F.SparseDocs, rows: np.ndarray, value: str = "log1p", cols: np.ndarray | None = None,
             chunk: int = 4096):
    """sd.rows(rows).to_csr(value, cols) (L2 rows) assembled chunk by chunk of rows into preallocated arrays. Every
    operation of to_csr (column restriction, log1p, the row norm, the scaling) is local to a row, so the matrix is
    the same bit for bit (tests/test_ssl_r1_confirm.py; checked again at run time on one task) while the peak memory
    is the final matrix plus one chunk instead of about four copies (the full-pool G+ matrix of AG News holds about
    1.2e9 entries)."""
    import scipy.sparse as sp

    rows = np.asarray(rows, np.int64)
    n = int(rows.shape[0])
    remap = None
    n_cols = sd.n_cols
    if cols is not None:
        remap = np.full(sd.n_cols, -1, np.int64)
        remap[np.asarray(cols, np.int64)] = np.arange(len(cols), dtype=np.int64)
        n_cols = len(cols)
    lens = np.zeros(n, np.int64)
    for a in range(0, n, int(chunk)):
        sub = sd.rows(rows[a:a + int(chunk)])
        if remap is None:
            lens[a:a + sub.n_doc] = np.diff(sub.indptr)
        else:
            keep = remap[sub.ids.astype(np.int64)] >= 0
            r = np.repeat(np.arange(sub.n_doc, dtype=np.int64), np.diff(sub.indptr))[keep]
            lens[a:a + sub.n_doc] = np.bincount(r, minlength=sub.n_doc)
    indptr = np.zeros(n + 1, np.int64)
    indptr[1:] = np.cumsum(lens)
    data = np.empty(int(indptr[-1]), np.float64)
    indices = None
    for a in range(0, n, int(chunk)):
        Xc = sd.rows(rows[a:a + int(chunk)]).to_csr(value, cols=cols)
        s, e = int(indptr[a]), int(indptr[min(n, a + int(chunk))])
        if Xc.nnz != e - s or not np.array_equal(np.diff(Xc.indptr), lens[a:a + Xc.shape[0]]):
            raise RuntimeError("chunk structure differs from the first pass")
        if indices is None:
            indices = np.empty(int(indptr[-1]), Xc.indices.dtype)
        data[s:e] = Xc.data
        indices[s:e] = Xc.indices
        del Xc
    if indices is None:
        indices = np.zeros(0, np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n_cols))


def full_matrix(store: "DevStore", fs: str, rows: np.ndarray):
    """LR input of a full-pool cell (sparse count codes only: <bank>.all, <bank>.dim, bow), memory lean."""
    src, _, var = fs.partition(".")
    if src in ("ppmi", "cbow", "cluster"):
        raise KeyError(f"{fs} is not a full-pool set (A9 item 8: epoch4, C0y, G+, bag of words)")
    cols = store.dim_cols(src) if var == "dim" else None
    return rows_csr(store.sparse(src), rows, "log1p", cols)


def probe_full(cj: dict, rec: R.Rec, store: "DevStore", kind: str, cid: str, features_run: Path, out: Path) -> Path:
    """The full-pool probes of A9 item 8 (dev stage of the configuration): LR (the registered probe, C by 4-fold CV
    inside the labelled sample) fitted on the WHOLE label pool of each task, draw 0 (as R1's budget 'full'), frozen
    and evaluated on the dev carve; the models are applied to the confirmatory set by the configuration's single
    confirm stage. The doc-set codes are dropped from memory during each fit and reloaded for the next cell."""
    p = cj["params"]
    fb = cj["full_budget"]
    E = int(p["epochs"])
    sets = [s.replace("epochE", f"epoch{E}") for s in fb["plans"]["common" if kind == "common" else "arm"]]
    draw = int(fb["draw"])
    if not needs_full(cj, kind, store.arm if kind == "arm" else None):
        raise SystemExit(f"{cid} has no full-pool row (A9 item 8)")
    rec.m("confirm.config_id", cid)
    rec.m("confirm.kind", kind)
    rec.m("probe.doc_set", "full")
    rec.m("probe.features_run", str(features_run))
    rec.m("probe.common_run", None)
    rec.m("probe.arm", store.arm)
    rec.m("probe.seed", store.seed)
    rec.m("probe.cells_planned", [[fs, "lr", "full"] for fs in sets])
    rec.m("probe.complete", False)
    rec.flush()
    # exactness of the memory-lean matrices against the R1 path (TREC: the smallest pool), before any fit
    t_chk = "trec" if "trec" in p["tasks"] else p["tasks"][-1]
    pool_y = np.asarray(store.labels[t_chk]["pool_y"], np.int64)
    lab = store.rows(t_chk, PR.label_sample(pool_y, "full", D.TASKS[t_chk]["code"], draw))
    same = {}
    for fs in sets:
        A = full_matrix(store, fs, lab)
        src, _, var = fs.partition(".")
        B = store.sparse(src).rows(lab).to_csr("log1p", cols=store.dim_cols(src) if var == "dim" else None)
        same[fs] = bool(A.shape == B.shape and np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
                        and np.array_equal(A.data.view(np.uint64), B.data.view(np.uint64)))
    rec.check("probe.full.lean_matrix_equals_r1_to_csr_bit_for_bit", all(same.values()), task=t_chk, **same)
    if not all(same.values()):
        raise SystemExit("the memory-lean matrix differs from the R1 path: no full-pool probe")
    n = 0
    with open(out / "cells.jsonl", "w", encoding="utf-8") as fh:
        for task in p["tasks"]:
            code = D.TASKS[task]["code"]
            pool_y = np.asarray(store.labels[task]["pool_y"], np.int64)
            dev_y = np.asarray(store.labels[task]["dev_y"], np.int64)
            dev = store.dev_rows(task)
            pr = PR.label_sample(pool_y, "full", code, draw)
            lab = store.rows(task, pr)
            yl = pool_y[pr]
            seed = PR.fold_seed(code, "full", draw)
            frozen: dict[str, CF.Frozen] = {}
            for fs in sets:
                t0 = time.time()
                Xl = full_matrix(store, fs, lab)
                Xd = full_matrix(store, fs, dev)
                store._cache.pop(fs.partition(".")[0], None)
                f = fit_cell("lr", Xl, yl, seed, p, cj, False)
                f.extra["edges"] = None
                pred = CF.frozen_predict(f, Xd)
                f.extra["dev_pred"] = pred.astype(np.int8)
                acc = float(np.mean(pred == dev_y))
                key = f"{fs}|lr|full|{draw}"
                frozen[key] = f
                row = {"config_id": cid, "arm": store.arm, "seed": store.seed, "set": fs, "clf": "lr", "task": task,
                       "budget": "full", "draw": draw, "split": "dev", "doc_set": "full", "acc": acc, "choice": f.choice,
                       "cv": f.cv, "n_labelled": int(yl.shape[0]), "n_eval": int(dev_y.shape[0]),
                       "class_counts": np.bincount(yl, minlength=D.TASKS[task]["n_classes"]).tolist(), "model_key": key,
                       "dev_pred_sha256": hashing.sha256_bytes(pred.astype("<i8").tobytes()), "nnz_labelled": int(Xl.nnz),
                       "wall_s": time.time() - t0}
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                n += 1
                del Xl, Xd
                rec.log(f"{task} {fs} full pool ({yl.shape[0]} labels): {time.time() - t0:.0f} s, peak {R.mem_info()}")
            with open(model_file(out, task), "wb") as mf:
                pickle.dump(frozen, mf, protocol=pickle.HIGHEST_PROTOCOL)
            rec.m(f"probe.frozen_models.{task}.sha256", hashing.sha256_file(model_file(out, task)), n_models=len(frozen))
            rec.flush()
    rec.m("probe.cells", n)
    rec.m("probe.cells_sha256", hashing.sha256_file(out / "cells.jsonl"))
    rec.m("probe.complete", n == len(sets) * len(p["tasks"]), expected=len(sets) * len(p["tasks"]))
    rec.m("memory", R.mem_info())
    rec.flush()
    return out


# ------------------------------------------------------------------ confirm (preflight, counted test access, frozen models applied)
def make_encoder(vocab: int):
    """BPE-8k encoder of confirmatory texts, exactly as R1 encodes the dev texts (F.encode with the WikiText-103
    tokenizer); loaded by the preflight, before the access (tests replace this function)."""
    from tmlm.data import bpe

    tok = bpe.load_tokenizer(int(vocab))
    return lambda texts: F.encode(tok, texts)


def confirm_docs(tasks: list[str], cid: str, detail: str, encode, reaccess_reason: str | None, rec: R.Rec):
    """The confirmatory doc set: ONE counted load per task (CF.load_confirm), then tokenisation."""
    parts, layout, labels, base = [], {}, {}, 0
    for task in tasks:
        cs = CF.load_confirm(task, cid, detail, reaccess_reason=reaccess_reason)
        rec.m(f"confirm.{task}.access", cs.info["access"])
        rec.m(f"confirm.{task}.source", {k: v for k, v in cs.info.items() if k != "access"})
        td = encode(cs.texts)
        parts.append(td)
        layout[task] = [base, base + td.n_doc]
        labels[task] = cs.y.astype(np.int64)
        base += td.n_doc
        R.dataset_hashes({f"{task}_confirm": cs.info.get("sha256") or cs.info.get("carve_idx_sha256")})
    docs, _ = F.concat(parts)
    return docs, layout, labels


def preflight_host(tasks: list[str], refs: dict) -> dict:
    """For every confirmatory TEST file that is not in the local cache yet (it will be downloaded inside the counted
    access): the host's file metadata at the pinned revision, fetched now (metadata only, no content), must still
    publish the referenced digest and size, so an unreachable host or a changed file stops the stage BEFORE the
    access. A cached file needs no network (its bytes are verified against the reference inside the access)."""
    out: dict[str, Any] = {"passed": True}
    for task in tasks:
        if CF.CONFIRM_SOURCES[task]["kind"] != "test_split":
            continue
        if CF.test_file_path(task).exists():
            out[task] = "cached"
            continue
        try:
            ref = refs[task]
            meta = CF.test_file_metadata(task)
            lfs = meta.get("lfs") or {}
            got = lfs.get("sha256") if ref["reference"] == "sha256" else meta.get("blob_id")
            ok = got == ref["digest"] and meta.get("size") is not None and int(meta["size"]) == int(ref["bytes"])
            out[task] = {"published": got, "size": meta.get("size"), "reference": ref["digest"], "equal": bool(ok)}
        except Exception as exc:  # unreachable host, malformed reference: no access
            ok = False
            out[task] = {"error": f"{type(exc).__name__}: {exc}"}
        out["passed"] = bool(out["passed"] and ok)
    return out


def _refuse(rec: R.Rec, name: str, ok: bool, msg: str, **info) -> None:
    rec.check(name, ok, **info)
    if not ok:
        rec.flush()
        raise SystemExit(f"{msg}: no test access")


def preflight_confirm(args, cj: dict, rec: R.Rec, probe_run: Path, probe_full_run: Path | None) -> dict:
    """Everything the confirm stage needs, loaded and checked BEFORE the counted access (a failure here costs no
    access): the dev probe runs (and for A3, A2 and common the full-pool probe run, A9 item 8) and their features runs
    are clean; the frozen model files have their logged sha256; the test references exist (A9 item 10) and the
    SST-2 train file matches its hash; the tokenizer loads; the prefix, arm tables and every checkpoint load with
    the state hash the train run logged and the dev features used; every artefact file the codes need exists and
    loads (G+ selection, random bank, single counts, PPMI-SVD, CBOW, dimension-matched columns); the stored dev
    predictions are reproduced by the loaded models. Returns the loaded objects for the stage."""
    p = cj["params"]
    cbow = ROOT / cj["baselines"]["cbow"]["file"]
    ok, info = run_ok(probe_run)
    _refuse(rec, "confirm.dev_probe_run_ok", ok, f"the dev stage of this configuration did not succeed ({info})", **info)
    cid = str(metric(probe_run, "confirm.config_id"))
    kind = str(metric(probe_run, "confirm.kind"))
    arm0 = str(metric(probe_run, "probe.arm")) if kind == "arm" else None
    full = needs_full(cj, kind, arm0)
    _refuse(rec, "confirm.full_pool_probe_present_iff_registered", (probe_full_run is not None) == full,
            "A9 item 8: A3, A2 and common need their full-pool dev probe run (--probe-full-run), every other configuration none",
            needs_full=full, given=str(probe_full_run) if probe_full_run else None)
    units = []
    for pr, doc_set in [(probe_run, "r1")] + ([(probe_full_run, "full")] if probe_full_run else []):
        ok, info = run_ok(pr)
        _refuse(rec, f"confirm.preflight.{doc_set}.probe_run_ok", ok, f"{pr} is not clean ({info})", **info)
        _refuse(rec, f"confirm.preflight.{doc_set}.same_configuration",
                str(metric(pr, "confirm.config_id")) == cid and str(metric_or(pr, "probe.doc_set", "r1")) == doc_set,
                f"{pr} is not the {doc_set} probe run of {cid}")
        fr = R.rp(str(metric(pr, "probe.features_run")))
        common = metric_or(pr, "probe.common_run")
        cr = R.rp(str(common)) if common else None
        fok, finfo = run_ok(fr)
        _refuse(rec, f"confirm.preflight.{doc_set}.features_run_ok", fok, f"the dev features run {fr} did not succeed ({finfo})", **finfo)
        frozen = {}
        for task in p["tasks"]:
            mf = model_file(pr, task)
            h = hashing.sha256_file(mf) if mf.exists() else None
            logged = metric_or(pr, f"probe.frozen_models.{task}.sha256")
            _refuse(rec, f"confirm.preflight.{doc_set}.frozen_models_{task}_unchanged", h is not None and h == logged,
                    f"frozen models of {task} changed or missing after the dev stage ({h} != {logged})")
            with open(mf, "rb") as fh:
                frozen[task] = pickle.load(fh)
        rows = [json.loads(x) for x in open(pr / "cells.jsonl", encoding="utf-8") if x.strip()]
        units.append({"doc_set": doc_set, "probe_run": pr, "features_run": fr, "common_run": cr,
                      "dev": DevStore(fr, cr, p, cbow), "rows": rows, "frozen": frozen, "repro": {}})
    ctx: dict[str, Any] = {"cid": cid, "kind": kind, "units": units}
    try:
        ctx["references"] = CF.preflight_references(list(p["tasks"]))
    except CF.ConfirmPreflightFailed as exc:
        _refuse(rec, "confirm.preflight.references", False, f"reference missing or wrong ({exc})", error=str(exc))
    rec.check("confirm.preflight.references", True, **{t: v for t, v in ctx["references"].items()})
    host = preflight_host(list(p["tasks"]), ctx["references"])
    _refuse(rec, "confirm.preflight.host_publishes_the_referenced_test_files", host["passed"],
            f"a test file must be downloaded inside the access and the host does not confirm its reference now ({host})", **host)
    ctx["encode"] = make_encoder(int(p["vocab"]))
    srcs = {r["set"].partition(".")[0] for u in units for r in u["rows"]}
    names = sorted(s for s in srcs if s not in ("bow", "cluster", "ppmi", "cbow"))
    ctx["names"] = names
    xp = np
    if args.engine == "cuda":
        import cupy as xp  # noqa: F811
    ctx["xp"] = xp
    missing: list[str] = []
    need = (lambda path: missing.append(str(path)) if not Path(path).exists() else None)
    for u in units:
        for r in u["rows"]:
            src, _, var = r["set"].partition(".")
            if var == "dim":
                need(u["dev"].fr / f"dim_{src}.npy")
            if src in names:
                need(u["dev"].fr / f"features_{src}.npz")
            if src == "ppmi":
                need((u["dev"].br or u["dev"].fr) / "ppmi_svd.npy")
            if src == "cbow":
                need(cbow)
    main = units[0]["dev"]
    if kind == "common":
        need(main.fr / "gplus_select.npz")
    elif kind == "rand":
        need(main.fr / "random_bank.npz")
        need(main.fr / "single_counts.npy")
    if "cbow" in srcs or kind == "common":
        need(cbow)
    _refuse(rec, "confirm.preflight.artefact_files_exist", not missing, f"artefact files missing: {sorted(set(missing))}",
            missing=sorted(set(missing)))
    if kind == "arm":
        train_run = R.rp(str(metric(main.fr, "features.train_run")))
        for u in units[1:]:
            _refuse(rec, "confirm.preflight.full_features_same_train_run",
                    R.rp(str(metric(u["dev"].fr, "features.train_run"))) == train_run, "the full-pool features run belongs to another train run")
        arm, seed = C.train_run_identity(train_run)
        base, _ = C.CONTROL_ARMS[arm]
        prefix = R.load_prefix(p, rec)
        ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
        ctx["tables"] = ad.feature_tables()
        ckpts = {}
        for name in names:
            cfg, st, meta = C.load_named_checkpoint(train_run, name, arm, seed, p, rec)
            for u in units:
                f = u["dev"].fr / f"features_{name}.npz"
                if f.exists():
                    with np.load(f) as z:
                        dev_hash = str(z["state_hash"])
                    _refuse(rec, f"confirm.{name}.state_hash_equals_{u['doc_set']}_features", dev_hash == meta["state_hash"],
                            f"checkpoint {name} is not the state the dev codes were made with", dev=dev_hash, checkpoint=meta["state_hash"])
            ckpts[name] = (cfg, st, meta)
        _refuse(rec, "confirm.preflight.checkpoint_hashes_equal_train_run",
                not [n for n in rec.failed if n.endswith(".state_hash_equals_train_run")], "a checkpoint state hash differs from the train run")
        ctx["ckpts"] = ckpts
    elif kind == "common":
        ctx["kind_map"] = main.kind_map()
        ctx["gplus"] = K.GPlus.load(main.fr / "gplus_select.npz")
        E = np.load(main.fr / "ppmi_svd.npy")
        W = np.load(cbow)
        _refuse(rec, "confirm.preflight.dense_tables_load", E.shape[0] == int(p["vocab"]) and W.shape == (int(p["vocab"]), int(cj["baselines"]["cbow"]["dim"])),
                "PPMI-SVD or CBOW table has the wrong shape", ppmi=list(E.shape), cbow=list(W.shape))
    elif kind == "rand":
        bank = CF.RandomBank.load(main.fr / "random_bank.npz")
        single = np.load(main.fr / "single_counts.npy")
        ctx["kind_map"] = main.kind_map()
        ctx["firer"] = CF.RandomBankFirer(xp, bank, single, ctx["kind_map"])
    # the stored dev predictions, recomputed from the loaded models
    bad, n = 0, 0
    for u in units:
        for task in p["tasks"]:
            drows = u["dev"].dev_rows(task)
            dev_y = np.asarray(u["dev"].labels[task]["dev_y"], np.int64)
            cache: dict = {}
            for r in [r for r in u["rows"] if r["task"] == task]:
                f = u["frozen"][task][r["model_key"]]
                if u["doc_set"] == "full":
                    Xd = full_matrix(u["dev"], r["set"], drows)
                else:
                    Xd = dev_matrix(u["dev"], r["set"], f.kind, drows, f.extra.get("edges"), cache)
                dpred = CF.frozen_predict(f, Xd)
                same = bool(np.array_equal(dpred.astype(np.int8), f.extra["dev_pred"])) and float(np.mean(dpred == dev_y)) == r["acc"]
                u["repro"][(task, r["model_key"])] = same
                bad += int(not same)
                n += 1
        u["dev"]._cache = {k: v for k, v in u["dev"]._cache.items() if k == "kind_map"}   # free the dev codes
    _refuse(rec, "confirm.dev_predictions_reproduced_by_frozen_models", bad == 0 and n > 0,
            f"{bad} of {n} frozen cells do not reproduce their stored dev predictions", mismatches=bad, cells=n)
    # the post-access path itself (codes, probe inputs, frozen models) on dev documents, before the access
    try:
        dry = preflight_dry_run(ctx, cj, args, rec)
    except Exception as exc:  # any error of the test-time path costs no access
        dry = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    _refuse(rec, "confirm.preflight.dry_run_on_dev_documents_reproduces_dev_predictions", bool(dry.get("passed")),
            f"the test-time path does not reproduce the dev predictions on dev documents ({dry})", **dry)
    _refuse(rec, "confirm.preflight.no_failed_check", not rec.failed, f"failed checks before the access: {rec.failed}",
            failed=list(rec.failed))
    rec.m("confirm.preflight", {"passed": True, "units": [u["doc_set"] for u in units], "cells": n, "names": names,
                                "dry_run": dry})
    rec.flush()
    return ctx


def confirm_sparse_codes(ctx: dict, docs: F.TokenDocs, cj: dict, args, rec: R.Rec, tag: str = "test") -> dict:
    """Clause-like count codes of the confirmatory documents for every source the frozen cells read, from the
    objects the preflight loaded (`tag` names the engine checks: 'test' after the access, 'preflight' in the dry run
    on dev documents before it)."""
    p = cj["params"]
    sp: dict[str, F.SparseDocs] = {}
    xp = ctx["xp"]
    kind = ctx["kind"]
    if kind == "arm":
        corpus = F.corpus_arrays(docs, ctx["tables"])
        for name in ctx["names"]:
            cfg, st, meta = ctx["ckpts"][name]
            eng = R.make_engine(args.engine, cfg, corpus)
            eng.load_state(st)
            sp[name] = R.bank_codes(eng, cfg, docs, int(p["forward_chunk"]), rec, f"{tag}.{name}")
            R.pipeline_checks(eng, cfg, docs, sp[name], rec, f"{tag}.{name}")
            del eng
    elif kind == "common":
        if "gplus" in ctx["names"]:
            sp["gplus"] = K.gplus_features(xp, ctx["gplus"], docs.tok, docs.doc_start, ctx["kind_map"], chunk=int(p["gplus_feature_chunk"]))
    elif kind == "rand":
        sp["rand"] = CF.random_bank_doc_counts(xp, ctx["firer"], docs.tok, docs.doc_start, int(cj["random_bank"]["chunk_positions"]))
    return sp


def subset_docs(docs: F.TokenDocs, rows: np.ndarray) -> F.TokenDocs:
    """The documents `rows` of a doc set, in that order, as their own TokenDocs (tokens copied, starts rebased)."""
    ds = docs.doc_start.astype(np.int64)
    rows = np.asarray(rows, np.int64)
    parts = [np.asarray(docs.tok[ds[r]:ds[r + 1]]) for r in rows.tolist()]
    tok = np.concatenate(parts) if parts else np.asarray(docs.tok[:0])
    st = np.zeros(rows.shape[0] + 1, np.int64)
    st[1:] = np.cumsum([x.shape[0] for x in parts]) if parts else []
    return F.TokenDocs(tok=tok.astype(np.uint16), doc_start=st.astype(np.uint32), end_id=docs.end_id)


def preflight_dry_run(ctx: dict, cj: dict, args, rec: R.Rec) -> dict:
    """The post-access path run BEFORE the access on DEV documents: the first n dev documents of every task
    (test_discipline.preflight_dry_run_docs) are coded by confirm_sparse_codes from the objects the preflight loaded
    (engines, G+ selection, random bank), turned into probe inputs by ConfirmReps (bag of words, cluster bags in the
    dev column space, PPMI-SVD and CBOW mean vectors, dimension-matched columns) and classified by the frozen models,
    exactly as the confirmatory documents will be. Every prediction must equal the stored dev prediction of that
    document, so the test-time path is shown to reproduce the dev-time path, and a code or device error of that path
    stops the stage before any access."""
    p = cj["params"]
    k = int(cj["test_discipline"].get("preflight_dry_run_docs", 16))
    tasks = list(p["tasks"])
    u0 = ctx["units"][0]
    parts = [subset_docs(u0["dev"].docs, u0["dev"].dev_rows(t)[:k]) for t in tasks]
    docs, _ = F.concat(parts)
    layout, base = {}, 0
    for t, part in zip(tasks, parts):
        layout[t] = (base, base + part.n_doc)
        base += part.n_doc
    failed0 = list(rec.failed)
    sp = confirm_sparse_codes(ctx, docs, cj, args, rec, tag="preflight")
    bad: list[str] = []
    n = 0
    for name, sd in sp.items():         # the codes themselves equal the stored dev codes of these documents
        ref = u0["dev"].sparse(name)
        for t in tasks:
            a, b = layout[t]
            x, y = sd.rows(np.arange(a, b, dtype=np.int64)), ref.rows(u0["dev"].dev_rows(t)[:k])
            if not (np.array_equal(np.diff(x.indptr), np.diff(y.indptr)) and np.array_equal(x.ids, y.ids)
                    and np.array_equal(x.cnt, y.cnt)):
                bad.append(f"codes:{name}:{t}")
        u0["dev"]._cache.pop(name, None)
    for u in ctx["units"]:
        for t, part in zip(tasks, parts):   # the same dev documents in this unit's doc set (full pool: checked by fullfeat)
            mine = subset_docs(u["dev"].docs, u["dev"].dev_rows(t)[:k])
            if not (np.array_equal(mine.tok, part.tok) and np.array_equal(mine.doc_start, part.doc_start)):
                bad.append(f"{u['doc_set']}:{t}:documents differ from the r1 unit's dev documents")
        test = ConfirmReps(docs, sp, u["dev"], p, int(p["vocab"]), ctx.get("kind_map"))
        for t in tasks:
            a, b = layout[t]
            trows = np.arange(a, b, dtype=np.int64)
            cache: dict = {}
            for r in [r for r in u["rows"] if r["task"] == t]:
                f = u["frozen"][t][r["model_key"]]
                if u["doc_set"] == "full":
                    Xt = full_matrix(test, r["set"], trows)
                else:
                    Xt = dev_matrix(test, r["set"], f.kind, trows, f.extra.get("edges"), cache)
                pred = CF.frozen_predict(f, Xt).astype(np.int8)
                n += 1
                if not np.array_equal(pred, np.asarray(f.extra["dev_pred"])[: b - a]):
                    bad.append(f"{u['doc_set']}:{t}:{r['model_key']}")
    # the post-access order is: frozen probes applied (tmu imported) THEN the confirmatory documents coded on the device;
    # code the same documents again now, so a device-context error of that order stops the stage before any access
    sp2 = confirm_sparse_codes(ctx, docs, cj, args, rec, tag="preflight_recode")
    for name, sd in sp2.items():
        s1 = sp[name]
        if not (np.array_equal(sd.indptr, s1.indptr) and np.array_equal(sd.ids, s1.ids) and np.array_equal(sd.cnt, s1.cnt)):
            bad.append(f"recode:{name}")
    new_failed = [x for x in rec.failed if x not in failed0]
    return {"docs_per_task": k, "documents": int(docs.n_doc), "cells": n, "mismatches": bad[:50], "n_mismatches": len(bad),
            "failed_checks": new_failed, "recoded_after_probes": True, "passed": bool(n > 0 and not bad and not new_failed)}


def stage_confirm(args, cj: dict, rec: R.Rec, probe_run: Path, probe_full_run: Path | None = None) -> Path:
    p = cj["params"]
    out = out_dir("confirm")
    tasks = list(p["tasks"])
    ctx = preflight_confirm(args, cj, rec, probe_run, probe_full_run)
    cid, kind = ctx["cid"], ctx["kind"]
    rec.m("confirm.config_id", cid)
    rec.m("confirm.kind", kind)
    rec.m("confirm.probe_run", str(probe_run))
    rec.m("confirm.probe_full_run", str(probe_full_run) if probe_full_run else None)
    before = {t: len(CF.prior_accesses(cid, t)) for t in tasks}
    rec.m("confirm.prior_accesses_before", before, note="lines of the test-access log for (this configuration, task)")
    rec.flush()
    detail = f"stage=confirm; probe_run={probe_run.name}" + (f"; probe_full_run={probe_full_run.name}" if probe_full_run else "")
    docs, layout, labels = confirm_docs(tasks, cid, detail, ctx["encode"], args.reaccess_reason, rec)
    after = {t: len(CF.prior_accesses(cid, t)) for t in tasks}
    rec.m("confirm.prior_accesses_after", after)
    rec.check("confirm.one_counted_access_per_configuration_and_task", all(after[t] - before[t] == 1 for t in tasks),
              before=before, after=after)
    np.savez(out / "confirm_tokens.npz", tok=docs.tok, doc_start=docs.doc_start, end_id=np.int64(docs.end_id),
             layout=json.dumps(layout), labels=json.dumps({t: v.tolist() for t, v in labels.items()}))
    sp = confirm_sparse_codes(ctx, docs, cj, args, rec)
    n = 0
    with open(out / "cells.jsonl", "w", encoding="utf-8") as fh:
        for u in ctx["units"]:
            test = ConfirmReps(docs, sp, u["dev"], p, int(p["vocab"]), ctx.get("kind_map"))
            for task in tasks:
                frozen = u["frozen"][task]
                a, b = layout[task]
                trows = np.arange(a, b, dtype=np.int64)
                ty = labels[task]
                preds = {}
                test_cache: dict = {}
                for r in [r for r in u["rows"] if r["task"] == task]:
                    f = frozen[r["model_key"]]
                    if u["doc_set"] == "full":
                        Xt = full_matrix(test, r["set"], trows)
                    else:
                        Xt = dev_matrix(test, r["set"], f.kind, trows, f.extra.get("edges"), test_cache)
                    tpred = CF.frozen_predict(f, Xt)
                    preds[r["model_key"]] = tpred.astype(np.int8)
                    row = {**{k: r[k] for k in ("config_id", "arm", "seed", "set", "clf", "task", "budget", "draw", "choice",
                                                "n_labelled", "model_key")},
                           "split": "test", "doc_set": u["doc_set"], "acc": float(np.mean(tpred == ty)), "n_eval": int(ty.shape[0]),
                           "dev_acc": r["acc"], "dev_prediction_reproduced": u["repro"][(task, r["model_key"])]}
                    fh.write(json.dumps(row, default=str) + "\n")
                    n += 1
                tag = "" if u["doc_set"] == "r1" else "_full"
                np.savez_compressed(out / f"test_predictions{tag}_{task}.npz", y=ty, keys=np.array(list(preds)),
                                    pred=np.stack(list(preds.values())) if preds else np.zeros((0, 0)))
                rec.log(f"{task} ({u['doc_set']}): {len(preds)} frozen cells applied")
    rec.m("confirm.cells", n)
    rec.m("confirm.cells_sha256", hashing.sha256_file(out / "cells.jsonl"))
    rec.flush()
    return out


# ------------------------------------------------------------------ Q_con (A7 word code)
def qcon_sets(pairs_by_ds: dict[str, list], words: list[str], covered: np.ndarray) -> dict[str, dict]:
    """Per dataset the covered pairs as word indices (both words covered) and the gold scores."""
    index = {w: i for i, w in enumerate(words)}
    out = {}
    for name, pairs in pairs_by_ds.items():
        a, b, g = [], [], []
        for w1, w2, s in pairs:
            i, j = index.get(w1), index.get(w2)
            if i is not None and j is not None and covered[i] and covered[j]:
                a.append(i)
                b.append(j)
                g.append(s)
        out[name] = {"a": np.asarray(a, np.int64), "b": np.asarray(b, np.int64), "gold": np.asarray(g, np.float64),
                     "n_total": len(pairs), "n_covered": len(a)}
    return out


def qcon_word_codes(eng, engine_kind: str, occ: CF.Occurrences, n_tok: int, doc_of_pos: np.ndarray) -> tuple:
    rounds = CF.occurrence_rounds(occ.start, doc_of_pos)
    if engine_kind == "cuda":
        return CF.word_codes_cuda(eng, occ, n_tok, rounds)
    return CF.word_codes_api(eng, occ, n_tok, rounds)


def qcon_eval(label: str, pairs_by_ds: dict[str, list], tokenize, prefix, word_id, banks: dict, E: dict, qc: dict,
              engine_kind: str, rec: R.Rec, cache: dict, dims: dict | None = None) -> dict:
    """Q_con of every representation on the pair sets `pairs_by_ds` (one common covered pair set). The word code of
    a word depends only on its own occurrences (each hidden alone in its document), so codes are cached per word and
    only words not coded before are run through the bank. With `dims` ({bank: dimension-matched clause ids}) the
    dimension-matched word code is reported as '<bank>.dim' next to the all-clauses code '<bank>' (A9 item 2)."""
    words = sorted({w for pairs in pairs_by_ds.values() for w1, w2, _ in pairs for w in (w1, w2)})
    targets = {w: tokenize(w) for w in words}
    occ = CF.build_occurrences(np.asarray(prefix.tok), word_id, targets, int(qc["max_occurrences"]))
    covered = occ.n_total >= int(qc["min_occurrences"])
    sets = qcon_sets(pairs_by_ds, words, covered)
    rec.m(f"qcon.{label}.coverage", {k: {"covered": v["n_covered"], "total": v["n_total"]} for k, v in sets.items()},
          words=len(words), covered_words=int(covered.sum()), occurrences_selected=int(occ.start.shape[0]))
    ds = np.asarray(prefix.doc_start, np.int64)
    dop = np.repeat(np.arange(ds.shape[0] - 1), np.diff(ds))
    res = {}
    for name, (cfg, st, make) in banks.items():
        wc = cache.setdefault(name, {})
        todo = [i for i, w in enumerate(words) if w not in wc]
        if todo:
            keep = np.isin(occ.widx, np.asarray(todo, np.int64))
            sub = CF.Occurrences(occ.words, occ.widx[keep], occ.start[keep], occ.length[keep], occ.n_total)
            eng = make()
            t0 = time.time()
            w, j, c = qcon_word_codes(eng, engine_kind, sub, int(prefix.tok.shape[0]), dop)
            rec.m(f"qcon.{label}.{name}.codes_wall_s", time.time() - t0, words_coded=len(todo), occurrences=int(keep.sum()))
            del eng
            for i in todo:
                m = w == i
                wc[words[i]] = (j[m], c[m])
        ww = np.concatenate([np.full(wc[x][0].shape[0], i, np.int64) for i, x in enumerate(words)]) if words else np.zeros(0, np.int64)
        jj = np.concatenate([wc[x][0] for x in words]) if words else np.zeros(0, np.int64)
        cc = np.concatenate([wc[x][1] for x in words]) if words else np.zeros(0, np.int64)
        X = CF.ppmi_rows(ww, jj, cc, len(words), int(cfg.n_clauses))
        res[name] = CF.q_con_from_similarity(sets, lambda a, b, X=X: CF.pair_cosines_sparse(X, a, b))
        rec.m(f"qcon.{label}.{name}", res[name], note="Spearman per dataset on the covered pairs; q_con = mean of the three")
        if dims is not None and name in dims:
            wd, jd, cd = CF.restrict_columns(ww, jj, cc, dims[name])
            Xd = CF.ppmi_rows(wd, jd, cd, len(words), int(np.asarray(dims[name]).shape[0]))
            res[f"{name}.dim"] = CF.q_con_from_similarity(sets, lambda a, b, X=Xd: CF.pair_cosines_sparse(X, a, b))
            rec.m(f"qcon.{label}.{name}.dim", res[f"{name}.dim"], note="dimension-matched word code (the bank's dim_<name>.npy columns), reported (A9 item 2)")
    tok_lists = [targets[w] for w in words]
    for name, M in E.items():
        Wv = CF.word_vectors_mean(M, tok_lists)
        res[name] = CF.q_con_from_similarity(sets, lambda a, b, Wv=Wv: (Wv[a] * Wv[b]).sum(axis=1))
        rec.m(f"qcon.{label}.{name}", res[name], note="word vector = mean of the subtoken vectors")
    rec.flush()
    return res


def word_tokenizer(vocab: int):
    """word -> BPE-8k ids of ' ' + word, the mid-sentence form (tests replace this function)."""
    from tmlm.data import bpe

    tok = bpe.load_tokenizer(int(vocab))
    return lambda w: list(tok.encode(" " + w).ids)


def stage_qcon(args, cj: dict, rec: R.Rec, train_run: Path, probe_run: Path, common_run: Path) -> Path:
    """Q_con of one configuration (A7 word code; docs/13 A9 item 6): the dev Q_con (SimLex-999, WordSim-353, MEN dev)
    first; then ONE counted access of the MEN test pairs for this configuration; the confirmatory Q_con on
    SimLex-999 and WordSim-353 in full and MEN in full (its 2,000 dev and 1,000 test pairs), all clauses (decides
    H17.3) and dimension-matched (reported)."""
    from tmlm.data import corpus_store as CS
    from tmlm.ssl import concepts as KC

    p = cj["params"]
    qc = cj["qcon"]
    out = out_dir("qcon")
    ok, info = run_ok(probe_run)
    rec.check("qcon.dev_probe_run_ok", ok, **info)
    if not ok:
        raise SystemExit(f"the dev stage of this configuration did not succeed ({info}): no MEN test access")
    arm, seed = C.train_run_identity(train_run)
    cid = config_id("arm", arm, seed)
    if str(metric(probe_run, "confirm.config_id")) != cid:
        raise SystemExit("the probe run belongs to another configuration")
    rec.m("confirm.config_id", cid)
    men_ref = CF.men_test_reference()          # before any work: the reference of the counted MEN file exists
    rec.m("qcon.men_test_reference", men_ref)
    features_run = R.rp(str(metric(probe_run, "probe.features_run")))
    dims = {name: np.load(features_run / f"dim_{name}.npy") for name in qc["checkpoints"]}
    base, _ = C.CONTROL_ARMS[arm]
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    corpus = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    word_id = np.asarray(CS.load_word_id(R.CORPUS, int(p["vocab"]), "train"))[: int(prefix.tok.shape[0])]
    tokenize = word_tokenizer(int(p["vocab"]))
    banks = {}
    for name in qc["checkpoints"]:
        cfg, st, _ = C.load_named_checkpoint(train_run, name, arm, seed, p, rec)
        banks[name] = (cfg, st, (lambda cfg=cfg, st=st: _loaded(R.make_engine(args.engine, cfg, corpus), st)))
    E = {"ppmi_svd": np.load(common_run / "ppmi_svd.npy"), "cbow": np.load(ROOT / cj["baselines"]["cbow"]["file"])}
    parsers = {"simlex": KC.parse_simlex, "ws353": KC.parse_ws353, "men_lemma": KC.parse_men_lemma}
    common_sets = {}
    for name, d in qc["datasets"].items():
        path = ROOT / d["path"]
        common_sets[name] = parsers[d["parser"]](path)
        R.dataset_hashes({f"{name}:{d['path']}": hashing.sha256_file(path)})
    men_dev = KC.parse_men_lemma(ROOT / qc["men_dev"]["path"])
    rec.m("qcon.dataset_accesses", {**{k: 1 for k in common_sets}, "men_dev": 1}, note="SimLex-999 and WordSim-353 have no split")
    cache: dict = {}
    dev_res = qcon_eval("dev", {**common_sets, "men": men_dev}, tokenize, prefix, word_id, banks, E, qc, args.engine, rec,
                        cache, dims)
    rec.m("qcon.dev_complete", True)
    rec.flush()
    key = f"qcon_{cid}"
    before = len(CF.prior_accesses(key, "men_test"))
    men_test, minfo = CF.load_men_test(key, f"stage=qcon; probe_run={probe_run.name}", reaccess_reason=args.reaccess_reason)
    after = len(CF.prior_accesses(key, "men_test"))
    rec.m("qcon.men_test_access", minfo)
    rec.check("qcon.one_counted_men_access_for_this_configuration", after - before == 1, before=before, after=after)
    men_full = list(men_dev) + list(men_test)
    rec.m("qcon.men_full_pairs", {"dev": len(men_dev), "test": len(men_test), "full": len(men_full)},
          note="A9 item 6: MEN in full (2,000 dev and 1,000 test pairs) for the confirmatory Q_con")
    conf = qcon_eval("confirm", {**common_sets, "men": men_full}, tokenize, prefix, word_id, banks, E, qc, args.engine, rec,
                     cache, dims)
    summary = {"dev": {k: v["q_con"] for k, v in dev_res.items()}, "confirm": {k: v["q_con"] for k, v in conf.items()}}
    rec.m("qcon.summary", summary, label="measured")
    (out / "qcon.json").write_text(json.dumps({"dev": dev_res, "confirm": conf}, indent=1, default=str) + "\n")
    rec.flush()
    return out


def _loaded(eng, st):
    eng.load_state(st)
    return eng


# ------------------------------------------------------------------ Q_int (NPMI coherence, word intrusion)
def vocab_texts(vocab: int) -> list[str]:
    """Surface text of every BPE token id (tests replace this function)."""
    from tmlm.data import bpe
    from tmlm.ssl import concepts as KC

    return KC.token_texts(bpe.token_bytes(bpe.load_tokenizer(int(vocab))))


def stage_qint(args, cj: dict, rec: R.Rec, train_run: Path, features_run: Path, probe_run: Path) -> Path:
    """Q_int of one configuration (docs/13 A9 item 7): the concept group of each of the 8,192 dimension-matched
    clauses = its top 10 words by PPMI with the clause's firings (at least 5 counts); NPMI over a 10-token window
    with the WikiText-103 train tokens AFTER the pretraining prefix as the PRIMARY reference (the validation split is
    the secondary reference). No test access: checked on this process's access-log lines (the log's pid), which
    other parallel jobs' counted accesses cannot change."""
    from tmlm.data import corpus_store as CS
    from tmlm.data.evalsets import test_access_count
    from tmlm.ssl import concepts as KC

    p = cj["params"]
    qi = cj["qint"]
    out = out_dir("qint")
    ok, info = run_ok(probe_run)
    rec.check("qint.dev_probe_run_ok", ok, **info)
    arm, seed = C.train_run_identity(train_run)
    cid = config_id("arm", arm, seed)
    rec.m("confirm.config_id", cid)
    rec.m("qint.primary_reference", qi["primary_reference"])
    own0 = len(CF.own_accesses())
    rec.m("qint.test_accesses_run_wide_before", test_access_count(), note="reported only; other parallel jobs may access")
    base, _ = C.CONTROL_ARMS[arm]
    V = int(p["vocab"])
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, V, float(p["f1_t"]))
    corpus = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    texts = vocab_texts(V)
    word_ok = KC.word_like_mask(texts, int(qi["min_word_len"])) & (np.asarray(prefix.stop)[:V] == 0)
    rec.m("qint.word_like_tokens", int(word_ok.sum()))
    _, _, cnt_pos = C.qpred_split(prefix)
    tt = np.asarray(prefix.tok).astype(np.int64)
    zero = np.zeros(int(prefix.tok.shape[0]), np.uint8)
    refs = {}
    for rname, spec in qi["references"].items():
        if spec["split"] == "validation":
            ref = CS.load_split(R.CORPUS, V, "validation")
            rt, rd = np.asarray(ref.tok), np.asarray(ref.doc_start)
        elif spec["split"] == "train_remainder":
            full = CS.load_split(R.CORPUS, V, "train")
            ds = np.asarray(full.doc_start, np.int64)
            cut = int(prefix.tok.shape[0])
            d0 = int(np.searchsorted(ds, cut))
            rt = np.asarray(full.tok)[cut:]
            rd = (ds[d0:] - cut).astype(np.int64)
            if int(ds[d0]) != cut:
                raise SystemExit("the prefix does not end at a document boundary")
        else:
            raise KeyError(spec["split"])
        t0 = time.time()
        refs[rname] = CF.NPMI(CF.cooc_window(rt, rd, V, int(qi["window"])))
        rec.m(f"qint.reference.{rname}", {"tokens": int(rt.shape[0]), "docs": int(rd.shape[0] - 1), "window": int(qi["window"]),
                                          "wall_s": time.time() - t0})
    saved = {}
    for name in qi["checkpoints"]:
        cfg, st, _ = C.load_named_checkpoint(train_run, name, arm, seed, p, rec)
        eng = R.make_engine(args.engine, cfg, corpus)
        eng.load_state(st)
        t0 = time.time()
        counts, n_ev, _ = C.count_table(eng, args.engine, cnt_pos, tt[cnt_pos], int(cfg.n_clauses), V, int(p["forward_chunk"]), zero)
        del eng
        rec.m(f"qint.{name}.count_pass_wall_s", time.time() - t0)
        rec.check(f"qint.{name}.counted_events_equal_table", int(counts.sum()) == int(n_ev))
        import scipy.sparse as sp
        O = sp.csr_matrix(counts)
        del counts
        clauses = np.load(features_run / f"dim_{name}.npy")
        groups, qual = CF.top_words_ppmi(O, clauses, word_ok, int(qi["top_words"]), int(qi["min_count"]))
        g = groups[qual]
        rec.m(f"qint.{name}.groups", {"clauses": int(clauses.shape[0]), "qualifying": int(qual.sum()),
                                      "fraction_qualifying": float(qual.mean()) if qual.size else None})
        for rname, npmi in refs.items():
            coh = CF.group_coherence(g, npmi)
            intr = CF.intruder_task(g, npmi, int(qi["intruder_seed"]), int(qi["intruder_keep"]))
            saved[f"{name}.{rname}.npmi"] = coh
            rec.m(f"qint.{name}.{rname}.npmi_mean", float(np.nanmean(coh)) if coh.size else None, groups=int(coh.shape[0]))
            rec.m(f"qint.{name}.{rname}.intruder_accuracy", intr["accuracy"], scored=intr["n_groups_scored"], chance=intr["chance"])
        saved[f"{name}.groups"] = g
        saved[f"{name}.clauses"] = clauses[qual]
        rec.m(f"qint.{name}.examples", [[texts[int(t)] for t in row if t >= 0] for row in g[:10]], note="first 10 concept groups")
        rec.flush()
    np.savez_compressed(out / "qint_groups.npz", **saved)
    rec.m("qint.test_accesses_run_wide_after", test_access_count(), note="reported only; other parallel jobs may access")
    rec.check("qint.no_test_access_by_this_process", len(CF.own_accesses()) == own0, own_before=own0,
              own_after=len(CF.own_accesses()))
    rec.flush()
    return out


# ------------------------------------------------------------------ report
def read_cells(runs: list[Path]) -> list[dict]:
    rows = []
    for d in runs:
        with open(d / "cells.jsonl", encoding="utf-8") as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    return rows


def q_table(rows: list[dict], tasks: list[str]) -> dict:
    """(arm, seed, set, clf, budget, split) -> {draw: Q_text}."""
    q: dict = {}
    for split in ("dev", "test"):
        for k, v in R.summarise([r for r in rows if r.get("split") == split], tasks).items():
            q[k + (split,)] = v
    return q


def mean_q(d: dict | None) -> float | None:
    return float(np.mean([d[k] for k in sorted(d)])) if d else None


def diff_units(q: dict, a: tuple, b_of_seed, seeds: list[int], split: str = "test", budget: str = "32") -> dict:
    """{seed: per-draw differences in points} of a = (arm, set, clf) minus b_of_seed(seed) -> (arm, seed, set, clf)."""
    out = {}
    for s in seeds:
        A = q.get((a[0], s, a[1], a[2], budget, split))
        bkey = b_of_seed(s)
        B = q.get(bkey[:4] + (budget, split)) if bkey else None
        if not A or not B:
            continue
        common = sorted(set(A) & set(B))
        out[s] = 100.0 * np.array([A[d] - B[d] for d in common])
    return out


def compare(q: dict, a: tuple, b_of_seed, seeds: list[int], st: dict, split: str = "test", budget: str = "32") -> dict:
    u = diff_units(q, a, b_of_seed, seeds, split, budget)
    return {"boot": CF.two_level_bootstrap(u, int(st["n_boot"]), int(st["seed"])) if u else None,
            "t": CF.seed_t_interval(u) if u else None, "units": {str(k): v.tolist() for k, v in u.items()}}


def stronger(q: dict, cands: list[tuple], split: str, budget: str, require_all: bool = True) -> tuple | None:
    """The candidate (arm, seed, set, clf) with the higher mean Q_text over draws (ties: the first). With
    `require_all` (the default) a candidate without cells makes the result None, so a missing control makes the
    seed's unit missing (the row INCOMPLETE) instead of silently comparing against the other candidates only."""
    best, bm = None, None
    if not cands:
        return None
    for c in cands:
        v = mean_q(q.get(c + (budget, split)))
        if v is None:
            if require_all:
                return None
            continue
        if bm is None or v > bm:
            best, bm = c, v
    return best


def young_sets(q: dict, arm: str, seed: int, code: str, split: str, budget: str) -> list[tuple]:
    return sorted({k[:4] for k in q if k[0] == arm and k[1] == seed and k[3] == "lr" and k[4] == budget and k[5] == split
                   and k[2].endswith("." + code) and (k[2].startswith("young_b") or k[2].startswith("c0y."))})


def young_expected(q: dict, arms: list[str], seed: int, code: str, budget: str) -> list[tuple] | None:
    """The young banks (C0y and ladder rungs) of `arms` at `seed` as the DEV cells list them (the configuration's
    frozen plan); None when an arm has none (its dev stage is missing): the caller's row is then incomplete."""
    out: list[tuple] = []
    for a in arms:
        s = young_sets(q, a, seed, code, "dev", budget)
        if not s:
            return None
        out += s
    return out


def expected_accesses(cj: dict) -> list[str]:
    """Every (configuration, task) key of A9 item 10: 28 configurations x 3 tasks, plus one MEN-test access per Q_con
    configuration."""
    p = cj["params"]
    arms = cj["configurations"]["arms"]
    cids = [config_id("arm", a, s) for a in arms for s in p["seeds"]] + ["common"]
    cids += [config_id("rand", seed=s) for s in cj["random_bank"]["seeds"]]
    keys = [CF.access_key(c, t).rstrip(";") for c in cids for t in p["tasks"]]
    keys += [CF.access_key(f"qcon_{config_id('arm', a, s)}", "men_test").rstrip(";") for a, spec in arms.items() if spec.get("qcon")
             for s in p["seeds"]]
    return keys


def access_audit(root: Path | None = None, expected: list[str] | None = None) -> dict:
    from tmlm.data.evalsets import access_log_path

    path = access_log_path(root)
    per: dict[str, int] = {}
    reacc: list[str] = []
    reasoned: dict[str, int] = {}
    total = 0
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            pur = str(r.get("purpose", ""))
            if pur.startswith(CF.ACCESS_TAG) or pur.startswith(CF.MEN_TAG):
                total += 1
                key = pur.split("; ")[1] + "; " + pur.split("; ")[2]
                per[key] = per.get(key, 0) + 1
                if "REACCESS" in pur:
                    reacc.append(pur)
                    reasoned[key] = reasoned.get(key, 0) + 1
    out = {"total_confirmatory_accesses": total, "per_config_task": per, "reaccesses": reacc,
           "every_pair_once": all(v == 1 for v in per.values()),
           "every_extra_access_has_a_logged_reason": all(v - 1 == reasoned.get(k, 0) for k, v in per.items())}
    if expected is not None:
        out["expected_pairs"] = len(expected)
        out["missing_pairs"] = [k for k in expected if k not in per]
        out["unexpected_pairs"] = [k for k in per if k not in set(expected)]
    return out


def compact(v: dict | None) -> dict | None:
    """Point, interval and verdict of a row (the side-by-side table)."""
    if not isinstance(v, dict):
        return None
    b = v.get("boot") or {}
    t = v.get("t") or {}
    return {"verdict": v.get("verdict"), "mean": b.get("mean", v.get("mean")), "boot_lo": b.get("lo"), "boot_hi": b.get("hi"),
            "t_lo": t.get("lo"), "t_hi": t.get("hi")}


def read_skipped(path: str | None) -> list[str]:
    if not path:
        return []
    f = Path(path)
    return [x.rstrip("\n") for x in open(f, encoding="utf-8")] if f.exists() else []


def stage_report(args, cj: dict, rec: R.Rec) -> Path:
    p = cj["params"]
    st = cj["statistics"]
    vt = cj["verdicts"]
    out = out_dir("report")
    split_list = lambda s: [R.rp(x) for x in (s or "").split(",") if x.strip()]  # noqa: E731
    rows = read_cells(split_list(args.probe_runs) + split_list(args.confirm_runs))
    tasks = list(p["tasks"])
    q = q_table(rows, tasks)
    BK = f"epoch{int(p['epochs'])}"   # the final bank
    seeds = [int(s) for s in p["seeds"]]
    n_seeds = len(seeds)
    arms = cj["configurations"]["arms"]
    top_two = [a for a, s in arms.items() if s["role"] in ("top1", "top2")]
    top_two.sort(key=lambda a: arms[a]["role"])
    code_v = vt["code"]                       # A9 item 2: "all" decides, "dim" is printed next to it
    other = "dim" if code_v == "all" else "all"
    lc = float(vt["literal_credit"]["min_points"])
    # ---- Q_con and Q_int inputs
    qcon: dict = {}
    for d in split_list(args.qcon_runs):
        cid = str(metric(d, "confirm.config_id"))
        qcon[cid] = metric(d, "qcon.summary")
    qint: dict = {}
    for d in split_list(args.qint_runs):
        cid = str(metric(d, "confirm.config_id"))
        qint[cid] = np.load(d / "qint_groups.npz")
    # ---- the winner (A9 item 1: fixed A3; the seeds 1 to 3 dev rule is printed, never used)
    dev_q = {a: (100.0 * np.mean([mean_q(q.get((a, s, f"{BK}.all", "lr", "32", "dev"))) for s in seeds])
                 if all(q.get((a, s, f"{BK}.all", "lr", "32", "dev")) for s in seeds) else None) for a in top_two}
    dev_qc = {a: (float(np.mean([qcon[f"{a}_seed{s}"]["dev"][BK] for s in seeds]))
                  if all(f"{a}_seed{s}" in qcon and qcon[f"{a}_seed{s}"]["dev"].get(BK) is not None for s in seeds) else None)
              for a in top_two}
    win = CF.winner_by_rule(cj["winner"]["rule"], cj["winner"]["seed100_winner"], top_two, dev_q, dev_qc,
                            float(cj["winner"]["tie_margin_points"]))
    win["note"] = "docs/13 A9 item 1: the winner is A3, fixed by the seed-100 tournament; dev_seeds_1_3 is printed, never used"
    rec.m("report.winner", win, label="derived")
    W = win["winner"]
    verdicts: dict[str, Any] = {"winner": win}
    twin = arms[W].get("twin_arm")
    # ---- H17.1 and the A8 rows, both codes
    rows_a8: dict = {}
    for code in CODES:
        bank = (W, f"{BK}.{code}", "lr")
        ctl_of = (lambda s, code=code: stronger(q, [(W, s, f"c0y.{code}", "lr"), ("common", 0, f"gplus.{code}", "lr")], "test", "32"))
        h171 = compare(q, bank, ctl_of, seeds, st)
        h171["stronger_control_per_seed"] = {str(s): ctl_of(s) for s in seeds}
        rows_a8[f"H17.1.{code}"] = CF.verdict_min_ci(f"H17.1 ({code})", h171["boot"], h171["t"], float(vt["H17.1"]["min_points"]), n_seeds)
        rows_a8[f"H17.1.{code}"]["inputs"] = h171
        b196 = int(cj["a8"]["exposure_matched_batch"])
        exp = compare(q, bank, lambda s, code=code: (W, s, f"young_b{b196}.{code}", "lr"), seeds, st)
        rows_a8[f"minus_exposure_matched.{code}"] = CF.estimate_row(f"winner - young_b{b196} ({code})", exp["boot"], exp["t"], n_seeds)
        rows_a8[f"minus_exposure_matched.{code}"]["inputs"] = exp
        lad_of = (lambda s, code=code: stronger(q, young_expected(q, [W], s, code, "32") or [], "test", "32"))
        lad = compare(q, bank, lad_of, seeds, st)
        lad["max_rung_per_seed"] = {str(s): lad_of(s) for s in seeds}
        rows_a8[f"ladder.{code}"] = CF.verdict_min_ci(f"winner - max(young ladder) ({code})", lad["boot"], lad["t"], lc, n_seeds,
                                                      rule=f"A8 literal-learning component: point >= {lc} and two-level bootstrap lo > 0")
        rows_a8[f"ladder.{code}"]["inputs"] = lad
        any_of = (lambda s, code=code: stronger(q, young_expected(q, list(cj["a8"]["young_any_arm"]), s, code, "32") or [], "test", "32"))
        anyy = compare(q, bank, any_of, seeds, st)
        anyy["best_young_per_seed"] = {str(s): any_of(s) for s in seeds}
        rows_a8[f"minus_best_young_any_arm.{code}"] = CF.estimate_row(f"winner - best young bank of any arm ({code})", anyy["boot"],
                                                                      anyy["t"], n_seeds)
        rows_a8[f"minus_best_young_any_arm.{code}"]["inputs"] = anyy
        tw = compare(q, bank, lambda s, code=code: (twin, s, f"{BK}.{code}", "lr"), seeds, st) if twin else {"boot": None, "t": None}
        rows_a8[f"twin.{code}"] = CF.verdict_min_ci(f"winner - twin {twin} ({code})", tw["boot"], tw["t"], lc, n_seeds,
                                                    rule=f"A8 literal-learning component: point >= {lc} and two-level bootstrap lo > 0")
        rows_a8[f"twin.{code}"]["inputs"] = tw
    verdicts["H17.1"] = rows_a8[f"H17.1.{code_v}"]
    verdicts[f"H17.1.{other}_reported"] = {**rows_a8[f"H17.1.{other}"], "decides": False,
                                           "note": "A9 item 2: reported next to the deciding all-clauses code"}
    verdicts["A8_rows"] = {k: v for k, v in rows_a8.items() if not k.startswith("H17.1")}
    verdicts["A8_literal_learning_credit"] = CF.literal_learning_label({k: rows_a8[k] for k in ("ladder.all", "ladder.dim", "twin.all", "twin.dim")}, lc)
    # ---- H17.2 (winner minus bag of words, same probe) and H17.4 (A1 minus A0, paired by seed)
    for hid, a_of, b_fun in (("H17.2", lambda code: (W, f"{BK}.{code}", "lr"), lambda code: (lambda s: ("common", 0, "bow", "lr"))),
                             ("H17.4", lambda code: ("A1", f"{BK}.{code}", "lr"), lambda code: (lambda s, code=code: ("A0", s, f"{BK}.{code}", "lr")))):
        for code in CODES:
            h = compare(q, a_of(code), b_fun(code), seeds, st)
            v = CF.verdict_min_ci(f"{hid} ({code})", h["boot"], h["t"], float(vt[hid]["min_points"]), n_seeds)
            v["inputs"] = h
            if code == code_v:
                verdicts[hid] = v
            else:
                verdicts[f"{hid}.{code}_reported"] = {**v, "decides": False, "note": "A9 item 2: reported next to the deciding all-clauses code"}
    # ---- H17.11 (TM probe minus LR probe on the winner's dimension-matched code; also against LR on all clauses)
    h11 = vt["H17.11"]
    for lr_set, key in ((h11["lr_set"], "H17.11"), (h11["reported_lr_set"], "H17.11.vs_all_reported")):
        u = diff_units(q, (W, h11["tm_set"].replace("epoch4", BK), "tm"), lambda s, ls=lr_set: (W, s, ls.replace("epoch4", BK), "lr"), seeds)
        per_seed = {str(s): float(v.mean()) for s, v in u.items()}
        v = CF.verdict_min_mean(f"H17.11 (TM dim - LR {lr_set})", per_seed, float(h11["min_points"]), n_seeds)
        v["boot"] = CF.two_level_bootstrap(u, int(st["n_boot"]), int(st["seed"])) if u else None
        if key != "H17.11":
            v["decides"] = False
        verdicts[key] = v
    # ---- H17.3 (Q_con: SimLex-999, WordSim-353, MEN in full; mean over seeds; all clauses decides, dim reported)
    get = (lambda key: {str(s): (qcon.get(f"{W}_seed{s}", {}).get("confirm", {}) or {}).get(key) for s in seeds})
    verdicts["H17.3"] = CF.verdict_h17_3(get(BK), get("ppmi_svd"), get("c0y"), n_seeds,
                                         float(vt["H17.3"]["min_vs_ppmi"]), float(vt["H17.3"]["min_vs_c0y"]))
    verdicts["H17.3.dim_reported"] = {**CF.verdict_h17_3(get(f"{BK}.dim"), get("ppmi_svd"), get("c0y.dim"), n_seeds,
                                                         float(vt["H17.3"]["min_vs_ppmi"]), float(vt["H17.3"]["min_vs_c0y"])),
                                      "decides": False, "note": "A9 item 2: the dimension-matched word code, reported"}
    verdicts["H17.3_reported"] = {"cbow": get("cbow"), "dev_variant": {str(s): (qcon.get(f"{W}_seed{s}", {}) or {}).get("dev") for s in seeds}}
    # ---- H17.12 (Q_int NPMI of the winner minus C0y; A9 item 7 primary reference; the other reference reported)
    for ref_name in [cj["qint"]["primary_reference"]] + [r for r in cj["qint"]["references"] if r != cj["qint"]["primary_reference"]]:
        units = {s: (qint[f"{W}_seed{s}"][f"{BK}.{ref_name}.npmi"], qint[f"{W}_seed{s}"][f"c0y.{ref_name}.npmi"]) for s in seeds
                 if f"{W}_seed{s}" in qint and f"{BK}.{ref_name}.npmi" in qint[f"{W}_seed{s}"].files}
        units = {s: (a[~np.isnan(a)], c[~np.isnan(c)]) for s, (a, c) in units.items()}
        b = CF.two_level_bootstrap_unpaired(units, int(st["n_boot"]), int(st["seed"])) if units else None
        t = CF.seed_t_interval({s: np.array([np.mean(a) - np.mean(c)]) for s, (a, c) in units.items()}) if units else None
        v = CF.verdict_above_ci(f"H17.12 ({ref_name})", b, t, n_seeds)
        if ref_name == cj["qint"]["primary_reference"]:
            verdicts["H17.12"] = v
        else:
            verdicts[f"H17.12.{ref_name}_reported"] = {**v, "decides": False, "note": "A9 item 7: secondary reference"}
    # ---- reported controls: the random kind_map arms A6 (A7) and A3R (A9 item 9); no threshold
    rep: dict = {}
    for ctl in cj["reported_controls"]:
        for code in CODES:
            c = compare(q, (W, f"{BK}.{code}", "lr"), lambda s, code=code, ctl=ctl: (ctl, s, f"{BK}.{code}", "lr"), seeds, st)
            rep[f"winner_minus_{ctl}.{code}"] = {**CF.estimate_row(f"winner - {ctl} ({code})", c["boot"], c["t"], n_seeds), "inputs": c}
            rep[f"{ctl}.{code}.q_text_32"] = {str(s): mean_q(q.get((ctl, s, f"{BK}.{code}", "lr", "32", "test"))) for s in seeds}
    verdicts["reported_controls"] = rep
    # ---- A9 item 8: Q_text at the full label pool (draw 0), reported with seed-level intervals, no threshold
    fb = cj["full_budget"]
    full: dict = {"note": "docs/13 A9 item 8: LR at the full label pool, draw 0; reported, no threshold", "q_text": {}, "differences": {}}
    for arm in fb["arms"]:
        for fs in [f"{BK}.all", f"{BK}.dim", "c0y.all", "c0y.dim"]:
            for split in ("dev", "test"):
                full["q_text"][f"{arm}.{fs}.{split}"] = {str(s): mean_q(q.get((arm, s, fs, "lr", "full", split))) for s in seeds}
    for fs in ("gplus.all", "gplus.dim", "bow"):
        for split in ("dev", "test"):
            full["q_text"][f"common.{fs}.{split}"] = mean_q(q.get(("common", 0, fs, "lr", "full", split)))
    for code in CODES:
        bank = (W, f"{BK}.{code}", "lr")
        ctl_of = (lambda s, code=code: stronger(q, [(W, s, f"c0y.{code}", "lr"), ("common", 0, f"gplus.{code}", "lr")], "test", "full"))
        for name, b_of in ((f"winner_minus_max_c0y_gplus.{code}", ctl_of), (f"winner_minus_bow.{code}", lambda s: ("common", 0, "bow", "lr"))):
            c = compare(q, bank, b_of, seeds, st, "test", "full")
            full["differences"][name] = {**CF.estimate_row(f"{name} (full pool)", c["boot"], c["t"], n_seeds), "inputs": c}
    verdicts["full_pool"] = full
    # ---- the side-by-side table (A9 item 2): every registered row, all clauses (decides) and dimension-matched
    side = {
        "H17.1": {"all": compact(rows_a8["H17.1.all"]), "dim": compact(rows_a8["H17.1.dim"])},
        "H17.2": {code_v: compact(verdicts["H17.2"]), other: compact(verdicts[f"H17.2.{other}_reported"])},
        "H17.4": {code_v: compact(verdicts["H17.4"]), other: compact(verdicts[f"H17.4.{other}_reported"])},
        "H17.3": {"all": {"verdict": verdicts["H17.3"].get("verdict"), "winner_minus_ppmi": (verdicts["H17.3"].get("winner_minus_ppmi") or {}).get("mean"),
                          "winner_minus_c0y": (verdicts["H17.3"].get("winner_minus_c0y") or {}).get("mean")},
                  "dim": {"verdict": verdicts["H17.3.dim_reported"].get("verdict"),
                          "winner_minus_ppmi": (verdicts["H17.3.dim_reported"].get("winner_minus_ppmi") or {}).get("mean"),
                          "winner_minus_c0y": (verdicts["H17.3.dim_reported"].get("winner_minus_c0y") or {}).get("mean")}},
        "A8": {k: compact(v) for k, v in verdicts["A8_rows"].items()},
    }
    verdicts["side_by_side"] = {"decides": code_v, "rows": side}
    # ---- Q_text tables, Q_pred, references, access audit, chain skips
    table = {".".join(str(x) for x in k): {"mean": mean_q(v), "draws": {str(d): v[d] for d in sorted(v)}} for k, v in q.items()}
    (out / "q_text_table.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    qpred = {}
    for d in split_list(args.qpred_runs):
        m = json.loads((d / "metrics.json").read_text())
        arm, seed = m["control.arm"]["value"], int(m["control.seed"]["value"])
        for k, v in m.items():
            if k.startswith("qpred.") and k.endswith(".loglin_top1"):
                qpred[f"{arm}.{seed}.{k[len('qpred.'):-len('.loglin_top1')]}"] = v["value"]
    refs = {}
    for name, (run, key) in cj["qpred_references"]["values"].items():
        try:
            refs[name] = {"value": metric(ROOT / run, key), "run": run, "key": key}
        except (KeyError, FileNotFoundError):
            refs[name] = None
    rec.m("report.q_pred", qpred, label="measured", note="T1 loglin masked top-1, frozen V 8000 validation targets (A8 item 1)")
    rec.m("report.q_pred_references", refs, note=cj["qpred_references"]["note"])
    audit = access_audit(expected=expected_accesses(cj))
    rec.m("report.test_access_audit", audit, label="measured")
    rec.check("report.every_configuration_task_accessed_once_or_reaccess_reasoned", audit["every_extra_access_has_a_logged_reason"],
              **{k: v for k, v in audit.items() if k != "per_config_task"})
    census = [str(d) for d in split_list(args.census_runs)]
    rec.m("report.census_runs", census, note="A8 item 6 diagnostics: metrics diag.* of each census run")
    skipped = read_skipped(args.skipped_file)
    rec.m("report.chain_skips_and_failures", skipped, file=args.skipped_file,
          note="every step the chain skipped or that failed (S5); an input missing because of one of these makes its row INCOMPLETE")
    verdicts["chain_skips_and_failures"] = skipped
    for k, v in verdicts.items():
        rec.m(f"verdict.{k}", v, label="derived")
    rec.log("VERDICTS " + json.dumps({k: (v.get("verdict") or v.get("label") or v.get("winner")) for k, v in verdicts.items()
                                      if isinstance(v, dict) and any(x in v for x in ("verdict", "label", "winner"))}, default=str))
    rec.log(f"CHAIN SKIPS AND FAILURES: {len(skipped)} line(s)" + "".join(f"\n  {x}" for x in skipped))
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=str) + "\n")
    rec.flush()
    return out


# ------------------------------------------------------------------ main
def cuda_context_first() -> str:
    """Import tmu's classifier before any CuPy work. tmu imports pycuda.autoinit, which pushes a new CUDA context and makes
    every CuPy module loaded earlier in the process fail with CUDA_ERROR_INVALID_HANDLE (reproduced 2026-09-25:
    a RawModule launched before the import fails after it; with tmu imported first, CuPy works throughout). The
    confirm stage uses the CUDA engine (CuPy) before and after applying frozen tmu probes, so the order is fixed here."""
    try:
        import tmu.models.classification.coalesced_classifier  # noqa: F401
    except ImportError as exc:
        return f"tmu not importable: {exc}"
    # pycuda.autoinit (pulled in by tmu) leaves its own non-primary context current, and the CUDA engine's RawModule
    # kernels then fail with CUDA_ERROR_INVALID_HANDLE (reproduced 2026-09-26: every confirmatory train step). tmu runs
    # on the CPU here, so the pycuda context is popped and destroyed and CuPy works in the primary context.
    try:
        import atexit
        import pycuda.autoinit as _ai
    except Exception:  # noqa: BLE001
        return "tmu imported before any CuPy work (no pycuda context)"
    _ai.context.pop()
    _ai.context.detach()
    atexit.unregister(_ai._finish_up)
    return "tmu imported before any CuPy work; the pycuda.autoinit context popped (CuPy uses the primary context)"


def main(argv: list[str] | None = None) -> int:
    ctx_note = cuda_context_first()
    ap = argparse.ArgumentParser(description="confirmatory R1 (docs/13 A7, A8, A9)")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--engine", default="cuda", choices=["cuda", "cpu", "oracle"])
    ap.add_argument("--train-run", default=None)
    ap.add_argument("--features-run", default=None)
    ap.add_argument("--common-run", default=None)
    ap.add_argument("--probe-run", default=None)
    ap.add_argument("--probe-full-run", default=None, help="confirm: the configuration's full-pool dev probe run (A3, A2, common)")
    ap.add_argument("--checkpoints", default=None)
    ap.add_argument("--probe-runs", default=None)
    ap.add_argument("--confirm-runs", default=None)
    ap.add_argument("--qpred-runs", default=None)
    ap.add_argument("--qcon-runs", default=None)
    ap.add_argument("--qint-runs", default=None)
    ap.add_argument("--census-runs", default=None)
    ap.add_argument("--skipped-file", default=None, help="report: the chain's list of skipped or failed steps")
    ap.add_argument("--reaccess-reason", default=None, help="only to repeat a failed counted access; written into the log")
    args = ap.parse_args(argv)
    cj = load_config(args.config)
    if not str(cj.get("status", "")).startswith("CONFIRMATORY"):
        raise SystemExit("this script runs the confirmatory config only")
    register(cj)
    rec = Rec(False)
    rec.m("cuda_context_order", ctx_note)
    rec.m("stage", args.stage)
    rec.m("config_params", cj["params"])
    check_config(cj, rec)
    tr = R.rp(args.train_run)
    if args.stage == "testhash":
        stage_testhash(args, cj, rec)
    elif args.stage == "train":
        stage_train(args, cj, rec)
    elif args.stage == "census":
        CA.stage_census(args, cj, rec, tr)
    elif args.stage == "features":
        stage_features(args, cj, rec, tr)
    elif args.stage == "fullfeat":
        stage_fullfeat(args, cj, rec)
    elif args.stage == "common":
        stage_common(args, cj, rec)
    elif args.stage == "randbank":
        stage_randbank(args, cj, rec)
    elif args.stage == "probe":
        stage_probe(args, cj, rec, R.rp(args.features_run))
    elif args.stage == "confirm":
        stage_confirm(args, cj, rec, R.rp(args.probe_run), R.rp(args.probe_full_run))
    elif args.stage == "qpred":
        if args.checkpoints is None:
            arm, _ = C.train_run_identity(tr)
            args.checkpoints = arm_spec(cj, arm)["qpred_checkpoints"]
        stage_qpred(args, cj, rec, tr, R.rp(args.features_run))
    elif args.stage == "qcon":
        stage_qcon(args, cj, rec, tr, R.rp(args.probe_run), R.rp(args.common_run))
    elif args.stage == "qint":
        stage_qint(args, cj, rec, tr, R.rp(args.features_run), R.rp(args.probe_run))
    else:
        stage_report(args, cj, rec)
    rec.m("wall_total_s", time.time() - T0)
    rec.m("failed_checks", rec.failed)
    rec.flush()
    rec.log(f"stage {args.stage} done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
