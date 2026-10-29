"""The E4.3 dev go-bar (registration revision 2, review issue 4 and decision 5), fixed before any Qwen row exists.

Input: the out-of-fold predictions that train_guard.py --oof-rows writes for A1's executed train-task writes (the
train_guard.py folds over train tasks, seed 20261021, 5 folds; the TM at the CV pick with seed 1; LR and tree at their
picks), and A1's own results (judge-free DB reward per train task; a re-run replaces its tasks, e4guard/runs.py).
  G = the number of A1-FAILED train tasks (DB reward 0) where the learned part denies at least 1 executed NON-GOLD
      write (label DENY);
  L = the number of A1-PASSED train tasks (DB reward 1) where it denies at least 1 GOLD write (label ALLOW).
  GO iff G >= 8 (of the 74 train tasks) AND G - L >= 4, both for the TM (seed 1). Otherwise NO-GO: E4.3 and E4.5
  become a dev-only line and A2, A3 and A3t are NOT run on the test tasks.
Reported beside it, no rule: G and L for LR and the tree; the share of the learned part's denials that persist after a
simulated re-confirmation (bits.reconfirm), per model.
Revision 3 (the decision 1, 2026-10-21T07:24:12Z): E4.3 is LOW POWER. gobar.json carries "power": the
exact power of E4.3's registered rule at THIS go-bar's observed rates (g = G / n_train, l = L / n_train; trinomial;
need(a) from the pinned results/power_e43_rev3.json; scripts/power_e43.py), per model, beside the fixed figures (the
go-bar corner G 8, L 4; the effect needed for 80% power). Every E4.3 sentence carries it (registration section 15).
"outcome" is "go" or "no-go" (STAGE=train_guard writes "not testable" or "void" itself when it stops earlier).
Exit 0 = GO, 10 = NO-GO, 2 = the inputs do not fit (no decision).
Run: offline_env.sh .../taubench/.venv/bin/python -I -B gobar.py --oof runs/<c>/models/oof_rows.json \
       --a1 runs/<c>/A1_base --split-file <split_tasks.json> --out runs/<c>/gobar.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

from e4guard.runs import db_reward, load_arm  # noqa: E402

G_MIN, NET_MIN = 8, 4
MODELS = ("tm_seed1", "lr", "tree")
POWER_FILE = E4 / "results/power_e43_rev3.json"


def power_block(models: dict, n_train: int, power_file: Path = POWER_FILE) -> dict:
    """E4.3's exact power at this go-bar's observed rates, per model (pure apart from reading the pinned file)."""
    from power_e43 import power

    b = Path(power_file).read_bytes()
    d = json.loads(b)
    need = d["need_favourable_by_against"]
    out = {"what": "E4.3 (low power): exact power of the registered rule at this go-bar's observed rates; trinomial, "
                   "g = G / n_train, l = L / n_train (upper bounds: a denial need not flip a task)",
           "power_file": str(power_file), "power_file_sha256": hashlib.sha256(b).hexdigest(), "n_train_tasks": n_train,
           "at_go_bar_corner_G8_L4": d["at_go_bar_corner_G8_L4"], "at_go_bar_G8_L0": d["at_go_bar_G8_L0"],
           "effect_for_80pct_power": d["effect_for_80pct_power"], "at_observed_rates": {}}
    for m, r in models.items():
        out["at_observed_rates"][m] = power(r["G"] / n_train, r["L"] / n_train, need) if n_train else None
    return out


def go_bar(oof_rows: list[dict], a1_reward: dict, train_tasks: list[str]) -> dict:
    """Pure computation (unit-tested). a1_reward: {task: 0.0/1.0} for the train tasks."""
    out = {"n_train_tasks": len(train_tasks), "n_oof_rows": len(oof_rows), "rule": f"GO iff G >= {G_MIN} and "
           f"G - L >= {NET_MIN} (TM seed 1)", "models": {}}
    tt = set(map(str, train_tasks))
    for m in MODELS:
        G, L = set(), set()
        n_deny = n_persist = 0
        for r in oof_rows:
            t = str(r["task"])
            if t not in tt or f"deny_{m}" not in r:
                continue
            d = int(r[f"deny_{m}"])
            if d:
                n_deny += 1
                n_persist += int(r[f"deny_{m}_after_reconfirm"])
            if d and r["label"] == 1 and a1_reward.get(t) == 0.0:
                G.add(t)
            if d and r["label"] == 0 and a1_reward.get(t) == 1.0:
                L.add(t)
        out["models"][m] = {"G": len(G), "L": len(L), "G_minus_L": len(G) - len(L),
                            "G_tasks": sorted(G, key=int), "L_tasks": sorted(L, key=int),
                            "learned_denials": n_deny,
                            "persist_after_reconfirm": n_persist,
                            "persist_share": (n_persist / n_deny) if n_deny else None}
    tm = out["models"]["tm_seed1"]
    out["GO"] = bool(tm["G"] >= G_MIN and tm["G_minus_L"] >= NET_MIN)
    out["outcome"] = "go" if out["GO"] else "no-go"
    out["decision"] = "GO: run A2 (seeds 1-3), A3, A3t on the 40 test tasks" if out["GO"] else \
        "NO-GO: E4.3 and E4.5 are a dev-only line; A2, A3, A3t are not run on the test tasks"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--a1", required=True, help="A1's run folder (A1_base); its _rerun folder is merged")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--train-tasks", nargs="*", default=None, help="DRYRUN_ campaigns only (run_e4_gpu.sh test hook)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = json.loads(Path(a.split_file).read_text())
    train = [str(t) for t in sp["train"]]
    if a.train_tasks:
        if "DRYRUN_" not in a.a1:
            sys.stderr.write("REFUSING: --train-tasks is a dry-run hook\n")
            sys.exit(2)
        train = [t for t in train if t in set(a.train_tasks)]
    oof = json.loads(Path(a.oof).read_text())
    arm = load_arm(Path(a.a1))
    if arm is None:
        sys.stderr.write(f"no results.json in {a.a1}\n")
        sys.exit(2)
    missing = [t for t in train if t not in arm["sims"]]
    if missing:
        sys.stderr.write(f"A1 lacks {len(missing)} train tasks ({missing[:10]}...); no go-bar decision\n")
        sys.exit(2)
    rew = {t: db_reward(arm["sims"][t]) for t in train}
    res = go_bar(oof["rows"], rew, train)
    res["power"] = power_block(res["models"], len(train))           # the pinned file only (H2)
    res["inputs"] = {"oof": a.oof, "oof_sha256": hashlib.sha256(Path(a.oof).read_bytes()).hexdigest(),
                     "oof_meta": oof.get("meta"), "a1": a.a1, "a1_rerun_tasks": arm["rerun_tasks"]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "inputs"}, indent=1))
    sys.exit(0 if res["GO"] else 10)


if __name__ == "__main__":
    main()
