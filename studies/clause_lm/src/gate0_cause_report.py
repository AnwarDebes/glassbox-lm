"""Apply the rules of the Gate 0 cause plan (docs/plan_gate0_cause_2026-10-25.md) to the runs of src/gate0_cause.py.

Needs the plan's freeze record, docs/plan_gate0_cause_2026-10-25.frozen.json (scripts/freeze.py); without it no run
counts. Reads results/*_gate0_cause_<variant>_s8_seed<seed>/: config.json, metrics.json and the copies of
gate0_cause.py and runlog.py that src/runlog.py put into the folder. The pinned values below are literals, read from
the files on 2026-10-25; tests/test_gate0_cause.py checks them against the driver, the upstream copy and the two
environments, and tests/test_gate0_cause_report.py checks every refusal.

A folder counts only if each check passes (a failed check is printed as the reason):
  1. its UTC stamp is after the freeze time;
  2. config has the locked settings of M5 step 1 (folders without them, such as smoke runs, are listed and ignored);
  3. the copies of gate0_cause.py and runlog.py have the pinned SHA-256, and the freeze record pins the same values;
  4. config: emb_sha256, the work values of upstream_sha256 and cdllm_manifest_sha256 equal the pins; only the four
     training and validation arrays were read; the SHA-256 of variant_diff is the expected one of the variant; device
     is cuda for the torch arms and cpu for numba; the package listing is the pinned one of the arm's environment;
     variant and seed equal those of the folder name;
  5. metrics.json exists, and the variant's attribute was called (variant_calls: calls > 0, replaced for torch_noneg
     and numba_erode, not replaced for torch and numba).
A folder without metrics.json is unfinished: it may be rerun once with the same command; a second unfinished folder of
one variant and seed makes that key MISSING. The first folder of a key that passes every check counts; a later one is
refused. A folder refused by checks 3 to 5 is listed and does not block a later folder.

Signatures (literal statistics of the best-validation snapshot; val_acc is its validation accuracy):
  degenerate  val_acc <= 0.70 and clauses with a positive literal <= 400 (of 4,000) and median literals >= 100
  healthy     val_acc >= 0.78 and clauses with a positive literal >= 2,000 and median literals <= 10
Rules:
  driver check  A0 (torch, seed 42): MET if |val_acc - 0.8186666666666667| <= 0.01 and at least 2,000 clauses have a
                positive literal; NOT MET otherwise; MISSING without a counting folder
  arm A         torch_noneg, seeds 42, 123, 456: CONFIRMED if all three are degenerate, REFUTED if all three are
                healthy, MIXED otherwise, MISSING if a seed has no counting folder
  arm B         numba_erode, seed 42: CONFIRMED if healthy, REFUTED if degenerate, MIXED otherwise; NOT RUN if no folder
                with the locked settings exists, MISSING if one exists (also one from before the freeze) and none
                counts
Outcome, the first row that applies (printed as one line):
  1  driver check NOT MET or MISSING: every number is reported, no cause statement
  2  else arm A MIXED or MISSING, or arm B MIXED or MISSING: every number is reported, no cause statement
  3  else arm A or arm B REFUTED: the missing branch alone does not explain the failed Gate 0
  4  else arm A and arm B CONFIRMED: the missing branch accounts for the degenerate model in both kernels
  5  else (arm A CONFIRMED, arm B NOT RUN): removing the branch from torch reproduces the numba model; the numba side
     stays a reading of the code
Also printed, with no rule: whether A0 reproduces 0.8186666666666667 exactly, and the structure of every run (clauses
with a positive literal, median literals) beside its signature.

Usage: .venv-cdllm/bin/python src/gate0_cause_report.py [--no-log] [--results DIR] [--freeze PATH] [--json PATH]
Without --no-log it writes a run folder results/<UTC>_gate0_cause_report/ (src/runlog.py) with report.md.
"""
import argparse, datetime, glob, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEZE = os.path.join(LL, "docs", "plan_gate0_cause_2026-10-25.frozen.json")
A0_TARGET, A0_TOL = 0.8186666666666667, 0.01     # val_accuracy of 20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu
SEEDS_A = (42, 123, 456)
KEYS = ["Xtrain", "ytrain", "Xval", "yval"]
PIN = {"gate0_cause.py": "5b01c9e9f8ada227c60319575e98a6f0dab6440eedde24a93d6d087c26a6ebd3",
       "runlog.py": "d0b23bf35b801d8af2ca17c1a9623919506517053cdecd45c7aabd6e276a46be",
       "coalesced.py": "bfffb131874cbc0b6edff7e1152d25ae612df69a24c66cf66b26ecd46ca3ef2d",
       "torch_backend.py": "4828a9b515908793dbd5655d7eb3eed8b5ad306ab82cc2e96917db1669c69215",
       "cdllm_manifest": "5c648da62b8b760ac4a0edfd3adeb11e555993c3f73a14f78b65d560ee094dab",
       "emb": "15a987add2b590d8498768045d465167c808d8cbf8afd483330e89330444d0fc",
       "venv_torch": "d82154d3cfb0116431768da93895624bacf6c259a9cca0f3a97b62547cb377f6",    # .venv-cdllm-gpu
       "venv_numba": "5ef3616698497037843caaf26f2c49a1884ada42e7419f379dbe4076185c82cd"}    # .venv-cdllm
VARIANT_DIFF_SHA256 = {"torch": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                       "numba": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                       "torch_noneg": "18c6b91222e2cb4d1c4c174a3f6872407aedf38c607cad00597aafeb923d702c",
                       "numba_erode": "7ac04c1ceeaa1f7b5e44626a52cfc93de23f32c2b32063ce4d2bd40c154fb652"}
OUTCOMES = {1: "driver check {d}: every number is reported, no cause statement",
            2: "an arm is MIXED or MISSING (A {a}, B {b}): every number is reported, no cause statement",
            3: "the missing branch alone does not explain the failed Gate 0 (A {a}, B {b})",
            4: "the missing branch accounts for the degenerate model in both kernels (validation split; seeds 42, 123, "
               "456 for torch; seed 42 for numba)",
            5: "removing the branch from torch reproduces the numba model (validation split; seeds 42, 123, 456); the "
               "numba side stays a reading of the code"}
FOLDER = re.compile(r"^(\d{8}T\d{6}Z)_gate0_cause_(torch|torch_noneg|numba|numba_erode)_s8_seed(\d+)$")


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def signature(m):
    st, v = m["literal_stats"], m["val_acc"]
    if v <= 0.70 and st["clauses_with_positive_literal"] <= 400 and st["literals_median"] >= 100:
        return "degenerate"
    if v >= 0.78 and st["clauses_with_positive_literal"] >= 2000 and st["literals_median"] <= 10:
        return "healthy"
    return "neither"


def read_freeze(path):
    """(freeze time as a datetime, problem or None)."""
    if not os.path.exists(path):
        return None, f"no freeze record at {path}: no run counts"
    rec = json.load(open(path))
    t = datetime.datetime.strptime(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ")
    pinned = rec.get("pinned", {})
    for name in ("gate0_cause.py", "runlog.py"):
        if pinned.get(f"src/{name}") != PIN[name]:
            return None, f"the freeze record pins src/{name} as {pinned.get(f'src/{name}')}, the checker as {PIN[name]}"
    return t, None


def check_folder(d, variant, seed):
    """Reasons why a finished folder with the locked settings does not count ([] if it counts)."""
    why = []
    cfg = json.load(open(os.path.join(d, "config.json")))
    for name in ("gate0_cause.py", "runlog.py"):
        p = os.path.join(d, name)
        if not os.path.exists(p) or sha_file(p) != PIN[name]:
            why.append(f"copy of {name} is not the pinned file")
    up = cfg.get("upstream_sha256", {})
    for name in ("coalesced.py", "torch_backend.py"):
        if up.get(name, {}).get("work") != PIN[name]:
            why.append(f"upstream {name} is not the pinned file")
    if cfg.get("cdllm_manifest_sha256") != PIN["cdllm_manifest"]:
        why.append("cdllm manifest is not the pinned one")
    if cfg.get("emb_sha256") != PIN["emb"] or not cfg.get("emb_is_registered_file"):
        why.append("feature file is not the registered one")
    if cfg.get("emb_keys_read") != KEYS:
        why.append(f"feature-file keys read: {cfg.get('emb_keys_read')}")
    if hashlib.sha256(cfg.get("variant_diff", "").encode()).hexdigest() != VARIANT_DIFF_SHA256[variant]:
        why.append("variant diff is not the expected one")
    torch_arm = variant.startswith("torch")
    if cfg.get("device") != ("cuda" if torch_arm else "cpu"):
        why.append(f"device {cfg.get('device')}")
    if cfg.get("venv", {}).get("listing_sha256") != PIN["venv_torch" if torch_arm else "venv_numba"]:
        why.append("package listing is not the pinned one of the environment")
    if cfg.get("variant") != variant or cfg.get("seed") != seed:
        why.append(f"config variant {cfg.get('variant')} seed {cfg.get('seed')} differ from the folder name")
    met = json.load(open(os.path.join(d, "metrics.json")))
    calls = met.get("variant_calls") or {}
    if not calls.get("calls", 0) > 0 or calls.get("replaced") != (variant in ("torch_noneg", "numba_erode")):
        why.append(f"variant attribute calls: {calls}")
    return why, met


def collect(results, freeze_t):
    """(counting runs {(variant, seed): metrics}, problems, keys with a locked folder, keys refused)."""
    runs, problems, seen_locked, refused = {}, [], set(), set()
    unfinished = {}
    for d in sorted(glob.glob(os.path.join(results, "*_gate0_cause_*"))):
        name = os.path.basename(d)
        mt = FOLDER.match(name)
        if not mt:
            continue
        stamp = datetime.datetime.strptime(mt.group(1), "%Y%m%dT%H%M%SZ")
        key = (mt.group(2), int(mt.group(3)))
        cfg_p = os.path.join(d, "config.json")
        cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
        if not cfg.get("locked_settings_of_m5_step1"):
            problems.append(f"ignored (not the locked settings; a smoke run or other settings): {name}")
            continue
        seen_locked.add(key)                     # a locked folder exists: arm B is then MISSING, not NOT RUN
        if freeze_t is None or stamp <= freeze_t:
            problems.append(f"not counted (before the freeze or no freeze record): {name}")
            continue
        if key in runs:
            problems.append(f"REFUSED: a second counting run for {key}: {name}")
            continue
        if key in refused:
            problems.append(f"not counted ({key} is MISSING after two unfinished runs): {name}")
            continue
        if not os.path.exists(os.path.join(d, "metrics.json")):
            unfinished[key] = unfinished.get(key, 0) + 1
            if unfinished[key] >= 2:
                refused.add(key)
                problems.append(f"REFUSED: a second unfinished run for {key}, the key is MISSING: {name}")
            else:
                problems.append(f"unfinished (one rerun with the same command is allowed): {name}")
            continue
        why, met = check_folder(d, *key)
        if why:
            problems.append(f"not counted: {name}: " + "; ".join(why))
            continue
        runs[key] = {**met, "folder": name}
    return runs, problems, seen_locked, refused


def verdicts(runs, seen_locked):
    out = {}
    a0 = runs.get(("torch", 42))
    if a0 is None:
        out["driver_check"] = "MISSING"
    else:
        ok = abs(a0["val_acc"] - A0_TARGET) <= A0_TOL and a0["literal_stats"]["clauses_with_positive_literal"] >= 2000
        out["driver_check"] = "MET" if ok else "NOT MET"
    sig_a = [signature(runs[("torch_noneg", s)]) if ("torch_noneg", s) in runs else None for s in SEEDS_A]
    if None in sig_a:
        out["arm_A"] = "MISSING"
    elif all(s == "degenerate" for s in sig_a):
        out["arm_A"] = "CONFIRMED"
    elif all(s == "healthy" for s in sig_a):
        out["arm_A"] = "REFUTED"
    else:
        out["arm_A"] = "MIXED"
    b = runs.get(("numba_erode", 42))
    if b is None:
        out["arm_B"] = "MISSING" if ("numba_erode", 42) in seen_locked else "NOT RUN"
    else:
        out["arm_B"] = {"healthy": "CONFIRMED", "degenerate": "REFUTED"}.get(signature(b), "MIXED")
    return out


def outcome(v):
    """(row, text) of the ordered outcome table."""
    d, a, b = v["driver_check"], v["arm_A"], v["arm_B"]
    if d in ("NOT MET", "MISSING"):
        row = 1
    elif a in ("MIXED", "MISSING") or b in ("MIXED", "MISSING"):
        row = 2
    elif a == "REFUTED" or b == "REFUTED":
        row = 3
    elif a == "CONFIRMED" and b == "CONFIRMED":
        row = 4
    else:
        row = 5                                  # a == CONFIRMED and b == NOT RUN, the only case left
    return row, OUTCOMES[row].format(d=d, a=a, b=b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(LL, "results"))
    ap.add_argument("--freeze", default=FREEZE)
    ap.add_argument("--json")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    freeze_t, fproblem = read_freeze(a.freeze)
    runs, problems, seen_locked, refused = collect(a.results, freeze_t)
    if fproblem:
        problems.insert(0, fproblem)
    lines = []
    for (variant, seed), m in sorted(runs.items()):
        st = m["literal_stats"]
        lines.append(f"{variant:12s} seed {seed:3d}: val_acc {m['val_acc']:.4f}, clauses with a positive literal "
                     f"{st['clauses_with_positive_literal']} of {st['n_clauses']}, median literals "
                     f"{st['literals_median']:g}, best epoch {m.get('best_epoch')}, signature {signature(m)}  "
                     f"[{m['folder']}]")
    lines += [f"PROBLEM {p}" for p in problems]
    v = verdicts(runs, seen_locked)
    row, text = outcome(v)
    a0 = runs.get(("torch", 42))
    lines.append("A0 reproduces 0.8186666666666667 exactly (a reading, no rule): " +
                 ("no counting A0 run" if a0 is None else
                  f"{'yes' if a0['val_acc'] == A0_TARGET else 'no'} (val_acc {a0['val_acc']!r})"))
    lines.append(json.dumps(v))
    lines.append(f"OUTCOME {row}: {text}")
    report = "\n".join(lines)
    print(report)
    out = {"verdicts": v, "outcome_row": row, "outcome": text, "problems": problems,
           "freeze_utc": freeze_t.strftime("%Y-%m-%dT%H:%M:%SZ") if freeze_t else None,
           "runs": {f"{k[0]}_seed{k[1]}": r for k, r in runs.items()}}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if not a.no_log:
        from runlog import Run
        run = Run("gate0_cause_report", {"results": a.results, "freeze": a.freeze, "rules": __doc__}, script=__file__)
        open(os.path.join(run.dir, "report.md"), "w").write(report + "\n")
        run.finish(out)
    return v, row


if __name__ == "__main__":
    main()
