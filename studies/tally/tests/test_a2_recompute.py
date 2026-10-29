"""scripts/a2_recompute.py on synthetic run folders (no data file is read).

Run: python tests/test_a2_recompute.py

Synthetic run folders hold a logp.npz (21 arms, 12 blocks; the code 0.1 nats per token better than
`none`, the random arms equal to `none` up to tiny noise; KN5 alone at 7.1760 bits per word) and a
metrics.json computed from the same arrays with the formulas of scripts/a2_run.py. Every dev-test
case runs the script's main() in this process, with its data loader replaced by a sentinel that
raises, and with the study folder pointed at a temporary one; no case starts the script as a program
on a dev-test folder. Checks:
1. a valid run (program, default mode): every check matches, exit 0; an altered success reading or
   perplexity is a mismatch, exit 1;
2. the dev-test rescore is refused unless the folder is the finished registered look with the
   registration's freeze hash; a refused rescore writes no output and reaches no data;
3. the after-the-look check of the freeze record finds no difference on an unchanged layout and
   names a changed and an added src file;
4. a synthetic dev-test look: with the code exchange, the registered label, KN5 alone 7.1760 and an
   unchanged freeze record every check matches; KN5 alone 7.2, another code or another label is a
   mismatch.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
SCRIPT = os.path.join(STUDY, "scripts", "a2_recompute.py")
sys.path.insert(0, os.path.join(STUDY, "src"))
from tally.floor import stats  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def synthetic_run(d, split="valid", kn5_bits=7.1760, code="exchange", label=None):
    os.makedirs(d)
    rng = np.random.default_rng(3)
    n, nb = 6000, 12
    blocks = np.repeat(np.arange(nb), n // nb)
    none = np.log(rng.uniform(0.001, 0.3, n))
    arms = {"none": none, "exchange": none + 0.1, "kind_topic": none + 0.1, "kmeans200": none + 0.01,
            "predefined": none + 0.009, "topic": none + 0.001, "orgroups": none.copy()}
    for l in range(4):
        arms[f"exchange_level{l}"] = none + 0.02 * (l + 1)
    for s in range(1, 11):
        arms[f"random_s{s}"] = none + rng.normal(0, 1e-5, n)
    kn5 = np.full(n, -kn5_bits * np.log(2))
    unseen, dup = rng.random(n) < 0.2, rng.random(n) < 0.06
    seen = rng.random(n) < 0.99
    np.savez_compressed(os.path.join(d, "logp.npz"), blocks=blocks, unseen_bigram=unseen, duplicate=dup,
                        train_seen_word=seen, kn5_alone=kn5, **{f"arm__{k}": v for k, v in arms.items()})
    R = {k: {"all": stats.report(v, split)} for k, v in arms.items()}
    base = R["none"]["all"]["ppl"]
    rnd = [f"random_s{s}" for s in range(1, 11)]
    r_ppl = np.array([R[r]["all"]["ppl"] for r in rnd])
    lp_r = np.mean([arms[r] for r in rnd], 0)
    gain = 1 - R[code]["all"]["ppl"] / base
    marg = 1 - R[code]["all"]["ppl"] / r_ppl.mean()
    bb = stats.block_bootstrap(stats.bits_diff(arms[code], lp_r), blocks)
    metrics = {"arms": R, "cache": {"chosen": {"half_life": 100.0, "reset": "block", "skip": [43]}},
               "kn5_alone": stats.report(kn5, split), "logp_npz_sha256": sha(os.path.join(d, "logp.npz")),
               "reading": {"label": label or f"development reading on {split}; not a verdict", "best_code": code,
                           "gain_vs_none": gain, "margin_vs_random_mean": marg, "best_minus_random_mean_bits": bb,
                           "success_line": bool(gain >= 0.05 and marg >= 0.01 and bb["hi"] < 0),
                           "kill_line": bool(gain < 0.02)}}
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"split": split}, f)
    return metrics


def program(run_dir, out, *extra):
    return subprocess.run([sys.executable, SCRIPT, "--run", run_dir, "--out", out, *extra],
                          capture_output=True, text=True, timeout=300)


class Reached(Exception):
    pass


def load_module():
    spec = importlib.util.spec_from_file_location("a2_recompute_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def no_load(*a, **k):
        raise Reached("ptb.load")
    mod.ptb.load = no_load
    return mod


def in_process(mod, *args):
    """main() of the module with these arguments; its printed summary is kept out of the test output."""
    sys.argv = ["a2_recompute.py", *args]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
        return "returned"
    except Reached as r:
        return f"reached {r}"
    except SystemExit as e:
        return f"exit {e.code}"


def fake_frozen_study(root, metrics_path):
    """A study folder with registrations/A2.md, one src file and a freeze record of them."""
    os.makedirs(os.path.join(root, "src", "tally"))
    os.makedirs(os.path.join(root, "registrations"))
    open(os.path.join(root, "src", "tally", "x.py"), "w").write("x = 1\n")
    open(os.path.join(root, "registrations", "A2.md"), "w").write("reg\n")
    rec = {"registration": {"registrations/A2.md": sha(os.path.join(root, "registrations", "A2.md"))},
           "src": {"src/tally/x.py": sha(os.path.join(root, "src", "tally", "x.py"))},
           "extra": {os.path.relpath(metrics_path, root): sha(metrics_path)}}
    json.dump(rec, open(os.path.join(root, "registrations", "A2.frozen.json"), "w"))
    return rec


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. a valid run, as a program
        d = os.path.join(tmp, "20261025T000000Z_a2_valid_dev")
        metrics = synthetic_run(d)
        r = program(d, os.path.join(tmp, "out1.json"))
        res = json.load(open(os.path.join(tmp, "out1.json"))) if os.path.exists(os.path.join(tmp, "out1.json")) else {}
        check("valid run: every check matches and exit 0",
              r.returncode == 0 and res.get("comparison", {}).get("all_match") is True,
              f"exit {r.returncode} {r.stdout[-300:]} {r.stderr[-300:]}")
        check("valid run: the reading is success (gain about 9.5%)",
              res.get("recompute", {}).get("success") is True and metrics["reading"]["success_line"] is True)
        m2 = dict(metrics, reading=dict(metrics["reading"], success_line=False))
        json.dump(m2, open(os.path.join(d, "metrics.json"), "w"))
        r = program(d, os.path.join(tmp, "out2.json"))
        res = json.load(open(os.path.join(tmp, "out2.json")))
        check("an altered success reading is a mismatch (exit 1)",
              r.returncode == 1 and res["comparison"]["checks"]["same_success"] is False, f"exit {r.returncode}")
        m3 = json.loads(json.dumps(metrics))
        m3["arms"]["kmeans200"]["all"]["ppl"] *= 1.001
        json.dump(m3, open(os.path.join(d, "metrics.json"), "w"))
        r = program(d, os.path.join(tmp, "out3.json"))
        res = json.load(open(os.path.join(tmp, "out3.json")))
        check("an altered perplexity is a mismatch (exit 1)",
              r.returncode == 1 and res["comparison"]["checks"]["ppl_within_1e-12"] is False, f"exit {r.returncode}")
        json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"))

        # 2. the dev-test rescore guard, in this process, with the data-loader sentinel
        mod = load_module()
        fake = os.path.join(tmp, "fake_study")
        os.makedirs(fake)
        mod.STUDY = fake
        dv = os.path.join(tmp, "20261025T000001Z_a2_devtest_smoke")
        synthetic_run(dv, split="devtest", label="registered reading of A2 on devtest")
        e = mod.rescore_allowed(dv, "devtest", study=fake)
        check("dev-test rescore refused for a folder that is not the registered look",
              any("registered look" in x for x in e), f"{e}")
        dl = os.path.join(tmp, "20261025T000002Z_a2_devtest_final")
        synthetic_run(dl, split="devtest", label="registered reading of A2 on devtest")
        json.dump({"state": "finished"}, open(os.path.join(dl, "run.json"), "w"))
        json.dump({"registrations/A2.md": "0" * 64}, open(os.path.join(dl, "source_hashes.json"), "w"))
        e = mod.rescore_allowed(dl, "devtest", study=fake)
        check("dev-test rescore refused without a freeze record of A2", any("A2.frozen.json missing" in x for x in e),
              f"{e}")
        got = in_process(mod, "--run", dv, "--out", os.path.join(tmp, "o.json"), "--rescore")
        check("the program refuses a dev-test rescore of a folder that is not the look, before any data",
              got.startswith("exit refused") and not os.path.exists(os.path.join(tmp, "o.json")), got)
        check("a valid run may be rescored", mod.rescore_allowed(d, "valid", study=fake) == [])

        # 3. the after-the-look check of the freeze record
        rec = fake_frozen_study(fake, os.path.join(dl, "metrics.json"))
        check("after the look: an unchanged frozen layout matches", mod.record_matches(fake) == [],
              f"{mod.record_matches(fake)}")
        open(os.path.join(fake, "src", "tally", "x.py"), "w").write("x = 2\n")
        open(os.path.join(fake, "src", "tally", "y.py"), "w").write("y = 1\n")
        check("after the look: a changed and an added src file are named",
              mod.record_matches(fake) == ["src/tally/x.py", "src/tally/y.py"], f"{mod.record_matches(fake)}")
        open(os.path.join(fake, "src", "tally", "x.py"), "w").write("x = 1\n")
        os.remove(os.path.join(fake, "src", "tally", "y.py"))
        json.dump({"registrations/A2.md": rec["registration"]["registrations/A2.md"]},
                  open(os.path.join(dl, "source_hashes.json"), "w"))
        e = mod.rescore_allowed(dl, "devtest", study=fake)
        check("dev-test rescore allowed for the finished registered look", e == [], f"{e}")

        # 4. a synthetic dev-test look, default mode, in this process
        got = in_process(mod, "--run", dl, "--out", os.path.join(tmp, "look.json"))
        res = json.load(open(os.path.join(tmp, "look.json"))) if os.path.exists(os.path.join(tmp, "look.json")) else {}
        ck = res.get("comparison", {}).get("checks", {})
        check("dev-test look: code exchange, registered label, KN5 7.1760, freeze record unchanged: all match",
              got == "exit 0" and res.get("comparison", {}).get("all_match") is True
              and ck.get("code_is_exchange") and ck.get("registered_label") and ck.get("kn5_alone_equals_T0")
              and ck.get("freeze_record_still_matches"), f"{got} {ck}")
        for name, kw, key in (("KN5 alone 7.2 is a mismatch", dict(kn5_bits=7.2), "kn5_alone_equals_T0"),
                              ("another code is a mismatch", dict(code="kmeans200"), "code_is_exchange"),
                              ("another label is a mismatch", dict(label="development reading on devtest"),
                               "registered_label")):
            dk = os.path.join(tmp, f"bad_{key}", "20261025T000003Z_a2_devtest_final")
            kw.setdefault("label", "registered reading of A2 on devtest")
            synthetic_run(dk, split="devtest", **kw)
            out = os.path.join(tmp, f"bad_{key}.json")
            got = in_process(mod, "--run", dk, "--out", out)
            ck = json.load(open(out))["comparison"]["checks"] if os.path.exists(out) else {}
            check(f"dev-test look: {name} (exit 1)", got == "exit 1" and ck.get(key) is False, f"{got} {ck}")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
