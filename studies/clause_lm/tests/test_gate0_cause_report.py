"""src/gate0_cause_report.py on made-up run folders: signatures, every refusal, the rerun rule and every outcome row.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_gate0_cause_report.py [REPORT]

Run from studies/clause_lm. CPU only, about a second. REPORT defaults to src/gate0_cause_report.py; give another copy
(for example the version before the review of 2026-10-25) to test it the same way. Every run folder and the freeze record are made up in a
temporary folder (a valid folder holds byte copies of src/gate0_cause.py and src/runlog.py and a config with the pinned
values); no run of src/gate0_cause.py is read, and the checker's own run folder goes to a temporary folder. The pinned
diff hashes of the two modified variants are replaced by those of made-up diff texts here; tests/test_gate0_cause.py
checks the real ones.

Asserts:
- the signatures on and next to their bars (0.70, 400, 100; 0.78, 2,000, 10);
- each outcome row: 1 with the driver check NOT MET and with it MISSING; 2 with arm A MIXED, arm A MISSING (a seed
  absent), arm B MIXED and arm B MISSING (one unfinished folder and no rerun); 3 with arm A REFUTED and with arm B
  REFUTED; 4 with both arms CONFIRMED; 5 with arm A CONFIRMED and arm B NOT RUN (only a smoke folder of B);
- a folder is not counted, with its reason printed, when: its stamp is not after the freeze; there is no freeze
  record; the freeze record pins another driver; its copy of gate0_cause.py or runlog.py differs; an upstream kernel
  file, the cdllm manifest or the feature file differs; the feature file is not the registered one; a test array was
  read; the variant diff differs; a torch arm ran on cpu; the package listing differs; the variant or the seed differs
  from the folder name; the variant's attribute was not called, or the replaced flag is wrong;
- the rerun rule: one unfinished folder and a finished rerun counts; two unfinished folders make the key MISSING even
  when a finished third follows; a second counting folder is refused and the first kept; folders without the locked
  settings are ignored;
- the A0 reading says "yes" only for exactly 0.8186666666666667; the default run writes report.md into a run folder,
  --no-log writes none; the checker's pins equal the SHA-256 of src/gate0_cause.py and src/runlog.py.
"""

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
KEYS = ["Xtrain", "ytrain", "Xval", "yval"]
FREEZE_UTC = "2026-10-25T13:00:00Z"
FAKE_DIFF = {"torch": "", "numba": "", "torch_noneg": "made-up diff of torch_noneg\n",
             "numba_erode": "made-up diff of numba_erode\n"}


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def stats(pos, median):
    return {"n_clauses": 4000, "clauses_with_positive_literal": pos, "literals_median": median}


DEG = (0.55, stats(0, 1021.0))
HEALTHY = (0.82, stats(2954, 1.0))


class World:
    """A temporary results folder and freeze record, and a maker of run folders."""

    def __init__(self, rep, root, name):
        self.rep, self.res = rep, root / name / "results"
        self.res.mkdir(parents=True)
        self.freeze = root / name / "plan.frozen.json"
        self.freeze.write_text(json.dumps({"frozen_utc": FREEZE_UTC, "pinned": {
            "src/gate0_cause.py": rep.PIN["gate0_cause.py"], "src/runlog.py": rep.PIN["runlog.py"]}}))
        self.n = 0

    def add(self, variant, seed, val, st, stamp=None, finished=True, cfg=None, met=None, copy=None, locked=True):
        self.n += 1
        stamp = stamp or f"20261025T14{self.n // 60:02d}{self.n % 60:02d}Z"
        d = self.res / f"{stamp}_gate0_cause_{variant}_s8_seed{seed}"
        d.mkdir()
        torch_arm = variant.startswith("torch")
        c = {"variant": variant, "seed": seed, "locked_settings_of_m5_step1": locked, "emb_is_registered_file": True,
             "emb_sha256": self.rep.PIN["emb"], "emb_keys_read": KEYS,
             "upstream_sha256": {"coalesced.py": {"work": self.rep.PIN["coalesced.py"]},
                                 "torch_backend.py": {"work": self.rep.PIN["torch_backend.py"]}},
             "cdllm_manifest_sha256": self.rep.PIN["cdllm_manifest"], "variant_diff": FAKE_DIFF[variant],
             "device": "cuda" if torch_arm else "cpu",
             "venv": {"listing_sha256": self.rep.PIN["venv_torch" if torch_arm else "venv_numba"]}}
        c.update(cfg or {})
        (d / "config.json").write_text(json.dumps(c))
        for f in ("gate0_cause.py", "runlog.py"):
            shutil.copy(toyutil.SRC / f, d / f)
        for f, text in (copy or {}).items():
            (d / f).write_text(text)
        if finished:
            m = {"variant": variant, "val_acc": val, "literal_stats": st, "best_epoch": 20,
                 "variant_calls": {"attribute": "x", "replaced": variant in ("torch_noneg", "numba_erode"), "calls": 7}}
            m.update(met or {})
            (d / "metrics.json").write_text(json.dumps(m))
        return d

    def full(self, a0=(0.8186666666666667, stats(2954, 1.0)), a=(DEG, DEG, DEG), b=HEALTHY):
        if a0:
            self.add("torch", 42, *a0)
        for seed, sig in zip((42, 123, 456), a):
            if sig:
                self.add("torch_noneg", seed, *sig)
        if b:
            self.add("numba_erode", 42, *b)
        return self

    def report(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            v, row = self.rep.main(["--results", str(self.res), "--freeze", str(self.freeze), "--no-log", *extra])
        return v, row, buf.getvalue()


def main():
    with tempfile.TemporaryDirectory(prefix="clm_g0rep_") as tmp:
        root = Path(tmp)
        toyutil.use_src(root)                       # runlog.ROOT -> the temporary folder
        target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else toyutil.SRC / "gate0_cause_report.py"
        print(f"checker under test: {target}")
        rep, _ = toyutil.load_module_copy(target, "gate0_cause_report_under_test")
        rep.VARIANT_DIFF_SHA256 = {v: sha_text(t) for v, t in FAKE_DIFF.items()}
        n = [0]

        def world(name=None):
            n[0] += 1
            return World(rep, root, name or f"w{n[0]}")

        s = rep.signature
        check(s({"val_acc": 0.70, "literal_stats": stats(400, 100.0)}) == "degenerate", "degenerate at the bars")
        check(s({"val_acc": 0.78, "literal_stats": stats(2000, 10.0)}) == "healthy", "healthy at the bars")
        check(s({"val_acc": 0.7001, "literal_stats": stats(0, 1021.0)}) == "neither", "just above 0.70: neither")
        check(s({"val_acc": 0.82, "literal_stats": stats(1999, 1.0)}) == "neither", "1,999 positive clauses: neither")
        check(s({"val_acc": 0.82, "literal_stats": stats(2954, 10.5)}) == "neither", "median 10.5: neither")

        # outcome rows
        cases = [
            ("row 1, driver check NOT MET", world().full(a0=(0.7987, stats(2954, 1.0))), 1, "NOT MET"),
            ("row 1, driver check MISSING", world().full(a0=None), 1, "MISSING"),
            ("row 2, arm A MIXED", world().full(a=(DEG, HEALTHY, DEG)), 2, "MIXED"),
            ("row 2, arm A MISSING", world().full(a=(DEG, None, DEG)), 2, "MISSING"),
            ("row 2, arm B MIXED", world().full(b=(0.75, stats(2500, 2.0))), 2, "MIXED"),
            ("row 3, arm A REFUTED", world().full(a=(HEALTHY,) * 3), 3, "REFUTED"),
            ("row 3, arm B REFUTED", world().full(b=DEG), 3, "REFUTED"),
            ("row 4, both arms CONFIRMED", world().full(), 4, "both kernels"),
        ]
        w = world().full(b=None)
        w.add("numba_erode", 42, *HEALTHY, cfg={"locked_settings_of_m5_step1": False}, locked=False)
        cases.append(("row 5, arm A CONFIRMED, arm B NOT RUN (only a smoke folder)", w, 5, "reading of the code"))
        w = world().full(b=None)
        w.add("numba_erode", 42, *HEALTHY, finished=False)
        cases.append(("row 2, arm B MISSING (unfinished, no rerun)", w, 2, "MISSING"))
        for what, w, row, word in cases:
            v, got, out = w.report()
            line = [l for l in out.splitlines() if l.startswith("OUTCOME")]
            check(got == row and len(line) == 1 and line[0].startswith(f"OUTCOME {row}:") and word in line[0],
                  f"{what}: {line[0] if line else out[-200:]}")

        # refusals: the A0 folder is the one changed; the rest stay valid, so the driver check turns MISSING
        refusals = [
            ("stamp not after the freeze", dict(stamp="20261025T130000Z"), "before the freeze"),
            ("copy of gate0_cause.py differs", dict(copy={"gate0_cause.py": "edited\n"}), "copy of gate0_cause.py"),
            ("copy of runlog.py differs", dict(copy={"runlog.py": "edited\n"}), "copy of runlog.py"),
            ("upstream coalesced.py differs", dict(cfg={"upstream_sha256": {
                "coalesced.py": {"work": "0" * 64}, "torch_backend.py": {"work": rep.PIN["torch_backend.py"]}}}),
             "upstream coalesced.py"),
            ("upstream torch_backend.py differs", dict(cfg={"upstream_sha256": {
                "coalesced.py": {"work": rep.PIN["coalesced.py"]}, "torch_backend.py": {"work": "0" * 64}}}),
             "upstream torch_backend.py"),
            ("cdllm manifest differs", dict(cfg={"cdllm_manifest_sha256": "0" * 64}), "cdllm manifest"),
            ("feature file differs", dict(cfg={"emb_sha256": "0" * 64}), "feature file"),
            ("feature file not registered", dict(cfg={"emb_is_registered_file": False}), "feature file"),
            ("a test array was read", dict(cfg={"emb_keys_read": KEYS + ["Xtest"]}), "keys read"),
            ("variant diff differs", dict(cfg={"variant_diff": "another diff\n"}), "variant diff"),
            ("torch arm on cpu", dict(cfg={"device": "cpu"}), "device cpu"),
            ("package listing differs", dict(cfg={"venv": {"listing_sha256": "0" * 64}}), "package listing"),
            ("variant differs from the folder", dict(cfg={"variant": "torch_noneg"}), "differ from the folder"),
            ("seed differs from the folder", dict(cfg={"seed": 43}), "differ from the folder"),
            ("attribute not called", dict(met={"variant_calls": {"replaced": False, "calls": 0}}), "calls"),
            ("replaced flag wrong", dict(met={"variant_calls": {"replaced": True, "calls": 3}}), "calls"),
        ]
        for what, kw, word in refusals:
            w = world().full(a0=None)
            w.add("torch", 42, 0.8186666666666667, stats(2954, 1.0), **kw)
            v, row, out = w.report()
            probs = [l for l in out.splitlines() if l.startswith("PROBLEM") and "_gate0_cause_torch_s8_seed42" in l]
            check(v["driver_check"] == "MISSING" and row == 1 and len(probs) == 1 and word in probs[0],
                  f"refused: {what} ({probs[0][8:120] if probs else 'no problem line'})")
        w = world().full()
        w.freeze.unlink()
        v, row, out = w.report()
        check(v == {"driver_check": "MISSING", "arm_A": "MISSING", "arm_B": "MISSING"} and row == 1
              and "no freeze record" in out, f"no freeze record: no run counts ({v})")
        w = world().full()
        w.freeze.write_text(json.dumps({"frozen_utc": FREEZE_UTC, "pinned": {"src/gate0_cause.py": "1" * 64,
                                                                              "src/runlog.py": rep.PIN["runlog.py"]}}))
        v, row, out = w.report()
        check(v["driver_check"] == "MISSING" and v["arm_A"] == "MISSING" and "freeze record pins src/gate0_cause.py" in out,
              "a freeze record that pins another driver: no run counts")
        w = world().full(a0=None)
        w.add("torch", 42, 0.8186666666666667, stats(2954, 1.0))
        w.add("torch", 42, 0.80, stats(2954, 1.0))
        v, row, out = w.report()
        check(v["driver_check"] == "MET" and "REFUSED: a second counting run" in out and "0.8187" in out,
              "a second counting run is refused and the first kept")

        # rerun rule
        w = world().full(a=(DEG, DEG, None))
        w.add("torch_noneg", 456, *DEG, finished=False)
        w.add("torch_noneg", 456, *DEG)
        v, row, out = w.report()
        check(v["arm_A"] == "CONFIRMED" and "unfinished (one rerun" in out, "one unfinished folder and a rerun: counts")
        w = world().full(a=(DEG, DEG, None))
        for fin in (False, False, True):
            w.add("torch_noneg", 456, *DEG, finished=fin)
        v, row, out = w.report()
        check(v["arm_A"] == "MISSING" and "second unfinished run" in out and row == 2,
              "two unfinished folders make the key MISSING, a third finished one does not count")
        w = world().full()
        w.add("torch_noneg", 7, *HEALTHY, cfg={"locked_settings_of_m5_step1": False}, locked=False)
        v, row, out = w.report()
        check(row == 4 and "ignored (not the locked settings" in out, "a folder without the locked settings is ignored")

        # A0 reading, logging, pins
        v, row, out = world().full().report()
        check("exactly (a reading, no rule): yes" in out, "A0 reading: yes at exactly 0.8186666666666667")
        v, row, out = world().full(a0=(0.8187, stats(2954, 1.0))).report()
        check("exactly (a reading, no rule): no" in out and v["driver_check"] == "MET", "A0 reading: no at 0.8187")
        w = world().full()
        with contextlib.redirect_stdout(io.StringIO()):
            rep.main(["--results", str(w.res), "--freeze", str(w.freeze)])
        logged = sorted((root / "results").glob("*_gate0_cause_report"))
        check(len(logged) == 1 and "OUTCOME 4:" in (logged[0] / "report.md").read_text(),
              "the default run writes report.md into one run folder (none for the --no-log runs)")
        check(rep.PIN["gate0_cause.py"] == hashlib.sha256((toyutil.SRC / "gate0_cause.py").read_bytes()).hexdigest()
              and rep.PIN["runlog.py"] == hashlib.sha256((toyutil.SRC / "runlog.py").read_bytes()).hexdigest(),
              "the checker's pins equal the SHA-256 of src/gate0_cause.py and src/runlog.py")
    check.done()


if __name__ == "__main__":
    main()
