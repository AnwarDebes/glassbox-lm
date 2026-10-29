"""Tests of freeze_reg.py (standard library only).

run: cd studies/agent_vs_llm && ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -I \
         test_freeze_reg.py
1. --list: every pinned path exists now, except the three files written at the freeze (jobs_reg.txt,
   runs/llm_star_choice.json and runs/reg_rows.json); the test example files of all three datasets are among the pins (hashed, never parsed).
2. On a temporary registration with a small pin list: the record is written once (a second freeze is refused), its
   hashes equal hashlib's, --check says EQUAL, and --check reports CHANGED after one pinned byte changes.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import freeze_reg as FR  # noqa: E402

OUT = []


def check(name, ok):
    OUT.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(OUT[-1])
    return ok


def test_list():
    paths = FR.pinned_paths()
    later = {os.path.join(H, x) for x in FR.WRITTEN_AT_FREEZE}
    missing = [p for p in paths if not os.path.exists(p) and p not in later]
    check(f"{len(paths)} pinned paths; none missing except the four written at the freeze", not missing)
    for p in missing[:10]:
        print("   missing:", p)
    tests = [p for p in paths if p.endswith("examples_test.json.gz")]
    check("the three test example files are pinned", len(tests) == 3)
    check("llama.cpp shared libraries are pinned", sum(p.endswith(".so") or ".so." in p for p in paths) >= 3)
    check("AGENT's registered seed files (R22, R23, R24; seeds 1-5) are pinned",
          sum(("/R22star/" in p or "/R23abcd/" in p or "/R24mwoz/" in p) and p.endswith(".npz") and "/seed" in p
              for p in paths) == 15)
    check("both model files, llama-server and the C4 table are pinned",
          any(p.endswith("Qwen3.6-35B-A3B-Q4_K_M.gguf") for p in paths)
          and any(p.endswith("Qwen3.5-9B-Q8_0.gguf") for p in paths)
          and any(p.endswith("llama-server") for p in paths) and any(p.endswith("c4words_S1000_v1/table.npz") for p in paths))
    check("bge-base snapshot files are pinned", sum("bge-base-en-v1.5" in p or "/blobs/" in p for p in paths) >= 3)
    check("dev preds files that select LLM* are pinned", sum(p.endswith("preds.jsonl") for p in paths) >= 10)


def test_freeze_and_check():
    with tempfile.TemporaryDirectory() as tmp:
        reg = os.path.join(tmp, "REG_x.md")
        f1, f2 = os.path.join(tmp, "a.txt"), os.path.join(tmp, "b.bin")
        open(reg, "w").write("registration text\n")
        open(f1, "w").write("alpha\n")
        open(f2, "wb").write(bytes(range(256)) * 1000)
        orig = FR.pinned_paths
        FR.pinned_paths = lambda: [f1, f2]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = FR.main(["--reg", reg])
            rec = json.load(open(os.path.join(tmp, "REG_x.frozen.json")))
            check("record written", rc == 0 and rec["n_pins"] == 2)
            check("hashes equal hashlib's", rec["sha256"] == hashlib.sha256(open(reg, "rb").read()).hexdigest()
                  and rec["pins"][f2] == hashlib.sha256(open(f2, "rb").read()).hexdigest())
            check("freeze time has the form YYYY-MM-DDTHH:MM:SSZ",
                  len(rec["frozen_utc"]) == 20 and rec["frozen_utc"][10] == "T" and rec["frozen_utc"].endswith("Z"))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    FR.main(["--reg", reg])
                check("second freeze refused", False)
            except SystemExit as e:
                check("second freeze refused", "refused" in str(e))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = FR.main(["--check", os.path.join(tmp, "REG_x.frozen.json")])
            check("--check: all pins EQUAL", rc == 0 and "3 of 3 pins EQUAL" in buf.getvalue())
            open(f1, "w").write("alphA\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = FR.main(["--check", os.path.join(tmp, "REG_x.frozen.json")])
            check("--check: one changed file reported", rc == 1 and "CHANGED " + f1 in buf.getvalue())
        finally:
            FR.pinned_paths = orig


def test_snapshot_restore_lock():
    with tempfile.TemporaryDirectory() as tmp:
        reg = os.path.join(tmp, "REG_y.md")
        f1, f2 = os.path.join(tmp, "a.txt"), os.path.join(tmp, "sub", "b.bin")
        big = os.path.join(tmp, "models", "m.gguf")
        os.makedirs(os.path.dirname(f2))
        os.makedirs(os.path.dirname(big))
        open(reg, "w").write("registration text\n")
        open(f1, "w").write("alpha\n")
        open(f2, "wb").write(bytes(range(256)) * 100)
        open(big, "wb").write(b"model bytes")
        rec_path = os.path.join(tmp, "REG_y.frozen.json")
        snap = os.path.join(tmp, "snap")
        orig, orig_skip, orig_h = FR.pinned_paths, FR.SNAPSHOT_SKIP, FR.H
        FR.pinned_paths = lambda: [f1, f2, big]
        FR.SNAPSHOT_SKIP = (os.path.join(tmp, "models") + "/",)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                FR.main(["--reg", reg])
                rc = FR.main(["--snapshot", rec_path, "--dir", snap])
            man = json.load(open(os.path.join(snap, "SNAPSHOT.json")))
            copy1 = os.path.join(snap, f1.lstrip("/"))
            check("snapshot: the registration, the record and the 2 pins copied; the skipped model file not",
                  rc == 0 and man["n_files"] == 4 and big not in man["files"] and os.path.exists(copy1)
                  and not os.path.exists(os.path.join(snap, big.lstrip("/"))))
            check("snapshot copies are read-only and equal the record's hashes",
                  (os.stat(copy1).st_mode & 0o222) == 0
                  and FR.sha256(copy1) == json.load(open(rec_path))["pins"][f1])
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    FR.main(["--snapshot", rec_path, "--dir", snap])
                check("a second snapshot into the same folder is refused", False)
            except SystemExit as e:
                check("a second snapshot into the same folder is refused", "refused" in str(e))
            open(f1, "w").write("changed\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = FR.main(["--restore", rec_path, "--dir", snap])
                rc2 = FR.main(["--check", rec_path])
            kept = [x for x in os.listdir(tmp) if x.startswith("a.txt.changed_")]
            check("restore: a CHANGED pin is put back, the changed file kept, --check EQUAL again",
                  rc == 0 and "RESTORED " + f1 in buf.getvalue() and open(f1).read() == "alpha\n" and len(kept) == 1
                  and open(os.path.join(tmp, kept[0])).read() == "changed\n" and rc2 == 0)
            open(big, "wb").write(b"other bytes")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = FR.main(["--restore", rec_path, "--dir", snap])
            check("restore: a CHANGED pin with no snapshot copy is NOT RESTORABLE (exit 1)",
                  rc == 1 and "NOT RESTORABLE " + big in buf.getvalue())
            FR.H = tmp
            with contextlib.redirect_stdout(io.StringIO()):
                rc = FR.main(["--lock", rec_path])
            check("lock: write bits off for the registration, its record and the pins in the folder",
                  rc == 0 and all((os.stat(x).st_mode & 0o222) == 0 for x in (reg, rec_path, f1, f2, big)))
            for x in (reg, rec_path, f1, f2, big):
                os.chmod(x, 0o644)
        finally:
            FR.pinned_paths, FR.SNAPSHOT_SKIP, FR.H = orig, orig_skip, orig_h
            for root, dirs, files in os.walk(tmp):
                for x in dirs + files:
                    os.chmod(os.path.join(root, x), 0o755)


if __name__ == "__main__":
    test_list()
    test_freeze_and_check()
    test_snapshot_restore_lock()
    n_fail = sum(x.startswith("FAIL") for x in OUT)
    print(f"{len(OUT) - n_fail} passed, {n_fail} failed")
    print("PASSED" if n_fail == 0 else "FAILED")
    sys.exit(1 if n_fail else 0)
