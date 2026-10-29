"""Tests of queue_reg_jobs.py on temporary files (standard library only).

run: ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -I test_queue_reg_jobs.py
"""
import os
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import queue_reg_jobs as Q  # noqa: E402

OUT = []


def check(name, ok):
    OUT.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(OUT[-1])


jobs = ["a --x", "b cmd:echo b", "c cmd:echo c", "extra1 cmd:echo e1", "extra2 cmd:echo e2"]
log = ("2026-10-27T00:00:00Z queue start: jobs\n2026-10-27T00:00:01Z a start: --x\n2026-10-27T00:01:00Z a exit 0\n"
       "2026-10-27T00:01:10Z b start: cmd:echo b\n2026-10-27T00:02:00Z b exit 0\n2026-10-27T00:02:10Z c start: cmd:echo c\n")
new = ["reg_ft_star cmd:x", "reg_llm_star cmd:y", ""]
out, k = Q.plan(jobs, log, new)
check("the running job c and the lines before it stay at lines 1-3", out[:3] == jobs[:3] and k == 3)
check("the new lines come next (blank lines dropped), then the pending extras",
      out[3:] == ["reg_ft_star cmd:x", "reg_llm_star cmd:y", "extra1 cmd:echo e1", "extra2 cmd:echo e2"])
log2 = log + "2026-10-27T00:03:00Z c exit 0\n2026-10-27T00:03:10Z extra1 start: cmd:echo e1\n"
out2, k2 = Q.plan(jobs, log2, new)
check("when extra1 runs, the new lines go after it", k2 == 4 and out2[4:6] == ["reg_ft_star cmd:x", "reg_llm_star cmd:y"]
      and out2[6] == "extra2 cmd:echo e2")
try:
    Q.plan(jobs + ["reg_ft_star cmd:z"], log, new)
    check("a new name already in the file is refused", False)
except SystemExit as e:
    check("a new name already in the file is refused", "already" in str(e))
try:
    Q.plan(jobs + ["c cmd:again"], log, new)
    check("a last started name found twice is refused", False)
except SystemExit as e:
    check("a last started name found twice is refused", "exactly once" in str(e))
with tempfile.TemporaryDirectory() as tmp:
    jp, lp, np_ = (os.path.join(tmp, n) for n in ("jobs.txt", "queue.log", "new.txt"))
    open(jp, "w").write("\n".join(jobs) + "\n")
    open(lp, "w").write(log)
    open(np_, "w").write("\n".join(new) + "\n")
    Q.main(["--jobs", jp, "--log", lp, "--new", np_])
    got = open(jp).read().splitlines()
    check("the file is rewritten in place with 7 lines in the planned order", got == out and len(got) == 7)
n_fail = sum(x.startswith("FAIL") for x in OUT)
print(f"{len(OUT) - n_fail} passed, {n_fail} failed")
print("PASSED" if n_fail == 0 else "FAILED")
sys.exit(1 if n_fail else 0)
