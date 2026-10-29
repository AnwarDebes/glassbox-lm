"""sha256 of every E4 code file (e4guard/, scripts/, tests/) and of the read-only inputs it depends on."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import hashlib
import json
import sys
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
S = Path(f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
         "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992")


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = sorted([p for d in ("e4guard", "scripts", "tests") for p in (E4 / d).glob("*") if p.is_file()])
out = {"utc_written": sys.argv[2] if len(sys.argv) > 2 else None,
       "e4_code": {str(p.relative_to(E4)): h(p) for p in files},
       "read_only_inputs": {str(p): h(p) for p in [DL / "src/dl/tm.py", DL / "scripts/verify_policy.py",
                                                   DL / "src/dl/cpu_tm_fast.py", DL / "scripts/cpu_load.py",
                                                   S / "data/tau2/domains/retail/tasks.json",
                                                   S / "data/tau2/domains/retail/split_tasks.json",
                                                   S / "data/tau2/domains/retail/policy.md",
                                                   S / "data/tau2/domains/retail/db.json",
                                                   S / "src/tau2/domains/retail/tools.py",
                                                   S / "src/tau2/evaluator/evaluator_env.py",
                                                   S / "src/tau2/environment/environment.py"]}}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
