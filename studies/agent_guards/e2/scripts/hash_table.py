"""Full sha256 (64 hex) of every file E2 pins (rev2, review item 6): the E2 code and data, the stand-in models the
smoke's open-loop check uses, every project module on the import path of every entry point (data/pins/
import_closure.json), the C4 table, both venvs' package lists, and the llama-server build files. Any Python:

    python -I -B scripts/hash_table.py --md logs/rev2/hashes.md --json logs/rev2/hashes.json [--write-pins]

--write-pins rewrites the literal PINS block of scripts/e2_check.py (every file below except e2_check.py itself, whose
own hash goes into the registration's Appendix C). Never pinned: LOG.md, README.md, the registration drafts, logs/,
runs/ (the campaign's records) and models/final (created by the build stage; e2_check.py records and checks it).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
import argparse
import hashlib
import json
import re
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
HB = E2.parent
ROOT = Path(f"{_WS_ROOT}")
DL = ROOT / "glassbox-lm" / "studies" / "decision_layer"
SERVER_BIN = ROOT / "tools" / "llama.cpp" / "b11742" / "build-cuda129" / "bin"
SERVER_FILES = ["llama-server", "libllama-server-impl.so", "libllama.so.0.7.0", "libllama-common.so.0.7.0",
                "libggml.so.0.27.0", "libggml-base.so.0.27.0", "libggml-cpu.so.0.27.0", "libggml-cuda.so.0.27.0",
                "libmtmd.so.0.7.0"]


def groups() -> dict:
    closure = json.loads((E2 / "data" / "pins" / "import_closure.json").read_text())
    dep = sorted({ROOT / f for venv in ("agentdojo_venv", "dl_venv") for fs in closure[venv].values() for f in fs
                  if not (ROOT / f).is_relative_to(E2)})
    return {
        "E2 code": sorted((E2 / "src" / "e2guard").glob("*.py")) + sorted((E2 / "src" / "e2guard").glob("*.json"))
        + sorted((E2 / "scripts").glob("*.py")) + sorted((E2 / "scripts").glob("*.sh")) + [E2 / "run_e2_gpu_slot.sh"]
        + sorted((E2 / "tests").glob("*.py")) + sorted((E2 / "tests").glob("*.sh")),
        "E2 data": [E2 / "data" / n for n in ("tool_docs_v1.2.2.json", "tasks_v1.2.2.json", "attack_selections_v1.2.2.json",
                                              "rows_gt_v1.2.2.jsonl", "rows_gt_v1.2.2.meta.json",
                                              "rows_standin_traces.jsonl", "rows_standin_traces.meta.json")]
        + sorted((E2 / "data" / "query_bits").glob("*.json")) + sorted((E2 / "data" / "pins").glob("*")),
        "stand-in models and offline results (code tests; the smoke's open-loop decide() check)":
            sorted((E2 / "models" / "standin").glob("*.json")) + sorted((E2 / "results").glob("*")),
        "read-only dependencies (every project module on an entry point's import path, the table, the venv lists)":
            dep + [ROOT / "glassbox-lm" / "studies" / "text_understanding" / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz",
                   HB / "agentdojo" / "pip_freeze.txt", HB / "agentdojo" / "harness" / "replay.py",
                   DL / "scripts" / "cpu_load.py"],
        "llama-server build b11742 (commit 4c3c5810f87efe177949f1b85ae509d38a2a311f)": [SERVER_BIN / n for n in SERVER_FILES],
    }


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--write-pins", action="store_true")
    a = ap.parse_args()
    g = groups()
    table = {name: {rel(p): (sha(p) if p.exists() else "MISSING") for p in files} for name, files in g.items()}
    missing = [f for t in table.values() for f, h in t.items() if h == "MISSING"]
    assert not missing, f"missing files: {missing}"
    check = E2 / "scripts" / "e2_check.py"
    if a.write_pins:
        pins = {f: h for t in table.values() for f, h in t.items() if f != rel(check)}
        s = check.read_text()
        block = "PINS: dict[str, str] = {\n" + "".join(f"    {f!r}: {h!r},\n" for f, h in sorted(pins.items())) + "}\n"
        s2, n = re.subn(r"(# BEGIN PINS[^\n]*\n)PINS: dict\[str, str\] = \{.*?\}\n(# END PINS)",
                        lambda m: m.group(1) + block + m.group(2), s, flags=re.DOTALL)
        assert n == 1, "PINS block not found"
        check.write_text(s2)
    table["E2 code"][rel(check)] = sha(check)  # after --write-pins: the checker's own hash
    lines = []
    for name, t in table.items():
        lines += [f"\n**{name}**\n", "| file | sha256 |", "|---|---|"]
        lines += [f"| `{f}` | `{h}` |" for f, h in sorted(t.items())]
    Path(a.md).write_text("\n".join(lines) + "\n")
    Path(a.json).write_text(json.dumps(table, indent=1) + "\n")
    print(f"{sum(len(t) for t in table.values())} files; e2_check.py sha256 {table['E2 code'][rel(check)]}")


if __name__ == "__main__":
    main()
