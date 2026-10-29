"""Write the registration's hash tables H1-H5 (and an empty H6) with full sha256 literals, computed by check_pins.spec_hash
(the same function the checker uses at every stage). Output: a Markdown block for REGISTRATION_E4.md section 12.
Revision 3 (re-review item 8): H5 also pins EVERY dist-info of both venvs (one venv: row per site-packages: each
dist-info's name, version and RECORD sha256, every top-level .pth, every other top-level name); the lines behind each
venv: row are written to --venv-lists-dir for reading (check_pins.py --venv-list prints them too). H2 adds the E4.3
power file that gobar.py and summarize_e4.py read.
Run: .../taubench/.venv/bin/python -I -B make_hash_tables.py --out results/hash_tables_rev3.md \
       --venv-lists-dir results/venv_lists_rev3
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/

import argparse
import os
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4 / "scripts"))

from check_pins import spec_hash, venv_lines  # noqa: E402

TB = Path(f"{_REPO_ROOT}/studies/agent_guards/taubench")
S = TB / "data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992"
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
SPT = TB / ".venv/lib/python3.12/site-packages"
SPD = DL / ".venv-dl/lib/python3.11/site-packages"


def code_files():
    out = []
    for d in ("e4guard", "scripts", "tests"):
        out += sorted(str(p.relative_to(E4)) for p in (E4 / d).iterdir() if p.is_file() and p.suffix in (".py", ".sh"))
    return out


def dist(site: Path, prefix: str) -> str:
    m = sorted(site.glob(prefix + "-*.dist-info"))
    assert len(m) == 1, (prefix, m)
    return f"dist:{m[0]}"


TABLES = {
    "H1 E4 code (e4guard/, scripts/, tests/)": code_files,
    "H2 frozen pins and E4 data files": lambda: [
        "pins/server_expected.json", "pins/gguf_chat_template.json", "results/judge_tasks.json",
        "results/equivalence_code_vs_veto_rev2.json", "results/power_e43_rev3.json"],
    "H3 read-only inputs (files)": lambda: [
        str(DL / "src/dl/tm.py"), str(DL / "scripts/verify_policy.py"), str(DL / "src/dl/cpu_tm_fast.py"),
        str(DL / "scripts/cpu_load.py"),
        str(S / "data/tau2/domains/retail/tasks.json"), str(S / "data/tau2/domains/retail/split_tasks.json"),
        str(S / "data/tau2/domains/retail/policy.md"), str(S / "data/tau2/domains/retail/db.json"),
        str(S / "data/tau2/user_simulator/simulation_guidelines.md"),
        str(S / "data/tau2/user_simulator/simulation_guidelines_tools.md"),
        str(S / "src/tau2/domains/retail/tools.py"), str(S / "src/tau2/evaluator/evaluator_env.py"),
        str(S / "src/tau2/environment/environment.py"), str(S / "src/tau2/agent/llm_agent.py"),
        str(S / "src/tau2/user/user_simulator.py"), str(S / "src/tau2/user/user_simulator_base.py"),
        str(S / "src/tau2/orchestrator/orchestrator.py"), str(S / "src/tau2/evaluator/evaluator.py"),
        str(S / "src/tau2/evaluator/evaluator_nl_assertions.py"), str(S / "src/tau2/runner/batch.py"),
        str(S / "src/tau2/runner/progress.py"), str(S / "src/tau2/runner/simulation.py"),
        str(S / "src/tau2/runner/build.py"), str(S / "src/tau2/runner/helpers.py"),
        str(S / "src/tau2/runner/checkpoint.py"), str(S / "src/tau2/utils/llm_utils.py"),
        str(S / "src/tau2/data_model/message.py"), str(S / "src/tau2/data_model/simulation.py"),
        str(S / "src/tau2/registry.py"), str(S / "src/tau2/config.py"),
        str(S / "data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json"),
        str(TB / "data_untrusted/tau2-bench/SHA256SUMS.tarball"),
        str(TB / "data_untrusted/tau2-bench/SHA256SUMS.extracted"),
        str(TB / "data_untrusted/tau2-bench/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992.tar.gz")],
    "H4 tau2 trees (every file; no __pycache__ or .pyc)": lambda: [
        f"tree:{S / 'src/tau2'}", f"tree:{S / 'data/tau2/domains/retail'}", f"tree:{S / 'data/tau2/user_simulator'}"],
    "H5 installed packages, interpreters and the server": lambda: [
        dist(SPT, "litellm"), dist(SPT, "openai"), dist(SPT, "pydantic"), dist(SPT, "pydantic_core"),
        dist(SPT, "httpx"), dist(SPT, "numpy"), dist(SPT, "loguru"),
        os.path.realpath(TB / ".venv/bin/python"),
        dist(SPD, "torch"), dist(SPD, "scikit_learn"), dist(SPD, "numpy"), dist(SPD, "scipy"),
        dist(SPD, "threadpoolctl"), dist(SPD, "joblib"),
        os.path.realpath(DL / ".venv-dl/bin/python"),
        f"venv:{SPT}", f"venv:{SPD}",
        f"{_WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf",
        f"{_WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin/llama-server",
        f"{_REPO_ROOT}/studies/agent_guards/llm/build_b11742_bin.sha256"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--venv-lists-dir", required=True)
    a = ap.parse_args()
    t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    vd = Path(a.venv_lists_dir)
    vd.mkdir(parents=True, exist_ok=True)
    for name, site in (("taubench_venv_site_packages.tsv", SPT), ("decision_layer_venv_dl_site_packages.tsv", SPD)):
        (vd / name).write_text("".join("\t".join(x) + "\n" for x in venv_lines(site)))
    lines = [f"Computed {t0} by scripts/make_hash_tables.py with check_pins.spec_hash (the checker's own function).", ""]
    for name, specs in TABLES.items():
        lines += [f"### {name}", "", "| spec | sha256 | files |", "|---|---|---|"]
        for spec in specs():
            h, n = spec_hash(spec)
            if h == "MISSING":
                raise SystemExit(f"missing: {spec}")
            lines.append(f"| `{spec}` | `{h}` | {n} |")
        lines.append("")
    lines += ["### H6 the smoke outcome (EMPTY until the freeze: filled with exactly the five files of section "
              "11.4 from runs/SMOKE_<campaign>/; check_pins.py requires them from the freeze on)", "",
              "| spec | sha256 | files |", "|---|---|---|", ""]
    lines.append(f"Done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}.")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
