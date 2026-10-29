"""Write the queue lines of the registered test REG_agent_vs_llm and its report script, from the LLM* choice.

usage: python -I make_reg_jobs.py --choice runs/llm_star_choice.json --rows runs/reg_rows.json --reg REG_agent_vs_llm.md \
           --out jobs_reg.txt
Every line runs a script that refuses the test split unless REG_agent_vs_llm.frozen.json exists, and every job line
starts only after freeze_reg.py --check finds every pin unchanged (its output in runs/reg/pincheck_<name>.out). The
registration path is absolute. No agent is trained: AGENT's test predictions are the registered seed files of R22, R23
and R24 (check_reg.AGENT_DIR). Lines: FT (GPU), TFIDF (CPU), LLM* (reg_llm_job.sh), then reg_report_all (check_reg.py,
then reg_report.py and verify_reg.py for each dataset whose gate check_reg.py opened; the script is written next to
the job file as reg_report_all.sh).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
_HOME = _os.path.expanduser("~")
import argparse
import json
import os
import sys

H = f"{_REPO_ROOT}/studies/agent_vs_llm"
sys.path.insert(0, H)
PY = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
PY312 = f"{_HOME}/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
MODEL = {"Qwen3.5-9B-Q8_0": (f"{_WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf", "Q8_0"),
         "Qwen3.6-35B-A3B-Q4_K_M": (f"{_WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/gguf-b11742/"
                                    "Qwen3.6-35B-A3B-Q4_K_M.gguf", "Q4_K_M")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice", required=True)
    ap.add_argument("--rows", required=True, help="rows_rule.py output")
    ap.add_argument("--reg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import check_reg
    ch = json.load(open(a.choice))["datasets"]
    rows = {ds: v["rows"] for ds, v in json.load(open(a.rows))["datasets"].items()}
    reg = os.path.abspath(a.reg)
    record = os.path.splitext(reg)[0] + ".frozen.json"
    jobs_abs, rows_abs, choice_abs = os.path.abspath(a.out), os.path.abspath(a.rows), os.path.abspath(a.choice)

    def pin_gate(name):  # every pin of the freeze record unchanged, or the job does not start
        return (f"mkdir -p {H}/runs/reg && {PY312} -I {H}/freeze_reg.py --check {record} "
                f"> {H}/runs/reg/pincheck_{name}.out 2>&1 && ")

    lines = []
    for ds in ("star", "abcd", "multiwoz"):
        lines.append(f"reg_ft_{ds} cmd:{pin_gate(f'reg_ft_{ds}')}cd {H} && {PY} -I ft_baseline.py --dataset {ds} "
                     f"--model bge-base --lr 5e-5 --seed 1 --test-registration {reg} --out runs/reg/ft_{ds}")
        lines.append(f"reg_tfidf_{ds} cmd:{pin_gate(f'reg_tfidf_{ds}')}cd {H} && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 "
                     f"nice -n 19 {PY} -I tfidf_reg.py --dataset {ds} --test-registration {reg} --out runs/reg/tfidf_{ds}")
    for ds in ("star", "abcd", "multiwoz"):
        c = ch[ds]["llm_star"]
        path, q = MODEL[c["model"]]
        budget = -1 if c["thinking"] else 0
        flags = f"--k {c['k']}" + (" --think" if c["thinking"] else "") + ("" if c["policy_text"] else " --no-guidelines")
        if c.get("max_tokens"):
            flags += f" --max-tokens {c['max_tokens']}"
        lines.append(f"reg_llm_{ds} cmd:{pin_gate(f'reg_llm_{ds}')}bash {H}/reg_llm_job.sh {ds} {path} {q} {budget} "
                     f"{rows[ds]} {H}/runs/reg/llm_{ds} {reg} {flags}")
    script = os.path.join(os.path.dirname(jobs_abs), "reg_report_all.sh")
    gate = ("gate() { " + PY312 + " -I -c 'import json, sys; sys.exit(0 if json.load(open(\"runs/reg/checks.json\"))"
            "[\"gate\"][sys.argv[1]] else 1)' \"$1\"; }")
    sh = ["#!/usr/bin/env bash", "# REG_agent_vs_llm: checks, registered reports and the independent recompute "
          "(written by make_reg_jobs.py). A dataset is reported only when check_reg.py opens its gate.",
          "set -uo pipefail", f"cd {H}", "mkdir -p runs/reg",
          f"CUDA_VISIBLE_DEVICES= {PY312} -I check_reg.py --reg {reg} --rows {rows_abs} --choice {choice_abs} "
          f"--jobs {jobs_abs} --out runs/reg/checks.json > runs/reg/checks.out 2>&1",
          "[ -f runs/reg/checks.json ] || { echo 'check_reg.py wrote no checks.json: no report'; exit 1; }",
          gate, "rc=0"]
    for ds in ("star", "abcd", "multiwoz"):
        files = check_reg.agent_files(ds)
        common = (f"--dataset {ds} --split test --n {rows[ds]} --reg {reg} --agent {files[0]} "
                  f"--llm runs/reg/llm_{ds}/preds.jsonl --ft runs/reg/ft_{ds}/test_preds.npz "
                  f"--tfidf runs/reg/tfidf_{ds}/test_preds.npz")
        sh.append(f"if gate {ds}; then")
        sh.append(f"  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 {PY} -I reg_report.py {common} "
                  f"--agent-seeds {' '.join(files)} --out runs/reg/report_{ds}.json > runs/reg/report_{ds}.out 2>&1 || rc=1")
        sh.append(f"  nice -n 19 {PY312} -I verify_reg.py {common} --report runs/reg/report_{ds}.json "
                  f"> runs/reg/verify_{ds}.out 2>&1 || rc=1")
        sh.append("else")
        sh.append(f"  echo '{ds}: gate closed (a hard check failed; runs/reg/checks.out): no report for {ds}'; rc=1")
        sh.append("fi")
    sh.append('exit "${rc}"')
    with open(script, "w") as h:
        h.write("\n".join(sh) + "\n")
    lines.append(f"reg_report_all cmd:bash {script}")
    with open(a.out, "w") as h:
        h.write("\n".join(lines) + "\n")
    print("\n".join(l[:150] for l in lines))


if __name__ == "__main__":
    main()
