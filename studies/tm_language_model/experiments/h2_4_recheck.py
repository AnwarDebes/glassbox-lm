"""H2.4 and Gate 3 re-check on the saved receipts of the E11 run, with the receipt checker that binds every
decision to its generation run (bug TLM-B2, fixed 2026-10-25; docs/13 H2.4). The rules are those of the
docs/13 amendment that carries the words "re-check of H2.4" in its heading (draft:
docs/drafts/H2_4_recheck_note_DRAFT_2026-10-25.md), fixed before the re-check runs.

Nothing is generated, trained or scored, and no data split is opened: the inputs are the files of the E11
run folder (`bank.npz` and the fifteen `receipts_<decoder>_seed<N>.json`).

Steps, in this order (a step that fails stops the run before the next one):
1. Integrity: the sha256 of MANIFEST.json, bank.npz and every receipts file equals the literal pins of the
   config (`params.input_sha256`), and every file also equals its MANIFEST.json entry. Else exit 3.
2. Selection (no checker call): the tampered receipts of E11's selection for the fifteen modes of
   tmlm.certify.receipt.TAMPER_MODES (candidates = every token of every document in E11's file order,
   stride = candidates // tamper-per-mode, at most tamper-per-mode per mode, `tamper(..., seed=1)`). The count
   per mode must equal the config's `params.tamper_counts` (E11's counts for the twelve old modes, and the
   counts of the selection-only pass, written into the config before the run, for the three new ones).
   Else exit 5 ("invalid selection"), before any receipt is checked.
3. Receipts: `src/tmlm/certify/checker_standalone.py` runs as a SEPARATE PROCESS on every receipts file, with
   `--seed N` (N from the file name: the registered seeds 1 to 5, docs/13 section 3) and `--temperature 1.0`
   (the decoders' default, with which E11 ran: its config has no temperature field). The full per-file reports
   are kept in the output folder. A failing receipt is classified as failing items 1 to 5 (the checker as it
   ran in E11) or item 6 only (the binding). A checker process that raises an exception counts every receipt of
   its file as failed; one killed by a signal stops the run with exit 4 (a resource problem: re-run unchanged).
4. Tampering: every receipt of step 2 is checked for its tampered token, with the same expected seed and
   temperature; per mode and per decoder the counts of tampered receipts, of those that fail, of those that
   fail items 1 to 5 and of those that pass items 1 to 5 and fail only item 6 (the binding), and every accepted
   one by name.

Validity (else no verdict input is written): 11,400 receipts checked, 20 runs in every file, and the file of
the smoke (receipts_anyorder_seed1.json) gives the smoke's 640 of 640. Run with every file and the default
tamper count (the registered form), the metrics hold the counts and the verdict inputs; run on a subset
(`--files`) or with another tamper count it is a smoke and writes no verdict input. `--selection-only` runs
steps 1 and 2 and stops: it never calls the checker.

Exit status: 0 the re-check ran (whatever it found), 3 an input differs from its pin, 4 a checker process was
killed by a signal, 5 the selection differs from the declared counts.

    CUDA_VISIBLE_DEVICES= scripts/cpu_slot.sh 1 -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --scope group \\
        --limit-gib 40 -- scripts/run_logged.py --no-gpu-sampler --name h2_4_recheck \\
        --config experiments/configs/h2_4_recheck.json -- \\
        .venv-tm/bin/python -u experiments/h2_4_recheck.py --run-dir results/20260922T232251Z_E11
    selection only (no checker call):
    .venv-tm/bin/python experiments/h2_4_recheck.py --run-dir results/20260922T232251Z_E11 --selection-only --out-dir <scratch>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from engine_common import ROOT, measured

from tmlm.certify.receipt import TAMPER_MODES, tamper
from tmlm.decode.bank import Bank

CHECKER = ROOT / "src" / "tmlm" / "certify" / "checker_standalone.py"
CONFIG = ROOT / "experiments" / "configs" / "h2_4_recheck.json"
E11_SEEDS = (1, 2, 3, 4, 5)
E11_DECODERS = ("anyorder", "diffusion", "eliminate")
E11_TEMPERATURE = 1.0
E11_TAMPER_PER_MODE = 100
NEW_MODES = ("forged_draw", "temperature", "mode_switch")
NAME = re.compile(r"^receipts_(anyorder|diffusion|eliminate)_seed([0-9]+)\.json$")
EXIT_OK, EXIT_INPUTS, EXIT_KILLED, EXIT_SELECTION = 0, 3, 4, 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def e11_files() -> list[str]:
    """The receipts files in the order E11 wrote them (seeds outer, decoders inner)."""
    return [f"receipts_{dec}_seed{seed}.json" for seed in E11_SEEDS for dec in E11_DECODERS]


def seed_of(name: str) -> int:
    m = NAME.match(name)
    if not m:
        raise ValueError(f"not a receipts file name: {name}")
    return int(m.group(2))


def decoder_of(name: str) -> str:
    return NAME.match(name).group(1)


def load_checker():
    spec = importlib.util.spec_from_file_location("checker_standalone", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def peak_rss_gib() -> float | None:
    """VmHWM of this process (the peak resident memory), in GiB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0 ** 2, 2)
    except OSError:
        pass
    return None


def one_sided_upper_95(n_accepted: int, n: int) -> float | None:
    """With 0 accepted of n: the one-sided 95% upper bound 1 - 0.05^(1/n) on the acceptance rate."""
    if n == 0:
        return None
    return float(1.0 - 0.05 ** (1.0 / n)) if n_accepted == 0 else None


def integrity(run_dir: Path, names: list[str], pins: dict) -> tuple[list[str], dict]:
    """Mismatches of every input against the literal pins and against MANIFEST.json; the sha256 read."""
    got, bad = {}, []
    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for name in ["MANIFEST.json", "bank.npz", *names]:
        path = run_dir / name
        if not path.is_file():
            bad.append(f"{name}: missing")
            continue
        got[name] = sha256_file(path)
        if pins.get(name) != got[name]:
            bad.append(f"{name}: differs from its pin")
        if name != "MANIFEST.json" and (manifest.get(name, {}).get("sha256") != got[name]
                                        or manifest.get(name, {}).get("size") != path.stat().st_size):
            bad.append(f"{name}: differs from MANIFEST.json")
    return bad, got


def selection(all_docs: list[tuple[str, dict]], bank, tamper_per_mode: int) -> dict:
    """{mode: [(file, run, token, tampered doc)]}: E11's stride walk, tamper() only (no checker)."""
    out = {}
    cands = [(name, k, doc, i) for name, k, doc in all_docs for i in range(len(doc["tokens"]))]
    stride = max(1, len(cands) // max(1, tamper_per_mode))
    for mode in TAMPER_MODES:
        picked = []
        for name, k, doc, i in cands[::stride]:
            t = tamper(doc, i, mode, bank, seed=1)
            if t is None:
                continue
            picked.append((name, k, i, t))
            if len(picked) >= tamper_per_mode:
                break
        out[mode] = picked
    return out


def run(run_dir: Path, names: list[str], tamper_per_mode: int, temperature: float, out: Path, cfg: dict,
        selection_only: bool = False) -> tuple[dict, int]:
    params = cfg["params"]
    metrics: dict = {}
    out.mkdir(parents=True, exist_ok=True)
    smoke = names != e11_files() or tamper_per_mode != E11_TAMPER_PER_MODE or temperature != E11_TEMPERATURE
    metrics["smoke"] = measured(smoke, note="True: a subset or other settings, no verdict input is written")
    metrics["run_dir"] = measured(str(run_dir))
    metrics["files"] = measured(names)
    metrics["pid"] = measured(os.getpid())
    metrics["checker_sha256"] = measured(sha256_file(CHECKER))
    metrics["receipt_py_sha256"] = measured(sha256_file(ROOT / "src" / "tmlm" / "certify" / "receipt.py"))
    t_all = time.time()

    # 1. integrity against the literal pins and the manifest
    bad, got = integrity(run_dir, names, params["input_sha256"])
    metrics["input_sha256"] = measured(got)
    metrics["input_mismatches"] = measured(bad)
    if bad:
        print("INPUTS DIFFER FROM THEIR PINS: " + "; ".join(bad), flush=True)
        return metrics, EXIT_INPUTS

    # 2. selection, before any checker call
    bank = Bank.load(run_dir / "bank.npz")
    all_docs = [(name, k, doc) for name in names
                for k, doc in enumerate(json.loads((run_dir / name).read_text(encoding="utf-8"))["runs"])]
    metrics["runs_per_file"] = measured({name: sum(1 for n, _, _ in all_docs if n == name) for name in names})
    t0 = time.time()
    picked = selection(all_docs, bank, tamper_per_mode)
    counts = {mode: len(p) for mode, p in picked.items()}
    per_decoder = {mode: {dec: sum(1 for n, _, _, _ in p if decoder_of(n) == dec) for dec in E11_DECODERS}
                   for mode, p in picked.items()}
    metrics["selection.counts"] = measured(counts)
    metrics["selection.counts_per_decoder"] = measured(per_decoder)
    metrics["selection_wall_s"] = measured(time.time() - t0)
    if selection_only:
        print("selection only: " + json.dumps(counts), flush=True)
        return metrics, EXIT_OK
    if not smoke and counts != params["tamper_counts"]:
        metrics["selection.invalid"] = measured({m: (counts[m], params["tamper_counts"].get(m)) for m in counts
                                                 if counts[m] != params["tamper_counts"].get(m)})
        print("INVALID SELECTION: " + json.dumps(metrics["selection.invalid"]["value"]), flush=True)
        return metrics, EXIT_SELECTION

    # 3. every receipt, the checker as a separate process
    chk = load_checker()
    lb = chk.load_bank(str(run_dir / "bank.npz"))
    docs_of = {}
    for name, k, doc in all_docs:
        docs_of.setdefault(name, []).append(doc)
    n_tok = n_pass = 0
    kinds_total = {"items_1_to_5": 0, "binding_only": 0}
    t0 = time.time()
    for name in names:
        report = out / f"report_{name}"
        cmd = [sys.executable, str(CHECKER), "--bank", str(run_dir / "bank.npz"), "--receipts", str(run_dir / name),
               "--seed", str(seed_of(name)), "--temperature", repr(float(temperature)), "--json", str(report), "--quiet"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        n_file = sum(len(d["tokens"]) for d in docs_of[name])
        if proc.returncode < 0:
            metrics[f"receipts.{name}"] = measured("checker killed", signal=-proc.returncode, stderr_tail=proc.stderr[-2000:])
            print(f"{name}: CHECKER KILLED by signal {-proc.returncode}: re-run unchanged", flush=True)
            return metrics, EXIT_KILLED
        if not report.is_file():  # an exception inside the checker: every receipt of the file counts as failed
            metrics[f"receipts.{name}"] = measured(f"0/{n_file} pass (checker raised)", checker_exit=proc.returncode,
                                                   stderr_tail=proc.stderr[-2000:])
            n_tok += n_file
            print(f"{name}: CHECKER RAISED (exit {proc.returncode}); {n_file} receipts counted as failed", flush=True)
            continue
        rep = json.loads(report.read_text(encoding="utf-8"))
        kinds = {"items_1_to_5": 0, "binding_only": 0}
        for r in rep["results"]:
            if r["pass"]:
                continue
            doc = docs_of[name][r.get("run", 0)]
            kind = "items_1_to_5" if chk.rederivation_errors(lb, doc, r["index"]) else "binding_only"
            kinds[kind] += 1
        for k in kinds:
            kinds_total[k] += kinds[k]
        metrics[f"receipts.{name}"] = measured(f"{rep['n_pass']}/{rep['n_tokens']} pass", runs=rep["n_runs"],
                                               checker_exit=proc.returncode, failures_by_kind=kinds,
                                               report=report.name, report_sha256=sha256_file(report))
        n_tok += rep["n_tokens"]
        n_pass += rep["n_pass"]
        print(f"{name}: {rep['n_pass']}/{rep['n_tokens']} pass (checker exit {proc.returncode})", flush=True)
    metrics["receipts_checked"] = measured(n_tok)
    metrics["receipts_pass"] = measured(n_pass)
    metrics["receipts_failures_by_kind"] = measured(kinds_total)
    metrics["receipts_wall_s"] = measured(time.time() - t0)

    # 4. tampering: the receipts of step 2, one check each, with the reading by kind
    t0 = time.time()
    tamper_tot = tamper_fail = 0
    binding_only_new = {}
    for mode in TAMPER_MODES:
        stats = {dec: {"n": 0, "failed": 0, "fail_items_1_to_5": 0, "binding_only": 0} for dec in E11_DECODERS}
        accepted = []
        for name, k, i, t in picked[mode]:
            seed = seed_of(name)
            errs = chk.token_errors(lb, t, i, expected_seed=seed, expected_temperature=temperature)
            e15 = chk.rederivation_errors(lb, t, i)
            e6 = chk.binding_only_errors(t, i, expected_seed=seed, expected_temperature=temperature)
            s = stats[decoder_of(name)]
            s["n"] += 1
            s["failed"] += int(bool(errs))
            s["fail_items_1_to_5"] += int(bool(e15))
            s["binding_only"] += int(not e15 and bool(e6))
            if not errs:
                accepted.append({"file": name, "run": k, "token": i})
        n = sum(s["n"] for s in stats.values())
        f = sum(s["failed"] for s in stats.values())
        metrics[f"tamper.{mode}"] = measured(f"{f}/{n} failed", per_decoder=stats, accepted=accepted,
                                             upper_95_acceptance=one_sided_upper_95(n - f, n))
        if mode in NEW_MODES:
            binding_only_new[mode] = bool(n) and all(s["binding_only"] == s["n"] for s in stats.values())
        tamper_tot += n
        tamper_fail += f
        print(f"tamper {mode}: {f}/{n} failed", flush=True)
    metrics["tamper_wall_s"] = measured(time.time() - t0)
    metrics["tampered_total"] = measured(tamper_tot)
    metrics["tampered_failed"] = measured(tamper_fail)
    metrics["new_modes_pass_items_1_to_5_and_fail_only_the_binding"] = measured(binding_only_new)

    # validity and verdict inputs
    smoke_row = params["smoke_row"]
    valid = {"receipts_checked": n_tok == int(params["receipts_checked"]),
             "runs_per_file": all(v == int(params["runs_per_file"]) for v in metrics["runs_per_file"]["value"].values()),
             "smoke_row": metrics.get(f"receipts.{smoke_row['file']}", {}).get("value") == smoke_row["result"]}
    metrics["validity"] = measured(valid)
    if not smoke and all(valid.values()):
        metrics["H2_4_recheck_all_receipts_pass"] = measured(bool(n_tok > 0 and n_pass == n_tok))
        metrics["H2_4_recheck_all_tampered_fail"] = measured(bool(tamper_tot > 0 and tamper_fail == tamper_tot))
        metrics["H2_4_recheck_forged_draw_qualifier_lifted"] = measured(bool(all(binding_only_new.values())
                                                                              and tamper_fail == tamper_tot))
    metrics["total_wall_s"] = measured(time.time() - t_all)
    print(f"receipts {n_pass}/{n_tok} pass; tampered {tamper_fail}/{tamper_tot} fail; valid {valid}; smoke {smoke}",
          flush=True)
    return metrics, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "20260922T232251Z_E11"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--files", default=None, help="comma list of receipts file names (a smoke); default: all 15")
    ap.add_argument("--tamper-per-mode", type=int, default=E11_TAMPER_PER_MODE)
    ap.add_argument("--temperature", type=float, default=E11_TEMPERATURE)
    ap.add_argument("--selection-only", action="store_true", help="integrity and selection only; no checker call")
    ap.add_argument("--out-dir", default=None, help="output folder when not run under scripts/run_logged.py")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    names = args.files.split(",") if args.files else e11_files()
    for name in names:
        seed_of(name)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(os.environ.get("TMLM_RUN_DIR") or args.out_dir or "")
    if not str(out) or str(out) == ".":
        raise SystemExit("not under run_logged.py: give --out-dir (a scratch folder)")
    metrics, code = run(run_dir, names, args.tamper_per_mode, args.temperature, out, cfg, args.selection_only)
    metrics["peak_rss_gib"] = measured(peak_rss_gib(), note="VmHWM of the driver process")
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(metrics)
    else:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
