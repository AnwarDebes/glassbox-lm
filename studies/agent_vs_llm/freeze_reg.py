"""Freeze REG_agent_vs_llm: the full SHA-256 of the registration text and of every file its test depends on.

usage: python3.12 -I freeze_reg.py --reg REG_agent_vs_llm.md            (writes REG_agent_vs_llm.frozen.json once)
       python3.12 -I freeze_reg.py --check REG_agent_vs_llm.frozen.json  (recomputes every pin: EQUAL or CHANGED)
       python3.12 -I freeze_reg.py --list                                (prints the pinned paths only)
       python3.12 -I freeze_reg.py --snapshot REG_agent_vs_llm.frozen.json --dir DIR  (read-only copies, once)
       python3.12 -I freeze_reg.py --restore REG_agent_vs_llm.frozen.json --dir DIR   (a CHANGED pin back from DIR)
       python3.12 -I freeze_reg.py --lock REG_agent_vs_llm.frozen.json   (write bits off for the pins in this folder)
Standard library only. The freeze time is read from the clock when the record is written. An existing record is never
overwritten. Literal hashes are stored (no folder is checked against its own manifest).
The snapshot copies the registration, the record and every pin except the model files, the llama.cpp build, the
bge-base snapshot and the decision_layer data (SNAPSHOT_SKIP: large files no step here rewrites); each copy's hash must
equal the record's. --restore puts a CHANGED pin back from its copy, keeping the changed file as <path>.changed_<utc>
for the report; a CHANGED pin with no copy is reported NOT RESTORABLE (exit 1).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
_HOME = _os.path.expanduser("~")
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

H = f"{_REPO_ROOT}/studies/agent_vs_llm"
DL = f"{_REPO_ROOT}/studies/decision_layer"
M9 = f"{_WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742"
M35 = f"{_WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/gguf-b11742"
BGE = f"{_HOME}/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5"
SCRIPTS = ["agent_preds.py", "ft_baseline.py", "llm_next_action.py", "serve_headline.sh", "reg_report.py",
           "verify_reg.py", "make_reg_jobs.py", "tfidf_reg.py", "select_llm_star.py", "agent_speed.py", "ft_speed.py",
           "run_queue.sh", "freeze_reg.py", "rows_rule.py", "check_reg.py", "reg_llm_job.sh", "combine_reask.py",
           "llm_q3_job5.sh", "readability_dev.py", "tfidf_speed.py", "results_table.py", "design_effect_dev.py", "design_effect_dev.out",
           "code_since_R24.diff", "NOTE_review_fixes.md", "NOTE_review_fixes.frozen.json", "NOTE_review_fixes_add1.md",
           "NOTE_review_fixes_add1.frozen.json", "NOTE_cost_cpu.md",
           "lit_headline_2026-10-27.md", "test_reg_tools.py", "test_reg_tools.out", "test_freeze_reg.py",
           "test_freeze_reg.out", "test_llm_next_action.py", "test_llm_next_action.out", "test_combine_reask.py",
           "test_combine_reask.out", "test_readability.py", "test_readability.out", "queue_reg_jobs.py",
           "test_queue_reg_jobs.py", "test_queue_reg_jobs.out",
           "jobs_reg.txt", "reg_report_all.sh", "runs/llm_star_choice.json", "runs/reg_rows.json",
           "runs/env_versions.txt", "runs/results_table.md"]
WRITTEN_AT_FREEZE = ("jobs_reg.txt", "reg_report_all.sh", "runs/llm_star_choice.json", "runs/reg_rows.json",
                     "runs/queue_at_freeze.log")
CONFIRM = f"{_REPO_ROOT}/studies/decision_layer/results/confirm"
BINDIR = f"{_WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin"
SNAPSHOT_SKIP = (M9 + "/", M35 + "/", BINDIR + "/", BGE + "/", DL + "/data/")
# evidence the registration text names (review of v11, m1 and m11); none is on the rules' computation path
EVIDENCE = ["tfidf_lr.py", "abl_table.py", "q1_table.py", "q3_table.py", "hybrid_table.py", "cascade.py", "zipf_report.py",
            "NOTE_tfidf_dev.addendum.txt", "PLAN_zipf_synth.addendum.txt", "runs/llm_q3_L35t16/star_reask_rows.json",
            "archive/REG_history.txt", "runs/queue_at_freeze.log",
            "reviews/REG_agent_vs_llm_v11_diffcheck_2026-10-27.md"]
EXTERNAL_EVIDENCE = [
    f"{_REPO_ROOT}/studies/decision_layer/tests/test_r24_network_free.py",
    f"{_REPO_ROOT}/studies/decision_layer/logs/R24_network_free.out",
    f"{_REPO_ROOT}/studies/decision_layer/logs/test_r24_network_free_selftest.out",
    f"{_REPO_ROOT}/studies/decision_layer/scripts/table_run.py",
    f"{_REPO_ROOT}/studies/decision_layer/scripts/abcd_confirm.py",
    f"{_REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py",
    f"{_REPO_ROOT}/studies/decision_layer/registrations/R24_multiwoz_random_negatives.md",
    f"{_REPO_ROOT}/studies/decision_layer/registrations/R24_multiwoz_superiority.md",
    f"{_REPO_ROOT}/studies/decision_layer/registrations/R25_refined_table_agent.md",
    f"{_REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh",
    f"{_REPO_ROOT}/studies/text_understanding/w1/build_corpus_c4.py",
    f"{_REPO_ROOT}/studies/text_understanding/w1/build_corpus_c4_ext.py",
    f"{_REPO_ROOT}/studies/text_understanding/w1/corpus/c4_slices.json",
    f"{_REPO_ROOT}/studies/text_understanding/w1/corpus/c4_ext_slices.json",
    f"local:reviews/REG_agent_vs_llm_v9_review_2026-10-27.md",
    f"local:reviews/REG_agent_vs_llm_v11_review_2026-10-27.md"]
AGENT_DIRS = ("R22star/tm_mem_c4_s16", "R23abcd/tm_mem_c4_s16", "R24mwoz/tm_mem_c4_nrand")


def pinned_paths():
    p = [os.path.join(H, s) for s in SCRIPTS]
    p += sorted(glob.glob(os.path.join(DL, "src/dl/*.py")))
    for ds in ("star", "abcd", "multiwoz"):
        p += [os.path.join(DL, "data", ds, f"examples_{s}.json.gz") for s in ("train", "dev", "test")]
        p += [os.path.join(H, f"runs/agent_{ds}_c4_tm_s1/tm.npz")]
    for d in AGENT_DIRS:  # AGENT's registered test predictions (seeds 1-5), records and the seed-1 model
        p += [os.path.join(CONFIRM, d, f"seed{k}.{e}") for k in range(1, 6) for e in ("npz", "json")]
        p += [os.path.join(CONFIRM, d, "model_seed1.npz"), os.path.join(CONFIRM, d, "bit_names.json")]
    p += [os.path.join(DL, "data/tables/c4words_S1000_v1", f) for f in
          ("table.npz", "words.txt", "MANIFEST.sha256", "tokenize_ref.py")]
    p += [os.path.join(DL, "data/untrusted/abcd/guidelines.json")]
    p += sorted(glob.glob(os.path.join(DL, "data/untrusted/star/STAR-master/tasks/*/*.json")))
    p += [os.path.join(M9, "FILES.sha256"), os.path.join(M9, "Qwen3.5-9B-Q8_0.gguf"),
          os.path.join(M35, "FILES.sha256"), os.path.join(M35, "Qwen3.6-35B-A3B-Q4_K_M.gguf")]
    bindir = BINDIR
    p += [os.path.join(bindir, "llama-server")] + sorted(glob.glob(os.path.join(bindir, "*.so*")))
    snap = open(os.path.join(BGE, "refs/main")).read().strip()
    p += sorted(os.path.realpath(x) for x in glob.glob(os.path.join(BGE, "snapshots", snap, "*")) if os.path.isfile(x))
    p += sorted(glob.glob(os.path.join(H, "runs/llm_*/**/preds.jsonl"), recursive=True))
    p += sorted(glob.glob(os.path.join(H, "runs/llm_*/**/result.json"), recursive=True))
    p += sorted(x for x in glob.glob(os.path.join(H, "runs/dry/**/*"), recursive=True) if os.path.isfile(x))
    p += sorted(glob.glob(os.path.join(H, "runs/ft_*/*.json")) + glob.glob(os.path.join(H, "runs/ft_*/*.npz")))
    p += sorted(glob.glob(os.path.join(H, "runs/tfidf_*/result.json")))
    p += sorted(glob.glob(os.path.join(H, "runs/*.json")) + glob.glob(os.path.join(H, "runs/cost/*.json"))
                + glob.glob(os.path.join(H, "runs/readability/*.json")))
    for pat in ("PLAN_*.md", "AMEND_*.md", "NOTE_*.md", "PLAN_*.frozen.json", "AMEND_*.frozen.json", "NOTE_*.frozen.json"):
        p += sorted(glob.glob(os.path.join(H, pat)))
    p += sorted(glob.glob(os.path.join(H, "llm*job*.sh")))  # the dev job scripts behind LLM*'s selection
    p += [os.path.join(H, s) for s in EVIDENCE]
    p += sorted(glob.glob(os.path.join(H, "runs/llm_q3_*/job.log")))
    p += sorted(glob.glob(os.path.join(H, "archive/REG_agent_vs_llm_draft_v*.md")))
    p += sorted(glob.glob(os.path.join(H, "runs/agent_*/result.json")) + glob.glob(os.path.join(H, "runs/agent_*/*.npz")))
    p += sorted(glob.glob(os.path.join(H, "runs/ftseed_*/result.json"))
                + glob.glob(os.path.join(H, "runs/ftseed_*/dev_preds.npz")))
    p += EXTERNAL_EVIDENCE
    return list(dict.fromkeys(p))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def record_items(rec):
    return [(rec["registration_path"], rec["sha256"])] + sorted(rec["pins"].items())


def snapshot(record, dest):
    """Read-only copies of the registration, the record and every pin outside SNAPSHOT_SKIP; taken once."""
    rec = json.load(open(record))
    if os.path.exists(dest):
        sys.exit(f"refused: {dest} exists (a snapshot is taken once)")
    files = {}
    for path, want in record_items(rec) + [(os.path.abspath(record), sha256(record))]:
        if path.startswith(SNAPSHOT_SKIP):
            continue
        copy = os.path.join(dest, path.lstrip("/"))
        os.makedirs(os.path.dirname(copy), exist_ok=True)
        shutil.copy2(path, copy)
        have = sha256(copy)
        if have != want:
            sys.exit(f"refused: {path} differs from the record ({have[:16]} != {want[:16]}); the snapshot is incomplete")
        os.chmod(copy, 0o444)
        files[path] = want
    man = {"record": os.path.abspath(record), "record_sha256": sha256(record), "n_files": len(files),
           "skipped_prefixes": list(SNAPSHOT_SKIP), "files": files,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(dest, "SNAPSHOT.json"), "x") as h:
        json.dump(man, h, indent=1)
    os.chmod(os.path.join(dest, "SNAPSHOT.json"), 0o444)
    print(f"snapshot {man['utc']}: {len(files)} files -> {dest}")
    return 0


def restore(record, dest):
    """Put every CHANGED pin back from its snapshot copy (the changed file kept as <path>.changed_<utc>)."""
    rec = json.load(open(record))
    man = json.load(open(os.path.join(dest, "SNAPSHOT.json")))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bad = 0
    for path, want in record_items(rec):
        have = sha256(path) if os.path.exists(path) else "MISSING"
        if have == want:
            continue
        copy = os.path.join(dest, path.lstrip("/"))
        if man["files"].get(path) == want and os.path.exists(copy) and sha256(copy) == want:
            if os.path.exists(path):
                shutil.copy2(path, f"{path}.changed_{stamp}")
            tmp = f"{path}.restore_tmp"
            shutil.copy2(copy, tmp)
            os.replace(tmp, path)
            print(f"RESTORED {path} (was {have[:16]}; changed copy kept as {path}.changed_{stamp})")
        else:
            bad += 1
            print(f"NOT RESTORABLE {path} {have[:16]} != {want[:16]}")
    print(f"{bad} pins not restorable")
    return 1 if bad else 0


def lock(record):
    """Write bits off for the registration, its record and every pin in this folder (H)."""
    rec = json.load(open(record))
    n = 0
    for path in [os.path.abspath(record)] + [x for x, _ in record_items(rec)]:
        if path.startswith(H + "/") and os.path.exists(path):
            os.chmod(path, os.stat(path).st_mode & ~0o222)
            n += 1
    print(f"locked {n} files in {H} (write bits off)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reg")
    g.add_argument("--check")
    g.add_argument("--list", action="store_true")
    g.add_argument("--snapshot", help="the freeze record; copies go to --dir")
    g.add_argument("--restore", help="the freeze record; copies come from --dir")
    g.add_argument("--lock", help="the freeze record")
    ap.add_argument("--dir", help="the snapshot folder (--snapshot, --restore)")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(pinned_paths()))
        return 0
    if a.snapshot:
        return snapshot(a.snapshot, a.dir)
    if a.restore:
        return restore(a.restore, a.dir)
    if a.lock:
        return lock(a.lock)
    if a.check:
        rec = json.load(open(a.check))
        bad = 0
        for path, want in [(rec["registration_path"], rec["sha256"])] + sorted(rec["pins"].items()):
            have = sha256(path) if os.path.exists(path) else "MISSING"
            if have != want:
                bad += 1
                print(f"CHANGED {path} {have[:16]} != {want[:16]}")
        print(f"{1 + len(rec['pins']) - bad} of {1 + len(rec['pins'])} pins EQUAL; {bad} changed")
        return 1 if bad else 0
    reg = os.path.abspath(a.reg)
    out = os.path.splitext(reg)[0] + ".frozen.json"
    if os.path.exists(out):
        sys.exit(f"refused: {out} exists (a freeze record is written once)")
    paths = pinned_paths()
    missing = [x for x in paths if not os.path.exists(x)]
    if missing:
        sys.exit("refused: missing pinned files:\n" + "\n".join(missing))
    pins = {x: sha256(x) for x in paths}
    rec = {"registration": os.path.basename(reg), "registration_path": reg, "sha256": sha256(reg),
           "n_pins": len(pins), "pins": pins}
    rec["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out, "x") as h:
        json.dump(rec, h, indent=1)
    print(f"frozen {rec['frozen_utc']} {rec['registration']} {rec['sha256']} ({len(pins)} pins) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
