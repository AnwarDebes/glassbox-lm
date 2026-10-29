"""Entry script for ../../test_patches.py, run by review/bench/with_patch.py AFTER a patch file, i.e. the mechanism of
the GPU jobs (run_gpu_job.sh IMPL), here on the CPU (CUDA hidden). Checks:
  - the patch replaced dl.tm_init.CoTM and nothing else checked here changed (dl.tm.CoTM is the study's class, make_tm and
    TMConfig are untouched), and the search path no longer holds the entry fast_cotm.py adds (the decision_layer src by its
    path; with_patch.py itself put the run root's src link there);
  - make_tm (init half and init empty) now builds the patch's class, a FastCoTM whose switches equal the named
    configuration, on the requested device;
  - a short CPU training through make_tm equals dl.tm.CoTM's (ta, w, generator state, dev history, best_epoch),
    unchunked and in 2 chunks, on the first --rows rows of the decision_layer's cached ABCD view-B bits (tmcode/prof_tm.py loader);
  - --second-patch P: running another patch file afterwards in the same process is refused (SystemExit).
Writes --out (JSON); exit 0 if every check holds.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../../../.."))

import argparse
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

DL_SRC = f"{_REPO_ROOT}/studies/decision_layer/src"
PROF_TM = Path(f"{_REPO_ROOT}/studies/decision_layer/speed/tmcode/prof_tm.py")
# the published prof_tm.py (as run: e2774df7be58425e7c79fe36423f8d12d1d47bdfdfb2fcc9ab0068b1223ec6f1)
PROF_TM_SHA256 = "71f1a483b16cf8bba1c93d9591d9e8463a51df6d6c950203ec006f9423f32a41"

path_has_dl_src = DL_SRC in sys.path  # read before anything below can add it

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dl.tm as T  # noqa: E402
import dl.tm_init as ti  # noqa: E402


def same(a, b) -> dict:
    return {
        "ta": bool(np.array_equal(a.ta.numpy(), b.ta.numpy())),
        "w": bool(np.array_equal(a.w.numpy(), b.w.numpy())),
        "gen_state": bool(torch.equal(a.gen.get_state(), b.gen.get_state())),
        "history": [h["val_acc"] for h in a.history] == [h["val_acc"] for h in b.history],
        "best_epoch": getattr(a, "best_epoch", None) == getattr(b, "best_epoch", None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--clauses", type=int, default=64)
    ap.add_argument("--second-patch", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(2)
    fc = sys.modules.get("speed_p2tm_fast_cotm")
    cf = sys.modules.get("speed_p2tm_configs")
    res = {"name": a.name, "checks": {}, "cases": []}
    c = res["checks"]
    c["patch_modules_loaded"] = fc is not None and cf is not None
    if not c["patch_modules_loaded"]:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
        raise SystemExit(1)
    want = dict(getattr(cf, a.name))
    c["tm_init_CoTM_replaced"] = ti.CoTM is not T.CoTM
    c["replacement_subclasses_FastCoTM"] = isinstance(ti.CoTM, type) and issubclass(ti.CoTM, fc.FastCoTM)
    c["dl_tm_CoTM_is_the_studys"] = T.CoTM.__module__ == "dl.tm" and T.CoTM.__qualname__ == "CoTM" and fc.CoTM is T.CoTM
    c["make_tm_and_TMConfig_untouched"] = ti.make_tm.__module__ == "dl.tm_init" and ti.TMConfig is T.TMConfig
    c["search_path_restored"] = not path_has_dl_src
    if a.second_patch is not None:
        try:
            runpy.run_path(str(a.second_patch.resolve()), run_name="speed_patch")
            c["second_patch_refused"] = False
        except SystemExit as e:
            c["second_patch_refused"] = "already replaced" in str(e)
            res["second_patch_message"] = str(e)
    # data and the CPU no-op for fit()'s final torch.cuda.synchronize (prof_tm sets it; it also puts the decision_layer src
    # on sys.path, which is why the search-path check above runs first)
    got = hashlib.sha256(PROF_TM.read_bytes()).hexdigest()
    if got != PROF_TM_SHA256:
        raise SystemExit(f"refused: {PROF_TM} sha256 {got} != pinned {PROF_TM_SHA256}")
    spec = importlib.util.spec_from_file_location("speed_p2tm_prof_tm", PROF_TM)
    prof_tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prof_tm)
    Xtr, ytr, Xdv, ydv, C = prof_tm.load("abcd", a.rows, a.rows)
    f = Xtr.shape[1]
    m = 2 * f
    N = a.clauses
    cfg0 = T.TMConfig(n_clauses=N, T=N, s=16.0, epochs=2, batch_size=128, seed=1)
    tm_e = ti.make_tm(C, f, cfg0, init="empty", device="cpu")
    c["make_tm_empty_builds_patch_class"] = type(tm_e) is ti.CoTM and tm_e.opt == want and \
        bool((tm_e.ta == cfg0.n_states).all())
    for chunk_name, chunk in (("unchunked", 50_000_000), ("chunks2", (N // 2) * m)):
        T.CHUNK_ELEMS = chunk
        cfg = T.TMConfig(n_clauses=N, T=N, s=16.0, epochs=2, batch_size=128, seed=1)
        ref = T.CoTM(C, f, cfg, device="cpu")
        ref.fit(Xtr, ytr, Xdv, ydv)
        new = ti.make_tm(C, f, cfg, device="cpu")
        r = {"built_patch_class": type(new) is ti.CoTM, "opts_equal_config": new.opt == want,
             "device_cpu": new.dev.type == "cpu"}
        new.fit(Xtr, ytr, Xdv, ydv)
        r.update(same(ref, new))
        r["no_cuda_event_on_cpu"] = len(new._events) == 0
        res["cases"].append({"case": f"{a.name}/{chunk_name}", "pass": all(r.values()), **r,
                             "history": [h["val_acc"] for h in ref.history]})
    res["all_pass"] = all(c.values()) and all(x["pass"] for x in res["cases"])
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"name": a.name, "all_pass": res["all_pass"], "checks": c,
                      "cases": [(x["case"], x["pass"]) for x in res["cases"]]}), flush=True)
    raise SystemExit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
