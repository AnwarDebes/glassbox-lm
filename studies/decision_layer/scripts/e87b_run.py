#!/usr/bin/env python
"""e87 Part B (plan logs/plan_e87_receipts.md; dated note logs/plan_e87_note_2026-10-25.md): run a study script with
the exact literal budget kept inside tm.py's clause-chunked update, and write a record of the run. Added 2026-10-25,
before any Part B number.

    e87b_run.py --neg-sampling hardest|random --run-info FILE [--chunk-elems N] [--plain-select] -- SCRIPT [ARGS...]

tm.py, tm_chunkbudget.py, knob_run.py and every run script stay byte for byte. This wrapper patches dl.tm.CoTM in its
own process, then runs SCRIPT there (runpy, as __main__), as knob_run.py does:
- CoTM.__init__ sets neg_sampling on the TMConfig that the script built, before the original __init__ runs, so the
  row's cfg (asdict(cfg) in abcd_explore.py) records it. After it, the machine must hold an exact literal budget
  (max_literals > 0, budget_mode "exact") and no negated cap, or the run stops. Then dl.tm_chunkbudget.enable binds
  the clause-chunked update that keeps the exact budget (tm.py's own chunked update raises NotImplementedError for
  it). make_tm's trim start (abcd_explore.py --init trim) only edits ta afterwards and keeps the binding.
- In that chunked update, select_k (the rank selection of the crossings a clause may keep) is replaced by
  select_k_skip0, which returns dl.tm.select_k's result for every row but does not sort the rows that keep nothing
  (k == 0: a clause already at the budget). The random keys are drawn as before, so training is bit for bit the same
  (tests/test_e87b_run.py: the function on random inputs with ties, and two trainings compared array by array; the
  dry run repeats one tiny job with --plain-select, the unchanged select_k, and compares the two models on the GPU).
  With the trim start every clause starts at the budget, so most sorts are skipped. Only dl.tm_chunkbudget's own
  name select_k is rebound; tm.py's unchunked update keeps dl.tm.select_k.
- Every call of that chunked update is counted. One call is one training step (one minibatch), so a full run records
  epochs x ceil(train rows / batch size) calls; a machine that never reached the chunked path records 0.
- --chunk-elems N sets dl.tm.CHUNK_ELEMS for this process only (the dry run uses it so that a tiny machine runs the
  chunked path too). Without it, tm.py's value (50,000,000 clause-literal pairs) holds.
- --run-info FILE is refused if it exists before SCRIPT starts, and so is the model file that SCRIPT will save
  (results/explore/abcd_models/<tag>_seed<seed>.npz, from SCRIPT's --tag and --seed; --save is required). So a second
  try of a job can neither overwrite the first try's model nor its record. FILE is written only after SCRIPT ends
  without an error: the knobs, CHUNK_ELEMS, each machine's sizes and its count of chunked budget updates, the argv,
  the model file and its SHA-256, and the start and end times.
Tested by tests/test_e87b_run.py (CPU: a small machine with the chunked path forced keeps the budget and learns;
select_k_skip0 against dl.tm.select_k; two trainings equal bit for bit; the run record; the refusals).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import torch  # noqa: E402

import dl.tm_chunkbudget as cb  # noqa: E402
from dl import tm as dltm  # noqa: E402
from dl.tm_chunkbudget import enable  # noqa: E402

NEG = ("hardest", "random")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arg(rest: list[str], flag: str) -> str | None:
    """The value after the last occurrence of flag in rest (argparse keeps the last one too), or None."""
    idx = [i for i, a in enumerate(rest) if a == flag]
    return rest[idx[-1] + 1] if idx and idx[-1] + 1 < len(rest) else None


def select_k_skip0(keys: torch.Tensor, cr: torch.Tensor, k: torch.Tensor, mode: str = "threshold") -> torch.Tensor:
    """dl.tm.select_k's result for every row; in "rank" mode the rows with k == 0 (which keep nothing) are not sorted."""
    if mode != "rank":
        return dltm.select_k(keys, cr, k, mode)
    out = torch.zeros_like(cr)
    pos = (k > 0).nonzero().squeeze(1)
    if pos.numel():
        out[pos] = dltm.select_k(keys[pos], cr[pos], k[pos], mode)
    return out


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str], Path]:
    if "--" not in argv:
        raise SystemExit("e87b_run: put -- before the script to run")
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="e87b_run.py")
    ap.add_argument("--neg-sampling", required=True, choices=NEG)
    ap.add_argument("--run-info", required=True, type=Path)
    ap.add_argument("--chunk-elems", type=int)
    ap.add_argument("--plain-select", action="store_true", help="keep dl.tm.select_k in the chunked update (dry-run identity check)")
    args = ap.parse_args(argv[:i])
    rest = argv[i + 1 :]
    if not rest or not rest[0].endswith(".py"):
        raise SystemExit("e87b_run: no script after --")
    if args.chunk_elems is not None and args.chunk_elems < 1:
        raise SystemExit("e87b_run: --chunk-elems needs a positive number")
    tag, seed = arg(rest, "--tag"), arg(rest, "--seed")
    if "--save" not in rest or not tag or seed is None:
        raise SystemExit("e87b_run: the script needs --save, --tag and --seed (the model file is part of the record)")
    model = ROOT / "results" / "explore" / "abcd_models" / f"{tag}_seed{int(seed)}.npz"
    return args, rest, model


def install(neg: str, chunk_elems: int | None, state: dict, plain: bool = False) -> None:
    """Patch dl.tm.CoTM in this process: neg_sampling, the exact-budget check, tm_chunkbudget.enable and the count;
    unless plain, tm_chunkbudget's select_k becomes select_k_skip0."""
    if chunk_elems is not None:
        dltm.CHUNK_ELEMS = chunk_elems
    cb.select_k = dltm.select_k if plain else select_k_skip0
    orig_init = dltm.CoTM.__init__

    def init(self, n_classes, n_features, cfg, device="cuda"):
        cfg.neg_sampling = neg
        orig_init(self, n_classes, n_features, cfg, device=device)
        if cfg.max_literals <= 0 or cfg.budget_mode != "exact" or cfg.max_negated > 0:
            raise SystemExit(f"e87b_run: needs an exact literal budget and no negated cap (max_literals {cfg.max_literals}, "
                             f"budget_mode {cfg.budget_mode!r}, max_negated {cfg.max_negated})")
        enable(self)
        inner = self._automata_chunked
        rec = {"n_clauses": int(cfg.n_clauses), "literals": int(self.m), "pairs": int(cfg.n_clauses) * int(self.m),
               "chunk_elems": int(dltm.CHUNK_ELEMS), "chunked": int(cfg.n_clauses) * int(self.m) > dltm.CHUNK_ELEMS,
               "clauses_per_chunk": max(1, dltm.CHUNK_ELEMS // int(self.m)), "max_literals": int(cfg.max_literals),
               "budget_mode": cfg.budget_mode, "budget_select": cfg.budget_select, "chunked_budget_steps": 0}
        state["machines"].append(rec)

        def counted(*a):
            rec["chunked_budget_steps"] += 1
            return inner(*a)

        self._automata_chunked = counted

    dltm.CoTM.__init__ = init


def main() -> None:
    args, rest, model = parse(sys.argv[1:])
    for p in (args.run_info, model):
        if p.exists():
            raise SystemExit(f"e87b_run: {p} exists; not overwriting")
    state: dict = {"machines": []}
    install(args.neg_sampling, args.chunk_elems, state, plain=args.plain_select)
    t0, started = time.time(), utc()
    print(f"e87b_run: neg_sampling {args.neg_sampling}; exact budget in the chunked update (dl.tm_chunkbudget.enable, select "
          f"{'plain' if args.plain_select else 'skip0'}); "
          f"CHUNK_ELEMS {dltm.CHUNK_ELEMS}; run-info {args.run_info}; running {' '.join(rest)}", flush=True)
    sys.argv = list(rest)
    try:
        runpy.run_path(str(Path(rest[0]).resolve()), run_name="__main__")
    except SystemExit as e:  # a script that ends with sys.exit(0) has still finished; any other exit stops here
        if e.code not in (None, 0):
            raise
    if not state["machines"]:
        raise SystemExit("e87b_run: the script built no TM; no run-info")
    if not model.exists():
        raise SystemExit(f"e87b_run: the script wrote no model file {model}; no run-info")
    info = {"e87b_run": utc(), "started": started, "seconds": round(time.time() - t0, 1),
            "knobs": {"neg_sampling": args.neg_sampling, "chunk_budget": "dl.tm_chunkbudget.enable",
                      "select": "plain" if args.plain_select else "skip0"},
            "chunk_elems": int(dltm.CHUNK_ELEMS), "machines": state["machines"], "argv": rest,
            "model": str(model.relative_to(ROOT)), "model_sha256": sha256(model)}
    args.run_info.parent.mkdir(parents=True, exist_ok=True)
    args.run_info.write_text(json.dumps(info, indent=1))
    m = state["machines"][-1]
    print(f"e87b_run: wrote {args.run_info} ({m['chunked_budget_steps']} chunked budget updates; model {info['model_sha256'][:8]})", flush=True)


if __name__ == "__main__":
    main()
