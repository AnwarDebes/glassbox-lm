"""The E2 freeze step (registration rev3, section 0; the decision 4): between E2-0 and E2-1, after the
registration was completed from E2-0's open files, write the WRITE-ONCE record runs/E2_frozen.json. Standard library
only, CPU, no server:

    python -I -B scripts/freeze_e2.py --registration <path of the completed registration>

It refuses unless:
- E2-0 ended (its store-chain record exists) and no later stage has run (the chain holds E2-0's record only) and no
  stop is logged;
- runs/E2-0_gate.json says go (and not from a fake run); runs/budget.json exists and does not say stop;
  runs/E2-0_summary.json (the job file's copy of E2-0's open summary) exists;
- the registration file's first line does not say NOT FROZEN, and its text holds the full sha256 of
  runs/server_identity_ref.json, runs/E2-0_gate.json, runs/budget.json and runs/E2-0_summary.json (section 13 at the
  freeze) and of scripts/e2_check.py (Appendix C);
- rev4 (round-3 m-14): the code (e2_check.code_hashes) equals the code E2-0 ran, as its open driver/meta.json records
  it (code_sha256); rev5 (round-5 R5-3): with no store-chain record of E2-0 this check refuses explicitly too;
- rev4 (round-3 m-15): every earlier campaign moved aside (a runs.aside.* folder beside runs/) is named in the
  registration (section 15 lists each with its reason, open gate and budget);
- runs/E2_frozen.json does not exist yet (it is opened with mode "x": never overwritten).
The record: the registration's path (relative to ..), its sha256, the UTC time, the sha256 of the
identity record, the gate, the budget plan and its sealed detail, the summary copy, and E2-0's chain record.
e2_check.py --begin requires it for every stage after E2-0 (the registration must still match; the freeze time must
precede the stage start; the identity record, the gate, the budget plan and the summary copy must be unchanged).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "scripts"))

import e2_check as EC


def problems(reg: Path) -> list[str]:
    bad = []
    chain = EC.chain_records()
    if not chain or chain[0].get("stage") != "smoke":
        bad.append("E2-0 has not ended (no store-chain record of stage smoke)")
    elif len(chain) != 1:
        bad.append("a stage after E2-0 already ran: the freeze must come between E2-0 and E2-1")
    if EC.stop_log():
        bad.append("runs/stopped_stages.jsonl holds a stop (E2-0 reruns on a fresh campaign)")
    gate = json.loads(EC.SMOKE_GATE.read_text()) if EC.SMOKE_GATE.exists() else None
    if gate is None or gate.get("go") is not True or gate.get("fake") is not False:
        bad.append("runs/E2-0_gate.json is missing, not go, or from a fake run")
    bud = json.loads(EC.BUDGET.read_text()) if EC.BUDGET.exists() else None
    if bud is None or bud.get("stop") is not False:
        bad.append("runs/budget.json is missing or says stop (a larger cap or no E2 is chosen)")
    if not EC.IDENTITY_REF.exists():
        bad.append("runs/server_identity_ref.json is missing")
    if not EC.E2_0_SUMMARY.exists():
        bad.append("runs/E2-0_summary.json is missing (the job file copies E2-0's open summary there)")
    if not reg.exists():
        return bad + [f"no registration file {reg}"]
    text = reg.read_text()
    if "NOT FROZEN" in text.splitlines()[0]:
        bad.append("the registration's first line still says NOT FROZEN")
    for p in (EC.IDENTITY_REF, EC.SMOKE_GATE, EC.BUDGET, EC.E2_0_SUMMARY, E2 / "scripts" / "e2_check.py"):
        if p.exists() and EC.sha256_file(p) not in text:
            bad.append(f"the registration does not hold the sha256 of {p.name}")
    if chain and chain[0].get("stage") == "smoke":  # rev4 (round-3 m-14): the code must be the code E2-0 ran
        meta = Path(chain[0].get("run_dir") or "/nonexistent") / "driver" / "meta.json"
        if not meta.exists():
            bad.append(f"E2-0's open driver/meta.json is missing ({meta}): its code record cannot be checked")
        else:
            ran = json.loads(meta.read_text()).get("code_sha256") or {}
            now = EC.code_hashes()
            diff = sorted(k for k in set(ran) | set(now) if ran.get(k) != now.get(k))
            if diff:
                bad.append(f"the code differs from the code E2-0 ran (its driver/meta.json code_sha256): {diff[:4]}")
    else:  # rev5 (round-5 R5-3): never skipped silently (the first check above refuses this case as well)
        bad.append("the code cannot be compared with the code E2-0 ran: no store-chain record of stage smoke")
    # rev4 (round-3 m-15): every earlier campaign (runs.aside.*) is named in the registration (section 15)
    aside = sorted(p.name for p in EC.RUNS.parent.glob(EC.RUNS.name + ".aside.*"))
    unnamed = [n for n in aside if n not in text]
    if unnamed:
        bad.append(f"earlier campaigns not named in the registration (section 15): {', '.join(unnamed[:4])}")
    if EC.FROZEN.exists():
        bad.append("runs/E2_frozen.json exists (write-once: never overwritten)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = Path(a.registration).resolve()
    bad = problems(reg)
    for b in bad:
        print("STOP:", b)
    if bad:
        sys.exit(1)
    detail = EC.RUNS / "sealed" / "budget_detail.json"
    rec = {"registration": str(reg.relative_to(EC.ROOT)), "registration_sha256": EC.sha256_file(reg),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "identity_ref_sha256": EC.sha256_file(EC.IDENTITY_REF), "gate_sha256": EC.sha256_file(EC.SMOKE_GATE),
           "budget_sha256": EC.sha256_file(EC.BUDGET),
           "budget_detail_sha256": EC.sha256_file(detail) if detail.exists() else None,
           "e2_0_summary_sha256": EC.sha256_file(EC.E2_0_SUMMARY),
           "e2_0_chain_record": EC.chain_records()[0],
           "rule": "write-once; e2_check.py --begin requires it for every stage after E2-0"}
    with EC.FROZEN.open("x") as f:  # never overwritten
        f.write(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {EC.FROZEN} sha256 {EC.sha256_file(EC.FROZEN)} (registration sha256 {rec['registration_sha256']}, "
          f"{rec['utc']})")


if __name__ == "__main__":
    main()
