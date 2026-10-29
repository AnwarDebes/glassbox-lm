"""The frozen guard descriptions of the guard-aware A3 optimizer, one per arm and suite, with their sha256 (rev2,
review item 7). AgentDojo venv:  ../agentdojo/.venv/bin/python -I -B scripts/write_description_pins.py OUT.json"""
import hashlib
import json
import sys
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))


def main() -> None:
    from e2guard import attacks_a3 as A3
    from e2guard.facts import ToolTable

    out_path = Path(sys.argv[1])
    assert out_path.suffix == ".json", "the output must be a .json file"
    t = ToolTable()
    out = {f"{a}|{s}": {"sha256": A3.description_sha256(a, s, t), "text": A3.guard_description(a, s, t)}
           for a in A3.ARMS_WITH_DESCRIPTION for s in ("banking", "slack", "travel", "workspace")}
    out["rev1_text_all_arms"] = {"sha256": hashlib.sha256(A3.GUARD_INFO_REV1.encode()).hexdigest(),
                                 "text": A3.GUARD_INFO_REV1}
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["sha256"] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
