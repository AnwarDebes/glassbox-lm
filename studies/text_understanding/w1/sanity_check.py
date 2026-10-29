#!/usr/bin/env python
"""W1 neighbour sanity check (rule and word lists fixed in LOG.md before the runs, rule of 01:1xZ).

    sanity_check.py TABLE_NAME [TABLE_NAME ...]

For each table, the latest probe row in results_w1_probe.jsonl: a probe word passes when at least one of its 6
nearest neighbours is in its fixed list; the table passes when at least 4 of the 10 probe words pass. Prints one
JSON line per table and appends it to results_w1_checks.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LISTS = {
    "balance": "account accounts bank banking payment payments deposit money funds credit debit statement amount balances cash savings",
    "transfer": "transfers transferred transferring send sending payment deposit wire move moving account bank funds money",
    "alarm": "alarms alert alerts clock timer wake reminder siren security warning notification sensor detector smoke",
    "weather": "forecast rain temperature temperatures climate storm storms snow sunny wind winds cloudy conditions humidity",
    "french": "france paris spanish italian german english european language dutch belgian quebec portuguese",
    "card": "cards credit debit visa mastercard payment wallet pin bank gift account",
    "music": "songs song musical band bands album albums jazz rock pop guitar concert concerts musicians musician sound radio playlist singer",
    "flight": "flights airline airlines aircraft airport airports plane planes fly flying travel trip booking departure pilot jet",
    "cancel": "cancellation cancelled canceled cancelling refund refunds terminate delete stop unsubscribe postpone booking reservation order subscription",
    "email": "emails mail inbox message messages contact newsletter address gmail sms text phone outlook send",
}


def main() -> None:
    rows = [json.loads(x) for x in (HERE / "results_w1_probe.jsonl").read_text().splitlines() if x.strip()]
    for name in sys.argv[1:]:
        r = [x for x in rows if x["table"] == name]
        assert r, f"no probe row for {name}"
        r = r[-1]
        per = {}
        for w, allowed in LISTS.items():
            nb = r["neighbours"].get(w)
            per[w] = None if nb is None else [n for n in nb[:6] if n in set(allowed.split())]
        n_pass = sum(1 for v in per.values() if v)
        rec = {"check": "W1 neighbour sanity (rule of LOG.md 01:1xZ)", "table": name, "probe_slice": r["slice_for_vocab"],
               "probe_mrr": r["mrr"], "words_passing": n_pass, "of": len(LISTS), "passed": n_pass >= 4, "hits": per}
        print(json.dumps(rec))
        with open(HERE / "results_w1_checks.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
