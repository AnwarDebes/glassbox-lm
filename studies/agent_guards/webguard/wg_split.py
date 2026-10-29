#!/usr/bin/env python
"""WebGuard veto test, the split (option A, 2026-10-20 22:29:55Z; fixed BEFORE any feature).

    python -I wg_split.py

Reads glassbox-lm/studies/agent_guards/data_untrusted/webguard/4135c4a9/monitor_factory_train.json (the image-version train file)
only for each row's site (the website folder in its image path: .../downloads/<site>/...) and its label, and checks
that monitor_factory_train_a11y.json has the same 1,982 rows in the same order (the same label in every row; the a11y
file has no URL or image field, so the row index is the link). The 115 sites go through a numpy default_rng(20261021)
permutation; the first 80 are train, the next 17 dev and the last 18 test (70/15/15 by site, rounded). Every row
follows its site. Writes webguard/wg_split.json (the site lists, the row indices per part, the counts, the sha256 of the
inputs) and prints the counts. Refuses to overwrite. Derived from CC BY-NC 4.0 data: research only, never committed.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parent / "data_untrusted" / "webguard" / "4135c4a9"
N_TRAIN, N_DEV, N_TEST = 80, 17, 18


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    out = HERE / "wg_split.json"
    assert not out.exists(), f"{out} exists"
    img = json.loads((D / "monitor_factory_train.json").read_text())
    a11y = json.loads((D / "monitor_factory_train_a11y.json").read_text())
    assert len(img) == len(a11y) == 1982
    assert all(p["messages"][1]["content"] == q["messages"][1]["content"] for p, q in zip(img, a11y)), "rows not aligned"
    site = [re.search(r"/downloads/([^/]+)/", d["images"][0]).group(1) for d in img]
    label = [json.loads(d["messages"][1]["content"])["risky_level"] for d in img]
    sites = sorted(set(site))
    assert len(sites) == N_TRAIN + N_DEV + N_TEST == 115
    perm = np.random.default_rng(20261021).permutation(len(sites))
    parts = {"train": sorted(sites[i] for i in perm[:N_TRAIN]), "dev": sorted(sites[i] for i in perm[N_TRAIN : N_TRAIN + N_DEV]),
             "test": sorted(sites[i] for i in perm[N_TRAIN + N_DEV :])}
    of = {s: p for p, ss in parts.items() for s in ss}
    rows = {p: [i for i, s in enumerate(site) if of[s] == p] for p in parts}
    counts = {p: {"sites": len(parts[p]), "rows": len(rows[p]), "labels": dict(collections.Counter(label[i] for i in rows[p]))} for p in parts}
    res = {"seed": 20261021, "rule": "numpy default_rng(20261021).permutation over the sorted 115 site names; 80 / 17 / 18",
           "inputs": {"monitor_factory_train.json": sha16(D / "monitor_factory_train.json"),
                      "monitor_factory_train_a11y.json": sha16(D / "monitor_factory_train_a11y.json")},
           "sites": parts, "rows": rows, "counts": counts,
           "site_lists_sha256": hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16],
           "wg_split.py": sha16(Path(__file__).resolve())}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"counts": counts, "site_lists_sha256": res["site_lists_sha256"]}))


if __name__ == "__main__":
    main()
