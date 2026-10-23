#!/usr/bin/env python
"""e77 (scope logs/plan_e77_third_dataset_scope.md): MultiWOZ 2.2 next system action, in the format of the ABCD and STAR
examples (src/dl/abcd.py), so the same views and scripts apply.

    .venv-dl/bin/python -I scripts/multiwoz_prepare.py               # train and dev only; prints the counts
    .venv-dl/bin/python -I scripts/multiwoz_prepare.py --with-test   # only at R24, after its freeze

Data: data/untrusted/multiwoz22 (budzianowski/multiwoz at commit fe0c8e65, data/MultiWOZ_2.2; MIT; SOURCE.txt,
MANIFEST.sha256). The official train / dev / test folders are the splits. Read as JSON only.
- One example per SYSTEM turn that has a dialogue-act annotation. Turns with none are left out and counted.
- **Label:** the sorted set of the turn's act names (domain-act, no slots or values), joined by "+". A combination with
  fewer than 20 train occurrences becomes "OTHER". The rule and the threshold were fixed in the scope before any count.
- **The text fields,** as in star_prepare.py:
  - cust: the customer's utterances so far (stripped, lower-cased), joined by " | ";
  - recent: the last 3 utterances with speaker tags "user:" and "system:";
  - full: empty.
- **Memory:** prev holds the labels of the system's earlier turns in the dialogue, in order. These are the mapped
  labels, OTHER included: the analogue of STAR's wizard labels and ABCD's buttons.
- **subflow / flow:** the dialogue's services, sorted and joined by "+". They are used only by the intent centres of
  the bge-large view, and only on train.
- convo_id: the dialogue's index among the split's sorted dialogue ids. ast_count_bert / roberta = 1, as STAR's.
- The test folder is not read unless --with-test is given; dialog_acts.json entries are read only for the requested
  splits' dialogues.
Writes data/multiwoz/examples_<split>.json.gz and data/multiwoz/labels.json; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "untrusted" / "multiwoz22"
OUT = ROOT / "data" / "multiwoz"
MIN_TRAIN = 20
RECENT = 3


def dialogues(split: str) -> list[dict]:
    out = []
    for p in sorted((SRC / split).glob("dialogues_*.json")):
        out.extend(json.loads(p.read_text()))
    return sorted(out, key=lambda d: d["dialogue_id"])


def raw_examples(split: str, acts: dict) -> tuple[list[dict], Counter]:
    rows, skipped = [], Counter()
    for ci, d in enumerate(dialogues(split)):
        da = acts.get(d["dialogue_id"], {})
        utts, prev_raw = [], []
        services = "+".join(sorted(d.get("services", []))) or "none"
        for t in d["turns"]:
            sp = "user" if t["speaker"] == "USER" else "system"
            if sp == "system":
                a = (da.get(t["turn_id"]) or {}).get("dialog_act") or {}
                if not a:
                    skipped["system turn with no act"] += 1
                else:
                    raw = "+".join(sorted(a.keys()))
                    cust = " | ".join(x for s, x in utts if s == "user")
                    recent = " ".join(f"{s}: {x}" for s, x in utts[-RECENT:])
                    rows.append({"convo_id": ci, "dialogue_id": d["dialogue_id"], "k": len(prev_raw), "flow": services, "subflow": services,
                                 "raw_label": raw, "prev_raw": list(prev_raw), "cust": cust, "recent": recent, "full": "",
                                 "ast_count_bert": 1, "ast_count_roberta": 1})
                    prev_raw.append(raw)
            utts.append((sp, t["utterance"].strip().lower()))
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-test", action="store_true")
    args = ap.parse_args()
    write = ["test"] if args.with_test else ["train", "dev"]  # the label set always comes from train
    splits = ["train"] + [s for s in write if s != "train"] + (["dev"] if args.with_test else [])
    for s in write:
        if (OUT / f"examples_{s}.json.gz").exists():
            raise SystemExit(f"{OUT / f'examples_{s}.json.gz'} exists; not overwriting")
    acts_all = json.loads((SRC / "dialog_acts.json").read_text())
    ids = {s: {d["dialogue_id"] for d in dialogues(s)} for s in splits}
    acts = {k: v for k, v in acts_all.items() if any(k in ids[s] for s in splits)}  # only the requested splits' entries
    del acts_all
    raw = {s: raw_examples(s, acts) for s in splits}
    train_counts = Counter(r["raw_label"] for r in raw["train"][0])
    keep = sorted(lb for lb, n in train_counts.items() if n >= MIN_TRAIN)
    buttons = keep + ["OTHER"]
    lab = lambda x: x if x in keep else "OTHER"  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"what": "e77 MultiWOZ 2.2 next system action (multiwoz_prepare.py)", "min_train": MIN_TRAIN, "n_labels": len(buttons),
              "n_raw_combinations_train": len(train_counts), "splits": {}}
    for s in write:
        rows, skipped = raw[s]
        out = []
        for r in rows:
            y = lab(r["raw_label"])
            out.append({**{k: v for k, v in r.items() if k not in ("raw_label", "prev_raw")}, "button": y, "y": buttons.index(y),
                        "prev": [lab(p) for p in r["prev_raw"]], "raw_label": r["raw_label"]})
        p = OUT / f"examples_{s}.json.gz"
        with gzip.open(p, "wt") as f:
            json.dump({"rows": out, "buttons": buttons}, f)
        c = Counter(r["button"] for r in out)
        report["splits"][s] = {"dialogues": len(ids[s]), "examples": len(out), "system_turns_without_act": skipped["system turn with no act"],
                               "share_without_act": round(skipped["system turn with no act"] / (len(out) + skipped["system turn with no act"]), 4),
                               "share_OTHER": round(c["OTHER"] / len(out), 4), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    tr_lab = Counter(lab(r["raw_label"]) for r in raw["train"][0])
    majority = tr_lab.most_common(1)[0][0]
    report["majority_label_train"] = majority
    report["majority_accuracy_dev"] = round(sum(lab(r["raw_label"]) == majority for r in raw["dev"][0]) / len(raw["dev"][0]), 4)
    report["train_counts"] = dict(tr_lab.most_common())
    rp = OUT / ("labels_test.json" if args.with_test else "labels.json")
    if rp.exists():
        raise SystemExit(f"{rp} exists; not overwriting")
    rp.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "train_counts"}, indent=1))
    print("top 25 labels (train):", tr_lab.most_common(25))
    print("labels with 20-50 train examples:", sum(1 for v in tr_lab.values() if 20 <= v < 50), "of", len(tr_lab))


if __name__ == "__main__":
    main()
