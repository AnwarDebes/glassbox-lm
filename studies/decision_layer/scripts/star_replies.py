#!/usr/bin/env python
"""e73 (note logs/plan_e73_replies_star.md): the STAR agent produces its replies from the task's templates.

    star_replies.py --mode mapping                  # train only: derive the slot -> database field mapping
    star_replies.py --mode score --seeds 1 2 3 4 5  # dev only, after the note's freeze

STAR's wizards mostly replied by picking one of the task's reply templates (tasks/<task>/responses.json); some
templates have slots filled from the knowledge base's result. The agent (R21's STAR models, view B on the C4-word
table, CPU decision) picks a label. For a template label its reply is that template's text, with the slots filled
from the database result. The reply is compared with what the wizard really sent. It does not compose sentences:
"produces its replies from the task's templates", never "generates language".

Rules (fixed in the note before any dev score):
- **Template:** the template of label L is the one from the first task in the dialogue's WizardCapabilities whose
  responses.json has L.
- **Slots** ({name} or {name:spec}; the spec is ignored, the value string is inserted as is, since the database
  stores display strings):
  - filled from the latest return_item before the turn whose APIName is that template's task;
  - mapped slot -> Item field by the mapping file (results/explore/star_reply_mapping.json, derived from train with
    --mode mapping);
  - a slot that cannot be filled leaves no reply text (no match).
- **Value string:** str(v) for a scalar; ", ".join(str(x) for x in v) for a list.
- **Mapping (train, mechanical):** for each slotted template label, each slot is mapped to the Item field whose value
  string appears (as a substring, case kept) most often in the wizard's logged text over train picks. Picks with no
  such Item are left out. Ties go to the field name that sorts first.
- **Exact match:** the reply equals the logged text after str.strip(); case and inner spaces kept.
- **The ceiling:** the logged label's own template, filled the same way.
- **Open loop:** memory from the logged history. **Closed loop:** memory from the agent's own presses (R20's
  loop); the text stays as logged.

Data: data/untrusted/star/STAR-master (run with python -I from the study folder). Splits are star_prepare's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

STAR = ROOT / "data" / "untrusted" / "star" / "STAR-master"
MAP = ROOT / "results" / "explore" / "star_reply_mapping.json"  # the literal rule (degenerate; kept, not scored)
MAP_USED = ROOT / "results" / "explore" / "star_reply_mapping_positional.json"  # the derived mapping the note fixes
OUT = ROOT / "results" / "explore" / "star_replies.jsonl"
NOTE_FROZEN = ROOT / "logs" / "plan_e73_replies_star.frozen.json"
MODELS = ROOT / "results" / "confirm" / "R21star" / "tm_mem_c4"
SLOT = re.compile(r"\{([A-Za-z_0-9]+)(?::[^}]*)?\}")


def split_of(dialogue_id) -> str:  # as scripts/star_prepare.py
    h = int(hashlib.sha256(str(dialogue_id).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("dev" if h < 90 else "test")


def responses() -> dict[str, dict[str, str]]:
    return {t.name: json.loads((t / "responses.json").read_text()) for t in sorted((STAR / "tasks").iterdir()) if t.is_dir()}


def value_string(v) -> str:
    return ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)


def template_of(label: str, tasks: list[str], resp: dict) -> tuple[str, str] | None:
    """(task, template text) of label L: the first task of the dialogue that has L."""
    for t in tasks:
        if t in resp and label in resp[t]:
            return t, resp[t][label]
    return None


def walk(split: str):
    """Every labelled wizard action of the split, in star_prepare's order: (DialogueID, k, label, event, tasks,
    latest return_item per APIName at that point); and the free-text wizard replies, counted per dialogue."""
    acts, free = [], Counter()
    for f in sorted((STAR / "dialogues").glob("*.json")):
        d = json.loads(f.read_text())
        if split_of(d["DialogueID"]) != split:
            continue
        tasks = [c["Task"] for c in d["Scenario"]["WizardCapabilities"]]
        items: dict[str, dict] = {}
        k = 0
        for e in d["Events"]:
            if e["Agent"] == "KnowledgeBase" and e["Action"] == "return_item" and isinstance(e.get("Item"), dict):
                items[e.get("APIName") or e["Item"].get("APIName", "?")] = e["Item"]
                continue
            if e["Agent"] != "Wizard":
                continue
            if e["Action"] == "utter" and e.get("Text"):
                free[int(d["DialogueID"])] += 1
            label = e["ActionLabel"] if e["Action"] == "pick_suggestion" else (f"query:{e.get('APIName', '?')}" if e["Action"] == "query" else None)
            if label is None:
                continue
            acts.append({"convo_id": int(d["DialogueID"]), "k": k, "label": label, "kind": e["Action"], "text": e.get("Text", ""),
                         "tasks": tasks, "items": dict(items)})
            k += 1
    return acts, free


def fill(label: str, a: dict, resp: dict, mapping: dict) -> str | None:
    """The reply text of label L at action a, or None (a query label, an unknown label, or a slot that cannot be filled)."""
    if label.startswith("query:"):
        return None
    tt = template_of(label, a["tasks"], resp)
    if tt is None:
        return None
    task, tpl = tt
    slots = SLOT.findall(tpl)
    if not slots:
        return tpl
    item = a["items"].get(task)
    m = mapping.get(f"{task}/{label}", {})
    if item is None:
        return None
    vals = {}
    for s in slots:
        fld = m.get(s)
        if fld is None or fld not in item:
            return None
        vals[s] = value_string(item[fld])
    return SLOT.sub(lambda g: vals[g.group(1)], tpl)


def derive_mapping() -> None:
    if MAP.exists():
        raise SystemExit(f"{MAP} exists; not overwriting")
    resp = responses()
    acts, _ = walk("train")
    cnt: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_pick, n_item = Counter(), Counter()
    for a in acts:
        if a["kind"] != "pick_suggestion":
            continue
        tt = template_of(a["label"], a["tasks"], resp)
        if tt is None or not SLOT.findall(tt[1]):
            continue
        task, tpl = tt
        key = f"{task}/{a['label']}"
        n_pick[key] += 1
        item = a["items"].get(task)
        if item is None:
            continue
        n_item[key] += 1
        for s in SLOT.findall(tpl):
            for fld, v in item.items():
                vs = value_string(v)
                if vs and vs in a["text"]:
                    cnt[key][s][fld] += 1
    mapping, cover = {}, {}
    for key in sorted(n_pick):
        task, label = key.split("/", 1)
        tpl = resp[task][label]
        mapping[key] = {}
        for s in SLOT.findall(tpl):
            c = cnt[key][s]
            if c:
                best = max(c.values())
                mapping[key][s] = sorted(f for f, v in c.items() if v == best)[0]
    # coverage on train with the derived mapping: the filled template equals the logged text
    exact = Counter()
    for a in acts:
        if a["kind"] != "pick_suggestion":
            continue
        tt = template_of(a["label"], a["tasks"], resp)
        if tt is None or not SLOT.findall(tt[1]):
            continue
        key = f"{tt[0]}/{a['label']}"
        r = fill(a["label"], a, resp, mapping)
        exact[key] += int(r is not None and r.strip() == a["text"].strip())
    for key in sorted(n_pick):
        cover[key] = {"train_picks": n_pick[key], "with_item": n_item[key], "filled_exact": exact[key],
                      "share_exact": round(exact[key] / n_pick[key], 4), "slots": mapping[key]}
    out = {"what": "e73 slot -> Item field mapping, derived from STAR train picks (star_replies.py --mode mapping)",
           "rule": "per slotted template label, each slot -> the Item field whose value string appears most often in the logged text; ties to the first field name",
           "mapping": mapping, "coverage_train": cover, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    MAP.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    tot = sum(n_pick.values())
    print(f"{len(mapping)} slotted template labels, {tot} train picks; filled exactly {sum(exact.values())} ({sum(exact.values()) / tot:.3f})")
    for key, c in sorted(cover.items(), key=lambda kv: -kv[1]["train_picks"]):
        print(f"  {key}: {c['train_picks']} picks, {c['with_item']} with an item, exact {c['share_exact']:.3f}, slots {c['slots']}")
    print("sha256", hashlib.sha256(MAP.read_bytes()).hexdigest())


def template_regex(tpl: str):
    """The template as a pattern: literal parts escaped, each slot occurrence a lazy group g<i>; [(slot name) per group]."""
    parts, names, pos = [], [], 0
    for i, m in enumerate(SLOT.finditer(tpl)):
        parts.append(re.escape(tpl[pos : m.start()]))
        parts.append(f"(?P<g{i}>.+?)")
        names.append(m.group(1))
        pos = m.end()
    parts.append(re.escape(tpl[pos:]))
    return re.compile("".join(parts), re.S), names


def derive_mapping_positional() -> None:
    """Added 07:5x UTC after the literal rule proved unable to tell slots apart (every slot of a template got the same field).
    Positional rule: a train pick whose logged text (stripped) fully matches the template's pattern gives, for each slot, the
    text in that slot's own place; each slot is mapped to the Item field whose value string equals that text most often
    (ties to the field name that sorts first). Picks whose text does not match the pattern (edited) do not count."""
    path = MAP.with_name("star_reply_mapping_positional.json")
    if path.exists():
        raise SystemExit(f"{path} exists; not overwriting")
    resp = responses()
    acts, _ = walk("train")
    cnt: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_pick, n_item, n_match = Counter(), Counter(), Counter()
    for a in acts:
        if a["kind"] != "pick_suggestion":
            continue
        tt = template_of(a["label"], a["tasks"], resp)
        if tt is None or not SLOT.findall(tt[1]):
            continue
        task, tpl = tt
        key = f"{task}/{a['label']}"
        n_pick[key] += 1
        item = a["items"].get(task)
        if item is None:
            continue
        n_item[key] += 1
        rx, names = template_regex(tpl)
        mt = rx.fullmatch(a["text"].strip())
        if mt is None:
            continue
        n_match[key] += 1
        for gi, s in enumerate(names):
            seg = mt.group(f"g{gi}")
            for fld, v in item.items():
                if value_string(v) == seg:
                    cnt[key][s][fld] += 1
    mapping, cover = {}, {}
    for key in sorted(n_pick):
        task, label = key.split("/", 1)
        mapping[key] = {}
        for s in dict.fromkeys(SLOT.findall(resp[task][label])):
            c = cnt[key][s]
            if c:
                best = max(c.values())
                mapping[key][s] = sorted(f for f, v in c.items() if v == best)[0]
    exact = Counter()
    for a in acts:
        if a["kind"] != "pick_suggestion":
            continue
        tt = template_of(a["label"], a["tasks"], resp)
        if tt is None or not SLOT.findall(tt[1]):
            continue
        key = f"{tt[0]}/{a['label']}"
        r = fill(a["label"], a, resp, mapping)
        exact[key] += int(r is not None and r.strip() == a["text"].strip())
    for key in sorted(n_pick):
        cover[key] = {"train_picks": n_pick[key], "with_item": n_item[key], "text_matches_pattern": n_match[key],
                      "filled_exact": exact[key], "share_exact": round(exact[key] / n_pick[key], 4), "slots": mapping[key]}
    out = {"what": "e73 slot -> Item field mapping, positional rule, derived from STAR train picks (star_replies.py --mode mapping-positional)",
           "rule": "per slotted template label: picks whose logged text fully matches the template pattern give each slot's own text; "
                   "each slot -> the Item field whose value string equals it most often; ties to the first field name",
           "mapping": mapping, "coverage_train": cover, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    tot = sum(n_pick.values())
    print(f"{len(mapping)} slotted template labels, {tot} train picks; filled exactly {sum(exact.values())} ({sum(exact.values()) / tot:.3f})")
    for key, c in sorted(cover.items(), key=lambda kv: -kv[1]["train_picks"]):
        print(f"  {key}: {c['train_picks']} picks, {c['with_item']} with an item, {c['text_matches_pattern']} match the pattern, "
              f"exact {c['share_exact']:.3f}, slots {c['slots']}")
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


def score(seeds: list[int], tag: str) -> None:
    if not NOTE_FROZEN.exists():
        raise SystemExit("dev scoring starts only after logs/plan_e73_replies_star.md is frozen")
    frozen = json.loads(NOTE_FROZEN.read_text())
    map_sha = hashlib.sha256(MAP_USED.read_bytes()).hexdigest()
    from dl import count_text, cpu_tm_fast  # noqa: E402

    meta = count_text.patch("c4words-S1000-v1")
    from abcd_closed_loop import closed_loop  # noqa: E402
    from dl.abcd import ABCDView, load_examples  # noqa: E402

    resp = responses()
    mapping = json.loads(MAP_USED.read_text())["mapping"]
    tr, blist = load_examples("train", "star")
    dv, _ = load_examples("dev", "star")
    acts, free = walk("dev")
    assert len(acts) == len(dv) and all(a["convo_id"] == r["convo_id"] and a["k"] == r["k"] and a["label"] == r["button"]
                                       for a, r in zip(acts, dv)), "raw dev actions do not align with the dev examples"
    E = np.load(ROOT / "data" / "emb" / "star__none.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
    names = json.loads((MODELS / "bit_names.json").read_text())
    assert view.names == names
    Xdv = view.transform(dv, em["dev"])
    n_mem = 2 * len(blist) + 6
    picks = [i for i, a in enumerate(acts) if a["kind"] == "pick_suggestion"]
    slotted = {i for i in picks if (tt := template_of(acts[i]["label"], acts[i]["tasks"], resp)) and SLOT.findall(tt[1])}
    n_free = sum(free.values())
    ceil = {i: (r := fill(acts[i]["label"], acts[i], resp, mapping)) is not None and r.strip() == acts[i]["text"].strip() for i in picks}
    base = {"tag": tag, "kind": "replies", "table": meta["table"], "mapping_sha256": map_sha, "note_frozen_utc": frozen["frozen_utc"],
            "n_dev_actions": len(acts), "n_picks": len(picks), "n_slotted_picks": len(slotted), "n_free_text_replies": n_free,
            "free_text_share_of_replies": round(n_free / (len(picks) + n_free), 4),
            "ceiling_exact": sum(ceil.values()), "ceiling_exact_plain": sum(ceil[i] for i in picks if i not in slotted),
            "ceiling_exact_slotted": sum(ceil[i] for i in slotted)}
    print(json.dumps(base), flush=True)
    for seed in seeds:
        z = np.load(MODELS / f"model_seed{seed}.npz")
        pack = cpu_tm_fast.pack(z["ta"], z["w"])
        fn = lambda Z: cpu_tm_fast.class_sums(pack, Z)  # noqa: E731
        preds = {"open": fn(Xdv).argmax(1), "closed": closed_loop(fn, Xdv, dv, blist, n_mem)}
        for loop, p in preds.items():
            lab = [blist[int(c)] for c in p]
            ok_label = [lab[i] == acts[i]["label"] for i in range(len(acts))]
            ex = {i: (r := fill(lab[i], acts[i], resp, mapping)) is not None and r.strip() == acts[i]["text"].strip() for i in picks}
            q = [i for i, a in enumerate(acts) if a["kind"] == "query"]
            rec = {**base, "seed": seed, "loop": loop,
                   "label_acc_all": round(float(np.mean(ok_label)), 4),
                   "label_acc_picks": round(float(np.mean([ok_label[i] for i in picks])), 4),
                   "label_acc_queries": round(float(np.mean([ok_label[i] for i in q])), 4) if q else None,
                   "agent_exact": sum(ex.values()), "agent_exact_plain": sum(ex[i] for i in picks if i not in slotted),
                   "agent_exact_slotted": sum(ex[i] for i in slotted),
                   "agent_exact_share_of_picks": round(sum(ex.values()) / len(picks), 4),
                   "agent_exact_share_of_all_replies": round(sum(ex.values()) / (len(picks) + n_free), 4),
                   "agent_over_ceiling": round(sum(ex.values()) / max(sum(ceil.values()), 1), 4),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("seed", "loop", "label_acc_picks", "agent_exact_share_of_picks", "agent_over_ceiling",
                                                 "agent_exact_share_of_all_replies")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["mapping", "mapping-positional", "score"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--tag", default="e73")
    args = ap.parse_args()
    if args.mode == "mapping":
        derive_mapping()
    elif args.mode == "mapping-positional":
        derive_mapping_positional()
    else:
        score(args.seeds, args.tag)


if __name__ == "__main__":
    main()
