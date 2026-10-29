#!/usr/bin/env python
"""WebGuard veto test, the bits (webguard/PLAN_wg_veto.md "Bits"; CPU only; every fit uses the TRAIN part only).

    python -I wg_feats.py

Rows: monitor_factory_train_a11y.json (the accessibility tree and the target node) joined by row index with
monitor_factory_train.json (URL, tagHead); parts from wg_split.json (fixed before this script existed). Words: the C4
counting table ppw_S1000 (glassbox-lm/studies/text_understanding/w1/tables/ppw_S1000_d500_e0.5.npz, no neural network at any stage) and
its tokenizer (wordtab_feats.glove_tokens); a word without a live row is dropped. Concepts: spherical k-means, K 256,
25 iterations, CPU generator seeded 0, over the table rows of the TRAIN part's words (node, title, URL and tree words).
Blocks:
- node: existsT of the node's name words (per concept the max cosine; Thermo(8) fit on the train rows), one-hot role and
  one-hot tagHead (values seen >= 5 times in train, else "other"), and the veto bit (the fixed word list below);
- page: existsT of the page title and the URL's words (Thermo(8) fit on train), and count bits over the tree's names
  (per concept, the number of tree words whose nearest concept it is: >= 1, >= 4, >= 16).
Writes webguard/feats/wg_bits.npz (B uint8 [rows, bits], y with SAFE 0 / LOW 1 / HIGH 2, part 0 train / 1 dev / 2 test,
site) and wg_bits.json (bit names, block ranges, the veto bit, the domain groups for the solver, counts, hashes).
Labels are read for y only; no score is computed here. Refuses to overwrite. Derived from CC BY-NC 4.0 data: research
only, never committed.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/agent_guards/webguard
D = HERE.parent / "data_untrusted" / "webguard" / "4135c4a9"
EMB = Path(f"{_REPO_ROOT}/studies/text_understanding")
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
sys.path.insert(0, str(EMB / "w1"))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import wordtab_feats as WF  # noqa: E402
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402

K = 256
LABELS = ["SAFE", "LOW", "HIGH"]
VETO_WORDS = {"buy", "purchase", "checkout", "pay", "payment", "order", "donate", "subscribe", "bet", "wager", "loan",
              "register", "signup", "publish", "transfer"}
VETO_PAIRS = {("sign", "up"), ("place", "order"), ("create", "account")}
COUNT_LEVELS = (1, 4, 16)
MIN_ONEHOT = 5
URL_DROP = {"http", "https"}


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def parse_a11y(text: str) -> dict:
    i = text.rfind("The Accessibility Tree of the webpage is ")
    j = text.rfind("\nThe proposed action is")
    assert i >= 0 and j > i, "no tree or no action"
    tree = text[i + len("The Accessibility Tree of the webpage is ") : j]
    act = text[j + len("\nThe proposed action is") :].strip().rstrip(".").strip()
    m = re.match(r"^(\w+)\s+(?:\[(\d+)\]\s+)?(\S+)", act)  # 744 rows have no node index (and an empty tree)
    role = m.group(3).strip("'") if m else "none"
    q = [k for k, ch in enumerate(act) if ch == "'"]
    name = act[q[0] + 1 : q[-1]] if len(q) >= 2 else ""
    first = tree.split("\n", 1)[0]
    qt = [k for k, ch in enumerate(first) if ch == "'"]
    title = first[qt[0] + 1 : qt[-1]] if len(qt) >= 2 else ""
    names = re.findall(r"\[\d+\]\s+\S+\s+'([^']*)'", tree)
    return {"role": role, "name": name, "title": title, "tree_names": names, "action_verb": (m.group(1).lower() if m else "")}


def parse_img(text: str) -> dict:
    u = text[text.rfind("<image>") :]
    tag = re.search(r"The tagHead of the element is (.*?)\.\n", u)
    url = re.search(r"The url of the web page is (\S+?)\.\n", u)
    return {"tag": tag.group(1).strip() if tag else "", "url": url.group(1) if url else ""}


def veto(words: list[str]) -> int:
    ws = [w.lower() for w in words]
    return int(any(w in VETO_WORDS for w in ws) or any((a, b) in VETO_PAIRS for a, b in zip(ws, ws[1:])))


def main() -> None:
    out_npz, out_js = HERE / "feats" / "wg_bits.npz", HERE / "feats" / "wg_bits.json"
    assert not out_npz.exists() and not out_js.exists(), "outputs exist"
    split = json.loads((HERE / "wg_split.json").read_text())
    a11y = json.loads((D / "monitor_factory_train_a11y.json").read_text())
    img = json.loads((D / "monitor_factory_train.json").read_text())
    assert len(a11y) == len(img) == 1982
    part = np.full(len(a11y), -1, np.int64)
    for k, p in enumerate(("train", "dev", "test")):
        part[split["rows"][p]] = k
    assert (part >= 0).all()
    site = [re.search(r"/downloads/([^/]+)/", d["images"][0]).group(1) for d in img]
    y = np.array([LABELS.index(json.loads(d["messages"][1]["content"])["risky_level"]) for d in a11y], np.int64)
    rows = [{**parse_a11y(a["messages"][0]["content"]), **parse_img(b["messages"][0]["content"])} for a, b in zip(a11y, img)]
    zt = np.load(TABLE)
    words = [str(w) for w in zt["words"]]
    codes = zt["codes"].astype(np.float32)
    live = np.abs(codes).sum(1) > 0
    row_of = {w: i for i, w in enumerate(words) if live[i]}
    tok = WF.glove_tokens

    def ids(ws):
        return [row_of[w] for w in ws if w in row_of]

    node_w = [tok(r["name"]) for r in rows]
    page_w = [tok(r["title"]) + [w for w in re.findall(r"[a-z0-9]+", r["url"].lower()) if w not in URL_DROP] for r in rows]
    tree_w = [[w for nm in r["tree_names"] for w in tok(nm)] for r in rows]
    node_i, page_i, tree_i = [ids(w) for w in node_w], [ids(w) for w in page_w], [ids(w) for w in tree_w]
    tr = np.flatnonzero(part == 0)
    vocab = sorted({t for k in tr for t in node_i[k] + page_i[k] + tree_i[k]})
    E = torch.nn.functional.normalize(torch.as_tensor(codes[vocab]), dim=1)
    local = {g: j for j, g in enumerate(vocab)}
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    Cn = _spherical_kmeans(E, K, 25, gen)
    En_all = torch.nn.functional.normalize(torch.as_tensor(codes), dim=1)

    def exists(id_lists):
        out = np.zeros((len(id_lists), K), np.float32)
        for r, li in enumerate(id_lists):
            if li:
                out[r] = (En_all[li] @ Cn.T).max(0).values.numpy()
        return out

    def counts(id_lists):
        out = np.zeros((len(id_lists), K), np.int64)
        for r, li in enumerate(id_lists):
            if li:
                a = (En_all[li] @ Cn.T).argmax(1).numpy()
                out[r] = np.bincount(a, minlength=K)
        return out

    ex_node, ex_page, cnt_tree = exists(node_i), exists(page_i), counts(tree_i)
    th_node, th_page = Thermo(8).fit(ex_node[tr]), Thermo(8).fit(ex_page[tr])
    blocks, names, groups = [], [], {"thermo": [], "onehot": [], "counts": []}

    def add(bits, nm):
        start = sum(b.shape[1] for b in blocks)
        blocks.append(bits.astype(np.uint8))
        names.extend(nm)
        return start

    def add_thermo(th, X, label):
        for d, t in enumerate(th.th):
            if len(t):
                s0 = add((X[:, d : d + 1] >= t[None, :]), [f"{label} word ~ concept {d} >= {v:.3f}" for v in t])
                groups["thermo"].append(list(range(s0, s0 + len(t))))

    add_thermo(th_node, ex_node, "node")
    for key in ("role", "tag"):
        vals = Counter(rows[k][key] for k in tr)
        keep = sorted(v for v, c in vals.items() if c >= MIN_ONEHOT)
        cats = keep + ["other"]
        M = np.zeros((len(rows), len(cats)), np.uint8)
        for r, row in enumerate(rows):
            M[r, cats.index(row[key]) if row[key] in keep else len(keep)] = 1
        s0 = add(M, [f"node {key} = {c}" for c in cats])
        groups["onehot"].append(list(range(s0, s0 + len(cats))))
    veto_bit = add(np.array([[veto(w)] for w in node_w], np.uint8), ["node name has an irreversible-action word"])
    node_end = sum(b.shape[1] for b in blocks)
    add_thermo(th_page, ex_page, "page title/url")
    for d in range(K):
        s0 = add(np.stack([cnt_tree[:, d] >= c for c in COUNT_LEVELS], 1), [f"tree words nearest concept {d} >= {c}" for c in COUNT_LEVELS])
        groups["counts"].append(list(range(s0, s0 + len(COUNT_LEVELS))))
    B = np.concatenate(blocks, 1)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, B=B, y=y, part=part, site=np.array(site))
    near = (Cn @ E.T).topk(4, dim=1).indices.numpy()
    meta = {"n_bits": int(B.shape[1]), "node_bits": [0, node_end], "page_bits": [node_end, int(B.shape[1])], "veto_bit": veto_bit,
            "names": names, "groups": groups, "K": K, "train_vocab": len(vocab),
            "concepts": ["|".join(words[vocab[j]] for j in r) for r in near],
            "veto_words": sorted(VETO_WORDS), "veto_pairs": sorted(" ".join(p) for p in VETO_PAIRS),
            "veto_on": {p: int(B[part == k, veto_bit].sum()) for k, p in enumerate(("train", "dev", "test"))},
            "on_share": {"node": round(float(B[part == 0, :node_end].mean()), 4), "page": round(float(B[part == 0, node_end:].mean()), 4)},
            "rows_without_node_word": int(sum(1 for li in node_i if not li)), "rows_without_page_word": int(sum(1 for li in page_i if not li)),
            "parts": {p: int((part == k).sum()) for k, p in enumerate(("train", "dev", "test"))},
            "sha256": {"wg_feats.py": sha16(Path(__file__).resolve()), "wg_split.json": sha16(HERE / "wg_split.json"),
                       "table": sha16(TABLE), "a11y": sha16(D / "monitor_factory_train_a11y.json"),
                       "img": sha16(D / "monitor_factory_train.json"), "wg_bits.npz": None}}
    meta["sha256"]["wg_bits.npz"] = sha16(out_npz)
    out_js.write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("names", "groups", "concepts")}))


if __name__ == "__main__":
    main()
