"""ABCD (Action-Based Conversations Dataset, Chen et al., NAACL 2021): next-action examples.

Source: github.com/asappresearch/abcd, ``data/abcd_v1.1.json.gz`` (train 8,034 / dev 1,004 /
test 1,004 conversations), read from ``data/untrusted/abcd`` (downloaded 2026-10-16; parsed as
JSON only).

One example per action turn of a conversation (29,190 in train). What the decision layer sees
at that point is what the agent had seen: the agent and customer utterances so far (delexed
text, as in the paper's Action State Tracking task) and the agent's own earlier actions. The
target is the action's button (30 buttons, in the ontology's order, as the paper's code).

Text fields per example (for a frozen encoder):
- ``cust``: every customer utterance so far, joined (what the customer wants);
- ``recent``: the last 3 utterances before the action, with speaker tags (what is asked now);
- ``full``: every turn so far, actions included as "action: <button>", the most recent last.

Memory (the agent's own history, also in the paper's AST input): the buttons pressed so far.

The paper's AST example set differs: verify-identity and validate-purchase give one example
per slot value (3), and an example is kept only when its value can be found (their
``value_to_id`` with the BERT tokenizer). ``ast_count`` reproduces that count per action turn
(scripts/abcd_prepare.py), so the paper's B-Slot accuracy can be recomputed as a weighted
accuracy.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABCD_DIR = ROOT / "data" / "untrusted" / "abcd"
SPLITS = ("train", "dev", "test")
RECENT_TURNS = 3


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def merge_subflow(s: str) -> str:
    """The scenario's subflow without the FAQ variant number (boots_how_3 -> boots, timing_4 -> timing)."""
    s = re.sub(r"_(how|other)_\d+$", "", s)
    return re.sub(r"_\d+$", "", s)


def ontology(path: Path = ABCD_DIR) -> dict:
    return json.loads((path / "ontology.json").read_text())


def buttons(ont: dict) -> list[str]:
    """The 30 buttons in the order of the paper's ``prepare_action_labels``."""
    out = []
    for _section, bs in ont["actions"].items():
        out.extend(bs.keys())
    return out


@dataclass
class ActionExample:
    convo_id: int
    k: int  # index of this action within the conversation (0 = first action)
    flow: str
    subflow: str  # merged (57 values)
    button: str
    prev: list[str]
    cust: str
    recent: str
    full: str
    utts: list = field(default_factory=list)  # [(speaker, text)] before the action (for the AST count)


def load_convs(path: Path = ABCD_DIR) -> dict:
    p = path / "abcd_v1.1.json.gz"
    with gzip.open(p) as f:
        return json.load(f)


def action_examples(convs: list) -> list[ActionExample]:
    out = []
    for conv in convs:
        utts: list[tuple[str, str]] = []
        turns: list[str] = []
        prev: list[str] = []
        sc = conv["scenario"]
        for t in conv["delexed"]:
            sp, txt = t["speaker"], t["text"]
            if sp in ("agent", "customer"):
                utts.append((sp, txt))
                turns.append(f"{sp}: {txt}")
                continue
            button = t["targets"][2]
            cust = " | ".join(x for s, x in utts if s == "customer")
            recent = " ".join(f"{s}: {x}" for s, x in utts[-RECENT_TURNS:])
            out.append(ActionExample(
                convo_id=int(conv["convo_id"]), k=len(prev), flow=sc["flow"], subflow=merge_subflow(sc["subflow"]),
                button=button, prev=list(prev), cust=cust, recent=recent, full=" | ".join(turns), utts=list(utts)))
            prev.append(button)
            turns.append(f"action: {button}")
    return out


def memory_bits(examples: list[ActionExample], button_list: list[str], max_count: int = 6) -> tuple[np.ndarray, list[str]]:
    """done(b), last(b), and "at least k actions so far" (k = 1..max_count)."""
    idx = {b: i for i, b in enumerate(button_list)}
    B = len(button_list)
    X = np.zeros((len(examples), 2 * B + max_count), dtype=np.uint8)
    for r, e in enumerate(examples):
        for b in e.prev:
            X[r, idx[b]] = 1
        if e.prev:
            X[r, B + idx[e.prev[-1]]] = 1
        for k in range(1, max_count + 1):
            X[r, 2 * B + k - 1] = len(e.prev) >= k
    names = [f"done({b})" for b in button_list] + [f"last action = {b}" for b in button_list]
    names += [f"at least {k} actions so far" for k in range(1, max_count + 1)]
    return X, names


def sources(path: Path = ABCD_DIR) -> dict:
    return {f: _sha(path / f) for f in ("abcd_v1.1.json.gz", "ontology.json", "guidelines.json")}


# ----- readable boolean view for next-action decisions ----------------------------------------

PARTS = ("mem", "intent", "step", "words", "mem2", "fstep", "fwords")


def load_examples(split: str, dataset: str = "abcd") -> tuple[list[dict], list[str]]:
    """Examples of one split; ``dataset`` "star" reads the STAR examples of scripts/star_prepare.py (same format)."""
    p = ROOT / "data" / dataset / f"examples_{split}.json.gz"
    with gzip.open(p, "rt") as f:
        d = json.load(f)
    return d["rows"], d["buttons"]


# every part ABCDView knows, in the fixed order of its blocks (xcust and xrecent added 2026-10-17 for the
# no-network agent; before, an unknown part name was silently ignored)
PARTS = ("mem", "intent", "step", "words", "mem2", "fstep", "fwords", "xcust", "xrecent")


class ABCDView:
    """Named bits for one decision (next button):

    - ``mem``: done(b), last action = b, at least k actions so far (the agent's own history);
    - ``intent``: whitened similarity of the customer's words so far to the centre of each
      subflow's training examples (levels 5/25/50/75%), and "one of the r nearest is <subflow>";
    - ``step``: the same for the last 3 turns, with one centre per next button;
    - ``words``: presence of the K most frequent 1-2-grams of the last 3 turns.
    - ``xcust``, ``xrecent`` (no network at run time; src/dl/static_text.py): existsT bits of the
      customer's words so far and of the last 3 turns, "some word of the <field> is close to concept k
      at level j", a table lookup and a bitwise OR per text. Built only when asked for.
    """

    def __init__(self, parts=("mem", "intent", "step", "words"), lex_k: int = 1000, whiten: float = 0.001,
                 levels=(5, 25, 50, 75), ranks=(1, 3), seed: int = 0):
        from dl.boolean import Lex, Proto

        self.parts = tuple(parts)
        unknown = [q for q in self.parts if q not in PARTS]
        if unknown:
            raise ValueError(f"unknown part(s) {unknown}; known: {PARTS}")
        self.intent = Proto(k=1, levels=levels, ranks=ranks, seed=seed, whiten=whiten) if "intent" in parts else None
        self.step = Proto(k=1, levels=levels, ranks=ranks, seed=seed, whiten=whiten) if "step" in parts else None
        self.lex = Lex(lex_k) if "words" in parts else None
        # added after e25 (exploration): the same two kinds of bits on the whole conversation so far
        self.fstep = Proto(k=1, levels=levels, ranks=ranks, seed=seed, whiten=whiten) if "fstep" in parts else None
        self.flex = Lex(lex_k) if "fwords" in parts else None
        self.xcust = self.xrecent = None
        if "xcust" in parts or "xrecent" in parts:
            from dl.static_text import ExistsT

            self.xcust = ExistsT(seed=seed, what="customer's words so far") if "xcust" in parts else None
            self.xrecent = ExistsT(seed=seed, what="last 3 turns") if "xrecent" in parts else None

    def fit(self, rows: list[dict], emb: dict, buttons_: list[str]) -> "ABCDView":
        self.buttons = buttons_
        self.subflows = sorted({r["subflow"] for r in rows})
        names = []
        if "mem" in self.parts:
            names += memory_names(buttons_)
        if self.intent is not None:
            ys = np.array([self.subflows.index(r["subflow"]) for r in rows])
            self.intent.fit(emb["cust"], ys, texts=None, class_names=self.subflows)
            names += _rename(self.intent.names, "customer so far", "subflow")
        if self.step is not None:
            yb = np.array([r["y"] for r in rows])
            self.step.fit(emb["recent"], yb, texts=None, class_names=buttons_)
            names += _rename(self.step.names, "last 3 turns", "next-button")
        if self.lex is not None:
            self.lex.fit(None, texts=[r["recent"] for r in rows])
            names += [n.replace("contains", "last 3 turns contain") for n in self.lex.names]
        if "mem2" in self.parts:
            names += [f"action before last = {b}" for b in buttons_]
        if self.fstep is not None:
            yb = np.array([r["y"] for r in rows])
            self.fstep.fit(emb["full"], yb, texts=None, class_names=buttons_)
            names += _rename(self.fstep.names, "conversation so far", "next-button")
        if self.flex is not None:
            self.flex.fit(None, texts=[r["full"] for r in rows])
            names += [n.replace("contains", "conversation so far contains") for n in self.flex.names]
        if self.xcust is not None:
            self.xcust.fit([r["cust"] for r in rows])
            names += self.xcust.names
        if self.xrecent is not None:
            self.xrecent.fit([r["recent"] for r in rows])
            names += self.xrecent.names
        self.names = names
        return self

    def static_info(self) -> dict:
        """Provenance of the no-network parts (concept, threshold and code hashes), empty without them."""
        return {k: v.info for k, v in (("xcust", self.xcust), ("xrecent", self.xrecent)) if v is not None}

    def transform(self, rows: list[dict], emb: dict) -> np.ndarray:
        blocks = []
        if "mem" in self.parts:
            blocks.append(memory_from_rows(rows, self.buttons))
        if self.intent is not None:
            blocks.append(self.intent.transform(emb["cust"]))
        if self.step is not None:
            blocks.append(self.step.transform(emb["recent"]))
        if self.lex is not None:
            blocks.append(self.lex.transform(None, texts=[r["recent"] for r in rows]))
        if "mem2" in self.parts:
            idx = {b: i for i, b in enumerate(self.buttons)}
            M2 = np.zeros((len(rows), len(self.buttons)), dtype=np.uint8)
            for i, r in enumerate(rows):
                if len(r["prev"]) >= 2:
                    M2[i, idx[r["prev"][-2]]] = 1
            blocks.append(M2)
        if self.fstep is not None:
            blocks.append(self.fstep.transform(emb["full"]))
        if self.flex is not None:
            blocks.append(self.flex.transform(None, texts=[r["full"] for r in rows]))
        if self.xcust is not None:
            blocks.append(self.xcust.transform([r["cust"] for r in rows]))
        if self.xrecent is not None:
            blocks.append(self.xrecent.transform([r["recent"] for r in rows]))
        return np.concatenate(blocks, 1).astype(np.uint8)


def _rename(names: list[str], what: str, kind: str) -> list[str]:
    """Proto names ('sim_w(q, proto3[x]) >= t', 'one of the r nearest prototypes is x') in the decision's words."""
    out = []
    for n in names:
        n = re.sub(r"^sim_w\(q, proto\d+\[(.+?)\]\) >= ", rf"sim({what}, \1) >= ", n)
        n = re.sub(r"^one of the (\d+) nearest prototypes is (.+)$", rf"\2 is among the \1 nearest {kind} centres of the {what}", n)
        out.append(n)
    return out


def memory_names(buttons_: list[str], max_count: int = 6) -> list[str]:
    names = [f"done({b})" for b in buttons_] + [f"last action = {b}" for b in buttons_]
    return names + [f"at least {k} actions so far" for k in range(1, max_count + 1)]


def memory_from_rows(rows: list[dict], buttons_: list[str], max_count: int = 6) -> np.ndarray:
    idx = {b: i for i, b in enumerate(buttons_)}
    B = len(buttons_)
    X = np.zeros((len(rows), 2 * B + max_count), dtype=np.uint8)
    for r, e in enumerate(rows):
        for b in e["prev"]:
            X[r, idx[b]] = 1
        if e["prev"]:
            X[r, B + idx[e["prev"][-1]]] = 1
        for k in range(1, max_count + 1):
            X[r, 2 * B + k - 1] = len(e["prev"]) >= k
    return X


class Clusters:
    """Label-free named bits from an embedding: spherical k-means clusters of the training texts
    (no labels), each named by its member text nearest to the centre. Bits: similarity to each
    centre at the members' own quantile levels, and "cluster j is among the r nearest"."""

    def __init__(self, k: int = 64, levels=(25, 50, 75), ranks=(1, 3), seed: int = 0, what: str = "last 3 turns",
                 device: str = "cuda"):
        self.k, self.levels, self.ranks, self.seed, self.what, self.dev = k, tuple(levels), tuple(ranks), seed, what, device

    def fit(self, X: np.ndarray, texts: list[str]) -> "Clusters":
        import torch

        from dl.boolean import _spherical_kmeans

        gen = torch.Generator(device=self.dev)
        gen.manual_seed(self.seed)
        Z = torch.nn.functional.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.dev), dim=1)
        self.P = _spherical_kmeans(Z, self.k, 25, gen)
        S = Z @ self.P.T
        a = S.argmax(1)
        self.th = []
        for j in range(self.P.shape[0]):
            m = S[a == j, j] if bool((a == j).any()) else S[:, j]
            self.th.append(np.unique(np.percentile(m.cpu().numpy(), self.levels)).astype(np.float32))
        near = S.argmax(0).cpu().numpy()
        label = [texts[i][:60] for i in near]
        self.names = [f'sim({self.what}, "{label[j]}") >= {t:.3f}' for j, ts in enumerate(self.th) for t in ts]
        self.names += [f'cluster "{label[j]}" is among the {r} nearest to the {self.what}' for r in self.ranks for j in range(self.P.shape[0])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        Z = torch.nn.functional.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.dev), dim=1)
        S = Z @ self.P.T
        bits = [torch.cat([(S[:, j : j + 1] >= torch.as_tensor(t, device=self.dev)[None, :]) for j, t in enumerate(self.th)], 1)]
        order = S.argsort(1, descending=True)
        for r in self.ranks:
            b = torch.zeros_like(S, dtype=torch.bool)
            b.scatter_(1, order[:, :r], True)
            bits.append(b)
        return torch.cat(bits, 1).to(torch.uint8).cpu().numpy()


class ABCDFreeView:
    """Label-free named bits for learning from reward: memory, words of the last 3 turns, and
    unsupervised clusters of the last 3 turns and of the customer's words (no label of any kind)."""

    def __init__(self, parts=("mem", "words", "rclust", "cclust"), lex_k: int = 1000, k: int = 64, seed: int = 0):
        from dl.boolean import Lex

        self.parts = tuple(parts)
        self.lex = Lex(lex_k) if "words" in parts else None
        self.rc = Clusters(k, seed=seed, what="last 3 turns") if "rclust" in parts else None
        self.cc = Clusters(k, seed=seed + 1, what="customer so far") if "cclust" in parts else None

    def fit(self, rows: list[dict], emb: dict, buttons_: list[str]) -> "ABCDFreeView":
        self.buttons = buttons_
        names = memory_names(buttons_) if "mem" in self.parts else []
        if self.lex is not None:
            self.lex.fit(None, texts=[r["recent"] for r in rows])
            names += [n.replace("contains", "last 3 turns contain") for n in self.lex.names]
        if self.rc is not None:
            self.rc.fit(emb["recent"], [r["recent"] for r in rows])
            names += self.rc.names
        if self.cc is not None:
            self.cc.fit(emb["cust"], [r["cust"] for r in rows])
            names += self.cc.names
        self.names = names
        return self

    def transform(self, rows: list[dict], emb: dict) -> np.ndarray:
        blocks = []
        if "mem" in self.parts:
            blocks.append(memory_from_rows(rows, self.buttons))
        if self.lex is not None:
            blocks.append(self.lex.transform(None, texts=[r["recent"] for r in rows]))
        if self.rc is not None:
            blocks.append(self.rc.transform(emb["recent"]))
        if self.cc is not None:
            blocks.append(self.cc.transform(emb["cust"]))
        return np.concatenate(blocks, 1).astype(np.uint8)
