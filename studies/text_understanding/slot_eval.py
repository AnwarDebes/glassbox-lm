#!/usr/bin/env python
"""Span F1 for MASSIVE slot filling, the convention of the MASSIVE scorer (FitzGerald et al. 2023, github
alexa/massive): per-word labels are turned into BIO by convert_to_bio (a run of the same non-"Other" label is one
B- then I-; "Other" is O), and seqeval's micro F1 counts a predicted span as correct only if its type, start and end
all match. On BIO made by convert_to_bio a span is a maximal run of one non-"Other" label, which is what spans()
returns. Run this file to self-test."""


def spans(labels):
    """Set of (label, start, end) for the maximal runs of one label other than "Other"."""
    out, i, n = set(), 0, len(labels)
    while i < n:
        if labels[i] == "Other":
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        out.add((labels[i], i, j))
        i = j + 1
    return out


def span_f1(gold_seqs, pred_seqs):
    """Micro precision, recall and F1 over all utterances (lists of per-word label strings)."""
    tp = ng = npred = 0
    for g, p in zip(gold_seqs, pred_seqs):
        assert len(g) == len(p)
        G, P = spans(g), spans(p)
        tp += len(G & P)
        ng += len(G)
        npred += len(P)
    prec = tp / npred if npred else 0.0
    rec = tp / ng if ng else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "gold_spans": ng, "pred_spans": npred, "correct": tp}


def convert_to_bio(labels):
    """The MASSIVE scorer's convert_to_bio for word-level labels (no merged sub-word positions)."""
    out, prev = [], None
    for t in labels:
        if t == "Other":
            out.append("O")
        elif t != prev:
            out.append("B-" + t)
        else:
            out.append("I-" + t)
        prev = t
    return out


def _bio_spans(bio):
    """Chunks of a well-formed BIO sequence (conlleval: B- starts, I- of the same type continues)."""
    out, start, typ = set(), None, None
    for i, t in enumerate(bio + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if typ is not None:
                out.add((typ, start, i - 1))
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return out


if __name__ == "__main__":
    ex = ["Other", "Other", "Other", "Other", "time", "time", "date", "date"]  # wake me up at [time : five am] [date : this week]
    assert spans(ex) == {("time", 4, 5), ("date", 6, 7)}
    assert convert_to_bio(ex) == ["O", "O", "O", "O", "B-time", "I-time", "B-date", "I-date"]
    import random

    random.seed(0)
    for _ in range(2000):  # spans() equals the chunks of convert_to_bio on random label sequences
        seq = [random.choice(["Other", "a", "b"]) for _ in range(random.randint(0, 12))]
        assert spans(seq) == _bio_spans(convert_to_bio(seq)), seq
    r = span_f1([ex], [["Other"] * 4 + ["time", "time", "date", "Other"]])
    assert r["correct"] == 1 and r["gold_spans"] == 2 and r["pred_spans"] == 2 and abs(r["f1"] - 0.5) < 1e-12
    print("slot_eval self-test ok")
