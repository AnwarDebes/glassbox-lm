"""S0: splits, delexicalisation, tokens, conditions from dialog_act only, train-only reads, vocabulary."""

import hashlib
import json

import numpy as np
import pytest
from helpers import synth_dirs

from astruct import data
from astruct.paths import load_tokenize_ref

TOK = load_tokenize_ref().tokens


def test_split_rule():
    did = "PMUL4398.json"
    h = int.from_bytes(hashlib.sha256(did.encode("utf-8")).digest(), "big") % 100
    assert data.h100(did) == h
    assert [data.split_of(x) for x in (0, 5, 6, 11, 12, 99)] == [data.CAL, data.CAL, data.HOLD, data.HOLD, data.FIT, data.FIT]
    assert data.salted_mod(did, "es", 10) == int.from_bytes(hashlib.sha256(b"es:" + did.encode()).digest(), "big") % 10


def test_spans_overlap_and_range():
    utt = "try pizza hut in the north"
    si = [["Restaurant-Inform", "name", "pizza hut", 4, 13], ["Restaurant-Recommend", "name", "pizza hut", 4, 13],
          ["Restaurant-Inform", "area", "north", 21, 26], ["Restaurant-Inform", "food", "x", 30, 40]]
    kept, n_over, n_range = data.kept_spans(utt, si)
    assert [k[2] for k in kept] == ["[restaurant-name]", "[restaurant-area]"]
    assert (n_over, n_range) == (1, 1)
    toks = data.reply_tokens(utt, kept, TOK)
    assert toks == ["try", "[restaurant-name]", "in", "the", "[restaurant-area]"]
    assert data.token_count_ok(utt, kept, toks, TOK)
    # negative: a cut that loses a segment is caught by the count test
    assert not data.token_count_ok(utt, kept, toks[:-1], TOK)
    # the regex alone would split a placeholder into five tokens
    assert TOK("[restaurant-name]") == ["[", "restaurant", "-", "name", "]"]


def test_tokens_contractions_next_to_placeholders():
    utt = "Cambridge's best isn't far"
    kept, _, _ = data.kept_spans(utt, [["Attraction-Inform", "name", "Cambridge", 0, 9]])
    toks = data.reply_tokens(utt, kept, TOK)
    assert toks == ["[attraction-name]", "'s", "best", "is", "n't", "far"]
    assert data.token_count_ok(utt, kept, toks, TOK)


def test_conditions_from_dialog_act_only():
    e = {"dialog_act": {"Hotel-Inform": [["name", "acorn"], ["choice", "none"]], "general-reqmore": [["none", "none"]],
                        "Hotel-Request": [["area", "?"]]},
         "span_info": [["Hotel-Inform", "name", "acorn", 0, 5], ["Train-Inform", "day", "monday", 9, 15]]}
    c = data.conditions(e)
    assert c["acts"] == ["Hotel-Inform", "Hotel-Request", "general-reqmore"]
    assert ("Hotel-Inform", "choice") in c["pairs"] and ("general-reqmore", "none") in c["pairs"]
    assert c["expected"] == ["[hotel-name]"]  # none and ? are not values
    assert data.conditions({"dialog_act": e["dialog_act"]}) == c  # review 2 (iii): span_info deleted

    # negative: a rule that also read span_info would give other bits for this turn
    def leaky(entry):
        out = data.conditions(entry)
        out["acts"] = sorted(set(out["acts"]) | {x[0] for x in entry.get("span_info", [])})
        return out
    assert leaky(e) != leaky({"dialog_act": e["dialog_act"]})


def test_train_only_and_manifest(tmp_path):
    mw, c4 = synth_dirs("data_test", n=60, seed=5)
    opened = []

    def opener(p, *a, **k):
        opened.append(str(p))
        return open(p, *a, **k)
    _arrays, meta = data.make_corpus(mw, c4, TOK, log=lambda *a: None, opener=opener)
    assert opened and all(("/train/" in p) or p.endswith("dialog_acts.json") for p in opened)
    assert not any("/dev/" in p or "/test/" in p for p in opened)
    assert meta["source"]["n_act_entries_dropped"] == 40  # the decoy dev and test entries, dropped unused
    # every reply passed the span_info and token-count tests
    assert meta["checks"]["span_info_test_turns"] == meta["counts"]["replies"] == meta["checks"]["token_count_test_replies"]
    # a file that does not match MANIFEST.sha256 stops S0
    bad = tmp_path / "mw"
    import shutil
    shutil.copytree(mw, bad)
    p = bad / "train" / "dialogues_001.json"
    d = json.loads(p.read_text())
    d[0]["turns"][1]["utterance"] += " changed"
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="MANIFEST"):
        data.load_train(bad)


def test_vocab_and_unk():
    mw, c4 = synth_dirs("data_test", n=60, seed=5)
    arrays, meta = data.make_corpus(mw, c4, TOK, log=lambda *a: None)
    c = data.Corpus(arrays, meta)
    v = c.vocab
    assert v[:2] == ["<end>", "<unk>"] and all(data.is_placeholder(w) for w in v[2:2 + c.is_ph.sum()])
    # rebuild the fit counts: every non-placeholder word in the vocabulary has a fit count >= 2
    from collections import Counter
    tk = load_tokenize_ref().tokens
    dl, acts, _ = data.load_train(mw)
    recs, _ = data.build_replies(dl, acts, tk)
    cnt = Counter(t for r in recs if r["split"] == data.FIT for t in r["toks"])
    assert all(cnt[w] >= 2 for w in v[2:] if not data.is_placeholder(w))
    assert all(w in v for w in cnt if data.is_placeholder(w))
    # once-only words became <unk> in the token arrays, and every reply ends with <end>
    ends = arrays["tok"][arrays["tok_off"][1:] - 1]
    assert (ends == c.END).all()
    assert meta["counts"]["tokens_mapped_to_unk"] >= 1
    assert (arrays["tok"] == c.UNK).sum() >= 1
    # expected placeholders are in the vocabulary and come from values that are not none / ? / empty
    assert set(arrays["exp_ids"].tolist()) <= set(np.flatnonzero(c.is_ph).tolist())
    # training rows: fit dialogues with h in 12..33
    h, s, tr = arrays["r_h"], arrays["r_split"], arrays["r_train"]
    assert (tr == ((s == data.FIT) & (h >= 12) & (h <= 33))).all()


def test_only_s0_opens_multiwoz_and_never_dev_or_test():
    """Static check: the MultiWOZ folder is named only by paths.py (constants and the folder listing), freeze.py (it
    hashes MANIFEST.sha256) and S0; dev/ and test/ occur only in the listing (ls of the folder, never its files)."""
    from pathlib import Path
    A = Path(__file__).resolve().parents[1]
    files = sorted(A.glob("astruct/*.py")) + sorted(A.glob("*.py")) + sorted(A.glob("*.sh"))
    users = sorted(f.name for f in files if "mwoz" in f.read_text().lower() or "multiwoz" in f.read_text().lower())
    assert users == ["data.py", "freeze.py", "paths.py", "s0_data.py"]
    src = (A / "astruct" / "paths.py").read_text()
    assert src.count('MWOZ / "dev"') == 1 and src.count('MWOZ / "test"') == 1
    for f in files:
        if f.name != "paths.py" and f.suffix == ".py":
            t = f.read_text()
            assert '"dev"' not in t and '"test"' not in t and "/dev/" not in t and "/test/" not in t, f.name
    assert "MWOZ" not in (A / "freeze.py").read_text().replace('paths.MWOZ / "MANIFEST.sha256"', "")


def test_concepts_deterministic():
    mw, c4 = synth_dirs("data_test", n=60, seed=5)
    a1, _ = data.make_corpus(mw, c4, TOK, log=lambda *a: None)
    a2, _ = data.make_corpus(mw, c4, TOK, log=lambda *a: None)
    assert np.array_equal(a1["concept32"], a2["concept32"]) and np.array_equal(a1["concept64"], a2["concept64"])
    assert a1["concept64"].max() < 64 and (a1["concept64"] >= 0).sum() > 0
