"""Unit tests: pure-Python guard decisions (veto clauses, tau bias, twins), D6 rule handling, the replay store."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))

from e2guard.bits import FACT_BITS, veto_premises
from e2guard.d6_actgov import (
    ScriptedProposer,
    parse_rules,
    rule_literals,
    rules_fire,
)
from e2guard.models import GuardedTM, LRTwin, RuleSet, TreeTwin
from e2guard.store_llm import ReplayStore, body_key, canonical


def toy_tm(tau: int = 1) -> dict:
    """A random 'learned' clause set over the fact bits, exported the way train_loso.export_tm does."""
    rng = random.Random(0)
    F = len(FACT_BITS)
    clauses, W = [], []
    for _ in range(40):
        clauses.append(sorted(rng.sample(range(2 * F), rng.randint(1, 4))))
        W.append([rng.randint(-30, 30), rng.randint(-30, 30)])
    kind, vid = ["learned"] * 40, [None] * 40
    clauses.append([])
    W.append([0, 1 - tau])
    kind.append("bias")
    vid.append(None)
    M = sum(max(w[1], 0) for w in W) + sum(abs(w[0]) for w in W) + 1
    col = {n: i for i, n in enumerate(FACT_BITS)}
    for v in veto_premises():
        clauses.append([col[b] for b in v["bits"]])
        W.append([0, -M])
        kind.append("veto")
        vid.append(v["id"])
    return {"names": FACT_BITS, "clauses": clauses, "W": W, "clause_kind": kind, "veto_id": vid}


def test_veto_clause_beats_every_input_and_tau_shifts_the_margin():
    g = GuardedTM(toy_tm())
    col = {n: i for i, n in enumerate(FACT_BITS)}
    rng = random.Random(1)
    for _ in range(3000):
        x = [rng.randint(0, 1) for _ in FACT_BITS]
        d = g.decide(x)
        fired = [v["id"] for v in veto_premises() if all(x[col[b]] for b in v["bits"])]
        assert sorted(d["vetoes"]) == sorted(fired)
        if fired:
            assert not d["allow"]
        else:
            raw, _ = g.scores(x, with_veto=False, with_bias=False)
            assert d["allow"] == (raw[1] - raw[0] >= 1)
    g5 = GuardedTM(toy_tm(tau=5))
    for _ in range(500):
        x = [rng.randint(0, 1) for _ in FACT_BITS]
        if not g5.decide(x)["vetoes"]:
            raw, _ = g5.scores(x, with_veto=False, with_bias=False)
            assert g5.decide(x)["allow"] == (raw[1] - raw[0] >= 5)


def test_twins_and_rule_sets():
    lr = LRTwin({"names": ["a", "b"], "coef": [2.0, -3.0], "intercept": 0.5, "threshold": 0.7})
    assert lr.decide([1, 0])["allow"] and not lr.decide([0, 1])["allow"]
    tr = TreeTwin({"names": ["a"], "nodes": [{"f": 0, "l": 1, "r": 2}, {"p": 0.2}, {"p": 0.9}], "threshold": 0.5})
    assert tr.decide([1])["allow"] and not tr.decide([0])["allow"]
    rs = RuleSet({"names": ["a", "b"], "rules": [["a", "!b"]]})
    assert rs.decide([1, 0])["allow"] and not rs.decide([1, 1])["allow"] and not rs.decide([0, 0])["allow"]


def test_d6_parse_and_scripted_rounds():
    names = FACT_BITS
    rules, problems = parse_rules(json.dumps({"rules": [["kind=read"], ["kind=nope"], []]}), names)
    assert rules == [["kind=read"]] and len(problems) == 2
    assert parse_rules("not json", names)[0] == []
    assert rules_fire([["kind=read", "!prov[url]=not_found"]], {"kind=read"})
    assert not rules_fire([["kind=read", "!prov[url]=not_found"]], {"kind=read", "prov[url]=not_found"})
    F = len(names)
    assert rule_literals(["kind=read", "!kind=write"], names) == [names.index("kind=read"), names.index("kind=write") + F]
    p = ScriptedProposer()
    r1, pr1 = parse_rules(p.propose([]), names)
    r3 = [parse_rules(p.propose([]), names)[0] for _ in range(2)][-1]
    assert pr1 and ["kind=send_or_pay"] in r1 and ["kind=send_or_pay"] not in r3


def test_replay_store_round_trip(tmp_path):
    s = ReplayStore(tmp_path / "store.jsonl")
    body = {"b": 1, "a": [1, 2], "model": "m"}
    k = body_key(body)
    assert k == body_key({"model": "m", "a": [1, 2], "b": 1})  # key order does not matter
    assert canonical(body) == '{"a":[1,2],"b":1,"model":"m"}'
    s.put(k, {"x": 1}, {"utc": "t"})
    s.put(k, {"x": 2}, {"utc": "t2"})  # first reply wins
    s2 = ReplayStore(tmp_path / "store.jsonl")
    assert s2.get(k)["reply"] == {"x": 1}


def test_negative_a_too_small_veto_weight_lets_allow_win():
    """The veto test above must be able to fail: with M = 1 instead of the safe M, some input wins ALLOW under a premise."""
    d = toy_tm()
    for j, k in enumerate(d["clause_kind"]):
        if k == "veto":
            d["W"][j] = [0, -1]
    g = GuardedTM(d)
    col = {n: i for i, n in enumerate(FACT_BITS)}
    rng = random.Random(2)
    broken = 0
    for _ in range(3000):
        x = [rng.randint(0, 1) for _ in FACT_BITS]
        if [v for v in veto_premises() if all(x[col[b]] for b in v["bits"])] and g.decide(x)["allow"]:
            broken += 1
    assert broken > 0


def test_negative_d6_policy_refuses_rules_without_a_passed_solver_check(tmp_path):
    import pytest

    from e2guard.policies import D6Policy

    qb = {"names": [], "by_sha256": {}}
    (tmp_path / "q.json").write_text(json.dumps(qb))
    (tmp_path / "r.json").write_text(json.dumps({"names": FACT_BITS, "rules": [["kind=read"]],
                                                 "solver_check": {"passed": False}}))
    paths = {s: (str(tmp_path / "r.json"), str(tmp_path / "q.json")) for s in ("banking",)}
    with pytest.raises(AssertionError):
        D6Policy(paths)
    (tmp_path / "r2.json").write_text(json.dumps({"names": FACT_BITS + ["extra"], "rules": [],
                                                  "solver_check": {"passed": True}}))
    with pytest.raises(AssertionError):  # bit names that differ from the extractor's
        D6Policy({"banking": (str(tmp_path / "r2.json"), str(tmp_path / "q.json"))})


def test_negative_unknown_query_has_no_bits_and_duplicate_names_are_refused(tmp_path):
    import pytest

    from e2guard.bits import QueryBits

    (tmp_path / "q.json").write_text(json.dumps({"names": ["a", "b"], "by_sha256": {"abc": [0]}}))
    qb = QueryBits(tmp_path / "q.json")
    assert qb.on("abc") == {"a"}
    with pytest.raises(KeyError):
        qb.on("not-a-known-query")
    (tmp_path / "d.json").write_text(json.dumps({"names": ["a", "a"], "by_sha256": {}}))
    with pytest.raises(AssertionError):
        QueryBits(tmp_path / "d.json")
