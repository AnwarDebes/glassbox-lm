"""Tests of the training-row rule (veto-clean, deduplicated; scripts/build_rows_from_results.py; review issue 8), the
row builder END TO END on stand-in data, the E4.3 go-bar (scripts/gobar.py; review issue 4) and the simulated
re-confirmation (bits.reconfirm)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import sys
from pathlib import Path

import numpy as np
import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import build_rows_from_results as BR  # noqa: E402
import gobar as GB  # noqa: E402
from e4guard import bits as B  # noqa: E402

GPT41 = Path(f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
             "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/results/final/"
             "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")
SPLIT = json.loads(Path(f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/"
                        "extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/retail/"
                        "split_tasks.json").read_text())
SPLIT_OF = {t: s for s in ("train", "test") for t in SPLIT[s]}

ORDER = {"order_id": "#W1000001", "user_id": "ann_1", "status": "pending", "items": [{"item_id": "1111111111"}],
         "payment_history": [{"transaction_type": "payment", "amount": 10.0, "payment_method_id": "credit_card_1"}]}


def _sim(task, listing, reply, reason="no longer needed"):
    msgs = [{"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "My email is ann@x.com"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "name": "find_user_id_by_email",
                                                                   "arguments": {"email": "ann@x.com"}}]},
            {"role": "tool", "id": "a", "content": "ann_1"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "name": "get_order_details",
                                                                   "arguments": {"order_id": "#W1000001"}}]},
            {"role": "tool", "id": "b", "content": json.dumps(ORDER)},
            {"role": "assistant", "content": listing},
            {"role": "user", "content": reply},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "name": "cancel_pending_order",
                                                                   "arguments": {"order_id": "#W1000001",
                                                                                 "reason": reason}}]},
            {"role": "tool", "id": "c", "content": json.dumps({**ORDER, "status": "cancelled"})}]
    return {"task_id": task, "trial": 0, "messages": msgs}


def _results(sims):
    gold = {"evaluation_criteria": {"actions": [{"name": "cancel_pending_order",
                                                 "arguments": {"order_id": "#W1000001", "reason": "no longer needed"}}]}}
    return {"tasks": [{"id": "0", **gold}, {"id": "1", **gold}], "simulations": sims}


def test_rows_are_veto_clean_and_deduplicated_across_a0_and_a1():
    ok = _sim("0", "Cancel order #W1000001, reason no longer needed?", "Yes.")
    wrong = _sim("0", "Cancel order #W1000001, reason ordered by mistake?", "Yes.", reason="ordered by mistake")
    vetoed = _sim("1", "Cancel your order?", "Yes.")                  # no order id before a cancel: E4-R3a fires
    a0 = BR.write_rows(_results([ok, vetoed]), "qwen_A0", "train", SPLIT_OF)
    a1 = BR.write_rows(_results([ok, wrong]), "qwen_A1", "train", SPLIT_OF)   # task 0 as in A0, plus a wrong write
    assert [r["label"] for r in a0] == [0, 0] and a0[1]["veto_clean"] is False and "E4-R3a" in a0[1]["veto_clauses"]
    kept, c = BR.select_training_rows(a0 + a1)
    assert c["qwen_A0:vetoed"] == 1 and c["qwen_A1:duplicate"] == 1
    assert [(r["source"], r["label"]) for r in kept] == [("qwen_A0", 0), ("qwen_A1", 1)]
    assert all(not B.veto_fires(np.array(r["x"], np.uint8)) for r in kept)
    assert c["kept_DENY"] == 1 and c["kept_ALLOW"] == 1


@pytest.mark.skipif(not GPT41.exists(), reason="stand-in file missing")
def test_row_builder_end_to_end_on_standin_data(tmp_path):
    d = json.loads(GPT41.read_text())
    t0 = {"tasks": d["tasks"], "simulations": [s for s in d["simulations"] if s["trial"] == 0]}
    t1 = {"tasks": d["tasks"], "simulations": [s for s in d["simulations"] if s["trial"] == 1]}
    p0, p1 = tmp_path / "t0.json", tmp_path / "t1.json"
    p0.write_text(json.dumps(t0))
    p1.write_text(json.dumps(t1))
    a0 = BR.write_rows(t0, "qwen_A0", "train", SPLIT_OF)
    a1 = BR.write_rows(t1, "qwen_A1", "train", SPLIT_OF)
    kept, c = BR.select_training_rows(a0 + a1)
    n_clean = sum(r["veto_clean"] for r in a0 + a1)
    assert 0 < len(kept) <= n_clean < len(a0 + a1)
    assert all(r["split"] == "train" for r in kept)
    assert all(not B.veto_fires(np.array(r["x"], np.uint8)) for r in kept)
    keys = [(r["task"], r["tool"], r["args"], "".join(map(str, r["x"]))) for r in kept]
    assert len(keys) == len(set(keys))
    # the same through the command line, with --all-out
    import subprocess

    out, allo = tmp_path / "rows.json", tmp_path / "all.json"
    py = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"
    sf = (f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
          "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/retail/split_tasks.json")
    p = subprocess.run([py, "-I", "-B", str(E4 / "scripts/build_rows_from_results.py"), "--results", str(p0), str(p1),
                        "--sources", "qwen_A0", "qwen_A1", "--split", "train", "--split-file", sf, "--out", str(out),
                        "--all-out", str(allo)], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-500:]
    k = json.loads(out.read_text())
    assert k["counts"]["kept"] == len(kept) and len(k["rows"]) == len(kept)
    assert len(json.loads(allo.read_text())["rows"]) == len(a0 + a1)


def _oof(task, label, deny, persist=0):
    return {"task": task, "label": label, "deny_tm_seed1": deny, "deny_tm_seed1_after_reconfirm": persist,
            "deny_lr": 0, "deny_lr_after_reconfirm": 0, "deny_tree": 0, "deny_tree_after_reconfirm": 0}


def test_go_bar_counts_and_boundaries():
    train = [str(i) for i in range(74)]
    rew = {t: 0.0 for t in train[:40]} | {t: 1.0 for t in train[40:]}
    # 8 failed tasks with a denied non-gold write; 4 passed tasks with a denied gold write -> G 8, L 4, G - L 4: GO
    rows = [_oof(t, 1, 1, persist=int(t in ("0", "1"))) for t in train[:8]]
    rows += [_oof(t, 0, 1) for t in train[40:44]]
    rows += [_oof("9", 0, 1), _oof("50", 1, 1), _oof("60", 1, 0)]   # neither G nor L (wrong label/outcome pairs)
    r = GB.go_bar(rows, rew, train)
    tm = r["models"]["tm_seed1"]
    assert (tm["G"], tm["L"], tm["G_minus_L"], r["GO"], r["outcome"]) == (8, 4, 4, True, "go")
    assert tm["learned_denials"] == 14 and tm["persist_after_reconfirm"] == 2
    r = GB.go_bar(rows + [_oof("44", 0, 1)], rew, train)                # L 5: G - L 3 -> NO-GO
    assert r["models"]["tm_seed1"]["L"] == 5 and r["GO"] is False and r["outcome"] == "no-go"
    r = GB.go_bar([x for x in rows if x["task"] != "7"], rew, train)    # G 7 -> NO-GO
    assert r["models"]["tm_seed1"]["G"] == 7 and r["GO"] is False
    r = GB.go_bar(rows + [_oof("0", 1, 1)], rew, train)                  # a second denial on a counted task: still 8
    assert r["models"]["tm_seed1"]["G"] == 8


def test_reconfirm_flips_only_the_confirmation_bits_and_keeps_the_domain():
    from e4guard import facts as F

    st = F.FactState()
    for e in [{"kind": "agent_text", "text": "Hi!", "with_calls": False}, {"kind": "user_text", "text": "ann@x.com"},
              {"kind": "agent_call", "id": "a", "name": "find_user_id_by_email", "args": {"email": "ann@x.com"}},
              {"kind": "tool_result", "id": "a", "error": False, "content": "ann_1"},
              {"kind": "agent_call", "id": "b", "name": "get_order_details", "args": {"order_id": "#W1000001"}},
              {"kind": "tool_result", "id": "b", "error": False, "content": json.dumps({**ORDER, "status": "delivered"})},
              {"kind": "agent_text", "text": "Return the lamp? Shall I?", "with_calls": False},
              {"kind": "user_text", "text": "Hmm, not sure."}]:
        st.update(e)
    f = st.facts_for_call("return_delivered_order_items", {"order_id": "#W1000001", "item_ids": ["1111111111"],
                                                           "payment_method_id": "credit_card_1"})
    x = B.encode(f)
    y = B.reconfirm(x)
    changed = {B.NAMES[i] for i in np.flatnonzero(x != y)}
    assert changed <= {"user_yes", "user_negation", "action_word_listed", "order_id_listed", "items_listed",
                       "new_items_listed", "user_turns_ge2"}
    assert y[B.IDX["user_yes"]] == 1 and y[B.IDX["user_negation"]] == 0 and y[B.IDX["order_id_listed"]] == 1
    assert B.rows_hold(B.domain_rows(), np.stack([x, y])).all()
    assert "E4-R3a" in B.veto_fires(x) and not ({"E4-R3a", "E4-R3b"} & set(B.veto_fires(y)))
    r = B.encode(st.facts_for_call("get_order_details", {"order_id": "#W1000001"}))
    assert (B.reconfirm(r) == r).all()                                     # reads are unchanged
