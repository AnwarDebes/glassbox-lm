"""Unit tests for the E4 fact extractor, the yes-matcher and the code gates (pure Python; no tau2 needed).

Hand-written cases (not taken from any trajectory). Run:
  .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider <this folder>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import sys

sys.path.insert(0, f"{_REPO_ROOT}/studies/agent_guards/e4")

import pytest  # noqa: E402

from e4guard import facts as F  # noqa: E402
from e4guard import rules  # noqa: E402

YES = [
    "Yes", "yes.", "Yes, please go ahead.", "Yeah, go ahead and switch it.", "Yep!", "yup", "Sure, go ahead.",
    "OK", "Okay, proceed.", "Correct.", "That's correct, please proceed.", "That’s right.", "Everything is correct.",
    "Please go ahead.", "Go ahead and cancel it.", "Sounds good, do it.", "I confirm.", "Confirmed.",
    "No, that's all. Please go ahead.", "No problem, go ahead.", "Alright, let's do it.", "Absolutely.",
    "Yes, I'd like to return both items to the original card.", "Perfect, thank you. Please proceed with the exchange.",
    "Yes, that's correct. Also, I'm not sure about the lamp yet, can you tell me my options later?",
    "\u201cNo longer needed.\u201d Yes, please cancel it.", "That's fine, use the cheaper one.",
]
NOT_YES = [
    "", "No.", "No, wait. Don't do that yet.", "Nope", "Not yet.", "Wait, I need to check something.",
    "Hmm, let me think about it.", "Actually, no.", "Don't cancel it.", "That's not correct.", "That's wrong.",
    "Yes, but before you proceed, can you check the price?", "I'm not sure.", "Please cancel it.",
    "I want to make sure the address is right.", "Is that ok?", "Can you confirm the price first?",
    "Hold on a second.", "I changed my mind.", "Never mind.", "The order id is #W1234567.",
    "Hmm, I'm not sure, can I go ahead later?", "Yes, but don't process it yet.",
]


@pytest.mark.parametrize("t", YES)
def test_yes(t):
    assert F.is_explicit_yes(t), t


@pytest.mark.parametrize("t", NOT_YES)
def test_not_yes(t):
    assert not F.is_explicit_yes(t), t


def test_order_id_match():
    assert F.order_id_in_text("#W2378156", "I will cancel order #W2378156 now.")
    assert F.order_id_in_text("#W2378156", "order W2378156, ok?")
    assert F.order_id_in_text("#W2378156", "order #w2378156")
    assert not F.order_id_in_text("#W2378156", "order #W23781567")
    assert not F.order_id_in_text("#W2378156", "order #W237815")
    assert not F.order_id_in_text("#W2378156", "")
    assert F.token_in_text("10083", "New York, NY 10083.")
    assert not F.token_in_text("10083", "zip 100830")


def _ev(*items):
    out, n = [], 0
    for it in items:
        if it[0] == "a":
            out.append({"kind": "agent_text", "text": it[1], "with_calls": False})
        elif it[0] == "u":
            out.append({"kind": "user_text", "text": it[1]})
        elif it[0] == "c":
            n += 1
            out.append({"kind": "agent_call", "id": f"c{n}", "name": it[1], "args": it[2]})
            out.append({"kind": "tool_result", "id": f"c{n}", "error": it[4] if len(it) > 4 else False,
                        "content": it[3] if isinstance(it[3], str) else json.dumps(it[3])})
    return out


ORDER = {"order_id": "#W1000001", "user_id": "ann_1", "status": "pending", "items": [{"item_id": "1111111111"}],
         "payment_history": [{"transaction_type": "payment", "amount": 10.0, "payment_method_id": "credit_card_1"}]}
USER = {"user_id": "ann_1", "payment_methods": {"credit_card_1": {"source": "credit_card"},
                                                "gift_card_9": {"source": "gift_card"}}, "orders": ["#W1000001"]}


def state_of(events):
    st = F.FactState()
    for e in events:
        st.update(e)
    return st


def test_rules_happy_path_and_each_rule():
    base = _ev(("a", "Hi! How can I help you today?"), ("u", "My email is ann@x.com"),
               ("c", "find_user_id_by_email", {"email": "ann@x.com"}, "ann_1"),
               ("c", "get_user_details", {"user_id": "ann_1"}, USER),
               ("c", "get_order_details", {"order_id": "#W1000001"}, ORDER),
               ("a", "I will cancel order #W1000001, reason 'no longer needed'. Do you confirm?"), ("u", "Yes."))
    st = state_of(base)
    ok = {"order_id": "#W1000001", "reason": "no longer needed"}
    assert rules.violations(st.facts_for_call("cancel_pending_order", ok)) == []
    # R8
    assert rules.rule_ids(st.facts_for_call("cancel_pending_order", {**ok, "reason": "too expensive"})) == ["E4-R8"]
    # R2: another user's id
    assert rules.rule_ids(st.facts_for_call("get_user_details", {"user_id": "bob_2"})) == ["E4-R2"]
    # R1: nothing but lookups / transfer before authentication
    st0 = state_of(base[:2])
    assert rules.rule_ids(st0.facts_for_call("get_order_details", {"order_id": "#W1000001"})) == ["E4-R1"]
    assert rules.violations(st0.facts_for_call("find_user_id_by_email", {"email": "a@b.c"})) == []
    assert rules.violations(st0.facts_for_call("transfer_to_human_agents", {"summary": "x"})) == []
    # R3: the listing names another order; the user did not say yes
    st3 = state_of(base[:-2] + _ev(("a", "Cancel order #W9999999? Do you confirm?"), ("u", "Yes.")))
    assert rules.rule_ids(st3.facts_for_call("cancel_pending_order", ok)) == ["E4-R3"]
    st3b = state_of(base[:-1] + _ev(("u", "No, wait.")))
    assert rules.rule_ids(st3b.facts_for_call("cancel_pending_order", ok)) == ["E4-R3"]
    # R4/R5: no status read -> status unknown
    st4 = state_of(base[:4] + base[-2:])
    assert set(rules.rule_ids(st4.facts_for_call("cancel_pending_order", ok))) == {"E4-R4", "E4-R5"}
    # R6: return of a pending order
    ret = {"order_id": "#W1000001", "item_ids": ["1111111111"], "payment_method_id": "credit_card_1"}
    assert rules.rule_ids(st.facts_for_call("return_delivered_order_items", ret)) == ["E4-R6"]
    # R9: payment method not in the profile; refund to a non-original non-gift card
    ex = {"order_id": "#W1000001", "item_ids": ["1111111111"], "new_item_ids": ["2222222222"],
          "payment_method_id": "paypal_7"}
    assert "E4-R9" in rules.rule_ids(st.facts_for_call("modify_pending_order_items", ex))
    assert rules.violations(st.facts_for_call("modify_pending_order_items", {**ex, "payment_method_id": "gift_card_9"})) == []
    # R10: modify an item into itself
    same = {**ex, "payment_method_id": "credit_card_1", "new_item_ids": ["1111111111"]}
    assert rules.rule_ids(st.facts_for_call("modify_pending_order_items", same)) == ["E4-R10"]


def test_status_tracking_after_writes_r5_r7():
    after = dict(ORDER, status="pending (item modified)")
    ev = _ev(("a", "Hi!"), ("u", "ann@x.com"), ("c", "find_user_id_by_email", {"email": "ann@x.com"}, "ann_1"),
             ("c", "get_order_details", {"order_id": "#W1000001"}, ORDER),
             ("a", "Modify order #W1000001? Do you confirm?"), ("u", "yes"),
             ("c", "modify_pending_order_items", {"order_id": "#W1000001", "item_ids": ["1111111111"],
                                                   "new_item_ids": ["2222222222"], "payment_method_id": "credit_card_1"},
              after),
             ("a", "Change the address of order #W1000001 to zip 73301? Do you confirm?"), ("u", "yes"))
    st = state_of(ev)
    addr = {"order_id": "#W1000001", "address1": "1 A St", "address2": "", "city": "X", "state": "TX",
            "country": "USA", "zip": "73301"}
    assert rules.rule_ids(st.facts_for_call("modify_pending_order_address", addr)) == ["E4-R5"]
    again = {"order_id": "#W1000001", "item_ids": ["2222222222"], "new_item_ids": ["3333333333"],
             "payment_method_id": "credit_card_1"}
    assert set(rules.rule_ids(st.facts_for_call("modify_pending_order_items", again))) >= {"E4-R5", "E4-R7"}


def test_multi_call_message_is_sequential():
    """A read and a write in ONE model message: after sequentialize, the write sees the read's result."""
    msgs = [
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "ann@x.com"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "name": "find_user_id_by_email", "arguments": {"email": "ann@x.com"}},
            {"id": "b", "name": "get_order_details", "arguments": {"order_id": "#W1000001"}}]},
        {"role": "tool", "id": "a", "content": "ann_1"},
        {"role": "tool", "id": "b", "content": json.dumps(ORDER)},
    ]
    seq = list(F.replay_facts(F.events_from_message_dicts(msgs), sequential=True))
    assert seq[1][2]["authenticated"] is True and rules.violations(seq[1][2]) == []
    raw = list(F.replay_facts(F.events_from_message_dicts(msgs), sequential=False))
    assert raw[1][2]["authenticated"] is False and rules.rule_ids(raw[1][2]) == ["E4-R1"]


def test_text_with_calls_is_not_shown_to_user():
    msgs = [{"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "ann@x.com"},
            {"role": "assistant", "content": "Cancel #W1000001? Do you confirm?",
             "tool_calls": [{"id": "a", "name": "find_user_id_by_email", "arguments": {"email": "ann@x.com"}}]},
            {"role": "tool", "id": "a", "content": "ann_1"}]
    st = state_of(F.sequentialize(F.events_from_message_dicts(msgs)))
    assert st.last_agent_text == "Hi!"


# ---------------------------------------------------------------------------------- revision 2: text-literal E4-R3
def _base_state(listing, reply="Yes."):
    ev = _ev(("a", "Hi! How can I help you today?"), ("u", "My email is ann@x.com"),
             ("c", "find_user_id_by_email", {"email": "ann@x.com"}, "ann_1"),
             ("c", "get_user_details", {"user_id": "ann_1"}, USER),
             ("c", "get_order_details", {"order_id": "#W1000001"}, ORDER_D),
             ("c", "get_order_details", {"order_id": "#W1000002"}, ORDER_P),
             ("a", listing), ("u", reply))
    return state_of(ev)


ORDER_D = {"order_id": "#W1000001", "user_id": "ann_1", "status": "delivered", "items": [{"item_id": "1111111111"}],
           "payment_history": [{"transaction_type": "payment", "amount": 10.0, "payment_method_id": "credit_card_1"}]}
ORDER_P = {"order_id": "#W1000002", "user_id": "ann_1", "status": "pending", "items": [{"item_id": "1111111111"}],
           "payment_history": [{"transaction_type": "payment", "amount": 10.0, "payment_method_id": "credit_card_1"}]}
EXCH = {"order_id": "#W1000001", "item_ids": ["1111111111"], "new_item_ids": ["2222222222"],
        "payment_method_id": "credit_card_1"}
RET = {"order_id": "#W1000001", "item_ids": ["1111111111"], "payment_method_id": "credit_card_1"}
ADDR = {"order_id": "#W1000002", "address1": "1 A St", "address2": "", "city": "X", "state": "TX", "country": "USA",
        "zip": "73301"}


def test_r3_text_literal_exchange_needs_no_order_id_but_strict_monitor_fires():
    st = _base_state("I will exchange your blue lamp for the red one, paid with your card. Shall I go ahead?")
    f = st.facts_for_call("exchange_delivered_order_items", EXCH)
    assert rules.violations(f) == []                         # enforced reading: allowed
    assert f["order_id_listed"] is False and rules.strict_r3a_fires(f) is True   # monitor only


def test_r3_text_literal_return_and_cancel_need_the_order_id():
    st = _base_state("I will return the blue lamp to your card. Shall I go ahead?")
    assert rules.rule_ids(st.facts_for_call("return_delivered_order_items", RET)) == ["E4-R3"]
    st2 = _base_state("I will return the blue lamp of order #W1000001 to your card. Shall I go ahead?")
    f2 = st2.facts_for_call("return_delivered_order_items", RET)
    assert rules.violations(f2) == [] and f2["order_id_listed"] and not rules.strict_r3a_fires(f2)
    st3 = _base_state("I will cancel your pending order, reason 'no longer needed'. OK?")
    assert rules.rule_ids(st3.facts_for_call("cancel_pending_order", {"order_id": "#W1000002",
                                                                      "reason": "no longer needed"})) == ["E4-R3"]


def test_r3_text_literal_no_zip_check_on_address_writes():
    st = _base_state("I will change the shipping address of order #W1000002 to 1 A St, X, TX. Confirm?")
    f = st.facts_for_call("modify_pending_order_address", ADDR)
    assert rules.violations(f) == []                         # no zip in the listing: allowed now
    assert rules.strict_r3a_fires(f) is True                 # revision 1 would have denied it (monitor only)
    ua = {k: v for k, v in ADDR.items() if k != "order_id"}
    fu = st.facts_for_call("modify_user_address", {**ua, "user_id": "ann_1"})
    assert rules.violations(fu) == [] and fu["order_id_listed"] is False and rules.strict_r3a_fires(fu)


def test_r3_yes_must_follow_a_text_only_agent_message():
    st = state_of(_ev(("u", "yes"), ("c", "find_user_id_by_email", {"email": "ann@x.com"}, "ann_1"),
                      ("c", "get_order_details", {"order_id": "#W1000002"}, ORDER_P)))
    f = st.facts_for_call("modify_pending_order_address", ADDR)
    assert f["user_yes"] is False and "E4-R3" in rules.rule_ids(f)
    st2 = _base_state("Change the address of order #W1000002?", reply="No, wait.")
    assert "E4-R3" in rules.rule_ids(st2.facts_for_call("modify_pending_order_address", ADDR))


def test_non_string_order_id_never_crashes():
    st = _base_state("Exchange order 1000001?")
    f = st.facts_for_call("exchange_delivered_order_items", {**EXCH, "order_id": 1000001})
    assert f["order_id_listed"] is False and f["has_order_arg"] is False
    assert "E4-R4" in rules.rule_ids(f)                      # no status read for a non-string id


def test_veto_clauses_equal_code_on_the_new_r3_cases():
    from e4guard import bits as B

    for listing, tool, args in [("Exchange the lamp?", "exchange_delivered_order_items", EXCH),
                                ("Return the lamp?", "return_delivered_order_items", RET),
                                ("Return the lamp of #W1000001?", "return_delivered_order_items", RET),
                                ("New address 1 A St for #W1000002?", "modify_pending_order_address", ADDR)]:
        for reply in ("Yes.", "No, wait."):
            f = _base_state(listing, reply).facts_for_call(tool, args)
            assert rules.rule_ids(f) == B.veto_rules(B.encode(f)), (listing, tool, reply)


def test_r3_a_yes_in_the_opening_request_is_not_a_confirmation():
    """policy.md:16: the yes must answer a listing; tau2's greeting precedes the user's request and never counts."""
    ev = _ev(("a", "Hi! How can I help you today?"),
             ("u", "Hi, I'm Ann. Sure, it's fine to use my card: please change my order address to 1 A St."),
             ("c", "find_user_id_by_email", {"email": "ann@x.com"}, "ann_1"),
             ("c", "get_order_details", {"order_id": "#W1000002"}, ORDER_P))
    st = state_of(ev)
    assert F.is_explicit_yes("Hi, I'm Ann. Sure, it's fine to use my card: please change my order address to 1 A St.")
    f = st.facts_for_call("modify_pending_order_address", ADDR)
    assert f["user_yes"] is False and "E4-R3" in rules.rule_ids(f)
    st2 = state_of(ev + _ev(("a", "I will change the address of order #W1000002 to 1 A St, X, TX 73301. OK?"),
                            ("u", "Yes.")))
    assert st2.facts_for_call("modify_pending_order_address", ADDR)["user_yes"] is True
