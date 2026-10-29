"""Fact record -> named bits; the veto clauses (one or two per written rule); the MILP domain rows.

Pure Python + numpy (numpy only for vectors). Every bit has a readable name, so a clause reads as a rule.
The veto clauses are written here from the rule texts, independently of rules.py; their equality with the code
gates is checked exhaustively over the rule bits (scripts/check_equivalence.py) and their effect on a trained
model by MILP (scripts/milp_proof.py, verify_policy.build_and_solve).
"""

from __future__ import annotations

from . import facts as F

TOOL_BITS = [f"tool={t}" for t in F.RETAIL_TOOLS] + ["tool=__unknown__"]
CLASS_BITS = ["is_write", "is_order_write", "needs_auth", "needs_pending", "needs_delivered", "is_item_change",
              "uses_payment", "needs_oid_confirm"]
STATUS_VALUES = F.STATUSES + ["__other__", "__unknown__", "__no_order__"]
STATUS_BITS = [f"status={s}" for s in STATUS_VALUES]
RULE_FACT_BITS = ["has_order_arg", "authenticated", "uid_mismatch", "order_owner_mismatch", "order_id_listed", "user_yes",
                  "status_read", "item_change_done", "reason_valid", "new_items_differ", "pay_in_profile",
                  "pay_is_original", "pay_is_giftcard"]
EXTRA_BITS = ["second_user_seen", "action_word_listed", "items_listed", "new_items_listed", "order_in_user_text",
              "order_in_profile", "items_in_order", "new_items_seen_in_tools", "pay_seen_in_tools", "same_write_done",
              "order_written_before", "writes_ge1", "writes_ge2", "writes_ge3", "user_turns_ge2", "user_turns_ge4",
              "user_turns_ge8", "last_result_error", "user_negation", "intent_match", "user_said_cancel",
              "user_said_return", "user_said_exchange", "user_said_modify", "user_said_address", "user_said_payment",
              "n_items_ge2"]
NAMES = TOOL_BITS + CLASS_BITS + STATUS_BITS + RULE_FACT_BITS + EXTRA_BITS
IDX = {n: i for i, n in enumerate(NAMES)}
N_BITS = len(NAMES)

# the veto clauses: clause id -> (rule, literals); a literal is (bit name, True) for x or (bit name, False) for NOT x.
# Names carry the prefix "E4-" (no clash with the decision layer's R9b, R25, ...). E4-R3a is the text-literal reading
# (revision 2): the order id is required only before a cancel or a return (needs_oid_confirm, policy.md:90,120).
VETO = {
    "E4-R1":  ("E4-R1", [("needs_auth", True), ("authenticated", False)]),
    "E4-R2a": ("E4-R2", [("uid_mismatch", True)]),
    "E4-R2b": ("E4-R2", [("order_owner_mismatch", True)]),
    "E4-R3a": ("E4-R3", [("needs_oid_confirm", True), ("order_id_listed", False)]),
    "E4-R3b": ("E4-R3", [("is_write", True), ("user_yes", False)]),
    "E4-R4":  ("E4-R4", [("is_order_write", True), ("status_read", False)]),
    "E4-R5":  ("E4-R5", [("needs_pending", True), ("status=pending", False)]),
    "E4-R6":  ("E4-R6", [("needs_delivered", True), ("status=delivered", False)]),
    "E4-R7":  ("E4-R7", [("is_item_change", True), ("item_change_done", True)]),
    "E4-R8":  ("E4-R8", [("tool=cancel_pending_order", True), ("reason_valid", False)]),
    "E4-R9a": ("E4-R9", [("uses_payment", True), ("pay_in_profile", False)]),
    "E4-R9b": ("E4-R9", [("tool=return_delivered_order_items", True), ("pay_is_original", False),
                         ("pay_is_giftcard", False)]),
    "E4-R10": ("E4-R10", [("is_item_change", True), ("new_items_differ", False)]),
}
ITEM_TOOLS = ["exchange_delivered_order_items", "modify_pending_order_items", "return_delivered_order_items"]


def encode(f: dict):
    """fact record -> uint8 vector [N_BITS]."""
    import numpy as np

    x = np.zeros(N_BITS, np.uint8)
    x[IDX[f"tool={f['tool']}"]] = 1
    for b in CLASS_BITS:
        x[IDX[b]] = int(bool(f[b]))
    x[IDX[f"status={f['status']}"]] = 1
    for b in RULE_FACT_BITS + EXTRA_BITS:
        x[IDX[b]] = int(bool(f[b]))
    return x


def literal_index(name: str, positive: bool) -> int:
    """Index in the literal vector L = [x, 1 - x] (verify_policy's convention: i < F is x_i, F + i is NOT x_i)."""
    return IDX[name] if positive else N_BITS + IDX[name]


def veto_literals() -> dict:
    return {cid: [literal_index(n, p) for n, p in lits] for cid, (_, lits) in VETO.items()}


def veto_fires(x) -> list[str]:
    """Clause ids whose literals are all true on bit vector x."""
    out = []
    for cid, (_, lits) in VETO.items():
        if all(bool(x[IDX[n]]) == p for n, p in lits):
            out.append(cid)
    return out


def veto_rules(x) -> list[str]:
    return sorted({VETO[c][0] for c in veto_fires(x)})


# --------------------------------------------------------------------------------------------- the MILP domain
def domain_rows() -> list[tuple[dict, float, float]]:
    """Linear rows lo <= sum(coef * x) <= hi that every real fact vector satisfies (a superset of real inputs).
    Each row is justified in its comment by how facts.FactState computes the bits."""
    inf = float("inf")
    I = IDX
    rows = []

    def tools(names):
        return {I[f"tool={t}"]: 1 for t in names}

    rows.append(({I[b]: 1 for b in TOOL_BITS}, 1, 1))                       # exactly one tool
    rows.append(({I[b]: 1 for b in STATUS_BITS}, 1, 1))                     # exactly one status value
    # tool-class bits are functions of the tool
    for b, members in [("is_write", F.WRITE_TOOLS), ("is_order_write", F.ORDER_WRITE_TOOLS),
                       ("needs_pending", F.NEEDS_PENDING), ("needs_delivered", F.NEEDS_DELIVERED),
                       ("is_item_change", F.ITEM_CHANGE_TOOLS), ("uses_payment", F.PAYMENT_TOOLS),
                       ("needs_oid_confirm", F.NEEDS_OID_CONFIRM)]:
        rows.append(({I[b]: 1, **{k: -1 for k in tools(members)}}, 0, 0))
    rows.append(({I["needs_auth"]: 1, **tools(F.NO_AUTH_NEEDED)}, 1, 1))     # needs_auth = NOT (auth or transfer)
    rows.append(({I["status=__no_order__"]: 1, I["has_order_arg"]: 1}, 1, 1))  # no order arg <-> status none
    le = lambda a, b: rows.append(({I[a]: 1, I[b]: -1}, -inf, 0))          # a implies b
    le("uid_mismatch", "authenticated")
    le("order_owner_mismatch", "authenticated")
    le("order_owner_mismatch", "has_order_arg")
    le("second_user_seen", "authenticated")
    le("status_read", "has_order_arg")
    rows.append(({I["status_read"]: 1, I["status=__unknown__"]: 1, I["status=__no_order__"]: 1}, -inf, 1))
    rows.append(({I["order_owner_mismatch"]: 1, I["status=__unknown__"]: 1, I["status=__no_order__"]: 1}, -inf, 1))
    le("item_change_done", "has_order_arg")
    le("item_change_done", "order_written_before")
    le("reason_valid", "tool=cancel_pending_order")
    le("new_items_differ", "is_item_change")
    le("order_id_listed", "is_order_write")
    le("order_id_listed", "has_order_arg")
    le("action_word_listed", "is_write")
    le("intent_match", "is_write")
    for b in ("pay_in_profile", "pay_is_original", "pay_is_giftcard", "pay_seen_in_tools"):
        le(b, "uses_payment")
    le("pay_is_giftcard", "pay_in_profile")
    for b in ("items_listed", "items_in_order", "n_items_ge2"):
        rows.append(({I[b]: 1, **{k: -1 for k in tools(ITEM_TOOLS)}}, -inf, 0))
    for b in ("new_items_listed", "new_items_seen_in_tools"):
        le(b, "is_item_change")
    for b in ("order_in_user_text", "order_in_profile", "order_written_before"):
        le(b, "has_order_arg")
    le("order_in_profile", "authenticated")
    le("same_write_done", "is_write")
    le("same_write_done", "writes_ge1")
    le("order_written_before", "writes_ge1")
    le("writes_ge2", "writes_ge1")
    le("writes_ge3", "writes_ge2")
    le("user_turns_ge4", "user_turns_ge2")
    le("user_turns_ge8", "user_turns_ge4")
    rows.append(({I["user_yes"]: 1, I["user_negation"]: 1}, -inf, 1))      # a yes is never a refusal
    return rows


def rows_hold(rows, X):
    """[n] bool: every domain row holds for each bit vector (same test as verify_policy.rows_hold)."""
    import numpy as np

    Xf = np.asarray(X).astype(np.int64)
    ok = np.ones(len(Xf), bool)
    for coef, lo, hi in rows:
        v = sum(c * Xf[:, i] for i, c in coef.items())
        ok &= (v >= lo) & (v <= hi)
    return ok


def reconfirm(x):
    """A write's bit vector after a SIMULATED re-confirmation (review issue 4; fixed in revision 2, before any Qwen row).
    A learned denial tells the agent to re-check the ids and the payment and to confirm the exact action again
    (models._denial). Simulated: the agent re-lists the action naming its order id, items and new items, and the user
    answers with an explicit yes. Set: user_yes = 1, user_negation = 0, action_word_listed = 1 (writes),
    order_id_listed = 1 (order writes with an order argument), items_listed = 1 (item tools), new_items_listed = 1
    (item changes), user_turns_ge2 = 1 (one more user turn; ge4 and ge8 are left as they are). Every other bit (the
    call, its arguments, the tool outputs, the history) is unchanged. The result satisfies the domain rows whenever x
    does. Used only for a reported share (no rule reads it)."""
    import numpy as np

    y = np.array(x, np.uint8).copy()
    w = bool(y[IDX["is_write"]])
    if not w:
        return y
    y[IDX["user_yes"]], y[IDX["user_negation"]], y[IDX["action_word_listed"]] = 1, 0, 1
    if y[IDX["is_order_write"]] and y[IDX["has_order_arg"]]:
        y[IDX["order_id_listed"]] = 1
    if any(y[IDX[f"tool={t}"]] for t in ITEM_TOOLS):
        y[IDX["items_listed"]] = 1
    if y[IDX["is_item_change"]]:
        y[IDX["new_items_listed"]] = 1
    y[IDX["user_turns_ge2"]] = 1
    return y


def decode_rule_facts(x) -> dict:
    """A bit vector -> the fact record fields that rules.violations reads (for the exhaustive equivalence check)."""
    tool = next(n.split("=", 1)[1] for n in TOOL_BITS if x[IDX[n]])
    status = next(n.split("=", 1)[1] for n in STATUS_BITS if x[IDX[n]])
    f = {"tool": tool, "status": status}
    for b in CLASS_BITS + RULE_FACT_BITS + EXTRA_BITS:
        f[b] = bool(x[IDX[b]])
    return f
