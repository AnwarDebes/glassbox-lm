"""Trusted facts for the E4 retail guard (tau2-bench v1.0.1 retail). Pure Python: no tau2, no numpy.

The harness, not a model, computes every fact, from three trusted sources only:
  - the tool-call log: which calls ran, their arguments, whether the environment returned an error, and the
    structured JSON the retail tools returned (tau2 has no attacker, so database results are trusted here);
  - the authenticated user id: the user id returned by the first successful find_user_id_by_email /
    find_user_id_by_name_zip;
  - the conversation turns: the user's messages (the user is the principal) and the text the agent last showed
    to the user (a record of what the user saw, not a claim the guard believes).
A fact never comes from an LLM's words about the world (e.g. "the order is pending" in the agent's text is ignored).

Input is a list of neutral events (see `events_from_message_dicts`), so the same code serves:
  live tau2 runs (agent wrapper), scripted offline episodes, and saved trajectories (results.json files).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------------------------------------------
# the retail tool set (tau2 v1.0.1 src/tau2/domains/retail/tools.py; READ/WRITE types from @is_tool)
# ----------------------------------------------------------------------------------------------------------------
RETAIL_TOOLS = [
    "calculate", "cancel_pending_order", "exchange_delivered_order_items", "find_user_id_by_email",
    "find_user_id_by_name_zip", "get_item_details", "get_order_details", "get_product_details", "get_user_details",
    "list_all_product_types", "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items", "transfer_to_human_agents",
]
WRITE_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}
ORDER_WRITE_TOOLS = WRITE_TOOLS - {"modify_user_address"}
AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
NO_AUTH_NEEDED = AUTH_TOOLS | {"transfer_to_human_agents"}          # R1's exceptions
NEEDS_PENDING = {"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
                 "modify_pending_order_payment"}                    # R5
NEEDS_DELIVERED = {"return_delivered_order_items", "exchange_delivered_order_items"}  # R6
ITEM_CHANGE_TOOLS = {"exchange_delivered_order_items", "modify_pending_order_items"}  # R7
PAYMENT_TOOLS = {"exchange_delivered_order_items", "modify_pending_order_items", "modify_pending_order_payment",
                 "return_delivered_order_items"}                    # R9
ADDRESS_TOOLS = {"modify_pending_order_address", "modify_user_address"}
# E4-R3a (text-literal reading, revision 2): the user confirms THE ORDER ID only where the policy says so:
# cancel (policy.md:90) and return (policy.md:120). Exchange (:128-136) and modify (:94-114) have no order-id wording.
NEEDS_OID_CONFIRM = {"cancel_pending_order", "return_delivered_order_items"}
VALID_CANCEL_REASONS = {"no longer needed", "ordered by mistake"}   # policy.md:90, tools.py:181
STATUSES = ["pending", "pending (item modified)", "processed", "delivered", "cancelled", "return requested",
            "exchange requested"]                                    # all statuses the retail tools can produce
ORDER_RETURNING_TOOLS = {"get_order_details"} | ORDER_WRITE_TOOLS   # tools whose output is an Order

# ----------------------------------------------------------------------------------------------------------------
# text helpers: the explicit-yes matcher and the id-listing check (ordinary code; tests in tests/test_facts.py)
# ----------------------------------------------------------------------------------------------------------------
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": " - "})

# strong affirmative cues count anywhere in the message; weak ones only when the message opens with them
YES_STRONG = [
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\byup\b", r"\bgo ahead\b", r"\bplease do\b", r"\bdo it\b",
    r"\bdo that\b", r"\b(please|you can|you may|ok,? |okay,? )proceed\b", r"\bproceed with\b", r"\bi (can |do )?confirm\b",
    r"\bconfirmed\b", r"\b(that'?s|that is|it'?s|it is|this is|everything'?s|everything is|all|both are|they are)( all| both)? "
    r"(correct|right)\b", r"\bsounds (good|great|perfect|right)\b", r"\blet'?s do (it|that|this)\b",
    r"\b(that|this) works\b", r"\bi agree\b", r"\bplease (make|process|submit) (the|those|these|that|this|it|them)\b",
    r"\b(that'?s|that is|it'?s|it is|this is) (fine|ok|okay|good|great|perfect)\b",
]
# an explicit yes at the very start of the message (then only a hard refusal can undo it)
YES_OPENING = re.compile(r"^\W*(yes|yeah|yep|yup|correct|that'?s correct|that is correct|confirmed|i confirm)\b")
YES_WEAK_START = re.compile(r"^\W*(sure|ok|okay|alright|all right|correct|exactly|perfect|absolutely|definitely|"
                            r"right|great|confirm|proceed|affirmative)\b")
# a message that opens with one of these is not a yes ("No, ...", "Wait, ...", "Actually no", "Don't ...")
NEG_START = re.compile(r"^\W*(no\b|nope\b|not\b|wait\b|hold on\b|hang on\b|actually,? no\b|don'?t\b|do not\b|"
                       r"never ?mind\b|stop\b|cancel that\b)")
# "No, that's all/everything" (answers "anything else?") and "no problem" are not refusals of the listed action
NO_THATS_ALL = re.compile(r"^\W*no\W+((that'?s|that is|it'?s|nothing)\W*(all|everything|it|else)\b|problem\b|worries\b|"
                          r"longer needed\b)")
NEG_ANYWHERE = [
    r"\bdon'?t (do|proceed|go ahead|make|process|change|cancel)\b", r"\bdo not (do|proceed|go ahead|make|process)\b",
    r"\bnot yet\b", r"\bhold off\b", r"\bwait a (moment|minute|second)\b", r"\bnot (correct|right)\b",
    r"\bthat'?s (wrong|incorrect|not right)\b", r"\bincorrect\b", r"\bisn'?t (correct|right)\b",
    r"\bchanged my mind\b", r"\bbefore you (proceed|do|go ahead)\b",
]
# soft doubt: rejects a message only when it does not OPEN with an explicit yes (doubt about something else is common)
NEG_SOFT = [r"\bnot sure\b", r"\bunsure\b", r"\bi'?m not certain\b"]
_YES_RE = [re.compile(p) for p in YES_STRONG]
_NEG_RE = [re.compile(p) for p in NEG_ANYWHERE]
_SOFT_RE = [re.compile(p) for p in NEG_SOFT]


def normalize_text(s: str | None) -> str:
    return (s or "").translate(_QUOTES).lower().strip()


def is_explicit_yes(text: str | None) -> bool:
    """True when the user's message is an explicit confirmation. Rule-based on purpose (a fixed, testable function):
    a strong affirmative phrase anywhere or a weak one at the start; no refusal at the start (except "no, that's all"
    and "no problem"); no refusal phrase anywhere. A bare instruction ("please cancel it") is NOT counted."""
    t = normalize_text(text)
    if not t:
        return False
    if NEG_START.search(t) and not NO_THATS_ALL.search(t):
        return False
    if any(r.search(t) for r in _NEG_RE):
        return False
    if any(r.search(t) for r in _SOFT_RE) and not YES_OPENING.search(t):
        return False
    return any(r.search(t) for r in _YES_RE) or bool(YES_WEAK_START.search(t))


def has_negation(text: str | None) -> bool:
    t = normalize_text(text)
    return bool(t) and ((bool(NEG_START.search(t)) and not NO_THATS_ALL.search(t)) or any(r.search(t) for r in _NEG_RE)
                        or (any(r.search(t) for r in _SOFT_RE) and not YES_OPENING.search(t)))


def order_id_in_text(order_id: str, text: str | None) -> bool:
    """'#W2378156' is listed if the text contains '#W2378156' or 'W2378156' (case-insensitive, whole token)."""
    if not order_id or not text:
        return False
    core = re.escape(order_id.lstrip("#"))
    return re.search(r"(?<![A-Za-z0-9])#?" + core + r"(?![0-9])", text, re.IGNORECASE) is not None


def token_in_text(tok: str, text: str | None) -> bool:
    """An id-like token (item id, zip, payment method id) appears as a whole token."""
    if not tok or not text:
        return False
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(tok)) + r"(?![A-Za-z0-9_])", text) is not None


# intent words in the user's own messages (the principal), per write type; a fixed list, not learned
INTENT_WORDS = {
    "cancel": r"\bcancel",
    "return": r"\b(return|refund)",
    "exchange": r"\b(exchange|swap)",
    "modify": r"\b(modify|change|update|switch|replace)",
    "address": r"\b(address|ship|shipping|deliver to|moved?)\b",
    "payment": r"\b(payment|pay|card|paypal|gift card|credit)\b",
}
ACTION_VERBS = {   # the verb of the action in the agent's listing (a learned-part fact, not a rule)
    "cancel_pending_order": r"\bcancel",
    "return_delivered_order_items": r"\b(return|refund)",
    "exchange_delivered_order_items": r"\b(exchange|swap)",
    "modify_pending_order_items": r"\b(modify|change|update|switch|replace|exchange)",
    "modify_pending_order_address": r"\b(address|ship)",
    "modify_user_address": r"\b(address)",
    "modify_pending_order_payment": r"\b(payment|pay|card|paypal|gift card)",
}
TOOL_INTENT = {
    "cancel_pending_order": ["cancel"],
    "return_delivered_order_items": ["return"],
    "exchange_delivered_order_items": ["exchange", "modify"],
    "modify_pending_order_items": ["modify", "exchange"],
    "modify_pending_order_address": ["address"],
    "modify_user_address": ["address"],
    "modify_pending_order_payment": ["payment"],
}


# ----------------------------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------------------------
def events_from_message_dicts(messages: list[dict]) -> list[dict]:
    """tau2 messages (model_dump() dicts or results.json dicts) -> neutral events, in trajectory order.

    agent_text {text}; user_text {text}; agent_call {id, name, args}; tool_result {id, error, content};
    user_call / system are ignored (retail has no user tools). A message with text AND calls gives both events,
    text first (tau2 sends such a message to the environment; the user never sees its text)."""
    ev = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if m.get("content") and str(m.get("content")).strip():
                ev.append({"kind": "agent_text", "text": m["content"], "with_calls": bool(tcs)})
            for tc in tcs:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"__unparsed__": args}
                ev.append({"kind": "agent_call", "id": tc.get("id", ""), "name": tc.get("name", ""), "args": args or {}})
        elif role == "user":
            if m.get("tool_calls"):
                continue
            if m.get("content") is not None:
                ev.append({"kind": "user_text", "text": m.get("content") or ""})
        elif role == "tool":
            if m.get("requestor", "assistant") != "assistant":
                continue
            ev.append({"kind": "tool_result", "id": m.get("id", ""), "error": bool(m.get("error")),
                       "content": m.get("content")})
    return ev


def _json(content):
    if content is None:
        return None
    if isinstance(content, (dict, list)):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


# ----------------------------------------------------------------------------------------------------------------
# the state the harness keeps, and the per-call fact record
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class FactState:
    auth_user: str | None = None
    other_users: set = field(default_factory=set)
    profiles: dict = field(default_factory=dict)        # user_id -> {"pm": {pm_id: source}, "orders": [...]}
    orders: dict = field(default_factory=dict)          # order_id -> {"user_id", "status", "orig_pm", "items"}
    status_read: set = field(default_factory=set)       # order ids with a successful get_order_details
    item_changes: dict = field(default_factory=dict)    # order_id -> successful exchange/modify-items count
    writes_ok: list = field(default_factory=list)       # successful writes: (name, canonical args json)
    last_agent_text: str | None = None
    last_agent_text_i: int = -1
    last_user_text: str | None = None
    last_user_text_i: int = -1
    first_user_text_i: int = -1                         # the user's opening request (event index)
    user_texts: list = field(default_factory=list)
    tool_text: list = field(default_factory=list)       # raw tool outputs (for provenance of ids)
    n_user_turns: int = 0
    n_calls: int = 0
    last_result_error: bool = False
    pending: dict = field(default_factory=dict)         # call id -> (name, args) awaiting its result
    i: int = 0                                          # event counter

    # ------------------------------------------------------------------ updates
    def update(self, e: dict) -> None:
        self.i += 1
        k = e["kind"]
        if k == "agent_text":
            if not e.get("with_calls"):      # the user only sees text-only agent messages
                self.last_agent_text, self.last_agent_text_i = e["text"], self.i
        elif k == "user_text":
            self.last_user_text, self.last_user_text_i = e["text"], self.i
            if self.first_user_text_i < 0:
                self.first_user_text_i = self.i
            self.user_texts.append(e["text"])
            self.n_user_turns += 1
        elif k == "agent_call":
            self.pending[e["id"]] = (e["name"], e.get("args") or {})
            self.n_calls += 1
        elif k == "tool_result":
            name, args = self.pending.pop(e["id"], (None, {}))
            self.last_result_error = bool(e.get("error"))
            if name is None or e.get("error"):
                return
            content = e.get("content")
            if isinstance(content, str):
                self.tool_text.append(content)
            self._absorb(name, args, content)

    def _absorb(self, name: str, args: dict, content) -> None:
        if name in AUTH_TOOLS:
            uid = content.strip().strip('"') if isinstance(content, str) else None
            if uid:
                if self.auth_user is None:
                    self.auth_user = uid
                elif uid != self.auth_user:
                    self.other_users.add(uid)
            return
        obj = _json(content)
        if name in ("get_user_details", "modify_user_address") and isinstance(obj, dict) and "user_id" in obj:
            self.profiles[obj["user_id"]] = {
                "pm": {pid: (pm or {}).get("source") for pid, pm in (obj.get("payment_methods") or {}).items()},
                "orders": list(obj.get("orders") or []),
            }
        if name in ORDER_RETURNING_TOOLS and isinstance(obj, dict) and "order_id" in obj:
            ph = obj.get("payment_history") or []
            first_pay = next((p for p in ph if p.get("transaction_type") == "payment"), ph[0] if ph else None)
            self.orders[obj["order_id"]] = {
                "user_id": obj.get("user_id"), "status": obj.get("status"),
                "orig_pm": first_pay.get("payment_method_id") if first_pay else None,
                "items": [it.get("item_id") for it in (obj.get("items") or [])],
            }
            if name == "get_order_details":
                self.status_read.add(obj["order_id"])
        if name in WRITE_TOOLS:
            self.writes_ok.append((name, json.dumps(canon_args(args), sort_keys=True)))
            oid = args.get("order_id")
            if name in ITEM_CHANGE_TOOLS and oid:
                self.item_changes[oid] = self.item_changes.get(oid, 0) + 1

    # ------------------------------------------------------------------ the fact record for a proposed call
    def facts_for_call(self, name: str, args: dict | None) -> dict:
        args = args or {}
        oid = args.get("order_id") if isinstance(args.get("order_id"), str) else None
        auth = self.auth_user
        rec = self.orders.get(oid) if oid else None
        prof = self.profiles.get(auth) if auth else None
        pm = args.get("payment_method_id") if isinstance(args.get("payment_method_id"), str) else None
        status = (rec or {}).get("status")
        f = {
            "tool": name if name in RETAIL_TOOLS else "__unknown__",
            "is_write": name in WRITE_TOOLS,
            "is_order_write": name in ORDER_WRITE_TOOLS,
            "needs_auth": name not in NO_AUTH_NEEDED,
            "needs_pending": name in NEEDS_PENDING,
            "needs_delivered": name in NEEDS_DELIVERED,
            "is_item_change": name in ITEM_CHANGE_TOOLS,
            "uses_payment": name in PAYMENT_TOOLS,
            "needs_oid_confirm": name in NEEDS_OID_CONFIRM,
            "has_order_arg": oid is not None,
            "authenticated": auth is not None,
        }
        # R2: one user per conversation
        uid_args = [v for k, v in args.items() if k == "user_id"]
        f["uid_mismatch"] = auth is not None and any(v != auth for v in uid_args)
        f["order_owner_mismatch"] = auth is not None and rec is not None and rec.get("user_id") != auth
        f["second_user_seen"] = bool(self.other_users)
        # E4-R3 (text-literal reading, revision 2):
        #   order_id_listed: the agent's last text-only message names the call's order id (defined for every order
        #     write; the rule reads it only for cancel and return, policy.md:90,120);
        #   user_yes: the user's latest message is an explicit yes, sent AFTER a text-only agent message that the agent
        #     sent after the user's opening request (policy.md:16: list the action details, then the yes). tau2's
        #     opening greeting is written by the orchestrator, before the user speaks, and never counts as a listing.
        # strict_ids_listed is the reading of revision 1 (order id on every order write, plus the new zip on address
        # writes); it is a MONITOR fact only: logged and reported in every arm, never a bit, never enforced.
        f["order_id_listed"] = self._order_id_listed(name, args)
        f["strict_ids_listed"] = self._strict_ids_listed(name, args)
        f["user_yes"] = (self.first_user_text_i >= 0 and self.last_agent_text_i > self.first_user_text_i
                         and self.last_user_text_i > self.last_agent_text_i and is_explicit_yes(self.last_user_text))
        # R4-R7
        f["status_read"] = oid is not None and oid in self.status_read
        if oid is None:
            st = "__no_order__"
        elif rec is None:
            st = "__unknown__"
        else:
            st = status if status in STATUSES else "__other__"
        f["status"] = st
        f["item_change_done"] = oid is not None and self.item_changes.get(oid, 0) > 0
        # R8
        f["reason_valid"] = name == "cancel_pending_order" and args.get("reason") in VALID_CANCEL_REASONS
        # R10: each new item differs from the item it replaces (policy.md:112,132; the exchange tool does not check)
        olds, news = args.get("item_ids"), args.get("new_item_ids")
        f["new_items_differ"] = (name in ITEM_CHANGE_TOOLS and isinstance(olds, list) and isinstance(news, list)
                                 and len(olds) == len(news) and all(str(a) != str(b) for a, b in zip(olds, news)))
        # R9
        orig = (rec or {}).get("orig_pm")
        owner_ok = rec is not None and auth is not None and rec.get("user_id") == auth
        in_prof = prof is not None and pm is not None and pm in prof["pm"]
        f["pay_in_profile"] = name in PAYMENT_TOOLS and pm is not None and (in_prof or (owner_ok and pm == orig))
        f["pay_is_original"] = name in PAYMENT_TOOLS and pm is not None and orig is not None and pm == orig
        f["pay_is_giftcard"] = name in PAYMENT_TOOLS and in_prof and prof["pm"].get(pm) == "gift_card"
        # extra trusted facts for the learned part only (no written rule reads them)
        f.update(self._extra(name, args, oid, prof))
        return f

    def _order_id_listed(self, name: str, args: dict) -> bool:
        """E4-R3a's fact: an order write whose order id appears in the agent's last text-only message (what the user
        saw). The rule reads it for cancel and return only (NEEDS_OID_CONFIRM); for other order writes it is a
        learned-part input. policy.md:120's 'list of items' is not enforced (agents name items, not item ids)."""
        oid = args.get("order_id")
        if name not in ORDER_WRITE_TOOLS or not isinstance(oid, str):
            return False
        t = self.last_agent_text
        return bool(t) and order_id_in_text(oid, t)

    def _strict_ids_listed(self, name: str, args: dict) -> bool:
        """MONITOR ONLY (revision 1's strict R3 listing check, kept to report what it would have denied): the order id
        on every order write; the new zip on address writes (only the zip for modify_user_address)."""
        if name not in WRITE_TOOLS:
            return False
        t = self.last_agent_text
        if not t:
            return False
        ok = True
        if name in ORDER_WRITE_TOOLS:
            oid = args.get("order_id")
            ok = ok and isinstance(oid, str) and order_id_in_text(oid, t)
        if name in ADDRESS_TOOLS:
            ok = ok and token_in_text(str(args.get("zip") or ""), t)
        return ok

    def _extra(self, name, args, oid, prof) -> dict:
        t = self.last_agent_text or ""
        users = "\n".join(self.user_texts)
        users_n = normalize_text(users)
        tools = "\n".join(self.tool_text)
        items = [str(x) for x in (args.get("item_ids") or []) if isinstance(x, (str, int))]
        new_items = [str(x) for x in (args.get("new_item_ids") or []) if isinstance(x, (str, int))]
        pm = args.get("payment_method_id")
        canon = json.dumps(canon_args(args), sort_keys=True)
        intents = {k: bool(re.search(p, users_n)) for k, p in INTENT_WORDS.items()}
        want = TOOL_INTENT.get(name, [])
        rec = self.orders.get(oid) if oid else None
        n_w = len(self.writes_ok)
        verb = ACTION_VERBS.get(name)
        return {
            "action_word_listed": bool(verb) and re.search(verb, normalize_text(t)) is not None,
            "items_listed": bool(items) and all(token_in_text(x, t) for x in items),
            "new_items_listed": bool(new_items) and all(token_in_text(x, t) for x in new_items),
            "order_in_user_text": bool(oid) and order_id_in_text(oid, users),
            "order_in_profile": bool(oid) and prof is not None and oid in prof["orders"],
            "items_in_order": bool(items) and rec is not None and all(x in (rec.get("items") or []) for x in items),
            "new_items_seen_in_tools": bool(new_items) and all(token_in_text(x, tools) for x in new_items),
            "pay_seen_in_tools": isinstance(pm, str) and token_in_text(pm, tools),
            "same_write_done": (name, canon) in self.writes_ok,
            "order_written_before": bool(oid) and any(json.loads(a).get("order_id") == oid for _, a in self.writes_ok),
            "writes_ge1": n_w >= 1, "writes_ge2": n_w >= 2, "writes_ge3": n_w >= 3,
            "user_turns_ge2": self.n_user_turns >= 2, "user_turns_ge4": self.n_user_turns >= 4,
            "user_turns_ge8": self.n_user_turns >= 8,
            "last_result_error": self.last_result_error,
            "user_negation": has_negation(self.last_user_text),
            "intent_match": bool(want) and any(intents[w] for w in want),
            **{f"user_said_{k}": v for k, v in intents.items()},
            "n_items_ge2": len(items) >= 2,
        }


def canon_args(args: dict) -> dict:
    """Arguments with list values sorted (the retail tools store item lists sorted), for write identity checks."""
    out = {}
    for k, v in (args or {}).items():
        out[k] = sorted(v, key=str) if isinstance(v, list) else v
    return out


def sequentialize(events: list[dict]) -> list[dict]:
    """Put each tool result right after its call. tau2 runs the calls of one message in order, so this is the order
    in which they took effect; it is also how the E4 agent wrapper emits them (one call per message), so a call's
    facts include the results of the calls before it in the same message."""
    # pair each result with the earliest pending call of the same id (FIFO), as tau2's replay pairs them
    pending, pair = [], {}
    for j, e in enumerate(events):
        if e["kind"] == "agent_call":
            pending.append(j)
        elif e["kind"] == "tool_result":
            k = next((p for p in pending if events[p]["id"] == e["id"]), pending[0] if pending else None)
            if k is not None:
                pending.remove(k)
                pair[k] = j
    placed, out = set(pair.values()), []
    for j, e in enumerate(events):
        if e["kind"] == "tool_result":
            if j not in placed:
                out.append(e)            # a result without a call event (should not happen)
            continue
        out.append(e)
        if j in pair:
            out.append(events[pair[j]])
    return out


def replay_facts(events: list[dict], sequential: bool = True):
    """Walk a trajectory; yield (event index, call event, facts-before-the-call, result event or None)."""
    evs = sequentialize(events) if sequential else events
    st = FactState()
    results = {e["id"]: e for e in evs if e["kind"] == "tool_result"}
    for idx, e in enumerate(evs):
        if e["kind"] == "agent_call":
            yield idx, e, st.facts_for_call(e["name"], e.get("args")), results.get(e["id"])
        st.update(e)
