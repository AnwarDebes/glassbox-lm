"""Offline smoke test for tau2-bench v1.0.1 (no LLM, no network, no GPU).

What it does, per domain (mock, airline, retail, telecom):
  A. Inventory: load the environment, list agent tools (and user tools), with the
     tool type tau2 assigns in code (READ / WRITE / THINK / GENERIC) and whether
     the tool mutates state; count tasks, splits and reward bases.
  B. Scripted episodes: for every task of the 'base' split whose reward basis
     needs no LLM judge, build a trajectory by executing the task's gold actions
     in a fresh environment (agent actions as assistant tool calls, user actions
     as user tool calls), add one final agent message that contains every
     communicate_info string, end with the user's ###STOP###, and score it with
     tau2's own evaluator (EvaluationType.ALL).  Expected reward: 1.0.
     Negative controls (expected 0.0 where the condition applies):
       B2 'silent'     : same actions, no final message  -> COMMUNICATE check only
       B3 'no action'  : greeting + final message only     -> DB/ENV check only
       B4 'drop write' : gold actions minus the last state-changing call -> DB/ENV check
  C. Policy-enforcement probes: call tools directly on a fresh environment to see
     whether the TOOL itself refuses a policy-violating call (enforced in code) or
     accepts it (only the agent's own policy following stands in the way).

Run (from any folder; the script never imports from its own directory):
  CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/smoke_offline.py --out results/smoke_offline.json
"""

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tool_rows(toolkit, tools):
    from tau2.environment.toolkit import get_tool_types

    types = get_tool_types(toolkit) if toolkit is not None else {}
    rows = []
    for t in tools:
        schema = t.openai_schema
        fn = schema.get("function", schema)
        params = list((fn.get("parameters") or {}).get("properties", {}).keys())
        rows.append(
            {
                "name": t.name,
                "type": str(getattr(types.get(t.name), "value", types.get(t.name))),
                "mutates_state": bool(toolkit.tool_mutates_state(t.name)) if toolkit is not None else None,
                "params": params,
            }
        )
    return rows


def build_episode(dom, task, env_ctor, drop_last_write=False, with_actions=True, with_final=True):
    """Execute gold actions in a fresh env and return (trajectory, stats)."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    env = env_ctor()
    ist = task.initial_state
    init_data = ist.initialization_data if ist is not None else None
    init_actions = ist.initialization_actions if ist is not None else None
    history = list(ist.message_history) if (ist is not None and ist.message_history) else []
    env.set_state(initialization_data=init_data, initialization_actions=init_actions, message_history=history)

    actions = list(task.evaluation_criteria.actions or []) if task.evaluation_criteria else []
    if drop_last_write:
        idx = [i for i, a in enumerate(actions) if env._is_mutating_tool(a.name)]
        if not idx:
            return None, {"note": "no state-changing gold action"}
        actions = actions[: idx[-1]] + actions[idx[-1] + 1 :]

    traj = list(history)
    if not history:
        traj.append(DEFAULT_FIRST_AGENT_MESSAGE.model_copy(deep=True))
    n_err = 0
    if with_actions:
        for i, a in enumerate(actions):
            tc = ToolCall(id=f"gold_{i}", name=a.name, arguments=dict(a.arguments), requestor=a.requestor)
            if a.requestor == "user":
                msg = UserMessage(role="user", content=None, tool_calls=[tc])
            else:
                msg = AssistantMessage(role="assistant", content=None, tool_calls=[tc])
            resp = env.get_response(tc)
            n_err += int(bool(resp.error))
            traj.append(msg)
            traj.append(resp)
    if with_final:
        info = list(task.evaluation_criteria.communicate_info or []) if task.evaluation_criteria else []
        text = "Done. " + " ".join(info) if info else "Done."
        traj.append(AssistantMessage(role="assistant", content=text))
    traj.append(UserMessage(role="user", content="Thanks. ###STOP###"))
    return traj, {"n_actions": len(actions), "n_tool_errors": n_err}


def sim_run(task, traj):
    from tau2.data_model.simulation import SimulationRun, TerminationReason

    t = now()
    return SimulationRun(
        id=f"smoke_{task.id}",
        task_id=task.id,
        timestamp=t,
        start_time=t,
        end_time=t,
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=traj,
    )


def section_b(dom, tasks, env_ctor):
    """Gold episodes + negative controls.  Never calls an LLM: tasks whose reward needs the
    NL-assertion judge (NL_ASSERTION in the basis AND a non-empty nl_assertions list) are scored
    on their non-LLM parts only and counted separately."""
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    keys = ["gold_full", "gold_nonllm_part", "silent", "no_action", "drop_write"]
    out = {k: Counter() for k in keys}
    odd = []  # results that differ from the expectation, and exceptions
    judge_needed = []
    t0 = time.time()

    def env_reward(traj, task):
        return EnvironmentEvaluator.calculate_reward(environment_constructor=env_ctor, task=task,
                                                     full_trajectory=traj).reward

    for task in tasks:
        ec = task.evaluation_criteria
        if ec is None:
            out["gold_full"]["no_criteria"] += 1
            continue
        basis = set(ec.reward_basis)
        needs_judge = RewardType.NL_ASSERTION in basis and bool(ec.nl_assertions)
        env_basis = basis & {RewardType.DB, RewardType.ENV_ASSERTION}
        try:
            traj, st = build_episode(dom, task, env_ctor)
            if not needs_judge:
                ri = evaluate_simulation(simulation=sim_run(task, traj), task=task,
                                         evaluation_type=EvaluationType.ALL, solo_mode=False, domain=dom)
                out["gold_full"][f"reward={ri.reward}"] += 1
                if ri.reward != 1.0:
                    odd.append({"task": task.id, "check": "gold_full", "reward": ri.reward,
                                "breakdown": {str(getattr(k, "value", k)): v for k, v in (ri.reward_breakdown or {}).items()},
                                "tool_errors": st["n_tool_errors"]})
            else:
                judge_needed.append(task.id)
                r = 1.0
                if env_basis:
                    r *= env_reward(traj, task)
                if RewardType.COMMUNICATE in basis:
                    r *= CommunicateEvaluator.calculate_reward(task=task, full_trajectory=traj).reward
                out["gold_nonllm_part"][f"reward={r}"] += 1
                if r != 1.0:
                    odd.append({"task": task.id, "check": "gold_nonllm_part", "reward": r,
                                "tool_errors": st["n_tool_errors"]})
        except Exception as e:  # report, do not hide
            out["gold_full"]["exception"] += 1
            odd.append({"task": task.id, "check": "gold", "exception": repr(e)[:300]})
            continue

        # B2 silent: communicate check without the final message
        try:
            if RewardType.COMMUNICATE in basis and ec.communicate_info:
                traj2, _ = build_episode(dom, task, env_ctor, with_final=False)
                r2 = CommunicateEvaluator.calculate_reward(task=task, full_trajectory=traj2).reward
                out["silent"][f"reward={r2}"] += 1
                if r2 != 0.0:
                    odd.append({"task": task.id, "check": "silent", "reward": r2})
            else:
                out["silent"]["not_applicable"] += 1
        except Exception as e:
            out["silent"]["exception"] += 1
            odd.append({"task": task.id, "check": "silent", "exception": repr(e)[:300]})

        if not env_basis:
            out["no_action"]["not_applicable"] += 1
            out["drop_write"]["not_applicable"] += 1
            continue
        # B3 no action
        try:
            traj3, _ = build_episode(dom, task, env_ctor, with_actions=False)
            out["no_action"][f"reward={env_reward(traj3, task)}"] += 1
        except Exception as e:
            out["no_action"]["exception"] += 1
            odd.append({"task": task.id, "check": "no_action", "exception": repr(e)[:300]})
        # B4 drop the last state-changing gold action
        try:
            traj4, _ = build_episode(dom, task, env_ctor, drop_last_write=True)
            if traj4 is None:
                out["drop_write"]["no_write_in_gold"] += 1
            else:
                r4 = env_reward(traj4, task)
                out["drop_write"][f"reward={r4}"] += 1
                if r4 != 0.0:
                    odd.append({"task": task.id, "check": "drop_write", "reward": r4})
        except Exception as e:
            out["drop_write"]["exception"] += 1
            odd.append({"task": task.id, "check": "drop_write", "exception": repr(e)[:300]})
    return {
        "counts": {k: dict(v) for k, v in out.items()},
        "nl_judge_needed_tasks": judge_needed,
        "unexpected": odd,
        "seconds": round(time.time() - t0, 1),
    }


def probe(env, name, requestor="assistant", **kw):
    from tau2.data_model.message import ToolCall

    resp = env.get_response(ToolCall(id="probe", name=name, arguments=kw, requestor=requestor))
    return {"tool": name, "args": kw, "error": bool(resp.error), "output_head": (resp.content or "")[:160]}


def section_c():
    """Direct tool calls that break a written rule: does the tool refuse?"""
    from tau2.registry import registry

    res = []

    def rec(rule, src, expect, out):
        out.update({"rule": rule, "policy_source": src, "policy_says": expect,
                    "tool_refused": out["error"]})
        res.append(out)

    # ---------------- retail ----------------
    ctor = registry.get_env_constructor("retail")
    env = ctor()
    db = env.tools.db
    orders = db.orders
    pend = [o for o in orders.values() if o.status == "pending"]
    deliv = [o for o in orders.values() if o.status == "delivered"]
    # R1: no authentication needed by any tool
    o = pend[0]
    rec("authenticate user (email or name+zip) before anything", "retail/policy.md:10", "forbidden before auth",
        probe(env, "get_order_details", order_id=o.order_id))
    rec("authenticate before a write (here: change a user's default address)", "retail/policy.md:10,16",
        "forbidden before auth + explicit yes",
        probe(env, "modify_user_address", user_id=o.user_id, address1="1 Probe St", address2="", city="X",
              state="CA", country="USA", zip="00000"))
    # R2: cancel a delivered order
    rec("cancel only if status is pending", "retail/policy.md:88", "forbidden",
        probe(env, "cancel_pending_order", order_id=deliv[0].order_id, reason="no longer needed"))
    # R3: cancel with a reason that is not allowed
    env = ctor(); o = [x for x in env.tools.db.orders.values() if x.status == "pending"][0]
    rec("cancel reason must be 'no longer needed' or 'ordered by mistake'", "retail/policy.md:90", "forbidden",
        probe(env, "cancel_pending_order", order_id=o.order_id, reason="found it cheaper elsewhere"))
    # R4: modify items, then modify address / payment of the same order
    env = ctor(); db = env.tools.db
    cand = None
    for x in db.orders.values():
        if x.status != "pending":
            continue
        for it in x.items:
            prod = db.products[it.product_id]
            alt = [v for v in prod.variants.values() if v.available and v.item_id != it.item_id]
            if alt:
                cand = (x, it, alt[0])
                break
        if cand:
            break
    if cand:
        x, it, alt = cand
        pay = x.payment_history[0].payment_method_id
        first = probe(env, "modify_pending_order_items", order_id=x.order_id, item_ids=[it.item_id],
                      new_item_ids=[alt.item_id], payment_method_id=pay)
        first.update({"rule": "(setup) modify items once", "policy_source": "retail/policy.md:110",
                      "policy_says": "allowed once", "tool_refused": first["error"]})
        res.append(first)
        rec("after 'modify items' the order cannot be modified any more (address)", "retail/policy.md:110",
            "forbidden", probe(env, "modify_pending_order_address", order_id=x.order_id, address1="2 Probe St",
                               address2="", city="Y", state="NY", country="USA", zip="11111"))
        rec("after 'modify items' the order cannot be modified any more (items again)", "retail/policy.md:84,110",
            "forbidden", probe(env, "modify_pending_order_items", order_id=x.order_id, item_ids=[alt.item_id],
                               new_item_ids=[it.item_id], payment_method_id=pay))
        rec("after 'modify items' the order cannot be cancelled", "retail/policy.md:110", "forbidden",
            probe(env, "cancel_pending_order", order_id=x.order_id, reason="ordered by mistake"))
    # R5: exchange to a different product type
    env = ctor(); db = env.tools.db
    x = [o for o in db.orders.values() if o.status == "delivered"][0]
    it = x.items[0]
    other = [p for p in db.products.values() if p.product_id != it.product_id][0]
    other_item = [v for v in other.variants.values() if v.available][0]
    rec("exchange only to the same product type", "retail/policy.md:132", "forbidden",
        probe(env, "exchange_delivered_order_items", order_id=x.order_id, item_ids=[it.item_id],
              new_item_ids=[other_item.item_id], payment_method_id=x.payment_history[0].payment_method_id))
    # R5b: exchange an item for the very same item (policy: a different option of the same product)
    env = ctor(); db = env.tools.db
    x = [o for o in db.orders.values() if o.status == "delivered"][0]
    it = x.items[0]
    v = db.products[it.product_id].variants.get(it.item_id)
    if v is not None and v.available:
        out = probe(env, "exchange_delivered_order_items", order_id=x.order_id, item_ids=[it.item_id],
                    new_item_ids=[it.item_id], payment_method_id=x.payment_history[0].payment_method_id)
        out["facts"] = {"order_status_after": db.orders[x.order_id].status}
        rec("exchange only to a different option of the same product", "retail/policy.md:132", "forbidden", out)
    # R6: return refund to a payment method that is neither original nor gift card
    env = ctor(); db = env.tools.db
    found = None
    for x in db.orders.values():
        if x.status != "delivered":
            continue
        u = db.users[x.user_id]
        orig = x.payment_history[0].payment_method_id
        others = [pid for pid, pm in u.payment_methods.items()
                  if pid != orig and type(pm).__name__ != "GiftCard"]
        if others:
            found = (x, others[0])
            break
    if found:
        x, pid = found
        rec("return refund goes to original method or an existing gift card", "retail/policy.md:124", "forbidden",
            probe(env, "return_delivered_order_items", order_id=x.order_id, item_ids=[x.items[0].item_id],
                  payment_method_id=pid))

    # ---------------- airline ----------------
    ctor = registry.get_env_constructor("airline")
    env = ctor(); db = env.tools.db
    now_s = "2024-05-15T15:00:00"
    def created_old(r):
        return r.created_at < "2024-05-14T15:00:00"
    def any_cancelled(r):
        for f in r.flights:
            fl = db.flights.get(f.flight_number)
            st = fl.dates.get(f.date) if fl else None
            if st is not None and getattr(st, "status", "") == "cancelled":
                return True
        return False
    def any_flown(r):
        for f in r.flights:
            fl = db.flights.get(f.flight_number)
            st = fl.dates.get(f.date) if fl else None
            if st is not None and getattr(st, "status", "") in ("flying", "landed"):
                return True
        return False
    bad = [r for r in db.reservations.values()
           if r.status != "cancelled" and created_old(r) and r.cabin != "business"
           and r.insurance == "no" and not any_cancelled(r) and not any_flown(r)]
    if bad:
        r = bad[0]
        rec("cancel only if <24h since booking, airline-cancelled, business, or insured+covered reason",
            "airline/policy.md:143-149", "forbidden (none of the 4 conditions hold)",
            dict(probe(env, "cancel_reservation", reservation_id=r.reservation_id),
                 facts={"created_at": r.created_at, "now": now_s, "cabin": r.cabin, "insurance": r.insurance}))
    env = ctor(); db = env.tools.db
    pick = None
    for r in db.reservations.values():
        if r.status == "cancelled" or r.total_baggages <= 0:
            continue
        cc = [pid for pid, pm in db.users[r.user_id].payment_methods.items() if pm.source == "credit_card"]
        if cc:
            pick = (r, cc[0])
            break
    if pick:
        r, pid = pick
        before = r.total_baggages
        out = probe(env, "update_reservation_baggages", reservation_id=r.reservation_id,
                    total_baggages=before - 1, nonfree_baggages=0, payment_id=pid)
        out["facts"] = {"total_baggages_before": before, "total_baggages_after": r.total_baggages}
        rec("checked bags can be added but not removed", "airline/policy.md:123", "forbidden", out)
    env = ctor(); db = env.tools.db
    be = [r for r in db.reservations.values() if r.status != "cancelled" and r.cabin == "basic_economy"
          and not any_flown(r)]
    if be:
        r, new_flights, changed = be[0], None, False
        for cand_r in be:
            f0 = cand_r.flights[0]
            n = len(cand_r.passengers)
            for fl in db.flights.values():
                if fl.flight_number == f0.flight_number or fl.origin != f0.origin or fl.destination != f0.destination:
                    continue
                st = fl.dates.get(f0.date)
                if st is not None and getattr(st, "status", "") == "available" and \
                        st.available_seats.get("basic_economy", 0) >= n:
                    r = cand_r
                    new_flights = [{"flight_number": fl.flight_number, "date": f0.date}] + \
                        [{"flight_number": f.flight_number, "date": f.date} for f in cand_r.flights[1:]]
                    changed = True
                    break
            if changed:
                break
        if new_flights is None:
            new_flights = [{"flight_number": f.flight_number, "date": f.date} for f in r.flights]
        u = db.users[r.user_id]
        cc = [pid for pid, pm in u.payment_methods.items() if pm.source in ("credit_card", "gift_card")]
        if cc:
            old = [(f.flight_number, f.date) for f in r.flights]
            out = probe(env, "update_reservation_flights", reservation_id=r.reservation_id, cabin="basic_economy",
                        flights=new_flights, payment_id=cc[0])
            out["facts"] = {"cabin": r.cabin, "flights_before": old,
                            "flights_after": [(f.flight_number, f.date) for f in r.flights],
                            "segment_changed_in_request": changed}
            rec("basic economy flights cannot be modified (the API does not check)", "airline/policy.md:110,113",
                "forbidden", out)
    env = ctor(); db = env.tools.db
    reg = [u for u in db.users.values() if u.membership == "regular"]
    if reg:
        rec("no compensation for regular member without insurance in (basic) economy; only on request",
            "airline/policy.md:155-161", "forbidden for this user/reservation",
            dict(probe(env, "send_certificate", user_id=reg[0].user_id, amount=500),
                 facts={"membership": reg[0].membership, "note": "no complaint, no reservation facts checked"}))
    env = ctor(); db = env.tools.db
    r = [r for r in db.reservations.values() if r.status != "cancelled" and len(r.passengers) >= 1][0]
    extra = [p.model_dump() for p in r.passengers] + [{"first_name": "Probe", "last_name": "Extra", "dob": "1990-01-01"}]
    rec("the number of passengers cannot change", "airline/policy.md:127-128", "forbidden",
        probe(env, "update_reservation_passengers", reservation_id=r.reservation_id, passengers=extra))
    # book with 6 passengers: find a direct available flight with >= 6 economy seats
    env = ctor(); db = env.tools.db
    booked = None
    for fl in db.flights.values():
        for d, st in fl.dates.items():
            if getattr(st, "status", "") == "available" and st.available_seats.get("economy", 0) >= 6:
                booked = (fl, d, st)
                break
        if booked:
            break
    users_cc = [(u, pid) for u in db.users.values() for pid, pm in u.payment_methods.items()
                if pm.source == "credit_card"]
    if booked and users_cc:
        fl, d, st = booked
        u, pid = users_cc[0]
        price = st.prices["economy"] * 6
        pax = [{"first_name": f"P{i}", "last_name": "Probe", "dob": "1990-01-01"} for i in range(6)]
        rec("at most five passengers per reservation", "airline/policy.md:73", "forbidden",
            probe(env, "book_reservation", user_id=u.user_id, origin=fl.origin, destination=fl.destination,
                  flight_type="one_way", cabin="economy", flights=[{"flight_number": fl.flight_number, "date": d}],
                  passengers=pax, payment_methods=[{"payment_id": pid, "amount": price}], total_baggages=0,
                  nonfree_baggages=0, insurance="no"))
    env = ctor(); db = env.tools.db
    u = users_cc[0][0] if users_cc else None
    if u is not None and booked:
        fl, d, st = booked
        cert = [pid for pid, pm in u.payment_methods.items() if pm.source == "certificate"]
        rec("payment methods must already be in the user profile", "airline/policy.md:80", "forbidden",
            probe(env, "book_reservation", user_id=u.user_id, origin=fl.origin, destination=fl.destination,
                  flight_type="one_way", cabin="economy", flights=[{"flight_number": fl.flight_number, "date": d}],
                  passengers=[{"first_name": "A", "last_name": "B", "dob": "1990-01-01"}],
                  payment_methods=[{"payment_id": "credit_card_0000000", "amount": st.prices["economy"]}],
                  total_baggages=0, nonfree_baggages=0, insurance="no"))

    # ---------------- telecom ----------------
    ctor = registry.get_env_constructor("telecom")
    env = ctor(); db = env.tools.db
    line = [l for l in db.lines if str(getattr(l.status, "value", l.status)).lower() == "active"]
    cust_of = {}
    for c in db.customers:
        for lid in c.line_ids:
            cust_of[lid] = c
    if line:
        l = line[0]
        c = cust_of.get(l.line_id)
        if c is not None:
            rec("refuel at most 2 GB", "telecom/main_policy.md:134", "forbidden",
                probe(env, "refuel_data", customer_id=c.customer_id, line_id=l.line_id, gb_amount=5.0))
    env = ctor(); db = env.tools.db
    bills = {b.bill_id: b for b in db.bills}
    target = None
    for c in db.customers:
        sts = [str(getattr(bills[b].status, "value", bills[b].status)).lower() for b in c.bill_ids if b in bills]
        if any("awaiting" in s for s in sts):
            continue
        for b in c.bill_ids:
            if b in bills and str(getattr(bills[b].status, "value", bills[b].status)).lower() != "overdue":
                target = (c, bills[b])
                break
        if target:
            break
    if target:
        c, b = target
        before = str(getattr(b.status, "value", b.status))
        r_ = probe(env, "send_payment_request", customer_id=c.customer_id, bill_id=b.bill_id)
        r_["facts"] = {"bill_id": b.bill_id, "bill_status_before": before,
                       "bill_status_after": str(getattr(b.status, "value", b.status))}
        rec("send a payment request only for an overdue bill (tool does not check)", "telecom/main_policy.md:105,117",
            "forbidden", r_)
    # Build the forbidden state the way telecom tasks do (initialization function suspend_line_for_overdue_bill):
    # an active line is suspended for a new OVERDUE bill and its contract end date is set in the past.
    env = ctor(); db = env.tools.db
    bills2 = {b.bill_id: b for b in db.bills}
    made = None
    for c in db.customers:
        if any(str(getattr(bills2[b].status, "value", bills2[b].status)).lower() == "overdue"
               for b in c.bill_ids if b in bills2):
            continue
        for lid in c.line_ids:
            l = next((x for x in db.lines if x.line_id == lid), None)
            if l is not None and str(getattr(l.status, "value", l.status)).lower() == "active":
                try:
                    msg = env.tools.suspend_line_for_overdue_bill(customer_id=c.customer_id, line_id=lid,
                                                                  new_bill_id="B_PROBE", contract_ended=True)
                    made = (c, l, msg)
                except Exception:
                    made = None
                break
        if made:
            break
    if made:
        c, l, msg = made
        facts = {"setup": msg, "line_status_before": str(getattr(l.status, "value", l.status)),
                 "contract_end_date": str(l.contract_end_date), "now": "2025-02-25",
                 "bill_statuses": [str(getattr(b.status, "value", b.status)) for b in db.bills
                                   if b.bill_id in c.bill_ids]}
        r_ = probe(env, "resume_line", customer_id=c.customer_id, line_id=l.line_id)
        facts["line_status_after"] = str(getattr(l.status, "value", l.status))
        r_["facts"] = facts
        rec("lift a suspension only after all overdue bills are paid, never if the contract end date is past",
            "telecom/main_policy.md:125-126", "forbidden (overdue bill unpaid AND contract end date past)", r_)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="mock,airline,retail,telecom")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--only-c", action="store_true", help="run only the section C probes")
    args = ap.parse_args()

    t_start = time.time()
    import tau2
    from tau2.registry import registry
    from tau2.runner.helpers import load_task_splits, load_tasks
    from tau2.utils.utils import DATA_DIR

    report = {"started": now(), "python": sys.version.split()[0], "tau2_file": tau2.__file__,
              "data_dir": str(DATA_DIR), "domains": {}}
    for dom in ([] if args.only_c else [d for d in args.domains.split(",") if d]):
        print(f"[{now()}] domain {dom}", flush=True)
        d = {}
        try:
            ctor = registry.get_env_constructor(dom)
            env = ctor()
            pol = env.get_policy()
            d["policy_chars"] = len(pol)
            d["agent_tools"] = tool_rows(env.tools, env.get_tools())
            d["user_tools"] = tool_rows(env.user_tools, env.get_user_tools()) if env.user_tools is not None else []
            splits = load_task_splits(dom) or {}
            d["splits"] = {k: len(v) for k, v in splits.items()}
            tasks = load_tasks(dom, "base" if "base" in splits else None)
            d["n_tasks_base"] = len(tasks)
            rb = Counter()
            n_nl = n_comm = n_envass = n_user_act = n_write_gold = 0
            n_act = []
            for t in tasks:
                ec = t.evaluation_criteria
                if ec is None:
                    rb["<none>"] += 1
                    continue
                rb["+".join(sorted(str(x.value) for x in ec.reward_basis))] += 1
                n_nl += int(bool(ec.nl_assertions))
                n_comm += int(bool(ec.communicate_info))
                n_envass += int(bool(ec.env_assertions))
                acts = ec.actions or []
                n_act.append(len(acts))
                n_user_act += int(any(a.requestor == "user" for a in acts))
                n_write_gold += int(any(env._is_mutating_tool(a.name) for a in acts))
            d["reward_basis_counts"] = dict(rb)
            d["tasks_with_nl_assertions_listed"] = n_nl
            d["tasks_with_communicate_info"] = n_comm
            d["tasks_with_env_assertions"] = n_envass
            d["tasks_with_user_side_gold_actions"] = n_user_act
            d["tasks_with_state_changing_gold_action"] = n_write_gold
            d["gold_actions_per_task"] = {"min": min(n_act) if n_act else 0, "max": max(n_act) if n_act else 0,
                                          "mean": round(sum(n_act) / len(n_act), 2) if n_act else 0}
            from tau2.data_model.tasks import RewardType
            need = {t.id for t in tasks if t.evaluation_criteria is not None
                    and RewardType.NL_ASSERTION in t.evaluation_criteria.reward_basis
                    and t.evaluation_criteria.nl_assertions}
            d["tasks_needing_nl_judge"] = len(need)
            d["tasks_needing_nl_judge_by_split"] = {k: len(set(v) & need) for k, v in splits.items()}
            if not args.skip_b:
                d["B_scripted_episodes"] = section_b(dom, tasks, ctor)
        except Exception as e:
            d["error"] = repr(e)
            d["traceback"] = traceback.format_exc()[-3000:]
        report["domains"][dom] = d
    print(f"[{now()}] section C probes", flush=True)
    try:
        report["C_policy_probes"] = section_c()
    except Exception as e:
        report["C_error"] = repr(e)
        report["C_traceback"] = traceback.format_exc()[-3000:]
    report["finished"] = now()
    report["seconds_total"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=1, default=str)
    print(f"[{now()}] wrote {args.out} in {report['seconds_total']} s", flush=True)


if __name__ == "__main__":
    main()
