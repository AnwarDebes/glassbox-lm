# tau2-bench v1.0.1: written policy rules a TM guard could enforce

Source tree: `data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/`
(commit fc0055dc4e0a316c3f83133267fbd6faaa770992, tag v1.0.1). Line numbers are 1-based, read with `cat -n` on 2026-10-21.

- `airline/policy.md` = `data/tau2/domains/airline/policy.md`, `retail/policy.md` = `data/tau2/domains/retail/policy.md`.
- `telecom/main_policy.md` and `telecom/tech_support_manual.md` are in `data/tau2/domains/telecom/`.
- The agent receives `<main_policy>` + `<tech_support_policy>` concatenated (`src/tau2/domains/telecom/environment.py:124-132`).
- Agent-wide instruction (all domains): `src/tau2/agent/llm_agent.py:24-32`. It says each turn is either a message or a
  tool call, never both.

## Legend

**Class: what the guard must know.**
- **T (trusted):** the check uses only the harness's own bookkeeping. That means the action log (tool name, arguments,
  order, error flag), structured fields returned by read tools, the tool type tau2 declares in code (`@is_tool(ToolType.WRITE)`),
  the policy's fixed "current time", and message structure.
- **T+X:** a trusted core plus one fact that only free text gives. Examples: the user's "yes", the user's stated reason,
  "the user asked for compensation".
- **X:** the check needs reading free text. That is the user's words or the agent's own words.
- Device facts in telecom (airplane mode, SIM, APN and so on) are **X from the agent's side**. The user simulator calls
  the device tools itself, and the agent only sees what the user says.

**Tool enforces?** This column says whether the environment code already refuses the violating call.
- **probe:** checked by a direct tool call in `results/smoke_probes.json`.
- **code:** read in the tool source (`src/tau2/domains/<d>/tools.py:<line>`).
- **schema:** forced by the tool signature.
- "No" means only the agent's own policy following stops the violation. Those rules are where a guard adds value.

## Airline (`airline/policy.md`, current time fixed at line 3: 2024-05-15 15:00:00 EST)

| # | Rule (short) | Source | Class | Facts the guard needs | Tool enforces? |
|---|---|---|---|---|---|
| A1 | Before booking, flight change, baggage edit, cabin change or passenger update: list the details and get an explicit "yes" | policy.md:7 | T+X | pending call is WRITE (tool type); the agent's last message showed this call's arguments; the user's next reply is "yes" | No (code: no confirmation state in any tool) |
| A2 | One tool call at a time; never text and a tool call in the same turn | policy.md:11; llm_agent.py:24-32 | T | message structure | Orchestrator only with `--enforce-communication-protocol` (default off) |
| A3 | Transfer only if the request is out of scope: call `transfer_to_human_agents`, then send the fixed text 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' | policy.md:15; airline tools.py:532-541 (docstring) | T+X | the order of the two steps and the exact string are T; "cannot be handled" is X (except A25, which is T) | No |
| A4 | No information or procedures beyond what the user or tools gave; no subjective recommendations | policy.md:9 | X | agent text | No |
| A5 | Booking: obtain the user id from the user first | policy.md:65 | T | a `get_user_details(user_id)` succeeded before `book_reservation` with the same `user_id`; the id string occurs in a user message (provenance by exact match) | No |
| A6 | Ask trip type, origin, destination | policy.md:67 | X | conversation content | No |
| A7 | At most five passengers per reservation | policy.md:73 | T | `len(passengers) <= 5` in the arguments | **No (probe: 6-passenger booking accepted)** |
| A8 | First name, last name, date of birth for each passenger | policy.md:74 | T (presence) / X (from the user) | argument fields | schema (Passenger needs all three) |
| A9 | Cabin the same on all flights; all passengers fly the same flights | policy.md:70,75 | T | one `cabin` and one `flights` list per call | schema |
| A10 | At most 1 travel certificate, 1 credit card, 3 gift cards per reservation | policy.md:78 | T | count payment ids by `source` (from `get_user_details` output) | No (code: tools.py:187-318 checks profile and balance only) |
| A11 | Payment methods must already be in the user profile | policy.md:80 | T | payment ids in profile | **Yes (probe: "Payment method ... not found")** |
| A12 | Free checked bags by membership x cabin, $50 per extra bag; do not add bags the user does not need | policy.md:83-97 | T (arithmetic) / X (need) | `nonfree_baggages == max(0, total - free(membership, cabin) * n_passengers)`; membership from `get_user_details` | No (code: the tool trusts `nonfree_baggages`) |
| A13 | Ask whether the user wants travel insurance ($30 per passenger) | policy.md:100-101 | X (asked) / T (price) | conversation; insurance price arithmetic | price yes (code) |
| A14 | Modify or cancel: obtain user id and reservation id; the user must give the user id | policy.md:105-107, 135-137 | T | the reservation's `user_id` (from `get_reservation_details`) equals the authenticated user id | No |
| A15 | Basic economy flights cannot be modified ("the API does not check") | policy.md:110,113 | T | reservation `cabin == basic_economy` and the new `flights` differ from the current ones | **No (probe: segment HAT021 -> HAT089 accepted)** |
| A16 | Keep origin, destination and trip type when changing flights | policy.md:111,113 | T | new segments' origin/destination vs the reservation record | No (code: tools.py:592-690 has no such check) |
| A17 | Kept segments keep their old price | policy.md:112 | - | - | Yes (code: tools.py:629-640) |
| A18 | No cabin change if any flight was already flown | policy.md:116 | T | flight status `flying`/`landed` (`get_flight_status`) | No (code) |
| A19 | Cabin change for the whole reservation only; pay or refund the difference | policy.md:117-120 | - | - | Yes (schema + code) |
| A20 | Bags can be added, not removed | policy.md:123 | T | new `total_baggages >= ` current | **No (probe: 1 -> 0 accepted)** |
| A21 | No insurance after booking | policy.md:124 | - | - | Yes (no tool exists) |
| A22 | Passengers can change, their number cannot | policy.md:127-128 | T | count | **Yes (probe: "Number of passengers does not match")** |
| A23 | Flight change: a single gift card or credit card from the profile | policy.md:131 | T | `payment_id` source and profile | Partly (code: certificate refused tools.py:167, balance tools.py:171) |
| A24 | Cancel: get the reason (change of plan, airline cancelled, other) | policy.md:139 | X | user text; "airline cancelled" can be checked as T | No |
| A25 | Any segment already flown: cannot cancel, transfer | policy.md:141 | T | flight status | No (code) |
| A26 | Cancel only if: booked < 24 h ago, OR airline cancelled the flight, OR business, OR insured AND the reason is covered (health/weather, line 101). "The API does not check" | policy.md:143-149, 101 | T+X | `created_at` vs 2024-05-15 15:00; flight status `cancelled`; `cabin`; `insurance`; the reason (X) only for the insurance branch | **No (probe: economy, no insurance, booked 2024-05-06, accepted)** |
| A27 | Refund to the original payment methods | policy.md:152 | - | - | Yes (code: refunds mirror payment_history) |
| A28 | No compensation unless the user explicitly asks | policy.md:155 | X | user text | No |
| A29 | No compensation for a regular member without insurance in (basic) economy; only silver/gold, insured, or business | policy.md:157,161 | T | membership, insurance, cabin | **No (probe: $500 certificate to a regular member accepted)** |
| A30 | Confirm the facts before offering compensation | policy.md:159 | T | `get_reservation_details` and `get_flight_status` read before `send_certificate` | No |
| A31 | Cancelled flight: certificate = $100 x passengers | policy.md:163 | T+X | flight status `cancelled`; `amount == 100 * n_passengers`; the complaint (X) | No |
| A32 | Delayed flight, and the user changes or cancels: certificate = $50 x passengers, after the change/cancel | policy.md:165 | T+X | status `delayed`; a successful update/cancel on that reservation earlier in the log; amount; the complaint (X) | No |
| A33 | No compensation for any other reason | policy.md:167 | T | `send_certificate` allowed only when A31 or A32 holds | No |

## Retail (`retail/policy.md`)

| # | Rule (short) | Source | Class | Facts the guard needs | Tool enforces? |
|---|---|---|---|---|---|
| R1 | Authenticate first, by email or by name + zip, even if the user gives a user id | policy.md:10 | T | a successful `find_user_id_by_email` / `find_user_id_by_name_zip` returned user id U before any other user-data tool | **No (probe: `get_order_details` and `modify_user_address` work without it)** |
| R2 | After authentication, info about the user's own orders, products and profile | policy.md:12 | T | order `user_id == U` (from `get_order_details`) | No |
| R3 | One user per conversation; deny tasks for any other user | policy.md:14 | T | every acted-on `user_id` / order owner equals U; a second lookup returning another id is flagged | No |
| R4 | Before cancel, modify, return or exchange: list the details and get an explicit "yes" (also in the write tools' docstrings, tools.py:162,217,421,462,550) | policy.md:16 | T+X | pending call is WRITE; the details were shown; "yes" (X) | No |
| R5 | Nothing made up; no subjective recommendations | policy.md:18 | X | agent text | No |
| R6 | One tool call at a time; never text and a tool call together | policy.md:20 | T | message structure | Orchestrator flag only |
| R7 | Transfer only if out of scope: tool call, then the fixed message | policy.md:24 | T+X | as A3 | No |
| R8 | Product id and item id are different things | policy.md:62 | T | an id used as `item_id` must be an item of that order/product | Partly (code: "Item not found" / "Variant not found") |
| R9 | Act only on pending or delivered orders | policy.md:82 | T | order status | Yes, per tool (code) |
| R10 | Exchange and modify-items: once per order; collect all items first | policy.md:84 | T | number of successful exchange/modify-items calls on that order in the log | **Yes, indirectly (probe: second modify-items refused, since status changes)** |
| R11 | Cancel only a 'pending' order, and check the status first | policy.md:88 | T | status from `get_order_details`, read earlier in the log | Status: **yes (probe)**; "checked first": no |
| R12 | Cancel reason must be 'no longer needed' or 'ordered by mistake'; the user confirms the order id and reason | policy.md:90 | T (value) / X (from the user) | argument value | **Yes for the value (probe: "Invalid reason")** |
| R13 | Modify only a pending order (address, payment, item options only), check status first | policy.md:96-98 | T | status; tool set | Yes (code: tools.py:441,483,569) |
| R14 | Payment change: one method, different from the original; a gift card must cover the total | policy.md:102-104 | T | payment ids, balance | Yes (code: tools.py:579,584,591) |
| R15 | Modify-items only once; afterwards the order cannot be modified or cancelled | policy.md:110 | T | `modify_pending_order_items` succeeded on this order (log), or status `pending (item modified)` | Modify-items again and cancel: **yes (probe)**. **Address change after it: NO (probe accepted)**. Payment change: no (code: `_is_pending_order` is `"pending" in status`, tools.py:132-138) |
| R16 | Modify/exchange to an available item of the same product with a different option | policy.md:112,132 | T | product id of old and new item; old != new | Other product: **yes (probe: "Variant not found")**. Same item in modify: yes (code: tools.py:499). **Same item in exchange: NO (probe accepted, status -> exchange requested)** |
| R17 | A payment method for the price difference; a gift card must cover it | policy.md:114,134 | T | balance | Yes (code: tools.py:274,518) |
| R18 | Return only a 'delivered' order, check status first; the user confirms the order id and items | policy.md:118-120 | T (status) / X (items the user named) | status; provenance of item ids | Status: yes (code) |
| R19 | Refund to the original payment method or an existing gift card | policy.md:122-124 | T | payment id type | **Yes (probe: "Payment method should be the original payment method")** |
| R20 | Exchange only a 'delivered' order, check status first; remind the user to list all items | policy.md:130 | T (status) / X (reminder) | status | Status: yes (code: tools.py:241) |

The harness plan's "7 written rules" are the generic retail rules R1-R7 (lines 10-24). The rest are action-specific.

## Telecom (`telecom/main_policy.md`, current time fixed at line 3: 2025-02-25 12:08:00 EST; `tech_support_manual.md`)

| # | Rule (short) | Source | Class | Facts the guard needs | Tool enforces? |
|---|---|---|---|---|---|
| T1 | No information beyond the user or tools; no subjective recommendations | main:7 | X | agent text | No |
| T2 | One tool call at a time; never text and a tool call together | main:9 | T | message structure | Orchestrator flag only |
| T3 | Transfer only if out of scope, try everything first; tool call, then the fixed message | main:13,15; manual:9 | T+X | step order and string (T); "tried everything" partly T (diagnostic steps in the log), partly X | No |
| T4 | Lookup by phone, customer id, or full name + date of birth | main:94-99 | - | - | Yes (schema: `get_customer_by_name(full_name, dob)`) |
| T5 | Technical support: identify the customer first | main:159 | T | a successful `get_customer_by_*` before line or bill actions | No (code: tools take any `customer_id`; only line ownership is checked, tools.py:208-224) |
| T6 | Payment request only for an overdue bill, after checking status and amount ("the tool will not check") | main:105-107,117 | T | bill status `Overdue` read before `send_payment_request` | **No (probe: request accepted for a 'Paid' bill)** |
| T7 | Only one bill AWAITING PAYMENT at a time | main:116 | T | - | Yes (code: tools.py:388) |
| T8 | Say "paid" only after a read shows PAID | main:113 | T+X | a read after `make_payment` shows Paid (T); the agent's claim (X, or a harness-written message) | No |
| T9 | Lift a suspension only after all overdue bills are paid; never if the contract end date is past | main:125-126 | T | bill statuses (`get_bills_for_customer`); line `contract_end_date` vs 2025-02-25 | **No (probe: resume accepted with an Overdue bill and contract end 2025-01-31)** |
| T10 | After resume, the user must reboot | main:128 | X | user-side | No |
| T11 | Refuel at most 2 GB; ask the amount, confirm the price, apply it to the line of the phone number the user gave | main:134-138 | T (<= 2 GB; line belongs to that phone number) / X (price confirmed) | `gb_amount`; line record | **No (probe: 5 GB accepted)** |
| T12 | Change plan: know the line, list plans, the user picks one, price confirmed | main:141-149 | - | - | **No agent tool for it in v1.0.1** (tool inventory in smoke_offline.json) |
| T13 | Traveling user: check roaming; if off, enable at no cost | main:155; manual:136-140 | T (current roaming flag) / X (the user travels) | line `roaming_enabled` | No cost and idempotent: yes (code: tools.py:540-563) |
| T14 | Device troubleshooting order (airplane mode, SIM missing/locked -> escalate, APN reset + reboot, data off, data saver, VPN, network mode, MMS: 3G+, MMSC, Wi-Fi calling, app permissions) | manual:70-91, 123-162, 171-206 | X (agent side) | device state is reported by the user in words; T only if the guard is given the user-side tool log, which the benchmark's agent does not see | No |
| T15 | Data limit exceeded: change plan or refuel | manual:151-157 | T | `get_data_usage` vs plan limit | No |

Solo mode (`--agent llm_agent_solo --user dummy_user`, telecom only: all 114 base tasks have a ticket) gives the agent the
user's device tools too. It uses `main_policy_solo.md` (not extracted here) together with the same `tech_support_manual.md`
(`src/tau2/domains/telecom/utils.py:13-16`). In that mode T14 becomes T, because the agent itself reads the device.

## What the probes showed (results/smoke_probes.json, 2026-10-21, offline)

- 21 direct tool calls on fresh environments. Calls share one environment only where the rule is about a sequence, for
  example modify-items followed by an address change.
- **Refused by the tool:** 8 calls.
  - retail: cancel a delivered order; invalid cancel reason; second modify-items; cancel after modify-items; exchange to
    another product; refund to a non-original card.
  - airline: change the number of passengers; payment method not in the profile.
- **Accepted although the policy forbids them:** 12 calls.
  - retail (R1 x2, R15, R16).
  - airline (A7, A15, A20, A26, A29).
  - telecom (T6, T9, T11).
- **Allowed setup step:** 1 call.

So the rules with the most value for a guard are T-class rules that no tool enforces:
- A7, A10, A12, A15, A16, A18, A20, A25, A26 (the trusted part), A29-A33.
- R1-R3, R15 (address/payment after modify-items), R16 (same item in an exchange).
- T5, T6, T9, T11.
- The structural rules A2/R6/T2.
- The confirmation rules A1/R4 need one free-text fact: the "yes".
