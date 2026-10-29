# Registration E4: tau2-bench retail, local Qwen as the agent, a Tsetlin machine guard over the written retail rules

- **Frozen** 2026-10-23T01:27:37Z. Freeze record: `runs/E4_frozen.json` (sha256 of the registered text
  `84bdfe7a7bb2c8cde8b67404d3fa2a560c5019381070ad14c670f98d35d33582`).
- This published copy leaves out the draft history and the pre-freeze review items (the first two sections of the
  registered text) and is edited in wording only; every rule, bar and number is as registered. Both hashes are in
  `docs/hash-table.md`.
- Also left out: the operational sections of the registered text (11.1-11.4 and 11.6 stages, smoke gates, budget
  and freeze steps;
  12 hash tables; 18 offline code tests; 19 compute; 20 file list). Their rules ran as registered; the outcome of
  every stage is in `docs/results.md`.

## 1. Question, headline, order

### 1.1 Question (DESIGN E4)

How often does Qwen break retail's written policy? Does enforcing the rules pay? Does a learned TM veto add anything
on top of the rules?

### 1.2 Headline and multiplicity (fixed now)

- **Within E4, E4.3 alone asks the TM question:** does a learned TM veto on trusted facts add anything on top of the
  code gates?
- **E4.3 is low power** (section 10.3: 12.0% at the go-bar's corner; 80% power needs about +13.4 points net
  with nothing against) **and a supporting rule:** no claim is weakened or strengthened by E4.3 alone
  (decision 1, fixed before the freeze). A miss is not evidence that the TM veto adds nothing; a hit is a supporting result under E2's claim,
  never a headline on its own. Every E4.3 sentence carries its power figure (section 15).
- **E4.2 is a code-gate baseline of a known kind** (section 14). It carries no TM claim.
- **Superiority claims:** E4.2 (DB) and E4.3 (DB, low power). **Non-inferiority:** E4.5 (DB).
- **Readings with no rule:** every judged number (self-judged, unvalidated); A2 against the tree; A2 seeds 2 and 3; the
  two-level intervals; the readings without infrastructure or timeout ends; E4.2 without the smoke tasks.
- **No combined E4 "met" line. No multiplicity correction is applied; each bar stands alone.** Every bar is shown in
  one table (section 10.1; `summary.json` field `table`).

### 1.3 Order relative to E2, and the family across E2 and E4

- **Both are run** (decided 2026-10-21T07:24:12Z): E4 is a supporting step under E2's claim; no family-wise or
  combined claim across E2 and E4; each registration's rules are reported separately.
- **The family of guard claims across the two registrations:** E2's primary bar, E4.2 and E4.3. No multiplicity correction is
  applied across it and no family-wise or combined claim is made; each registration's rules are reported separately.
- DESIGN 2.5 puts E4 after E2 (order 4, after E2-1 to E2-5). E4's smoke and stages run only in the GPU slots of the
  project's queue. They run before E2's stages only if that order is decided and logged.
- **No E4 output feeds E2.** E2 fixes its own server settings and has no yes-matcher. Nothing measured in E4 (rates,
  budgets, matcher results) changes an E2 choice.
- Both use one shared server (`llm/serve_qwen_agent.sh`) with the same GGUF. E4 checks the server's identity
  against its own pins (section 4.1).
- **Not an either/or:** E2 and E4 are separate registrations on different benchmarks and questions. Neither one's
  outcome decides whether the other runs.

## 2. Benchmark, tasks, splits

| Item | Value |
|---|---|
| Benchmark | tau2-bench v1.0.1, commit `fc0055dc4e0a316c3f83133267fbd6faaa770992` (MIT), installed in `harness_build/taubench/.venv` (read-only use) |
| Domain | retail: 114 tasks, 16 agent tools (7 write tools) |
| Splits | `data/tau2/domains/retail/split_tasks.json`: train 74, test 40 (sha256 in section 12, H3) |
| Trials | 1 per task at temperature 0, so the score is pass^1 |
| Test ids (40) | 5, 9, 12, 17, 18, 26, 27, 32, 33, 36, 38, 39, 40, 42, 45, 49, 51, 53, 55, 56, 60, 61, 62, 64, 65, 68, 70, 71, 74, 77, 79, 86, 90, 94, 97, 100, 101, 102, 108, 111 |
| Train ids (74) | the other 74 of 0-113 (`split_tasks.json` "train") |
| Smoke tasks (10 to 20, by a fixed rule) | the first 10 train ids in `split_tasks.json`'s order: 0, 1, 2, 3, 4, 6, 7, 8, 10, 11 (judge tasks among them: 2, 3, 4); revision 4's extension rule (section 11.2, S0) adds the next train ids in that order, one at a time (13, 14, 15, 16, 19, 20, 21, 22, 23, 24), while A0's label candidates are fewer than 10, to at most 20 tasks |

**Which tasks need the LLM judge** (`scripts/check_judge_tasks.py`, `results/judge_tasks.json`; unchanged from
revision 1):

- Reward basis: DB + NL_ASSERTION for 112 tasks, DB only for 2 (tasks 33, 34). No other component.
- **40 tasks need the judge: 29 train, 11 test.** A static count and a behavioural count agree. tau2's own
  `evaluate_simulation(ALL)` ran on a gold episode of every task with the judge's `generate` stubbed; the stub was
  called exactly 40 times, on the same 40 ids, and contacted nothing.
- Test-split judge tasks: 36, 38, 39, 40, 45, 60, 62, 68, 70, 108, 111.

## 3. Scores

### 3.1 The primary score (fixed): judge-free DB pass^1

- The mean over tasks of the DB part of tau2's reward. A simulation scores 1 when it ends by user or agent stop and the
  agent database after replaying its trajectory equals the database after replaying the gold actions (tau2's
  `EnvironmentEvaluator`). Otherwise it scores 0.
- On the 74 tasks without assertions it equals tau2's full reward. On the 40 judge tasks it is an upper bound of tau2's
  reward (reward = DB x judge verdict).
- It is computed inline (`EvaluationType.ENV`), so a judge reply can never re-run a simulation.
- **Every bar uses it.** An earlier revision's option to make the judged score primary is deleted.

### 3.2 The judged score: a reading, "self-judged, unvalidated"

- tau2's NL judge with the same local Qwen, computed post hoc (`scripts/judge_posthoc.py`, JSON mode, temperature 0).
- It is reported beside every bar and is **never a bar or a headline**.
- It cannot be calibrated here: no v1.0.1 retail trajectory with official judge verdicts exists locally. The shipped
  retail files are 2025 runs on old task versions, which have no NL assertions.
- **A judge error scores NL 0 and is counted per arm:**
  - a reply that raises (not JSON, missing keys);
  - a reply whose results count differs from the assertion count, including an empty or missing list (tau2 would score
    that as met: `evaluator_nl_assertions.py:134` and `:51-52`);
  - a task listed as needing the judge whose task version has no assertion.
- The smoke runs the judge in health-only mode: parse rate, the share with the right results count, and seconds per
  call. It produces no score.

### 3.3 Missing and failed simulations (rule 9, fixed now)

- **End classes** (`e4guard/runs.py`):
  - "infra" = `infrastructure_error`, `unexpected_error` or `context_window_exceeded` (tau2 gives up after its
    retries; a context overflow or an unparsable tool call ends this way);
  - "timeout" = the 1800 s limit;
  - "normal" = every other end.
- An infra or timeout end **scores 0** and is **counted per arm**.
- **A bar is computed only if every arm of the pair holds every task of its set** (114 or 40). Otherwise it reads "not
  computed" and names the missing tasks; once the stage that runs a missing arm has failed its third attempt, it reads
  "not shown (stage failed)" (revision 5; section 11.6).
- Every bar is also read **without the tasks that ended infra or timeout in either arm** (a reading).
- **Re-run rule:** if an arm has more than 3 such ends, those simulations are re-run once from the store
  (`STAGE=rerun`) into `<arm>_rerun/` before the summary. The re-run replaces them and is logged. There is never a
  second re-run. (Revision 5: the re-run writes its task list when it starts; stopped before its end, it resumes on
  that list; section 11.3.)
- **A re-run left unfinished** (revision 6; the round-5 review's minor 1): after the `rerun` stage's third failed
  attempt (11.6) the re-run is used as it stands: the tasks it finished replace their first ends, and the others keep
  their first infra or timeout ends (score 0, counted). The summary lists them under "anomalies".

## 4. Agent, user simulator, server

### 4.1 The server, pinned (review issue 1, decision 1)

- **Every LLM role is the local Qwen3.5-9B Q8_0:** agent, user simulator and post-hoc judge. No paid service.
- **The server** is the shared llama-server b11742 (`/props` `build_info` `b1-4c3c581`), started by
  `llm/serve_qwen_agent.sh` with its defaults. Its exact argv (pinned):
  `.../build-cuda129/bin/llama-server -m .../Qwen3.5-9B-Q8_0.gguf --host 127.0.0.1 --port 18082 -a qwen3.5-9b-q8_0
  --jinja --reasoning-budget 0 -fa off -np 1 -c 65536 -ngl 99 -t 2 --no-webui --metrics -rea off`.
  - `-np 1` with 65536 tokens in the single slot (at least 64k, as the review asks). `cache_prompt` stays at its default.
  - Full paths: `pins/server_expected.json`.
- **Pinned in `pins/server_expected.json`** (its sha256 is in section 12, H2):
  - the GGUF sha256 `ff3d8ff122c21c37f2ca52502db705e36903c581657f1e8d3e930a9cfb4d50aa` (hashed again 2026-10-21T04:53:31-04:53:55Z);
  - the executable sha256 `2fc2870a983daedf21cc248a890752fb761fd9df85e04d68b1264bc61a84652b`;
  - all 34 files of its build directory (`llm/build_b11742_bin.sha256`, sha256 `5d9d8e33161234731c72f0cce7e449939420073eadb691244ed8b70ef06d9e2d`, re-checked 04:53:30Z);
  - `/props`: `total_slots` 1, `default_generation_settings.n_ctx` 65536, `build_info`, `model_alias`, `model_path`;
  - the `chat_template` sha256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. This is the GGUF's
    own `tokenizer.chat_template`, read offline with `scripts/gguf_meta.py`. It equals the `/props` value from the server's
    CPU smoke (04:02Z; server metadata only, no task output).
- **The identity check** (`e4_run.server_identity`) runs before any request. The judge and the repeat check run it
  too. It compares:
  - `GET /props` and `/v1/models`;
  - the one process whose argv holds `--port 18082` (its argv, executable and build files, read from `/proc`);
  - the GGUF at `model_path`.
- On any difference the run exits 2. The identity record goes into every `manifest.json`.
- **Again when every call ends** (revision 3; `e4_run.identity_end_check`): the pid, `/props`, `/v1/models`, the argv
  and the executable path, against the pins and the call's own start (no re-hashing). "same" goes on. A changed pid
  ("restart_same_identity") or anything else ("identity_changed") exits 2 after the results are written; section 11.6
  says what follows. A stage's first call checks its start and its last call its end. Every call's record is kept
  (`manifest.json`; a resumed call keeps the earlier one as `manifest.<k>.json`), and the summary lists the pids seen
  per arm.
- **A call stopped before its own end check** (revision 6; the round-5 review's Major 1. Revision 5 checked it at the
  run's next start, before the start check, so a server started wrongly between two slots read as an integrity stop):
  - **At the stop.** The operator's stop at a slot's end (TERM or INT to `run_e4_gpu.sh`, or a client ended by TERM or
    INT) and a failed client (exit 3) give the stopped call its end check at once, while the server that served it is
    still up (the operator stops the client only, never the server; 11.3). `scripts/end_check_at_stop.py` runs the
    light end check (`e4_run.identity_end_check`) against the pins and the call's own start record (the run's
    `manifest.json`, or the judge's `<out>.start.json`) and writes it there as `server_identity_end` with
    `"at_stop": true`. Only the stopped call's own record is checked: written at or after the client's start, and
    without its end. "same" or "restart_same_identity" leaves the stop as it was (exit 8, or a failed attempt);
    "identity_changed" is an integrity stop for that run (9.1): a `STOP [identity_changed] run=<n>` line, exit 2.
  - **At the next start, the full start check first** (`e4_run.start_checks`, in `e4_run.py` and in the judge). A
    server that is not the pinned one exits 2 tagged `[server]` (not counted, no integrity event; 11.6) and leaves
    every record untouched. Only with the pinned server up does a record that still has no end (a kill -9 or a crash
    of the script itself) get its deferred end check (`e4_run.deferred_end_check`, written with `"deferred": true`):
    "same" or "restart_same_identity" goes on; anything else exits 2 tagged `[identity_changed]` for that run (9.1). A
    record that cannot be read, or has no start record, is "identity_changed".
  - **What the deferred check can and cannot see.** It compares the pinned server as it is now with the call's start
    record. It sees a restart (a new pid) and a record it cannot read. A change made during the call and undone before
    the restart stays invisible to it; only the end check at the stop could have seen it.
  - The judge writes its start record first (`<out>.start.json`) and gets the same checks.
- **Why `-np 1`:** tau2 sends one request at a time (`max_concurrency` 1), so the batch is 1 with either setting.
  `-np 2` would only keep more prompt caches, and cache reuse is the nondeterminism source the server README warns
  about. Pairing between arms rests on the replay store, not on server determinism.
- **If the budget does not fit:** if the measured budget does not fit the slot after the cut order, it is decided and
  logged before the freeze; one option is a second smoke with `-np 2`, as a new revision of this text.
- **A planned server change (revision 4).** The GPU queue runs
  BG5 on this server after both registrations are ready and before the smokes: the speed protocol's qwen/S1,
  a lower server-CPU setting, accepted there only with identical replies. If BG5 adopts it, it enters E4 **only as a
  pins-only revision before the smoke**: a new `pins/server_expected.json` (the new argv; the GGUF, the executable and
  the build files unchanged), its H2 row, `check_pins.py --stage smoke` on the revised text, and the identity check
  against the new pins when the smoke's first call starts (S1). No rule, bar or threshold changes with it. **After the
  smoke, E4 keeps its pinned argv** until its last stage: a server started otherwise is exit 2 at every call.
- **E4 gives BG5 no request bodies** (revision 5; the round-4 review's m3). PROTOCOL.md runs BG5 "on fixed train / dev
  agent bodies from the E2 / E4 owners"; E4 provides none, so no Qwen reply to an E4 request exists before the smoke.

### 4.2 Request arguments (all arms, all roles)

- `temperature 0.0` (sent explicitly; checked against the stub);
- `max_tokens 2048`, `timeout 600`;
- `extra_body.chat_template_kwargs.enable_thinking = false`;
- the trial seed that tau2 derives from `--seed 300` (626729, the same for every arm).

### 4.3 tau2 settings

- max_steps 200, max_errors 10, max_retries 3, timeout 1800 s per simulation, max_concurrency 1;
- auto_resume on, auto_review off, hallucination_retries 0, enforce_communication_protocol off.

### 4.4 The agent (unchanged from revision 1)

- `e4guard/agent.py:GuardedAgent`, a subclass of tau2's `LLMAgent` with the same system prompt and the same `generate`
  call, in every arm.
- **One call per message.** A model message with several tool calls stays whole in the agent's private context, but
  the calls reach the environment one per message, in order (policy.md:20).
- **A denied call never reaches the environment and never enters the trajectory.** tau2's scorer replays every mutating
  call of the trajectory, so a denied write left there would be executed at scoring time. The model instead gets the
  denial text in its private context.
- **Retries:** at most 3 model calls after denials in one turn; then a fixed message to the user.
- **A0 is the same class in monitor mode.** Offline, with a fake deterministic LLM, A0 sent byte-identical requests to
  stock `LLMAgent` (section 18.2). The smoke repeats this with the real model (gate S6).
- **New in revision 2:** every verdict is logged with the live fact record's bit vector and sha256 and the monitor
  facts. Every construction of the agent is logged, one per tau2 attempt, so tau2's re-runs are counted.

## 5. The written rules (E4-R1 to E4-R10: A1's code gates = A2's veto clauses)

| Rule | policy.md | The check, on trusted facts only | Veto clause(s) | Tool enforces it? (taubench probes) |
|---|---|---|---|---|
| E4-R1 | :10 | before a successful `find_user_id_by_email` / `find_user_id_by_name_zip`, only those two and `transfer_to_human_agents` | E4-R1: `needs_auth AND NOT authenticated` | no (reads and `modify_user_address` work without it) |
| E4-R2 | :14 | a `user_id` argument other than the authenticated user; an order whose owner (from its own record) is another user | E4-R2a: `uid_mismatch`; E4-R2b: `order_owner_mismatch` | no |
| E4-R3 | :16, :90, :120 | **text-literal (revision 2):** before a cancel or a return, the agent's last text-only message names the call's order id; before every write, the user's latest message is an explicit yes answering a text-only agent message sent after the user's opening request | E4-R3a: `needs_oid_confirm AND NOT order_id_listed`; E4-R3b: `is_write AND NOT user_yes` | no |
| E4-R4 | :88, :96, :118, :130 | `get_order_details` on that order succeeded before an order write | E4-R4: `is_order_write AND NOT status_read` | no |
| E4-R5 | :88, :96, :110 | cancel / modify only when the order's latest known status is exactly `pending` | E4-R5: `needs_pending AND NOT status=pending` | partly (an address change after modify-items is accepted) |
| E4-R6 | :118, :130 | return / exchange only when it is `delivered` | E4-R6: `needs_delivered AND NOT status=delivered` | yes |
| E4-R7 | :84, :110 | exchange or modify-items at most once per order | E4-R7: `is_item_change AND item_change_done` | yes, indirectly |
| E4-R8 | :90 | cancel reason exactly `no longer needed` or `ordered by mistake` | E4-R8: `tool=cancel AND NOT reason_valid` | yes |
| E4-R9 | :102, :114, :122-124, :134 | the payment method is in the user's profile (or is the order's original method); a refund goes to the original method or a gift card of the user | E4-R9a: `uses_payment AND NOT pay_in_profile`; E4-R9b: `tool=return AND NOT pay_is_original AND NOT pay_is_giftcard` | yes |
| E4-R10 | :112, :132 | each new item differs from the item it replaces | E4-R10: `is_item_change AND NOT new_items_differ` | no for exchange (a same-item exchange is accepted) |

- **13 veto clauses in all.** Code: `e4guard/rules.py` (plain if-statements) and `e4guard/bits.py:VETO`. They were
  written apart and proved equal: 0 mismatches over 1,392,640 vectors in revision 2 (IC2, section 9).

### 5.1 E4-R3, the text-literal reading (revision 2; review decision 2)

- **The order id is required only where the policy says so:** before a cancel (:90) and before a return (:120).
  Exchange (:128-136) and modify (:94-114) have no order-id wording.
- **Every write needs an explicit yes** from the user, answering a text-only agent message that the agent sent after
  the user's opening request (:16: "list the action details and obtain explicit user confirmation (yes)"). tau2's
  greeting is written by the orchestrator before the user speaks, so it never counts as a listing.
  - This last condition is new in revision 2. In the scripted offline re-run, 4 planted "no confirmation" writes passed
    without it (tasks 20, 22, 37 train; 36 test). The user's opening request held a yes-like phrase, and the greeting
    counted as the agent's message (study log, revision 2 entry). Fixed before any Qwen data.
- **No zip check:** no line of the policy asks for it.
- **:120's "the list of items" is not enforced.** Agents name items, not item ids: in the stand-in, item ids were named
  before only 11% of writes.
- **Revision 1's strict reading is a monitor fact in every arm** (order id on every order write, the new zip on
  address writes). It is logged with each verdict as `monitor.strict_r3a`, reported per arm (A0 also by tool, in the
  smoke), and never enforced. It is not a bit.
- **Why, and when.** The reason is the policy text. Declared: the change was made after the stand-in counts were seen
  and before any Qwen number. It was not chosen from the smoke, which gives only about 5-15 writes.
  - Stand-in counts (gpt-4.1, 534 writes in its reward-1 runs): the strict reading would deny 145 (27%) by E4-R3 (revision
    1). The review counted 126 order-id/zip-only denials, 106 of them on exchange and modify writes.
  - The text-literal reading would deny 35 of 534 (6.6%), 81 (15.2%) by any rule (section 18.3).

### 5.2 Explicit yes: fixed now

- `facts.is_explicit_yes` is ordinary code. It is tested with 55 hand-written cases from revision 1 plus revision 2's
  cases (62 facts tests).
- It requires a strong affirmative phrase anywhere ("yes", "go ahead", "that's correct", ...) or a weak one at the start
  ("ok", "sure", ...). It rejects any refusal ("no", "wait", "don't ...", "before you proceed", ...). A bare instruction
  ("please cancel it") is not a yes.

### 5.3 The yes-matcher is frozen with this registration; stage 2b is deleted

- No matcher revision happens after this registration, before or after A0.
- The smoke's label check (gate S8, section 11.2) is a **stop/go gate, not a tuning step.**
- If S8 fails, E4 is not frozen in this form. Any matcher change would be a new draft revision that names the smoke as
  seen and runs a new smoke on other train tasks.

### 5.4 Written rules not enforced, and why

- :16 "list the action details and obtain explicit user confirmation (yes)": checked only as E4-R3. The yes must
  answer a text-only agent message sent after the user's opening request (E4-R3b), and before a cancel or a return
  that message must name the call's order id (E4-R3a). What else the listing says (the details) is not checked.
- :20 "one tool call at a time": handled by the one-call-per-message split, not by a denial.
- :20 "no text and a tool call together": counted (smoke and summary), not denied.
- :24 the transfer message: a text rule.
- :18 "nothing made up, no subjective comments": needs free text.
- :120 "the list of items": agents name items, not ids.
- :84 "collect all items into a list": not checkable from one call.
- :110 and :130 "remind the customer to confirm all items": a text rule.
- :104, :114, :134 gift-card balance: the tool enforces it.

## 6. Trusted facts and bits

- **Facts come only from the harness's record of the trajectory** (`e4guard/facts.py`):
  - the tool calls and the structured JSON the tools returned (database results are trusted: tau2 has no attacker);
  - the authenticated user id;
  - the user's messages (the user is the principal);
  - the text the agent last showed the user.
- A model's claim about the world is never a fact.
- **75 named bits** (`e4guard/bits.py`; 74 in revision 1):
  - one-hot tool (17, including unknown);
  - 8 tool-class bits (new: `needs_oid_confirm` = cancel or return);
  - one-hot order status (10);
  - 13 rule facts (`order_id_listed` replaces revision 1's `ids_listed`);
  - 27 facts for the learned part only.
- **The MILP domain:** 48 linear rows that every real input satisfies (one-hot groups, tool-class
  definitions, implications, thermometer chains).

## 7. Arms

| Arm | What decides | Run on |
|---|---|---|
| A0 | Qwen alone; the guard computes verdicts and logs them but never blocks | all 114 tasks |
| A1 | code gates E4-R1 to E4-R10; a denied call returns its reason to the agent | all 114 tasks |
| A2 (seeds 1, 2, 3) | the **guarded CoTM**: learned clauses + the 13 veto clauses (weight -M on ALLOW, M = ALLOW's positive weights + abs(DENY weights) + 1) decide every write; reads: veto clauses only. **Seed 1 is the registered bar**; seeds 2 and 3 give the two-level interval | the 40 test tasks, only after the go-bar says GO |
| A3 | LR twin of A2's learned part, same bits, same veto overlay | the 40 test tasks, after GO |
| A3t | depth-limited tree twin, same bits, same veto overlay (a reading) | the 40 test tasks, after GO |

- On E4-R1 to E4-R10, A1's code and A2's veto clauses are the same function. The TM can add something only through
  its learned part.
- **Paired replay** (`e4guard/replay.py`): one store per campaign maps sha256(request) to the reply, shared by every
  arm and role.
  - A1 reuses A0's replies until its first denial.
  - A2, A3 and A3t reuse A1's replies until their first different verdict.
  - The key leaves out transport-only arguments (`api_base`, `api_key`, `timeout`, retries).
- **The registered campaign's store does not exist at the freeze.** The smoke has its own campaign and store
  (`SMOKE_<campaign>`). `freeze_e4.py` refuses a registered store that exists, and `check_pins.py --stage a0`
  refuses one that exists before the a0 marker.
- **Run order on the test tasks:** A2 seed 1, A3, A2 seed 2, A2 seed 3, A3t. The arms the budget never cuts run first.

## 8. The learned part and its twins (review issue 8, decision 6)

- **Rows:** every write that A0 or A1 **executed** on the **74 train tasks** and on which **no veto clause fires**.
  Those are the only writes the learned part ever decides. They are **deduplicated across A0 and A1** by (task, tool,
  canonical arguments, bit vector), keeping the first copy (A0 before A1). Without this, A1 rows would duplicate A0
  rows on every task without a denial.
  - Label: DENY if the write is not one of the task's gold actions with matching arguments (lists compared as
    multisets), else ALLOW (`scripts/build_rows_from_results.py`).
  - Test-task rows are never used for training or choices.
  - `train_guard.py` refuses any row on which a veto clause fires.
- **Minimum:** at least 10 veto-clean, deduplicated train rows of each class. Otherwise the learned part is not trained,
  E4.3 and E4.5 read "not testable", and A2, A3 and A3t are not run. (train_guard.py exits 3; the stage writes
  `gobar.json` "not testable" and exits 0, so the loop goes on to `judge` and `summary` for E4.2. No integrity flag
  follows: IC2 (b) and IC6 read "not applicable", section 9.)
- **CoTM:** the project's `glassbox-lm/studies/decision_layer/src/dl/tm.py`, imported unedited, `device="cpu"`,
  `half=False`, 2 threads.
  - Settings: 200 clauses, T 100, 60 epochs, batch 16, negated literals on.
  - s from {2, 3, 5, 8, 12, 16, 24}. These grids were widened in revision 1 after the stand-in CV picked the top end;
    they are frozen now.
  - Its `fit()` calls `torch.cuda.synchronize()`, which raises with no GPU visible, so the training process makes that
    one function a no-op. The file is not touched.
- **LR:** scikit-learn LogisticRegression, C from {0.01, 0.1, 1, 10, 100, 1000}.
- **Tree:** max_depth from {2, 3, 4, 6, 8}.
- **Choices, the same rule for the TM, LR and the tree:**
  - 5-fold CV over train tasks (folds from a seeded task permutation, seed 20261021), on balanced accuracy;
  - ties go to the earlier grid value; the TM uses seed 1 in the CV;
  - then a refit on all rows: TM seeds 1, 2 and 3, LR, tree.
- **Reported beside each pick** (review issue 8): out-of-fold balanced accuracy, DENY recall and allow_kept (the share
  of gold-matching rows kept), from the same folds. The TM's refit fold predictions must equal the CV's own (asserted).
- **Exactness, checked on every row:** the exported model gives the same votes or predictions as the trained object,
  and the deployed decision runs the exported model.
- **MILP, per trained TM seed** (`scripts/milp_proof.py`; `verify_policy.build_and_solve` imported unedited; its
  sha256 is now recorded in the output):
  - the pre-checks D, E and W, as in revision 1;
  - 13 checks. The guarded TM must be infeasible on 13 of 13, and the domain must hold, for every seed; else STOP
    (exit 5): IC2 (b) failed, E4.3 and E4.5 are void (section 9.1), E4.2 stands.

## 9. Integrity checks: outcome forced by construction, not evidence (review issue 5)

- **IC1 (was bar E4.1):** 0 executed calls that our code reads as breaking E4-R1 to E4-R10, in A1, A2 (each seed), A3
  and A3t.
  - The facts are recomputed from the saved trajectory with the same `facts.py` and `rules.py` the guard used, so an
    error in fact extraction passes IC1. That is why IC3-IC6 exist.
  - A0's executed violations per task, by rule, are reported (a reading).
- **IC2 (was bar E4.4):**
  - (a) code gates == veto clauses, exhaustive: 17 tools x 10 statuses x 2^13 rule-fact combinations = 1,392,640
    vectors; revision 2: **0 mismatches** (2026-10-21T05:25:49-05:26:23Z, 33.8 s; `results/equivalence_code_vs_veto_rev2.json`);
  - (b) the guarded TM is proved to deny whatever a veto clause denies: 13 of 13 per seed. With M = sum + 1 this is
    arithmetic.
- **IC3:** live `code_rules` == `veto_rules` on every logged verdict of every arm.
- **IC4:** live facts == the facts recomputed from the saved trajectory on every executed call. The calls are matched
  in order, and the sha256 of each fact record is compared.
- **IC5:** on every task where A1 denied nothing (both arms ending normally), A1's trajectory and final DB equal A0's.
  The DB is replayed as tau2's evaluator builds it.
- **IC6:** on every test task where the learned part made no learned-only denial, A2, A3 and A3t's trajectory and DB
  equal A1's.
- **Each check touches only some bars** (revision 3; re-review item 1; `summarize_e4.SCOPE`):

| Check | Touches | Why |
|---|---|---|
| IC1, IC3, IC4, IC5 | E4.2, E4.3, E4.5 | the facts, the gates and the replay pairing are shared by every arm |
| IC2 (a) | E4.2, E4.3, E4.5 | it tests A1's gate function (`rules.py`), which E4.2 runs and E4.3 and E4.5 compare against |
| IC2 (b), IC6 | E4.3, E4.5 | the learned part and the test arms only |

- A check that does not apply reads **"not applicable"**, never a failure: IC2 (b) when no learned part was trained (E4.3
  not testable) or no MILP check ran; IC6 when no test arm ran. Test-arm folders are read only after GO.
- In the end-to-end test on stand-in data, IC5 fires as it should: there, A0 and A1 are different trials, not replays
  (section 18.5).

### 9.1 What an integrity failure leads to (written now; decision 3; revision 4)

- **A failed check is a bug.** Revision 4 leaves no choice to anyone after the freeze: every step below is mechanical.
- **Integrity events, derived and never chosen:**
  - a check that reads False in the campaign's `summary.json` (IC1-IC6);
  - an integrity stop in its `run_e4_gpu.log` at a registered stage: exit 2 "identity_changed" when a call ends, at
    the stop of a call stopped before its own end check (revision 6: the end check at the stop, 4.1), or at the next
    start of a run whose earlier call still has no end record, with the pinned server up (the deferred end check,
    4.1); or exit 7 "pins_changed" when a stage starts (section 11.6; `run_e4_gpu.sh` tags every STOP line;
    `check_pins.stop_effects` reads them). The operator's stop (`[slot_end]`, exit 8) is not one, and neither is a
    server that is not the pinned one when a call starts (`[server]`, exit 2; revision 6: also at a resume, where it
    leaves the earlier call's record untouched).
- **The bars an event voids, by fixed tables:**

| Event | Void bars |
|---|---|
| a failed check | the bars its scope table names (above) |
| identity_changed in a run of A0_base or A1_base (or their re-runs) | E4.2, E4.3, E4.5 |
| identity_changed in a test arm's run (or its re-run) | E4.3, E4.5 |
| identity_changed in a judge call | none; that arm's judged readings are "not available" |
| pins_changed at stage S | the bars fed by the stage X that ran before S (the last "pins ok" line of the log before the stop): X's runs as above; train_guard: E4.3, E4.5; judge: none (the judged readings of the arms it judged are "not available"); the smoke (the freeze check came after it) and the summary: none |

- **A void bar is printed without its numbers.** `summary.json` and its table carry no difference and no interval for it.
  Its readings, the pass rates of an arm that feeds only void bars, and the go-bar counts are withheld too. No claim
  line uses it.
- **After an integrity stop that voids a bar, only judge and summary run in that campaign.** `run_e4_gpu.sh`'s
  integrity guard refuses every other stage (exit 7). No further test-split run is spent in a campaign whose bars are
  void, and the record below covers every event.
- **The record, after the campaign's summary, is always written when a bar is void** (fixed now, like
  the slots). First the cause is found and logged (the study log and the research log). Then
  `scripts/integrity_failure.py` writes `runs/<campaign>/integrity_failure.json`, once and read-only. It holds:
  - every event (from `summary.json` and the log; it refuses a summary older than the log's last integrity stop);
  - the void bars, the logged cause and the UTC time from the clock;
  - the files the logged fix changes (`--fix-changes`; none is a valid answer, for example a server restored to its
    pinned identity), and the files restored byte for byte (`--restored`; each must hash to its registered literal
    again).
- **The decision is mechanical** (the round-3 review's M1 (a)). The test is the logged list of files the fix changes,
  compared with every spec of H1-H5 (a file spec, or a file under a `tree:`, `dist:` or `venv:` spec). A byte-for-byte
  restoration is no change.
  - **No listed file is named in H1-H5: the one integrity re-run IS run**, in the next GPU slots (section 11.3), whatever
    the numbers. It is the campaign `<campaign>_icr` on a fresh store, every registered stage a0 to summary: a fresh
    store holds no replies to pair with, so every arm the voided bars compare must run in it (A1 replays A0, the test
    arms replay A1, and the TM is trained on its A0 and A1 rows). It runs the frozen code and pins (`check_pins.py`
    must pass), the same smoke outcome and the same cut.
  - **Otherwise** the void bars read "not shown (integrity failure)". A fix is a new registration revision, a new
    campaign and a raised test-split use count; the files the record lists are that revision's change, named in its
    draft history.
  - A cause that is not found is logged as such, with what was checked. Its fix changes no file, so the re-run is run.
- Then the summary of the campaign runs again (CPU): it shows the record's decision on every void bar.
- `check_pins.py` refuses every stage of `<campaign>_icr` unless the record exists, names the frozen campaign, requires
  the re-run, and its evidence holds now (every named check False in `summary.json`, every named stop a line of the
  log). Its a0 stage refuses a store that exists.
- **The re-run's bars replace the voided ones,** named as coming from `<campaign>_icr`. When only E4.3 and E4.5 were
  void, E4.2 stands from the first campaign and the re-run's E4.2 is a reading: the re-run's summary labels every bar
  the record did not void "a reading (the bar stands from <campaign>)" (revision 5; the round-4 review's m4).
- **If an integrity event occurs in the re-run,** its void bars read "not shown (integrity failure)" and no claim line of
  section 15 uses them. There is no second re-run.
- **The re-run is a second Qwen use of the retail test split:** the use count rises to 2 and is reported.
- The smoke's S7 (IC3 and IC4 on the smoke) is a smoke gate: a failure there stops E4 before the freeze.
- A MILP STOP at `train_guard` (exit 5) is IC2 (b): E4.3 and E4.5 are void; `judge` and `summary` still run for E4.2
  (section 11.6).

## 10. Bars (registered before any Qwen run; all paired)

**The registered statistic** (`summarize_e4.paired`):

- paired task differences (1 trial) in points;
- a percentile bootstrap over tasks: 10,000 resamples, `numpy.random.default_rng(20261021)`,
  `idx = rng.integers(0, n, size=(10000, n))`;
- the 95% interval is the 2.5th and 97.5th percentiles of the resampled means.

### 10.1 Every bar and reading, in one table

| Name | Kind | Arms, tasks | Rule | Prior: DESIGN, then ours now |
|---|---|---|---|---|
| **E4.2** | superiority, judge-free DB | A1 - A0, all 114 | met iff diff >= +3.0 points **and** the interval's lower end > 0 | about 45%; now about 30% (already informed by 2607.07405's retail control) |
| **E4.3** | superiority, judge-free DB; **low power** (12.0% at the go-bar's corner, 10.3) | A2 seed 1 - A1, the 40 test tasks, only after GO | met iff diff >= +2.0 points **and** lower end > 0; its power at the observed go-bar rates beside it | about 15%; now about 5% (stand-in, section 17) |
| **E4.5** | non-inferiority, judge-free DB | A2 seed 1 - A3, the 40 test tasks, only after GO | non-inferior iff lower end > -5.0 points (-2 tasks); "not informative" when fewer than 4 test tasks differ | - |
| E4.2 judged | reading | A1 - A0, 114 | none; "self-judged, unvalidated" | - |
| E4.2 without infra/timeout | reading | A1 - A0, tasks ending normally in both | none | - |
| E4.2 without the smoke tasks | reading | A1 - A0, 94 to 104 tasks (the 10 to 20 smoke tasks left out) | none | - |
| E4.3 / E4.5 two-level | reading | seeds 1-3 within A2, then tasks | none (beside the bar) | - |
| E4.3 / E4.5 judged, and without infra/timeout | reading | as the bar | none | - |
| A2 seeds 2, 3 - A1 (and - A3) | reading | 40 | none | - |
| A2 - tree (A3t) | reading | 40 | none | - |
| IC1-IC6 | integrity checks | section 9 | must hold; a failure voids the bars it touches (9, 9.1) | by construction |

### 10.2 E4.5 (review issue 7, decision 9)

- The margin is **-2 tasks = -5.0 points** (revision 1's -2.0 points was less than one task, 2.5 points, and identical
  arms met it).
- **"Not informative"** when fewer than 4 test tasks differ in DB outcome between A2 and A3. Then there is no verdict
  either way.
- Reported beside it: each twin's operating point on veto-clean write proposals of the test tasks. That is the learned
  part's deny rate on non-gold proposals and its allow rate on gold proposals, for A2 seeds 1-3, A3 and A3t.
- Power, from the table below: with 0 tasks where A3 is better, A2 must be better on 4. With 1, A2 needs 4. With 2, 6.
  With 3, 8. With 4 or more, no count up to 8 is enough.

### 10.3 Power of the registered statistic (`scripts/power_table.py`, `results/power_table_rev2.json`)

The smallest number of favourable task flips that meets each rule, by the number of flips against. Flips are laid out
as +1 on the first tasks, then -1, then 0. These numbers equal the review's own.

| Flips against | E4.2 (114 tasks): favourable needed, diff [interval] | E4.3 (40 tasks): favourable needed, diff [interval] |
|---|---|---|
| 0 | 4: +3.51 [+0.88, +7.02] | 4: +10.0 [+2.5, +20.0] |
| 1 | 7: +5.26 [+0.88, +10.53] | 7: +15.0 [+2.5, +27.5] |
| 2 | 9: +6.14 [+0.88, +11.40] | 9: +17.5 [+2.5, +32.5] |
| 3 | 11: +7.02 [+0.88, +13.16] | 10: +17.5 [+2.44, +35.0] |
| 4 | 13: +7.89 [+0.88, +14.91] | 13: +22.5 [+2.5, +42.5] |

- **E4.3 needs a net of about 4 or more test tasks** (10 points or more). Every report says so.
- The stand-in predicts about 0 (section 17), hence the go-bar.
- **E4.3 is low power** (revision 3; decision 1; `scripts/power_e43.py`, `results/power_e43_rev3.json`).
  The exact power of E4.3's rule, when each of the 40 test tasks flips favourably with probability g = G/74 and
  against with l = L/74. These are the go-bar's train-task rates read as per-task rates, as a pre-freeze review read them;
  both are upper bounds, since a learned denial need not flip a task. The sum is exact over the trinomial (the
  binomial with flips against added; with nothing against it is the binomial itself). need(a), the favourable flips
  the rule needs with a against, is `results/power_table_rev2.json`'s E4.3 column for a = 0-6 (the table above shows
  0-4) and the same function beyond it:

| Where | G, L (of 74) | Power of E4.3 |
|---|---|---|
| the go-bar's corner (fewest favourable, most against, still GO) | 8, 4 | 12.0% |
| the go-bar with nothing against | 8, 0 | 64.1% |
| the boundary G - L = 4, from G 9 to its far end G 39 (G + L <= 74: G counts A1-failed, L A1-passed train tasks) | 9, 5 ... 24, 20 ... 39, 35 | 10.7% ... 5.8% ... 4.2% (the lowest; `results/power_rev4.json`) |
| the net-rate reading of a pre-freeze review (G - L = 4 with nothing against; not a GO point) | 4, 0 | 16.9% |
| the stand-in's observed go-bar (gpt-4.1 trial 0 as A1; section 17) | 3, 5 | 0.6% |

- **The effect needed for 80% power** (the net favourable rate g - l):
  - nothing against: +13.4 points (5.3 net tasks of 40);
  - 2.5% of the tasks against: +20.4 points (8.2 net tasks);
  - 4/74 against (the go-bar's corner): +24.6 points (9.8 net tasks).
- A Monte Carlo check (the registered statistic on random flip layouts, 2,000 draws each) agrees: G 8, L 4:
  11.0% (se 0.7%) against the exact 12.0%; G 8, L 0: 64.5% against 64.1%; net 4 of 74: 17.1%
  against 16.9%; the 80% point: 79.5% against 80.0%.
- **E4.2's power** (revision 4; the round-3 review's m1; `scripts/power_rev4.py`, `results/power_rev4.json`). The same
  model on the 114 tasks: each task flips +1 with probability g and -1 with l. need(a) is E4.2's column of the table
  above for a = 0-6 (asserted) and the same function beyond it. The effect needed for 80% power (the net rate g - l):
  - nothing against: +4.8 points (5.5 net tasks of 114);
  - 2.5% of the tasks against: +9.8 points (11.2 net tasks);
  - 5% of the tasks against: +12.1 points (13.8 net tasks).
  A Monte Carlo check of the first (2,000 draws): 79.0% (se 0.9%) against the exact 80.0%. An independent recompute in
  the pre-freeze review gave the same three figures.
- **At the observed rates:** at STAGE=train_guard, `gobar.py` writes the exact power at Qwen's observed G and L (TM
  seed 1, LR and the tree) into `gobar.json` ("power"). The summary puts it on E4.3's row, and every E4.3 sentence
  carries it (section 15).

### 10.4 The E4.3 go-bar (review issue 4, decision 5; fixed now, before any Qwen row exists)

- At `STAGE=train_guard`, take the out-of-fold predictions on **A1's executed train-task writes**. Use the
  `train_guard.py` folds (5 folds over train tasks, seed 20261021) and the TM at the CV pick with seed 1. Each write is
  predicted by the model of the fold its task belongs to (`scripts/gobar.py`).
  - **G** = the number of A1-failed train tasks (DB 0) where the learned part denies at least 1 executed **non-gold**
    write.
  - **L** = the number of A1-passed train tasks (DB 1) where it denies at least 1 **gold** write.
- If a fold's training part lacks a class, that fold has no model. Its writes count as not denied (conservative for
  GO), and the count is reported (`models/oof_rows.json`, `meta.rows_in_folds_without_a_model`).
- **GO iff G >= 8 (of 74) and G - L >= 4.**
- **The go-bar is kept as it is** (decision 1, 2026-10-21T07:24:12Z), **and it is not a power gate.** At
  its corner (G 8, L 4) E4.3's power is 12.0%; along the boundary G - L = 4 it falls to 5.8% at G 24, L 20, and down to
  4.2% at its far end, G 39, L 35 (G + L <= 74); at G 8, L 0 it is 64.1% (10.3). A GO near the bar still leaves E4.3 a
  likely miss.
- **Why it is kept:** decision 1; this text adds no reason of its own. The facts beside the decision:
  - E4.3 stays a supporting rule, labelled low power, and every E4.3 sentence carries its power figure (section 15),
    so a miss near the bar is never read as evidence of no effect;
  - the retail test split's first Qwen use is spent whatever the go-bar says: A0 and A1 run on all 114 tasks for E4.2.
    A GO near the bar adds only the A2, A3 and A3t runs on the 40 test tasks (about 44 GPU minutes, section 19);
  - the other option, a go-bar at the power level (for example G - L >= 10 of 74, about +13.5 points), was
    offered and not taken.
- Otherwise **NO-GO**: E4.3 and E4.5 become a dev-only line, and A2, A3 and A3t are **not** run on the test tasks.
- Reported beside it, no rule:
  - G and L for LR and the tree;
  - the share of the learned part's denials that **persist after a simulated re-confirmation**.
- **The simulated re-confirmation** (`bits.reconfirm`, fixed now). A learned denial tells the agent to re-check and
  re-confirm (`models._denial`). Simulated: the agent re-lists the action with its order id, items and new items, and
  the user answers yes.
  - The bits set: `user_yes` 1, `user_negation` 0, `action_word_listed` 1, `order_id_listed` 1 (order writes),
    `items_listed` 1 (item tools), `new_items_listed` 1 (item changes), `user_turns_ge2` 1.
  - All else is unchanged.

### 10.5 E4.2 readings (review issue 0)

- E4.2 is also read without the smoke tasks: the first 10 train tasks, or up to 20 after the extension rule of S0 (the
  freeze record's `smoke_tasks`). Their transcripts are seen at the label check (S8), in the smoke's own campaign; the
  registered A0 re-runs them fresh. This is a reading with no rule.

### 10.6 The two-level interval (review issue 6, decision 4)

- `summarize_e4.two_level`, the decision layer's convention:
  - per resample, each arm's seeds are drawn with replacement (A2: 3 seeds; A1 and A3: 1 each);
  - then the tasks are drawn with replacement;
  - the difference is the mean over the drawn tasks of (A2 averaged over its drawn seeds) minus the other arm.
- 10,000 resamples, seed 20261021. Reported beside E4.3 and E4.5.
- A bar met only on seed 1 is reported with this interval beside it.
- If fewer than two A2 seeds hold every test task (the budget cut A2 seeds 2-3, or a seed arm is incomplete), the
  reading is "not available (seeds cut or incomplete): seeds available [...] (need 2 or more)", the code's own text
  (`tests/test_summary.py` compares the two).

### 10.7 Reported beside the bars (no rule)

- Per arm: ends by class, tau2 re-runs, denials by rule, learned-only denials, fallbacks, agent model calls and tokens
  per task, judge errors.
- The strict E4-R3 monitor count (how often revision 1's reading would have denied).
- Executed violations by rule (A0).
- False blocks: gold-matching writes among the denied ones.

## 11. The report step (from the operational sections)

### 11.5 The report step (exact commands; this block is, line for line, the last comment of `run_e4_gpu.sh`; revision 6)

```text
THE REPORT STEP (registration revision 6, section 11.5: the same text, line for line; tests/test_report_step.py)
cd studies/agent_guards/e4
C=e4reg_<yyyymmdd>    # the one campaign name: e4reg_ and the UTC date of the smoke; its smoke is SMOKE_$C
# 1. the smoke (E4-0), before the freeze and with no slot: its decision reads PENDING until step 2
CAMPAIGN=$C STAGE=smoke bash scripts/run_e4_gpu.sh
# 2. the freeze (section 11.4): the labels into runs/SMOKE_$C/confirmations_labels.json, the slot, then
scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/smoke_report.py --campaign SMOKE_$C \
    --slot-minutes <slot> --out runs/SMOKE_$C/smoke_report.json
#    on GO: REGISTRATION_E4.md = this file with H6, the F table and the status line filled (section 11.4), then
scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/freeze_e4.py --registration REGISTRATION_E4.md
# 3. the registered stages, in order; the loop stops after a failed stage (a non-zero exit)
for s in a0 a1 rerun train_guard a2a3 rerun judge summary; do
  CAMPAIGN=$C STAGE=$s bash scripts/run_e4_gpu.sh || break
done
# 4. after a stop: section 11.6, by the exit code. After exit 2, 3, 4, 6 or 8 the cause is logged, then the loop of
#    step 3 runs again with its list starting at the stopped stage (after exit 8: in the next GPU slot); a stage
#    with 3 failed attempts (exit 3 or 6) logs SKIP [stage_failed] and exits 0, so the loop goes on. The operator
#    stops a stage with kill -TERM <pid of run_e4_gpu.sh> or Ctrl-C, never the server (exit 8, never counted).
#    After an integrity stop (section 9.1) only judge and summary run:
for s in judge summary; do
  CAMPAIGN=$C STAGE=$s bash scripts/run_e4_gpu.sh || break
done
#    a void bar in runs/$C/summary.json: the cause found and logged, then the record and its mechanical decision
scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/integrity_failure.py --campaign $C \
    --cause '<the logged cause>' --fix-changes <every file the logged fix changes; none is a valid answer>
#    the summary of $C again (it shows the decision); when the record requires it, every stage of ${C}_icr
CAMPAIGN=$C STAGE=summary bash scripts/run_e4_gpu.sh
for s in a0 a1 rerun train_guard a2a3 rerun judge summary; do
  CAMPAIGN=${C}_icr STAGE=$s bash scripts/run_e4_gpu.sh || break
done
# 5. the outputs: runs/$C/summary.json ("table": E4.2, E4.3 with its power, E4.5; "readings"; "integrity" with
#    "checks", "stop_events" and "by_bar"; "gobar"; "arms"; "anomalies"); runs/$C/gobar.json;
#    runs/$C/models/train_report.json (meta.cv.picks, meta.oof_at_picks); runs/SMOKE_$C/smoke_report.json;
#    runs/E4_frozen.json; runs/$C/integrity_failure.json (only after an integrity failure); the pins checks
#    runs/$C/pins_check_<stage>.json; the arm folders runs/$C/{A0_base,A1_base,A2s1_test,A2s2_test,A2s3_test,
#    A3_test,A3t_test}[_rerun]/ with results.json, guard_log.jsonl, manifest.json, run_summary.json, judged.json
```

- `tests/test_report_step.py` checks that this block and the script's last comment are the same text, that there is
  one campaign name and that every stage loop stops after a failed stage.

## 13. Departures from the design of E4 (all deliberate)

1. **Primary score:** the judge-free DB pass^1; the judged score is a reading only (review decision 0). DESIGN said
   "tau2 1.x's own reward", pending the judge check (section 3).
2. **E4-R3's form.**
   - DESIGN's draft rule: the listing names "the action's order and item ids" before any write.
   - Revision 1: the order id on every order write plus the zip.
   - Revision 2: the text-literal reading. The order id is required before a cancel and a return only; a yes is
     required before every write, answering an agent text sent after the user's request; no zip check.
   - The reason is the policy text (section 5.1).
3. **E4-R10 added** (policy.md:112, :132). The exchange tool accepts a same-item exchange.
4. **A3t (the tree twin)** added as a reading. DESIGN's arms list LR only.
5. **One call per message** in every arm, A0 included (section 4.4).
6. **The judge runs post hoc,** not inline, so a non-JSON reply can never re-run a simulation.
7. **E4-R2's order-owner check reads the order's own record only.**
8. **Training rows:**
   - DESIGN named "the train split's executed writes".
   - Revision 2 uses the veto-clean ones, deduplicated across A0 and A1 (review decision 6).
9. **`--seed 300`** against DESIGN 1.7's seed 0. 300 is tau2's own default (`config.DEFAULT_SEED`), from which tau2
   derives the trial seed 626729. At temperature 0 decoding is greedy, so the seed should not change replies. Kept for
   comparability with tau2's default runs.
10. **`-np`:** revision 1 reopened it. Revision 2 returns to DESIGN 1.7's `-np 1` (no departure now; listed for the
    record).
11. **Stage 2b** (a matcher revision after A0) was added in revision 1 and is deleted in revision 2 (no departure
    now).
12. **The fallback after 3 retries** (a fixed message to the user). DESIGN does not say how a denial loop ends; it must
    end.
13. **CV on balanced accuracy.** DESIGN names no CV metric. DENY rows are the minority.
14. **A2 runs TM seeds 1-3 on the test tasks** (review decision 4). DESIGN implies one A2 run.
15. **E4.5's margin is -5.0 points (-2 tasks)** with a "not informative" rule. DESIGN said -2.0 points, which is less
    than one task.
16. **The E4.3 go-bar** (review decision 5). DESIGN has none.
17. **E4.1 and E4.4 are integrity checks, not bars** (review issue 5).
18. **The smoke uses 10 train tasks in its own campaign,** with gates S0-S9. DESIGN 2.5 sets only E2-0's parse-rate
    bar.
19. **Missing and failed simulations** follow rule 9 (section 3.3). DESIGN says nothing about them.
20. **What follows an integrity failure or a stop** (sections 9.1 and 11.6, revisions 3 and 4): void bars by fixed
    tables, printed without their numbers; only judge and summary after an integrity stop; the write-once record; the
    one integrity re-run on a fresh store, run whenever the logged fix changes no pinned file; then "not shown". DESIGN
    says nothing about either.
21. **S8 with fewer than 10 candidates is NO-GO** (decision 6). DESIGN has no matcher check.
22. **The smoke's extension rule** (revision 4): 10 to 20 train tasks, by the count of label candidates only. DESIGN
    2.5 sets a smoke of fixed size.

## 14. Prior art

- **The design's section 3** (searched 2026-10-21, 00:48-01:01 UTC) covers browser and computer-use guards.
- **For tau2 and retail, the dated check:** `lit_tau2_retail_2026-10-21.md` (searched 2026-10-21 05:17-05:30 UTC;
  web search only; the arXiv and DBLP indexes were not searched).
- **Run-time rule enforcement on tau-bench / tau2-bench is a known kind of method:**
  - Agent-C (arXiv 2512.23738): 100% conformance on retail and airline;
  - deterministic gates (2607.07405): tau2 airline, gpt-4o-mini 29.6% to 42.0%;
  - PolicyGuard (2606.29225): airline;
  - PolicyGuide (2608.19861): airline, retail and telecom; mean Pass^4 0.42 to 0.62;
  - solver-aided compliance (2603.20449);
  - ToolGuard (EMNLP 2025 industry track).
- **New and directly relevant:** 2607.07405 section 6.1 used retail as a **negative control**. "On eight retail tasks,
  a retail gate suite moves performance from 40/64 (62.5%) to 37/64 (57.8%)", an interval that includes zero, because
  "tools already enforce many relevant preconditions". The gates, the model and the seeds of that section are not
  named.
  - **The paper's own caveat:** the drop "is largely explained by an encoding bug that blocked corrected retries the
    underlying tool would have accepted" (as read, `lit_tau2_retail_2026-10-21.md`). And the control covers only eight
    tasks. So it says little about gates on rules the tools do not enforce.
  - Our probes show retail's tools do not enforce E4-R1 (reads, `modify_user_address`), E4-R2, E4-R3, E4-R4, E4-R5
    after modify-items, and E4-R10 for exchanges. E4.2's gain can come only from these.
- **No published tau2 retail number of Qwen3.5-9B** was found. Its model card's "TAU2-Bench 79.1" is self-reported:
  thinking on by default, the official user simulator, and no domain mix stated. We use none of it as a comparison.
- **So A1 is a known kind of baseline.** E4 measures it with a local 9B model and asks only whether a learned TM veto
  adds anything.
- No TM work on agent guards was found in DESIGN 3.7's search (2026-10-21).

## 15. What a result may be called (DESIGN 2.6, completed)

| Result | It may be called | Never |
|---|---|---|
| E4.2 met, E4.3 not met | "Code gates for 10 of retail's written rules (E4-R1 to E4-R10, E4-R3 in its text-literal reading) raised a local 9B agent's judge-free pass^1 by N points on tau2-bench v1.0.1 retail (interval [a, b]); the learned TM veto showed no gain on top (E4.3: M points, interval [c, d]; low power: P% at the go-bar's observed rates)." | "the TM enforces the policy better than code" (on E4-R1 to E4-R10 they are the same function); "policy-compliant agent" (the text rules are not enforced); "provably compliant" (the proof covers facts to verdict; the yes-matcher and id matching are heuristics) |
| E4.2 not met | "Code gates for 10 of retail's written rules did not raise a local 9B agent's judge-free pass^1 (N points, interval [a, b])." | any softer wording |
| E4.3 met | "On 40 held-out retail tasks, a learned TM veto on trusted facts added N points of judge-free pass^1 over the code gates (interval [a, b]; two-level interval with TM seeds 1-3 [c, d]; low power: P% at the go-bar's observed rates); a supporting result under E2's claim." | "the TM understands the policy"; a judged-only result as the headline; a standalone headline (E4.3 is a supporting rule); "first" |
| E4.3 computed, not met | "On 40 held-out retail tasks, a learned TM veto showed no gain over the code gates (N points, interval [a, b]; low power: P% at the go-bar's observed rates)." | "the TM veto does not help" or any wording that reads a miss as evidence of no effect |
| E4.3 NO-GO | "Not tested on the test split: the dev go-bar was not met (G = x, L = y; E4.3's power at these rates: P%)." | any test-split wording |
| E4.3 not testable | "Not testable: fewer than 10 veto-clean training rows of a class; the learned part was not trained (E4.3 is low power: 12.0% at the go-bar's corner, G 8, L 4; no rates were observed)." | any test-split wording |
| a bar void after an integrity failure | "Void in campaign c (integrity failure: the checks); from the integrity re-run c_icr: ..."; for E4.3 with its power figure, as in the E4.3 rows above | any number of the void campaign as a result |
| in `c_icr`, a bar that c's record did not void | "a reading (the bar stands from c)": the bar is c's (revision 5) | the re-run's number as the bar |
| a bar not shown | "Not shown (integrity failure)", "not shown (pins changed after the freeze)", "not shown (stage failed)" (an arm left incomplete by a stage after its third failed attempt; train_guard's: E4.3 and E4.5; 11.6) or "not shown (not run)" (a stage that has not run yet, 11.3); for E4.3 with its power figure ("low power: P% at the go-bar's observed rates", or 12.0% at the go-bar's corner when no rates were observed) | any number for it |
| E4.5 non-inferior | "A2's learned part was non-inferior to LR's at -5 points (2 tasks) on 40 tasks; deny and allow rates: ..." | "beats LR" (non-inferiority only); "better than LR" |
| E4.5 not informative | "Not informative: fewer than 4 test tasks differed between A2 and A3." | any verdict |
| any judged number | "self-judged by the same local model, unvalidated" | a comparison with leaderboard numbers (other agent setup, user simulator and judge); the model card's 79.1 |
| any | - | "first"; "state of the art"; "beats" without a met superiority bar |

## 16. Provenance and selection caveats

### 16.1 Provenance (review issue 16)

- **Qwen also plays the user.** DESIGN says this biases only absolute scores. But the guard's cost depends on how the
  simulated user answers a re-confirmation request, so it **can bias the paired comparison too** (A1 against A0, A2
  against A1).
- **Qwen3.5's training data may include tau-bench-style retail tasks.** This cannot be checked.
- **The judge is the same model as the agent:** the judged score is self-judged and unvalidated.
- **Not comparable with leaderboard numbers** or with the model card's TAU2-Bench 79.1.

### 16.2 Selection on stand-in data (review issue 14)

- These were designed after reading gpt-4.1 trajectories on the old task versions:
  - the 27 learned-part bits;
  - E4-R3's order-id-only form (revision 1) and the text-literal change (revision 2, for the policy text, with the
    stand-in counts on screen);
  - the yes-matcher.
- The s and C grids were widened after the stand-in CV picked the edge, **while the stand-in test-task metrics were on
  screen** (study log, 02:28-02:31Z).
- So: **selection on stand-in data (another model, old task versions); stand-in test-task numbers were visible.**
- **The go-bar is optimistic:** its G and L use the same folds on which the CV chose s, C and the depth.
- **Local only, tested:** `e4_run.py` refuses any endpoint that is not 127.0.0.1 or localhost (the dry run's
  non-local refusal, exit 2), and every LLM-calling script sends all other HTTP to a dead proxy with no API key
  (`e4_run.local_only_env`). No paid service is used.

## 17. What was known before the first Qwen run (each item: already informed by ...)

**Priors and earlier results:**

- DESIGN's priors: E4.2 about 45%, E4.3 about 15% (our guesses).
- *Already informed by 2607.07405 (airline):* deterministic gates took gpt-4o-mini from 29.6% to 42.0%
  (self-reported).
- *Already informed by 2607.07405 section 6.1 (retail):* gates were a negative control, 40/64 to 37/64 with an interval
  including zero. Retail's tools already self-enforce many rules. But the paper itself says the drop "is largely
  explained by an encoding bug that blocked corrected retries the underlying tool would have accepted", and the control
  covers only eight tasks. It weighs only lightly toward a small or null E4.2.
- *Already informed by our taubench probes:* the tools refuse breaks of E4-R6, E4-R8 and E4-R9 and most of E4-R5 and
  E4-R7. So E4.2's DB gain must come mainly from E4-R1 to E4-R4, E4-R5 after modify-items, and E4-R10.

**Stand-in numbers seen** (gpt-4.1, old task versions, code tests):

- *Revision 1 (E4-R3's cost):* the strict E4-R3 would deny 145 of 534 writes in reward-1 runs (27%), 172 by any rule. The
  review counted 126 order-id/zip-only, 106 of them on exchange and modify writes.
- *Revision 2 (E4-R3's cost):* the text-literal E4-R3 would deny 35 of 534, 81 (15.2%) by any rule. This
  informed only the wording of the declaration; the reason for the change is the policy text.
- *Revision 1 (the stand-in models):* on 328 test-task rows, TM balanced accuracy 0.633 / 0.669 / 0.672 (seeds 1-3), LR
  0.656, tree 0.726; TM seed 1 DENY recall 0.32, allow_kept 0.94.
- *Revision 1 (gold-episode losses):* A2 2, A3 14 (DB 1.0 on 100/114), A3t 44 (70/114). The TM denied 4 of 71
  plausible-wrong writes and does not separate the right option or payment.
- *Revision 2 re-run* (veto-clean deduplicated rows; section 18): rows after the veto-clean filter (146 of gpt-4.1's 743 executed writes fire a veto) and deduplication: train 590 (96 DENY), test 282. The CV picked TM s 24 (the top of the frozen grid again; the grid is NOT widened), LR C 10, tree depth 6. Out of fold at the picks: TM seed 1 balanced accuracy 0.601, DENY recall 0.281, allow_kept 0.921; LR 0.549 / 0.146 / 0.951; tree 0.693 / 0.469 / 0.917. Test-task rows (descriptive): TM seeds 1-3 balanced accuracy 0.634 / 0.671 / 0.636, LR 0.564, tree 0.667. Scripted gold episodes with DB 1.0: A2 85/114, A3 101/114, A3t 69/114 (revision 1: 112, 100, 70); plausible-wrong episodes with a denial: A2 42/71, A3 20/71, A3t 56/71.
- *Revision 3, E4.3's power (no data):* 12.0% at the go-bar's corner, 0.6% at the stand-in's go-bar (G 3, L 5);
  80% power needs +13.4 points net with nothing against (section 10.3).
- *Revision 4, E4.2's power and the go-bar's whole boundary (no data):* E4.2's 80% power needs +4.8 points net with
  nothing against, +9.8 with 2.5% against, +12.1 with 5% against; E4.3's power on the boundary G - L = 4 falls to 4.2%
  at G 39, L 35 (sections 10.3, 10.4). The round-3 review, read before this revision, gave the same figures.
- *Revision 2 stand-in go-bar* (code test, gpt-4.1 trial 0 as "A1"): G 3, L 5 -> NO-GO (G - L = -2; 11 learned denials, 7 of them persist after a simulated re-confirmation). This predicts NO-GO on
  Qwen too.

**Predicted outcomes:**

- IC1 and IC2 pass by construction.
- E4.2: our guess is small or null, possibly negative from E4-R3's friction (the tools already enforce several
  rules; the 2607.07405 control weighs only lightly, as above).
- E4.3 is NO-GO at the go-bar, or a miss (low power: 12.0% at the go-bar's corner, 0.6% at the stand-in's
  observed rates).
- E4.5, if run, is "not informative" or decided by the twins' operating points. Stand-in revision 1: LR blocked 12
  more gold episodes than A2; stand-in revision 2: A2 (s 24) blocked 16 more than LR.

**Earlier reads of the retail TEST tasks (none by Qwen):**

1. The taubench folder rescored the shipped gpt-4.1 retail file, all tasks (non-Qwen).
2. `e4/.tmp/explore_gpt41.py` (sha256 da367e104134eafa9a1956e6111b00ecc00a5ecc55d65a19e9d1788f47b18980) read the
   pre-write user messages of all 456 stand-in simulations, test tasks included, while the guard's code was written
   (study log entry 2026-10-21T01:46:39Z, "Design facts found while writing it"; its own run was not logged, and its
   file time is not used).
3. The offline scripted episodes (revisions 1 and 2) used every task's gold actions, test tasks included. That covers
   the gold plans, the violation plans and the plausible-wrong writes. Their DB rewards (114/114 gold) are code checks.
4. The stand-in training scored 328 test-task rows (revision 1) and revision 2's re-run scored them again
   (descriptive).
5. `check_judge_tasks.py` read every task's assertions and ran gold episodes.
6. In revision 2, attempt 1 of the offline A0 re-run listed test task 36 among the 4 planted writes that passed. Only
   train tasks 20, 22 and 37 were inspected.

**The coming reads:**

- **Seen before the freeze, by design: everything in `smoke_report.json`** (pinned by H6). These are Qwen numbers on the
  10 to 20 smoke train tasks: the label candidates per task (S0's extension rule), denials by rule, recoveries and
  fallbacks, the count of tasks whose final DB differs from A0's, A0's executed writes per task and their non-gold and
  veto-clean shares, the projected DENY rows, the parse rate and
  the other tool-call health counts, the rule firing by tool, seconds and tokens per call, the repeat check's
  difference rate, the judge's health, and the matcher's precision and recall on the labelled candidates. No reward,
  pass rate or judged score is in it. The smoke may change only the budget (the cut), stop/go and its own size (S0's
  rule, by candidate counts).
- **This is the first Qwen use of the retail test split.**
- A0's and A1's outcomes on the 40 test tasks are seen before stage 5 by design (stages a0 and a1 print `pass1_db` over
  all 114 tasks). So are the test arms' own (revision 5; the round-4 review's m1): each of A2 seeds 1-3, A3 and A3t
  prints its `pass1_db` on the 40 test tasks when it ends (into `run_e4_gpu.log`), before the summary. **No choice uses
  them:** training, CV and the go-bar use train tasks only, and every step after an integrity failure (9.1; revision 4)
  and after any stop (11.3, 11.6; revision 5) is mechanical. Revision 6 (the round-5 review's Major 1): a resume that
  meets a server that is not the pinned one is a `[server]` stop, never an integrity stop, so no operator step at a
  resume can move a bar to the integrity re-run.
- **BG5** (the speed queue's server check, 4.1) gets no E4 request bodies, so no Qwen reply to an E4 request exists
  before the smoke (revision 5; the round-4 review's m3).
- `dump_call_bits.py` reads every A0 and A1 call's bits, test tasks included, for the MILP domain pre-check. These are
  bits only: no label, no reward.
- A2, A3 and A3t run on the 40 test tasks only after GO.
- The new use count of the retail test split by Qwen: 1 (this registration); 2 if the integrity re-run of section 9.1
  runs.

**Server facts seen:** the server's CPU smoke `/props` (04:02Z): server metadata only, no task output.

