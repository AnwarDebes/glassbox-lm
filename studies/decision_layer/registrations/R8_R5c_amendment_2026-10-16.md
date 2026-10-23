# R8 and R5c amendment, written 2026-10-16T11:34:06Z, before any R8 or R5c result was opened

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

No change to the arms, runs or rules.

- **Source changes after the freezes (R8 11:07:16Z, R5c 11:07:45Z):** src/dl/tm.py gained `budget_select` / `select_k` (11:30). It only acts when a literal budget or a negated-literal cap is set, which no R8 or R5c arm uses, and its default is the old rule. R8 and R5c jobs that started after 11:30 import this tm.py; their code path is unchanged. scripts/report_agent.py gained `r5c()` (the frozen R5c rules, written after the freeze). scripts/confirm_router.py, tm_explore.py and rules_check.py changed for other registrations.
- **R8:** the rule baseline `rule2` is tuned by coordinate ascent on the GPU and takes about an hour under the present load; nothing else.

## Notes added 2026-10-16T14:31:51Z, after the results (internal audit 4)

- R8's test episodes are the CLINC150 test queries (4,500 in scope, 1,000 out of scope) that R6 used as its episodes and that the routers of R2, R4, R4b, R7, R7b and R7c scored. R8's arms and rules were chosen on val only; this note completes the disclosure.
- R8's "rule2" was not explored before the freeze (only e22b_rule was); its val score first appeared in the registered run, where the registered val rule selected it.
- R8.2's name, "the TM learns when to stop asking", says more than its result: the test is met by out-of-scope episodes alone, and in scope the TM equals always asking (+0.006 [-0.003, +0.014]).
