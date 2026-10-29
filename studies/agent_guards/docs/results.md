# Results of the agent_guards study

Rule guards and Tsetlin machine guards around a local LLM agent (Qwen3.5-9B Q8_0 through llama.cpp), on two public
agent benchmarks and one web-safety set. Each number is read from the file named beside it.

## E4: tau2-bench retail, code gates over the written retail rules (registered; frozen 2026-10-23T01:27:37Z)

Campaign `e4reg_20261023` (`runs/e4reg_20261023/summary.json`); every stage ended with exit 0 at its first attempt.

| Bar | Outcome | Numbers |
|---|---|---|
| E4.2 (code gates A1 against the plain agent A0, judge-free DB pass^1, 114 test tasks) | **not met** | A1 66.67% against A0 68.42%: -1.75 points, 95% [-4.39, 0.00]; 2 tasks differ, both in A0's favour |
| E4.3 (the learned TM veto on top of the gates) | not run on the test tasks | NO-GO at the dev go-bar (TM seed 1: G = 4, L = 4; E4.3's power at these rates 2.3%), so A2, A3 and A3t did not run |
| E4.5 | not run | the same go-bar |

Beside the bars: the gates stopped the executed rule violations of A0 (17 in A0: E4-R3 8, E4-R9 4, E4-R10 2, E4-R5 2,
E4-R1 1; 0 in A1), and the readings without the infrastructure or timeout tasks and without the smoke tasks give the
same -1.75 and -1.92 points. The judged score is a reading only ("self-judged by the same local model, unvalidated").

## E2: AgentDojo v1.2.2, a TM tool guard against prompt injection (registered; frozen 2026-10-23T01:47:25Z)

**Incomplete.** E2-1 (the plain agent D0) completed. E2-2 (the D1 tool filter) stopped on 2026-10-26 at 14:54:37Z and
its one rerun at 15:10:51Z, both with exit code 3 at the same point (`runs/stopped_stages.jsonl`). By the frozen
registration a stage that stops twice is incomplete and E2 stops there, so no E2 bar has an outcome. The blind smoke
E2-0 passed its gate (`runs/E2-0_gate.json`, `runs/E2-0_summary.json`).

## WebGuard veto test (dev plan; CPU; a first guard check, not a benchmark)

`webguard/results/wg_eval.txt`: on 253 test rows from 18 sites of our own site split, the TM (s 16, seeds 1-3) reaches
accuracy 63.77 and logistic regression 64.03; TM - LR -0.26, site-bootstrap 95% interval [-11.11, 10.78], so the bar
(lower end above -5) is **not shown**: too few sites.

## The local LLM

- The GGUF conversion of Qwen3.5-9B (Q8_0, llama.cpp b11742) against the original weights in fp32 (`llm/hf_ref_ppl.json`):
  identical token ids (21,480), largest relative perplexity difference 0.0055 over 8 chunks: "conversion OK".
- Tool calls on our own 10-tool smoke set (`llm/tool_smoke.jsonl`): 50 of 50 with valid JSON, the required arguments
  and the expected values.
