# agent_guards: rule and Tsetlin machine guards around a local LLM agent

Can written rules, enforced in code, and a Tsetlin machine veto learned on top of them make a local LLM agent follow
a policy or resist prompt injection? Two registered tests on public agent benchmarks, tau2-bench retail (E4) and
AgentDojo (E2), and a first veto check on WebGuard. Every LLM role runs on a local Qwen3.5-9B (llama.cpp); no paid
service is used, and every script sends other network traffic to a dead local proxy.

**Status:** E4 done (its headline bar not met); E2 incomplete (it stopped at its second stage); WebGuard not shown.
Numbers in [docs/results.md](docs/results.md).

## Files

| Path | What |
|---|---|
| `e4/` | E4: `REGISTRATION_E4.md`, `e4guard/` (facts, rules, bits, models, the scripted user), `scripts/`, `tests/`, `pins/`, `results/` (final offline and stand-in rows), `runs/` (the freeze record and the campaign's summary and checks) |
| `e2/` | E2: `REGISTRATION_E2.md`, `src/e2guard/`, `scripts/`, `tests/`, `logs/` (stand-in chains), `runs/` (the freeze record, the smoke, the budget and the stop records) |
| `webguard/` | the WebGuard veto test: plan, split, features, runs and its evaluation |
| `llm/` | the llama.cpp build, the GGUF conversion and its check against the original weights, the shared server and its tests |
| `taubench/`, `agentdojo/`, `miniwob/` | offline harness scripts for tau2-bench v1.0.1, AgentDojo v1.2.2 and MiniWoB++ (with `POLICY_RULES.md` and `LICENCES.md`) |

## Running

Each benchmark runs in its own virtual environment beside its folder (`taubench/.venv`, `agentdojo/.venv`), installed
offline from the upstream releases; the TM side runs in `../decision_layer/.venv-dl`. E4's scripts run through
`e4/scripts/offline_env.sh`. The benchmark data (downloaded into `*/data_untrusted/`), the model files, the trajectories
and the run folders are not distributed.
