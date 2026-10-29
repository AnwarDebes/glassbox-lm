"""Dump user tasks (id, prompt, ground-truth calls) and injection tasks (id, goal, ground-truth calls) of a benchmark
version to JSON. Ground truth is computed on the default environment with no injections. Read-only use of agentdojo.
Usage: python -I -B dump_tasks.py VERSION OUT.json"""
import json
import sys

from agentdojo.task_suite.load_suites import get_suites


def calls(task, env):
    return [{"function": c.function, "args": c.args} for c in task.ground_truth(env.model_copy(deep=True))]


version, out = sys.argv[1], sys.argv[2]
res = {}
for name, suite in get_suites(version).items():
    env = suite.load_and_inject_default_environment({})
    res[name] = {
        "user_tasks": {t.ID: {"prompt": t.PROMPT, "gt": calls(t, env)} for t in suite.user_tasks.values()},
        "injection_tasks": {t.ID: {"goal": t.GOAL, "gt": calls(t, env)} for t in suite.injection_tasks.values()},
    }
with open(out, "w") as f:
    json.dump(res, f, indent=1, default=str)
print({k: (len(v["user_tasks"]), len(v["injection_tasks"])) for k, v in res.items()})
