"""Dump every tool of the AgentDojo suites (name, docstring, parameters with types and descriptions) to JSON.
Read-only use of the installed agentdojo package. Usage: python -I -B dump_tool_docs.py VERSION OUT.json"""
import json
import sys

from agentdojo.task_suite.load_suites import get_suites

version, out = sys.argv[1], sys.argv[2]
res = {}
for name, suite in get_suites(version).items():
    tools = []
    for t in suite.tools:
        schema = t.parameters.model_json_schema()
        tools.append({"name": t.name, "description": t.description, "parameters": schema.get("properties", {}),
                      "required": schema.get("required", [])})
    res[name] = tools
with open(out, "w") as f:
    json.dump(res, f, indent=1)
print({k: len(v) for k, v in res.items()}, sum(len(v) for v in res.values()))
