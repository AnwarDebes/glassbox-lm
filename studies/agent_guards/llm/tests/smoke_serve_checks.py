"""Checks for the CPU smoke of serve_qwen_agent.sh against its running llama-server (the real model): /props and
/v1/models, the two template renderings (/apply-template), a short reply sent without the thinking switch (like E2's
D1 filter call), and one tool call sent with enable_thinking=false (like E2's and E4's agent calls).
One PASS/FAIL line per check; all of it goes to --out (JSON). Exit 0 only when every check passes."""
import argparse
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--alias", required=True)
ap.add_argument("--np", type=int, required=True)
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--reasoning", choices=["auto", "off"], required=True)
ap.add_argument("--render-only", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = f"http://127.0.0.1:{a.port}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
checks = []


def call(path, body=None, timeout=1200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    t = time.time()
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read()), round(time.time() - t, 2)


def check(name, ok, info):
    checks.append({"check": name, "ok": bool(ok), "info": info})
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(info)[:600], flush=True)


props, _ = call("/props")
gen = props.get("default_generation_settings", {})
check("props: slots and context per slot", props.get("total_slots") == a.np and gen.get("n_ctx") == a.ctx,
      {"total_slots": props.get("total_slots"), "n_ctx": gen.get("n_ctx"), "build_info": props.get("build_info"),
       "model_path": props.get("model_path")})
models, _ = call("/v1/models")
ids = [d.get("id") for d in models.get("data", [])]
check("alias listed by /v1/models", a.alias in ids, {"ids": ids})

msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with the single word READY."}]
plain, _ = call("/apply-template", {"messages": msgs})
off, _ = call("/apply-template", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})
p_plain, p_off = plain["prompt"], off["prompt"]
THINK_OFF, THINK_ON = "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n<think>\n"
check("render with enable_thinking=false ends with an empty think block", p_off.endswith(THINK_OFF), {"tail": p_off[-60:]})
if a.reasoning == "auto":
    check("render without the switch: the template's thinking is on", p_plain.endswith(THINK_ON), {"tail": p_plain[-60:]})
else:
    check("render without the switch equals enable_thinking=false (-rea off)", p_plain == p_off, {"tail": p_plain[-60:]})

if not a.render_only:
    r, secs = call("/v1/chat/completions", {"model": a.alias, "messages": msgs, "temperature": 0, "seed": 0, "max_tokens": 16})
    m = r["choices"][0]["message"]
    content, reasoning = m.get("content") or "", m.get("reasoning_content") or ""
    check("reply without the switch has no thinking text",
          content.strip() != "" and "<think>" not in content and "</think>" not in content and reasoning.strip() == "",
          {"content": content, "reasoning_content": reasoning, "finish_reason": r["choices"][0].get("finish_reason"),
           "seconds": secs, "usage": r.get("usage"), "timings": r.get("timings")})
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city name."}},
                       "required": ["city"]}}}]
    body = {"model": a.alias, "messages": [{"role": "user", "content": "What is the weather in Oslo right now? Use the tool."}],
            "tools": tools, "tool_choice": "auto", "temperature": 0, "seed": 0, "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False}}
    r, secs = call("/v1/chat/completions", body)
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    try:
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (ValueError, KeyError, TypeError):
        args = {}
    check("one tool call parsed (get_weather, city Oslo)",
          len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and "oslo" in str(args.get("city", "")).lower()
          and not (m.get("reasoning_content") or "").strip(),
          {"tool_calls": calls, "content": m.get("content"), "reasoning_content": m.get("reasoning_content"),
           "finish_reason": r["choices"][0].get("finish_reason"), "seconds": secs, "usage": r.get("usage"),
           "timings": r.get("timings")})

with open(a.out, "w") as f:
    json.dump({"args": vars(a), "checks": checks}, f, indent=1)
n_ok = sum(c["ok"] for c in checks)
print(f"checks: {n_ok} passed, {len(checks) - n_ok} failed", flush=True)
sys.exit(0 if n_ok == len(checks) else 1)
