"""Run the tau2 CLI (tau2-bench v1.0.1) with EVERY LLM role on one local OpenAI-compatible endpoint.

Roles and how each is pointed at the local server:
  - agent (llm_agent)            : --agent-llm openai/<model>  --agent-llm-args '{"api_base": ..., "api_key": ...}'
  - user simulator               : --user-llm  openai/<model>  --user-llm-args  '{"api_base": ..., "api_key": ...}'
  - NL-assertion judge (reward!) : no CLI flag in v1.0.1. tau2/config.py:24 hard-codes gpt-4.1-2025-04-14 and
                                   tau2/evaluator/evaluator_nl_assertions.py:8,122 reads it; this wrapper rebinds
                                   those module globals to the local model before the CLI starts (threads share it).
  - review / hallucination LLMs  : only with --auto-review (default: a paid vendor API model, config.py:32) -> refused here.

Guard rails (the wrapper exits instead of running):
  - the endpoint host must be 127.0.0.1 or localhost;
  - --agent-llm-args and --user-llm-args must both be given and carry a local api_base;
  - --agent-llm / --user-llm must be 'openai/...' model strings;
  - --auto-review is refused.
This wrapper never starts a server.

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -I -B scripts/tau2_local.py --base http://127.0.0.1:18080/v1 \
      --model <served-model-id> -- run --domain retail ... (any tau2 CLI arguments)
"""

import argparse
import json
import sys
import urllib.parse

LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def die(msg):
    sys.stderr.write(f"tau2_local.py: refusing: {msg}\n")
    sys.exit(3)


def arg_value(rest, flag):
    if flag in rest:
        i = rest.index(flag)
        if i + 1 < len(rest):
            return rest[i + 1]
    return None


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--base", required=True, help="http://127.0.0.1:<port>/v1")
    ap.add_argument("--model", required=True, help="served model id (see GET /v1/models)")
    ap.add_argument("--api-key", default="sk-local-no-key")
    known, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if urllib.parse.urlparse(known.base).hostname not in LOCAL_HOSTS:
        die(f"endpoint {known.base} is not local")
    if rest and rest[0] == "run":
        if "--auto-review" in rest:
            die("--auto-review calls an external paid review model (config.py:32)")
        for flag in ("--agent-llm", "--user-llm"):
            v = arg_value(rest, flag)
            if v is None or not v.startswith("openai/"):
                die(f"{flag} must be given as openai/<served-model-id> (got {v!r})")
        for flag in ("--agent-llm-args", "--user-llm-args"):
            v = arg_value(rest, flag)
            if v is None:
                die(f"{flag} missing (tau2's default would contact api.openai.com)")
            try:
                host = urllib.parse.urlparse(json.loads(v).get("api_base", "")).hostname
            except Exception as e:  # noqa: BLE001
                die(f"{flag} is not valid JSON: {e}")
            if host not in LOCAL_HOSTS:
                die(f"{flag} has no local api_base")

    # Point the NL-assertion judge at the local model (module globals are read at call time).
    import tau2.evaluator.evaluator_nl_assertions as nl

    nl.DEFAULT_LLM_NL_ASSERTIONS = "openai/" + known.model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, "api_base": known.base, "api_key": known.api_key}
    sys.stderr.write(f"tau2_local.py: NL-assertion judge -> {nl.DEFAULT_LLM_NL_ASSERTIONS} at {known.base}\n")

    from tau2.cli import main as tau2_main

    sys.argv = ["tau2"] + rest
    return tau2_main()


if __name__ == "__main__":
    sys.exit(main())
