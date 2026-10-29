"""Time the parts of one offline AgentDojo run (workspace, v1.2.2). No LLM."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
import yaml
from agentdojo.task_suite.load_suites import get_suite
from replay import ReplayLLM
from pipelines import build_pipeline
suite = get_suite("v1.2.2", "workspace")
print("yaml C loader available:", hasattr(yaml, "CSafeLoader"))
t = time.time(); env = suite.load_and_inject_default_environment({}); print("load_and_inject_default_environment: %.2f s" % (time.time() - t))
t = time.time(); env2 = env.model_copy(deep=True); print("model_copy(deep=True): %.3f s" % (time.time() - t))
ut = suite.get_user_task_by_id("user_task_0")
t = time.time(); gt = ut.ground_truth(env); print("ground_truth: %.3f s, %d calls" % (time.time() - t, len(gt)))
replay = ReplayLLM(suite); pipe, _ = build_pipeline(replay, "none")
t = time.time(); r = suite.run_task_with_pipeline(pipe, ut, None, {}); print("run_task_with_pipeline (benign):", r, "%.2f s" % (time.time() - t))
if len(sys.argv) > 1:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(sys.argv[1])
    import json
    from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
    s = json.dumps([_function_to_openai(f) for f in suite.tools])
    t = time.time(); n = len(tok.encode(s).ids); print("tokenize tool schemas: %d tokens, %.3f s" % (n, time.time() - t))
