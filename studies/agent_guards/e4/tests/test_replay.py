"""Offline test of the paired replay store with a fake completion (no network, no server)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import sys
import tempfile

sys.path.insert(0, f"{_REPO_ROOT}/studies/agent_guards/e4")


def _resp(name):
    from litellm import ModelResponse

    return ModelResponse(**{"id": "x", "object": "chat.completion", "created": 1, "model": "m",
                            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                "role": "assistant", "content": None, "tool_calls": [
                                    {"id": "call_1", "type": "function",
                                     "function": {"name": name, "arguments": "{\"order_id\": \"#W1\"}"}}]}}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


def test_record_then_replay_and_transport_args_ignored():
    import tau2.utils.llm_utils as lu
    from tau2.data_model.message import SystemMessage, UserMessage

    from e4guard.replay import ReplayMiss, ReplayStore

    calls = []

    def fake(**kw):
        calls.append(kw)
        return _resp(f"tool_{len(calls)}")

    with tempfile.TemporaryDirectory(dir=f"{_REPO_ROOT}/studies/agent_guards/e4/.tmp") as d:
        st = ReplayStore(f"{d}/s.sqlite")
        saved = lu.completion
        try:
            lu.completion = st.wrap(fake)
            msgs = [SystemMessage(role="system", content="<policy>x</policy>"), UserMessage(role="user", content="hi")]
            a = lu.generate(model="openai/m", messages=msgs, temperature=0.0, seed=7, api_base="http://127.0.0.1:1/v1",
                            num_retries=0)
            b = lu.generate(model="openai/m", messages=msgs, temperature=0.0, seed=7, api_base="http://127.0.0.1:2/v1",
                            num_retries=0)
            assert len(calls) == 1 and a.tool_calls[0].name == b.tool_calls[0].name == "tool_1"
            c = lu.generate(model="openai/m", messages=msgs, temperature=0.0, seed=8, num_retries=0)
            assert len(calls) == 2 and c.tool_calls[0].name == "tool_2"     # the seed is part of the key
            assert st.stats == {"hit": 1, "miss": 2, "stored": 2, "replay_miss": 0}
            st2 = ReplayStore(f"{d}/s.sqlite", mode="replay_only")
            lu.completion = st2.wrap(fake)
            e = lu.generate(model="openai/m", messages=msgs, temperature=0.0, seed=7, num_retries=0)
            assert e.tool_calls[0].name == "tool_1" and len(calls) == 2
            try:
                lu.generate(model="openai/m", messages=msgs, temperature=0.0, seed=9, num_retries=0)
                raise AssertionError("expected a miss")
            except ReplayMiss:
                pass
            assert st2.stats["replay_miss"] == 1 and st2.stats["hit"] == 1
            role = st.db.execute("SELECT role FROM calls LIMIT 1").fetchone()[0]
            assert role == "agent"
        finally:
            lu.completion = saved


def test_role_guess_names_the_judge():
    from e4guard.replay import guess_role

    assert guess_role([{"role": "system", "content": "<policy>x</policy>"}]) == "agent"
    assert guess_role([{"role": "system", "content": "# User Simulation Guidelines\n<scenario>x</scenario>"}]) == "user"
    assert guess_role([{"role": "system", "content": "TASK - You will be given a list of expected outcomes"}]) == "judge"
    assert guess_role([{"role": "user", "content": "hi"}]) == "other"
