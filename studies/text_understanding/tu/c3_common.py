"""Step 3c, shared pieces (tu/PLAN_3c.md): the prompt, the decoding settings, the request body and the parser.

Imported by c3_generate.py (the client) and c3_pilot_check.py; nothing here sends or reads anything by itself.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
sys.path.insert(0, str(EMB / "w1"))
import build_corpus_c4 as B  # noqa: E402

SETS = ["clinc150", "banking77", "hwu64"]
N_PARA = 5
# Prompt v4 (2026-10-20, LOG entry after 22:32Z): v3 plus one sentence against label-word injection. In pilot v3
# (c3/pilot_v3) 23.5% of the paraphrases added a word of the intent label that the request did not have (Banking77
# 39%), e.g. "help my money isnt showing up" -> "... after the bank transfer".
# Prompt v3: the fixed instructions are in the system message and only the label and the text are in a user-role
# message. llama-server keeps a state checkpoint of this hybrid model at the start of the last user-role message, so the
# whole system message is reused from the cache. Pilot v1 (label and text first; 5 rows in c3/pilot_v1) and v2
# (instructions first in a user-role message; 3 rows in c3/pilot_v2) reused only the 16 tokens of the system message.
SYSTEM = ("You rewrite requests that users send to a virtual assistant. For each request, write 5 different ways a "
          "user could say it. Keep the same intent and the same details (names, numbers, dates, places). The intent "
          "label only tells you what the request means: do not add words or details from the label that are not in "
          "the request. Vary the words and the sentence form; short and casual forms are fine. Do not answer the "
          "request. Write one per line, with no numbering, no quotes and nothing else.")
USER = "Intent label: {label}\nRequest: {text}"
DECODE = {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0, "max_tokens": 192}
MARK = re.compile(r"^\s*(?:\d{1,2}\s*[.):]|[-*•])\s*")
QUOTES = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}


def label_text(intent: str) -> str:
    return intent.replace("_", " ")


def seed_of(set_name: str, index: int) -> int:
    return 1_000_003 * SETS.index(set_name) + index + 1


def body(set_name: str, index: int, text: str, intent: str) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(label=label_text(intent), text=text)}],
            **DECODE, "seed": seed_of(set_name, index), "chat_template_kwargs": {"enable_thinking": False}}


def parse(raw: str, source: str) -> tuple[list[str], dict]:
    """Lines of the model's answer -> at most N_PARA paraphrases, and the count of each drop reason."""
    n_src = max(1, len(source.split()))
    max_words = max(2 * n_src, n_src + 12)
    seen = {B.key(source)}
    out: list[str] = []
    why = {"lines": 0, "empty": 0, "header": 0, "too_long": 0, "same_as_source_or_earlier": 0, "over_5": 0}
    for line in (raw or "").splitlines():
        s = MARK.sub("", line.strip()).strip()
        if len(s) >= 2 and (s[0], s[-1]) in QUOTES:
            s = s[1:-1].strip()
        if not s or not B.norm(s):
            why["empty"] += 1
            continue
        why["lines"] += 1
        if s.endswith(":"):
            why["header"] += 1
            continue
        if len(s.split()) > max_words:
            why["too_long"] += 1
            continue
        k = B.key(s)
        if k in seen:
            why["same_as_source_or_earlier"] += 1
            continue
        if len(out) == N_PARA:
            why["over_5"] += 1
            continue
        seen.add(k)
        out.append(s)
    return out, why
