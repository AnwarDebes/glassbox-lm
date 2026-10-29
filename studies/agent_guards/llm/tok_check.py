#!/usr/bin/env python
"""Tokenizer check for the converted GGUF (CPU only): llama.cpp's llama-tokenize (vocab only) against the model's own
tokenizer.json (HF tokenizers) on the 100 paragraphs of ppl_paragraphs.txt plus 20 fixed strings with special cases.
    python -I tok_check.py   (run with the converter venv's python, which has tokenizers 0.22.2)
Prints the number of texts whose ids are identical, and the first mismatch."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
import json, os, subprocess, sys, tempfile
from pathlib import Path
from tokenizers import Tokenizer
HERE = Path(__file__).resolve().parent
B = Path(f"{_WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin")
M = Path(f"{_WS_ROOT}/tools/models_untrusted/qwen3.5-9b")
tok = Tokenizer.from_file(str(M / "c2022362" / "tokenizer.json"))
paras = [p for p in (HERE / "ppl_paragraphs.txt").read_text(encoding="utf-8").split("\n\n") if p.strip()]
extra = ["Hello world", "  leading spaces", "tabs\tand\nnewlines", "naïve café 東京 😀", "don't won't it's", "12345 678.90 1e-9",
         "<tool_call>{\"name\": \"f\"}</tool_call>", "x" * 300, "UPPER lower MiXeD", "a,b;c:d!e?f", "https://www.uia.no/path?q=1",
         "Æ Ø Å æ ø å", "def f(x):\n    return x ** 2", "€100 £50 ¥1000", "e.g. i.e. etc.", "—–-", "   ", "​ zero width",
         "Grüße aus Grimstad", "русский текст"]
texts = paras + extra
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
same, first = 0, None
for i, t in enumerate(texts):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(t)
        fn = f.name
    out = subprocess.run([str(B / "llama-tokenize"), "-m", str(M / "gguf-b11742" / "Qwen3.5-9B-Q8_0.gguf"), "-f", fn, "--ids",
                          "--no-bos", "--log-disable"], capture_output=True, text=True, env=env)
    os.unlink(fn)
    line = [l for l in out.stdout.strip().split("\n") if l.startswith("[")][-1]
    got = json.loads(line)
    want = tok.encode(t, add_special_tokens=False).ids
    if got == want:
        same += 1
    elif first is None:
        first = {"i": i, "text": t[:120], "llama": got[:30], "hf": want[:30]}
print(json.dumps({"texts": len(texts), "identical": same, "first_mismatch": first}))
