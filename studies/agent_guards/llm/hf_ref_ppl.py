#!/usr/bin/env python
"""Conversion check for Qwen3.5-9B (CPU only; local files only; no remote code; no network).

    python -I hf_ref_ppl.py LLAMA_IDS.txt OUT.json

Why: the GPU smoke test compares llama.cpp's CUDA and CPU backends on the SAME GGUF, so it cannot see a conversion
bug (the converter warned that its transformers 4.57.6 does not know qwen3_5). This is the reference side: the
original Hugging Face weights (revision c2022362, bf16 safetensors, upcast to fp32) in transformers' own
Qwen3_5ForConditionalGeneration (transformers 5.20.0, trust_remote_code=False), on the first 8 chunks of
llm/ppl_paragraphs.txt, chunked and scored as llama-perplexity does at -c 512: the file's tokens (no BOS: the GGUF has
add_bos_token false) in windows of 512, each window alone, the last 255 tokens of each window scored (the logits at
positions 256..510). The forward pass reads llama-tokenize's ids, so both sides score the same tokens; the script also
reports whether the HF tokenizer gives the same ids for the whole file. It prints the running perplexity after each
chunk beside llama.cpp's Q8_0 values from ppl_cpu.log ([1]..[8]).
Bar (fixed before the run, LOG.md): conversion OK when the running perplexity after 8 chunks is within 2% of
llama.cpp's [8] (32.6690) and every running value [1]..[8] is within 4%; over 5% after 8 chunks = conversion bug
suspected; in between = unclear. Q8_0 against fp32 alone usually moves perplexity by well under 1%.
Writes OUT.json (refuses to overwrite).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/

import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
Q = Path(f"{_WS_ROOT}/tools/models_untrusted/qwen3.5-9b/c2022362")
NCTX, NCH = 512, 8


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> None:
    t0 = time.time()
    ids_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    assert not out.exists(), f"{out} exists"
    txt_p = HERE / "ppl_paragraphs.txt"
    assert sha16(txt_p) == "72a8d08a1fab8ac7", "ppl_paragraphs.txt changed"
    text = txt_p.read_bytes().decode("utf-8")
    m = re.search(r"\[1\][0-9.]+(,\[[0-9]+\][0-9.]+)*", (HERE / "ppl_cpu.log").read_text())
    ref = {int(k): float(v) for k, v in re.findall(r"\[([0-9]+)\]([0-9.]+)", m.group(0))}
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import transformers

    tok = AutoTokenizer.from_pretrained(Q, local_files_only=True, trust_remote_code=False)
    hf_ids = tok(text, add_special_tokens=False)["input_ids"]
    l_ids = json.loads(ids_p.read_text())
    diff = next((i for i, (a, b) in enumerate(zip(hf_ids, l_ids)) if a != b), None)
    res = {"llama_ids": len(l_ids), "hf_ids": len(hf_ids), "ids_identical": hf_ids == l_ids, "first_difference": diff,
           "llama_chunks": len(l_ids) // NCTX, "transformers": transformers.__version__, "torch": torch.__version__,
           "threads": torch.get_num_threads()}
    print(json.dumps(res), flush=True)
    assert len(l_ids) // NCTX == 41, "llama-perplexity reported 41 chunks"
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(Q, local_files_only=True, trust_remote_code=False,
                                                                   dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"], info["missing_keys"][:10]
    res["missing_keys"] = 0
    res["unexpected_keys"] = len(info["unexpected_keys"])
    res["unexpected_prefixes"] = sorted({k.split(".")[0] for k in info["unexpected_keys"]})
    res["load_seconds"] = round(time.time() - t0, 1)
    model.eval()
    nll, cnt, rows = 0.0, 0, []
    for c in range(NCH):
        t1 = time.time()
        x = torch.tensor([l_ids[c * NCTX : (c + 1) * NCTX]], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=x, use_cache=False).logits[0, NCTX // 2 : NCTX - 1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = x[0, NCTX // 2 + 1 : NCTX]
        nll += float(-lp[torch.arange(len(tgt)), tgt].sum())
        cnt += len(tgt)
        ppl = math.exp(nll / cnt)
        rows.append({"chunk": c + 1, "ppl_hf_fp32": round(ppl, 4), "ppl_llama_q8_0": ref[c + 1],
                     "rel_diff": round(ppl / ref[c + 1] - 1, 4), "seconds": round(time.time() - t1, 1)})
        print(json.dumps(rows[-1]), flush=True)
    d8 = abs(rows[-1]["rel_diff"])
    ok = d8 <= 0.02 and all(abs(r["rel_diff"]) <= 0.04 for r in rows)
    res.update({"rows": rows, "scored_tokens": cnt, "max_abs_rel_diff": max(abs(r["rel_diff"]) for r in rows),
                "verdict": "conversion OK" if ok else ("conversion bug suspected" if d8 > 0.05 else "unclear"),
                "hf_ref_ppl.py": sha16(Path(__file__).resolve()), "seconds": round(time.time() - t0, 1),
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
