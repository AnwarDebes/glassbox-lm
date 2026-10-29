"""CPU time of one decision of the fine-tuned transformer arm (one thread, batch 1); cost only, no accuracy.

usage: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -I ft_speed.py --dataset abcd \
           --model bge-base --out runs/cost/ft_abcd_cpu.json [--n 300]
The fine-tuned weights are not saved by ft_baseline.py, and they do not change the compute: the timed network is the
same architecture (the pretrained encoder plus a linear head of the dataset's label count, as ft_baseline.py builds
it), fed the same text (ft_baseline.render, max_len 256, truncation). Timed: tokenization plus the forward pass, per
dev row, batch 1, torch.set_num_threads(1), after 20 warm-up rows. Rough timing on a shared machine.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--model", choices=sorted(F.MODELS), default="bge-base")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(1)
    xdv, _, n_lab, sha = F.load(a.dataset, "dev")
    name = F.MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_lab, local_files_only=True).eval()
    n_params = sum(p.numel() for p in model.parameters())
    idx = random.Random(0).sample(range(len(xdv)), min(a.n, len(xdv)))
    times, lens = [], []
    with torch.inference_mode():
        for warm in (True, False):
            for i in (idx[:20] if warm else idx):
                t0 = time.perf_counter()
                enc = tok(xdv[i], truncation=True, max_length=a.max_len, return_tensors="pt")
                model(**enc).logits.argmax(-1)
                t1 = time.perf_counter()
                if not warm:
                    times.append((t1 - t0) * 1000)
                    lens.append(int(enc["input_ids"].shape[1]))
    res = {"dataset": a.dataset, "model": name, "n_params": n_params, "n_labels": n_lab, "dev_sha256": sha,
           "n": len(times), "threads": torch.get_num_threads(), "max_len": a.max_len,
           "median_tokens": float(np.median(lens)), "total_median_ms": float(np.median(times)),
           "total_p90_ms": float(np.percentile(times, 90)),
           "label": "rough: shared machine, one CPU thread, batch 1; pretrained weights with a new head (same compute)",
           "torch": torch.__version__, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
