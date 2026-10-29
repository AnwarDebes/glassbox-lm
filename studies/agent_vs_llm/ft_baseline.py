"""Fine-tune a transformer encoder end to end on next-action prediction (STAR or ABCD), dev only.

usage (from any folder, in decision_layer's .venv-dl):
  HF_HUB_OFFLINE=1 python -I ft_baseline.py --dataset star --model roberta-base --lr 3e-5 --seed 1 --out DIR
      [--epochs 4] [--batch 32] [--max-len 256] [--limit-train N] [--limit-dev N]

Reads data/<dataset>/examples_{train,dev}.json.gz of the decision_layer study; the test file is never opened.
Input text (plan PLAN_ft_dev.md): previous actions, the recent exchange, then the tail of the customer's utterances.
Writes DIR/result.json (dev accuracy per epoch, best and final epoch, timings, config, input hashes).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time

DATA = f"{_REPO_ROOT}/studies/decision_layer/data"
MODELS = {
    "roberta-base": "roberta-base",
    "bert-base-uncased": "bert-base-uncased",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}
CUST_CHARS = 800
N_PREV = 10


def render(row):
    """The text one example shows the model: actions, recent exchange, then the customer's last words."""
    prev = row.get("prev") or []
    acts = " ; ".join(str(a) for a in prev[-N_PREV:]) or "none"
    cust = (row.get("cust") or "")[-CUST_CHARS:]
    return f"actions: {acts} | recent: {row.get('recent') or ''} | customer: {cust}"


def load(dataset, split, limit=None, allow_test=False):
    if split not in ("train", "dev") and not (allow_test and split == "test"):
        raise ValueError("only train and dev are allowed (test only under a frozen registration)")
    path = os.path.join(DATA, dataset, f"examples_{split}.json.gz")
    with open(path, "rb") as h:
        raw = h.read()
    d = json.loads(gzip.decompress(raw))
    rows = d["rows"][:limit] if limit else d["rows"]
    return [render(r) for r in rows], [int(r["y"]) for r in rows], len(d["buttons"]), hashlib.sha256(raw).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-registration", help="also score TEST at the best dev epoch; needs the .frozen.json")
    ap.add_argument("--dry-run-dev", action="store_true", help="with --test-registration: dev stands in for test")
    a = ap.parse_args(argv)
    if a.test_registration and not a.dry_run_dev:
        fr = os.path.splitext(a.test_registration)[0] + ".frozen.json"
        if not os.path.exists(fr):
            raise SystemExit(f"refused: no freeze record {fr}; the test split is read only under a frozen registration")

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    dev_ = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)

    xtr, ytr, n_lab, h_tr = load(a.dataset, "train", a.limit_train)
    xdv, ydv, n_lab2, h_dv = load(a.dataset, "dev", a.limit_dev)
    xte = yte = None
    if a.test_registration:
        xte, yte, _, h_te = load(a.dataset, "dev" if a.dry_run_dev else "test", a.limit_dev, allow_test=True)
    assert n_lab == n_lab2
    name = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_lab, local_files_only=True).to(dev_)
    t0 = time.time()
    enc_tr = tok(xtr, truncation=True, max_length=a.max_len)
    enc_dv = tok(xdv, truncation=True, max_length=a.max_len)
    enc_te = tok(xte, truncation=True, max_length=a.max_len) if xte is not None else None
    tok_s = time.time() - t0
    trunc_tr = sum(len(ids) >= a.max_len for ids in enc_tr["input_ids"]) / len(xtr)

    def batches(enc, ys, order, bs):
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            feats = [{k: enc[k][j] for k in enc.keys()} for j in idx]
            b = tok.pad(feats, return_tensors="pt")
            yield {k: v.to(dev_) for k, v in b.items()}, torch.tensor([ys[j] for j in idx], device=dev_)

    steps_per_epoch = math.ceil(len(xtr) / a.batch)
    total = steps_per_epoch * a.epochs
    nd = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in nd)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in nd)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(a.seed)
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t1 = time.time()
        order = rng.permutation(len(xtr)).tolist()
        loss_sum = 0.0
        for b, y in batches(enc_tr, ytr, order, a.batch):
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**b)
                loss = torch.nn.functional.cross_entropy(out.logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss) * len(y)
        train_s = time.time() - t1
        model.eval()
        correct = 0
        t2 = time.time()
        top_i, top_p = [], []
        with torch.no_grad():
            for b, y in batches(enc_dv, ydv, list(range(len(xdv))), 128):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**b).logits.float()
                pred = logits.argmax(-1)
                pr, ix = torch.softmax(logits, -1).topk(min(5, n_lab), -1)
                top_i.append(ix.cpu().numpy())
                top_p.append(pr.cpu().numpy())
                correct += int((pred == y).sum())
        acc = correct / len(xdv)
        if not hist or acc > max(r["dev_acc"] for r in hist):
            best_preds = (np.concatenate(top_i).astype(np.int16), np.concatenate(top_p).astype(np.float16))
            if enc_te is not None:  # test scored at every new best dev epoch; only the best one is kept
                ti, tp = [], []
                with torch.no_grad():
                    for b, y in batches(enc_te, yte, list(range(len(xte))), 128):
                        with torch.autocast("cuda", dtype=torch.float16):
                            lg = model(**b).logits.float()
                        pr, ix = torch.softmax(lg, -1).topk(min(5, n_lab), -1)
                        ti.append(ix.cpu().numpy())
                        tp.append(pr.cpu().numpy())
                test_preds = (np.concatenate(ti).astype(np.int16), np.concatenate(tp).astype(np.float16))
        hist.append({"epoch": ep, "dev_acc": acc, "train_loss": loss_sum / len(xtr), "train_s": round(train_s, 1),
                     "eval_s": round(time.time() - t2, 1)})
        print(json.dumps(hist[-1]), flush=True)
    best = max(hist, key=lambda r: r["dev_acc"])
    res = {
        "plan": "PLAN_ft_dev.md", "dataset": a.dataset, "model": name, "lr": a.lr, "seed": a.seed, "epochs": a.epochs,
        "batch": a.batch, "max_len": a.max_len, "n_train": len(xtr), "n_dev": len(xdv), "n_labels": n_lab,
        "limit_train": a.limit_train, "limit_dev": a.limit_dev, "train_sha256": h_tr, "dev_sha256": h_dv,
        "truncated_share_train": round(trunc_tr, 4), "tokenize_s": round(tok_s, 1), "history": hist,
        "best_epoch": best["epoch"], "best_dev_acc": best["dev_acc"], "final_dev_acc": hist[-1]["dev_acc"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "argv": sys.argv, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(a.out, "result.json"), "w") as h:
        json.dump(res, h, indent=1)
    # dev predictions of the best epoch (note NOTE_ft_preds.md): top-5 label ids and probabilities, gold labels
    np.savez_compressed(os.path.join(a.out, "dev_preds.npz"), top_ids=best_preds[0], top_probs=best_preds[1],
                        gold=np.array(ydv, dtype=np.int16), best_epoch=best["epoch"])
    if enc_te is not None:
        np.savez_compressed(os.path.join(a.out, "test_preds.npz"), top_ids=test_preds[0], top_probs=test_preds[1],
                            gold=np.array(yte, dtype=np.int16), best_epoch=best["epoch"], dry_run_dev=a.dry_run_dev,
                            registration=a.test_registration)
    print("best", best["epoch"], round(best["dev_acc"], 4), "final", round(hist[-1]["dev_acc"], 4), flush=True)


if __name__ == "__main__":
    main()
